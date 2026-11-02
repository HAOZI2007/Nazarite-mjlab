"""PPO with FR-Net mass-contact auxiliary supervision."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from rsl_rl.algorithms import PPO
from tensordict import TensorDict
from torch import nn

from .frnet_actor import FRNetActor


class FRNetPPO(PPO):
    """Optimize the normal PPO objective and the MCP prediction objective jointly."""

    @property
    def frnet_actor(self) -> FRNetActor:
        return cast(FRNetActor, self.actor)

    def __init__(
        self,
        *args,
        mcp_loss_coef: float = 0.25,
        mass_loss_coef: float = 1.0,
        contact_loss_coef: float = 1.0,
        reconstruction_loss_coef: float = 1.0,
        vae_kl_loss_coef: float = 0.01,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not isinstance(self.actor, FRNetActor):
            raise TypeError("FRNetPPO requires FRNetActor as its actor model")
        if self.rnd is not None or self.symmetry is not None:
            raise ValueError(
                "FRNetPPO currently does not combine MCP with RND or symmetry"
            )
        self.mcp_loss_coef = mcp_loss_coef
        self.mass_loss_coef = mass_loss_coef
        self.contact_loss_coef = contact_loss_coef
        self.reconstruction_loss_coef = reconstruction_loss_coef
        self.vae_kl_loss_coef = vae_kl_loss_coef
        if self.frnet_actor.use_vae:
            # Extra training-only fields use the existing shuffled TensorDict storage,
            # keeping targets aligned with PPO samples without changing upstream RSL.
            shape = (self.storage.num_transitions_per_env, self.storage.num_envs)
            self.storage.observations["frnet_next_proprio"] = torch.zeros(
                *shape, self.frnet_actor.proprio_dim, device=self.device
            )
            self.storage.observations["frnet_next_valid"] = torch.zeros(
                *shape, 1, dtype=torch.bool, device=self.device
            )

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        """Pair history at t with the actual next observation, including rollout end."""
        if self.frnet_actor.use_vae:
            if self.transition.observations is None:
                raise RuntimeError("Call act() before process_env_step()")
            # The env supplies clean proprioception in exactly the actor frame order.
            # Done observations may already belong to a reset episode; exclude them.
            next_proprio = obs["frnet_proprio"]
            valid = ~dones.reshape(-1, 1).bool()
            transition_obs = self.transition.observations.clone(recurse=False)
            transition_obs["frnet_next_proprio"] = torch.where(
                valid, next_proprio, torch.zeros_like(next_proprio)
            )
            transition_obs["frnet_next_valid"] = valid
            self.transition.observations = transition_obs
        super().process_env_step(obs, rewards, dones, extras)

    def _vae_losses(
        self, observations: TensorDict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reconstruction, mean, logvar = self.frnet_actor.get_vae_outputs()
        target = self.actor.obs_normalizer(observations["frnet_next_proprio"])
        valid = observations["frnet_next_valid"].squeeze(-1)
        # A whole minibatch may terminate. Keep a differentiable zero in that case.
        squared_error = (reconstruction - target).square().mean(dim=-1)
        reconstruction_loss = torch.where(valid, squared_error, 0.0).sum() / (
            valid.sum().clamp(min=1)
        )
        # KL(q(z|history) || N(0,I)): sum latent dimensions, then average examples.
        kl_loss = (-0.5 * (1 + logvar - mean.square() - logvar.exp()).sum(-1)).mean()
        return reconstruction_loss, kl_loss

    def update(self) -> dict[str, float]:
        """Run PPO updates with supervised mass and contact prediction losses."""
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_mass_loss = 0.0
        mean_contact_loss = 0.0
        mean_contact_accuracy = 0.0
        mean_reconstruction_loss = 0.0
        mean_vae_kl_loss = 0.0
        mean_next_valid = 0.0

        generator = self.storage.mini_batch_generator(
            self.num_mini_batches,
            self.num_learning_epochs,
        )
        for batch in generator:
            assert batch.observations is not None
            assert batch.actions is not None
            assert batch.advantages is not None
            assert batch.returns is not None
            assert batch.values is not None
            assert batch.old_actions_log_prob is not None
            assert batch.old_distribution_params is not None

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (
                        batch.advantages.std() + 1.0e-8
                    )

            self.actor(batch.observations, stochastic_output=True)
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations)
            entropy = self.actor.output_entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl_mean = torch.mean(
                        self.actor.get_kl_divergence(
                            batch.old_distribution_params,
                            self.actor.output_distribution_params,
                        )
                    )
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(
                            kl_mean,
                            op=torch.distributed.ReduceOp.SUM,
                        )
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        learning_rate_tensor = torch.tensor(
                            self.learning_rate,
                            device=self.device,
                        )
                        torch.distributed.broadcast(learning_rate_tensor, src=0)
                        self.learning_rate = learning_rate_tensor.item()
                    for parameter_group in self.optimizer.param_groups:
                        parameter_group["lr"] = self.learning_rate

            ratio = torch.exp(
                actions_log_prob - torch.squeeze(batch.old_actions_log_prob)
            )
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio,
                1.0 - self.clip_param,
                1.0 + self.clip_param,
            )
            surrogate_loss = torch.maximum(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(
                    -self.clip_param,
                    self.clip_param,
                )
                value_loss = torch.maximum(
                    (values - batch.returns).pow(2),
                    (value_clipped - batch.returns).pow(2),
                ).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            targets = batch.observations["frnet_aux"]
            mass_target = targets[:, :4]
            contact_target = targets[:, 4:]
            mass_prediction, contact_logits, _latent = (
                self.frnet_actor.get_mcp_outputs()
            )
            mass_loss = F.mse_loss(mass_prediction, mass_target)
            contact_loss = F.binary_cross_entropy_with_logits(
                contact_logits,
                contact_target,
            )
            mcp_loss = (
                self.mass_loss_coef * mass_loss + self.contact_loss_coef * contact_loss
            )
            if self.frnet_actor.use_vae:
                reconstruction_loss, vae_kl_loss = self._vae_losses(batch.observations)
                mcp_loss = (
                    mcp_loss
                    + self.reconstruction_loss_coef * reconstruction_loss
                    + self.vae_kl_loss_coef * vae_kl_loss
                )
                mean_reconstruction_loss += reconstruction_loss.item()
                mean_vae_kl_loss += vae_kl_loss.item()
                mean_next_valid += (
                    batch.observations["frnet_next_valid"].float().mean().item()
                )

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy.mean()
                + self.mcp_loss_coef * mcp_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            with torch.no_grad():
                contact_accuracy = (
                    ((torch.sigmoid(contact_logits) >= 0.5) == (contact_target >= 0.5))
                    .to(dtype=torch.float32)
                    .mean()
                )
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_mass_loss += mass_loss.item()
            mean_contact_loss += contact_loss.item()
            mean_contact_accuracy += contact_accuracy.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        losses = {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "mcp_mass": mean_mass_loss / num_updates,
            "mcp_contact": mean_contact_loss / num_updates,
            "mcp_contact_accuracy": mean_contact_accuracy / num_updates,
        }
        if self.frnet_actor.use_vae:
            losses.update(
                {
                    "mcp_reconstruction": mean_reconstruction_loss / num_updates,
                    "mcp_vae_kl": mean_vae_kl_loss / num_updates,
                    "mcp_next_valid_fraction": mean_next_valid / num_updates,
                }
            )
        return losses
