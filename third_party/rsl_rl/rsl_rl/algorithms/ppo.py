# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import torch
import torch.nn as nn
from itertools import chain
from tensordict import TensorDict
from typing import Any

from rsl_rl.env import VecEnv
from rsl_rl.extensions import RandomNetworkDistillation, Symmetry, resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import compile_model, resolve_callable, resolve_obs_groups, resolve_optimizer


class PPO:
    """Proximal Policy Optimization algorithm.

    Reference:
        - Schulman et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
    """

    actor: MLPModel
    """The actor model."""

    critic: MLPModel
    """The critic model."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        diagnostics_enabled: bool = True,
        diagnostics_report_threshold: float = 100.0,
        diagnostics_max_abs: float = 1.0e6,
        strict_gradient_checks: bool = True,
        recover_on_nonfinite: bool = True,
        recovery_lr_factor: float = 0.5,
        max_consecutive_recoveries: int = 20,
        device: str = "cpu",
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        """Initialize the algorithm with models, storage, and optimization settings."""
        # Device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND extension
        self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg) if rnd_cfg else None

        # Symmetry extension
        if symmetry_cfg is not None and (actor.is_recurrent or critic.is_recurrent):
            raise ValueError("Symmetry augmentation is not supported for recurrent policies.")
        self.symmetry = Symmetry(**symmetry_cfg) if symmetry_cfg else None

        # PPO components
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)

        # Handles to the uncompiled modules for state_dict operations and export. If compilation is disabled, these
        # simply alias ``self.actor`` / ``self.critic``.
        self._raw_actor = self.actor
        self._raw_critic = self.critic

        # Create the optimizer
        self.optimizer = resolve_optimizer(optimizer)(
            chain(self.actor.parameters(), self.critic.parameters()), lr=learning_rate
        )  # type: ignore

        # Add storage
        self.storage = storage
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        self.diagnostics_enabled = diagnostics_enabled
        self.diagnostics_report_threshold = diagnostics_report_threshold
        self.diagnostics_max_abs = diagnostics_max_abs
        self.strict_gradient_checks = strict_gradient_checks
        self.recover_on_nonfinite = recover_on_nonfinite
        self.recovery_lr_factor = recovery_lr_factor
        self.max_consecutive_recoveries = max_consecutive_recoveries
        self.recovery_count = 0
        self.consecutive_recoveries = 0

    @staticmethod
    def _clone_to_cpu(value: Any) -> Any:
        """Clone nested state to CPU so a recovery snapshot is independent."""
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: PPO._clone_to_cpu(item) for key, item in value.items()}
        if isinstance(value, list):
            return [PPO._clone_to_cpu(item) for item in value]
        if isinstance(value, tuple):
            return tuple(PPO._clone_to_cpu(item) for item in value)
        return copy.deepcopy(value)

    def _make_update_snapshot(self) -> dict[str, Any]:
        """Capture all state that can be changed by one PPO update."""
        snapshot = {
            "actor": self._clone_to_cpu(self._raw_actor.state_dict()),
            "critic": self._clone_to_cpu(self._raw_critic.state_dict()),
            "optimizer": self._clone_to_cpu(self.optimizer.state_dict()),
            "learning_rate": self.learning_rate,
        }
        if self.rnd:
            snapshot["rnd"] = self._clone_to_cpu(self.rnd.state_dict())
            snapshot["rnd_optimizer"] = self._clone_to_cpu(self.rnd.optimizer.state_dict())
            snapshot["rnd_learning_rates"] = [group["lr"] for group in self.rnd.optimizer.param_groups]
        return snapshot

    def _restore_update_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Restore the state captured before a failed PPO update."""
        self._raw_actor.load_state_dict(snapshot["actor"])
        self._raw_critic.load_state_dict(snapshot["critic"])
        self.optimizer.load_state_dict(snapshot["optimizer"])
        if self.rnd:
            self.rnd.load_state_dict(snapshot["rnd"])
            self.rnd.optimizer.load_state_dict(snapshot["rnd_optimizer"])

        # Restoring the optimizer can restore its old learning rate, so apply the
        # reduced rate after loading the optimizer state.
        self.learning_rate = max(1.0e-8, snapshot["learning_rate"] * self.recovery_lr_factor)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

        self.optimizer.zero_grad(set_to_none=True)
        if self.rnd:
            self.rnd.optimizer.zero_grad(set_to_none=True)
            for param_group, learning_rate in zip(
                self.rnd.optimizer.param_groups,
                snapshot["rnd_learning_rates"],
                strict=True,
            ):
                param_group["lr"] = max(1.0e-8, learning_rate * self.recovery_lr_factor)
        self.storage.clear()

    def _diagnose_tensor(
        self,
        name: str,
        tensor: torch.Tensor,
        *,
        env_dim: int,
    ) -> None:
        """Report per-environment maxima and stop on invalid rollout values."""
        if not self.diagnostics_enabled:
            return

        env_view = tensor.movedim(env_dim, 0).reshape(tensor.shape[env_dim], -1)
        finite = torch.isfinite(env_view)
        finite_per_env = finite.all(dim=1)
        bad_envs = torch.where(~finite_per_env)[0]
        finite_abs = torch.where(finite, env_view.abs(), torch.zeros_like(env_view))
        max_abs = finite_abs.amax(dim=1)

        if bad_envs.numel() > 0:
            bad_list = bad_envs[:20].detach().cpu().tolist()
            raise FloatingPointError(
                f"PPO rollout {name} contains NaN/Inf for envs={bad_list}; "
                f"max_finite_abs={max_abs[bad_envs].max().item():.6g}"
            )

        top_k = min(5, max_abs.numel())
        top_values, top_envs = torch.topk(max_abs, k=top_k)
        if top_values[0] >= self.diagnostics_report_threshold:
            pairs = [
                f"env={int(env_id)} max_abs={value:.6g}"
                for env_id, value in zip(
                    top_envs.detach().cpu().tolist(),
                    top_values.detach().cpu().tolist(),
                    strict=False,
                )
            ]
            print(f"[PPO diagnostics] {name}: " + ", ".join(pairs))

        if top_values[0] > self.diagnostics_max_abs:
            raise FloatingPointError(
                f"PPO rollout {name} exceeds diagnostics_max_abs="
                f"{self.diagnostics_max_abs:.6g}; env={int(top_envs[0])}, "
                f"max_abs={top_values[0].item():.6g}"
            )

    def _check_gradients(self, model: nn.Module, model_name: str) -> None:
        """Raise with parameter names when a model gradient is not finite."""
        bad_parameters: list[str] = []
        max_grad = 0.0
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach()
            if not torch.isfinite(gradient).all():
                bad_parameters.append(name)
                continue
            max_grad = max(max_grad, gradient.abs().max().item())

        if bad_parameters:
            raise FloatingPointError(
                f"{model_name} gradient contains NaN/Inf in parameters "
                f"{bad_parameters[:10]}; max_finite_grad={max_grad:.6g}"
            )

    @staticmethod
    def _check_parameters(model: nn.Module, model_name: str) -> None:
        """Raise when an optimizer step produced non-finite parameters."""
        for name, parameter in model.named_parameters():
            if not torch.isfinite(parameter).all():
                raise FloatingPointError(f"{model_name} parameter contains NaN/Inf after optimizer step: {name}")

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions and store transition data."""
        # Record the hidden states for recurrent policies
        self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic.get_hidden_state())
        # Compute the actions and values
        self.transition.actions = self.actor(obs, stochastic_output=True).detach()
        self.transition.values = self.critic(obs).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()  # type: ignore
        self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
        # Record observations before env.step()
        self.transition.observations = obs
        return self.transition.actions  # type: ignore

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Record one environment step and update the normalizers."""
        # Update the normalizers
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        diagnostics = extras.get("diagnostics", {})
        if self.transition.actions is not None:
            invalid_actions = diagnostics.get("action_invalid")
            if invalid_actions is not None and invalid_actions.any():
                # The environment replaces these actions with zero and
                # terminates the affected episodes. Keep the stored
                # transition consistent with the action that was actually
                # applied, and recompute its old log probability under the
                # already-produced policy distribution.
                self.transition.actions[invalid_actions] = 0.0
                safe_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()
                self.transition.actions_log_prob[invalid_actions] = safe_log_prob[invalid_actions]

        if self.diagnostics_enabled:
            observation_diagnostics = diagnostics.get("observations", {})
            for name, tensor in observation_diagnostics.items():
                self._diagnose_tensor(name, tensor, env_dim=0)
            if self.transition.actions is not None:
                self._diagnose_tensor("actions", self.transition.actions, env_dim=0)

        # Record the rewards and dones
        # Note: We clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Compute the intrinsic rewards
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards

        # Bootstrapping on time outs
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),  # type: ignore
                1,
            )

        # Record the transition
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute return and advantage targets from stored transitions."""
        st = self.storage
        if self.diagnostics_enabled:
            for name, tensor in st.observations.items():
                self._diagnose_tensor(f"observation/{name}", tensor, env_dim=1)
            self._diagnose_tensor("rewards", st.rewards, env_dim=1)
            self._diagnose_tensor("values", st.values, env_dim=1)
            for name, tensor in obs.items():
                self._diagnose_tensor(f"last_observation/{name}", tensor, env_dim=0)
        # Compute values for the last step
        critic_hidden_state = self.critic.get_hidden_state()
        last_values = self.critic(obs).detach()
        # Restore the critic's hidden state so the next rollout is not affected by the forward pass
        self.critic.reset(hidden_state=critic_hidden_state)
        # Compute returns and advantages
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            # If we are at the last step, bootstrap the return value
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - st.dones[step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            # Return: R_t = A(s_t, a_t) + V(s_t)
            st.returns[step] = advantage + st.values[step]
        # Compute the advantages
        st.advantages = st.returns - st.values
        if self.diagnostics_enabled:
            self._diagnose_tensor("returns", st.returns, env_dim=1)
            self._diagnose_tensor("advantages", st.advantages, env_dim=1)
        # Normalize the advantages if per minibatch normalization is not used
        if not self.normalize_advantage_per_mini_batch:
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

    def update(self) -> dict[str, float]:
        """Run one PPO update, recovering from non-finite numerical failures."""
        if not self.recover_on_nonfinite:
            return self._update_impl()

        snapshot = self._make_update_snapshot()
        try:
            loss_dict = self._update_impl()
        except FloatingPointError as error:
            self._restore_update_snapshot(snapshot)
            self.recovery_count += 1
            self.consecutive_recoveries += 1
            print(
                f"[PPO recovery] Skipped non-finite update #{self.recovery_count}: {error}. "
                f"Learning rate reduced to {self.learning_rate:.6g}."
            )
            if self.max_consecutive_recoveries > 0 and self.consecutive_recoveries > self.max_consecutive_recoveries:
                raise FloatingPointError(
                    f"PPO exceeded max_consecutive_recoveries={self.max_consecutive_recoveries}; last failure: {error}"
                ) from error
            return {
                "value": 0.0,
                "surrogate": 0.0,
                "entropy": 0.0,
                "update_skipped": 1.0,
                "recovery_count": float(self.recovery_count),
                "consecutive_recoveries": float(self.consecutive_recoveries),
            }

        self.consecutive_recoveries = 0
        loss_dict["update_skipped"] = 0.0
        loss_dict["recovery_count"] = float(self.recovery_count)
        loss_dict["consecutive_recoveries"] = 0.0
        return loss_dict

    def _update_impl(self) -> dict[str, float]:
        """Run optimization epochs over stored batches and return mean losses."""
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None

        # Get mini-batch generator
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # Iterate over mini-batches
        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            # Check if we should normalize advantages per mini-batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

            # Perform symmetric augmentation if enabled
            if self.symmetry:
                self.symmetry.augment_batch(batch, original_batch_size)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: We need to do this because we updated the policy with new parameters
            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            # Note: We only keep the following tensors for the original samples in case of symmetry augmentation
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            # Compute KL divergence and adapt the learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate only on the main process
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
            surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            if self.strict_gradient_checks:
                for name, value in (
                    ("surrogate_loss", surrogate_loss),
                    ("value_loss", value_loss),
                    ("entropy", entropy),
                ):
                    if not torch.isfinite(value).all():
                        raise FloatingPointError(f"PPO {name} contains NaN/Inf during update")

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
            if self.strict_gradient_checks and not torch.isfinite(loss).all():
                raise FloatingPointError("PPO total loss contains NaN/Inf during update")

            # RND loss
            rnd_loss = self.rnd.compute_loss(batch.observations[:original_batch_size]) if self.rnd else None  # type: ignore
            if self.strict_gradient_checks and rnd_loss is not None and not torch.isfinite(rnd_loss).all():
                raise FloatingPointError("PPO RND loss contains NaN/Inf during update")

            # Symmetry loss
            if self.symmetry:
                symmetry_loss = self.symmetry.compute_loss(self.actor, batch, original_batch_size)
                if self.strict_gradient_checks and not torch.isfinite(symmetry_loss).all():
                    raise FloatingPointError("PPO symmetry loss contains NaN/Inf during update")
                if self.symmetry.use_mirror_loss:
                    loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss

            # Compute the gradients for PPO
            self.optimizer.zero_grad()
            loss.backward()
            # Compute the gradients for RND
            if self.rnd:
                self.rnd.optimizer.zero_grad()
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO
            if self.strict_gradient_checks:
                self._check_gradients(self.actor, "Actor")
                self._check_gradients(self.critic, "Critic")
                if self.rnd:
                    self._check_gradients(self.rnd, "RND")
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                self.max_grad_norm,
                error_if_nonfinite=False,
            )
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.critic.parameters(),
                self.max_grad_norm,
                error_if_nonfinite=False,
            )
            if self.strict_gradient_checks and (
                not torch.isfinite(actor_grad_norm).all() or not torch.isfinite(critic_grad_norm).all()
            ):
                raise FloatingPointError("PPO gradient norm contains NaN/Inf before optimizer step")
            self.optimizer.step()
            if self.strict_gradient_checks:
                self._check_parameters(self.actor, "Actor")
                self._check_parameters(self.critic, "Critic")
            # Apply the gradients for RND
            if self.rnd:
                self.rnd.optimizer.step()
                if self.strict_gradient_checks:
                    self._check_parameters(self.rnd, "RND")

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        # Construct the loss dictionary
        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        # Clear the storage
        self.storage.clear()

        return loss_dict

    def train_mode(self) -> None:
        """Set train mode for learnable models."""
        self.actor.train()
        self.critic.train()
        if self.rnd:
            self.rnd.train()

    def eval_mode(self) -> None:
        """Set evaluation mode for learnable models."""
        self.actor.eval()
        self.critic.eval()
        if self.rnd:
            self.rnd.eval()

    def save(self) -> dict:
        """Return a dict of all models for saving."""
        saved_dict = {
            "actor_state_dict": self._raw_actor.state_dict(),
            "critic_state_dict": self._raw_critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.rnd:
            saved_dict["rnd_state_dict"] = self.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.rnd.optimizer.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models from a saved dict."""
        # If no load_cfg is provided, load all models and states
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": True,
                "iteration": True,
                "rnd": True,
            }

        # Load the specified models
        if load_cfg.get("actor"):
            self._raw_actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critic"):
            self._raw_critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        if load_cfg.get("optimizer"):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        if load_cfg.get("rnd") and self.rnd:
            self.rnd.load_state_dict(loaded_dict["rnd_state_dict"], strict=strict)
            self.rnd.optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        return load_cfg.get("iteration", False)

    def get_policy(self) -> MLPModel:
        """Get the policy model."""
        return self._raw_actor

    def compile(self, mode: str | None = None) -> None:
        """Compile actor and critic with ``torch.compile``.

        See :func:`~rsl_rl.utils.compile_model` for the set of accepted modes.

        Args:
            mode: ``torch.compile`` mode. Defaults to ``None``, in which case compilation is disabled.
        """
        self.actor = compile_model(self._raw_actor, mode)  # type: ignore
        self.critic = compile_model(self._raw_critic, mode)  # type: ignore

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPO:
        """Construct the PPO algorithm."""
        # Resolve class callables
        alg_class: type[PPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        # Resolve observation groups
        default_sets = ["actor", "critic"]
        if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # Resolve RND config if used
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)

        # Resolve symmetry config if used
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        # Initialize the policy
        actor: MLPModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        actor.model_name = "Actor"
        print(f"Actor Model: {actor}")
        if cfg["algorithm"].pop("share_cnn_encoders", None):  # Share CNN encoders between actor and critic
            cfg["critic"]["cnns"] = actor.cnns  # type: ignore
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        critic.model_name = "Critic"
        print(f"Critic Model: {critic}")

        # Initialize the storage
        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        # Initialize the algorithm
        alg: PPO = alg_class(actor, critic, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"])

        # Compile the algorithm's models if requested
        alg.compile(cfg.get("torch_compile_mode"))

        return alg

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        # Obtain the model parameters on current GPU
        model_params = [self._raw_actor.state_dict(), self._raw_critic.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        # Broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # Load the model parameters on all GPUs from source GPU
        self._raw_actor.load_state_dict(model_params[0])
        self._raw_critic.load_state_dict(model_params[1])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[2])

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        all_params = chain(self.actor.parameters(), self.critic.parameters())
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())
        all_params = list(all_params)
        grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
        all_grads = torch.cat(grads)
        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # Copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # Update the offset for the next parameter
                offset += numel
