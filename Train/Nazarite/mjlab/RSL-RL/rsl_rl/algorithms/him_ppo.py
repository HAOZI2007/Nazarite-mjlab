# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""HIM-enhanced PPO implementation for the modern rsl_rl API.

The estimator lifecycle intentionally follows native HIMLoco: its own
supervised/contrastive optimizer is stepped before the PPO step and receives
the current adaptive PPO learning rate.  Estimator outputs are detached in
the actor, so the PPO loss does not apply a second estimator gradient.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import chain

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.models import HIMActorModel, MLPModel
from rsl_rl.storage import HIMRolloutStorage
from rsl_rl.utils import reduce_gradients_in_buckets, resolve_class, resolve_obs_groups, resolve_optimizer


class HIMPPO:
    """PPO with the native HIM estimator update lifecycle."""

    def __init__(
        self,
        actor: HIMActorModel,
        critic: MLPModel,
        storage: HIMRolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        estimator_learning_rate: float | None = None,
        max_grad_norm: float = 1.0,
        estimator_max_grad_norm: float | None = None,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        use_mixed_precision: bool = False,
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
        grad_reduce_bucket_mb: float = 25,
        estimator_obs_groups: list[str] | None = None,
        estimator_velocity_slice: tuple[int, int] | list[int] | None = None,
        estimator_target_slice: tuple[int, int] | list[int] | None = None,
        estimator_target_slices: tuple[tuple[int, int], ...] | list[tuple[int, int]] | None = None,
        **kwargs,
    ) -> None:
        """Initialize PPO and the HIM estimator training configuration."""
        self.use_mixed_precision = use_mixed_precision
        del kwargs

        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        self.gpu_global_rank = int(multi_gpu_cfg.get("global_rank", 0)) if multi_gpu_cfg else 0
        self.gpu_world_size = int(multi_gpu_cfg.get("world_size", 1)) if multi_gpu_cfg else 1
        self.grad_reduce_bucket_mb = grad_reduce_bucket_mb

        self.actor = actor.to(device)
        self.critic = critic.to(device)
        if self.actor.is_recurrent or self.critic.is_recurrent:
            raise ValueError("HIMPPO currently supports feed-forward HIM actor and critic models only")
        self._raw_actor = self.actor
        self._raw_critic = self.critic
        self.estimator = self.actor.estimator
        # Keep the runner/logger contract even when RND is not configured.
        self.rnd = None
        self.intrinsic_rewards = None

        # Native HIMLoco keeps estimator parameters in the PPO optimizer.  The
        # estimator also owns a second optimizer for its HIM objective; since
        # HIMActorModel detaches estimator outputs, PPO contributes no gradient
        # to these parameters, matching the original implementation.
        self.optimizer = resolve_optimizer(optimizer)(
            chain(self.actor.parameters(), self.critic.parameters()),
            lr=learning_rate,
        )
        # This is only the estimator optimizer's initial LR.  Each mini-batch
        # update below synchronizes it to the current PPO LR, as in HIMLoco.
        if estimator_learning_rate is not None:
            self.estimator.learning_rate = float(estimator_learning_rate)
            for param_group in self.estimator.optimizer.param_groups:
                param_group["lr"] = self.estimator.learning_rate
        if estimator_max_grad_norm is not None:
            self.estimator.max_grad_norm = float(estimator_max_grad_norm)

        self.storage = storage
        self.transition = HIMRolloutStorage.Transition()
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        if self.num_learning_epochs < 1 or self.num_mini_batches < 1:
            raise ValueError("num_learning_epochs and num_mini_batches must be positive")
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
        if isinstance(estimator_obs_groups, str):
            estimator_obs_groups = [estimator_obs_groups]
        self.estimator_obs_groups = list(estimator_obs_groups or [])
        self.estimator_velocity_slice = self._normalize_slice(estimator_velocity_slice)
        self.estimator_target_slices = self._normalize_target_slices(
            estimator_target_slice,
            estimator_target_slices,
        )

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions and record the current transition."""
        self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic.get_hidden_state())
        self.transition.actions = self.actor(obs, stochastic_output=True).detach()
        self.transition.values = self.critic(obs).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()
        self.transition.distribution_params = tuple(param.detach() for param in self.actor.output_distribution_params)
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Record rewards, termination flags, and the post-step observation."""
        self.transition.next_observations, self.transition.next_observations_valid = self._resolve_next_observations(
            obs, dones, extras
        )
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones.clone()
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
            )
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute generalized advantage and return targets."""
        last_values = self.critic(obs).detach()
        advantage = 0
        for step in reversed(range(self.storage.num_transitions_per_env)):
            next_values = (
                last_values
                if step == self.storage.num_transitions_per_env - 1
                else self.storage.values[step + 1]
            )
            next_is_not_terminal = 1.0 - self.storage.dones[step].float()
            delta = (
                self.storage.rewards[step]
                + next_is_not_terminal * self.gamma * next_values
                - self.storage.values[step]
            )
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            self.storage.returns[step] = advantage + self.storage.values[step]
        self.storage.advantages = self.storage.returns - self.storage.values
        if not self.normalize_advantage_per_mini_batch:
            self.storage.advantages = (self.storage.advantages - self.storage.advantages.mean()) / (
                self.storage.advantages.std() + 1e-8
            )

    def update(self) -> dict[str, float]:
        """Optimize PPO and HIM losses over the collected rollout."""
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_estimation_loss = 0.0
        mean_swap_loss = 0.0
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

            self.actor(batch.observations, stochastic_output=True)
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations)
            distribution_params = self.actor.output_distribution_params
            entropy = self.actor.output_entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                self._update_learning_rate(batch.old_distribution_params, distribution_params)

            history = self.actor.get_observation_tensor(batch.observations)
            history = self.actor.obs_normalizer(history)
            velocity_target, target_obs = self._get_estimator_targets(batch.next_observations)
            valid_estimator_samples = batch.next_observations_valid.reshape(-1).bool()
            update_estimator = torch.tensor(
                int(valid_estimator_samples.any()), dtype=torch.int32, device=self.device
            )
            if self.is_multi_gpu:
                torch.distributed.all_reduce(update_estimator, op=torch.distributed.ReduceOp.MAX)
            if bool(update_estimator.item()):
                estimation_loss, swap_loss = self.estimator.update(
                    history[valid_estimator_samples],
                    velocity_target=velocity_target[valid_estimator_samples],
                    target_obs=target_obs[valid_estimator_samples],
                    # Native HIMLoco synchronizes the estimator optimizer
                    # with the adaptive PPO learning rate at every update.
                    learning_rate=self.learning_rate,
                    gradient_reducer=self._reduce_estimator_gradients if self.is_multi_gpu else None,
                )
            else:
                estimation_loss = 0.0
                swap_loss = 0.0

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self._all_parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_estimation_loss += estimation_loss
            mean_swap_loss += swap_loss

        num_updates = self.num_learning_epochs * self.num_mini_batches
        if num_updates < 1:
            raise ValueError("num_learning_epochs and num_mini_batches must be positive")
        observations = self.storage.observations.flatten(0, 1)
        self.actor.update_normalization(observations)
        self.critic.update_normalization(observations)
        self.storage.clear()
        return {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "estimation": mean_estimation_loss / num_updates,
            "swap": mean_swap_loss / num_updates,
        }

    def train_mode(self) -> None:
        """Set all HIM and PPO modules to training mode."""
        self.actor.train()
        self.critic.train()

    def eval_mode(self) -> None:
        """Set all HIM and PPO modules to evaluation mode."""
        self.actor.eval()
        self.critic.eval()

    def save(self) -> dict:
        """Return actor, critic, estimator, and optimizer states."""
        return {
            "actor_state_dict": self._raw_actor.state_dict(),
            "critic_state_dict": self._raw_critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "estimator_state_dict": self.estimator.state_dict(),
            "estimator_optimizer_state_dict": self.estimator.optimizer.state_dict(),
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load selected model and optimizer states from a checkpoint."""
        if load_cfg is None:
            load_cfg = {"actor": True, "critic": True, "optimizer": True, "estimator": True, "iteration": True}
        if load_cfg.get("actor"):
            self._load_actor_state(loaded_dict, strict, load_cfg.get("estimator", True))
        if load_cfg.get("critic"):
            self._load_critic_state(loaded_dict, strict)
        if load_cfg.get("optimizer") and "optimizer_state_dict" in loaded_dict:
            optimizer_state = loaded_dict["optimizer_state_dict"]
            # Older MjLab HIM checkpoints excluded estimator parameters from
            # this optimizer.  Keep their model weights loadable and start a
            # fresh PPO optimizer when the saved slot count is incompatible.
            current_count = len(self.optimizer.param_groups[0]["params"])
            saved_groups = optimizer_state.get("param_groups", [])
            saved_count = len(saved_groups[0].get("params", [])) if saved_groups else 0
            if current_count == saved_count:
                self.optimizer.load_state_dict(optimizer_state)
                self.learning_rate = self.optimizer.param_groups[0]["lr"]
        if load_cfg.get("estimator"):
            self._load_estimator_state(loaded_dict, strict)
            if "estimator_optimizer_state_dict" in loaded_dict:
                self.estimator.optimizer.load_state_dict(loaded_dict["estimator_optimizer_state_dict"])
        return bool(load_cfg.get("iteration", False))

    def get_policy(self) -> HIMActorModel:
        """Return the HIM actor used for inference."""
        return self._raw_actor

    def compile(self, mode: str | None = None) -> None:
        """Leave HIM uncompiled because its TensorDict estimator is stateful."""
        del mode

    def broadcast_parameters(self) -> None:
        """Broadcast actor, critic, and estimator parameters in distributed training."""
        if not self.is_multi_gpu:
            return
        for model in (self._raw_actor, self._raw_critic):
            for tensor in model.state_dict().values():
                torch.distributed.broadcast(tensor, src=0)

    def reduce_parameters(self) -> None:
        """Average PPO and estimator gradients across distributed workers."""
        if self.is_multi_gpu:
            reduce_gradients_in_buckets(
                self._all_parameters(),
                self.gpu_world_size,
                self.grad_reduce_bucket_mb,
            )

    def _reduce_estimator_gradients(self, parameters: Iterable[nn.Parameter]) -> None:
        """Average estimator gradients across distributed workers."""
        reduce_gradients_in_buckets(parameters, self.gpu_world_size, self.grad_reduce_bucket_mb)

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "HIMPPO":
        """Construct HIMPPO from the same configuration contract as modern PPO."""
        alg_class, alg_cfg = resolve_class(cfg["algorithm"])
        actor_class, actor_cfg = resolve_class(cfg["actor"])
        critic_class, critic_cfg = resolve_class(cfg["critic"])
        # Project-level dataclasses include optional fields for CNN/RNN and
        # custom DELTA models. Remove unset options before constructing the
        # plain HIM actor and MLP critic. This mirrors MjlabOnPolicyRunner's
        # config cleanup, which this standalone HIM runner does not inherit.
        for model_cfg in (actor_cfg, critic_cfg):
            for key in (
                "cnn_cfg",
                "distribution_cfg",
                "proprio_dim",
                "map_height",
                "map_width",
            ):
                if model_cfg.get(key) is None:
                    model_cfg.pop(key, None)
            for key in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
                if model_cfg.get("rnn_type") is None:
                    model_cfg.pop(key, None)
        default_sets = ["actor", "critic"]
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        actor: HIMActorModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg).to(device)
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **critic_cfg).to(device)
        storage = HIMRolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
        if not alg_cfg.get("estimator_obs_groups"):
            alg_cfg["estimator_obs_groups"] = list(cfg["obs_groups"]["critic"])
        algorithm = alg_class(
            actor,
            critic,
            storage,
            device=device,
            multi_gpu_cfg=cfg.get("multi_gpu"),
            **alg_cfg,
        )
        algorithm.compile(cfg.get("torch_compile_mode"))
        return algorithm

    def _all_parameters(self) -> Iterable[nn.Parameter]:
        """Yield all actor/critic parameters, including the HIM estimator."""
        return chain(self.actor.parameters(), self.critic.parameters())

    def _get_estimator_targets(self, next_observations: TensorDict | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Build velocity and one-step targets from post-step observation groups."""
        if next_observations is None:
            raise ValueError("HIM batches must contain next_observations")
        groups = self.estimator_obs_groups
        if not groups:
            groups = self.actor.obs_groups
        next_critic_obs = torch.cat([next_observations[group] for group in groups], dim=-1)
        velocity_slice = self.estimator_velocity_slice or (
            self.actor.num_one_step_obs,
            self.actor.num_one_step_obs + 3,
        )
        target_slices = self.estimator_target_slices or ((3, self.actor.num_one_step_obs + 3),)
        velocity_target = self._slice_features(next_critic_obs, velocity_slice)
        target_obs = torch.cat([self._slice_features(next_critic_obs, slc) for slc in target_slices], dim=-1)
        if velocity_target.shape[-1] != 3 or target_obs.shape[-1] != self.actor.num_one_step_obs:
            raise ValueError(
                "Invalid HIM estimator target configuration: expected velocity width 3 and target width "
                f"{self.actor.num_one_step_obs}, got {velocity_target.shape[-1]} and {target_obs.shape[-1]}"
            )
        return velocity_target, target_obs

    def _update_learning_rate(self, old_params, new_params) -> None:
        """Adapt PPO learning rate from the Gaussian KL divergence."""
        with torch.inference_mode():
            kl = self.actor.get_kl_divergence(old_params, new_params)
            kl_mean = torch.mean(kl)
            if self.is_multi_gpu:
                torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                kl_mean /= self.gpu_world_size
            if self.gpu_global_rank == 0:
                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)
            if self.is_multi_gpu:
                learning_rate = torch.tensor(self.learning_rate, device=self.device)
                torch.distributed.broadcast(learning_rate, src=0)
                self.learning_rate = learning_rate.item()
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

    def _resolve_next_observations(
        self,
        obs: TensorDict,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> tuple[TensorDict, torch.Tensor]:
        """Resolve true post-step observations and the samples valid for estimator training."""
        next_observations = obs.clone()
        done_mask = dones.reshape(-1).bool().to(self.device)
        valid = (~done_mask).view(-1, 1)
        terminal_observations = self._get_terminal_observations(extras)
        if terminal_observations is None or not done_mask.any():
            return next_observations, valid

        terminal_observations = terminal_observations.to(self.device)
        copied = False
        required_groups = self.estimator_obs_groups or self.actor.obs_groups
        terminal_batch = terminal_observations.batch_size[0] if terminal_observations.batch_size else 0
        for group, value in next_observations.items():
            if group not in terminal_observations:
                continue
            terminal_value = terminal_observations[group]
            if terminal_batch == next_observations.batch_size[0]:
                value[done_mask] = terminal_value[done_mask]
            elif terminal_batch == int(done_mask.sum().item()):
                value[done_mask] = terminal_value
            else:
                raise ValueError(
                    "terminal observations must contain either all environments or only done environments, got "
                    f"batch {terminal_batch} for {next_observations.batch_size[0]} envs"
                )
            copied = True
        if copied and all(group in terminal_observations for group in required_groups):
            valid[done_mask] = True
        return next_observations, valid

    @staticmethod
    def _get_terminal_observations(extras: dict) -> TensorDict | None:
        """Return terminal observations from common VecEnv extras keys, if present."""
        for key in ("terminal_observations", "terminal_observation", "final_observations", "final_observation"):
            value = extras.get(key)
            if isinstance(value, TensorDict):
                return value
            if isinstance(value, dict):
                tensors = list(value.values())
                if tensors and all(isinstance(item, torch.Tensor) for item in tensors):
                    return TensorDict(value, batch_size=[tensors[0].shape[0]])
        return None

    @staticmethod
    def _slice_features(tensor: torch.Tensor, slc: tuple[int, int]) -> torch.Tensor:
        start, stop = slc
        if start < 0 or stop <= start or stop > tensor.shape[-1]:
            raise ValueError(f"Invalid slice {(start, stop)} for tensor width {tensor.shape[-1]}")
        return tensor[:, start:stop]

    @staticmethod
    def _normalize_slice(value: tuple[int, int] | list[int] | None) -> tuple[int, int] | None:
        if value is None:
            return None
        if len(value) != 2:
            raise ValueError(f"Expected a two-element slice, got {value}")
        return int(value[0]), int(value[1])

    @classmethod
    def _normalize_target_slices(
        cls,
        target_slice: tuple[int, int] | list[int] | None,
        target_slices: tuple[tuple[int, int], ...] | list[tuple[int, int]] | None,
    ) -> tuple[tuple[int, int], ...] | None:
        if target_slices is not None:
            return tuple(cls._normalize_slice(slc) for slc in target_slices)  # type: ignore[arg-type]
        if target_slice is not None:
            if len(target_slice) == 2 and all(isinstance(item, int) for item in target_slice):
                return (cls._normalize_slice(target_slice),)  # type: ignore[return-value]
            return tuple(cls._normalize_slice(slc) for slc in target_slice)  # type: ignore[arg-type]
        return None

    def _load_actor_state(self, loaded_dict: dict, strict: bool, load_estimator: bool = True) -> None:
        if "actor_state_dict" in loaded_dict:
            state_dict = loaded_dict["actor_state_dict"]
            if not load_estimator:
                state_dict = {
                    key: value for key, value in state_dict.items() if not key.startswith("estimator.")
                }
            self._raw_actor.load_state_dict(state_dict, strict=strict if load_estimator else False)
            return
        if "model_state_dict" not in loaded_dict:
            if strict:
                raise KeyError("actor_state_dict")
            return
        state_dict = self._legacy_actor_state_dict(loaded_dict["model_state_dict"], load_estimator)
        self._raw_actor.load_state_dict(state_dict, strict=False if state_dict else strict)

    def _load_critic_state(self, loaded_dict: dict, strict: bool) -> None:
        if "critic_state_dict" in loaded_dict:
            self._raw_critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
            return
        if "model_state_dict" not in loaded_dict:
            if strict:
                raise KeyError("critic_state_dict")
            return
        state_dict = self._legacy_critic_state_dict(loaded_dict["model_state_dict"])
        self._raw_critic.load_state_dict(state_dict, strict=False if state_dict else strict)

    def _load_estimator_state(self, loaded_dict: dict, strict: bool) -> None:
        if "estimator_state_dict" in loaded_dict:
            self.estimator.load_state_dict(loaded_dict["estimator_state_dict"], strict=strict)
            return
        if "model_state_dict" not in loaded_dict:
            return
        state_dict = self._legacy_prefixed_state_dict(loaded_dict["model_state_dict"], "estimator.", "")
        if state_dict:
            self.estimator.load_state_dict(state_dict, strict=False)

    @staticmethod
    def _legacy_prefixed_state_dict(state_dict: dict, old_prefix: str, new_prefix: str) -> dict:
        return {
            f"{new_prefix}{key[len(old_prefix):]}": value
            for key, value in state_dict.items()
            if key.startswith(old_prefix)
        }

    def _legacy_actor_state_dict(self, state_dict: dict, load_estimator: bool = True) -> dict:
        actor_state = self._legacy_prefixed_state_dict(state_dict, "actor.", "mlp.")
        if load_estimator:
            actor_state.update(self._legacy_prefixed_state_dict(state_dict, "estimator.", "estimator."))
        if "std" in state_dict and hasattr(self._raw_actor.distribution, "std_param"):
            actor_state["distribution.std_param"] = state_dict["std"]
        return actor_state

    def _legacy_critic_state_dict(self, state_dict: dict) -> dict:
        return self._legacy_prefixed_state_dict(state_dict, "critic.", "mlp.")
