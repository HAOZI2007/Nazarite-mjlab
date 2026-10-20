"""On-policy WMP runner layered on top of the existing HIM PPO lifecycle."""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import fields
from typing import Any

import torch
from tensordict import TensorDict

from nazarite.rl.him_runner import HIMOnPolicyRunner
from nazarite.rl.wmp.amp import AdversarialMotionPrior
from nazarite.rl.wmp.replay import WorldModelReplay
from nazarite.rl.wmp.world_model import WMPWorldModel, WMPWorldModelCfg

# Proprioception arrives straight from the simulator, so one stiff contact can put
# a huge (or infinite) joint velocity in it. ``PropEncoder`` applies
# ``sign(x) * log1p(|x|)``, which keeps 1e6 finite but keeps infinity infinite; the
# RSSM then samples a categorical distribution whose logits are NaN and CUDA trips
# a device-side assert in multinomial, killing the run at an unrelated later line.
# The clamp is a no-op for real data: |qdot| above 100 rad/s means the simulation
# already failed, and log1p(100) = 4.6 still leaves the encoder its useful range.
_WM_PROP_LIMIT = 100.0
# A single non-finite feature would poison every downstream rollout. Zeroing it
# recovers from a transient bad step, but NaN weights would zero it every time,
# which is silent corruption rather than a crash: stop and say so instead.
_WM_MAX_BAD_FEATURES = 50


class WMPHIMOnPolicyRunner(HIMOnPolicyRunner):
    """Run HIM PPO with online visual RSSM features and a separate WM update."""

    def __init__(
        self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"
    ) -> None:
        super().__init__(env, train_cfg, log_dir, device)
        if self.is_distributed:
            raise NotImplementedError(
                "WMP-HIM currently supports single-process training only"
            )
        values = dict(train_cfg.get("world_model", {}))
        cfg_fields = {item.name for item in fields(WMPWorldModelCfg)}
        wm_cfg = WMPWorldModelCfg(
            **{key: value for key, value in values.items() if key in cfg_fields}
        )
        if wm_cfg.action_dim != self.env.num_actions:
            raise ValueError(
                f"World Model action_dim {wm_cfg.action_dim} != Go2 actions {self.env.num_actions}"
            )
        if wm_cfg.prop_dim != 33:
            raise ValueError(
                f"Go2 World Model prop_dim must be 33, got {wm_cfg.prop_dim}"
            )
        self.world_model = WMPWorldModel(wm_cfg).to(self.device)
        self.world_model_optimizer = self.world_model.optimizer
        chunk = int(wm_cfg.action_chunk)
        if chunk < 1:
            raise ValueError(f"World Model action_chunk must be >= 1, got {chunk}")
        self._wm_runtime_cfg = {
            "action_chunk": chunk,
            "capacity_per_env": int(values.get("capacity_per_env", 64)),
            "train_start_frames": int(values.get("train_start_frames", 4096)),
            "updates_per_iteration": int(values.get("updates_per_iteration", 1)),
            "batch_size": int(values.get("batch_size", 8)),
            "batch_length": int(values.get("batch_length", 32)),
            "pretrain_updates": int(values.get("pretrain_updates", 0)),
        }
        self.replay: WorldModelReplay | None = None
        self._wm_state: dict[str, torch.Tensor] | None = None
        self._wm_prev_action: torch.Tensor | None = None
        self._wm_prev_reward: torch.Tensor | None = None
        self._wm_is_first: torch.Tensor | None = None
        self._wm_action_hist: torch.Tensor | None = None
        self._wm_feature: torch.Tensor | None = None
        self._wm_step_index = 0
        self._wm_pretrained = False

        amp_values = dict(train_cfg.get("amp", {}))
        self.amp: AdversarialMotionPrior | None = None
        self._amp_runtime_cfg = {
            "updates_per_iteration": int(amp_values.get("updates_per_iteration", 1)),
        }
        if amp_values.get("enabled", False):
            self.amp = AdversarialMotionPrior(
                device=self.device,
                motion_dir=amp_values.get("motion_dir"),
                hidden_dims=tuple(amp_values.get("hidden_dims", (1024, 512))),
                style_coefficient=float(amp_values.get("style_coefficient", 0.15)),
                batch_size=int(amp_values.get("batch_size", 256)),
                learning_rate=float(amp_values.get("learning_rate", 1.0e-4)),
                grad_penalty=float(amp_values.get("grad_penalty", 10.0)),
                seed=amp_values.get("seed"),
            )
        self._amp_prev: torch.Tensor | None = None
        self._amp_style_sum = 0.0
        self._amp_style_steps = 0
        self._amp_pairs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def _new_replay(self) -> WorldModelReplay:
        cfg = self.world_model.cfg
        return WorldModelReplay(
            num_envs=self.env.num_envs,
            capacity_per_env=self._wm_runtime_cfg["capacity_per_env"],
            image_shape=(cfg.image_height, cfg.image_width, cfg.image_channels),
            prop_dim=cfg.prop_dim,
            action_dim=cfg.flat_action_dim,
            device=self.device,
        )

    @staticmethod
    def _prop_from_estimator(estimator_obs: torch.Tensor) -> torch.Tensor:
        if estimator_obs.ndim != 2 or estimator_obs.shape[-1] != 48:
            raise ValueError(
                f"Expected Go2 HIM estimator input [N, 48], got {tuple(estimator_obs.shape)}"
            )
        # [gyro, gravity, command, joint position, joint velocity, action, true velocity]
        # The world model is fed everything except the action and the true
        # velocity: 9 + 12 + 12. The gait clock used to sit at 9:11, which is why
        # the joint block starts at 9 rather than 11.
        return torch.cat((estimator_obs[:, :9], estimator_obs[:, 9:33]), dim=-1)

    def _camera_depth(self) -> torch.Tensor:
        """Read raw camera depth in metres."""
        sensor = self.env.unwrapped.scene["wmp_depth_camera"]
        depth = sensor.data.depth
        if depth is None:
            raise RuntimeError("WMP depth camera produced no depth tensor")
        expected = (self.env.num_envs, 64, 64, 1)
        if tuple(depth.shape) != expected:
            raise ValueError(
                f"Expected WMP depth shape {expected}, got {tuple(depth.shape)}"
            )
        return depth.to(self.device)

    def _wm_chunk_buffers(self, device: torch.device) -> None:
        """Create the action-fold buffers once, for training and inference alike."""
        chunk = int(self._wm_runtime_cfg["action_chunk"])
        # Compare ``torch.device`` objects, never a device with a string: the
        # training loop passes ``self.device`` (a str) while inference passes a
        # ``torch.device``, and ``torch.device("cuda:0") == "cuda:0"`` is False.
        # That mismatch silently recreated the buffers on every call, which reset
        # the step index and meant the chunk never filled: the world model was fed
        # one real action plus zero padding instead of a 5-step interval.
        target = torch.device(device)
        # ``getattr``: the buffers are created lazily, and a runner built for
        # playback-only (or by a test double) may never have run ``__init__``.
        hist = getattr(self, "_wm_action_hist", None)
        if hist is not None and hist.device == target and hist.shape[1] == chunk:
            return
        self._wm_action_hist = torch.zeros(
            (self.env.num_envs, chunk, self.env.num_actions), device=target
        )
        self._wm_feature = torch.zeros(
            (self.env.num_envs, self.world_model.feature_dim), device=target
        )
        self._wm_step_index = 0
        self._wm_bad_features = 0

    def _fold_wm_action(
        self,
        image: torch.Tensor,
        prop: torch.Tensor,
        action: torch.Tensor,
        is_first: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Fold one policy action into the current chunk, stepping the WM when full.

        Returns the feature to feed the actor (held between camera frames) and the
        stacked action vector when a world-model step actually happened, else
        ``None``.  Both the training loop and the inference policy go through this
        one function: duplicating the fold logic is how the play path ended up
        feeding a 12-dim action to a network that now expects 60.
        """
        self._wm_chunk_buffers(device)
        assert self._wm_action_hist is not None
        assert self._wm_feature is not None
        chunk = int(self._wm_runtime_cfg["action_chunk"])
        slot = self._wm_step_index % chunk
        self._wm_action_hist[:, slot].copy_(action)
        self._wm_step_index += 1
        # The first observation of a run has no history to fold, so it flushes a
        # partial chunk instead of leaving the policy blind for a full interval.
        if slot + 1 != chunk and self._wm_state is not None:
            return self._wm_feature, None
        stacked = self._wm_action_hist.reshape(self.env.num_envs, -1)
        # Sanitize before the encoder sees them: an infinite joint velocity would
        # otherwise reach the categorical sample in the RSSM.
        prop = torch.nan_to_num(
            prop, nan=0.0, posinf=_WM_PROP_LIMIT, neginf=-_WM_PROP_LIMIT
        ).clamp(-_WM_PROP_LIMIT, _WM_PROP_LIMIT)
        image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        self._wm_state = self.world_model.observe_step(
            image, prop, stacked, is_first, self._wm_state
        )
        feature = self.world_model.dynamics.feature(self._wm_state)
        if not bool(torch.isfinite(feature).all()):
            # Drop the history and hand the actor zeros for this interval: one
            # NaN feature becomes a NaN action, and from there a NaN policy.
            bad = int(getattr(self, "_wm_bad_features", 0)) + 1
            self._wm_bad_features = bad
            if bad > _WM_MAX_BAD_FEATURES:
                raise RuntimeError(
                    f"The world model produced non-finite features for {bad} "
                    "consecutive steps, so its weights are probably NaN. Look at "
                    "`world_model/loss` just before this point; a run that reaches "
                    "here cannot recover by zeroing the feature."
                )
            self._wm_state = None
            feature = torch.zeros_like(feature)
        else:
            self._wm_bad_features = 0
        self._wm_feature.copy_(feature)
        return self._wm_feature, stacked

    def _encode_current_observation(self, obs: TensorDict) -> TensorDict:
        """Advance the world model at the camera rate and cache its feature.

        The camera only refreshes every ``action_chunk`` policy steps, so the
        world model is stepped once per chunk with the actions taken inside it
        folded into one vector, and the feature is held in between.  Stepping it
        per policy step (what the 2026-10-17 runs did) trained the dynamics model
        on five identical frames per transition, which is why ``kl_dyn_rep`` sat
        on the free-bits ceiling instead of learning a 0.1 s transition.
        """
        assert self.replay is not None
        assert self._wm_prev_action is not None
        assert self._wm_prev_reward is not None
        assert self._wm_is_first is not None

        image = WMPWorldModel.normalize_depth(self._camera_depth())
        prop = self._prop_from_estimator(obs["estimator"])
        feature, stacked = self._fold_wm_action(
            image, prop, self._wm_prev_action, self._wm_is_first, self.device
        )
        if stacked is not None:
            image_u8 = ((image + 0.5).clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
            self.replay.add(
                image_u8,
                prop,
                stacked,
                self._wm_prev_reward,
                self._wm_is_first,
            )
            self._wm_is_first.zero_()

        augmented = obs.clone()
        augmented["world_model_feature"] = feature
        return augmented

    def _update_world_model(self) -> dict[str, float]:
        assert self.replay is not None
        values: list[dict[str, float]] = []
        for _ in range(self._wm_runtime_cfg["updates_per_iteration"]):
            if not self.replay.can_sample(self._wm_runtime_cfg["batch_length"]):
                break
            batch = self.replay.sample(
                self._wm_runtime_cfg["batch_size"], self._wm_runtime_cfg["batch_length"]
            )
            values.append(self.world_model.train_batch(batch))
        # Always report the bad-feature counter: when a world model stops updating
        # entirely the loss keys vanish from TensorBoard, and this card makes that
        # visible instead of leaving a silently shrinking group of six.
        metrics: dict[str, float] = {
            "world_model/bad_features": float(getattr(self, "_wm_bad_features", 0))
        }
        if values:
            metrics.update(
                {
                    f"world_model/{key}": sum(item[key] for item in values)
                    / len(values)
                    for key in values[0]
                }
            )
        return metrics

    def _update_amp(self) -> dict[str, float]:
        """One discriminator step per iteration, on this iteration's transitions."""
        if not self._amp_pairs:
            return {}
        current = torch.cat([pair[0] for pair in self._amp_pairs], dim=0)
        following = torch.cat([pair[1] for pair in self._amp_pairs], dim=0)
        valid = torch.cat([pair[2] for pair in self._amp_pairs], dim=0)
        self._amp_pairs.clear()
        if int(valid.sum()) < 2:
            return {}
        current = current[valid]
        following = following[valid]
        values = [
            self.amp.update(current, following)
            for _ in range(self._amp_runtime_cfg["updates_per_iteration"])
        ]
        metrics = {
            f"amp/{key}": sum(item[key] for item in values) / len(values)
            for key in values[0]
        }
        metrics["amp/bad_values"] = float(self.amp.bad_values)
        return metrics

    def _pretrain_world_model(self) -> dict[str, float]:
        """Train the freshly warmed-up world model in one burst.

        The replay buffer is only useful once it holds a few thousand frames, but
        by then the policy has already been updated a hundred times on a feature
        that is close to noise.  The reference implementation gives the world
        model ``pretrain`` updates at that moment; doing it here costs one
        longer iteration and shortens how long the policy is fed a stale feature.
        """
        updates = self._wm_runtime_cfg["pretrain_updates"]
        if updates <= 0:
            return {}
        values: list[dict[str, float]] = []
        for _ in range(updates):
            if not self.replay.can_sample(self._wm_runtime_cfg["batch_length"]):
                break
            batch = self.replay.sample(
                self._wm_runtime_cfg["batch_size"], self._wm_runtime_cfg["batch_length"]
            )
            values.append(self.world_model.train_batch(batch))
        if not values:
            return {}
        return {
            f"world_model/pretrain_{key}": sum(item[key] for item in values)
            / len(values)
            for key in values[0]
        }

    def learn(
        self, num_learning_iterations: int, init_at_random_ep_len: bool = False
    ) -> None:
        if self.replay is None:
            self.replay = self._new_replay()
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs = self.env.get_observations().to(self.device)
        self._wm_state = None
        self._wm_prev_action = torch.zeros(
            (self.env.num_envs, self.env.num_actions), device=self.device
        )
        self._wm_prev_reward = torch.zeros(self.env.num_envs, device=self.device)
        self._wm_is_first = torch.ones(self.env.num_envs, device=self.device)
        self._wm_action_hist = torch.zeros(
            (
                self.env.num_envs,
                self._wm_runtime_cfg["action_chunk"],
                self.env.num_actions,
            ),
            device=self.device,
        )
        self._wm_feature = torch.zeros(
            (self.env.num_envs, self.world_model.feature_dim), device=self.device
        )
        self._wm_step_index = 0
        self._wm_pretrained = False
        if self.amp is not None:
            self._amp_prev = obs["amp"].to(self.device).clone()
            self._amp_pairs.clear()
        policy_obs = self._encode_current_observation(obs)
        self.alg.train_mode()
        self.world_model.train()
        if self.is_distributed:
            self.alg.broadcast_parameters()
        self.logger.init_logging_writer()
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations

        for it in range(start_it, total_it):
            self.env.unwrapped.wmp_learning_iteration = it
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(policy_obs)
                    next_obs, rewards, dones, extras = self.env.step(
                        actions.to(self.env.device)
                    )
                    next_obs = next_obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    if self.amp is not None:
                        amp_now = next_obs["amp"].to(self.device)
                        # Deliberately *not* scaled by ``step_dt``. The reward
                        # manager folds dt into every task term, and the
                        # reference AMP adds its style reward on top of that
                        # result: 0.01 of style therefore competes with ~0.03 of
                        # tracking, a third of the objective. Scaling it by dt
                        # as well divides it by 50 and leaves it at one percent,
                        # where the policy simply ignores it (2026-10-20 run).
                        style = self.amp.style_reward(self._amp_prev, amp_now)
                        rewards = rewards + style
                        self._amp_style_sum += float(style.mean())
                        self._amp_style_steps += 1
                        # A reset teleports the robot, so a pair that straddles one
                        # is not a transition and must not train the discriminator.
                        self._amp_pairs.append(
                            (
                                self._amp_prev.detach().clone(),
                                amp_now.detach().clone(),
                                ~dones.bool(),
                            )
                        )
                        self._amp_prev = amp_now
                    self.alg.process_env_step(next_obs, rewards, dones, extras)
                    self.logger.process_env_step(
                        rewards, dones, extras, self.alg.intrinsic_rewards
                    )
                    self._wm_prev_action.copy_(actions.to(self.device))
                    self._wm_prev_action[dones.bool()] = 0.0
                    if dones.any():
                        # A reset starts a new chunk: folding the previous
                        # episode's actions into it would feed the dynamics model
                        # a transition that never happened.
                        self._wm_action_hist[dones.bool()] = 0.0
                    self._wm_prev_reward.copy_(rewards)
                    self._wm_is_first.copy_(dones.float())
                    obs = next_obs
                    policy_obs = self._encode_current_observation(obs)
                self.alg.compute_returns(policy_obs)
            collect_time = time.time() - start
            start = time.time()
            loss_dict = self.alg.update()
            loss_dict.update(self._update_amp())
            if self.amp is not None and self._amp_style_steps:
                # The style reward bypasses the reward manager, so without this
                # line nothing in the logs says how much of the return it is.
                loss_dict["amp/style_reward"] = (
                    self._amp_style_sum / self._amp_style_steps
                )
                self._amp_style_sum = 0.0
                self._amp_style_steps = 0
            if self.replay.total_frames >= self._wm_runtime_cfg["train_start_frames"]:
                loss_dict.update(self._update_world_model())
                if not self._wm_pretrained:
                    self._wm_pretrained = True
                    loss_dict.update(self._pretrain_world_model())
            self.current_learning_iteration = it
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=time.time() - start,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=None,
            )
            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                assert self.logger.log_dir is not None
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))
        if self.logger.writer is not None:
            assert self.logger.log_dir is not None
            self.save(
                os.path.join(
                    self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"
                )
            )
            self.logger.stop_logging_writer()

    def _clamp_learning_rate(self) -> None:
        """Keep the adaptive learning rate above ``min_learning_rate``.

        rsl-rl's own schedule clamps at 1e-5, and the 2026-10-18 run reached it
        after roughly 2500 of its 3592 iterations: the last third of the run
        contributed almost no policy change, which locked in the crouched gait
        that the reward rebalance is meant to remove.
        """
        floor = float(self.cfg.get("min_learning_rate", 1.0e-4))
        if self.alg.learning_rate >= floor:
            return
        self.alg.learning_rate = floor
        for group in self.alg.optimizer.param_groups:
            group["lr"] = floor

    def export_policy_to_onnx(
        self, path: str, filename: str = "policy.onnx", verbose: bool = False
    ) -> None:
        del path, filename, verbose
        warnings.warn(
            "WMP-HIM ONNX export is not available yet: deployment must include the "
            "depth encoder and recurrent World Model state update.",
            RuntimeWarning,
        )

    def save(self, path: str, infos: dict | None = None) -> None:
        super().save(path, infos)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint["world_model_state_dict"] = self.world_model.state_dict()
        checkpoint["world_model_optimizer_state_dict"] = (
            self.world_model_optimizer.state_dict()
        )
        checkpoint["world_model_config"] = vars(self.world_model.cfg)
        if self.amp is not None:
            checkpoint["amp_state_dict"] = self.amp.state_dict()
        torch.save(checkpoint, path)

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        infos = super().load(path, load_cfg, strict, map_location)
        checkpoint = torch.load(
            path, map_location=map_location or self.device, weights_only=False
        )
        if "world_model_state_dict" not in checkpoint:
            raise KeyError("Checkpoint does not contain world_model_state_dict")
        try:
            self.world_model.load_state_dict(
                checkpoint["world_model_state_dict"], strict=strict
            )
        except RuntimeError as error:
            raise RuntimeError(
                "World model weights do not fit this runner's configuration. The "
                "most likely cause is a checkpoint trained before action chunking: "
                "its RSSM consumed one 12-dim action per step, while the current "
                "config folds an action_chunk of camera-interval actions into a "
                "single transition. Retrain the stage instead of resuming it."
            ) from error
        if load_cfg is None or load_cfg.get("load_optimizer", True):
            state = checkpoint.get("world_model_optimizer_state_dict")
            if state is not None:
                self.world_model_optimizer.load_state_dict(state)
        self._load_amp_state(checkpoint)
        return infos

    def _load_amp_state(self, checkpoint: dict) -> None:
        """Restore the discriminator, loudly.

        The reference implementation leaves the discriminator out of its
        checkpoints, so a resumed run silently restarts it from random weights and
        the style reward changes meaning mid-training. Say so instead.
        """
        state = checkpoint.get("amp_state_dict")
        if self.amp is None:
            if state is not None:
                warnings.warn(
                    "Checkpoint carries AMP weights but this run has AMP disabled; "
                    "they are ignored.",
                    RuntimeWarning,
                )
            return
        if state is None:
            warnings.warn(
                "Checkpoint has no AMP weights: the discriminator, its optimiser and "
                "its normaliser restart from scratch.",
                RuntimeWarning,
            )
            return
        self.amp.load_state_dict(state)

    def get_inference_policy(self, device: str | None = None) -> Any:
        target_device = torch.device(device or self.device)
        self.alg.eval_mode()
        self.world_model.eval()
        self._wm_state = None
        self._wm_prev_action = torch.zeros(
            (self.env.num_envs, self.env.num_actions), device=target_device
        )
        prev_action = self._wm_prev_action
        assert prev_action is not None
        # Inference owns its own fold buffers so a runner that was never trained
        # in this process (a checkpoint loaded for playback) still works.
        self._wm_chunk_buffers(target_device)
        first = torch.ones(self.env.num_envs, device=target_device)

        def policy(
            obs: TensorDict,
            depth: torch.Tensor | None = None,
            is_first: torch.Tensor | None = None,
        ):
            nonlocal first
            # Both the live camera path and an explicitly supplied deployment
            # frame use raw metric depth. Normalize exactly once here.
            depth_tensor = (
                self._camera_depth() if depth is None else depth.to(target_device)
            )
            if depth_tensor.ndim == 3:
                depth_tensor = depth_tensor.unsqueeze(-1)
            depth_normalized = WMPWorldModel.normalize_depth(depth_tensor)
            prop = self._prop_from_estimator(obs["estimator"].to(target_device))
            reset_mask = (
                first if is_first is None else is_first.to(target_device).float()
            )
            with torch.inference_mode():
                # Same fold-and-hold path as training. The play loop used to step
                # the world model with a single 12-dim action, which crashed once
                # the network learned to expect a folded 5x12 camera interval.
                feature, _ = self._fold_wm_action(
                    depth_normalized,
                    prop,
                    prev_action,
                    reset_mask,
                    target_device,
                )
                augmented = obs.to(target_device).clone()
                augmented["world_model_feature"] = feature
                actions = self.alg.get_policy().act_inference(augmented)
            prev_action.copy_(actions)
            first = torch.zeros_like(first)
            return actions

        def reset() -> None:
            nonlocal first
            self._wm_state = None
            self._wm_step_index = 0
            if self._wm_action_hist is not None:
                self._wm_action_hist.zero_()
            prev_action.zero_()
            first = torch.ones_like(first)

        policy.reset = reset  # type: ignore[attr-defined]
        return policy
