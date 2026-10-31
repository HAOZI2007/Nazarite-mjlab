"""XZ1 HIM terrain teacher, following Extreme Parkour's scan MLP."""

import copy
import json
import math
from pathlib import Path
from typing import cast

import torch
from rsl_rl.models.him_actor_model import HIMActorModel
from rsl_rl.modules import MLP
from tensordict import TensorDict
from torch import nn

from nazarite.mdp.xz1_diagnostics import capture_failure_scenes
from nazarite.rl.him_runner import HIMOnPolicyRunner


def teacher_scan_contract(history_length=1, update_period=0.02):
    contract = {
        "points": 132,
        "latent": 32,
        "offset": 0.3,
        "clip": [-1.0, 1.0],
        "order": "x_major_y_minor",
        "privileged_not_camera": True,
    }
    if history_length > 1:
        contract.update(
            {
                "history_length": history_length,
                "history_order": "oldest_first",
                "latent_order": "current_first",
                "history_dt": update_period,
                "reset_fill": "repeat_first",
            }
        )
    # Preserve the exact metadata of existing 50 Hz single/six-frame policies.
    if update_period != 0.02:
        contract["update_period"] = update_period
    return contract


class XZ1TerrainTeacherActor(HIMActorModel):
    """Keep HIM unchanged; append 32-D embeddings for each terrain frame.

    Reference: extreme-parkour/rsl_rl/modules/actor_critic.py, Actor.scan_encoder.
    Unlike the later visual student this teacher requires oracle height scans.
    Raw temporal scans travel with the rollout observations so PPO can recompute
    all embeddings with the same current encoder, without stale latent caches.
    """

    def __init__(
        self,
        *args,
        scan_group: str = "teacher_scan",
        scan_history_length: int = 1,
        scan_update_period: float = 0.02,
        **kwargs,
    ):
        if scan_history_length < 1:
            raise ValueError("scan_history_length must be positive")
        self.scan_history_length = scan_history_length
        if not math.isfinite(scan_update_period) or scan_update_period <= 0:
            raise ValueError("scan_update_period must be finite and positive")
        self.scan_update_period = scan_update_period
        super().__init__(*args, **kwargs)
        self.scan_group = scan_group
        self.scan_encoder = nn.Sequential(
            nn.Linear(132, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.Tanh(),
        )
        if self.scan_history_length > 1:
            self.mlp = MLP(
                self.num_one_step_obs
                + 3
                + self.estimator.num_latent
                + 32 * self.scan_history_length,
                self.distribution.input_dim,
                kwargs.get("hidden_dims", (512, 256, 128)),
                kwargs.get("activation", "elu"),
            )
            self.distribution.init_mlp_weights(self.mlp)

    @staticmethod
    def _actor_input_dim(input_dim: int) -> int:
        return input_dim + 32

    def get_latent(self, obs: TensorDict, masks=None, hidden_state=None):
        latent = super().get_latent(obs, masks, hidden_state)
        scan = obs[self.scan_group]
        expected = (
            (132,) if self.scan_history_length == 1 else (self.scan_history_length, 132)
        )
        if tuple(scan.shape[1:]) != expected:
            raise ValueError(
                f"Expected [batch, {expected}] teacher scan, received {scan.shape}"
            )
        encoded = self.scan_encoder(scan)
        if self.scan_history_length > 1:
            encoded = encoded.flip(-2).flatten(-2)
        return torch.cat((latent, encoded), dim=-1)

    def as_onnx(self, verbose: bool = False):
        del verbose
        return _TeacherExport(self)

    def as_jit(self):
        return _TeacherExport(self)


class _TeacherExport(nn.Module):
    """Two-input export; oracle scan must never disappear at deployment."""

    input_names = ("obs_history", "teacher_scan")
    output_names = ("actions",)
    bounds: torch.Tensor

    def __init__(self, actor: XZ1TerrainTeacherActor):
        super().__init__()
        self.obs_normalizer = copy.deepcopy(actor.obs_normalizer)
        self.estimator_encoder = copy.deepcopy(actor.estimator.encoder)
        self.scan_encoder = copy.deepcopy(actor.scan_encoder)
        self.mlp = copy.deepcopy(actor.mlp)
        self.output = copy.deepcopy(
            cast(nn.Module, actor.distribution.as_deterministic_output_module())
        )
        self.frame_size = actor.num_one_step_obs
        self.history_dim = actor.obs_dim
        self.action_clip = actor.action_clip
        self.scan_history_length = actor.scan_history_length
        self.register_buffer(
            "bounds", cast(torch.Tensor, actor.observation_bounds).detach().clone()
        )

    def forward(self, obs_history: torch.Tensor, teacher_scan: torch.Tensor):
        history = self.obs_normalizer(obs_history.clamp(-self.bounds, self.bounds))
        estimate = self.estimator_encoder(history)
        velocity = estimate[..., :3]
        latent = nn.functional.normalize(estimate[..., 3:], p=2.0, dim=-1)
        terrain = self.scan_encoder(teacher_scan)
        if self.scan_history_length > 1:
            terrain = terrain.flip(-2).flatten(-2)
        inputs = torch.cat(
            (
                history[..., : self.frame_size],
                velocity,
                latent,
                terrain,
            ),
            dim=-1,
        )
        actions = self.output(self.mlp(inputs))
        if self.action_clip is not None:
            actions = actions.clamp(-self.action_clip, self.action_clip)
        return actions

    def get_dummy_inputs(self):
        scan = (
            torch.zeros(1, 132)
            if self.scan_history_length == 1
            else torch.zeros(1, self.scan_history_length, 132)
        )
        return torch.zeros(1, self.history_dim), scan


def expand_him_state(actor: XZ1TerrainTeacherActor, source: dict) -> dict:
    """Strict warm start: preserve old policy exactly with zero scan columns."""
    target = actor.state_dict()
    expected = {key for key in target if not key.startswith("scan_encoder.")}
    if set(source) != expected:
        raise ValueError(
            "Warm start requires an unmodified XZ1 HIM actor; "
            f"missing={expected - set(source)}, extra={set(source) - expected}"
        )
    merged = dict(target)
    for key, value in source.items():
        if not torch.isfinite(value).all():
            raise ValueError(f"Non-finite warm-start tensor: {key}")
        if key == "mlp.0.weight":
            if value.shape != (target[key].shape[0], 66):
                raise ValueError("Warm-start actor must have the XZ1 66-D HIM input")
            value = torch.cat(
                (
                    value,
                    value.new_zeros(value.shape[0], 32 * actor.scan_history_length),
                ),
                dim=1,
            )
        if value.shape != target[key].shape:
            raise ValueError(f"Warm-start shape mismatch for {key}: {value.shape}")
        merged[key] = value
    return merged


def expand_teacher_state(actor: XZ1TerrainTeacherActor, source: dict) -> dict:
    """Preserve current-frame actions when adding older scan features."""
    target = actor.state_dict()
    if set(source) != set(target):
        raise ValueError("Warm-start teacher actor keys differ")
    merged = {}
    for key, value in source.items():
        if not torch.isfinite(value).all():
            raise ValueError(f"Invalid warm-start teacher tensor: {key}")
        if key == "mlp.0.weight" and value.shape != target[key].shape:
            if (
                value.shape[0] != target[key].shape[0]
                or value.shape[1] != 98
                or actor.scan_history_length <= 1
            ):
                raise ValueError("Only single-frame teacher expansion is supported")
            value = torch.cat(
                (
                    value,
                    value.new_zeros(
                        value.shape[0], target[key].shape[1] - value.shape[1]
                    ),
                ),
                dim=1,
            )
        if value.shape != target[key].shape:
            raise ValueError(f"Invalid warm-start teacher tensor: {key}")
        merged[key] = value
    return merged


class XZ1TerrainTeacherRunner(HIMOnPolicyRunner):
    """Separate weight-only HIM/teacher warm starts from exact resumption."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        checkpoint = train_cfg.get("initial_checkpoint")
        if train_cfg.get("resume"):
            checkpoint = None  # Explicit resume takes priority over task defaults.
        super().__init__(env, train_cfg, log_dir, device)
        actor = cast(XZ1TerrainTeacherActor, self.alg.get_policy())
        scan_cfg = self.env.unwrapped.cfg.observations[actor.scan_group]
        params = scan_cfg.terms["height_scan"].params
        env_length = params.get("history_length", scan_cfg.history_length or 1)
        env_period = params.get("update_period", self.env.unwrapped.step_dt)
        if env_length != actor.scan_history_length or not math.isclose(
            env_period, actor.scan_update_period, abs_tol=1e-9
        ):
            raise ValueError("Teacher environment and actor scan timing/history differ")
        self.warm_start_source = None
        if checkpoint:
            self.warm_start(checkpoint)

    def _numerical_contract(self):
        actor = cast(XZ1TerrainTeacherActor, self.alg.get_policy())
        return {
            **super()._numerical_contract(),
            "teacher_scan": teacher_scan_contract(
                actor.scan_history_length, actor.scan_update_period
            ),
        }

    def warm_start(self, path: str):
        """Load HIM/teacher weights; neither optimizer nor curricula are resumed."""
        if (
            self.current_learning_iteration
            or self.alg.optimizer.state
            or self.alg.estimator.optimizer.state
        ):
            raise ValueError("Warm start requires a fresh runner/optimizers")
        source = torch.load(path, map_location=self.device, weights_only=False)
        contract = (source.get("infos") or {}).get("him_numerics")
        expected = HIMOnPolicyRunner._numerical_contract(self)
        legacy_teacher = {**expected, "teacher_scan": teacher_scan_contract()}
        from_teacher = contract in (self._numerical_contract(), legacy_teacher)
        if contract != expected and not from_teacher:
            raise ValueError("Warm-start HIM numerical/history contract mismatch")
        actor = self.alg.get_policy()
        if not isinstance(actor, XZ1TerrainTeacherActor):
            raise TypeError("Teacher runner requires XZ1TerrainTeacherActor")
        if from_teacher:
            actor_state = expand_teacher_state(actor, source["actor_state_dict"])
        else:
            actor_state = expand_him_state(actor, source["actor_state_dict"])
        critic_state = source["critic_state_dict"]
        current = self.alg.critic.state_dict()
        if set(critic_state) != set(current):
            raise ValueError("Warm-start critic keys differ")
        for key, value in critic_state.items():
            if value.shape != current[key].shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid warm-start critic tensor: {key}")
        self.alg.actor.load_state_dict(actor_state, strict=True)
        self.alg.critic.load_state_dict(critic_state, strict=True)
        # actor_state contains estimator encoder, target and prototypes. Do NOT
        # call HIMPPO.load: it also loads estimator Adam state independently.
        self._validate_numerics()
        self.warm_start_source = str(path)
        print(f"XZ1 teacher weight-only warm start: {path}; new iteration = 0")

    def _terrain_names(self):
        terrain = self.env.unwrapped.scene.terrain
        assert terrain is not None
        generator = terrain.cfg.terrain_generator
        assert generator is not None
        return list(generator.sub_terrains)

    def _curriculum_contract(self):
        terms = self.env.unwrapped.cfg.curriculum
        if "terrain_levels" not in terms:
            return None
        terrain = self.env.unwrapped.scene.terrain
        assert terrain is not None
        return {
            "version": 4,
            "gap_failure": "actual_foot_pit_bottom_contact",
            "type_assignment": "fixed",
            "randomize_completed_levels": terms["terrain_levels"].params.get(
                "randomize_completed_levels", False
            ),
            "terrain_names": self._terrain_names(),
            "num_rows": terrain.max_terrain_level,
        }

    def _record_nonfinite_envs(self, env_ids, observations):
        env = self.env.unwrapped
        diagnostics = getattr(env, "_xz1_course_diagnostics", None)
        pending = diagnostics.pending_scenes if diagnostics is not None else {}
        scenes = {
            index: pending.pop(index) for index in env_ids.tolist() if index in pending
        }
        missing = [index for index in env_ids.tolist() if index not in scenes]
        if missing:
            scenes.update(
                capture_failure_scenes(
                    env,
                    torch.tensor(missing, device=env.device),
                    phase="before_recovery",
                )
            )
        if diagnostics is not None:
            diagnostics.record_recovery(env_ids)
        log_dir = self.logger.log_dir
        if log_dir is None:
            return
        directory = Path(log_dir) / "numerics"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (
            f"recovery_{self._nan_recoveries:06d}_step_{env.common_step_counter:010d}.pt"
        )
        evidence = {
            "version": 1,
            "iteration": getattr(
                self, "_rollout_iteration", self.current_learning_iteration
            ),
            "rollout_step": getattr(self, "_rollout_step", None),
            "scenes": scenes,
            "observations": observations[env_ids].detach().cpu().clone(),
        }
        torch.save(evidence, path)
        print(f"[HIM recovery] saved {len(scenes)} environment scenes to {path}")

    def _log_training_diagnostics(self, iteration):
        diagnostics = getattr(self.env.unwrapped, "_xz1_course_diagnostics", None)
        if diagnostics is None or self.logger.writer is None:
            return
        for name, value in diagnostics.scalars().items():
            self.logger.writer.add_scalar(name, value, iteration)
        if iteration % self.cfg["save_interval"] == 0:
            assert self.logger.log_dir is not None
            path = Path(self.logger.log_dir) / "terrain_outcomes.json"
            path.write_text(
                json.dumps(
                    {
                        "iteration": iteration,
                        "terrain_names": diagnostics.names,
                        "columns": [
                            "episodes",
                            "success",
                            "timeout",
                            "numerical_failure",
                        ],
                        "counts_by_type_and_level": diagnostics.counts.cpu().tolist(),
                    },
                    indent=2,
                )
            )

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        source = torch.load(path, map_location="cpu", weights_only=False)
        if (source.get("infos") or {}).get(
            "him_numerics"
        ) != self._numerical_contract():
            raise ValueError(
                "Teacher scan/history contract differs; use initial_checkpoint "
                "for a single-frame teacher weight-only warm start"
            )
        if load_cfg is None or load_cfg.get("iteration", True):
            names = self._terrain_names()
            if (source.get("infos") or {}).get("xz1_terrain_names") != names:
                raise ValueError(
                    "Teacher resume terrain layout differs or is unversioned; "
                    "use initial_checkpoint for a weight-only warm start"
                )
            contract = self._curriculum_contract()
            if (
                contract is not None
                and (source.get("infos") or {}).get("xz1_course_curriculum") != contract
            ):
                raise ValueError(
                    "Teacher curriculum contract changed; use initial_checkpoint "
                    "for a weight-only warm start with fresh curriculum"
                )
        infos = super().load(path, load_cfg, strict, map_location)
        self.warm_start_source = (infos or {}).get("teacher_warm_start")
        return infos

    def save(self, path, infos=None):
        names = self._terrain_names()
        super().save(
            path,
            {
                **(infos or {}),
                "teacher_warm_start": self.warm_start_source,
                "xz1_terrain_names": names,
                "xz1_course_curriculum": self._curriculum_contract(),
            },
        )
