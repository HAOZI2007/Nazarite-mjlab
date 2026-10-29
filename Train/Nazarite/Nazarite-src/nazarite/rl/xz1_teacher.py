"""XZ1 HIM terrain teacher, following Extreme Parkour's scan MLP."""

import copy
import json
from pathlib import Path
from typing import cast

import torch
from rsl_rl.models.him_actor_model import HIMActorModel
from tensordict import TensorDict
from torch import nn

from nazarite.mdp.xz1_diagnostics import capture_failure_scenes
from nazarite.rl.him_runner import HIMOnPolicyRunner


class XZ1TerrainTeacherActor(HIMActorModel):
    """Keep HIM unchanged; append a 32-D privileged terrain embedding.

    Reference: extreme-parkour/rsl_rl/modules/actor_critic.py, Actor.scan_encoder.
    Unlike the later visual student this teacher requires oracle height scans.
    """

    def __init__(self, *args, scan_group: str = "teacher_scan", **kwargs):
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

    @staticmethod
    def _actor_input_dim(input_dim: int) -> int:
        return input_dim + 32

    def get_latent(self, obs: TensorDict, masks=None, hidden_state=None):
        latent = super().get_latent(obs, masks, hidden_state)
        scan = obs[self.scan_group]
        if scan.ndim != 2 or scan.shape[-1] != 132:
            raise ValueError(
                f"Expected [batch, 132] teacher scan, received {scan.shape}"
            )
        return torch.cat((latent, self.scan_encoder(scan)), dim=-1)

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
        self.register_buffer(
            "bounds", cast(torch.Tensor, actor.observation_bounds).detach().clone()
        )

    def forward(self, obs_history: torch.Tensor, teacher_scan: torch.Tensor):
        history = self.obs_normalizer(obs_history.clamp(-self.bounds, self.bounds))
        estimate = self.estimator_encoder(history)
        velocity = estimate[..., :3]
        latent = nn.functional.normalize(estimate[..., 3:], p=2.0, dim=-1)
        inputs = torch.cat(
            (
                history[..., : self.frame_size],
                velocity,
                latent,
                self.scan_encoder(teacher_scan),
            ),
            dim=-1,
        )
        actions = self.output(self.mlp(inputs))
        if self.action_clip is not None:
            actions = actions.clamp(-self.action_clip, self.action_clip)
        return actions

    def get_dummy_inputs(self):
        return torch.zeros(1, self.history_dim), torch.zeros(1, 132)


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
            value = torch.cat((value, value.new_zeros(value.shape[0], 32)), dim=1)
        if value.shape != target[key].shape:
            raise ValueError(f"Warm-start shape mismatch for {key}: {value.shape}")
        merged[key] = value
    return merged


class XZ1TerrainTeacherRunner(HIMOnPolicyRunner):
    """Separate weight-only HIM/teacher warm starts from exact resumption."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        checkpoint = train_cfg.get("initial_checkpoint")
        if checkpoint and train_cfg.get("resume"):
            raise ValueError("Choose initial_checkpoint OR resume, not both")
        super().__init__(env, train_cfg, log_dir, device)
        self.warm_start_source = None
        if checkpoint:
            self.warm_start(checkpoint)

    def _numerical_contract(self):
        return {
            **super()._numerical_contract(),
            "teacher_scan": {
                "points": 132,
                "latent": 32,
                "offset": 0.3,
                "clip": [-1.0, 1.0],
                "order": "x_major_y_minor",
                "privileged_not_camera": True,
            },
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
        from_teacher = contract == self._numerical_contract()
        if contract != expected and not from_teacher:
            raise ValueError("Warm-start HIM numerical/history contract mismatch")
        actor = self.alg.get_policy()
        if not isinstance(actor, XZ1TerrainTeacherActor):
            raise TypeError("Teacher runner requires XZ1TerrainTeacherActor")
        if from_teacher:
            actor_state = source["actor_state_dict"]
            target = actor.state_dict()
            if set(actor_state) != set(target):
                raise ValueError("Warm-start teacher actor keys differ")
            for key, value in actor_state.items():
                if value.shape != target[key].shape or not torch.isfinite(value).all():
                    raise ValueError(f"Invalid warm-start teacher tensor: {key}")
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
        if load_cfg is None or load_cfg.get("iteration", True):
            source = torch.load(path, map_location="cpu", weights_only=False)
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
