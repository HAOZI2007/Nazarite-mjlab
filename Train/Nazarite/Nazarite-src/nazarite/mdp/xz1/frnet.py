"""XZ1 flat recovery: geometry-safe resets and nominal-pose action targets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import mujoco
import numpy as np
import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul
from nazarite.mdp.fr_net.frnet_terminations import stable_recovery_mask
from nazarite.mdp.xz1.frnet_curriculum import RECOVERY_STAGES

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.metrics_manager import MetricsTermCfg


FALL_CASES = ("left_side", "right_side", "back", "belly", "semi_fallen", "standing")
"""Pose-bank axis order. Back means supine; belly means upright but folded."""


def collision_bottoms(
    model: mujoco.MjModel, data: mujoco.MjData, prefix: str
) -> np.ndarray:
    """Lowest world Z of each collidable robot primitive, excluding visual meshes."""
    bottoms = []
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] == 0:
            continue
        body_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom_id]
        )
        if not body_name or not body_name.startswith(prefix):
            continue
        if not (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]):
            continue
        kind = model.geom_type[geom_id]
        size = model.geom_size[geom_id]
        z_axis = data.geom_xmat[geom_id].reshape(3, 3)[2]
        if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            extent = size[0]
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            extent = np.abs(z_axis) @ size
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            extent = size[0] + abs(z_axis[2]) * size[1]
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            extent = np.linalg.norm(z_axis[:2]) * size[0] + abs(z_axis[2]) * size[1]
        elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            extent = np.linalg.norm(z_axis * size)
        else:
            raise ValueError("XZ1 flat recovery requires primitive collision geometry")
        bottoms.append(data.geom_xpos[geom_id, 2] - extent)
    if not bottoms:
        raise ValueError("No collidable XZ1 primitives found")
    return np.asarray(bottoms)


def build_flat_reset_bank(
    model: mujoco.MjModel,
    joint_names: tuple[str, ...],
    default_joint_pos: np.ndarray,
    *,
    prefix: str = "robot/",
    seed: int = 42,
    samples_per_case: int = 64,
    clearance: float = 0.008,
    side_roll_range: tuple[float, float] | None = None,
    back_roll_range: tuple[float, float] | None = None,
    joint_noise: float = 0.12,
    folded_probability: float = 0.0,
) -> np.ndarray:
    """Precompute near-ground poses from actual XZ1 collision geometry once.

    Layout is [6 families, samples, root_z + quaternion(wxyz) + 12 joints].
    Runtime sampling adds arbitrary yaw and XY; neither changes ground clearance.
    """
    if samples_per_case < 1 or clearance <= 0:
        raise ValueError("samples_per_case and clearance must be positive")
    if joint_noise < 0 or not 0 <= folded_probability <= 1:
        raise ValueError("Invalid joint diversity settings")
    rng = np.random.default_rng(seed)
    data = mujoco.MjData(model)
    root = model.joint(prefix + "floating_base_joint").qposadr[0]
    joint_ids = np.array([model.joint(prefix + name).id for name in joint_names])
    addresses = model.jnt_qposadr[joint_ids]
    limits = model.jnt_range[joint_ids]
    bank = np.zeros((len(FALL_CASES), samples_per_case, 5 + len(joint_names)))
    rolls = (math.pi / 2, -math.pi / 2, math.pi, 0.0, 0.65, 0.0)
    for case_id, case in enumerate(FALL_CASES):
        for sample in range(samples_per_case):
            joints = default_joint_pos.copy()
            roll = rolls[case_id]
            pitch = 0.0
            if case != "standing":
                roll += rng.uniform(-0.12, 0.12)
                pitch = rng.uniform(-0.12, 0.12)
                joints += rng.uniform(-joint_noise, joint_noise, len(joint_names))
            if side_roll_range is not None and case in ("left_side", "right_side"):
                roll = rng.uniform(*side_roll_range) * (
                    1 if case == "left_side" else -1
                )
            if back_roll_range is not None and case == "back":
                roll = rng.uniform(*back_roll_range) * rng.choice((-1, 1))
            if case == "semi_fallen":
                roll *= rng.choice((-1.0, 1.0))
                pitch = rng.uniform(-0.5, 0.5)
            folded = (
                folded_probability > 0
                and case in ("left_side", "right_side", "back")
                and (rng.random() < folded_probability)
            )
            if case == "belly" or folded:
                # Fold the knees and splay the legs so this is self-standing
                # practice, rather than another already-standing reset.
                for j, name in enumerate(joint_names):
                    if "thigh" in name:
                        joints[j] = rng.uniform(1.60, 1.90)
                    elif "shank" in name:
                        joints[j] = rng.uniform(-2.38, -2.28)
                    else:
                        joints[j] = 0.6 if name.startswith(("FL", "RL")) else -0.6
            if folded:
                joints += rng.uniform(-joint_noise, joint_noise, len(joint_names))
            joints = np.clip(joints, limits[:, 0] + 0.02, limits[:, 1] - 0.02)
            cr, sr = math.cos(roll / 2), math.sin(roll / 2)
            cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
            quat = np.array([cr * cp, sr * cp, cr * sp, -sr * sp])
            data.qpos[:] = model.qpos0
            data.qpos[root : root + 3] = 0.0
            data.qpos[root + 3 : root + 7] = quat
            data.qpos[addresses] = joints
            mujoco.mj_kinematics(model, data)
            height = -collision_bottoms(model, data, prefix).min() + clearance
            bank[case_id, sample] = np.concatenate(([height], quat, joints))
    return bank


class FlatRecoveryReset(ManagerTermBase):
    """Sample a cached pose bank on the simulation device, including partial resets."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        robot = env.scene["robot"]
        model = env.sim.mj_model
        names = tuple(robot.joint_names)
        defaults = robot.data.default_joint_pos[0].detach().cpu().numpy()
        seed = env.cfg.seed if env.cfg.seed is not None else 42
        bank = build_flat_reset_bank(model, names, defaults, seed=seed)
        self.bank = torch.as_tensor(bank, dtype=torch.float32, device=env.device)
        self.curriculum_banks: torch.Tensor | None = None
        if cfg.params.get("use_curriculum", False):
            banks = [
                build_flat_reset_bank(
                    model,
                    names,
                    defaults,
                    seed=seed + level + 1,
                    side_roll_range=(
                        math.radians(stage.side_roll[0]),
                        math.radians(stage.side_roll[1]),
                    ),
                    back_roll_range=(
                        math.radians(stage.back_roll[0]),
                        math.radians(stage.back_roll[1]),
                    ),
                    joint_noise=stage.joint_noise,
                    folded_probability=0.35,
                )
                for level, stage in enumerate(RECOVERY_STAGES)
            ]
            self.curriculum_banks = torch.as_tensor(
                np.stack(banks), dtype=torch.float32, device=env.device
            )
        self._probabilities = torch.tensor(
            [stage.probabilities for stage in RECOVERY_STAGES], device=env.device
        )
        env.extras["frnet_reset_level"] = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
        env.extras["frnet_reset_kind"] = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
        env.extras["frnet_recovery_mask"] = torch.ones(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        env.extras["frnet_reset_xy"] = torch.zeros(env.num_envs, 2, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        standing_probability: float = 0.05,
        semi_fallen_probability: float = 0.15,
        fall_case: str = "mixed",
        use_curriculum: bool = False,
        rehearsal_probability: float = 0.20,
    ) -> None:
        if not (
            0 <= standing_probability <= 1
            and 0 <= semi_fallen_probability <= 1
            and standing_probability + semi_fallen_probability <= 1
        ):
            raise ValueError("Reset probabilities must be nonnegative and sum to <= 1")
        if not 0 <= rehearsal_probability <= 1:
            raise ValueError("rehearsal_probability must be in [0, 1]")
        env_ids = resolve_env_ids(env, env_ids)
        count = len(env_ids)
        if count == 0:
            return
        levels = torch.full((count,), -1, dtype=torch.long, device=env.device)
        kind = torch.randint(4, (count,), device=env.device)
        if fall_case == "mixed" and use_curriculum:
            if self.curriculum_banks is None:
                raise RuntimeError("Curriculum pose banks were not configured")
            current_level = int(env.extras.get("frnet_curriculum_level", 0))
            levels.fill_(current_level)
            if current_level > 0:
                rehearse = torch.rand(count, device=env.device) < rehearsal_probability
                previous = torch.randint(current_level, (count,), device=env.device)
                levels = torch.where(rehearse, previous, levels)
            kind = torch.multinomial(self._probabilities[levels], 1).squeeze(-1)
        elif fall_case == "mixed":
            sample = torch.rand(count, device=env.device)
            kind[sample < standing_probability] = 5
            kind[
                (sample >= standing_probability)
                & (sample < standing_probability + semi_fallen_probability)
            ] = 4
        elif fall_case in FALL_CASES:
            kind[:] = FALL_CASES.index(fall_case)
        else:
            raise ValueError(f"Unknown fall case: {fall_case}")
        indices = torch.randint(self.bank.shape[1], (count,), device=env.device)
        if use_curriculum and fall_case == "mixed":
            assert self.curriculum_banks is not None
            poses = self.curriculum_banks[levels, kind, indices]
        else:
            poses = self.bank[kind, indices]
        yaw = (torch.rand(count, device=env.device) * 2 - 1) * math.pi
        yaw_quat = quat_from_euler_xyz(
            torch.zeros_like(yaw), torch.zeros_like(yaw), yaw
        )
        position = env.scene.env_origins[env_ids].clone()
        position[:, :2] += (torch.rand(count, 2, device=env.device) - 0.5) * 0.5
        position[:, 2] += poses[:, 0]
        quat = quat_mul(yaw_quat, poses[:, 1:5])
        robot = env.scene["robot"]
        robot.write_root_link_pose_to_sim(
            torch.cat((position, quat), dim=-1), env_ids=env_ids
        )
        robot.write_root_link_velocity_to_sim(
            torch.zeros(count, 6, device=env.device), env_ids=env_ids
        )
        robot.write_joint_state_to_sim(
            poses[:, 5:], torch.zeros(count, 12, device=env.device), env_ids=env_ids
        )
        env.extras["frnet_reset_kind"][env_ids] = kind
        env.extras["frnet_reset_level"][env_ids] = levels
        env.extras["frnet_recovery_mask"][env_ids] = kind != 5
        # Snapshot the sampled root position, not the terrain/env origin. Only
        # reset envs change their reference, so partial resets preserve drift.
        env.extras["frnet_reset_xy"][env_ids] = position[:, :2]


@dataclass(kw_only=True)
class RecoveryJointPositionActionCfg(JointPositionActionCfg):
    """Default-pose offsets, with control-step target slew and hard joint limits."""

    max_target_delta: float = 0.15
    settle_steps: int = 6
    ramp_steps: int = 6

    def build(self, env: ManagerBasedRlEnv) -> RecoveryJointPositionAction:
        return RecoveryJointPositionAction(self, env)


class RecoveryJointPositionAction(JointPositionAction):
    """Latch a target once per control step; substeps keep the same PD target."""

    def __init__(self, cfg: RecoveryJointPositionActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        if cfg.max_target_delta <= 0 or min(cfg.settle_steps, cfg.ramp_steps) < 0:
            raise ValueError("Invalid recovery action slew/settle/ramp settings")
        self._target = self._entity.data.default_joint_pos[:, self._target_ids].clone()
        self._reset_pos = self._target.clone()
        self._age = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._target[ids] = self._entity.data.joint_pos[ids][:, self._target_ids]
        self._reset_pos[ids] = self._target[ids]
        self._age[ids] = 0

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        cfg = cast(RecoveryJointPositionActionCfg, self.cfg)
        bias = self._entity.data.encoder_bias[:, self._target_ids]
        desired = self._processed_actions - bias
        recovering = self._env.extras["frnet_recovery_mask"]
        alpha = (
            (self._age - cfg.settle_steps + 1).float() / max(cfg.ramp_steps, 1)
        ).clamp(0, 1)
        alpha = torch.where(
            self._age < cfg.settle_steps, torch.zeros_like(alpha), alpha
        )
        alpha = torch.where(recovering, alpha, torch.ones_like(alpha))
        desired = torch.lerp(self._reset_pos, desired, alpha.unsqueeze(-1))
        self._target += (desired - self._target).clamp(
            -cfg.max_target_delta, cfg.max_target_delta
        )
        limits = self._entity.data.soft_joint_pos_limits[:, self._target_ids]
        self._target.clamp_(min=limits[..., 0], max=limits[..., 1])
        self._age += 1

    def apply_actions(self) -> None:
        self._entity.set_joint_position_target(self._target, joint_ids=self._target_ids)


def reset_fraction(env: ManagerBasedRlEnv, case: str) -> torch.Tensor:
    """Log sampling proportions separately from recovery success."""
    return (env.extras["frnet_reset_kind"] == FALL_CASES.index(case)).float()


class StableRecoveryMetric(ManagerTermBase):
    """Observe sustained recovery without ending an episode or adding reward."""

    def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        duration = float(cfg.params.get("hold_duration_s", 0.8))
        if duration <= 0:
            raise ValueError("hold_duration_s must be positive")
        self._required_steps = max(1, math.ceil(duration / env.step_dt))
        self._stable_steps = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
        self._track_progress = bool(cfg.params.get("track_progress", False))
        if self._track_progress:
            for key in ("frnet_episode_success", "frnet_final_stable"):
                env.extras[key] = torch.zeros(
                    env.num_envs, dtype=torch.bool, device=env.device
                )
            env.extras["frnet_episode_steps"] = torch.zeros_like(self._stable_steps)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self._stable_steps[slice(None) if env_ids is None else env_ids] = 0
        if self._track_progress:
            ids = slice(None) if env_ids is None else env_ids
            for key in (
                "frnet_episode_success",
                "frnet_final_stable",
                "frnet_episode_steps",
            ):
                self._env.extras[key][ids] = 0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        hold_duration_s: float = 0.8,
        fallen_only: bool = False,
        track_progress: bool = False,
        **condition_params,
    ) -> torch.Tensor:
        del hold_duration_s, track_progress
        stable = stable_recovery_mask(env, **condition_params)
        self._stable_steps = torch.where(stable, self._stable_steps + 1, 0)
        result = self._stable_steps.ge(self._required_steps)
        if self._track_progress:
            env.extras["frnet_episode_success"] |= result
            env.extras["frnet_final_stable"].copy_(result)
            env.extras["frnet_episode_steps"] += 1
        if fallen_only:
            result &= env.extras["frnet_recovery_mask"]
        return result.float()
