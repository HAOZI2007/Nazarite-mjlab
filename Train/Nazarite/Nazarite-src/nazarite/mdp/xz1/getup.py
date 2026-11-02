# SPDX-License-Identifier: Apache-2.0
# Adapted from mjlab_playground/getup (Copyright 2025, The mjlab Developers).
# XZ1 adaptations: primitive-safe reset banks, independent settle clock,
# terrain-relative height, stricter stability diagnostics.
"""Playground getup rewards, random drops and relative-position control for XZ1."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import mujoco
import numpy as np
import torch

from mjlab.envs.mdp.actions.actions import (
    RelativeJointPositionAction,
    RelativeJointPositionActionCfg,
)
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.sensor import ContactSensor
from mjlab.tasks.velocity.mdp.rewards import self_collision_cost
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul
from nazarite.mdp.xz1.frnet import FALL_CASES, build_flat_reset_bank, collision_bottoms

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.metrics_manager import MetricsTermCfg
    from mjlab.managers.reward_manager import RewardTermCfg


@dataclass(kw_only=True)
class SettleRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    settle_steps: int = 25

    def build(self, env: ManagerBasedRlEnv) -> SettleRelativeJointPositionAction:
        return SettleRelativeJointPositionAction(self, env)


class SettleRelativeJointPositionAction(RelativeJointPositionAction):
    """Recompute q + scale * action every physics substep, as in Playground.

    Use an independent control-step clock: the runner randomizes episode length
    initially, which must not skip the drop/settle window. No nominal-pose offset,
    ramp, target slew, target clamp or target accumulator is added here.
    """

    def __init__(
        self, cfg: SettleRelativeJointPositionActionCfg, env: ManagerBasedRlEnv
    ):
        super().__init__(cfg, env)
        if cfg.settle_steps < 0:
            raise ValueError("settle_steps must be nonnegative")
        self._age = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._settling = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        self._age[slice(None) if env_ids is None else env_ids] = 0

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        cfg = cast(SettleRelativeJointPositionActionCfg, self.cfg)
        self._settling = (self._age < cfg.settle_steps) & self._env.extras[
            "settle_mask"
        ]
        self._age += 1
        self._env.extras["getup_action_age"] = self._age

    def apply_actions(self) -> None:
        current = self._entity.data.joint_pos[:, self._target_ids]
        bias = self._entity.data.encoder_bias[:, self._target_ids]
        delta = torch.where(self._settling[:, None], 0, self._processed_actions)
        self._entity.set_joint_position_target(
            current + delta - bias, joint_ids=self._target_ids
        )


def build_drop_bank(
    model: mujoco.MjModel,
    joint_names: tuple[str, ...],
    *,
    seed: int,
    samples: int = 2048,
    prefix: str = "robot/",
) -> np.ndarray:
    """Uniform random SO(3) and soft-range joints; cache exact ground clearance.

    Store [minimum_safe_root_height, quaternion(wxyz), joints]. The requested
    drop height is applied at runtime. This finite-bank adaptation avoids
    spawning long XZ1 limbs through the plane at a fixed 0.5 m root height.
    """
    if samples < 1:
        raise ValueError("samples must be positive")
    rng = np.random.default_rng(seed)
    data = mujoco.MjData(model)
    root = model.joint(prefix + "floating_base_joint").qposadr[0]
    ids = np.array([model.joint(prefix + name).id for name in joint_names])
    addresses = model.jnt_qposadr[ids]
    limits = model.jnt_range[ids]
    middle = limits.mean(-1)
    half_width = (limits[:, 1] - limits[:, 0]) * 0.95 / 2
    bank = np.zeros((samples, 5 + len(joint_names)))
    for i in range(samples):
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        joints = rng.uniform(middle - half_width, middle + half_width)
        data.qpos[:] = model.qpos0
        data.qpos[root : root + 3] = 0
        data.qpos[root + 3 : root + 7] = quat
        data.qpos[addresses] = joints
        mujoco.mj_kinematics(model, data)
        height = -collision_bottoms(model, data, prefix).min() + 0.008
        bank[i] = np.concatenate(([height], quat, joints))
    return bank


class RandomDropReset(ManagerTermBase):
    """60% random drops / 40% stance in train; full random drops in play.

    Explicit fall_case overrides are near-ground standardized evaluation starts,
    not the default training distribution. Existing four-case evaluation remains
    comparable; random-drop generalization is measured separately.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        robot = env.scene["robot"]
        seed = env.cfg.seed if env.cfg.seed is not None else 42
        names = tuple(robot.joint_names)
        self.drop_bank = torch.as_tensor(
            build_drop_bank(env.sim.mj_model, names, seed=seed),
            dtype=torch.float32,
            device=env.device,
        )
        self.pose_bank = torch.as_tensor(
            build_flat_reset_bank(
                env.sim.mj_model,
                names,
                robot.data.default_joint_pos[0].detach().cpu().numpy(),
                seed=seed,
            ),
            dtype=torch.float32,
            device=env.device,
        )
        for key in ("settle_mask", "frnet_recovery_mask"):
            env.extras[key] = torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            )
        env.extras["frnet_reset_kind"] = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        fall_probability: float = 0.6,
        fall_height: float = 0.5,
        velocity_range: float = 0.5,
        fall_case: str = "mixed",
    ) -> None:
        if not 0 <= fall_probability <= 1 or fall_height <= 0 or velocity_range < 0:
            raise ValueError("Invalid drop reset configuration")
        ids = resolve_env_ids(env, env_ids)
        n = len(ids)
        if not n:
            return
        robot = env.scene["robot"]
        fallen = torch.rand(n, device=env.device) < fall_probability
        kinds = torch.full((n,), -1, dtype=torch.long, device=env.device)
        if fall_case == "mixed":
            indices = torch.randint(len(self.drop_bank), (n,), device=env.device)
            poses = self.drop_bank[indices].clone()
            poses[:, 0].clamp_(min=fall_height)
        elif fall_case in FALL_CASES:
            kinds.fill_(FALL_CASES.index(fall_case))
            fallen.fill_(fall_case != "standing")
            indices = torch.randint(self.pose_bank.shape[1], (n,), device=env.device)
            poses = self.pose_bank[kinds, indices]
            # Keep the four near-ground evaluation groups comparable with the
            # legacy reset: only the random-drop group has initial velocities.
            velocity_range = 0.0
        else:
            raise ValueError(f"Unknown fall_case: {fall_case}")
        default = robot.data.default_root_state[ids]
        position = env.scene.env_origins[ids].clone()
        position[:, 2] += poses[:, 0]
        yaw = (torch.rand(n, device=env.device) * 2 - 1) * math.pi
        zeros = torch.zeros_like(yaw)
        quat = quat_mul(quat_from_euler_xyz(zeros, zeros, yaw), poses[:, 1:5])
        # The standing bank is also geometry-safe for the actual XZ1 default pose.
        standing = self.pose_bank[FALL_CASES.index("standing"), 0]
        standing_position = env.scene.env_origins[ids].clone()
        standing_position[:, 2] += max(float(default[0, 2]), float(standing[0])) + 0.02
        position = torch.where(fallen[:, None], position, standing_position)
        quat = torch.where(fallen[:, None], quat, default[:, 3:7])
        velocities = (torch.rand(n, 6, device=env.device) * 2 - 1) * velocity_range
        velocities = torch.where(fallen[:, None], velocities, default[:, 7:13])
        joint_pos = torch.where(
            fallen[:, None], poses[:, 5:], robot.data.default_joint_pos[ids]
        )
        joint_vel = (torch.rand_like(joint_pos) * 2 - 1) * velocity_range
        joint_vel = torch.where(
            fallen[:, None], joint_vel, robot.data.default_joint_vel[ids]
        )
        robot.write_root_link_pose_to_sim(torch.cat((position, quat), -1), env_ids=ids)
        robot.write_root_link_velocity_to_sim(velocities, env_ids=ids)
        robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=ids)
        env.extras["settle_mask"][ids] = fallen
        env.extras["frnet_recovery_mask"][ids] = fallen
        env.extras["frnet_reset_kind"][ids] = torch.where(fallen, kinds, 5)


def orientation_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    gravity = env.scene["robot"].data.projected_gravity_b
    error = gravity.clone()
    error[:, 2] += 1
    return torch.exp(-2 * error.square().sum(-1))


def height_reward(env: ManagerBasedRlEnv, desired_height: float) -> torch.Tensor:
    height = env.scene["base_height_scan"].data.heights[:, 0]
    return torch.expm1(height.clamp(max=desired_height)) / math.expm1(desired_height)


def _upright(env: ManagerBasedRlEnv, threshold: float) -> torch.Tensor:
    gravity = env.scene["robot"].data.projected_gravity_b
    error = gravity.clone()
    error[:, 2] += 1
    return error.square().sum(-1) < threshold


def _feet_contact(env: ManagerBasedRlEnv, force_threshold: float) -> torch.Tensor:
    sensor = env.scene["feet_ground_contact"]
    if not isinstance(sensor, ContactSensor):
        raise TypeError("feet_ground_contact must be a ContactSensor")
    force = sensor.data.force
    if force is None:
        raise ValueError("feet_ground_contact must expose contact forces")
    return force.norm(dim=-1) >= force_threshold


def _control_active(env: ManagerBasedRlEnv, settle_steps: int) -> torch.Tensor:
    age = env.extras.get("getup_action_age")
    if age is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return (age > 0) & (~env.extras["settle_mask"] | (age > settle_steps))


def _standing_for_support(
    env: ManagerBasedRlEnv,
    min_height: float,
    upright_gravity_z_max: float,
    max_angular_velocity: float,
) -> torch.Tensor:
    data = env.scene["robot"].data
    return (
        (data.projected_gravity_b[:, 2] <= upright_gravity_z_max)
        & (env.scene["base_height_scan"].data.heights[:, 0] >= min_height)
        & (data.root_link_ang_vel_b.norm(dim=-1) <= max_angular_velocity)
    )


def feet_support_reward(
    env: ManagerBasedRlEnv,
    force_threshold: float = 5.0,
    min_height: float = 0.25,
    upright_gravity_z_max: float = -0.90,
    max_angular_velocity: float = 1.5,
    settle_steps: int = 25,
) -> torch.Tensor:
    """Favor concurrent support after getup, with full reward for all four feet.

    Squaring the contact fraction gives partial credit while making additional
    supporting feet more valuable. Height/upright gates exclude belly crawling.
    """
    fraction = _feet_contact(env, force_threshold).float().mean(-1)
    valid = _standing_for_support(
        env, min_height, upright_gravity_z_max, max_angular_velocity
    ) & _control_active(env, settle_steps)
    return fraction.square() * valid.float()


class SynchronizedLandingReward(ManagerTermBase):
    """One bonus per recovery episode for four recent, concurrent touchdowns.

    Detect touchdowns from force samples at control frequency. Each foot must
    first be airborne for min_air_time after settle; all four must then be on
    the ground with their latest touchdown times within landing_window_s.
    Return an impulse divided by dt so the configured weight is the actual
    one-off bonus under the task's default dt-scaled reward manager.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        contacts = _feet_contact(env, cfg.params.get("force_threshold", 5.0))
        if contacts.shape != (env.num_envs, 4):
            raise ValueError("Synchronized landing requires exactly four foot sensors")
        self._air_steps = torch.zeros_like(contacts, dtype=torch.long)
        self._last_touch = torch.full_like(self._air_steps, -1_000_000)
        self._clock = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._paid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._air_steps[ids] = 0
        self._last_touch[ids] = -1_000_000
        self._clock[ids] = 0
        self._paid[ids] = False

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        landing_window_s: float = 0.12,
        min_air_time: float = 0.04,
        force_threshold: float = 5.0,
        min_height: float = 0.25,
        upright_gravity_z_max: float = -0.90,
        max_angular_velocity: float = 1.5,
        settle_steps: int = 25,
    ) -> torch.Tensor:
        if landing_window_s <= 0 or min_air_time <= 0:
            raise ValueError("Landing window and minimum air time must be positive")
        contacts = _feet_contact(env, force_threshold)
        active = _control_active(env, settle_steps) & env.extras["frnet_recovery_mask"]
        self._clock += 1
        touchdown = (
            contacts
            & (self._air_steps * env.step_dt >= min_air_time - 1e-6)
            & active[:, None]
        )
        self._last_touch = torch.where(
            touchdown, self._clock[:, None], self._last_touch
        )
        self._air_steps = torch.where(
            ~contacts & active[:, None], self._air_steps + 1, 0
        )
        recent = (
            (self._clock[:, None] - self._last_touch) * env.step_dt
            <= landing_window_s + 1e-6
        ).all(-1)
        valid = _standing_for_support(
            env, min_height, upright_gravity_z_max, max_angular_velocity
        )
        bonus = active & valid & contacts.all(-1) & recent & ~self._paid
        self._paid |= bonus
        return bonus.float() / env.step_dt


class GatedPostureReward(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        names = env.scene["robot"].joint_names
        self.std = torch.tensor(
            [cfg.params["std"][name.rsplit("_", 2)[-2]] for name in names],
            device=env.device,
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        std: dict[str, float],
        orientation_threshold: float = 0.01,
    ) -> torch.Tensor:
        del std
        data = env.scene["robot"].data
        error = ((data.joint_pos - data.default_joint_pos) / self.std).square().mean(-1)
        return _upright(env, orientation_threshold).float() * torch.exp(-error)


class GetupSuccess(ManagerTermBase):
    """Playground's ever-upright-and-high metric, separate from strict stability."""

    def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        self.success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self.success[slice(None) if env_ids is None else env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        desired_height: float,
        height_tolerance: float = 0.02,
        orientation_threshold: float = 0.05,
    ) -> torch.Tensor:
        high = env.scene["base_height_scan"].data.heights[:, 0] > (
            desired_height - height_tolerance
        )
        self.success |= _upright(env, orientation_threshold) & high
        return self.success.float()


def mechanical_power(env: ManagerBasedRlEnv) -> torch.Tensor:
    data = env.scene["robot"].data
    return (data.actuator_force * data.joint_vel).abs().sum(-1)


def energy_termination(
    env: ManagerBasedRlEnv, threshold: float = float("inf"), settle_steps: int = 25
) -> torch.Tensor:
    age = env.extras.get("getup_action_age")
    if age is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return (age > settle_steps) & (mechanical_power(env) > threshold)


def joint_speed_rms(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.scene["robot"].data.joint_vel.square().mean(-1).sqrt()


def joint_torque_rms(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.scene["robot"].data.actuator_force.square().mean(-1).sqrt()


def action_delta_rms(env: ManagerBasedRlEnv) -> torch.Tensor:
    manager = env.action_manager
    return (manager.action - manager.prev_action).square().mean(-1).sqrt()


def base_angular_speed(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.scene["robot"].data.root_link_ang_vel_b.norm(dim=-1)


def self_collision_penalty(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Preserve Playground's four-substep cost scale with XZ1's ten substeps."""
    return self_collision_cost(env, sensor_name) * (4 / env.cfg.decimation)
