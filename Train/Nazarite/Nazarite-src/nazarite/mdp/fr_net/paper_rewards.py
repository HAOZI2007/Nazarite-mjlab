"""FR-Net Table I reward kernels, without task-specific extra shaping.

Weights belong to the environment config. Vector squares are summed over
joints; ``||.||_2`` in the paper denotes an unsquared Euclidean norm. Contact
indicators use actual simulator contact matches, without force thresholds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_ROBOT = SceneEntityCfg("robot")


def base_orientation(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Table I: gx² + gy²."""
    return env.scene["robot"].data.projected_gravity_b[:, :2].square().sum(-1)


def upright_orientation(env: ManagerBasedRlEnv, epsilon: float = 0.15) -> torch.Tensor:
    """Equation (8): exp(-(gz + 1)² / (2 epsilon²))."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    gravity_z = env.scene["robot"].data.projected_gravity_b[:, 2]
    return torch.exp(-(gravity_z + 1).square() / (2 * epsilon**2))


def target_posture(
    env: ManagerBasedRlEnv,
    epsilon: float = 0.15,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """Equation (8): exp(-sum((q-qstand)²)) gated by |gz+1| < epsilon."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    data = env.scene[asset_cfg.name].data
    error = (
        data.joint_pos[:, asset_cfg.joint_ids]
        - data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    upright = (data.projected_gravity_b[:, 2] + 1).abs() < epsilon
    return torch.exp(-error.square().sum(-1)) * upright.float()


def _contact_matches(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    sensor = env.scene[sensor_name]
    if not isinstance(sensor, ContactSensor):
        raise TypeError(f"Expected ContactSensor for '{sensor_name}'")
    found = sensor.data.found
    if found is None:
        raise RuntimeError(f"Contact sensor '{sensor_name}' needs the found field")
    # Multiple slots for one foot still count as one contacting foot.
    return found.reshape(env.num_envs, -1, sensor.cfg.num_slots).gt(0).any(-1)


def feet_contact(
    env: ManagerBasedRlEnv, sensor_name: str = "feet_ground_contact"
) -> torch.Tensor:
    """Equation (9): count feet in contact, in [0, 4]."""
    return _contact_matches(env, sensor_name).float().sum(-1)


def body_contact(
    env: ManagerBasedRlEnv,
    sensor_names: tuple[str, ...] = (
        "trunk_ground_touch",
        "hip_ground_touch",
        "thigh_ground_touch",
        "shank_ground_touch",
    ),
) -> torch.Tensor:
    """Equation (9): a single indicator for any non-foot terrain contact."""
    contacts = [_contact_matches(env, name).any(-1) for name in sensor_names]
    return torch.stack(contacts, dim=-1).any(-1).float()


def safety_force(
    env: ManagerBasedRlEnv, sensor_name: str = "shank_ground_touch"
) -> torch.Tensor:
    """Equation (10): sum of four knee-body world-XY force L2 norms, in N.

    XZ1's shank bodies start at the knee. Their ground-contact sensor must use
    netforce reduction so XY refers to world axes even when the robot rolls.
    """
    sensor = env.scene[sensor_name]
    if not isinstance(sensor, ContactSensor):
        raise TypeError(f"Expected ContactSensor for '{sensor_name}'")
    if sensor.cfg.reduce != "netforce":
        raise ValueError("FR-Net safety_force requires global netforce contacts")
    force = sensor.data.force
    if force is None:
        raise RuntimeError(f"Contact sensor '{sensor_name}' needs the force field")
    return torch.linalg.vector_norm(force[..., :2], dim=-1).sum(-1)


def body_bias(
    env: ManagerBasedRlEnv, reset_position_key: str = "frnet_reset_xy"
) -> torch.Tensor:
    """Equation (10): clip(||base_xy - reset_base_xy||_2, 0, 4), in m."""
    position = env.scene["robot"].data.root_link_pos_w[:, :2]
    initial_position = env.extras[reset_position_key]
    return torch.linalg.vector_norm(position - initial_position, dim=-1).clamp(0, 4)


def position_limits(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
    """Table I: number of joints outside the actual joint position limits."""
    data = env.scene[asset_cfg.name].data
    position = data.joint_pos[:, asset_cfg.joint_ids]
    limits = data.joint_pos_limits[:, asset_cfg.joint_ids]
    return ((position < limits[..., 0]) | (position > limits[..., 1])).float().sum(-1)


def angular_velocity_limit(
    env: ManagerBasedRlEnv,
    threshold: float = 0.8,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """Table I's qdot limit: sum(max(abs(joint_velocity) - 0.8, 0)).

    This is joint angular velocity in rad/s, not the base IMU angular velocity
    and not a fraction of an actuator velocity limit.
    """
    if threshold < 0:
        raise ValueError("threshold must be nonnegative")
    velocity = env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]
    return (velocity.abs() - threshold).clamp(min=0).sum(-1)
