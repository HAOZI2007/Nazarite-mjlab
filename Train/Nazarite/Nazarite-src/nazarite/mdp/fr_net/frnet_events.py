"""Reset events specific to quadrupedal fall recovery."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, sample_uniform

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def reset_fallen_root_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    roll_range: tuple[float, float] = (-3.14159, 3.14159),
    fallen_pitch_range: tuple[float, float] = (0.8, 2.4),
    pitch_sign: float | None = None,
    yaw_range: tuple[float, float] = (-3.14159, 3.14159),
    xy_offset_range: tuple[float, float] = (0.0, 0.0),
    height_offset_range: tuple[float, float] = (0.18, 0.28),
    minimum_root_height_above_origin: float = 0.50,
    linear_velocity_range: tuple[float, float] = (-0.25, 0.25),
    angular_velocity_range: tuple[float, float] = (-1.0, 1.0),
    standing_probability: float = 0.0,
    semi_fallen_probability: float = 0.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Reset each robot into a collision-safe side, back, or belly fall.

    The default root height describes an upright robot.  Reusing it with a
    random fallen orientation puts the trunk or legs inside the floor, which
    creates unrecoverable MuJoCo contacts.  Start above the flat spawn platform
    instead, then let gravity settle the robot into a real fallen state.
    """
    if minimum_root_height_above_origin <= 0.0:
        raise ValueError("minimum_root_height_above_origin must be positive")
    if not 0.0 <= standing_probability <= 1.0:
        raise ValueError("standing_probability must be in [0, 1]")
    if not 0.0 <= semi_fallen_probability <= 1.0:
        raise ValueError("semi_fallen_probability must be in [0, 1]")
    if standing_probability + semi_fallen_probability > 1.0:
        raise ValueError("standing and semi-fallen probabilities must sum to <= 1")
    env_ids = resolve_env_ids(env, env_ids)
    asset = env.scene[asset_cfg.name]
    root_state = asset.data.default_root_state[env_ids].clone()
    num_resets = len(env_ids)

    mode_sample = torch.rand(num_resets, device=env.device)
    standing = mode_sample < standing_probability
    semi_fallen = (mode_sample >= standing_probability) & (
      mode_sample < standing_probability + semi_fallen_probability
    )
    if "frnet_recovery_mask" not in env.extras:
      env.extras["frnet_recovery_mask"] = torch.ones(
        env.num_envs, dtype=torch.bool, device=env.device
      )
    env.extras["frnet_recovery_mask"][env_ids] = ~standing
    roll = sample_uniform(*roll_range, (num_resets,), device=env.device)
    roll = torch.where(standing, torch.zeros_like(roll), roll)
    roll = torch.where(
        semi_fallen,
        sample_uniform(-1.0, 1.0, (num_resets,), device=env.device),
        roll,
    )
    pitch_magnitude = sample_uniform(
        *fallen_pitch_range, (num_resets,), device=env.device
    )
    pitch_magnitude = torch.where(
        semi_fallen,
        sample_uniform(0.35, 0.80, (num_resets,), device=env.device),
        pitch_magnitude,
    )
    pitch_magnitude = torch.where(
        standing, torch.zeros_like(pitch_magnitude), pitch_magnitude
    )
    if pitch_sign is None:
        pitch_direction = torch.where(
            torch.rand(num_resets, device=env.device) < 0.5,
            -torch.ones(num_resets, device=env.device),
            torch.ones(num_resets, device=env.device),
        )
    elif pitch_sign in (-1.0, 1.0):
        pitch_direction = torch.full((num_resets,), pitch_sign, device=env.device)
    else:
        raise ValueError("pitch_sign must be None, -1.0, or 1.0")
    pitch = pitch_direction * pitch_magnitude
    yaw = sample_uniform(*yaw_range, (num_resets,), device=env.device)
    orientation = quat_from_euler_xyz(roll, pitch, yaw)

    position = root_state[:, :3]
    position += env.scene.env_origins[env_ids]
    # A terrain recovery episode must not always begin on the central flat
    # platform.  Keep the offset well inside the 8 m terrain tile so the robot
    # can fall onto low stairs, grid blocks, or scattered boxes without reaching
    # the tile border.
    position[:, :2] += sample_uniform(
        *xy_offset_range, (num_resets, 2), device=env.device
    )
    fallen_height_offset = sample_uniform(
        *height_offset_range, (num_resets,), device=env.device
    )
    position[:, 2] += torch.where(
        standing, torch.zeros_like(fallen_height_offset), fallen_height_offset
    )
    # Every recovery terrain has a flat center platform at its environment
    # origin.  Sensor readings still describe the previous state during reset,
    # so this origin-relative guard is the reliable way to prevent an initial
    # body/terrain overlap on both planes and generated terrain.
    minimum_height = env.scene.env_origins[env_ids, 2] + torch.where(
        standing,
        torch.zeros_like(fallen_height_offset),
        torch.full_like(fallen_height_offset, minimum_root_height_above_origin),
    )
    position[:, 2] = torch.maximum(position[:, 2], minimum_height)
    velocity = torch.cat(
        (
            sample_uniform(*linear_velocity_range, (num_resets, 3), device=env.device),
            sample_uniform(*angular_velocity_range, (num_resets, 3), device=env.device),
        ),
        dim=-1,
    )

    asset.write_root_link_pose_to_sim(
        torch.cat((position, orientation), dim=-1),
        env_ids=env_ids,
    )
    asset.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)
