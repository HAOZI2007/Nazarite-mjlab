"""Reference WMP/Legged-Gym termination rules adapted to mjlab."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def reference_time_out(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Match the source check's strict episode-length comparison."""
    return env.episode_length_buf > env.max_episode_length


def base_contact(
    env: ManagerBasedRlEnv,
    sensor_name: str = "nonfoot_ground_touch",
    force_threshold: float = 1.0,
    require_fall_state: bool = False,
    min_base_height: float = 0.18,
    inverted_gravity_z: float = 0.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate on hard base contact, optionally gated by a fallen state.

    The reference task terminates on any base contact.  A Go2's forward base
    collision geometry can lightly touch the 290 mm low frame while the robot
    is still upright, so WMP training can enable the fallen-state gate without
    changing the source force threshold.
    """
    sensor = env.scene[sensor_name]
    force = sensor.data.force_history
    if force is None:
        force = sensor.data.force
    if force is None:
        raise RuntimeError(f"Contact sensor '{sensor_name}' has no force data")

    force_norm = torch.linalg.vector_norm(force, dim=-1)
    # Flatten contact slots and the optional substep history dimensions.
    peak_force = force_norm.reshape(env.num_envs, -1).amax(dim=1)
    contact = peak_force > force_threshold
    if not require_fall_state:
        return contact
    asset = env.scene[asset_cfg.name]
    low_base = asset.data.root_link_pos_w[:, 2] < min_base_height
    inverted = asset.data.projected_gravity_b[:, 2] > inverted_gravity_z
    return contact & (low_base | inverted)


def velocity_violation(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    terrain_level_threshold: int = 3,
    velocity_error_threshold: float = 1.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate wrong-way motion on terrain rows above level three."""
    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    if terrain is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    command = env.command_manager.get_command(command_name)
    assert command is not None
    velocity_error = asset.data.root_link_lin_vel_b[:, 0] - command[:, 0]
    moving_against_command = (
        (velocity_error > velocity_error_threshold) & (command[:, 0] < 0.0)
    ) | ((velocity_error < -velocity_error_threshold) & (command[:, 0] > 0.0))
    return moving_against_command & (terrain.terrain_levels > terrain_level_threshold)


def fall(
    env: ManagerBasedRlEnv,
    downward_velocity_threshold: float = -3.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Match source fall checks: rapid downward motion or inverted gravity."""
    asset = env.scene[asset_cfg.name]
    downward = asset.data.root_link_lin_vel_w[:, 2] < downward_velocity_threshold
    inverted = asset.data.projected_gravity_b[:, 2] > 0.0
    return downward | inverted


def forward_course_bounds(
    env: ManagerBasedRlEnv,
    terrain_names: tuple[str, ...],
    lateral_margin: float = 0.60,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Prevent forward-course agents from escaping into an adjacent tile."""
    terrain = env.scene.terrain
    if terrain is None or terrain.cfg.terrain_generator is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    generator = terrain.cfg.terrain_generator
    names = list(generator.sub_terrains)
    selected = [names.index(name) for name in terrain_names if name in names]
    if not selected:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    type_ids = torch.tensor(
        selected, dtype=terrain.terrain_types.dtype, device=env.device
    )
    forward_course = torch.isin(terrain.terrain_types, type_ids)
    asset = env.scene[asset_cfg.name]
    lateral_offset = (
        asset.data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]
    ).abs()
    lateral_limit = float(generator.size[1]) / 2.0 - lateral_margin
    return forward_course & (lateral_offset > lateral_limit)


def stall_timeout(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    speed_fraction: float = 0.4,
    command_threshold: float = 0.1,
    time_limit: float = 2.0,
    min_terrain_level: int = 2,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate an episode that spends too long far below its command.

    Speeding up by reward alone is a soft pressure, and the 2026-10-18 policy
    answered it by surviving: it crawled at roughly a third of the command and
    collected the progress bonus anyway. This is the hard floor. ``time_limit``
    is deliberately long (2 s) so that the deliberate slow-down in front of a
    trench, and the flight phase of a jump, are not punished.
    """
    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    if terrain is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    command = env.command_manager.get_command(command_name)
    assert command is not None
    speed = torch.linalg.vector_norm(asset.data.root_link_lin_vel_b[:, :2], dim=1)
    target = torch.linalg.vector_norm(command[:, :2], dim=1)
    active = (target > command_threshold) & (
        terrain.terrain_levels >= int(min_terrain_level)
    )
    stalled = active & (speed < float(speed_fraction) * target)
    count = _stall_counter(env)
    # Replaced rather than updated in place: this term is also evaluated while an
    # inference policy drives the environment, and the counter is first allocated
    # inside the training loop's ``torch.inference_mode()`` block, where an in
    # place write afterwards raises "Inplace update to inference tensor".
    count = torch.where(stalled, count + 1, torch.zeros_like(count))
    # ``episode_length_buf`` restarts at zero, so clamping to it also clears the
    # counter at reset without needing a reset hook.
    count = torch.minimum(count, env.episode_length_buf)
    env.__dict__["_wmp_stall_counter"] = count
    limit = max(round(float(time_limit) / env.step_dt), 1)
    return count >= limit


def _stall_counter(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Return the lazily allocated per-environment stall counter."""
    counter = getattr(env, "_wmp_stall_counter", None)
    if counter is None or counter.shape[0] != env.num_envs:
        counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env.__dict__["_wmp_stall_counter"] = counter
    return counter


__all__ = [
    "base_contact",
    "fall",
    "forward_course_bounds",
    "reference_time_out",
    "stall_timeout",
    "velocity_violation",
]
