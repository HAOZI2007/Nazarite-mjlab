"""Termination terms for recovery without prematurely ending a fall."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor, TerrainHeightSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def base_below_safe_height(
  env: ManagerBasedRlEnv,
  minimum_height: float = 0.02,
  height_sensor_name: str = "base_height_scan",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """End only definite terrain penetration, measured relative to local ground."""
  del asset_cfg
  sensor = env.scene[height_sensor_name]
  if not isinstance(sensor, TerrainHeightSensor):
    raise TypeError(f"Expected TerrainHeightSensor for '{height_sensor_name}'")
  if sensor.data.heights.shape != (env.num_envs, 1):
    raise RuntimeError(
      f"'{height_sensor_name}' returned {tuple(sensor.data.heights.shape)}, "
      f"expected {(env.num_envs, 1)}"
    )

  # TerrainHeightSensor returns zero if a downward ray starts inside a terrain
  # geometry.  Unlike world-Z checks, this also works on raised stairs/boxes.
  relative_height = torch.nan_to_num(sensor.data.heights[:, 0], nan=0.0)
  return relative_height < minimum_height


def stable_recovery_mask(
  env: ManagerBasedRlEnv,
  upright_gravity_z_max: float = -0.85,
  min_relative_height: float = 0.25,
  min_foot_contacts: int = 3,
  contact_force_threshold: float = 5.0,
  max_angular_velocity: float = 1.5,
  max_joint_position_error: float = 0.35,
  foot_sensor_name: str = "feet_ground_contact",
  height_sensor_name: str = "base_height_scan",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Return environments that are genuinely standing and dynamically settled."""
  if min_foot_contacts not in (1, 2, 3, 4):
    raise ValueError("min_foot_contacts must be between 1 and 4")

  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor = env.scene[foot_sensor_name]
  if not isinstance(contact_sensor, ContactSensor):
    raise TypeError(f"Expected ContactSensor for '{foot_sensor_name}'")
  if contact_sensor.data.force is None:
    raise RuntimeError(f"Contact sensor '{foot_sensor_name}' has no force data")

  height_sensor = env.scene[height_sensor_name]
  if not isinstance(height_sensor, TerrainHeightSensor):
    raise TypeError(f"Expected TerrainHeightSensor for '{height_sensor_name}'")
  if height_sensor.data.heights.shape != (env.num_envs, 1):
    raise RuntimeError(
      f"'{height_sensor_name}' returned {tuple(height_sensor.data.heights.shape)}, "
      f"expected {(env.num_envs, 1)}"
    )

  force_norm = torch.linalg.vector_norm(contact_sensor.data.force, dim=-1)
  if force_norm.shape != (env.num_envs, 4):
    raise RuntimeError(
      f"'{foot_sensor_name}' returned force shape {tuple(force_norm.shape)}, "
      f"expected {(env.num_envs, 4)}"
    )
  foot_contacts = (force_norm >= contact_force_threshold).sum(dim=-1)
  relative_height = torch.nan_to_num(height_sensor.data.heights[:, 0], nan=0.0)
  joint_error = torch.mean(
    torch.abs(
      asset.data.joint_pos[:, asset_cfg.joint_ids]
      - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    ),
    dim=-1,
  )
  angular_velocity = torch.linalg.vector_norm(asset.data.root_link_ang_vel_b, dim=-1)
  finite = (
    torch.isfinite(asset.data.root_link_pos_w).all(dim=-1)
    & torch.isfinite(asset.data.root_link_ang_vel_b).all(dim=-1)
  )

  return (
    finite
    & (asset.data.projected_gravity_b[:, 2] <= upright_gravity_z_max)
    & (relative_height >= min_relative_height)
    & (foot_contacts >= min_foot_contacts)
    & (angular_velocity <= max_angular_velocity)
    & (joint_error <= max_joint_position_error)
  )


def soft_recovery_mask(
  env: ManagerBasedRlEnv,
  upright_gravity_z_max: float = -0.80,
  min_relative_height: float = 0.24,
  min_foot_contacts: int = 3,
  contact_force_threshold: float = 3.0,
  max_angular_velocity: float = 4.0,
  foot_sensor_name: str = "feet_ground_contact",
  height_sensor_name: str = "base_height_scan",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Return a learnable recovery milestone without requiring final posture.

  Soft success means the robot has flipped upright and obtained meaningful
  multi-foot support.  Joint-pose and low-angular-velocity requirements remain
  in ``stable_recovery_mask`` so the strict metric still measures final quality.
  """
  stable = stable_recovery_mask(
    env,
    upright_gravity_z_max=upright_gravity_z_max,
    min_relative_height=min_relative_height,
    min_foot_contacts=min_foot_contacts,
    contact_force_threshold=contact_force_threshold,
    max_angular_velocity=max_angular_velocity,
    max_joint_position_error=float("inf"),
    foot_sensor_name=foot_sensor_name,
    height_sensor_name=height_sensor_name,
    asset_cfg=asset_cfg,
  )
  return stable


def recovery_condition(
  env: ManagerBasedRlEnv,
  condition: str,
  upright_gravity_z_max: float = -0.85,
  min_relative_height: float = 0.27,
  min_foot_contacts: int = 3,
  contact_force_threshold: float = 5.0,
  max_angular_velocity: float = 3.0,
  max_joint_position_error: float = 0.35,
  foot_sensor_name: str = "feet_ground_contact",
  height_sensor_name: str = "base_height_scan",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Expose instantaneous recovery sub-conditions as [0, 1] metrics."""
  if condition not in {
    "upright",
    "height",
    "contacts",
    "angular_velocity",
    "joint_pose",
    "strict",
  }:
    raise ValueError(f"Unknown recovery condition: {condition}")

  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor = env.scene[foot_sensor_name]
  height_sensor = env.scene[height_sensor_name]
  if not isinstance(contact_sensor, ContactSensor):
    raise TypeError(f"Expected ContactSensor for '{foot_sensor_name}'")
  if not isinstance(height_sensor, TerrainHeightSensor):
    raise TypeError(f"Expected TerrainHeightSensor for '{height_sensor_name}'")
  if contact_sensor.data.force is None:
    return torch.zeros(env.num_envs, device=env.device)

  force_norm = torch.linalg.vector_norm(contact_sensor.data.force, dim=-1)
  contacts = force_norm.ge(contact_force_threshold).sum(dim=-1).ge(min_foot_contacts)
  height = torch.nan_to_num(height_sensor.data.heights[:, 0], nan=0.0).ge(
    min_relative_height
  )
  upright = asset.data.projected_gravity_b[:, 2].le(upright_gravity_z_max)
  angular_velocity = torch.linalg.vector_norm(asset.data.root_link_ang_vel_b, dim=-1)
  angular = angular_velocity.le(max_angular_velocity)
  joint_error = torch.mean(
    torch.abs(
      asset.data.joint_pos[:, asset_cfg.joint_ids]
      - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    ),
    dim=-1,
  )
  joint_pose = joint_error.le(max_joint_position_error)
  conditions = {
    "upright": upright,
    "height": height,
    "contacts": contacts,
    "angular_velocity": angular,
    "joint_pose": joint_pose,
    "strict": upright & height & contacts & angular & joint_pose,
  }
  return conditions[condition].to(dtype=torch.float32)


class StableRecoveryTermination(ManagerTermBase):
  """Terminate only after a recovery pose has remained stable for a time window."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    hold_duration_s = float(cfg.params.get("hold_duration_s", 0.3))
    if hold_duration_s <= 0.0:
      raise ValueError("hold_duration_s must be positive")
    self._required_stable_steps = max(1, math.ceil(hold_duration_s / env.step_dt))
    self._stable_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    self._stable_steps[slice(None) if env_ids is None else env_ids] = 0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    hold_duration_s: float = 0.3,
    **kwargs,
  ) -> torch.Tensor:
    del hold_duration_s
    stable = stable_recovery_mask(env, **kwargs)
    self._stable_steps = torch.where(
      stable, self._stable_steps + 1, torch.zeros_like(self._stable_steps)
    )
    return self._stable_steps >= self._required_stable_steps


class SoftRecoveryTermination(StableRecoveryTermination):
  """Terminate at a supported-upright milestone to drive terrain learning."""

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    hold_duration_s: float = 0.15,
    **kwargs,
  ) -> torch.Tensor:
    del hold_duration_s
    stable = soft_recovery_mask(env, **kwargs)
    self._stable_steps = torch.where(
      stable, self._stable_steps + 1, torch.zeros_like(self._stable_steps)
    )
    return self._stable_steps >= self._required_stable_steps
