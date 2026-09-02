"""Reward terms for the FR-Net fall-recovery task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor, TerrainHeightSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def upright_gaussian(
  env: ManagerBasedRlEnv,
  std: float = 0.15,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward the base gravity vector approaching ``[0, 0, -1]``."""
  asset: Entity = env.scene[asset_cfg.name]
  gravity = asset.data.projected_gravity_b
  target = gravity.new_tensor((0.0, 0.0, -1.0))
  error = torch.sum(torch.square(gravity - target), dim=-1)
  return torch.exp(-error / max(float(std) ** 2, 1.0e-6))


def base_height_relative_gaussian(
  env: ManagerBasedRlEnv,
  target_height: float = 0.32,
  std: float = 0.08,
  height_sensor_name: str = "base_height_scan",
) -> torch.Tensor:
  """Reward base clearance above the local terrain, rather than world Z."""
  sensor = env.scene[height_sensor_name]
  if not isinstance(sensor, TerrainHeightSensor):
    raise TypeError(f"Expected TerrainHeightSensor for '{height_sensor_name}'")
  if sensor.data.heights.shape != (env.num_envs, 1):
    raise RuntimeError(
      f"'{height_sensor_name}' returned {tuple(sensor.data.heights.shape)}, "
      f"expected {(env.num_envs, 1)}"
    )

  # The probe frame is base_link, so terrain clearance is exactly the base
  # height relative to the highest nearby terrain sample. A ray that starts
  # inside terrain reports zero clearance, yielding no height reward.
  relative_height = torch.nan_to_num(sensor.data.heights[:, 0], nan=0.0)
  height_error = torch.square(relative_height - target_height)
  return torch.exp(-height_error / max(float(std) ** 2, 1.0e-6))


def foot_contact_count(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 5.0,
) -> torch.Tensor:
  """Return the number of feet with a robust ground contact in [0, 4]."""
  sensor = env.scene[sensor_name]
  if not isinstance(sensor, ContactSensor):
    raise TypeError(f"Expected ContactSensor for '{sensor_name}'")
  if sensor.data.force is None:
    return torch.zeros(env.num_envs, device=env.device)
  force_norm = torch.linalg.vector_norm(sensor.data.force, dim=-1)
  return (force_norm >= force_threshold).to(dtype=torch.float32).sum(dim=-1)


def recovered_stand_pose(
  env: ManagerBasedRlEnv,
  upright_threshold: float = -0.8,
  std: float = 0.45,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward the nominal standing pose only after the body is mostly upright."""
  asset: Entity = env.scene[asset_cfg.name]
  joint_error = torch.mean(
    torch.square(
      asset.data.joint_pos[:, asset_cfg.joint_ids]
      - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    ),
    dim=-1,
  )
  upright = asset.data.projected_gravity_b[:, 2] <= upright_threshold
  return upright.to(dtype=joint_error.dtype) * torch.exp(
    -joint_error / max(float(std) ** 2, 1.0e-6)
  )


def upright_stability_support(
  env: ManagerBasedRlEnv,
  upright_threshold: float = -0.8,
  contact_force_threshold: float = 5.0,
  angular_velocity_std: float = 1.5,
  foot_sensor_name: str = "feet_ground_contact",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward quiet multi-foot support after the trunk has become upright.

  The upright/height rewards are sufficient to learn a flip, but they do not
  distinguish a settled stand from an upright pose that is still windmilling.
  This term is gated by orientation, then rewards both low base angular velocity
  and distributing support across all four feet.
  """
  if angular_velocity_std <= 0.0:
    raise ValueError("angular_velocity_std must be positive")

  asset: Entity = env.scene[asset_cfg.name]
  sensor = env.scene[foot_sensor_name]
  if not isinstance(sensor, ContactSensor):
    raise TypeError(f"Expected ContactSensor for '{foot_sensor_name}'")
  if sensor.data.force is None:
    return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

  force_norm = torch.linalg.vector_norm(sensor.data.force, dim=-1)
  if force_norm.shape != (env.num_envs, 4):
    raise RuntimeError(
      f"'{foot_sensor_name}' returned force shape {tuple(force_norm.shape)}, "
      f"expected {(env.num_envs, 4)}"
    )
  contact_fraction = (force_norm >= contact_force_threshold).float().mean(dim=-1)
  angular_velocity = torch.linalg.vector_norm(asset.data.root_link_ang_vel_b, dim=-1)
  stillness = torch.exp(-torch.square(angular_velocity / angular_velocity_std))
  upright = asset.data.projected_gravity_b[:, 2] <= upright_threshold
  return upright.to(dtype=stillness.dtype) * contact_fraction * stillness


def nonnegative_total_reward_correction(
  env: ManagerBasedRlEnv,
  source_term_names: tuple[str, ...],
) -> torch.Tensor:
  """Add exactly enough reward to clamp the current total reward rate at zero.

  This term must be the final reward term.  ``RewardManager`` has already
  evaluated every source term by then, and its step buffer stores their weighted,
  unscaled contributions.  Adding ``max(-sum(sources), 0)`` keeps all individual
  regularization costs active while changing the rollout reward from ``raw`` to
  ``max(raw, 0)``.  It deliberately avoids the harmful alternative of deleting
  torque, action-rate, or angular-velocity penalties term by term.
  """
  if not source_term_names:
    raise ValueError("source_term_names must not be empty")

  reward_manager = env.reward_manager
  indices: list[int] = []
  for name in source_term_names:
    if name not in reward_manager.active_terms:
      raise ValueError(f"Reward term '{name}' is not active")
    indices.append(reward_manager.active_terms.index(name))

  source_reward_rate = reward_manager._step_reward[:, indices].sum(dim=-1)
  return torch.clamp(-source_reward_rate, min=0.0)


def stable_recovery_completion_bonus(
  env: ManagerBasedRlEnv,
  termination_name: str = "recovery_success",
) -> torch.Tensor:
  """Emit one unscaled terminal bonus when stable recovery has completed.

  The termination manager runs before the reward manager. Its success flag is
  therefore the single source of truth for both the terminal transition and
  this reward. RewardManager scales terms by ``env.step_dt``; dividing here
  makes the configured reward weight the actual one-time bonus in return.
  """
  try:
    completed = env.termination_manager.get_term(termination_name)
  except ValueError:
    # Fixed-fall evaluation deliberately clears training terminations.
    return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
  return completed.to(dtype=torch.float32) / env.step_dt
