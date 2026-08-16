"""Useful methods for MDP terminations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.nan_guard import NanGuard

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def time_out(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Terminate when the episode length exceeds its maximum."""
  return env.episode_length_buf >= env.max_episode_length


def bad_orientation(
  env: ManagerBasedRlEnv,
  limit_angle: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
  """Terminate when the asset's orientation exceeds the limit angle."""
  asset: Entity = env.scene[asset_cfg.name]
  projected_gravity = asset.data.projected_gravity_b
  return (
    torch.acos(torch.clamp(-projected_gravity[:, 2], -1.0, 1.0)).abs() > limit_angle
  )


def root_height_below_minimum(
  env: ManagerBasedRlEnv,
  minimum_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate when the asset's root height is below the minimum height."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2] < minimum_height


def state_limit(
  env: ManagerBasedRlEnv,
  max_joint_vel: float,
  max_joint_acc: float,
  max_root_lin_vel: float,
  max_root_ang_vel: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate environments whose finite state has become physically invalid."""
  asset: Entity = env.scene[asset_cfg.name]
  joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  joint_acc = asset.data.joint_acc[:, asset_cfg.joint_ids]
  root_lin_vel = asset.data.root_link_lin_vel_b
  root_ang_vel = asset.data.root_link_ang_vel_b
  return (
    (joint_vel.abs().amax(dim=1) > max_joint_vel)
    | (joint_acc.abs().amax(dim=1) > max_joint_acc)
    | (root_lin_vel.abs().amax(dim=1) > max_root_lin_vel)
    | (root_ang_vel.abs().amax(dim=1) > max_root_ang_vel)
  )


def nan_detection(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Terminate environments that have NaN/Inf values in their physics state."""
  return NanGuard.detect_nans(env.sim.data)
