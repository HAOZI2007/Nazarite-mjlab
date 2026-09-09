"""Observation terms for the closed-loop SMP teacher."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.utils.lab_api.math import quat_apply_inverse

from .reference import ReferenceMotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _command(env: ManagerBasedRlEnv, command_name: str) -> ReferenceMotionCommand:
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, ReferenceMotionCommand):
    raise TypeError(f"'{command_name}' is not a ReferenceMotionCommand")
  return command


def reference_joint_pos_rel(
  env: ManagerBasedRlEnv, command_name: str = "reference"
) -> torch.Tensor:
  command = _command(env, command_name)
  robot: Entity = env.scene[command.cfg.entity_name]
  default_pos = robot.data.default_joint_pos[:, command.joint_ids]
  return command.joint_pos - default_pos


def reference_joint_vel(
  env: ManagerBasedRlEnv, command_name: str = "reference"
) -> torch.Tensor:
  return _command(env, command_name).joint_vel


def reference_root_pos_b(
  env: ManagerBasedRlEnv, command_name: str = "reference"
) -> torch.Tensor:
  command = _command(env, command_name)
  robot: Entity = env.scene[command.cfg.entity_name]
  delta_w = command.root_pos_w - robot.data.root_link_pos_w
  return quat_apply_inverse(robot.data.root_link_quat_w, delta_w)


def reference_root_lin_vel_b(
  env: ManagerBasedRlEnv, command_name: str = "reference"
) -> torch.Tensor:
  return _command(env, command_name).root_lin_vel_b


def reference_foot_target_base(
  env: ManagerBasedRlEnv, command_name: str = "reference"
) -> torch.Tensor:
  return _command(env, command_name).foot_target_base.reshape(env.num_envs, -1)


def robot_foot_pos_base(
  env: ManagerBasedRlEnv, command_name: str = "reference"
) -> torch.Tensor:
  command = _command(env, command_name)
  robot: Entity = env.scene[command.cfg.entity_name]
  foot_pos_w = robot.data.site_pos_w[:, command.site_ids]
  root_pos_w = robot.data.root_link_pos_w[:, None, :]
  root_quat_w = robot.data.root_link_quat_w[:, None, :].expand(-1, foot_pos_w.shape[1], -1)
  foot_pos_b = quat_apply_inverse(root_quat_w, foot_pos_w - root_pos_w)
  return foot_pos_b.reshape(env.num_envs, -1)


def reference_contact(
  env: ManagerBasedRlEnv, command_name: str = "reference"
) -> torch.Tensor:
  return _command(env, command_name).contact


def reference_phase(
  env: ManagerBasedRlEnv, command_name: str = "reference"
) -> torch.Tensor:
  phase = _command(env, command_name).phase
  return torch.stack(
    (phase, torch.sin(2.0 * torch.pi * phase), torch.cos(2.0 * torch.pi * phase)), dim=-1
  )
