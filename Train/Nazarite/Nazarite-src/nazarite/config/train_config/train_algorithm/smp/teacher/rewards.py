"""Reward terms for training a closed-loop reference-tracking teacher."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.sensor import ContactSensor

from .observations import robot_foot_pos_base
from .reference import ReferenceMotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _command(env: ManagerBasedRlEnv, command_name: str) -> ReferenceMotionCommand:
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, ReferenceMotionCommand):
    raise TypeError(f"'{command_name}' is not a ReferenceMotionCommand")
  return command


def reference_joint_position_tracking(
  env: ManagerBasedRlEnv, command_name: str = "reference", std: float = 0.25
) -> torch.Tensor:
  command = _command(env, command_name)
  robot: Entity = env.scene[command.cfg.entity_name]
  error = command.joint_pos - robot.data.joint_pos[:, command.joint_ids]
  return torch.exp(-torch.mean(torch.square(error), dim=-1) / (std * std))


def reference_joint_velocity_tracking(
  env: ManagerBasedRlEnv, command_name: str = "reference", std: float = 2.0
) -> torch.Tensor:
  command = _command(env, command_name)
  robot: Entity = env.scene[command.cfg.entity_name]
  error = command.joint_vel - robot.data.joint_vel[:, command.joint_ids]
  return torch.exp(-torch.mean(torch.square(error), dim=-1) / (std * std))


def reference_foot_position_tracking(
  env: ManagerBasedRlEnv, command_name: str = "reference", std: float = 0.08
) -> torch.Tensor:
  command = _command(env, command_name)
  error = command.foot_target_base - robot_foot_pos_base(env, command_name).reshape(
    env.num_envs, 4, 3
  )
  return torch.exp(-torch.mean(torch.square(error), dim=(-1, -2)) / (std * std))


def reference_root_velocity_tracking(
  env: ManagerBasedRlEnv, command_name: str = "reference", std: float = 0.5
) -> torch.Tensor:
  command = _command(env, command_name)
  robot: Entity = env.scene[command.cfg.entity_name]
  error = command.root_lin_vel_b - robot.data.root_link_lin_vel_b
  return torch.exp(-torch.mean(torch.square(error), dim=-1) / (std * std))


def reference_contact_tracking(
  env: ManagerBasedRlEnv,
  command_name: str = "reference",
  sensor_name: str = "feet_ground_contact",
  force_threshold: float = 5.0,
) -> torch.Tensor:
  command = _command(env, command_name)
  sensor = env.scene[sensor_name]
  if not isinstance(sensor, ContactSensor) or sensor.data.force is None:
    raise TypeError(f"'{sensor_name}' must be a ContactSensor with force data")
  force = torch.linalg.vector_norm(sensor.data.force, dim=-1)
  if force.shape[1] != 4:
    force = force.reshape(env.num_envs, 4, -1).amax(dim=-1)
  actual_contact = (force >= force_threshold).to(dtype=torch.float32)
  return 1.0 - torch.mean(torch.abs(actual_contact - command.contact), dim=-1)


def upright_tracking(env: ManagerBasedRlEnv, std: float = 0.25) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  gravity_xy = robot.data.projected_gravity_b[:, :2]
  return torch.exp(-torch.sum(torch.square(gravity_xy), dim=-1) / (std * std))
