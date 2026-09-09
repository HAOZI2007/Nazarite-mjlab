"""Termination terms for the reference-tracking teacher."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity

from .reference import ReferenceMotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def reference_finished(
  env: ManagerBasedRlEnv, command_name: str = "reference"
) -> torch.Tensor:
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, ReferenceMotionCommand):
    raise TypeError(f"'{command_name}' is not a ReferenceMotionCommand")
  return command.finished


def bad_orientation(
  env: ManagerBasedRlEnv, limit_angle: float = 1.2
) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  gravity = robot.data.projected_gravity_b
  angle = torch.acos(torch.clamp(-gravity[:, 2], -1.0, 1.0))
  return angle > limit_angle


def root_height_below_minimum(
  env: ManagerBasedRlEnv, minimum_height: float = 0.16
) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  return robot.data.root_link_pos_w[:, 2] < minimum_height
