"""Bounded joint-position actions for FR-Net recovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from mjlab.envs.mdp.actions.actions import (
  RelativeJointPositionAction,
  RelativeJointPositionActionCfg,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class SafeRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
  """Relative position action with recovery settling and target bounds."""

  max_delta: float = 0.10
  max_offset_from_default: float = 1.0
  settle_steps: int = 12
  ramp_steps: int = 12

  def build(self, env: ManagerBasedRlEnv) -> SafeRelativeJointPositionAction:
    return SafeRelativeJointPositionAction(self, env)


class SafeRelativeJointPositionAction(RelativeJointPositionAction):
  """Limit per-step and accumulated joint targets during recovery.

  Fallen environments briefly hold their reset joint positions, then linearly
  ramp policy authority. Standing environments receive actions immediately.
  """

  def __init__(self, cfg: SafeRelativeJointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    if cfg.max_delta <= 0.0 or cfg.max_offset_from_default <= 0.0:
      raise ValueError("action bounds must be positive")
    if cfg.settle_steps < 0 or cfg.ramp_steps < 0:
      raise ValueError("settle_steps and ramp_steps must be non-negative")

  def apply_actions(self) -> None:
    cfg = cast(SafeRelativeJointPositionActionCfg, self.cfg)
    current_pos = self._entity.data.joint_pos[:, self._target_ids]
    default_pos = self._entity.data.default_joint_pos[:, self._target_ids]
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]

    delta = self._processed_actions.clamp(-cfg.max_delta, cfg.max_delta)
    recovery_mask = self._env.extras.get("frnet_recovery_mask")
    if recovery_mask is None:
      recovery_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    elapsed = self._env.episode_length_buf
    if cfg.settle_steps > 0:
      settling = recovery_mask & (elapsed < cfg.settle_steps)
      delta = torch.where(settling.unsqueeze(-1), torch.zeros_like(delta), delta)
    if cfg.ramp_steps > 0:
      ramp_start = cfg.settle_steps
      ramp = ((elapsed - ramp_start + 1).float() / cfg.ramp_steps).clamp(0.0, 1.0)
      ramp = torch.where(recovery_mask, ramp, torch.ones_like(ramp))
      delta = delta * ramp.unsqueeze(-1)

    target = current_pos + delta
    target = torch.maximum(
      target,
      default_pos - cfg.max_offset_from_default,
    )
    target = torch.minimum(
      target,
      default_pos + cfg.max_offset_from_default,
    )
    self._entity.set_joint_position_target(target - encoder_bias, joint_ids=self._target_ids)
