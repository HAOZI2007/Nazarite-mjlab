"""Curriculum terms specific to FR-Net recovery on generated terrain."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.events import resolve_env_ids

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def terrain_levels_from_recovery_success(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
  success_termination_name: str = "recovery_success",
  low_level_exploration_probability: float = 0.25,
) -> dict[str, torch.Tensor]:
  """Advance only episodes that completed a stable recovery.

  The curriculum manager runs before reset events.  It can therefore consume
  the previous episode's termination flag, choose the next terrain origin, and
  let ``reset_fallen_root_state`` spawn at that new origin in the same reset.

  Failed level-0 episodes are retried at level 1 with a small probability.  A
  pure success-gated curriculum otherwise collapses permanently to the flat
  start level before the policy has ever produced its first strict recovery
  success.  Levels 2 and above still require a genuine stable recovery.
  """
  if not 0.0 <= low_level_exploration_probability <= 1.0:
    raise ValueError("low_level_exploration_probability must be in [0, 1]")
  terrain = env.scene.terrain
  if terrain is None or terrain.terrain_origins is None:
    raise RuntimeError("FR-Net terrain curriculum requires generated terrain")

  if isinstance(env_ids, slice):
    resolved_env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
  else:
    resolved_env_ids = resolve_env_ids(env, env_ids)
  levels = terrain.terrain_levels
  retry_low_level = torch.zeros(
    len(resolved_env_ids), dtype=torch.bool, device=env.device
  )
  if env.common_step_counter != 0:
    success = env.termination_manager.get_term(success_termination_name)[
      resolved_env_ids
    ]
    retry_low_level = (
      ~success
      & (levels[resolved_env_ids] == 0)
      & (
        torch.rand(len(resolved_env_ids), device=env.device)
        < low_level_exploration_probability
      )
    )
    terrain.update_env_origins(
      resolved_env_ids,
      move_up=success | retry_low_level,
      move_down=(~success) & (~retry_low_level),
    )

  result: dict[str, torch.Tensor] = {
    "mean_level": levels.float().mean(),
    "max_level": levels.max(),
    "low_level_retry_fraction": retry_low_level.float().mean(),
  }
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None
  for terrain_type, name in enumerate(terrain_generator.sub_terrains):
    mask = terrain.terrain_types == terrain_type
    if mask.any():
      result[f"{name}_level"] = levels[mask].float().mean()
  return result
