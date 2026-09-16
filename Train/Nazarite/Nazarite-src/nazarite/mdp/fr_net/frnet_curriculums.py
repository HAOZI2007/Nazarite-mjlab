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
  success_termination_name: str = "soft_recovery_success",
  low_level_exploration_probability: float = 0.25,
  min_episodes_per_type: int = 128,
  promotion_success_rate: float = 0.70,
  demotion_success_rate: float = 0.35,
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
  if min_episodes_per_type < 1:
    raise ValueError("min_episodes_per_type must be positive")
  if not 0.0 <= demotion_success_rate < promotion_success_rate <= 1.0:
    raise ValueError("curriculum success thresholds are invalid")
  terrain = env.scene.terrain
  if terrain is None or terrain.terrain_origins is None:
    raise RuntimeError("FR-Net terrain curriculum requires generated terrain")

  if isinstance(env_ids, slice):
    resolved_env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
  else:
    resolved_env_ids = resolve_env_ids(env, env_ids)
  levels = terrain.terrain_levels
  terrain_types = terrain.terrain_types[resolved_env_ids]
  success = torch.zeros(len(resolved_env_ids), dtype=torch.bool, device=env.device)
  if env.common_step_counter != 0:
    success = env.termination_manager.get_term(success_termination_name)[
      resolved_env_ids
    ]

  # Keep type-level statistics on the environment instead of in the config.
  # This avoids upgrading a difficult boxes column merely because flat
  # episodes are successful.  The counters are reset only when a new env is
  # constructed, so they survive individual episode resets.
  terrain_generator = terrain.cfg.terrain_generator
  if terrain_generator is None:
    raise RuntimeError("FR-Net terrain curriculum requires a terrain generator")
  num_types = len(terrain_generator.sub_terrains)
  state = getattr(env, "frnet_curriculum_state", None)
  if state is None:
    state = {
      "trials": torch.zeros(num_types, dtype=torch.float32, device=env.device),
      "successes": torch.zeros(num_types, dtype=torch.float32, device=env.device),
    }
    env.__dict__["frnet_curriculum_state"] = state
  trials_by_type = state["trials"]
  successes_by_type = state["successes"]
  if env.common_step_counter != 0:
    for terrain_type in range(num_types):
      mask = terrain_types == terrain_type
      if mask.any():
        trials_by_type[terrain_type] += mask.float().sum()
        successes_by_type[terrain_type] += success[mask].float().sum()

  retry_low_level = torch.zeros(
    len(resolved_env_ids), dtype=torch.bool, device=env.device
  )
  if env.common_step_counter != 0:
    trials = trials_by_type[terrain_types]
    successes = successes_by_type[terrain_types]
    success_rate = successes / trials.clamp_min(1.0)
    enough_statistics = trials >= min_episodes_per_type
    promote = enough_statistics & (success_rate >= promotion_success_rate)
    demote = enough_statistics & (success_rate <= demotion_success_rate)
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
      move_up=promote | retry_low_level,
      move_down=demote & (~retry_low_level),
    )

  result: dict[str, torch.Tensor] = {
    "mean_level": levels.float().mean(),
    "max_level": levels.max(),
    "low_level_retry_fraction": retry_low_level.float().mean(),
  }
  type_trials = trials_by_type[terrain_types]
  type_successes = successes_by_type[terrain_types]
  result["batch_success_rate"] = success.float().mean()
  result["batch_ready_fraction"] = (
    (type_trials >= min_episodes_per_type).float().mean()
  )
  result["batch_type_success_rate"] = (
    (type_successes / type_trials.clamp_min(1.0)).mean()
  )
  for terrain_type, name in enumerate(terrain_generator.sub_terrains):
    mask = terrain.terrain_types == terrain_type
    if mask.any():
      result[f"{name}_level"] = levels[mask].float().mean()
  return result
