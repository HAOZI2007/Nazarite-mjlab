"""Nazarite curriculum hooks."""

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg


_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


def terrain_levels_vel_strict(
  env, env_ids: torch.Tensor, command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
):
  """Advance terrain levels without impossible distance requirements.

  The native curriculum compares walked distance against
  ``|command| * episode_length * 0.5``.  On Nazarite's 8 m tiles that value
  can exceed the physically available half-tile distance, so a successful
  traversal is incorrectly downgraded.  The target is capped to a safe
  fraction of the tile length while preserving command-dependent early-fall
  detection.
  """
  asset: Entity = env.scene[asset_cfg.name]
  terrain = env.scene.terrain
  assert terrain is not None and terrain.cfg.terrain_generator is not None
  generator = terrain.cfg.terrain_generator
  command = env.command_manager.get_command(command_name)
  assert command is not None

  distance = torch.linalg.vector_norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
    dim=1,
  )
  tile_half = float(generator.size[0]) * 0.5
  move_up = distance > tile_half
  command_distance = torch.linalg.vector_norm(command[env_ids, :2], dim=1)
  target_distance = command_distance * env.max_episode_length_s * 0.5
  # A failed traversal should only downgrade after covering less than 60% of
  # the available half-tile, never require more distance than the tile offers.
  target_distance = torch.minimum(
    target_distance,
    torch.full_like(target_distance, tile_half * 0.6),
  )
  move_down = (distance < target_distance) & ~move_up
  if env.common_step_counter == 0:
    move_up = torch.zeros_like(move_up)
    move_down = torch.zeros_like(move_down)

  terrain.update_env_origins(env_ids, move_up, move_down)
  levels = terrain.terrain_levels.float()
  result = {"mean": torch.mean(levels), "max": torch.max(levels)}
  names = list(generator.sub_terrains.keys())
  origins = terrain.terrain_origins
  if origins is not None and origins.shape[1] == len(names):
    for i, name in enumerate(names):
      mask = terrain.terrain_types == i
      if mask.any():
        result[name] = torch.mean(levels[mask])
  return result


__all__ = ["terrain_levels_vel_strict"]
