"""HIM-specific termination conditions."""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg


def fell_from_single_bridge(
    env,
    terrain_name: str = "single_bridge",
    obstacle_x: float = 4.40,
    bridge_length: float = 3.0,
    bridge_width: float = 0.30,
    bridge_height: float = 0.10,
    base_height_target: float = 0.32,
    behavior_command_name: str = "behavior",
    drop_margin: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate bridge environments once the robot drops below bridge level."""
    terrain = env.scene.terrain
    generator = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
    terrain_types = getattr(terrain, "terrain_types", None)
    origins = getattr(terrain, "env_origins", None)
    if generator is None or not isinstance(terrain_types, torch.Tensor) or origins is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    names = list(generator.sub_terrains)
    if terrain_name not in names:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    bridge_mask = terrain_types == names.index(terrain_name)
    asset: Entity = env.scene[asset_cfg.name]
    local_pos = asset.data.root_link_pos_w - origins
    # The obstacle terrain spawn origin is one metre from the patch edge.
    bridge_center_x = obstacle_x - 1.0
    in_bridge_x = (local_pos[:, 0] - bridge_center_x).abs() <= bridge_length * 0.5
    outside_bridge_sides = local_pos[:, 1].abs() > bridge_width * 0.5 + 0.05

    behavior = env.command_manager.get_command(behavior_command_name)
    height_offset = behavior[:, 0] if behavior.ndim == 2 and behavior.shape[1] else 0.0
    minimum_root_height = bridge_height + base_height_target + height_offset - drop_margin
    root_height = asset.data.root_link_pos_w[:, 2] - origins[:, 2]
    fell_height = root_height < minimum_root_height
    # A side fall may settle on the floor at nearly the nominal base height,
    # so also terminate when the robot leaves the narrow bridge footprint and
    # has dropped halfway from the bridge top to the floor.
    side_fall_height = bridge_height + base_height_target + height_offset - drop_margin * 0.5
    fell_side = outside_bridge_sides & (root_height < side_fall_height)
    fell = bridge_mask & in_bridge_x & (fell_height | fell_side)
    if hasattr(env, "extras") and "log" in env.extras:
        env.extras["log"]["HIM/bridge_fall"] = fell.float().mean()
    return fell


__all__ = ["fell_from_single_bridge"]
