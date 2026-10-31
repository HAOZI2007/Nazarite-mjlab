"""XZ1-specific reset events."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.events import reset_root_state_uniform
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def reset_root_state_uniform_facing_high_wall(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]] | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Use the usual randomized reset, but face high-wall environments +x.

    Terrain sub-patches are aligned to world +x. The robot therefore needs a
    fixed zero yaw on the high-wall type; other terrain types retain the
    configured random yaw distribution.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    terrain = env.scene.terrain
    generator = terrain.cfg.terrain_generator if terrain is not None else None
    terrain_names = list(generator.sub_terrains) if generator is not None else []
    if "high_wall" not in terrain_names:
        reset_root_state_uniform(env, env_ids, pose_range, velocity_range, asset_cfg)
        return

    high_wall_type = terrain_names.index("high_wall")
    high_wall_mask = terrain.terrain_types[env_ids] == high_wall_type
    high_wall_ids = env_ids[high_wall_mask]
    other_ids = env_ids[~high_wall_mask]

    if other_ids.numel() > 0:
        reset_root_state_uniform(env, other_ids, pose_range, velocity_range, asset_cfg)
    if high_wall_ids.numel() > 0:
        wall_pose_range = dict(pose_range)
        wall_pose_range["yaw"] = (0.0, 0.0)
        reset_root_state_uniform(
            env, high_wall_ids, wall_pose_range, velocity_range, asset_cfg
        )


__all__ = ["reset_root_state_uniform_facing_high_wall"]
