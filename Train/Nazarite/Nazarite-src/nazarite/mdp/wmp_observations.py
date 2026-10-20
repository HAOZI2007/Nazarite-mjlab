"""Observation helpers for the WMP-HIM training task."""

from __future__ import annotations

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def empty_world_model_feature(env, feature_dim: int = 512) -> torch.Tensor:
    """Reserve the RSSM feature group; the runner replaces it before policy use."""
    return torch.zeros(
        (env.num_envs, feature_dim), device=env.device, dtype=torch.float32
    )


def amp_state(env, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """The 30 numbers the motion discriminator scores.

    Joints, base linear velocity, base angular velocity and joint velocity, in that
    order, all absolute: no noise, no history, no terrain. The reference task feeds
    the discriminator exactly this much, and the clip loader produces the same
    layout, so the two can be compared without reordering anything.
    """
    asset = env.scene[asset_cfg.name]
    return torch.cat(
        (
            asset.data.joint_pos,
            asset.data.root_link_lin_vel_b,
            asset.data.root_link_ang_vel_b,
            asset.data.joint_vel,
        ),
        dim=-1,
    )
