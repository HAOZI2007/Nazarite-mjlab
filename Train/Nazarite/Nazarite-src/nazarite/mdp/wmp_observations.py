"""Observation helpers for the WMP-HIM training task."""

from __future__ import annotations

import torch


def empty_world_model_feature(env, feature_dim: int = 512) -> torch.Tensor:
    """Reserve the RSSM feature group; the runner replaces it before policy use."""
    return torch.zeros(
        (env.num_envs, feature_dim), device=env.device, dtype=torch.float32
    )
