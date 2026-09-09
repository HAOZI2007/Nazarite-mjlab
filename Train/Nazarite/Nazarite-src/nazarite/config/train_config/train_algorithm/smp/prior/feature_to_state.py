"""Invert Go2 SMP feature windows for generative state initialization."""

from __future__ import annotations

import torch
from torch.nn import functional

from mjlab.utils.lab_api.math import quat_from_matrix

from .features import GO2_SMP_FEATURE_DIM


def split_features(window: torch.Tensor) -> dict[str, torch.Tensor]:
    if window.shape[-1] != GO2_SMP_FEATURE_DIM:
        raise ValueError(
            f"expected feature_dim={GO2_SMP_FEATURE_DIM}, got {window.shape[-1]}"
        )
    return {
        "root_pos": window[..., 0:3],
        "root_rot_6d": window[..., 3:9],
        "joint_pos": window[..., 9:21],
        "foot_pos": window[..., 21:33],
        "root_lin_vel": window[..., 33:36],
        "root_ang_vel": window[..., 36:39],
    }


def rotation_6d_to_matrix(rotation: torch.Tensor) -> torch.Tensor:
    """Invert SMP's [col0,col2] rotation representation robustly."""
    column_0 = functional.normalize(rotation[..., :3], dim=-1)
    column_2 = rotation[..., 3:6]
    column_2 = column_2 - (column_0 * column_2).sum(dim=-1, keepdim=True) * column_0
    column_2 = functional.normalize(column_2, dim=-1)
    column_1 = torch.cross(column_2, column_0, dim=-1)
    return torch.stack((column_0, column_1, column_2), dim=-1)


def rotation_6d_to_quaternion(rotation: torch.Tensor) -> torch.Tensor:
    return quat_from_matrix(rotation_6d_to_matrix(rotation))
