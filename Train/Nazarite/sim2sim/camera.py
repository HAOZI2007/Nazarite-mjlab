"""Common viewer camera helpers for the standalone sim2sim runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FollowCameraConfig:
    """Fixed viewing angles with a target that follows the robot base."""

    distance: float = 2.5
    azimuth: float = 135.0
    elevation: float = -20.0
    target_height: float = 0.12

    def __post_init__(self) -> None:
        if self.distance <= 0.0:
            raise ValueError("camera distance must be positive")


def follow_robot(viewer: Any, base_position: np.ndarray, config: FollowCameraConfig) -> None:
    """Move the viewer target to the robot while keeping a stable viewpoint."""
    base_position = np.asarray(base_position, dtype=np.float64)
    if base_position.shape != (3,):
        raise ValueError(f"base_position must have shape (3,), got {base_position.shape}")

    viewer.cam.lookat[:] = base_position + np.array(
        [0.0, 0.0, config.target_height], dtype=np.float64
    )
    viewer.cam.distance = config.distance
    viewer.cam.azimuth = config.azimuth
    viewer.cam.elevation = config.elevation
