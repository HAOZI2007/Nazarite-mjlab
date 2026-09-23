"""Stepping-stones scene and DELTA BEV map sampling for MuJoCo sim2sim."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from ..mujoco_io import MuJoCoIO
from .config import (
    MAP_CHANNELS,
    MAP_HEIGHT,
    MAP_RAY_HEIGHT,
    MAP_WIDTH,
    MAP_X_RANGE,
    MAP_Y_RANGE,
    MAP_Z_SCALE,
)

_BASE_BODY_NAME = "base_link"


def _add_box(
    spec: mujoco.MjSpec,
    name: str,
    position: tuple[float, float, float],
    half_size: tuple[float, float, float],
    rgba: tuple[float, float, float, float],
) -> None:
    geom = spec.worldbody.add_geom()
    geom.name = name
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.pos[:] = position
    geom.size[:] = half_size
    geom.rgba[:] = rgba
    geom.friction[:] = [0.8, 0.02, 0.01]


def add_stepping_stones_scene(spec: mujoco.MjSpec) -> None:
    """Add a simple approach platform, a deep pit, and separated stones."""
    # The recessed floor makes gaps observable and physically meaningful.  It
    # is deliberately below the stone tops, matching the training terrain.
    _add_box(
        spec,
        "stepping_stones_pit_floor",
        (2.5, 0.0, -2.05),
        (7.5, 4.0, 0.05),
        (0.08, 0.08, 0.09, 1.0),
    )

    platform_x_min, platform_x_max = -2.0, 0.72
    _add_box(
        spec,
        "stepping_stones_approach_platform",
        ((platform_x_min + platform_x_max) / 2.0, 0.0, -1.0),
        ((platform_x_max - platform_x_min) / 2.0, 0.70, 1.0),
        (0.28, 0.46, 0.30, 1.0),
    )

    stone_top = 0.06
    stone_size = 0.82
    stone_height = stone_top + 2.0
    stone_x = (1.30, 2.25, 3.20, 4.15, 5.10, 6.05)
    stone_y = (-0.43, 0.43)
    for x_index, x in enumerate(stone_x):
        for y_index, y in enumerate(stone_y):
            # A small deterministic offset keeps the test course from being
            # perfectly symmetric without making it harder to reproduce.
            x_offset = 0.025 * ((x_index % 3) - 1)
            y_offset = 0.02 if (x_index + y_index) % 2 else -0.02
            _add_box(
                spec,
                f"stepping_stone_{x_index}_{y_index}",
                (x + x_offset, y + y_offset, -2.0 + stone_height / 2.0),
                (stone_size / 2.0, stone_size / 2.0, stone_height / 2.0),
                (0.20, 0.68, 0.34, 1.0),
            )

    # Low side rails show the course boundaries without filling the gaps.
    _add_box(
        spec,
        "stepping_stones_left_border",
        (2.5, -1.30, -1.0),
        (7.5, 0.06, 1.0),
        (0.12, 0.30, 0.16, 1.0),
    )
    _add_box(
        spec,
        "stepping_stones_right_border",
        (2.5, 1.30, -1.0),
        (7.5, 0.06, 1.0),
        (0.12, 0.30, 0.16, 1.0),
    )


def _body_rotation(quat_wxyz: np.ndarray) -> np.ndarray:
    rotation = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, np.asarray(quat_wxyz, dtype=np.float64))
    return rotation.reshape(3, 3)


def build_delta_map(io: MuJoCoIO) -> np.ndarray:
    """Sample a [16, 26, 4] robot-frame map using downward MuJoCo rays.

    Channels match ``delta_depth_image(project_to_bev=True)``: normalized cell
    x, normalized cell y, normalized hit height, and a validity confidence.
    """
    data = io.data
    base_position = np.asarray(data.qpos[:3], dtype=np.float64)
    rotation = _body_rotation(data.qpos[3:7])
    inverse_rotation = rotation.T
    x0, x1 = MAP_X_RANGE
    y0, y1 = MAP_Y_RANGE
    xs = np.linspace(
        x0 + 0.5 * (x1 - x0) / MAP_WIDTH,
        x1 - 0.5 * (x1 - x0) / MAP_WIDTH,
        MAP_WIDTH,
    )
    ys = np.linspace(
        y1 - 0.5 * (y1 - y0) / MAP_HEIGHT,
        y0 + 0.5 * (y1 - y0) / MAP_HEIGHT,
        MAP_HEIGHT,
    )
    output = np.zeros((MAP_HEIGHT, MAP_WIDTH, MAP_CHANNELS), dtype=np.float32)
    base_body_id = mujoco.mj_name2id(
        io.model, mujoco.mjtObj.mjOBJ_BODY, _BASE_BODY_NAME
    )
    if base_body_id < 0:
        raise RuntimeError(f"Could not resolve Go2 base body {_BASE_BODY_NAME!r}")
    for row, y in enumerate(ys):
        for column, x in enumerate(xs):
            output[row, column, 0] = (x - x0) / (x1 - x0) * 2.0 - 1.0
            output[row, column, 1] = (y - y0) / (y1 - y0) * 2.0 - 1.0
            local_origin = np.array([x, y, MAP_RAY_HEIGHT], dtype=np.float64)
            ray_origin = base_position + rotation @ local_origin
            ray_direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)
            distance = mujoco.mj_ray(
                io.model,
                data,
                ray_origin,
                ray_direction,
                np.ones(6, dtype=np.uint8),
                True,
                # Exclude the Go2 body tree while retaining world/static
                # geometry.  bodyexclude=0 would exclude the terrain itself.
                base_body_id,
                np.zeros(1, dtype=np.int32),
            )
            if distance < 0.0 or not math.isfinite(distance):
                continue
            hit_world = ray_origin + distance * ray_direction
            hit_body = inverse_rotation @ (hit_world - base_position)
            output[row, column, 2] = np.clip(
                hit_body[2] / MAP_Z_SCALE, -1.0, 1.0
            )
            output[row, column, 3] = 1.0
    if not np.all(np.isfinite(output)):
        raise FloatingPointError("DELTA map contains NaN or Inf")
    return output.reshape(-1)
