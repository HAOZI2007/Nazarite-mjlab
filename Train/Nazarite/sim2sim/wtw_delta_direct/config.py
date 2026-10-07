"""Deployment contract for Nazarite-WTW-Delta-Direct-Go2.

The values in this module mirror ``Nazarite_Wtw_Delta_Direct_Go2``.  Keeping
them separate from the older residual task prevents an accidental change in
map shape or camera geometry from silently producing a plausible-looking but
incompatible policy input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from nazarite.delta.d435 import DELTA_DIRECT_D435_INTRINSICS

from ..config import ACTION_DIM, ROOT
from ..wtw_delta_residual.depth_camera import (
    BevConfig,
    DepthCameraConfig,
    DepthObservation,
)

MAP_HEIGHT = 16
MAP_WIDTH = 26
MAP_CHANNELS = 5
MAP_DIM = MAP_HEIGHT * MAP_WIDTH * MAP_CHANNELS
MAP_X_RANGE = (0.0, 2.0)
MAP_Y_RANGE = (-0.6, 0.6)
MAP_Z_SCALE = 0.8
FILL_KERNEL_SIZE = 3

WTW_OBS_DIM = 498
DELTA_PROPRIO_DIM = 61
DELTA_MAP_DIM = MAP_DIM

DIRECT_CAMERA = DepthCameraConfig(
    name="sim2sim_wtw_delta_direct_depth_camera",
    width=72,
    height=128,
    pos_b=(0.40, 0.0, 0.05),
    pitch_deg=0.0,
    roll_deg=0.0,
    fx_px=DELTA_DIRECT_D435_INTRINSICS.fx,
    fy_px=DELTA_DIRECT_D435_INTRINSICS.fy,
    cx_px=DELTA_DIRECT_D435_INTRINSICS.cx,
    cy_px=DELTA_DIRECT_D435_INTRINSICS.cy,
    max_depth_m=5.0,
)

DIRECT_BEV = BevConfig(
    height=MAP_HEIGHT,
    width=MAP_WIDTH,
    x_range_m=MAP_X_RANGE,
    y_range_m=MAP_Y_RANGE,
    z_scale_m=MAP_Z_SCALE,
    fill_kernel_size=FILL_KERNEL_SIZE,
)

_POLICY_CANDIDATES = (
    ROOT
    / "logs/rsl_rl/go2_wtw_delta_direct/2026-09-24_22-11-00/model_9950.pt",
    ROOT
    / "logs/rsl_rl/go2_wtw_delta_direct/2026-09-24_16-39-18/model_*.pt",
)


def _latest_glob_candidate(pattern: Path) -> Path | None:
    if "*" not in pattern.name:
        return pattern if pattern.is_file() else None
    candidates = sorted(pattern.parent.glob(pattern.name))
    return candidates[-1] if candidates else None


POLICY = next(
    (candidate for pattern in _POLICY_CANDIDATES if (candidate := _latest_glob_candidate(pattern))),
    _POLICY_CANDIDATES[0],
)


def build_configs(args: Any) -> tuple[DepthCameraConfig, BevConfig]:
    """Apply CLI overrides while retaining the Direct task defaults."""
    width = DIRECT_CAMERA.width if args.depth_width is None else int(args.depth_width)
    height = DIRECT_CAMERA.height if args.depth_height is None else int(args.depth_height)
    scale_x = width / DIRECT_CAMERA.width
    scale_y = height / DIRECT_CAMERA.height
    roll = DIRECT_CAMERA.roll_deg if args.depth_roll is None else float(args.depth_roll)

    # A portrait stream created by a 90-degree optical-axis roll swaps the
    # calibrated horizontal/vertical axes. Explicit CLI intrinsics always win.
    portrait = height > width and abs(abs(roll) % 180.0 - 90.0) < 1.0e-6
    if portrait:
        fx = DIRECT_CAMERA.fy_px * width / DIRECT_CAMERA.height
        fy = DIRECT_CAMERA.fx_px * height / DIRECT_CAMERA.width
        cx = DIRECT_CAMERA.cy_px * width / DIRECT_CAMERA.height
        cy = DIRECT_CAMERA.cx_px * height / DIRECT_CAMERA.width
    else:
        fx = DIRECT_CAMERA.fx_px * scale_x
        fy = DIRECT_CAMERA.fy_px * scale_y
        cx = DIRECT_CAMERA.cx_px * scale_x
        cy = DIRECT_CAMERA.cy_px * scale_y

    camera = DepthCameraConfig(
        name=DIRECT_CAMERA.name,
        width=width,
        height=height,
        pos_b=(
            DIRECT_CAMERA.pos_b[0] if args.depth_pos_x is None else float(args.depth_pos_x),
            DIRECT_CAMERA.pos_b[1] if args.depth_pos_y is None else float(args.depth_pos_y),
            DIRECT_CAMERA.pos_b[2] if args.depth_pos_z is None else float(args.depth_pos_z),
        ),
        pitch_deg=(DIRECT_CAMERA.pitch_deg if args.depth_pitch is None else float(args.depth_pitch)),
        roll_deg=roll,
        fx_px=fx if args.depth_fx is None else float(args.depth_fx),
        fy_px=fy if args.depth_fy is None else float(args.depth_fy),
        cx_px=cx if args.depth_cx is None else float(args.depth_cx),
        cy_px=cy if args.depth_cy is None else float(args.depth_cy),
        max_depth_m=(DIRECT_CAMERA.max_depth_m if args.depth_max is None else float(args.depth_max)),
    )
    x_range = (
        MAP_X_RANGE[0] if args.bev_x_min is None else float(args.bev_x_min),
        MAP_X_RANGE[1] if args.bev_x_max is None else float(args.bev_x_max),
    )
    y_range = (
        MAP_Y_RANGE[0] if args.bev_y_min is None else float(args.bev_y_min),
        MAP_Y_RANGE[1] if args.bev_y_max is None else float(args.bev_y_max),
    )
    bev = BevConfig(
        height=MAP_HEIGHT if args.bev_height is None else int(args.bev_height),
        width=MAP_WIDTH if args.bev_width is None else int(args.bev_width),
        x_range_m=x_range,
        y_range_m=y_range,
        z_scale_m=(MAP_Z_SCALE if args.bev_z_scale is None else float(args.bev_z_scale)),
        fill_kernel_size=(FILL_KERNEL_SIZE if args.bev_fill_kernel is None else int(args.bev_fill_kernel)),
    )
    if (bev.height, bev.width) != (MAP_HEIGHT, MAP_WIDTH):
        raise ValueError(
            f"Direct policy requires BEV {MAP_HEIGHT}x{MAP_WIDTH}, got {bev.height}x{bev.width}"
        )
    return camera, bev


def build_policy_map(observation: DepthObservation, bev: BevConfig) -> np.ndarray:
    """Convert one diagnostic observation to Direct's flattened 5-channel map."""
    x0, x1 = bev.x_range_m
    y0, y1 = bev.y_range_m
    x = np.linspace(
        x0 + 0.5 * (x1 - x0) / bev.width,
        x1 - 0.5 * (x1 - x0) / bev.width,
        bev.width,
        dtype=np.float32,
    )
    y = np.linspace(
        y1 - 0.5 * (y1 - y0) / bev.height,
        y0 + 0.5 * (y1 - y0) / bev.height,
        bev.height,
        dtype=np.float32,
    )
    yy, xx = np.meshgrid(y, x, indexing="ij")
    xy = np.stack(
        (
            (xx - x0) / (x1 - x0) * 2.0 - 1.0,
            (yy - y0) / (y1 - y0) * 2.0 - 1.0,
        ),
        axis=-1,
    )
    output = np.concatenate(
        (
            xy,
            observation.bev_z[..., None],
            observation.support[..., None],
            observation.confidence[..., None],
        ),
        axis=-1,
    ).astype(np.float32)
    flat = output.reshape(-1)
    if flat.shape != (DELTA_MAP_DIM,) or not np.all(np.isfinite(flat)):
        raise FloatingPointError(f"Direct camera map must be finite {DELTA_MAP_DIM}D")
    return flat


if DELTA_MAP_DIM != MAP_HEIGHT * MAP_WIDTH * MAP_CHANNELS:
    raise RuntimeError("Direct map contract is inconsistent")
if ACTION_DIM != 12:
    raise RuntimeError("Go2 sim2sim expects 12 actions")
