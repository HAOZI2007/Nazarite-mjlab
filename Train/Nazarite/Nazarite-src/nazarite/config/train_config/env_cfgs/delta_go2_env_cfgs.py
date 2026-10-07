"""Shared DELTA terrain and camera configuration.

The standalone DELTA task was removed. These definitions are kept because the
WTW+DELTA Direct task uses the same terrain primitives and D435 camera contract.
"""

import math

from mjlab.sensor import CameraSensorCfg
from mjlab.terrains import TerrainGeneratorCfg
from mjlab.terrains.config import (
    box_random_grid,
    hf_pyramid_slope,
    nested_rings,
    open_stairs,
    pyramid_stairs,
    random_rough,
    wave_terrain,
)
from nazarite.delta.d435 import (
    D435_DEPTH_848X480,
    DELTA_D435_INTRINSICS,
    DELTA_DEPTH_HEIGHT,
    DELTA_DEPTH_WIDTH,
    PinholeIntrinsics,
    ZED_DEPTH_FOV_HORIZONTAL_DEG,
    ZED_DEPTH_FOV_VERTICAL_DEG,
    ZED_DEPTH_HEIGHT,
    ZED_DEPTH_WIDTH,
    pinhole_from_fov,
)
from nazarite.delta.terrains import BoxApproachSteppingStonesTerrainCfg


DELTA_TERRAINS_CFG = TerrainGeneratorCfg(
    # 4 m tiles give the 10 s episode enough opportunities to cross several
    # terrain types while keeping the terrain inside the D435/DELTA local map.
    size=(4.0, 4.0),
    border_width=20.0,
    num_rows=10,
    num_cols=8,
    curriculum=True,
    difficulty_range=(0.0, 1.0),
    sub_terrains={
        "hills": wave_terrain(
            proportion=0.125, amplitude_range=(0.0, 0.16), num_waves=3
        ),
        "slope": hf_pyramid_slope(proportion=0.125, slope_range=(0.0, 0.30)),
        "stairs": pyramid_stairs(proportion=0.125, step_height_range=(0.015, 0.16)),
        "steps": open_stairs(
            proportion=0.125,
            step_height_range=(0.02, 0.16),
            step_width_range=(0.45, 0.85),
        ),
        "rough_ground": random_rough(
            proportion=0.125,
            noise_range=(0.01, 0.12),
            noise_step=0.02,
            scale_with_difficulty=True,
        ),
        "gaps": nested_rings(
            proportion=0.125,
            num_rings=6,
            ring_width_range=(0.45, 0.75),
            gap_range=(0.08, 0.30),
            height_range=(0.05, 0.25),
        ),
        "stepping_stones": BoxApproachSteppingStonesTerrainCfg(
            proportion=0.125,
            stone_size_range=(0.55, 0.95),
            stone_distance_range=(0.08, 0.30),
            stone_height=0.12,
            stone_height_variation=0.05,
            stone_size_variation=0.10,
            displacement_range=0.06,
            approach_platform_length=1.20,
            approach_platform_width=1.40,
            approach_gap=0.25,
            platform_width=0.0,
        ),
        "grid_stones": box_random_grid(
            proportion=0.125,
            grid_width=0.45,
            grid_height_range=(0.0, 0.18),
            platform_width=0.0,
            fill_gaps=True,
            holes=False,
        ),
    },
    add_lights=True,
)


DELTA_DEPTH_CAMERA = CameraSensorCfg(
    name="delta_depth_camera",
    parent_body="robot/base_link",
    pos=(0.30, 0.0, 0.12),
    # D435 optical axis forward, pitched 20 degrees down in the robot frame.
    quat=(0.579228, 0.405580, -0.405580, -0.579228),
    width=DELTA_DEPTH_WIDTH,
    height=DELTA_DEPTH_HEIGHT,
    focal_length_px=(DELTA_D435_INTRINSICS.fx, DELTA_D435_INTRINSICS.fy),
    principal_point_px=(DELTA_D435_INTRINSICS.cx, DELTA_D435_INTRINSICS.cy),
    data_types=("depth",),
    enabled_geom_groups=(0,),
    use_shadows=False,
    use_textures=False,
)


def _quat_mul(a: tuple[float, float, float, float], b: tuple[float, float, float, float]):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _axis_angle_quat(axis: tuple[float, float, float], degrees: float):
    half = math.radians(float(degrees)) * 0.5
    scale = math.sin(half)
    return (math.cos(half), axis[0] * scale, axis[1] * scale, axis[2] * scale)


def make_delta_depth_camera(
    orientation: str = "horizontal",
    pitch_deg: float = 20.0,
    width: int = ZED_DEPTH_WIDTH,
    height: int = ZED_DEPTH_HEIGHT,
    pos: tuple[float, float, float] = (0.38, 0.0, 0.10),
) -> CameraSensorCfg:
    """Create the front-torso depth camera in horizontal or portrait layout.

    ``pitch_deg`` is positive downward in the projection convention. Portrait
    mode rotates the optical image plane by +90 degrees and swaps the rendered
    resolution to ``height x width`` while preserving the configured FOV.

    The attention task uses the ZED depth FOV (58.4 x 45.5 degrees). The
    legacy D435 camera constants above remain available to old tasks.
    """
    orientation = str(orientation).lower()
    if orientation not in ("horizontal", "vertical", "portrait"):
        raise ValueError("orientation must be 'horizontal' or 'vertical'")
    portrait = orientation in ("vertical", "portrait")
    render_width, render_height = (height, width) if portrait else (width, height)
    if portrait:
        intrinsics = pinhole_from_fov(
            render_width,
            render_height,
            ZED_DEPTH_FOV_VERTICAL_DEG,
            ZED_DEPTH_FOV_HORIZONTAL_DEG,
        )
        roll_deg = 90.0
    else:
        intrinsics = pinhole_from_fov(
            render_width,
            render_height,
            ZED_DEPTH_FOV_HORIZONTAL_DEG,
            ZED_DEPTH_FOV_VERTICAL_DEG,
        )
        roll_deg = 0.0
    base_quat = (-0.5, -0.5, 0.5, 0.5)
    pitch_quat = _axis_angle_quat((0.0, 1.0, 0.0), pitch_deg)
    pitched_quat = _quat_mul(pitch_quat, base_quat)
    # Roll around the actual optical axis, matching sim2sim's camera model.
    rotation = [0.0] * 9
    qw, qx, qy, qz = pitched_quat
    rotation[:] = [
        1.0 - 2.0 * (qy * qy + qz * qz),
        2.0 * (qx * qy - qz * qw),
        2.0 * (qx * qz + qy * qw),
        2.0 * (qx * qy + qz * qw),
        1.0 - 2.0 * (qx * qx + qz * qz),
        2.0 * (qy * qz - qx * qw),
        2.0 * (qx * qz - qy * qw),
        2.0 * (qy * qz + qx * qw),
        1.0 - 2.0 * (qx * qx + qy * qy),
    ]
    optical_axis = (-rotation[2], -rotation[5], -rotation[8])
    roll_quat = _axis_angle_quat(optical_axis, roll_deg)
    quat = _quat_mul(roll_quat, pitched_quat)
    return CameraSensorCfg(
        name="delta_depth_camera",
        parent_body="robot/base_link",
        pos=pos,
        quat=quat,
        width=render_width,
        height=render_height,
        focal_length_px=(intrinsics.fx, intrinsics.fy),
        principal_point_px=(intrinsics.cx, intrinsics.cy),
        data_types=("depth",),
        enabled_geom_groups=(0,),
        use_shadows=False,
        use_textures=False,
    )


DELTA_DEPTH_CAMERA_HORIZONTAL_160X120 = make_delta_depth_camera()
DELTA_DEPTH_CAMERA_VERTICAL_160X120 = make_delta_depth_camera("vertical")
