"""Common MuJoCo scene helpers shared by every deployment policy."""

from __future__ import annotations

import math

import mujoco


def add_simple_grid_scene(
    spec: mujoco.MjSpec,
    *,
    half_extent: float = 10.0,
    spacing: float = 0.5,
) -> None:
    """Add an infinite flat floor with a non-colliding square grid overlay."""
    if half_extent <= 0.0:
        raise ValueError("half_extent must be positive")
    if spacing <= 0.0:
        raise ValueError("spacing must be positive")

    floor = spec.worldbody.add_geom()
    floor.name = "sim2sim_floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size[:] = [half_extent, half_extent, 0.1]
    floor.rgba[:] = [0.13, 0.14, 0.16, 1.0]

    line_count = int(half_extent / spacing)
    line_width = 0.006
    line_height = 0.001
    minor_color = [0.28, 0.30, 0.33, 1.0]
    major_color = [0.43, 0.46, 0.50, 1.0]

    for index in range(-line_count, line_count + 1):
        coordinate = index * spacing
        color = major_color if index % 2 == 0 else minor_color

        x_line = spec.worldbody.add_site()
        x_line.name = f"grid_x_{index + line_count:02d}"
        x_line.type = mujoco.mjtGeom.mjGEOM_BOX
        x_line.pos[:] = [0.0, coordinate, line_height]
        x_line.size[:] = [half_extent, line_width, line_height]
        x_line.rgba[:] = color
        x_line.group = 1

        y_line = spec.worldbody.add_site()
        y_line.name = f"grid_y_{index + line_count:02d}"
        y_line.type = mujoco.mjtGeom.mjGEOM_BOX
        y_line.pos[:] = [coordinate, 0.0, line_height]
        y_line.size[:] = [line_width, half_extent, line_height]
        y_line.rgba[:] = color
        y_line.group = 1

    # Highlight the world axes without creating additional collision geometry.
    x_axis = spec.site(f"grid_x_{line_count:02d}")
    y_axis = spec.site(f"grid_y_{line_count:02d}")
    if x_axis is not None:
        x_axis.rgba[:] = [0.80, 0.22, 0.20, 1.0]
    if y_axis is not None:
        y_axis.rgba[:] = [0.20, 0.65, 0.25, 1.0]


def _add_box(
    spec: mujoco.MjSpec,
    name: str,
    position: tuple[float, float, float],
    half_size: tuple[float, float, float],
    rgba: tuple[float, float, float, float],
    *,
    angle: float = 0.0,
) -> None:
    geom = spec.worldbody.add_geom()
    geom.name = name
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.pos[:] = position
    geom.size[:] = half_size
    geom.rgba[:] = rgba
    geom.friction[:] = [0.8, 0.02, 0.01]
    if angle:
        geom.quat[:] = [math.cos(angle / 2.0), 0.0, math.sin(angle / 2.0), 0.0]


def add_stairs_and_slopes_scene(spec: mujoco.MjSpec) -> None:
    """Add a small obstacle course after the common grid floor.

    The robot starts at x=0.  Along +x the course contains a complete stair
    group (up stairs, high platform, down stairs), a flat separation, and a
    complete ramp group (up ramp, high platform, down ramp). All objects are
    ordinary collidable geoms.
    """
    add_simple_grid_scene(spec, half_extent=14.0)

    stair_color = (0.52, 0.38, 0.20, 1.0)
    ramp_color = (0.24, 0.45, 0.62, 1.0)
    stair_count = 5
    stair_length = 0.28
    stair_height = 0.06
    stair_width = 1.4

    # Stair group: ascending stairs, a high platform, then descending stairs.
    up_start = 1.4
    for index in range(stair_count):
        height = stair_height * (index + 1)
        _add_box(
            spec,
            f"stairs_up_{index + 1}",
            (
                up_start + stair_length * (index + 0.5),
                0.0,
                height / 2.0,
            ),
            (stair_length / 2.0, stair_width / 2.0, height / 2.0),
            stair_color,
        )

    platform_height = stair_height * stair_count
    _add_box(
        spec,
        "stairs_platform",
        (3.35, 0.0, platform_height / 2.0),
        (0.42, stair_width / 2.0, platform_height / 2.0),
        stair_color,
    )

    down_start = 4.25
    for index in range(stair_count):
        height = stair_height * (stair_count - index)
        _add_box(
            spec,
            f"stairs_down_{index + 1}",
            (
                down_start + stair_length * (index + 0.5),
                0.0,
                height / 2.0,
            ),
            (stair_length / 2.0, stair_width / 2.0, height / 2.0),
            stair_color,
        )

    def add_ramp(
        name: str,
        start_x: float,
        length: float,
        rise: float,
    ) -> None:
        angle = -math.atan2(rise, length)
        thickness = 0.08
        half_length = math.hypot(length, rise) / 2.0
        half_width = 1.4 / 2.0
        # Keep the low end nearly flush with the floor while preserving a
        # thin collidable slab under the ramp surface.
        center_z = abs(rise) / 2.0 + thickness * (
            1.0 - math.cos(angle)
        ) / 2.0
        _add_box(
            spec,
            name,
            (start_x + length / 2.0, 0.0, center_z),
            (half_length, half_width, thickness / 2.0),
            ramp_color,
            angle=angle,
        )

    # Ramp group: it is separated from the stair group by a flat section.
    ramp_height = 0.35 + 0.08
    add_ramp("ramp_up", 8.0, 1.8, 0.35)
    _add_box(
        spec,
        "ramp_platform",
        (10.15, 0.0, ramp_height / 2.0),
        (0.35, 1.4 / 2.0, ramp_height / 2.0),
        ramp_color,
    )
    add_ramp("ramp_down", 10.5, 1.8, -0.35)
