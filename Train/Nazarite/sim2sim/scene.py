"""Common MuJoCo scene helpers shared by every deployment policy."""

from __future__ import annotations

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
