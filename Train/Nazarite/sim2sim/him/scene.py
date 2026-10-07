"""HIM obstacle course: solid stairs and a low wall."""

from __future__ import annotations

import mujoco

from ..scene import _add_box, add_simple_grid_scene


def add_him_scene(spec: mujoco.MjSpec) -> None:
    # The ground below the floating treads is lower than the walking track.
    add_simple_grid_scene(spec, half_extent=18.0, floor_z=-0.7)
    width = 1.4
    half_width = width / 2.0
    slab = (0.36, 0.40, 0.42, 1.0)
    solid = (0.55, 0.37, 0.21, 1.0)

    def flat(name: str, start: float, end: float) -> None:
        _add_box(spec, name, ((start + end) / 2.0, 0.0, -0.06),
                 ((end - start) / 2.0, half_width, 0.06), slab)

    flat("him_start", -3.0, 1.4)
    step_depth = 0.30
    rise = 0.06
    for i in range(5):
        height = (i + 1) * rise
        _add_box(spec, f"him_solid_up_{i + 1}",
                 (1.4 + (i + 0.5) * step_depth, 0.0, height / 2.0),
                 (step_depth / 2.0, half_width, height / 2.0), solid)
    _add_box(spec, "him_solid_platform", (3.30, 0.0, 0.15),
             (0.40, half_width, 0.15), solid)
    for i in range(5):
        height = (5 - i) * rise
        _add_box(spec, f"him_solid_down_{i + 1}",
                 (3.7 + (i + 0.5) * step_depth, 0.0, height / 2.0),
                 (step_depth / 2.0, half_width, height / 2.0), solid)

    flat("him_finish", 5.2, 15.0)

    # x: thickness 100 mm; y: width 800 mm; z: height 320 mm.
    _add_box(spec, "him_wall_320x800x100", (12.3, 0.0, 0.16),
             (0.05, 0.40, 0.16), (0.65, 0.23, 0.22, 1.0))
