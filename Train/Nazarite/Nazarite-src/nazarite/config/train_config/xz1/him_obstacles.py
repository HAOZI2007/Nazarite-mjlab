"""XZ1 courses with Go2 WMP's start/obstacle/landing layout.

Use primitive collision boxes, including the segmented hollow tire rings from
Go2 WMP. Difficulty uses mjlab's native [0, 1] rows, without WMP's /0.9 remap.
Course profiles are shared by geometry, commands, curriculum and evaluation.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import (
    SubTerrainCfg,
    TerrainGeometry,
    TerrainOutput,
)
from nazarite.config.train_config.go2.wmp_terrain import _tire_ring
from nazarite.config.train_config.xz1.him_terrain import _floor, _interpolate

OBSTACLE_TERRAIN_NAMES = (
    "plum_blossom",
    "single_beam",
    "low_frame",
    "tire_array",
    "gap",
)
XZ1_TERRAIN_NAMES = (
    "flat",
    "high_wall",
    "narrow_stairs",
    "narrow_ramp",
    *OBSTACLE_TERRAIN_NAMES,
)
_DECK = (0.65, 0.52, 0.30, 1.0)
_GROUND = (0.28, 0.28, 0.30, 1.0)
GAP_BOTTOM_PREFIX = "xz1_gap_bottom_"


def lerp(difficulty, bounds):
    """Interpolation also accepts tensors for the vectorized runtime profile."""
    return bounds[0] + difficulty * (bounds[1] - bounds[0])


def _box(body, x0, x1, y0, y1, bottom, top, color=_DECK):
    if min(x1 - x0, y1 - y0, top - bottom) <= 0:
        raise ValueError("Obstacle box dimensions must be positive")
    geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=((x0 + x1) / 2, (y0 + y1) / 2, (bottom + top) / 2),
        size=((x1 - x0) / 2, (y1 - y0) / 2, (top - bottom) / 2),
        friction=(0.8, 0.1, 0.1),
    )
    return TerrainGeometry(geom=geom, color=color)


def _finish(cfg, geoms, height=0.0):
    start, end, width = cfg.course_profile(1.0)
    if not 0 < cfg.size[0] * cfg.spawn_x_fraction < start < end < cfg.size[0] - 0.8:
        raise ValueError("Course needs a safe spawn platform and a far landing")
    if not 0 < width <= cfg.size[1]:
        raise ValueError("Course width must fit the tile")
    return TerrainOutput(
        origin=np.array(
            [
                cfg.size[0] * cfg.spawn_x_fraction,
                cfg.size[1] / 2,
                height,
            ]
        ),
        geometries=geoms,
    )


@dataclass(kw_only=True)
class XZ1PlumBlossomTerrainCfg(SubTerrainCfg):
    """Six staggered stones; both launch and finish share their top height."""

    spawn_x_fraction: float = 0.125
    start_x: float = 1.4
    stone_size_range: tuple[float, float] = (0.50, 0.20)
    height_range: tuple[float, float] = (0.05, 0.20)
    row_spacing: float = 0.40
    stone_pitch: float = 0.60
    row_stagger: float = 0.30
    platform_length: float = 0.80
    width: float = 0.80

    def stone_centers(self):
        return tuple(
            (
                self.start_x + 0.20 + i * self.stone_pitch + row * self.row_stagger,
                (0.5 - row) * self.row_spacing,
            )
            for row in range(2)
            for i in range(3)
        )

    def course_profile(self, difficulty):
        del difficulty
        end = self.start_x + 0.4 + 2 * self.stone_pitch + self.row_stagger
        return self.start_x, end, self.width

    def function(self, difficulty, spec, rng):
        del rng
        body = spec.body("terrain")
        height = _interpolate(difficulty, self.height_range)
        side = _interpolate(difficulty, self.stone_size_range)
        start, end, width = self.course_profile(difficulty)
        y = self.size[1] / 2
        geoms = [TerrainGeometry(geom=_floor(body, self.size), color=_GROUND)]
        for x0, x1 in (
            (start - self.platform_length, start),
            (end, end + self.platform_length),
        ):
            geoms.append(_box(body, x0, x1, y - width / 2, y + width / 2, 0, height))
        for x, offset in self.stone_centers():
            geoms.append(
                _box(
                    body,
                    x - side / 2,
                    x + side / 2,
                    y + offset - side / 2,
                    y + offset + side / 2,
                    0,
                    height,
                )
            )
        return _finish(self, geoms, height)


@dataclass(kw_only=True)
class XZ1SingleBeamTerrainCfg(SubTerrainCfg):
    """One continuous bridge, with aligned launch and landing platforms."""

    spawn_x_fraction: float = 0.125
    start_x: float = 1.6
    length: float = 3.0
    width_range: tuple[float, float] = (0.80, 0.40)
    height_range: tuple[float, float] = (0.05, 0.10)
    platform_width: float = 0.80

    def course_profile(self, difficulty):
        return (
            self.start_x,
            self.start_x + self.length,
            lerp(difficulty, self.width_range),
        )

    def function(self, difficulty, spec, rng):
        del rng
        body = spec.body("terrain")
        start, end, width = self.course_profile(difficulty)
        height = _interpolate(difficulty, self.height_range)
        y = self.size[1] / 2
        geoms = [TerrainGeometry(geom=_floor(body, self.size), color=_GROUND)]
        for x0, x1 in ((0, start), (end, self.size[0])):
            geoms.append(
                _box(
                    body,
                    x0,
                    x1,
                    y - self.platform_width / 2,
                    y + self.platform_width / 2,
                    0,
                    height,
                )
            )
        geoms.append(_box(body, start, end, y - width / 2, y + width / 2, 0, height))
        return _finish(self, geoms, height)


@dataclass(kw_only=True)
class XZ1LowFrameTerrainCfg(SubTerrainCfg):
    """An 0.8 m square overhead frame; internal clearance decreases by row."""

    spawn_x_fraction: float = 0.125
    start_x: float = 2.0
    length: float = 0.80
    width: float = 0.80
    clearance_range: tuple[float, float] = (0.60, 0.29)
    beam_thickness: float = 0.04

    def course_profile(self, difficulty):
        del difficulty
        return self.start_x, self.start_x + self.length, self.width

    def function(self, difficulty, spec, rng):
        del rng
        body = spec.body("terrain")
        start, end, width = self.course_profile(difficulty)
        height = _interpolate(difficulty, self.clearance_range)
        y = self.size[1] / 2
        t = self.beam_thickness
        geoms = [TerrainGeometry(geom=_floor(body, self.size), color=_GROUND)]
        # Posts outside the internal opening; four roof rails, no solid ceiling.
        for x in (start, end):
            for side in (-1, 1):
                cy = y + side * (width + t) / 2
                geoms.append(
                    _box(body, x - t / 2, x + t / 2, cy - t / 2, cy + t / 2, 0, height)
                )
            geoms.append(
                _box(
                    body,
                    x - t / 2,
                    x + t / 2,
                    y - width / 2 - t,
                    y + width / 2 + t,
                    height,
                    height + t,
                )
            )
        for side in (-1, 1):
            cy = y + side * (width + t) / 2
            geoms.append(
                _box(body, start, end, cy - t / 2, cy + t / 2, height, height + t)
            )
        return _finish(self, geoms)


@dataclass(kw_only=True)
class XZ1TireArrayTerrainCfg(SubTerrainCfg):
    """Two by four hollow rings, using the Go2 WMP collision construction."""

    spawn_x_fraction: float = 0.125
    start_x: float = 1.8
    outer_diameter: float = 0.55
    inner_diameter: float = 0.33
    height_range: tuple[float, float] = (0.03, 0.165)
    pitch: float = 0.56
    segments: int = 12

    def course_profile(self, difficulty):
        del difficulty
        return (
            self.start_x,
            self.start_x + 3 * self.pitch + self.outer_diameter,
            self.pitch + self.outer_diameter,
        )

    def function(self, difficulty, spec, rng):
        del rng
        body = spec.body("terrain")
        height = _interpolate(difficulty, self.height_range)
        geoms = [TerrainGeometry(geom=_floor(body, self.size), color=_GROUND)]
        for i in range(4):
            for row in (-0.5, 0.5):
                geoms.extend(
                    _tire_ring(
                        body,
                        (
                            self.start_x + self.outer_diameter / 2 + i * self.pitch,
                            self.size[1] / 2 + row * self.pitch,
                            height / 2,
                        ),
                        self.outer_diameter,
                        self.inner_diameter,
                        height,
                        self.segments,
                    )
                )
        return _finish(self, geoms)


@dataclass(kw_only=True)
class XZ1GapTerrainCfg(SubTerrainCfg):
    """Repeated full-width gaps with explicit intermediate landing decks."""

    spawn_x_fraction: float = 0.125
    start_x: float = 1.8
    gap_width_range: tuple[float, float] = (0.02, 0.36)
    pit_depth_range: tuple[float, float] = (0.15, 0.36)
    num_gaps: int = 3
    landing_length: float = 0.80

    def course_profile(self, difficulty):
        gap = lerp(difficulty, self.gap_width_range)
        end = (
            self.start_x
            + self.num_gaps * gap
            + (self.num_gaps - 1) * self.landing_length
        )
        return self.start_x, end, self.size[1]

    def function(self, difficulty, spec, rng):
        del rng
        body = spec.body("terrain")
        gap = _interpolate(difficulty, self.gap_width_range)
        depth = _interpolate(difficulty, self.pit_depth_range)
        start, end, _ = self.course_profile(difficulty)
        geoms = [
            _box(body, 0, self.size[0], 0, self.size[1], -depth - 0.1, -depth, _GROUND)
        ]
        # Keep every bottom identifiable across the whole curriculum grid.
        # A vertical landing face must never be mistaken for the pit floor.
        bottom = geoms[0].geom
        assert bottom is not None
        bottom.name = f"{GAP_BOTTOM_PREFIX}{len(body.geoms)}"
        cursor = 0
        for index in range(self.num_gaps):
            edge = start + index * (gap + self.landing_length)
            geoms.append(_box(body, cursor, edge, 0, self.size[1], -depth, 0))
            cursor = edge + gap
        geoms.append(_box(body, end, self.size[0], 0, self.size[1], -depth, 0))
        return _finish(self, geoms)
