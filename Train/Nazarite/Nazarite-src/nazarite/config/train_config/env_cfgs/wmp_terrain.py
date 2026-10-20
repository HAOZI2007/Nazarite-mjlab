"""Dense, progressive obstacle-skill terrains for Go2 WMP training.

Each patch keeps the reference repository's 8 x 8 m curriculum-grid layout,
but repeats or expands obstacles to fill the useful training area. Exact
ROBOCON dimensions belong in a separate evaluation course, not this sampler.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import (
    SubTerrainCfg,
    TerrainGeneratorCfg,
    TerrainGeometry,
    TerrainOutput,
)
from mjlab.terrains.utils import make_plane

WMP_TERRAIN_NAMES = (
    "stairs",
    "low_frame",
    "high_wall",
    "plum_blossom",
    "gap",
    "tire_array",
    "single_beam",
    "ramp",
)
# Every WMP tile is a forward course: it has a start deck, an obstacle field and
# a finish deck. Stairs and ramp used to be excluded here, which left them on the
# omni-directional command sampler (lin_vel_y != 0, yaw != 0). Their expected +x
# displacement over an episode was therefore near zero while the curriculum
# promotion rule requires 4 m of +x progress, so those two columns could never
# level up no matter how good the policy was.
WMP_FORWARD_TERRAIN_NAMES = (
    "stairs",
    "low_frame",
    "high_wall",
    "plum_blossom",
    "gap",
    "tire_array",
    "single_beam",
    "ramp",
)

# Number of curriculum rows. The source uses ten rows of an 8 m tile, which is
# too coarse for the deep-pit tiles: on the gap tile a row step moves the trench
# width by 0.042 m, i.e. one step can jump from "trivially crossed" to "falls in".
# Twelve rows keeps the granularity at 0.048 m while the patch count only grows
# from 80 to 96 (10 x 8 -> 12 x 8). Twenty rows was tried first and was too
# expensive: it doubles the number of static terrain geoms (2313 -> 4623, and
# tire_array alone goes 1290 -> 2580), and every one of them is walked by the
# renderer's BVH refit on every strategy step.
# One flat column rides along with the obstacle course: an omni-directional and
# stand-still practice pad. It is deliberately not part of WMP_TERRAIN_NAMES --
# the curriculum, the forward-only command mask and the obstacle rewards all key
# off that tuple, and the pad must stay out of every one of them. Without it the
# terrain stage is 100% forward-only commands, so lin_vel_y and ang_vel_z never
# reach the actor and the policy cannot strafe or turn once deployed.
WMP_OMNI_TERRAIN_NAMES = ("omni_flat",)

WMP_NUM_ROWS = 12

_FLOOR = (0.42, 0.43, 0.44, 1.0)
_WOOD = (0.40, 0.25, 0.12, 1.0)
_WALL = (0.68, 0.25, 0.16, 1.0)
_BRICK = (0.48, 0.33, 0.22, 1.0)
_TIRE = (0.06, 0.07, 0.07, 1.0)
_STONE = (0.55, 0.53, 0.48, 1.0)
_REFERENCE_NUM_ROWS = 10
_REFERENCE_MAX_ROW_DIFFICULTY = (_REFERENCE_NUM_ROWS - 1) / _REFERENCE_NUM_ROWS


def _difficulty(value: float) -> float:
    # The source samples difficulty as row / num_rows, then its obstacle
    # generators normalize the final row to the full rule-size endpoint.
    return float(np.clip(value / _REFERENCE_MAX_ROW_DIFFICULTY, 0.0, 1.0))


def _lerp(value: float, low: float, high: float) -> float:
    return low + _difficulty(value) * (high - low)


def _box(
    body: mujoco.MjsBody,
    half_size: tuple[float, float, float],
    position: tuple[float, float, float],
    color: tuple[float, float, float, float],
    quat: tuple[float, float, float, float] | None = None,
) -> TerrainGeometry:
    geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=half_size,
        pos=position,
    )
    geom.rgba = color
    if quat is not None:
        geom.quat = quat
    return TerrainGeometry(geom=geom, color=color)


def _floor(body: mujoco.MjsBody, size: tuple[float, float]) -> TerrainGeometry:
    geom = make_plane(body, size, 0.0, center_zero=False)[0]
    return TerrainGeometry(geom=geom, color=_FLOOR)


def _floor_region(
    body: mujoco.MjsBody,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    top: float,
    thickness: float = 0.10,
) -> TerrainGeometry:
    return _box(
        body,
        ((x1 - x0) / 2, (y1 - y0) / 2, thickness / 2),
        ((x0 + x1) / 2, (y0 + y1) / 2, top - thickness / 2),
        _FLOOR,
    )


def _origin(size: tuple[float, float]) -> np.ndarray:
    return np.asarray((1.0, size[1] / 2, 0.0), dtype=np.float64)


def _add_cylinder(
    body: mujoco.MjsBody,
    radius: float,
    height: float,
    position: tuple[float, float, float],
    color: tuple[float, float, float, float],
) -> TerrainGeometry:
    geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=(radius, height / 2, 0.0),
        pos=position,
    )
    geom.rgba = color
    return TerrainGeometry(geom=geom, color=color)


def _trapezoid_prism(
    spec: mujoco.MjSpec,
    body: mujoco.MjsBody,
    x_points: tuple[float, float, float, float],
    y_bounds: tuple[float, float],
    height: float,
    color: tuple[float, float, float, float],
    rng: np.random.Generator,
) -> TerrainGeometry:
    """Create a closed ascent-deck-descent prism with solid collision volume."""
    x0, x1, x2, x3 = x_points
    y0, y1 = y_bounds
    vertices = np.asarray(
        (
            (x0, y0, 0.0),
            (x0, y1, 0.0),
            (x1, y0, height),
            (x1, y1, height),
            (x2, y0, height),
            (x2, y1, height),
            (x3, y0, 0.0),
            (x3, y1, 0.0),
        ),
        dtype=np.float64,
    )
    # The four vertices on each side describe a convex trapezoid. The end
    # points meet the floor, so no zero-area vertical end faces are required.
    faces = (
        0,
        6,
        4,
        0,
        4,
        2,  # -y side
        1,
        3,
        5,
        1,
        5,
        7,  # +y side
        0,
        1,
        7,
        0,
        7,
        6,  # bottom
        0,
        2,
        3,
        0,
        3,
        1,  # ascent
        2,
        4,
        5,
        2,
        5,
        3,  # summit deck
        4,
        6,
        7,
        4,
        7,
        5,  # descent
    )
    mesh = spec.add_mesh(
        name=f"wmp_ramp_{rng.integers(int(1e9))}",
        uservert=vertices.ravel().tolist(),
        userface=list(faces),
    )
    geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=mesh.name,
    )
    geom.rgba = color
    return TerrainGeometry(geom=geom, color=color)


def _tire_ring(
    body: mujoco.MjsBody,
    center: tuple[float, float, float],
    outer_diameter: float,
    inner_diameter: float,
    width: float,
    num_segments: int,
) -> list[TerrainGeometry]:
    """Build a horizontal tire from tangent boxes so its center stays hollow."""
    if not 0.0 < inner_diameter < outer_diameter:
        raise ValueError("Tire diameters must satisfy 0 < inner < outer.")
    if width <= 0.0 or num_segments < 6:
        raise ValueError("Tire width must be positive and num_segments at least 6.")

    outer_radius = outer_diameter / 2
    inner_radius = inner_diameter / 2
    centerline_radius = (outer_radius + inner_radius) / 2
    radial_thickness = outer_radius - inner_radius
    tangent_length = 2 * centerline_radius * math.sin(math.pi / num_segments)
    center_x, center_y, center_z = center
    geometries = []
    for segment in range(num_segments):
        angle = 2 * math.pi * segment / num_segments
        segment_x = center_x + centerline_radius * math.cos(angle)
        segment_y = center_y + centerline_radius * math.sin(angle)
        geometries.append(
            _box(
                body,
                (tangent_length * 0.53, radial_thickness / 2, width / 2),
                (segment_x, segment_y, center_z),
                _TIRE,
                (
                    math.cos((angle + math.pi / 2) / 2),
                    0.0,
                    0.0,
                    math.sin((angle + math.pi / 2) / 2),
                ),
            )
        )
    return geometries


@dataclass(kw_only=True)
class WmpSlalomTerrainCfg(SubTerrainCfg):
    """L-shaped 100 mm slalom posts at the rule 900 mm spacing."""

    post_diameter_range: tuple[float, float] = (0.06, 0.10)
    post_height_range: tuple[float, float] = (0.05, 0.10)
    post_spacing_range: tuple[float, float] = (0.60, 0.90)

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        del rng
        body = spec.body("terrain")
        geoms = [_floor(body, self.size)]
        diameter = _lerp(difficulty, *self.post_diameter_range)
        height = _lerp(difficulty, *self.post_height_range)
        spacing = _lerp(difficulty, *self.post_spacing_range)
        cx, cy = 4.8, self.size[1] / 2
        # Three posts form an L, preserving the event's turn-around geometry.
        positions = (
            (cx - spacing, cy - spacing / 2),
            (cx, cy - spacing / 2),
            (cx, cy + spacing / 2),
        )
        for x, y in positions:
            geoms.append(
                _add_cylinder(body, diameter / 2, height, (x, y, height / 2), _WOOD)
            )
        return TerrainOutput(origin=_origin(self.size), geometries=geoms)


@dataclass(kw_only=True)
class WmpLowFrameTerrainCfg(SubTerrainCfg):
    """Repeated full-width low-clearance frames with explicit flat approaches."""

    opening_height_range: tuple[float, float] = (0.60, 0.42)
    beam_thickness: float = 0.10
    obstacle_positions: tuple[float, ...] = (2.75, 4.65, 6.55)

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        del rng
        body = spec.body("terrain")
        geoms = [_floor(body, self.size)]
        height = _lerp(difficulty, *self.opening_height_range)
        beam_half_width = self.size[1] / 2
        post = 0.10
        for cx in self.obstacle_positions:
            post_z = (height + self.beam_thickness) / 2
            for y in (post / 2, self.size[1] - post / 2):
                geoms.append(
                    _box(
                        body,
                        (post / 2, post / 2, post_z),
                        (cx, y, post_z),
                        _WOOD,
                    )
                )
            geoms.append(
                _box(
                    body,
                    (
                        self.beam_thickness / 2,
                        beam_half_width,
                        self.beam_thickness / 2,
                    ),
                    (cx, self.size[1] / 2, height + self.beam_thickness / 2),
                    _WOOD,
                )
            )
        return TerrainOutput(origin=_origin(self.size), geometries=geoms)


@dataclass(kw_only=True)
class WmpStairsTerrainCfg(SubTerrainCfg):
    """One continuous ascent-platform-descent staircase across the tile."""

    step_height_range: tuple[float, float] = (0.04, 0.18)
    step_depth_range: tuple[float, float] = (0.55, 0.36)
    step_count: int = 4
    start_platform_length: float = 1.55
    top_platform_length: float = 0.70
    lane_margin: float = 0.0

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        del rng
        body = spec.body("terrain")
        geoms = [_floor(body, self.size)]
        step_height = _lerp(difficulty, *self.step_height_range)
        step_depth = _lerp(difficulty, *self.step_depth_range)
        cy = self.size[1] / 2
        half_width = self.size[1] / 2 - self.lane_margin
        x = self.start_platform_length
        for index in range(self.step_count):
            height = step_height * (index + 1)
            geoms.append(
                _box(
                    body,
                    (step_depth / 2, half_width, height / 2),
                    (x + step_depth / 2, cy, height / 2),
                    _STONE,
                )
            )
            x += step_depth

        top_height = step_height * self.step_count
        geoms.append(
            _floor_region(
                body,
                x,
                x + self.top_platform_length,
                self.lane_margin,
                self.size[1] - self.lane_margin,
                top_height,
            )
        )
        x += self.top_platform_length
        for index in range(self.step_count):
            height = step_height * (self.step_count - index - 1)
            if height > 0.0:
                geoms.append(
                    _box(
                        body,
                        (step_depth / 2, half_width, height / 2),
                        (x + step_depth / 2, cy, height / 2),
                        _STONE,
                    )
                )
            x += step_depth
        # Start and finish platforms are explicit and flush with the floor.
        geoms.append(
            _floor_region(
                body,
                0.0,
                self.start_platform_length,
                0.0,
                self.size[1],
                0.0,
            )
        )
        geoms.append(_floor_region(body, x, self.size[0], 0.0, self.size[1], 0.0))
        return TerrainOutput(origin=_origin(self.size), geometries=geoms)


@dataclass(kw_only=True)
class WmpHighWallTerrainCfg(SubTerrainCfg):
    """Repeated full-width step walls with clear approach and landing zones."""

    wall_height_range: tuple[float, float] = (0.10, 0.42)
    wall_width: float | None = None
    wall_thickness: float = 0.14
    obstacle_positions: tuple[float, ...] = (2.75, 4.65, 6.55)

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        del rng
        body = spec.body("terrain")
        geoms = [_floor(body, self.size)]
        height = _lerp(difficulty, *self.wall_height_range)
        wall_width = self.wall_width or self.size[1]
        for obstacle_x in self.obstacle_positions:
            geoms.append(
                _box(
                    body,
                    (self.wall_thickness / 2, wall_width / 2, height / 2),
                    (obstacle_x, self.size[1] / 2, height / 2),
                    _WALL,
                )
            )
        return TerrainOutput(origin=_origin(self.size), geometries=geoms)


@dataclass(kw_only=True)
class WmpGapTerrainCfg(SubTerrainCfg):
    """Several full-width jump trenches with stable takeoff/landing decks.

    The reference gap starts at 0.10 m of width over a 0.5-0.8 m pit, so the
    easiest row is already crossable by simply walking. The port started at
    0.20 m over a 1.20 m pit, which is simultaneously too wide to step over and
    so deep that any misstep is an unrecoverable base contact: no curriculum
    level ever advanced on this tile. Both ends are re-anchored here.
    """

    gap_width_range: tuple[float, float] = (0.02, 0.55)
    gap_count: int = 3
    # Long approach deck: the robot spawns at tile x = 1.0 +/- 0.5 and starts
    # from rest, so the first trench must sit far enough away to build speed.
    start_platform_length: float = 2.60
    landing_length: float = 0.45
    # A shallow pit keeps a failed step recoverable instead of terminal.
    pit_depth_range: tuple[float, float] = (0.35, 0.60)

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        del rng
        body = spec.body("terrain")
        geoms = []
        gap = _lerp(difficulty, *self.gap_width_range)
        pit_depth = _lerp(difficulty, *self.pit_depth_range)
        x = self.start_platform_length
        for _ in range(self.gap_count):
            geoms.append(_floor_region(body, x, x + gap, 0.0, self.size[1], -pit_depth))
            x += gap
            geoms.append(
                _floor_region(
                    body,
                    x,
                    x + self.landing_length,
                    0.0,
                    self.size[1],
                    0.0,
                )
            )
            x += self.landing_length
        geoms.insert(
            0,
            _floor_region(
                body, 0.0, self.start_platform_length, 0.0, self.size[1], 0.0
            ),
        )
        geoms.append(_floor_region(body, x, self.size[0], 0.0, self.size[1], 0.0))
        return TerrainOutput(origin=_origin(self.size), geometries=geoms)


@dataclass(kw_only=True)
class WmpRampTerrainCfg(SubTerrainCfg):
    """Solid trapezoidal ramp with ascent, summit deck and descent."""

    angle_range: tuple[float, float] = (8.0, 20.0)
    length_range: tuple[float, float] = (1.5, 1.9)
    width: float | None = None
    start_x: float = 1.65
    summit_length: float = 0.85

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        body = spec.body("terrain")
        geoms = [_floor(body, self.size)]
        cy = self.size[1] / 2
        angle = math.radians(_lerp(difficulty, *self.angle_range))
        length = _lerp(difficulty, *self.length_range)
        width = self.width or self.size[1]
        rise = length * math.sin(angle)
        geoms.append(
            _trapezoid_prism(
                spec,
                body,
                (
                    self.start_x,
                    self.start_x + length,
                    self.start_x + length + self.summit_length,
                    self.start_x + 2 * length + self.summit_length,
                ),
                (cy - width / 2, cy + width / 2),
                rise,
                _WOOD,
                rng,
            )
        )
        return TerrainOutput(origin=_origin(self.size), geometries=geoms)


@dataclass(kw_only=True)
class WmpRubblePitTerrainCfg(SubTerrainCfg):
    """L-like brick-and-timber patch, with rule-height 150 mm low borders."""

    debris_count_range: tuple[int, int] = (8, 28)
    debris_size_range: tuple[float, float] = (0.12, 0.24)
    debris_height_range: tuple[float, float] = (0.025, 0.15)
    border_height_range: tuple[float, float] = (0.05, 0.15)

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        body = spec.body("terrain")
        lane_width = 2.0
        geoms = [_floor(body, self.size)]
        cy = self.size[1] / 2
        region_x0, region_x1 = 3.2, 5.2
        border_h = _lerp(difficulty, *self.border_height_range)
        # A short side return makes the boundary L-shaped; the random fill is
        # concentrated over the two-metre contest obstacle footprint.
        border_t = 0.05
        geoms.append(
            _box(
                body,
                (0.05, lane_width / 2, border_h / 2),
                (region_x0, cy, border_h / 2),
                _WOOD,
            )
        )
        geoms.append(
            _box(
                body,
                (0.45, border_t / 2, border_h / 2),
                (region_x1 - 0.45, cy + lane_width / 2 - border_t / 2, border_h / 2),
                _WOOD,
            )
        )
        count = round(_lerp(difficulty, *self.debris_count_range))
        min_size, max_size = self.debris_size_range
        min_height, max_height = self.debris_height_range
        for index in range(count):
            length = float(rng.uniform(min_size, max_size))
            width = float(rng.uniform(min_size * 0.65, max_size * 0.75))
            height = _lerp(difficulty, min_height, max_height) * float(
                rng.uniform(0.65, 1.0)
            )
            x = float(rng.uniform(region_x0 + 0.10, region_x1 - 0.10))
            y = float(
                rng.uniform(
                    cy - lane_width / 2 + width / 2,
                    cy + lane_width / 2 - width / 2,
                )
            )
            color = _BRICK if index % 2 == 0 else _WOOD
            yaw = float(rng.uniform(-math.pi / 4, math.pi / 4))
            geoms.append(
                _box(
                    body,
                    (length / 2, width / 2, height / 2),
                    (x, y, height / 2),
                    color,
                    (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)),
                )
            )
        return TerrainOutput(origin=_origin(self.size), geometries=geoms)


@dataclass(kw_only=True)
class WmpTireArrayTerrainCfg(SubTerrainCfg):
    """A moderate field of rule-sized hollow tire rings."""

    tire_outer_diameter: float = 0.55
    tire_inner_diameter: float = 0.33
    tire_width: float = 0.165
    pitch_range: tuple[float, float] = (1.15, 0.90)
    ring_segments: int = 8

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        del rng
        body = spec.body("terrain")
        geoms = [_floor(body, self.size)]
        difficulty = _difficulty(difficulty)
        pitch = _lerp(difficulty, *self.pitch_range)
        cy = self.size[1] / 2
        x_center = 4.20
        columns = 3 + round(difficulty)
        rows = 4 + round(difficulty)
        for row in range(rows):
            center_y = cy + (row - (rows - 1) / 2) * pitch
            for column in range(columns):
                center_x = x_center + (column - (columns - 1) / 2) * pitch
                geoms.extend(
                    _tire_ring(
                        body=body,
                        center=(center_x, center_y, self.tire_width / 2),
                        outer_diameter=self.tire_outer_diameter,
                        inner_diameter=self.tire_inner_diameter,
                        width=self.tire_width,
                        num_segments=self.ring_segments,
                    )
                )
        return TerrainOutput(origin=_origin(self.size), geometries=geoms)


@dataclass(kw_only=True)
class WmpPlumBlossomTerrainCfg(SubTerrainCfg):
    """Dense alternating stepping-stone matrix over a recessed pit."""

    # The easiest row must be almost a floor: with 0.62 m stones and 0.72/0.70 m
    # pitch the gaps are 0.10 m (x) and 0.08 m (y), so row 0 is walkable and the
    # curriculum has somewhere to start. The hardest row (0.50 m stones on
    # 0.82/0.80 m pitch) is a genuine 0.30 m stepping-stone course.
    block_size_range: tuple[float, float] = (0.62, 0.50)
    x_pitch_range: tuple[float, float] = (0.72, 0.82)
    y_pitch_range: tuple[float, float] = (0.70, 0.80)
    # Keep every stone's top flush with the start/finish platforms (z = 0).
    # The pit remains recessed, so stepping stones still provide support over
    # the gap without becoming raised obstacles themselves.
    block_height_range: tuple[float, float] = (0.0, 0.0)
    start_platform_length: float = 1.65
    finish_platform_start: float = 6.45
    pit_depth_range: tuple[float, float] = (0.30, 0.70)

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        del rng
        body = spec.body("terrain")
        geoms = []
        block_size = _lerp(difficulty, *self.block_size_range)
        x_pitch = _lerp(difficulty, *self.x_pitch_range)
        y_pitch = _lerp(difficulty, *self.y_pitch_range)
        top_height = _lerp(difficulty, *self.block_height_range)
        pit_depth = _lerp(difficulty, *self.pit_depth_range)
        geoms.extend(
            (
                _floor_region(
                    body, 0.0, self.start_platform_length, 0.0, self.size[1], 0.0
                ),
                _floor_region(
                    body,
                    self.start_platform_length,
                    self.finish_platform_start,
                    0.0,
                    self.size[1],
                    -pit_depth,
                ),
                _floor_region(
                    body,
                    self.finish_platform_start,
                    self.size[0],
                    0.0,
                    self.size[1],
                    0.0,
                ),
            )
        )
        x_positions = np.arange(
            self.start_platform_length + x_pitch / 2,
            self.finish_platform_start - x_pitch / 3,
            x_pitch,
        )
        y_positions = np.arange(y_pitch / 2, self.size[1], y_pitch)
        for row, x in enumerate(x_positions):
            offset = 0.0 if row % 2 == 0 else y_pitch / 2
            for y in y_positions:
                center_y = float(y + offset)
                if center_y > self.size[1] - block_size / 2:
                    continue
                geoms.append(
                    _box(
                        body,
                        (
                            block_size / 2,
                            block_size / 2,
                            (top_height + pit_depth) / 2,
                        ),
                        (float(x), center_y, (top_height - pit_depth) / 2),
                        _STONE,
                    )
                )
        return TerrainOutput(origin=_origin(self.size), geometries=geoms)


@dataclass(kw_only=True)
class WmpSingleBeamTerrainCfg(SubTerrainCfg):
    """One centered narrow beam spanning a full-width deep pit."""

    bridge_width_range: tuple[float, float] = (0.60, 0.34)
    bridge_height_range: tuple[float, float] = (0.05, 0.10)
    pit_depth_range: tuple[float, float] = (0.30, 0.60)
    pit_start: float = 2.0
    pit_end: float = 6.2

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        del rng
        body = spec.body("terrain")
        difficulty = _difficulty(difficulty)
        pit_depth = _lerp(difficulty, *self.pit_depth_range)
        geoms = []
        cy = self.size[1] / 2
        geoms.extend(
            (
                _floor_region(body, 0.0, self.pit_start, 0.0, self.size[1], 0.0),
                _floor_region(
                    body,
                    self.pit_start,
                    self.pit_end,
                    0.0,
                    self.size[1],
                    -pit_depth,
                ),
                _floor_region(body, self.pit_end, self.size[0], 0.0, self.size[1], 0.0),
            )
        )
        width = _lerp(difficulty, *self.bridge_width_range)
        height = _lerp(difficulty, *self.bridge_height_range)
        length = self.pit_end - self.pit_start
        geoms.append(
            _box(
                body,
                (length / 2, width / 2, height / 2),
                (self.pit_start + length / 2, cy, height / 2),
                _WOOD,
            )
        )
        return TerrainOutput(origin=_origin(self.size), geometries=geoms)


class WmpFlatTerrainCfg(SubTerrainCfg):
    """Plain start deck used by the first stage of the two-stage schedule.

    Stage one exists to answer one question -- can the policy walk at the
    commanded speed with a periodic, upright gait -- and a course of obstacles
    confounds that question with terrain difficulty. The deck keeps the same
    tile size so the checkpoint transfers to the obstacle course unchanged.
    """

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        del difficulty, rng
        body = spec.body("terrain")
        return TerrainOutput(
            origin=_origin(self.size), geometries=[_floor(body, self.size)]
        )


WMP_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    # The source terrain uses independent 8 x 8 m patches in a row/column grid.
    size=(8.0, 8.0),
    num_rows=WMP_NUM_ROWS,
    num_cols=len(WMP_TERRAIN_NAMES) + len(WMP_OMNI_TERRAIN_NAMES),
    border_width=25.0,
    color_scheme="none",
    # mjlab samples rows over [0, 1]; cap the raw value at .9 to match the
    # reference row / num_rows curriculum for ten rows. _difficulty() maps
    # that raw range back to the full geometry scale.
    difficulty_range=(0.0, _REFERENCE_MAX_ROW_DIFFICULTY),
    sub_terrains={
        "stairs": WmpStairsTerrainCfg(proportion=1.0),
        "low_frame": WmpLowFrameTerrainCfg(proportion=1.0),
        "high_wall": WmpHighWallTerrainCfg(proportion=1.0),
        "plum_blossom": WmpPlumBlossomTerrainCfg(proportion=1.0),
        "gap": WmpGapTerrainCfg(proportion=1.0),
        "tire_array": WmpTireArrayTerrainCfg(proportion=1.0),
        "single_beam": WmpSingleBeamTerrainCfg(proportion=1.0),
        "ramp": WmpRampTerrainCfg(proportion=1.0),
        # Omni-directional and stand-still practice. The order matters: the
        # obstacle families keep column indices 0-7, so every terrain-name mask
        # in the rewards and the curriculum keeps pointing at the same tiles.
        "omni_flat": WmpFlatTerrainCfg(proportion=1.0),
    },
)


WMP_FLAT_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=False,
    size=(8.0, 8.0),
    num_rows=1,
    num_cols=1,
    border_width=25.0,
    color_scheme="none",
    difficulty_range=(0.0, _REFERENCE_MAX_ROW_DIFFICULTY),
    sub_terrains={"flat": WmpFlatTerrainCfg(proportion=1.0)},
)


__all__ = [
    "WMP_FLAT_TERRAINS_CFG",
    "WMP_FORWARD_TERRAIN_NAMES",
    "WMP_OMNI_TERRAIN_NAMES",
    "WMP_TERRAINS_CFG",
    "WMP_TERRAIN_NAMES",
    "WmpFlatTerrainCfg",
    "WmpGapTerrainCfg",
    "WmpHighWallTerrainCfg",
    "WmpLowFrameTerrainCfg",
    "WmpPlumBlossomTerrainCfg",
    "WmpRampTerrainCfg",
    "WmpSingleBeamTerrainCfg",
    "WmpStairsTerrainCfg",
    "WmpTireArrayTerrainCfg",
]
