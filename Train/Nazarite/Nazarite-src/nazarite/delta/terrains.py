"""Terrain primitives used only by DELTA-based Nazarite tasks."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeometry,
  TerrainOutput,
)
from mjlab.terrains.primitive_terrains import BoxSteppingStonesTerrainCfg
from mjlab.terrains.utils import make_border
from mjlab.utils.color import brand_ramp


_MUJOCO_GREEN = (0.25, 0.80, 0.45)


@dataclass(kw_only=True)
class BoxStripedGapsTerrainCfg(SubTerrainCfg):
  """Alternating transverse support strips and deep gaps.

  The layout resembles a zebra crossing viewed from above: light strips are
  collision geometry and dark strips are openings to a recessed floor.  The
  stripes span the lateral direction, so a robot travelling along +x must
  repeatedly cross the gaps.
  """

  strip_width_range: tuple[float, float] = (0.65, 0.42)
  gap_width_range: tuple[float, float] = (0.06, 0.28)
  center_platform_width: float = 0.9
  border_width: float = 0.25
  floor_depth: float = 1.0

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    del rng
    body = spec.body("terrain")
    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    strip_width = (
      self.strip_width_range[0]
      + difficulty * (self.strip_width_range[1] - self.strip_width_range[0])
    )
    gap_width = (
      self.gap_width_range[0]
      + difficulty * (self.gap_width_range[1] - self.gap_width_range[0])
    )
    if strip_width <= 0.0 or gap_width < 0.0:
      raise ValueError("Strip width must be positive and gap width non-negative")

    size_x, size_y = self.size
    inner_x_min = self.border_width
    inner_x_max = size_x - self.border_width
    inner_y_size = size_y - 2.0 * self.border_width
    if inner_x_max <= inner_x_min or inner_y_size <= 0.0:
      raise ValueError("border_width leaves no usable striped terrain area")

    support_color = (0.92, 0.92, 0.92, 1.0)
    pit_color = (0.015, 0.015, 0.015, 1.0)
    geometries: list[TerrainGeometry] = []

    # A recessed floor makes the dark bands genuine holes instead of merely
    # black surface markings.
    floor_thickness = 0.08
    floor = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(size_x / 2.0, size_y / 2.0, floor_thickness / 2.0),
      pos=(size_x / 2.0, size_y / 2.0, -self.floor_depth - floor_thickness / 2.0),
    )
    geometries.append(TerrainGeometry(geom=floor, color=pit_color))

    support_half_height = self.floor_depth / 2.0

    def add_support(x_min: float, x_max: float) -> None:
      x_min = max(x_min, inner_x_min)
      x_max = min(x_max, inner_x_max)
      if x_max - x_min <= 1.0e-4:
        return
      geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=((x_max - x_min) / 2.0, inner_y_size / 2.0, support_half_height),
        pos=((x_min + x_max) / 2.0, size_y / 2.0, -support_half_height),
      )
      geometries.append(TerrainGeometry(geom=geom, color=support_color))

    center_x = size_x / 2.0
    center_width = max(self.center_platform_width, strip_width)
    center_min = center_x - center_width / 2.0
    center_max = center_x + center_width / 2.0
    add_support(center_min, center_max)

    # Moving away from the spawn platform: hole, support, hole, support, ...
    cursor = center_max
    while cursor < inner_x_max:
      cursor += gap_width
      add_support(cursor, cursor + strip_width)
      cursor += strip_width

    cursor = center_min
    while cursor > inner_x_min:
      cursor -= gap_width
      add_support(cursor - strip_width, cursor)
      cursor -= strip_width

    if self.border_width > 0.0:
      border = make_border(
        body,
        self.size,
        (size_x - 2.0 * self.border_width, size_y - 2.0 * self.border_width),
        self.floor_depth,
        (size_x / 2.0, size_y / 2.0, -support_half_height),
      )
      geometries.extend(TerrainGeometry(geom=geom, color=support_color) for geom in border)

    origin = np.array((center_x, size_y / 2.0, 0.0))
    return TerrainOutput(origin=origin, geometries=geometries)


@dataclass(kw_only=True)
class BoxApproachSteppingStonesTerrainCfg(BoxSteppingStonesTerrainCfg):
  """Stepping stones reached from a dedicated starting platform.

  The generic stepping-stones terrain places its spawn origin in the middle of
  the tile.  That is a poor reset position when the centre is a pit.  This
  DELTA-only variant reserves the rear of the tile for a solid platform and
  starts the stone field in front of it along +x.  Other terrain presets keep
  the generic implementation and are therefore unaffected.
  """

  approach_platform_length: float = 1.2
  """Length of the solid platform along the robot's forward (+x) direction."""
  approach_platform_width: float = 1.4
  """Width of the starting platform along y."""
  approach_gap: float = 0.25
  """Clear gap between the platform edge and the first stone."""

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")
    geometries: list[TerrainGeometry] = []
    difficulty = float(np.clip(difficulty, 0.0, 1.0))

    size_x, size_y = self.size
    inner_min_x, inner_max_x = self.border_width, size_x - self.border_width
    inner_min_y, inner_max_y = self.border_width, size_y - self.border_width
    if inner_max_x <= inner_min_x or inner_max_y <= inner_min_y:
      raise ValueError("border_width leaves no usable stepping-stones area")

    platform_length = float(self.approach_platform_length)
    platform_width = float(self.approach_platform_width)
    approach_gap = float(self.approach_gap)
    if platform_length <= 0.0 or platform_width <= 0.0:
      raise ValueError("Approach platform dimensions must be positive")
    if approach_gap < 0.0:
      raise ValueError("approach_gap must be non-negative")

    platform_length = min(platform_length, inner_max_x - inner_min_x)
    platform_width = min(platform_width, inner_max_y - inner_min_y)
    platform_min_x = inner_min_x
    platform_max_x = platform_min_x + platform_length
    center_y = size_y / 2.0

    stone_size_variation = self.stone_size_variation * difficulty
    displacement_range = self.displacement_range * difficulty
    stone_height_variation = self.stone_height_variation * difficulty
    s_min, s_max = self.stone_size_range
    avg_stone_size = s_max - difficulty * (s_max - s_min)
    if avg_stone_size <= 0.0:
      raise ValueError("stone_size_range must produce a positive stone size")

    # The floor is deliberately deep, so a gap is a real hole in the terrain.
    floor_h = 0.1
    floor_geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(size_x / 2.0, size_y / 2.0, floor_h / 2.0),
      pos=(size_x / 2.0, size_y / 2.0, -self.floor_depth - floor_h / 2.0),
    )
    geometries.append(TerrainGeometry(geom=floor_geom, color=(0.1, 0.1, 0.1, 1.0)))

    z_center = (self.stone_height - self.floor_depth) / 2.0
    half_height = (self.stone_height + self.floor_depth) / 2.0
    if self.border_width > 0.0:
      border_center = (size_x / 2.0, size_y / 2.0, z_center)
      border_boxes = make_border(
        body,
        self.size,
        (size_x - 2.0 * self.border_width, size_y - 2.0 * self.border_width),
        self.stone_height + self.floor_depth,
        border_center,
      )
      border_rgba = (0.16, 0.52, 0.29, 1.0)
      geometries.extend(TerrainGeometry(geom=box, color=border_rgba) for box in border_boxes)

    # A solid platform supports the reset pose.  Its top is flush with the
    # nominal stone height and its footprint is entirely behind the first gap.
    platform_geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(platform_length / 2.0, platform_width / 2.0, half_height),
      pos=(platform_min_x + platform_length / 2.0, center_y, z_center),
    )
    geometries.append(
      TerrainGeometry(geom=platform_geom, color=brand_ramp(_MUJOCO_GREEN, 0.5))
    )

    # Keep the first row at a fixed gap from the platform.  As difficulty
    # increases stones shrink while the grid count stays stable, widening the
    # intervening gaps without moving the spawn platform.
    first_center_x = platform_max_x + approach_gap + avg_stone_size / 2.0
    last_center_x = inner_max_x - avg_stone_size / 2.0
    if last_center_x < first_center_x:
      raise ValueError("Approach platform leaves no room for stepping stones")
    nominal_spacing = s_max + self.stone_distance_range[0]
    num_x = max(1, int(np.floor((last_center_x - first_center_x) / nominal_spacing)) + 1)
    x_centers = np.linspace(first_center_x, last_center_x, num_x)

    nominal_y_spacing = s_max + self.stone_distance_range[0]
    first_center_y = inner_min_y + avg_stone_size / 2.0
    last_center_y = inner_max_y - avg_stone_size / 2.0
    num_y = max(
      1,
      int(np.floor((last_center_y - first_center_y) / nominal_y_spacing)) + 1,
    )
    y_centers = np.linspace(first_center_y, last_center_y, num_y)

    for base_x in x_centers:
      for base_y in y_centers:
        size_stone_x = avg_stone_size + rng.uniform(
          -stone_size_variation, stone_size_variation
        )
        size_stone_y = avg_stone_size + rng.uniform(
          -stone_size_variation, stone_size_variation
        )
        px = base_x + rng.uniform(-displacement_range, displacement_range)
        py = base_y + rng.uniform(-displacement_range, displacement_range)
        # Clamp randomized stones so neither the platform nor the outer border
        # can accidentally absorb a stone and remove the intended first gap.
        px = np.clip(
          px,
          platform_max_x + approach_gap + size_stone_x / 2.0,
          inner_max_x - size_stone_x / 2.0,
        )
        py = np.clip(
          py,
          inner_min_y + size_stone_y / 2.0,
          inner_max_y - size_stone_y / 2.0,
        )

        h = self.floor_depth + self.stone_height + rng.uniform(
          -stone_height_variation, stone_height_variation
        )
        geom = body.add_geom(
          type=mujoco.mjtGeom.mjGEOM_BOX,
          size=(size_stone_x / 2.0, size_stone_y / 2.0, h / 2.0),
          pos=(px, py, -self.floor_depth + h / 2.0),
        )
        geometries.append(
          TerrainGeometry(geom=geom, color=brand_ramp(_MUJOCO_GREEN, rng.uniform(0.4, 0.7)))
        )

    origin = np.array(
      (platform_min_x + platform_length / 2.0, center_y, self.stone_height)
    )
    return TerrainOutput(origin=origin, geometries=geometries)
