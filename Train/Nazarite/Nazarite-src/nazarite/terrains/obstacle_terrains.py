"""Primitive obstacle layouts used by the HIM blind-locomotion task.

Box obstacles use simple geoms for portability. The gravel pit additionally
uses two compact heightfields to model fine, foot-scale surface roughness. The
tire array uses tangent boxes to approximate an annulus; the resulting outer
and inner radii are 0.275 m and 0.165 m, respectively, while the axial width is
the specified 0.165 m.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeometry,
  TerrainOutput,
)
from mjlab.terrains.utils import make_plane

_FLOOR = (0.45, 0.45, 0.45, 1.0)
_WOOD = (0.20, 0.65, 0.25, 1.0)
_WALL = (0.72, 0.30, 0.18, 1.0)
_GRAVEL = (0.38, 0.36, 0.32, 1.0)
_TIRE = (0.08, 0.10, 0.08, 1.0)
_BRIDGE = (0.52, 0.30, 0.12, 1.0)


def _set_rgba(geom: mujoco.MjsGeom, color: tuple[float, ...]) -> None:
  geom.rgba = color


def _box(
  body: mujoco.MjsBody,
  size: tuple[float, float, float],
  pos: tuple[float, float, float],
  color: tuple[float, float, float, float],
  quat: tuple[float, float, float, float] | None = None,
) -> TerrainGeometry:
  geom = body.add_geom(
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=size,
    pos=pos,
  )
  if quat is not None:
    geom.quat = quat
  _set_rgba(geom, color)
  return TerrainGeometry(geom=geom, color=color)


def _floor(
  body: mujoco.MjsBody,
  size: tuple[float, float],
) -> TerrainGeometry:
  return TerrainGeometry(geom=make_plane(body, size, 0.0, center_zero=False)[0], color=_FLOOR)


def _rough_hfield(
  spec: mujoco.MjSpec,
  body: mujoco.MjsBody,
  size: tuple[float, float],
  center: tuple[float, float],
  rng: np.random.Generator,
  height_range: tuple[float, float],
  horizontal_scale: float,
  vertical_scale: float,
) -> TerrainGeometry:
  """Create a small quantized heightfield for fine gravel-like roughness."""
  width_cells = max(2, int(round(size[0] / horizontal_scale)))
  length_cells = max(2, int(round(size[1] / horizontal_scale)))
  min_height, max_height = height_range
  max_step = max(1, int(round(max_height / vertical_scale)))
  min_step = max(0, int(round(min_height / vertical_scale)))
  if min_step > max_step:
    raise ValueError("height_range must be ordered and compatible with vertical_scale")

  # Independent low-amplitude cells make the surface irregular at foot scale,
  # while quantization keeps the generated terrain reproducible and stable.
  heights = rng.integers(
    min_step,
    max_step + 1,
    size=(width_cells, length_cells),
    dtype=np.int16,
  )
  peak_step = max(int(heights.max()), 1)
  peak_height = peak_step * vertical_scale
  field = spec.add_hfield(
    name=f"gravel_hfield_{uuid.uuid4().hex}",
    size=[size[0] * 0.5, size[1] * 0.5, peak_height, peak_height],
    nrow=width_cells,
    ncol=length_cells,
    userdata=(heights.astype(np.float32) / peak_step).flatten().tolist(),
  )
  geom = body.add_geom(
    type=mujoco.mjtGeom.mjGEOM_HFIELD,
    hfieldname=field.name,
    pos=(center[0], center[1], 0.0),
  )
  return TerrainGeometry(geom=geom, hfield=field, color=_GRAVEL)


def _origin(cfg: SubTerrainCfg) -> np.ndarray:
  # Spawn near the near edge so the robot has a long approach to the obstacle.
  return np.array([1.0, cfg.size[1] * 0.5, 0.0], dtype=np.float64)


@dataclass(kw_only=True)
class BoxHeightLimitFrameTerrainCfg(SubTerrainCfg):
  """0.8 m x 0.8 m clear opening with a 0.29 m ceiling."""

  inner_height: float = 0.29
  inner_width: float = 0.80
  inner_length: float = 0.80
  obstacle_x: float = 4.40

  def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
    del difficulty, rng
    body = spec.body("terrain")
    geoms = [_floor(body, self.size)]
    cx, cy = self.obstacle_x, self.size[1] * 0.5
    post_t = 0.10
    post_h = self.inner_height
    post_z = post_h * 0.5
    half_w = self.inner_width * 0.5
    half_l = self.inner_length * 0.5
    post_x = half_l + post_t * 0.5
    post_y = half_w + post_t * 0.5
    for x in (cx - post_x, cx + post_x):
      for y in (cy - post_y, cy + post_y):
        geoms.append(_box(body, (post_t * 0.5, post_t * 0.5, post_z), (x, y, post_z), _WOOD))
    beam_h = 0.08
    geoms.append(_box(body, (half_l + post_t, post_t * 0.5, beam_h * 0.5), (cx, cy - post_y, post_h + beam_h * 0.5), _WOOD))
    geoms.append(_box(body, (half_l + post_t, post_t * 0.5, beam_h * 0.5), (cx, cy + post_y, post_h + beam_h * 0.5), _WOOD))
    geoms.append(_box(body, (post_t * 0.5, half_w + post_t, beam_h * 0.5), (cx - half_l, cy, post_h + beam_h * 0.5), _WOOD))
    geoms.append(_box(body, (post_t * 0.5, half_w + post_t, beam_h * 0.5), (cx + half_l, cy, post_h + beam_h * 0.5), _WOOD))
    return TerrainOutput(origin=_origin(self), geometries=geoms)


@dataclass(kw_only=True)
class BoxHighWallTerrainCfg(SubTerrainCfg):
  """A wall crossing the complete terrain tile.

  Only the wall height and longitudinal thickness are task parameters.  The
  lateral span is taken from the tile width so the robot cannot bypass the
  obstacle around either end.
  """

  wall_height: float = 0.32
  wall_thickness: float = 0.10
  obstacle_x: float = 4.40

  def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
    del difficulty, rng
    body = spec.body("terrain")
    cy = self.size[1] * 0.5
    geoms = [_floor(body, self.size)]
    geoms.append(_box(body, (self.wall_thickness * 0.5, self.size[1] * 0.5, self.wall_height * 0.5), (self.obstacle_x, cy, self.wall_height * 0.5), _WALL))
    return TerrainOutput(origin=_origin(self), geometries=geoms)


@dataclass(kw_only=True)
class BoxGravelTerrainCfg(SubTerrainCfg):
  """Full-tile fine gravel terrain with foot-scale height variation."""

  roughness_range: tuple[float, float] = (0.008, 0.045)
  roughness_horizontal_scale: float = 0.04
  roughness_vertical_scale: float = 0.004

  def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
    del difficulty
    body = spec.body("terrain")
    geoms = [_floor(body, self.size)]
    geoms.append(
      _rough_hfield(
        spec,
        body,
        self.size,
        (self.size[0] * 0.5, self.size[1] * 0.5),
        rng,
        self.roughness_range,
        self.roughness_horizontal_scale,
        self.roughness_vertical_scale,
      )
    )
    return TerrainOutput(origin=_origin(self), geometries=geoms)


# Keep old symbols importable for downstream configs and saved experiment
# metadata. Both now generate the full-tile gravel surface.
BoxLGravelPitTerrainCfg = BoxGravelTerrainCfg
BoxLBrickMulchPitTerrainCfg = BoxGravelTerrainCfg


@dataclass(kw_only=True)
class BoxTireArrayTerrainCfg(SubTerrainCfg):
  """Full-tile array of 165/70R13 tires, represented by tangent boxes.

  The original obstacle layout used one 2 x 4 group in the center of the
  tile.  HIM needs substantially more obstacle coverage, so the default now
  tiles the same tire geometry across the complete sub-terrain patch.
  """

  outer_diameter: float = 0.55
  inner_diameter: float = 0.33
  tire_width: float = 0.165
  grid_spacing: float | None = None
  segments: int = 12

  def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
    del difficulty, rng
    body = spec.body("terrain")
    geoms = [_floor(body, self.size)]
    outer_r = self.outer_diameter * 0.5
    inner_r = self.inner_diameter * 0.5
    center_r = (outer_r + inner_r) * 0.5
    radial_half = (outer_r - inner_r) * 0.5
    spacing = self.grid_spacing if self.grid_spacing is not None else self.outer_diameter
    if spacing < self.outer_diameter:
      raise ValueError("grid_spacing must be at least the tire outer diameter")
    if self.segments < 8:
      raise ValueError("segments must be at least 8 for a stable tire ring")
    count = self.segments
    x_centers = np.arange(outer_r, self.size[0] - outer_r + 1.0e-9, spacing)
    y_centers = np.arange(outer_r, self.size[1] - outer_r + 1.0e-9, spacing)
    for cy in y_centers:
      for cx in x_centers:
        for i in range(count):
          theta = 2.0 * math.pi * i / count
          px = cx + center_r * math.cos(theta)
          py = cy + center_r * math.sin(theta)
          half_tangent = center_r * math.pi / count * 1.12
          quat = (math.cos(theta * 0.5), 0.0, 0.0, math.sin(theta * 0.5))
          geoms.append(_box(body, (radial_half, half_tangent, self.tire_width * 0.5), (px, py, self.tire_width * 0.5), _TIRE, quat))
    return TerrainOutput(origin=_origin(self), geometries=geoms)


@dataclass(kw_only=True)
class BoxSingleBridgeTerrainCfg(SubTerrainCfg):
  """3.0 m long, 0.3 m wide, 0.1 m high single timber bridge."""

  bridge_length: float = 3.0
  bridge_width: float = 0.30
  bridge_height: float = 0.10
  obstacle_x: float = 4.40

  def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
    del difficulty, rng
    body = spec.body("terrain")
    cy = self.size[1] * 0.5
    geoms = [_floor(body, self.size)]
    geoms.append(_box(body, (self.bridge_length * 0.5, self.bridge_width * 0.5, self.bridge_height * 0.5), (self.obstacle_x, cy, self.bridge_height * 0.5), _BRIDGE))
    return TerrainOutput(origin=_origin(self), geometries=geoms)


__all__ = [
  "BoxHeightLimitFrameTerrainCfg",
  "BoxHighWallTerrainCfg",
  "BoxGravelTerrainCfg",
  "BoxLGravelPitTerrainCfg",
  "BoxLBrickMulchPitTerrainCfg",
  "BoxSingleBridgeTerrainCfg",
  "BoxTireArrayTerrainCfg",
]
