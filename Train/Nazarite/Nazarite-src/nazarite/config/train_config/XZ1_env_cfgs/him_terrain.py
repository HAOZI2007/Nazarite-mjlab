"""Competition obstacles used by the XZ1 HIM terrain curriculum."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import (
    SubTerrainCfg,
    TerrainGeometry,
    TerrainOutput,
)


def _floor(body, size: tuple[float, float]):
    return body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(size[0] / 2, size[1] / 2, 0.025),
        pos=(size[0] / 2, size[1] / 2, -0.025),
    )


def _tire_mesh(
    spec: mujoco.MjSpec,
    name: str,
    outer_diameter: float,
    inner_diameter: float,
    tire_width: float,
) -> mujoco.MjsMesh:
    """Create a horizontal tire ring with the rule dimensions.

    The mesh is a rounded rectangular torus: 550 mm outside diameter and
    165 mm axial width, matching a 165/70R13 tire envelope.
    """
    outer_radius = outer_diameter / 2.0
    inner_radius = inner_diameter / 2.0
    if not 0.0 < inner_radius < outer_radius or tire_width <= 0.0:
        raise ValueError("Tire dimensions must satisfy 0 < inner < outer and width > 0.")
    major_radius = (outer_radius + inner_radius) / 2.0
    radial_radius = (outer_radius - inner_radius) / 2.0
    axial_radius = tire_width / 2.0
    n_major, n_minor = 24, 8
    vertices: list[list[float]] = []
    for i in range(n_major):
        u = 2.0 * np.pi * i / n_major
        for j in range(n_minor):
            v = 2.0 * np.pi * j / n_minor
            radius = major_radius + radial_radius * np.cos(v)
            vertices.append([radius * np.cos(u), radius * np.sin(u), axial_radius * np.sin(v)])

    faces: list[int] = []
    for i in range(n_major):
        for j in range(n_minor):
            a = i * n_minor + j
            b = ((i + 1) % n_major) * n_minor + j
            c = ((i + 1) % n_major) * n_minor + (j + 1) % n_minor
            d = i * n_minor + (j + 1) % n_minor
            faces.extend((a, b, c, a, c, d))
    return spec.add_mesh(
        name=name,
        uservert=np.asarray(vertices, dtype=np.float64).ravel().tolist(),
        userface=faces,
    )


@dataclass(kw_only=True)
class BoxTireArrayTerrainCfg(SubTerrainCfg):
    """Dense rule-sized tire array after a clear launch platform."""

    tire_outer_diameter: float = 0.55
    tire_inner_diameter: float = 0.33
    tire_width: float = 0.165
    tire_pitch: float = 0.56
    tire_height: float = 0.165
    startup_zone_length: float = 4.0
    spawn_x_fraction: float = 0.25

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        del difficulty
        if not 0.0 <= self.startup_zone_length < self.size[0]:
            raise ValueError("startup_zone_length must be in [0, terrain length).")
        if self.tire_pitch <= 0.0:
            raise ValueError("tire_pitch must be positive.")
        body = spec.body("terrain")
        floor = _floor(body, self.size)
        mesh = _tire_mesh(
            spec,
            f"xz1_tire_ring_{rng.integers(int(1e9))}",
            self.tire_outer_diameter,
            self.tire_inner_diameter,
            self.tire_width,
        )
        geometries = [TerrainGeometry(geom=floor, color=(0.28, 0.28, 0.30, 1.0))]

        # Keep the first half of the patch clear for spawning, then fill the
        # remaining half with the rule-dimension tires at the configured pitch.
        tire_radius = self.tire_outer_diameter / 2.0
        x_min = self.startup_zone_length + tire_radius
        x_max = self.size[0] - tire_radius
        y_min = tire_radius
        y_max = self.size[1] - tire_radius

        def centered_lattice(lower: float, upper: float) -> np.ndarray:
            count = max(1, int(np.floor((upper - lower) / self.tire_pitch)) + 1)
            center = (lower + upper) / 2.0
            return center + (np.arange(count) - (count - 1) / 2.0) * self.tire_pitch

        tire_x = centered_lattice(x_min, x_max)
        tire_y = centered_lattice(y_min, y_max)
        for x in tire_x:
            for y in tire_y:
                geom = body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_MESH,
                    meshname=mesh.name,
                    pos=(x, y, self.tire_height / 2),
                )
                geom.friction = (0.8, 0.1, 0.1)
                geometries.append(TerrainGeometry(geom=geom, color=(0.04, 0.04, 0.04, 1.0)))

        # Spawn in the center of the clear half; the tire field starts ahead.
        if not 0.0 <= self.spawn_x_fraction <= 1.0:
            raise ValueError("spawn_x_fraction must be in [0, 1].")
        origin = np.array(
            [self.size[0] * self.spawn_x_fraction, self.size[1] / 2, 0.0]
        )
        return TerrainOutput(origin=origin, geometries=geometries)


@dataclass(kw_only=True)
class BoxHighWallTerrainCfg(SubTerrainCfg):
    """A 300 mm high wall spanning the complete terrain width.

    The wall height is curriculum-scaled up to the 300 mm rule value. Its
    length is always the full patch width, so the policy cannot bypass it by
    walking around either end.
    """

    wall_height_range: tuple[float, float] = (0.05, 0.30)
    wall_thickness: float = 0.08
    spawn_x_fraction: float = 0.25

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        del rng
        body = spec.body("terrain")
        floor = _floor(body, self.size)
        height = self.wall_height_range[0] + difficulty * (
            self.wall_height_range[1] - self.wall_height_range[0]
        )
        wall = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            # Full width across y; only the x thickness is 80 mm.
            size=(self.wall_thickness / 2, self.size[1] / 2, height / 2),
            pos=(self.size[0] / 2, self.size[1] / 2, height / 2),
        )
        wall.friction = (0.8, 0.1, 0.1)
        geometries = [
            TerrainGeometry(geom=floor, color=(0.28, 0.28, 0.30, 1.0)),
            TerrainGeometry(geom=wall, color=(0.20, 0.65, 0.30, 1.0)),
        ]
        if not 0.0 <= self.spawn_x_fraction <= 1.0:
            raise ValueError("spawn_x_fraction must be in [0, 1].")
        origin = np.array([self.size[0] * self.spawn_x_fraction, self.size[1] / 2, 0.0])
        return TerrainOutput(origin=origin, geometries=geometries)


__all__ = ["BoxHighWallTerrainCfg", "BoxTireArrayTerrainCfg"]
