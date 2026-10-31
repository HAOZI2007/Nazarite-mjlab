from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
from nazarite.config.train_config.xz1.him_env_cfg import Nazarite_HIM_Complex_Terrain_XZ1
from nazarite.config.train_config.xz1.him_terrain import BoxTireArrayTerrainCfg, BoxTrapezoidRampTerrainCfg
from nazarite.mdp.shared import events as custom_events

from mjlab.managers.scene_entity_config import SceneEntityCfg


def test_tire_array_keeps_half_tile_clear_and_fills_the_other_half():
  cfg = BoxTireArrayTerrainCfg(
    size=(8.0, 8.0),
    tire_outer_diameter=0.55,
    tire_inner_diameter=0.33,
    tire_width=0.165,
    tire_pitch=0.56,
    startup_zone_length=4.0,
  )
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  output = cfg.function(0.0, spec, np.random.default_rng(0))
  tires = [
    geom.geom
    for geom in output.geometries
    if geom.geom.type == mujoco.mjtGeom.mjGEOM_MESH
  ]

  assert len(tires) == 7 * 14
  assert np.isclose(output.origin[0], 2.0)
  tire_x = np.array([geom.pos[0] for geom in tires])
  tire_y = np.array([geom.pos[1] for geom in tires])
  assert np.min(tire_x) - cfg.tire_outer_diameter / 2 >= cfg.startup_zone_length
  assert np.max(tire_x) + cfg.tire_outer_diameter / 2 <= cfg.size[0]
  assert np.min(tire_y) - cfg.tire_outer_diameter / 2 >= 0.0
  assert np.max(tire_y) + cfg.tire_outer_diameter / 2 <= cfg.size[1]
  spec.compile()


def test_trapezoid_ramp_is_a_solid_full_width_wedge():
  cfg = BoxTrapezoidRampTerrainCfg(
    size=(8.0, 8.0),
    start_x=1.6,
    height_range=(0.06, 0.36),
    slope_angle_range=(10.0, 20.0),
    deck_length=0.8,
    spawn_x_fraction=0.075,
  )
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  output = cfg.function(1.0, spec, np.random.default_rng(0))
  spec.compile()

  assert output.origin[0] == pytest.approx(0.6)
  # One floor box plus one closed prism mesh.
  assert len(output.geometries) == 2
  prism = [g for g in output.geometries if g.geom.type == mujoco.mjtGeom.mjGEOM_MESH]
  assert len(prism) == 1
  mesh = spec.mesh(prism[0].geom.meshname)
  assert mesh.name.startswith("xz1_ramp_")
  # Six closed faces, two triangles each: the prism has no open interior.
  assert len(mesh.userface) == 6 * 2 * 3
  vertices = np.asarray(mesh.uservert).reshape(-1, 3)
  assert vertices[:, 2].max() == pytest.approx(0.36)
  assert vertices[:, 2].min() == pytest.approx(0.0)
  # Full patch width in y, so the robot has to climb instead of walking around.
  assert vertices[:, 1].min() == pytest.approx(0.0)
  assert vertices[:, 1].max() == pytest.approx(cfg.size[1])
  # Ascend, deck and mirrored descent share one summit height.
  assert np.count_nonzero(np.isclose(vertices[:, 2], 0.36)) == 4
  summit_x = np.unique(vertices[np.isclose(vertices[:, 2], 0.36), 0])
  assert summit_x[1] - summit_x[0] == pytest.approx(0.8)
  assert summit_x[0] - cfg.start_x == pytest.approx(0.36 / np.tan(np.deg2rad(20.0)))

  # Lower difficulty reduces the height and the ascent angle.
  low_spec = mujoco.MjSpec()
  low_spec.worldbody.add_body(name="terrain")
  low = cfg.function(0.0, low_spec, np.random.default_rng(0))
  low_prism = [g for g in low.geometries if g.geom.type == mujoco.mjtGeom.mjGEOM_MESH][
    0
  ]
  low_vertices = np.asarray(low_spec.mesh(low_prism.geom.meshname).uservert).reshape(
    -1, 3
  )
  assert low_vertices[:, 2].max() == pytest.approx(0.06)


def test_highest_curriculum_level_generates_competition_obstacle_profiles():
  cfg = Nazarite_HIM_Complex_Terrain_XZ1()
  generator = cfg.scene.terrain.terrain_generator
  level = cfg.curriculum["terrain_levels"].params["sparse_max_level"]
  lower, upper = generator.difficulty_range
  difficulty = lower + (upper - lower) * level / (generator.num_rows - 1)
  assert difficulty == pytest.approx(1.0)
  for name in ("high_wall", "trapezoid_ramp"):
    terrain = generator.sub_terrains[name]
    terrain.size = generator.size
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    output = terrain.function(difficulty, spec, np.random.default_rng(0))
    model = spec.compile()
    assert model.ngeom == 2
    if name == "high_wall":
      wall = output.geometries[1].geom
      assert wall.size[0] * 2 == pytest.approx(0.08)
      assert wall.pos[2] + wall.size[2] == pytest.approx(0.30)
    else:
      ramp = output.geometries[1].geom
      vertices = np.asarray(spec.mesh(ramp.meshname).uservert).reshape(-1, 3)
      summit = vertices[np.isclose(vertices[:, 2], 0.36)]
      assert len(summit) == 4
      assert np.ptp(summit[:, 0]) == pytest.approx(0.8)
      run = summit[:, 0].min() - vertices[:, 0].min()
      assert np.rad2deg(np.arctan(0.36 / run)) == pytest.approx(20.0)


def test_ramp_rejects_a_ramp_that_cannot_fit_the_tile():
  cfg = BoxTrapezoidRampTerrainCfg(
    size=(8.0, 8.0), start_x=6.0, height_range=(0.30, 0.30), deck_length=2.0
  )
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  with pytest.raises(ValueError, match="does not fit"):
    cfg.function(1.0, spec, np.random.default_rng(0))


def test_high_wall_reset_uses_fixed_yaw_only_for_wall_envs(monkeypatch):
  calls = []

  def record_reset(_env, env_ids, pose_range, velocity_range, asset_cfg):
    calls.append((env_ids.clone(), dict(pose_range), velocity_range, asset_cfg))

  monkeypatch.setattr(custom_events, "reset_root_state_uniform", record_reset)
  generator = SimpleNamespace(sub_terrains={"flat": object(), "high_wall": object()})
  terrain = SimpleNamespace(
    cfg=SimpleNamespace(terrain_generator=generator),
    terrain_types=torch.tensor([0, 1, 1]),
  )
  env = SimpleNamespace(
    num_envs=3,
    device=torch.device("cpu"),
    scene=SimpleNamespace(terrain=terrain),
  )
  pose_range = {"x": (-0.5, 0.5), "yaw": (-3.14, 3.14)}
  asset_cfg = SceneEntityCfg("robot")

  custom_events.reset_root_state_uniform_facing_high_wall(
    env, torch.tensor([0, 1, 2]), pose_range, {}, asset_cfg
  )

  assert len(calls) == 2
  assert torch.equal(calls[0][0], torch.tensor([0]))
  assert calls[0][1]["yaw"] == (-3.14, 3.14)
  assert torch.equal(calls[1][0], torch.tensor([1, 2]))
  assert calls[1][1]["yaw"] == (0.0, 0.0)
  assert pose_range["yaw"] == (-3.14, 3.14)
