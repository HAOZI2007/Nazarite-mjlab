"""Check XZ1 automatic-reset supervision and terrain geometry contracts."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock
from xml.etree import ElementTree

import mujoco
import numpy as np
import pytest
import torch
from nazarite.config.robot_config.xz1_cfg import XZ1_XML, get_spec, get_xz1_cfg
from nazarite.config.train_config.xz1.him_env_cfg import Nazarite_HIM_Complex_Terrain_XZ1
from nazarite.config.train_config.xz1.him_rl_cfg import xz1_him_runner_cfg
from nazarite.config.train_config.xz1.terminal_observations import TerminalEstimatorRecorder
from rsl_rl.algorithms.him_ppo import HIMPPO
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv, mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGenerator


def test_him_timing_changes_are_local_and_keep_control_frequency():
  shared = get_xz1_cfg()
  old_delays = [
    (a.delay_max_lag, a.delay_update_period) for a in shared.articulation.actuators
  ]
  cfg = Nazarite_HIM_Complex_Terrain_XZ1()
  assert cfg.sim.mujoco.timestep * cfg.decimation == pytest.approx(0.02)
  assert cfg.auto_reset
  assert cfg.sim.mujoco.solver == "newton"
  for actuator in cfg.scene.entities["robot"].articulation.actuators:
    # Nine 2 ms steps are 18 ms; allow float rounding at the boundary.
    assert actuator.delay_max_lag * cfg.sim.mujoco.timestep <= 0.018 + 1e-9
    assert actuator.delay_update_period * cfg.sim.mujoco.timestep == pytest.approx(0.02)
    actuator.stiffness = 999.0
  assert [
    (a.delay_max_lag, a.delay_update_period) for a in shared.articulation.actuators
  ] == old_delays
  assert all(a.stiffness != 999.0 for a in shared.articulation.actuators)
  assert "him_terminal" not in Nazarite_HIM_Complex_Terrain_XZ1(play=True).recorders


def test_generated_terrain_groups_and_geometry_budget():
  cfg = Nazarite_HIM_Complex_Terrain_XZ1()
  terrain_cfg = deepcopy(cfg.scene.terrain.terrain_generator)
  terrain_cfg.seed = 42
  generator = TerrainGenerator(terrain_cfg, device="cpu")
  spec = mujoco.MjSpec()
  generator.compile(spec)
  model = spec.compile()
  scan = next(sensor for sensor in cfg.scene.sensors if sensor.name == "terrain_scan")
  assert scan.include_geom_groups == (0,)
  assert np.all(model.geom_group == 0)
  assert set(model.geom_type).issubset(
    {mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_MESH}
  )
  # Wide and narrow ramps each use one closed prism per difficulty row;
  # everything else remains boxes, including the narrow staircase.
  assert np.count_nonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_MESH) == (
    2 * terrain_cfg.num_rows
  )
  assert model.ngeom < 2200
  assert generator.terrain_origins.shape == (
    terrain_cfg.num_rows,
    len(terrain_cfg.sub_terrains),
    3,
  )
  stairs = terrain_cfg.sub_terrains["pyramid_stairs"]
  patch_spec = mujoco.MjSpec()
  patch_spec.worldbody.add_body(name="terrain")
  output = stairs.function(1.0, patch_spec, np.random.default_rng(42))
  assert len(output.geometries) == 25

  # The last difficulty row is the rule-height 300 mm wall.
  wall = terrain_cfg.sub_terrains["high_wall"]
  wall_spec = mujoco.MjSpec()
  wall_spec.worldbody.add_body(name="terrain")
  wall.function(1.0, wall_spec, np.random.default_rng(42))
  wall_model = wall_spec.compile()
  assert max(wall_model.geom_size[:, 2]) == pytest.approx(0.15)  # half height

  # The full-width ramp's last row reaches the tallest, steepest form.
  ramp = terrain_cfg.sub_terrains["trapezoid_ramp"]
  ramp_spec = mujoco.MjSpec()
  ramp_spec.worldbody.add_body(name="terrain")
  ramp_out = ramp.function(1.0, ramp_spec, np.random.default_rng(42))
  prism = [g for g in ramp_out.geometries if g.geom.type == mujoco.mjtGeom.mjGEOM_MESH]
  assert len(prism) == 1
  assert ramp_out.origin[0] == pytest.approx(terrain_cfg.size[0] * 0.075)
  assert ramp_out.origin[1] == pytest.approx(terrain_cfg.size[1] / 2)


def test_terrain_scan_groups_exclude_every_robot_geom():
  """Only terrain geoms may be raycast -- never the robot's own meshes."""
  cfg = Nazarite_HIM_Complex_Terrain_XZ1()
  root = ElementTree.parse(XZ1_XML).getroot()
  # Walk bodies only: the <default> block also declares geom templates, which
  # carry no group attribute and are never part of the compiled robot.
  robot_geoms = [
    geom.get("group", "0")
    for body in root.iter("body")
    for geom in body.findall("geom")
  ]
  assert robot_geoms, "the XZ1 XML should contain geoms"
  assert "0" not in robot_geoms
  for sensor in cfg.scene.sensors:
    if sensor.name in ("terrain_scan", "feet_terrain_height"):
      assert sensor.include_geom_groups == (0,)


def test_estimator_uses_independent_history_free_critic_prefix():
  cfg = Nazarite_HIM_Complex_Terrain_XZ1()
  group = cfg.observations["estimator"]
  assert not group.history_length
  assert not group.enable_corruption
  assert tuple(group.terms) == tuple(cfg.observations["critic"].terms)[:8]
  assert xz1_him_runner_cfg().algorithm.estimator_obs_groups == ("estimator",)
  for name, term in group.terms.items():
    assert not term.history_length and not term.delay_max_lag
    assert term.noise is None
    assert (
      term.params["source_func"]
      is cfg.observations["critic"].terms[name].params["source_func"]
    )
    term.params["numerics_tag"] = "changed"
    assert (
      cfg.observations["critic"]
      .terms[name]
      .params["numerics_tag"]
      .startswith("critic/")
    )


def test_done_only_targets_replace_reset_rows_without_mutating_policy_inputs():
  alg = object.__new__(HIMPPO)
  alg.device = "cpu"
  alg.storage = SimpleNamespace(next_observation_groups=("estimator",))
  alg.estimator_obs_groups = ["estimator"]
  alg.estimator_target_slices = ((0, 6), (9, 47), (47, 50))
  alg.estimator_velocity_slice = (47, 50)
  alg.actor = SimpleNamespace(num_one_step_obs=47)
  post_reset = torch.zeros(3, 50)
  terminal = torch.arange(100, dtype=torch.float32).reshape(2, 50)
  obs = TensorDict(
    {"estimator": post_reset, "actor": torch.zeros(3, 282)}, batch_size=[3]
  )
  resolved, valid = alg._resolve_next_observations(
    obs,
    torch.tensor([1, 0, 1]),
    {"terminal_observations": TensorDict({"estimator": terminal}, batch_size=[2])},
  )
  assert valid.all()
  assert tuple(resolved.keys()) == ("estimator",)
  torch.testing.assert_close(resolved["estimator"][[0, 2]], terminal)
  torch.testing.assert_close(resolved["estimator"][1], torch.zeros(50))
  assert not obs["estimator"].any()
  velocity, targets = alg._get_estimator_targets(resolved)
  assert velocity.shape == (3, 3) and targets.shape == (3, 47)
  torch.testing.assert_close(velocity[[0, 2]], terminal[:, 47:50])


def test_recorder_clones_terminal_rows_and_clears_stale_extras():
  cfg = Nazarite_HIM_Complex_Terrain_XZ1()
  sensor_model = SimpleNamespace(
    sensor=lambda _name: SimpleNamespace(type=[mujoco.mjtSensor.mjSENS_GYRO])
  )
  values = torch.arange(150, dtype=torch.float32).reshape(3, 50)
  env = SimpleNamespace(
    observation_manager=SimpleNamespace(
      cfg=cfg.observations, compute_group=lambda _name: values
    ),
    sim=SimpleNamespace(mj_model=sensor_model),
    command_manager=SimpleNamespace(compute=Mock()),
    step_dt=0.02,
    extras={},
  )
  recorder = TerminalEstimatorRecorder(cfg.recorders["him_terminal"], env)
  assert recorder._kinematics_only
  recorder._refresh_terminal_state = Mock()
  ids = torch.tensor([0, 2])
  recorder.record_pre_reset(ids)
  saved = values[ids].clone()
  values.zero_()
  recorder.record_post_reset(ids)
  recorder.record_post_step()
  torch.testing.assert_close(env.extras["terminal_observations"]["estimator"], saved)
  env.command_manager.compute.assert_called_once_with(dt=0.02, env_ids=ids)
  recorder.record_post_step()
  assert "terminal_observations" not in env.extras
  recorder.record_pre_reset(ids)
  recorder.record_post_step()
  recorder.record_post_reset(ids)
  assert "terminal_observations" not in env.extras


def _physics_only_xz1_spec():
  """Use the real XZ1 dynamics without CAD visuals to keep CPU tests small."""
  spec = get_spec()
  for geom in list(spec.geoms):
    if geom.contype == 0 and geom.conaffinity == 0:
      spec.delete(geom)
  for mesh in list(spec.meshes):
    spec.delete(mesh)
  return spec


def _terminal_velocity_push(env, env_ids):
  del env_ids
  velocity = torch.zeros(env.num_envs, 6, device=env.device)
  velocity[:, 0] = 3.0
  env.scene["robot"].write_root_link_velocity_to_sim(velocity)


def _make_parity_cfg(auto_reset):
  cfg = Nazarite_HIM_Complex_Terrain_XZ1()
  cfg.seed = 42
  cfg.scene.num_envs = 2
  cfg.scene.entities["robot"].spec_fn = _physics_only_xz1_spec
  cfg.scene.terrain = TerrainEntityCfg(terrain_type="plane")
  # The parity test uses a plane rather than terrain-generator columns.
  cfg.commands["twist"].forward_only_terrain_names = ()
  cfg.observations["actor"].enable_corruption = False
  cfg.events = {
    "reset": EventTermCfg(func=mdp.reset_scene_to_default, mode="reset"),
    "terminal_push": EventTermCfg(func=_terminal_velocity_push, mode="step"),
  }
  cfg.rewards = {}
  cfg.metrics = {}
  cfg.curriculum = {}
  cfg.terminations = {"time_out": cfg.terminations["time_out"]}
  cfg.auto_reset = auto_reset
  if not auto_reset:
    cfg.recorders = {}
  for actuator in cfg.scene.entities["robot"].articulation.actuators:
    actuator.delay_max_lag = 0
  return cfg


def test_lightweight_terminal_matches_manual_refresh_after_push_and_partial_reset():
  results = {}
  for auto_reset in (False, True):
    env = ManagerBasedRlEnv(cfg=_make_parity_cfg(auto_reset), device="cpu")
    try:
      env.reset(seed=42)
      env.command_manager.get_command("twist")[:] = torch.tensor([0.5, 0.0, 0.0])
      env.episode_length_buf[0] = env.max_episode_length - 1
      action = torch.full((2, 12), 0.15)
      obs, _, terminated, truncated, extras = env.step(action)
      assert (terminated | truncated).tolist() == [True, False]
      assert obs["actor"].shape == (2, 6, 47)
      assert obs["critic"].shape == (2, 244)
      assert obs["estimator"].shape == (2, 50)
      if auto_reset:
        recorder = env.recorder_manager.get_term("him_terminal")
        assert recorder._kinematics_only
        results[True] = extras["terminal_observations"]["estimator"].clone()
        assert env.episode_length_buf[0] == 0
        assert not torch.equal(results[True][0], obs["estimator"][0])
        assert extras["terminal_observations"].batch_size == torch.Size([1])
        assert torch.isfinite(obs["critic"]).all()
        env.step(torch.zeros(2, 12))
        assert "terminal_observations" not in env.extras
      else:
        results[False] = obs["estimator"][:1].clone()
        torch.testing.assert_close(obs["estimator"], obs["critic"][:, :50])
    finally:
      env.close()
  # In particular the 3 m/s post-integration push must reach the velocity
  # sensor; copying the old observation buffer would give an incorrect target.
  assert results[True][0, 47].item() == pytest.approx(3.0, abs=1e-4)
  torch.testing.assert_close(results[True], results[False], atol=1e-5, rtol=1e-5)
