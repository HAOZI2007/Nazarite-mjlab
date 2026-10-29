"""Regression tests for XZ1 HIM rewards, randomization, and episode timing."""

import math
import re
from types import SimpleNamespace
from unittest.mock import MagicMock
from xml.etree import ElementTree

import mujoco
import numpy as np
import pytest
import torch
from nazarite.config.robot_config.xz1_cfg import (
  XZ1_BASE_BODY,
  XZ1_FOOT_GEOMS,
  XZ1_FOOT_SITES,
  XZ1_INIT_STATE,
  XZ1_XML,
  get_xz1_cfg,
)
from nazarite.config.train_config.XZ1_env_cfgs import him_course
from nazarite.config.train_config.XZ1_env_cfgs.him_env_cfg import (
  XZ1_HIM_BASE_CONTACT_HISTORY,
  XZ1_HIM_BASE_CONTACT_LOAD_STEPS,
  XZ1_HIM_GAIT_OFFSETS,
  XZ1_HIM_GAIT_PERIOD,
  XZ1_HIM_JOINT_NAMES,
  XZ1_HIM_STANCE_PRECOMPENSATION,
  XZ1_HIM_STANCE_TARGET,
  Nazarite_HIM_Complex_Terrain_XZ1,
)
from nazarite.config.train_config.XZ1_env_cfgs.him_rl_cfg import xz1_him_runner_cfg
from nazarite.mdp import rewards, terminations

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.utils.lab_api.string import resolve_matching_names_values


def _init_joint_positions() -> dict[str, float]:
  """Resolve XZ1_INIT_STATE's wildcard joint map to the twelve leg joints."""
  joint_pos = XZ1_INIT_STATE.joint_pos
  assert joint_pos is not None, "XZ1_INIT_STATE.joint_pos must list the legs."
  _, names, values = resolve_matching_names_values(joint_pos, XZ1_HIM_JOINT_NAMES)
  return dict(zip(names, values, strict=True))


def _contact_sensor(**data_fields) -> MagicMock:
  sensor = MagicMock(spec=ContactSensor)
  sensor.data = SimpleNamespace(**data_fields)
  return sensor


def test_current_self_collision_ignores_old_history_samples():
  force_history = torch.zeros(2, 2, 3, 3)
  force_history[0, 0, 0, 0] = 5.0
  force_history[1, 1, -1, 0] = 5.0
  sensor = _contact_sensor(force_history=force_history, force=None, found=None)
  env = SimpleNamespace(
    num_envs=2,
    device=torch.device("cpu"),
    scene={"self_collision": sensor},
  )

  result = rewards.current_self_collision_cost(
    env, "self_collision", force_threshold=1.0
  )

  torch.testing.assert_close(result, torch.tensor([0.0, 1.0]))


def test_ri4438_linear_tracking_combines_xy_and_leaves_vertical_to_separate_term(
  monkeypatch,
):
  command = torch.tensor([[0.4, -0.2, 0.0], [0.4, -0.2, 0.0]])
  actual = torch.tensor([[0.4, -0.2, 2.0], [0.9, -0.2, 0.0]])
  monkeypatch.setattr(
    rewards,
    "safe_base_lin_vel",
    lambda _env, _asset_cfg: actual,
  )
  env = SimpleNamespace(
    num_envs=2,
    device=torch.device("cpu"),
    command_manager=SimpleNamespace(get_command=lambda _name: command),
  )

  result = rewards.ri4438_track_linear_velocity(
    env,
    std=0.5,
    command_name="twist",
    asset_cfg=SceneEntityCfg("robot"),
  )

  torch.testing.assert_close(
    result, torch.tensor([1.0, torch.exp(torch.tensor(-1.0)).item()])
  )


def test_ri4438_yaw_tracking_ignores_roll_pitch_and_penalizes_yaw_error(monkeypatch):
  command = torch.tensor([[0.0, 0.0, 0.3]] * 3)
  actual = torch.tensor([[0.0, 0.0, 0.3], [2.0, -3.0, 0.3], [2.0, -3.0, 0.8]])
  monkeypatch.setattr(rewards, "safe_base_ang_vel", lambda _env, _asset_cfg: actual)
  env = SimpleNamespace(
    num_envs=3,
    device=torch.device("cpu"),
    command_manager=SimpleNamespace(get_command=lambda _name: command),
  )

  result = rewards.track_yaw_velocity(env, std=0.5, command_name="twist")

  torch.testing.assert_close(
    result, torch.tensor([1.0, 1.0, torch.exp(torch.tensor(-1.0)).item()])
  )


def test_him_feet_air_time_reads_and_matches_foot_gait_phase():
  target_time = 0.6 * (1.0 - 0.56)
  contact_time = torch.tensor([[target_time, 0.0, target_time, 0.0]])
  air_time = torch.tensor([[0.0, target_time, 0.0, target_time]])
  sensor = _contact_sensor(
    current_air_time=air_time,
    current_contact_time=contact_time,
  )
  gait = SimpleNamespace(
    params={
      "period": 0.6,
      "offset": [0.0, 0.5, 0.0, 0.5],
      "threshold": 0.56,
      "command_name": "twist",
    }
  )
  command_manager = SimpleNamespace(
    get_command=lambda _name: torch.tensor([[0.5, 0.0, 0.0]])
  )
  env = SimpleNamespace(
    num_envs=1,
    device=torch.device("cpu"),
    step_dt=0.02,
    episode_length_buf=torch.tensor([3]),
    scene={"feet": sensor},
    cfg=SimpleNamespace(rewards={"foot_gait": gait}),
    command_manager=command_manager,
  )

  result = rewards.him_feet_air_time(
    env,
    sensor_name="feet",
    gait_name="foot_gait",
    command_name="twist",
    command_threshold=0.1,
  )

  assert result.item() == pytest.approx(target_time)

  sensor.data.current_contact_time = torch.tensor(
    [[0.0, target_time, 0.0, target_time]]
  )
  sensor.data.current_air_time = torch.tensor([[target_time, 0.0, target_time, 0.0]])
  wrong_pair = rewards.him_feet_air_time(
    env,
    sensor_name="feet",
    gait_name="foot_gait",
    command_name="twist",
    command_threshold=0.1,
  )
  assert wrong_pair.item() == 0.0


def test_terrain_relative_clearance_penalizes_stationary_dragging_swing_feet():
  feet = torch.tensor(
    [[[-0.2, 0.2, 0.022], [0.2, 0.2, 0.022], [-0.2, -0.2, 0.022], [0.2, -0.2, 0.022]]]
  )
  asset = SimpleNamespace(
    data=SimpleNamespace(
      site_pos_w=feet,
      root_link_pos_w=torch.zeros(1, 3),
      root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    )
  )
  scan = SimpleNamespace(
    data=SimpleNamespace(
      distances=torch.ones(1, 4),
      hit_pos_w=torch.cat((feet[..., :2], torch.zeros(1, 4, 1)), dim=-1),
      normals_w=torch.tensor([[[0.0, 0.0, 1.0]] * 4]),
    )
  )
  gait = SimpleNamespace(
    params={
      "period": 0.8,
      "offset": [0.0, 0.5, 0.0, 0.5],
      "threshold": 0.5,
    }
  )
  env = SimpleNamespace(
    num_envs=1,
    device=torch.device("cpu"),
    step_dt=0.02,
    episode_length_buf=torch.tensor([30]),
    scene={"robot": asset, "terrain_scan": scan},
    cfg=SimpleNamespace(rewards={"foot_gait": gait}),
    command_manager=SimpleNamespace(
      get_command=lambda _name: torch.tensor([[0.5, 0.0, 0.0]])
    ),
  )
  asset_cfg = SceneEntityCfg("robot", site_ids=[0, 1, 2, 3])

  result = rewards.terrain_relative_feet_clearance(
    env,
    target_height=0.08,
    gait_name="foot_gait",
    scan_sensor_name="terrain_scan",
    command_name="twist",
    command_threshold=0.1,
    asset_cfg=asset_cfg,
    foot_radius=0.022,
  )

  assert result.item() == pytest.approx(0.16, abs=1.0e-6)


# XZ1-specific weights based on the Go2 HIM reference, including positive pose
# tracking outside high walls and narrow-course support regularization.
XZ1_REFERENCE_REWARD_WEIGHTS = {
  # Steering retuned by hand after the stall diagnosis: linear tracking pays
  # more, angular tracking pays less (it was handing out 0.98 per step whenever
  # the command was zero), orientation is softened and gait credit is tenfold.
  "track_linear_velocity": 1.5,
  "track_angular_velocity": 0.8,
  "body_orientation_l2": -0.5,
  "pose": 0.2,
  "body_ang_vel": -0.05,
  "angular_momentum": -0.025,
  "is_terminated": -10.0,
  "joint_acc_l2": -2.5e-7,
  "joint_pos_limits": -1.0,
  "action_rate_l2": -0.015,
  "smoothness": -0.02,
  "joint_torques_l2": -2.0e-5,
  "hip_pos": -0.05,
  "narrow_support_edge": -0.2,
  "foot_gait": 0.1,
  "foot_clearance": -0.01,
  "foot_slip": -0.05,
  "soft_landing": -1e-3,
  "stand_still": -1.0,
  "leg_collision": -1.0,
  # Added on top of the reference set to price retrograde motion: the Gaussian
  # tracking reward is convex, so rocking through the command pays when steady
  # tracking is impossible.
  "reverse_velocity": -1.0,
  # Ported from the UW-HIMLoco dog stack: hovering-foot cost, wedged-foot cost
  # and left/right hip symmetry.
  "prolonged_air_time": -0.5,
  "feet_stumble": -0.5,
  "hip_mirror_symmetry": -0.2,
}


def _leg_contact_sensor(force_history: torch.Tensor) -> MagicMock:
  sensor = _contact_sensor(
    force_history=force_history,
    force=torch.zeros(force_history.shape[0], force_history.shape[1], 3),
    found=None,
  )
  sensor.history_length = force_history.shape[2]
  return sensor


def test_reverse_velocity_penalty_only_prices_motion_against_the_command():
  class _Asset:
    def __init__(self, velocity):
      self.data = SimpleNamespace(root_link_lin_vel_b=velocity)

  def fake_env(velocity, command):
    return SimpleNamespace(
      num_envs=velocity.shape[0],
      device=velocity.device,
      scene={"robot": _Asset(velocity)},
      command_manager=SimpleNamespace(get_command=lambda _name: command),
    )

  velocity = torch.tensor(
    [
      [0.9, 0.0, 0.0],  # tracking -> free
      [-0.5, 0.0, 0.2],  # retrograde -> 0.5
      [0.0, 0.6, 0.0],  # lateral drift -> not retrograde
      [-0.3, 0.0, 0.0],  # no command -> inactive
    ]
  )
  command = torch.tensor(
    [
      [1.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
    ]
  )

  result = rewards.reverse_velocity_penalty(
    fake_env(velocity, command), command_name="twist"
  )

  assert result.tolist() == pytest.approx([0.0, 0.5, 0.0, 0.0])


def test_sustained_leg_collision_requires_contact_for_the_whole_window():
  # History index 0 is the newest substep.
  force_history = torch.zeros(3, 1, 4, 3)
  force_history[0, :, :, 0] = 50.0  # env0: on the ground for the whole window
  force_history[1, :, 0, 0] = 50.0  # env1: touched only in the newest substep
  force_history[2, :, 1:, 0] = 50.0  # env2: contact ended before the newest step
  sensor = _leg_contact_sensor(force_history)
  env = SimpleNamespace(num_envs=3, device=torch.device("cpu"), scene={"leg": sensor})

  result = rewards.sustained_leg_terrain_contact(env, "leg", force_threshold=1.0)

  torch.testing.assert_close(result, torch.tensor([1.0, 0.0, 0.0]))


def test_sustained_leg_collision_ignores_low_force_and_clamps_short_windows():
  force_history = torch.zeros(2, 2, 4, 3)
  force_history[0, :, :, 0] = 50.0
  force_history[1, :, :, 0] = 0.5
  sensor = _leg_contact_sensor(force_history)
  env = SimpleNamespace(num_envs=2, device=torch.device("cpu"), scene={"leg": sensor})

  clamped = rewards.sustained_leg_terrain_contact(
    env, "leg", force_threshold=1.0, window_length=64
  )

  torch.testing.assert_close(clamped, torch.tensor([1.0, 0.0]))


def test_sustained_leg_collision_without_history_is_zero():
  sensor = _contact_sensor(
    force=torch.full((2, 8, 3), 50.0),
    force_history=None,
    found=None,
  )
  env = SimpleNamespace(num_envs=2, device=torch.device("cpu"), scene={"leg": sensor})

  result = rewards.sustained_leg_terrain_contact(env, "leg")

  torch.testing.assert_close(result, torch.zeros(2))


def test_xz1_him_config_matches_reference_reward_set():
  cfg = Nazarite_HIM_Complex_Terrain_XZ1()

  assert tuple(cfg.rewards) == tuple(XZ1_REFERENCE_REWARD_WEIGHTS)
  for name, weight in XZ1_REFERENCE_REWARD_WEIGHTS.items():
    assert cfg.rewards[name].weight == pytest.approx(weight), name

  tracking = cfg.rewards["track_linear_velocity"].params
  assert cfg.rewards["track_linear_velocity"].func is rewards.him_track_linear_velocity
  assert tracking["std"] == pytest.approx(0.5)
  assert cfg.rewards["track_angular_velocity"].params["std"] == pytest.approx(2.0**-0.5)

  # Pose tracking uses terrain relief and masks high walls in the callable.
  pose = cfg.rewards["pose"]
  assert pose.func is him_course.XZ1TerrainPosture
  assert pose.weight == pytest.approx(0.2)
  assert pose.params["walking_threshold"] == 0.1
  assert pose.params["running_threshold"] == 1.5

  generator = cfg.scene.terrain.terrain_generator
  terrain_names = tuple(generator.sub_terrains)
  assert terrain_names == (
    "flat",
    "pyramid_stairs",
    "random_grid",
    "trapezoid_ramp",
    "high_wall",
    "narrow_stairs",
    "narrow_ramp",
  )
  assert cfg.commands["twist"].forward_only_terrain_names == ("high_wall",)
  # Heading mode: the yaw reference is a sampled heading angle, so the linear
  # command ranges still cover every direction while net displacement tracks
  # real progress again. Half of the envs keep plain yaw-rate tracking.
  twist = cfg.commands["twist"]
  assert twist.heading_command is True
  assert twist.rel_heading_envs == pytest.approx(0.5)
  assert twist.heading_control_stiffness == pytest.approx(0.5)
  assert twist.ranges.heading == (pytest.approx(-math.pi), pytest.approx(math.pi))
  assert twist.ranges.lin_vel_x == (-1.0, 1.0)
  assert twist.ranges.lin_vel_y == (-1.0, 1.0)
  assert cfg.commands["twist"].forward_only_x_min == 0.3
  # Difficulty climbs along the rows, so the last row is the 300 mm wall of the
  # competition rule and the ramp reaches its steepest, tallest form there.
  assert generator.num_rows == 10
  assert generator.difficulty_range == (0.0, 1.0)
  wall = generator.sub_terrains["high_wall"]
  assert wall.wall_height_range == (0.05, 0.30)
  ramp = generator.sub_terrains["trapezoid_ramp"]
  assert ramp.height_range == (0.06, 0.36)
  assert ramp.slope_angle_range == (10.0, 20.0)
  assert ramp.deck_length == pytest.approx(0.8)
  grid = generator.sub_terrains["random_grid"]
  assert grid.fill_gaps is True
  assert grid.grid_height_range == (0.0, 0.10)
  assert XZ1_HIM_GAIT_OFFSETS == (0.0, 0.5, 0.0, 0.5)

  # The terrain curriculum follows the legged_gym reference: a promotion needs
  # two consecutive successes, a demotion three failures, and the success
  # distance is capped at the 4.0 m half tile.
  levels = cfg.curriculum["terrain_levels"].params
  assert levels["max_success_distance"] == pytest.approx(4.0)
  assert levels["success_streak_length"] == 2
  assert levels["failure_streak_length"] == 3
  assert levels["flat_terrain_name"] == "flat"
  assert levels["sparse_max_level"] == generator.num_rows - 1 == 9

  # Reference physics timing: 2 ms MuJoCo steps, ten of them per 20 ms action.
  assert cfg.sim.mujoco.timestep == 0.002
  assert cfg.decimation == 10
  assert cfg.sim.mujoco.timestep * cfg.decimation == pytest.approx(0.02)
  # Solver and contact budgets match the reference Go2 HIM run.
  assert cfg.sim.mujoco.solver == "newton"
  assert cfg.sim.mujoco.tolerance == pytest.approx(1e-8)
  assert cfg.sim.mujoco.ls_iterations == 20
  assert cfg.sim.mujoco.ls_tolerance == pytest.approx(0.01)
  assert cfg.sim.mujoco.ccd_iterations == 50
  assert cfg.sim.nconmax == 256
  assert cfg.sim.njmax == 1500
  assert cfg.sim.contact_sensor_maxmatch == 500

  air_time = cfg.rewards["prolonged_air_time"]
  assert air_time.func is rewards.prolonged_air_time
  assert air_time.params["sensor_name"] == "feet_ground_contact"
  assert air_time.params["max_air_time"] == pytest.approx(0.60)
  stumble = cfg.rewards["feet_stumble"]
  assert stumble.func is rewards.feet_stumble_reference
  assert stumble.params["sensor_name"] == "feet_ground_contact"
  assert stumble.params["horizontal_ratio"] == pytest.approx(5.0)
  symmetry = cfg.rewards["hip_mirror_symmetry"]
  assert symmetry.func is rewards.hip_mirror_symmetry
  assert symmetry.params["asset_cfg"].name == "robot"

  clearance = cfg.rewards["foot_clearance"]
  assert clearance.func is velocity_mdp.feet_clearance
  assert clearance.weight == -0.01
  # 0.08 was tuned around a 22 mm foot radius; the sphere is now 24 mm, and the
  # Target raised to 0.25 so the swing clears the tallest wall band the
  # curriculum can reach (0.23 m at level 6).
  assert clearance.params["target_height"] == pytest.approx(0.25)
  assert clearance.params["height_sensor_name"] == "feet_terrain_height"
  assert clearance.params["asset_cfg"].site_names == XZ1_FOOT_SITES
  sensors = {sensor.name: sensor for sensor in cfg.scene.sensors}
  feet_height = sensors["feet_terrain_height"]
  assert tuple(ref.name for ref in feet_height.frame) == XZ1_FOOT_SITES
  assert feet_height.reduction == "min"

  # Leg collision is a sustained-contact cost over thigh and shank geoms only,
  # not the illegal_contact termination repurposed as a reward.
  leg_collision = cfg.rewards["leg_collision"]
  assert leg_collision.func is rewards.sustained_leg_terrain_contact
  assert leg_collision.weight == -1.0
  assert leg_collision.params["force_threshold"] == 1.0
  assert leg_collision.params["sensor_name"] == "leg_ground_contact"
  assert "self_collision_contact" not in sensors

  leg_sensor = sensors["leg_ground_contact"]
  assert leg_sensor.history_length == 20
  assert leg_collision.params["window_length"] == 2 * cfg.decimation

  leg_geoms = {
    geom.get("name")
    for geom in ElementTree.parse(XZ1_XML).iter("geom")
    if geom.get("name")
  }
  matched = {name for name in leg_geoms if re.findall(leg_sensor.primary.pattern, name)}
  # Each thigh carries one collision box; each shank carries two thin cylinders
  # (upper and lower), so the sensor sees twelve leg geoms and no foot or base
  # geometry.
  expected_leg_geoms = {
    f"{leg}_thigh_link_primitive_1" for leg in ("FL", "FR", "RR", "RL")
  }
  expected_leg_geoms |= {
    f"{leg}_shank_link_primitive_{index}"
    for leg in ("FL", "FR", "RR", "RL")
    for index in (1, 2)
  }
  assert matched == expected_leg_geoms

  # A 0.25 m lift needs half the 0.6 s period for the swing, so the gait reward
  # must not expect the foot back on the ground sooner than that.
  gait = cfg.rewards["foot_gait"]
  assert gait.params["threshold"] == pytest.approx(0.50)
  assert gait.params["period"] == pytest.approx(XZ1_HIM_GAIT_PERIOD)
  air = cfg.rewards["prolonged_air_time"]
  assert air.params["max_air_time"] == pytest.approx(0.60)

  # Only sustained vertical support ends the episode, allowing wall impacts
  # and brief scrapes: 150 ms loaded inside a 200 ms window.
  assert list(cfg.terminations) == [
    "time_out",
    "base_contact",
    "fell_over",
    "left_narrow_course",
    "narrow_course_complete",
  ]
  base_term = cfg.terminations["base_contact"]
  assert base_term.func is terminations.sustained_base_contact
  assert base_term.params["sensor_name"] == "base_ground_contact"
  assert base_term.params["force_threshold"] == pytest.approx(5.0)
  assert base_term.params["min_loaded_steps"] == XZ1_HIM_BASE_CONTACT_LOAD_STEPS
  assert base_term.params["min_vertical_force_ratio"] == pytest.approx(0.7)
  assert XZ1_HIM_BASE_CONTACT_HISTORY == round(0.20 / cfg.sim.mujoco.timestep)
  assert XZ1_HIM_BASE_CONTACT_LOAD_STEPS == round(0.15 / cfg.sim.mujoco.timestep)
  base_sensor = sensors["base_ground_contact"]
  assert base_sensor.history_length == XZ1_HIM_BASE_CONTACT_HISTORY
  assert base_sensor.primary.pattern == XZ1_BASE_BODY
  assert base_sensor.secondary.pattern == "terrain"
  assert base_sensor.reduce == "netforce"

  actor = cfg.observations["actor"].terms
  assert actor["base_ang_vel"].delay_max_lag == 2
  assert actor["joint_vel"].delay_hold_prob == pytest.approx(0.3)
  for term in cfg.observations["critic"].terms.values():
    assert term.delay_max_lag == 0

  assert cfg.episode_length_s == 20.0

  # The reference run clips raw actions at 10.0 in the runner, not in the env
  # action term (whose `clip` bounds processed joint targets instead).
  runner = xz1_him_runner_cfg()
  assert runner.clip_actions == 10.0
  assert runner.actor.action_clip == 10.0
  assert cfg.actions["joint_pos"].clip is None


def test_xz1_him_randomization_matches_reference_run():
  """Every event mirrors go2_him/2026-09-28_20-59-53/params/env.yaml."""
  cfg = Nazarite_HIM_Complex_Terrain_XZ1()
  events = cfg.events

  assert list(events) == [
    "reset_base",
    "reset_robot_joints",
    "push_robot",
    "foot_friction",
    "encoder_bias",
    "base_com",
    "pd_gains",
    "effort_limits",
    "pseudo_inertia",
  ]

  reset = events["reset_base"]
  assert reset.mode == "reset"
  # XZ1_INIT_STATE is the loaded stance the servo actually holds, so the root
  # spawn is a plain 1-5 cm terrain clearance lift above the 0.30 m stance.
  assert reset.params["pose_range"] == {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (0.01, 0.05),
    "yaw": (-3.14, 3.14),
  }
  assert XZ1_INIT_STATE.pos[2] == pytest.approx(0.30)
  assert reset.params["velocity_range"] == {}

  joints = events["reset_robot_joints"]
  assert joints.mode == "reset"
  assert joints.func.__name__ == "reset_joints_by_offset"
  assert joints.params["position_range"] == (0.0, 0.0)
  assert joints.params["velocity_range"] == (0.0, 0.0)

  # The zero action must hold the loaded stance: the servo target is the init
  # pose pushed past it by the calibrated static pre-compensation.
  action = cfg.actions["joint_pos"]
  assert action.use_default_offset is False
  assert action.offset == XZ1_HIM_STANCE_TARGET
  assert set(action.offset) == set(XZ1_HIM_JOINT_NAMES)
  assert set(XZ1_HIM_STANCE_PRECOMPENSATION) == set(XZ1_HIM_JOINT_NAMES)
  assert all(abs(offset) < 0.25 for offset in XZ1_HIM_STANCE_PRECOMPENSATION.values())
  # Knees and thighs have to reach past the stance to carry the body, and the
  # rear legs carry more, so their pre-compensation is the larger one.
  assert all(
    XZ1_HIM_STANCE_PRECOMPENSATION[name] > 0.0
    for name in XZ1_HIM_JOINT_NAMES
    if "shank" in name or "thigh" in name
  )
  assert (
    XZ1_HIM_STANCE_PRECOMPENSATION["RR_shank_joint"]
    > XZ1_HIM_STANCE_PRECOMPENSATION["FL_shank_joint"]
  )
  init_joint_pos = _init_joint_positions()
  for name in XZ1_HIM_JOINT_NAMES:
    assert action.offset[name] == pytest.approx(
      init_joint_pos[name] + XZ1_HIM_STANCE_PRECOMPENSATION[name]
    )

  push = events["push_robot"]
  assert push.mode == "interval"
  assert push.interval_range_s == (5.0, 6.0)
  assert push.params["velocity_range"] == {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.4, 0.4),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
  }

  friction = events["foot_friction"]
  assert friction.mode == "startup"
  assert friction.params["operation"] == "abs"
  assert friction.params["ranges"] == (0.3, 1.6)
  assert friction.params["shared_random"] is True
  assert friction.params["asset_cfg"].geom_names == XZ1_FOOT_GEOMS

  bias = events["encoder_bias"]
  assert bias.mode == "startup"
  assert bias.params["bias_range"] == (-0.015, 0.015)

  com = events["base_com"]
  assert com.mode == "startup"
  assert com.params["operation"] == "add"
  assert com.params["ranges"] == {
    0: (-0.05, 0.05),
    1: (-0.05, 0.05),
    2: (-0.05, 0.05),
  }
  assert com.params["asset_cfg"].body_names == (XZ1_BASE_BODY,)

  gains = events["pd_gains"]
  assert gains.mode == "startup"
  assert gains.params["kp_range"] == (0.9, 1.1)
  assert gains.params["kd_range"] == (0.9, 1.1)
  assert gains.params["operation"] == "scale"

  effort = events["effort_limits"]
  assert effort.mode == "startup"
  assert effort.params["effort_limit_range"] == (0.9, 1.1)
  assert effort.params["operation"] == "scale"

  inertia = events["pseudo_inertia"]
  assert inertia.mode == "startup"
  assert inertia.params["distribution"] == "uniform"
  assert inertia.params["alpha_range"] == pytest.approx(
    (0.5 * math.log(0.8), 0.5 * math.log(1.2))
  )


def test_zero_action_holds_the_calibrated_stance():
  """The pre-compensated target has to stand at 0.300 m with a level trunk.

  This is the measured claim behind ``XZ1_HIM_STANCE_PRECOMPENSATION``: a
  position servo needs a steady-state error to carry the body, so a zero action
  only holds XZ1_INIT_STATE if the target sits past it by that error.
  """
  init = _init_joint_positions()

  spec = Entity(get_xz1_cfg()).spec
  spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0.0, 0.0, 0.05))
  model = spec.compile()
  data = mujoco.MjData(model)

  qpos_adr = {
    name: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
    for name in init
  }
  qpos = np.zeros(model.nq)
  qpos[2] = XZ1_INIT_STATE.pos[2]
  qpos[3] = 1.0
  for name, value in init.items():
    qpos[qpos_adr[name]] = value
  data.qpos[:] = qpos
  mujoco.mj_forward(model, data)

  ctrl = np.zeros(model.nu)
  for actuator in range(model.nu):
    joint = model.actuator_trnid[actuator, 0]
    ctrl[actuator] = XZ1_HIM_STANCE_TARGET[
      mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    ]

  # Four seconds, matching the calibration: the hips hold their splay through
  # the foot friction cone, so the trunk creeps down about a millimetre per
  # second after the knee has taken up the load.
  for _ in range(int(4.0 / 0.002)):
    data.ctrl[:] = ctrl
    mujoco.mj_step(model, data)

  base_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, XZ1_BASE_BODY)
  qw, qx, qy, qz = data.xquat[base_body]
  pitch = np.degrees(np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0)))
  roll = np.degrees(np.arctan2(2 * (qw * qx + qy * qz), 1.0 - 2 * (qx * qx + qy * qy)))

  assert data.xpos[base_body, 2] == pytest.approx(0.300, abs=1e-3)
  # Residual tilt of the friction-settled splay: a third of a degree, against
  # the 5.4 deg nose-up crouch the servo used to fall into.
  assert abs(pitch) < 0.5
  assert abs(roll) < 0.5
  assert data.ncon == 4


def test_hip_mirror_symmetry_ignores_symmetric_splay_only():
  """Mirror-symmetric abduction is free; a left/right mismatch is charged."""
  names = ["FL_hip_joint", "FR_hip_joint", "RR_hip_joint", "RL_hip_joint"]
  asset = SimpleNamespace(
    data=SimpleNamespace(
      joint_pos=torch.tensor(
        [
          [0.10, -0.10, -0.10, 0.10],
          [0.10, 0.05, -0.10, 0.10],
        ]
      ),
      default_joint_pos=torch.zeros(2, 4),
    ),
    find_joints=lambda _pattern: ([0, 1, 2, 3], names),
  )
  env = SimpleNamespace(num_envs=2, device="cpu", scene={"robot": asset})

  cost = rewards.hip_mirror_symmetry(env)

  # Env 0 splays symmetrically (FL +0.10 / FR -0.10, RL +0.10 / RR -0.10).
  assert cost[0].item() == pytest.approx(0.0, abs=1e-6)
  # Env 1 has FR at +0.05 instead of -0.10, so the front pair is off by 0.15.
  assert cost[1].item() == pytest.approx(0.15, abs=1e-6)


def _base_sensor(loaded: torch.Tensor) -> MagicMock:
  """Fake contact sensor whose force history is loaded where ``loaded`` is 1."""
  force_history = torch.zeros(loaded.shape[0], 1, loaded.shape[1], 3)
  force_history[:, 0, :, 2] = loaded * 20.0
  sensor = MagicMock()
  sensor.data = SimpleNamespace(force_history=force_history)
  return sensor


def test_sustained_base_contact_counts_loaded_substeps_not_a_single_run():
  """Chattered belly load still ends the episode; a short press does not."""
  # Every other substep loaded: 13 of 25 loaded, never two in a row.
  chattered = torch.zeros(1, 25)
  chattered[:, ::2] = 1.0
  env = SimpleNamespace(
    num_envs=1, device="cpu", scene={"base_ground_contact": _base_sensor(chattered)}
  )
  env.extras = {}
  assert bool(terminations.sustained_base_contact(env, min_loaded_steps=10).all())

  # Four loaded substeps is a stumble, not a slide.
  brief = torch.zeros(1, 25)
  brief[:, :4] = 1.0
  env = SimpleNamespace(
    num_envs=1, device="cpu", scene={"base_ground_contact": _base_sensor(brief)}
  )
  env.extras = {}
  assert not bool(terminations.sustained_base_contact(env, min_loaded_steps=10).any())

  # Force threshold is honoured: 4 N never counts, 20 N does.
  weak = torch.full((1, 25), 4.0)
  strong = torch.full((1, 25), 20.0)
  for force, expected in ((weak, False), (strong, True)):
    force_history = torch.zeros(1, 1, 25, 3)
    force_history[:, 0, :, 2] = force
    sensor = MagicMock()
    sensor.data = SimpleNamespace(force_history=force_history)
    env = SimpleNamespace(
      num_envs=1, device="cpu", scene={"base_ground_contact": sensor}
    )
    env.extras = {}
    got = bool(terminations.sustained_base_contact(env, min_loaded_steps=10).any())
    assert got is expected


def test_base_contact_distinguishes_wall_impacts_from_trunk_support():
  cfg = Nazarite_HIM_Complex_Terrain_XZ1()
  term = cfg.terminations["base_contact"]
  history = torch.zeros(6, 1, XZ1_HIM_BASE_CONTACT_HISTORY, 3)
  # Horizontal wall impact, wall with friction, upward/downward floor support,
  # and a trunk resting on the competition's 20 degree slope.
  history[0, :, :, 0] = 200.0
  history[1, :, :, :] = torch.tensor([200.0, 0.0, 30.0])
  history[2, :, :, 2] = 20.0
  history[3, :, :, 2] = -20.0
  history[4, :, :, :] = torch.tensor(
    [20.0 * math.sin(math.radians(20.0)), 0.0, 20.0 * math.cos(math.radians(20.0))]
  )
  # A short scrape has 74 loaded substeps, below the configured 75.
  history[5, :, : XZ1_HIM_BASE_CONTACT_LOAD_STEPS - 1, 2] = 20.0
  sensor = _contact_sensor(force_history=history)
  env = SimpleNamespace(
    num_envs=6,
    device="cpu",
    scene={"base_ground_contact": sensor},
    extras={"log": {}},
  )
  torch.testing.assert_close(
    term.func(env, **term.params),
    torch.tensor([False, False, True, True, True, False]),
  )
  # Interrupted support still triggers at exactly 75 substeps in the window.
  history[5].zero_()
  history[5, :, :, 2] = 20.0
  history[5, :, ::4, 2] = 0.0
  assert term.func(env, **term.params)[5].item()
