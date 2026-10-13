"""Regression tests for XZ1-specific HIM reward terms."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from nazarite.config.robot_config.xz1_cfg import (
  XZ1_FOOT_GEOMS,
  XZ1_FOOT_SITES,
  XZ1_HIP_BODIES,
  XZ1_THIGH_BODIES,
  XZ1_XML,
)
from nazarite.config.train_config.XZ1_env_cfgs.him_env_cfg import (
  XZ1_HIM_GAIT_OFFSETS,
  Nazarite_HIM_Complex_Terrain_XZ1,
)
from nazarite.mdp import rewards

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.velocity import mdp as velocity_mdp


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


def test_xz1_him_config_wires_reference_reward_contract():
  cfg = Nazarite_HIM_Complex_Terrain_XZ1()
  gait = cfg.rewards["foot_gait"]
  air_time = cfg.rewards["feet_air_time"]

  assert cfg.rewards["self_collision"].func is rewards.current_self_collision_cost
  terrain_names = tuple(cfg.scene.terrain.terrain_generator.sub_terrains)
  assert terrain_names == ("stairs_up", "stairs_down", "tire_array", "high_wall")
  assert cfg.commands["twist"].forward_only_terrain_names == (
    "tire_array",
    "high_wall",
  )
  assert cfg.commands["twist"].forward_only_x_min == 0.3
  assert "discrete_grid" not in terrain_names
  assert cfg.scene.terrain.terrain_generator.sub_terrains["tire_array"].tire_outer_diameter == 0.55
  assert cfg.scene.terrain.terrain_generator.sub_terrains["tire_array"].tire_inner_diameter == 0.33
  assert cfg.scene.terrain.terrain_generator.sub_terrains["tire_array"].tire_width == 0.165
  assert cfg.scene.terrain.terrain_generator.sub_terrains["tire_array"].startup_zone_length == 4.0
  assert cfg.rewards["foot_clearance"].weight == -0.5
  assert cfg.rewards["foot_clearance"].params["max_height"] == 0.40
  assert cfg.rewards["foot_clearance"].params["lookahead_distance"] == 0.45
  assert cfg.scene.terrain.terrain_generator.sub_terrains["high_wall"].wall_thickness == 0.08
  assert cfg.scene.terrain.terrain_generator.sub_terrains["high_wall"].wall_height_range[1] == 0.30
  assert cfg.events["reset_base"].func.__name__ == "reset_root_state_uniform_facing_high_wall"
  assert cfg.curriculum["terrain_levels"].params["high_wall_crossing_clearance"] == 0.4
  assert air_time.func is rewards.him_feet_air_time
  assert air_time.params["gait_name"] == "foot_gait"
  assert gait.weight == 0.15
  assert cfg.rewards["pose"].func is rewards.terrain_variable_posture
  assert cfg.rewards["pose"].params["scan_sensor_name"] == "terrain_scan"
  assert cfg.rewards["pose"].params["terrain_tolerance"][r".*_thigh_joint"] == 0.15
  assert "track_velocity_x" not in cfg.rewards
  assert "track_velocity_y" not in cfg.rewards
  assert "track_yaw_velocity" not in cfg.rewards
  assert cfg.rewards["track_linear_velocity"].func is rewards.ri4438_track_linear_velocity
  assert cfg.rewards["track_linear_velocity"].weight == 2.0
  assert cfg.rewards["track_angular_velocity"].func is rewards.track_yaw_velocity
  assert cfg.rewards["track_angular_velocity"].weight == 1.0
  assert "nonfoot_ground_touch" not in cfg.rewards
  assert cfg.rewards["leg_collision"].func is velocity_mdp.illegal_contact
  assert cfg.rewards["leg_collision"].weight == -0.5
  assert cfg.rewards["leg_collision"].params == {
    "sensor_name": "leg_ground_contact", "force_threshold": 1.0
  }
  sensor_names = {sensor.name for sensor in cfg.scene.sensors or ()}
  assert "nonfoot_ground_touch" not in sensor_names
  assert "leg_ground_contact" in sensor_names
  leg_sensor = next(s for s in cfg.scene.sensors if s.name == "leg_ground_contact")
  assert leg_sensor.history_length == 1
  assert leg_sensor.primary.mode == "geom"
  assert cfg.rewards["foot_clearance"].func is rewards.terrain_relative_feet_clearance
  assert "rr_thigh_torque_ratio" in cfg.metrics
  assert "rl_thigh_torque_ratio" in cfg.metrics
  assert "rear_thigh_ground_contact" in cfg.metrics
  assert any(
    sensor.name == "rear_thigh_ground_contact"
    for sensor in cfg.scene.sensors or ()
  )
  import xml.etree.ElementTree as ET
  root = ET.parse(XZ1_XML).getroot()
  import re
  matched_leg_geoms = {
    geom.attrib["name"]
    for geom in root.iter("geom")
    if re.fullmatch(leg_sensor.primary.pattern, geom.attrib.get("name", ""))
  }
  assert matched_leg_geoms == {
    f"{leg}_{part}_link_primitive_1"
    for leg in ("FL", "FR", "RR", "RL")
    for part in ("hip", "thigh", "calf")
  }
  calf_ranges = {
    joint.attrib["name"]: tuple(float(v) for v in joint.attrib["range"].split())
    for joint in root.iter("joint")
    if joint.attrib.get("name", "").endswith("_calf_joint")
  }
  assert calf_ranges == {
    f"{leg}_calf_joint": (-2.7227, -0.83776)
    for leg in ("FL", "FR", "RR", "RL")
  }
  assert (
    cfg.observations["actor"].terms["phase"].params["period"] == gait.params["period"]
  )
  assert XZ1_HIM_GAIT_OFFSETS == (0.0, 0.5, 0.0, 0.5)
  assert XZ1_FOOT_SITES == ("FL", "FR", "RR", "RL")
  leg_order = ("FL", "FR", "RR", "RL")
  assert XZ1_FOOT_GEOMS == tuple(f"{leg}_foot_link_primitive_1" for leg in leg_order)
  assert XZ1_HIP_BODIES == tuple(f"{leg}_hip_link" for leg in leg_order)
  assert XZ1_THIGH_BODIES == tuple(f"{leg}_thigh_link" for leg in leg_order)

  self_collision_sensor = next(
    sensor
    for sensor in cfg.scene.sensors or ()
    if sensor.name == "self_collision_contact"
  )
  assert self_collision_sensor.history_length == 1
