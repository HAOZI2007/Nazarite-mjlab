"""Regression tests for XZ1-specific HIM reward terms."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from nazarite.config.train_config.XZ1_env_cfgs.him_env_cfg import (
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


def test_him_feet_air_time_reads_and_matches_foot_gait_phase():
  target_time = 0.6 * (1.0 - 0.56)
  contact_time = torch.tensor([[target_time, 0.0, 0.0, target_time]])
  air_time = torch.tensor([[0.0, target_time, target_time, 0.0]])
  sensor = _contact_sensor(
    current_air_time=air_time,
    current_contact_time=contact_time,
  )
  gait = SimpleNamespace(
    params={
      "period": 0.6,
      "offset": [0.0, 0.5, 0.5, 0.0],
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
    [[0.0, target_time, target_time, 0.0]]
  )
  sensor.data.current_air_time = torch.tensor([[target_time, 0.0, 0.0, target_time]])
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
      "offset": [0.0, 0.5, 0.5, 0.0],
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
  assert air_time.func is rewards.him_feet_air_time
  assert air_time.params["gait_name"] == "foot_gait"
  assert cfg.rewards["pose"].func is velocity_mdp.variable_posture
  assert cfg.rewards["track_yaw_velocity"].func is rewards.him_track_angular_velocity
  assert cfg.rewards["foot_clearance"].func is rewards.terrain_relative_feet_clearance
  assert (
    cfg.observations["actor"].terms["phase"].params["period"] == gait.params["period"]
  )

  self_collision_sensor = next(
    sensor
    for sensor in cfg.scene.sensors or ()
    if sensor.name == "self_collision_contact"
  )
  assert self_collision_sensor.history_length == 1
