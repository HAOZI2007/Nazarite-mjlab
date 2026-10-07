from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from tensordict import TensorDict

from nazarite.config.train_config.env_cfgs.wtw_delta_attention_env_cfg import (
  Nazarite_Wtw_Delta_Go2,
)
from nazarite.config.train_config.wtw_delta_attention_rl_cfg import (
  wtw_delta_go2_runner_cfg,
)
from nazarite.delta.critic_model import DeltaPrivilegedCriticModel
from nazarite.delta.wtw_delta_attention_model import WtwDeltaAttentionModel
from nazarite.delta.terrains import PillarApproachTerrainCfg
from nazarite.mdp.delta_observations import delta_privileged_terrain_map


def test_attention_task_observation_contract_and_elevation_sensor():
  cfg = Nazarite_Wtw_Delta_Go2(
    play=True,
    bev_variant="25x40",
    camera_orientation="vertical",
    camera_pitch_deg=0.0,
  )
  assert set(cfg.observations) == {
    "wtw_proprio",
    "critic_privileged",
    "delta_map",
    "delta_privileged_map",
  }
  elevation = next(sensor for sensor in cfg.scene.sensors if sensor.name == "delta_privileged_scan")
  assert elevation.name == "delta_privileged_scan"
  plum_cfg = cfg.scene.terrain.terrain_generator.sub_terrains["plum_stones"]
  assert plum_cfg.__class__.__name__ == "PillarApproachTerrainCfg"
  assert plum_cfg.pillar_spacing_x == 0.42
  assert plum_cfg.pillar_spacing_y == 0.34
  assert plum_cfg.pillar_radius_range == (0.09, 0.14)
  assert cfg.observations["delta_map"].terms["delta_map"].params["map_height"] == 25
  assert cfg.observations["delta_map"].terms["delta_map"].params["map_width"] == 40
  assert cfg.observations["delta_map"].terms["delta_map"].params["x_range_m"] == (0.0, 1.5)
  assert cfg.observations["delta_map"].terms["delta_map"].params["y_range_m"] == (-0.5, 0.5)
  assert cfg.observations["delta_map"].terms["delta_map"].params["map_channels"] == 3
  assert cfg.observations["delta_map"].terms["delta_map"].params[
    "coordinate_normalization"
  ] == "paper"
  assert "terrain_goal" not in cfg.commands
  assert "tracking_goal_vel" not in cfg.rewards
  assert "tracking_yaw" not in cfg.rewards
  assert cfg.rewards["track_velocity_x"].weight == 1.0
  assert cfg.rewards["track_velocity_y"].weight == 1.0
  assert cfg.rewards["track_yaw_velocity"].weight == 2.0
  assert wtw_delta_go2_runner_cfg().actor.distribution_cfg["std_range"] is None
  # wtw_body_height returns a negative squared error; it must attenuate the
  # positive objective rather than be summed into the positive reward bucket.
  assert "wtw_body_height" not in cfg.reward_composition["positive_terms"]
  assert "wtw_body_height" in cfg.reward_composition["negative_terms"]
  clearance = cfg.rewards["delta_swing_clearance"]
  assert clearance.weight == -0.35
  assert clearance.params["terrain_map_cache_attr"] == "_delta_privileged_map_cache"
  assert cfg.reward_composition["terminal_penalty"] == -50.0

  default_cfg = Nazarite_Wtw_Delta_Go2(play=True)
  assert default_cfg.observations["delta_map"].terms["delta_map"].params["map_height"] == 25
  assert default_cfg.observations["delta_map"].terms["delta_map"].params["map_width"] == 40


def test_plum_pillar_terrain_is_uniform_and_compilable():
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  cfg = PillarApproachTerrainCfg(size=(8.0, 8.0))
  output = cfg.function(0.0, spec, np.random.default_rng(0))
  cylinders = [
    item.geom
    for item in output.geometries
    if item.geom is not None and item.geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER
  ]
  assert len(cylinders) > 0
  assert len({round(float(geom.size[0]), 6) for geom in cylinders}) == 1
  spec.compile()


def test_attention_actor_and_privileged_critic_forward_for_both_maps():
  for height, width in ((16, 26), (25, 40)):
    obs = TensorDict(
      {
        "wtw_proprio": torch.zeros(2, 498),
        "delta_map": torch.zeros(2, height * width * 3),
        "critic_privileged": torch.zeros(2, 211),
        "delta_privileged_map": torch.zeros(2, height * width * 3),
      },
      batch_size=[2],
    )
    actor = WtwDeltaAttentionModel(
      obs,
      {"actor": ["wtw_proprio", "delta_map"]},
      "actor",
      12,
      map_height=height,
      map_width=width,
      map_channels=3,
      map_extent=(1.5, 1.0),
      map_center=(1.20, 0.0),
      distribution_cfg=None,
      prior_group=None,
      delta_proprio_group=None,
    )
    critic = DeltaPrivilegedCriticModel(
      obs,
      {"critic": ["critic_privileged", "delta_privileged_map"]},
      "critic",
      1,
      map_height=height,
      map_width=width,
      map_channels=3,
      map_extent=(1.5, 1.0),
      map_center=(1.20, 0.0),
    )
    assert actor(obs).shape == (2, 12)
    assert critic(obs).shape == (2, 1)


def test_privileged_map_rasterizes_hits_in_requested_body_frame_roi():
  hit_pos = torch.tensor(
    [[[0.10, -0.40, 0.30], [1.40, 0.40, 0.35], [2.00, 0.00, 0.50]]]
  )
  sensor = SimpleNamespace(
    data=SimpleNamespace(hit_pos_w=hit_pos, distances=torch.ones(1, 3)),
  )
  asset = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=torch.zeros(1, 3),
      root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    )
  )
  env = SimpleNamespace(
    scene={"delta_privileged_scan": sensor, "robot": asset},
    extras={"log": {}},
    num_envs=1,
  )
  result = delta_privileged_terrain_map(
    env,
    map_height=4,
    map_width=6,
    x_range_m=(0.0, 1.5),
    y_range_m=(-0.5, 0.5),
  ).reshape(1, 4, 6, 5)
  assert torch.isfinite(result).all()
  assert result[..., 3].sum().item() == 2.0
  assert result[..., 0].min() >= -1.0 and result[..., 0].max() <= 1.0


def test_privileged_three_channel_map_skips_support_diagnostics():
  hit_pos = torch.tensor(
    [[[0.60, -0.20, 0.30], [1.20, 0.20, 0.35]]]
  )
  sensor = SimpleNamespace(
    data=SimpleNamespace(hit_pos_w=hit_pos, distances=torch.ones(1, 2)),
  )
  asset = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=torch.zeros(1, 3),
      root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    )
  )
  env = SimpleNamespace(
    scene={"delta_privileged_scan": sensor, "robot": asset},
    extras={"log": {}},
    num_envs=1,
  )
  result = delta_privileged_terrain_map(
    env,
    map_height=4,
    map_width=6,
    x_range_m=(0.0, 1.5),
    y_range_m=(-0.5, 0.5),
    map_channels=3,
  ).reshape(1, 4, 6, 3)
  assert result.shape == (1, 4, 6, 3)
  assert torch.isfinite(result).all()
  assert not any(key.startswith("DELTA/map_support_") for key in env.extras["log"])


def test_privileged_map_preserves_negative_relative_elevation():
  hit_pos = torch.tensor(
    [[[0.60, 0.00, -0.35], [1.20, 0.00, -0.55]]]
  )
  sensor = SimpleNamespace(
    data=SimpleNamespace(hit_pos_w=hit_pos, distances=torch.ones(1, 2)),
  )
  asset = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=torch.zeros(1, 3),
      root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    )
  )
  env = SimpleNamespace(
    scene={"delta_privileged_scan": sensor, "robot": asset},
    extras={"log": {}},
    num_envs=1,
  )
  result = delta_privileged_terrain_map(
    env,
    map_height=4,
    map_width=6,
    x_range_m=(0.0, 1.5),
    y_range_m=(-0.5, 0.5),
    map_channels=3,
  ).reshape(1, 4, 6, 3)
  assert result[..., 2].min().item() < -0.5
