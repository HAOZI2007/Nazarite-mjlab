"""Behavioral checks for XZ1-only terrain-aware pose regularization."""

from types import SimpleNamespace

import pytest
import torch
from nazarite.config.train_config.xz1.him_env_cfg import XZ1_HIM_JOINT_NAMES, XZ1_HIM_STANCE_PRECOMPENSATION, Nazarite_HIM_Complex_Terrain_XZ1

from mjlab.utils.lab_api.string import resolve_matching_names_values


class _Scene(dict):
  terrain: SimpleNamespace


@pytest.fixture
def pose_setup():
  cfg = Nazarite_HIM_Complex_Terrain_XZ1()
  term = cfg.rewards["pose"]
  names = list(XZ1_HIM_JOINT_NAMES)
  term.params["asset_cfg"].joint_ids = list(range(len(names)))
  init = cfg.scene.entities["robot"].init_state.joint_pos
  assert init is not None
  _, _, values = resolve_matching_names_values(init, names)
  default = torch.tensor([values] * 7)
  asset = SimpleNamespace(
    data=SimpleNamespace(
      joint_pos=default.clone(),
      default_joint_pos=default.clone(),
      root_link_pos_w=torch.zeros(7, 3),
    ),
    find_joints=lambda *_args: (list(range(12)), names),
  )
  assert cfg.scene.terrain is not None
  generator = cfg.scene.terrain.terrain_generator
  assert generator is not None
  terrain = SimpleNamespace(
    cfg=SimpleNamespace(terrain_generator=generator),
    terrain_types=torch.arange(7),
  )
  scan = SimpleNamespace(
    distances=torch.ones(7, 4),
    hit_pos_w=torch.tensor(
      [[[-0.1, -0.1, 0.0], [-0.1, 0.1, 0.0], [0.1, -0.1, 0.0], [0.1, 0.1, 0.0]]] * 7
    ),
    normals_w=torch.tensor([[[0.0, 0.0, 1.0]] * 4] * 7),
  )
  scene = _Scene(robot=asset, terrain_scan=SimpleNamespace(data=scan))
  scene.terrain = terrain
  command = torch.tensor([[0.5, 0.0, 0.0]] * 7)
  env = SimpleNamespace(
    scene=scene,
    num_envs=7,
    device="cpu",
    command_manager=SimpleNamespace(get_command=lambda _name: command),
  )
  reward = term.func(term, env)
  return term, env, reward, command, names


def test_pose_tracks_initial_state_instead_of_precompensated_servo_target(pose_setup):
  term, env, reward, _, names = pose_setup
  baseline = reward(env, **term.params)
  wall = list(env.scene.terrain.cfg.terrain_generator.sub_terrains).index("high_wall")
  expected = torch.ones(7)
  expected[wall] = 0
  torch.testing.assert_close(baseline, expected)

  offset = torch.tensor([XZ1_HIM_STANCE_PRECOMPENSATION[name] for name in names])
  env.scene["robot"].data.joint_pos += offset
  servo_target_reward = reward(env, **term.params)
  assert (servo_target_reward[expected.bool()] < 1).all()
  assert servo_target_reward[wall] == 0


def test_high_wall_mask_follows_adaptive_terrain_reassignment(pose_setup):
  term, env, reward, _, _ = pose_setup
  terrain = env.scene.terrain
  wall = list(terrain.cfg.terrain_generator.sub_terrains).index("high_wall")
  terrain.terrain_types[:] = wall
  env.scene["robot"].data.joint_pos += 0.5
  assert torch.equal(reward(env, **term.params), torch.zeros(7))
  terrain.terrain_types[:] = 0
  assert (reward(env, **term.params) > 0).all()


def test_pose_constrains_hips_more_than_leg_flexion(pose_setup):
  term, env, reward, _, names = pose_setup
  env.scene.terrain.terrain_types[:] = 0
  for row, suffix in enumerate(("hip_joint", "thigh_joint", "shank_joint")):
    ids = [i for i, name in enumerate(names) if name.endswith(suffix)]
    env.scene["robot"].data.joint_pos[row, ids] += 0.4
  result = reward(env, **term.params)
  assert 0 < result[0] < result[1] < result[2] < 1
  assert torch.equal(result[3:], torch.ones(4))


def test_rough_terrain_relief_is_moving_only_and_small_for_hips(pose_setup):
  term, env, reward, command, names = pose_setup
  env.scene.terrain.terrain_types[:] = 0
  hips = [i for i, name in enumerate(names) if name.endswith("hip_joint")]
  env.scene["robot"].data.joint_pos[:, hips] += 0.4
  command[1] = 0
  flat = reward(env, **term.params)
  env.scene["terrain_scan"].data.hit_pos_w[:, :, 2] = torch.tensor(
    [-0.12, 0.12, -0.12, 0.12]
  )
  rough = reward(env, **term.params)
  assert flat[0] < rough[0] < 0.5  # Relief must not erase a large hip deviation.
  assert rough[1] == flat[1]  # Standing has no terrain tolerance.


def test_missing_scan_hits_fall_back_to_finite_pose_tracking(pose_setup):
  term, env, reward, _, _ = pose_setup
  env.scene.terrain.terrain_types[:] = 0
  env.scene["robot"].data.joint_pos += 0.2
  baseline = reward(env, **term.params)
  scan = env.scene["terrain_scan"].data
  scan.distances[:] = -1
  scan.hit_pos_w[:] = float("nan")
  scan.distances[0, 0] = 1
  scan.hit_pos_w[0, 0] = torch.tensor([0.1, 0.1, 0.12])
  fallback = reward(env, **term.params)
  assert torch.isfinite(fallback).all()
  torch.testing.assert_close(fallback, baseline)


def test_pose_config_covers_all_twelve_joints_with_positive_tolerances(pose_setup):
  term, _, reward, _, names = pose_setup
  assert len(names) == 12
  assert reward.values.shape == (4, 12)
  assert (reward.values[:3] > 0).all()
  for i, name in enumerate(names):
    if name.endswith("hip_joint"):
      assert reward.values[1, i] == pytest.approx(0.20)
      assert reward.values[3, i] == pytest.approx(0.04)
  assert term.params["scan_sensor_name"] == "terrain_scan"
