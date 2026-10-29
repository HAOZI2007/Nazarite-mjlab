"""Geometry, command and reward regressions for XZ1's narrow support courses."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
from nazarite.config.train_config.XZ1_env_cfgs import him_course as course
from nazarite.config.train_config.XZ1_env_cfgs.him_env_cfg import (
  Nazarite_HIM_Complex_Terrain_XZ1,
)
from nazarite.mdp import curriculums, rewards

from mjlab.managers.scene_entity_config import SceneEntityCfg


class _Scene(dict):
  pass


def test_play_keeps_geometry_types_and_reset_order_consistent():
  cfg = Nazarite_HIM_Complex_Terrain_XZ1(play=True)
  generator = cfg.scene.terrain.terrain_generator
  assert cfg.scene.num_envs == 1
  assert generator.curriculum is True
  assert generator.num_cols == len(generator.sub_terrains) == 7
  assert not cfg.curriculum
  assert next(iter(cfg.events)) == "randomize_terrain"
  assert cfg.events["reset_base"].func is course.reset_xz1_course
  assert cfg.terminations["narrow_course_complete"].time_out is True


def _env(terrain_names, level=9):
  cfg = Nazarite_HIM_Complex_Terrain_XZ1()
  generator = cfg.scene.terrain.terrain_generator
  names = list(generator.sub_terrains)
  count = len(terrain_names)
  asset = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=torch.zeros(count, 3),
      heading_w=torch.zeros(count),
      joint_pos=torch.zeros(count, 4),
      default_joint_pos=torch.zeros(count, 4),
      site_pos_w=torch.zeros(count, 4, 3),
    ),
    find_joints=lambda *_args, **_kwargs: (list(range(4)), list(course.HIP_NAMES)),
    find_sites=lambda *_args, **_kwargs: (list(range(4)), []),
  )
  terrain = SimpleNamespace(
    cfg=SimpleNamespace(terrain_generator=generator),
    terrain_types=torch.tensor([names.index(name) for name in terrain_names]),
    terrain_levels=torch.full((count,), level, dtype=torch.long),
    terrain_origins=torch.zeros(10, len(names), 3),
    max_terrain_level=10,
    env_origins=torch.zeros(count, 3),
  )
  scene = _Scene(robot=asset, terrain=terrain)
  scene.terrain = terrain
  scene["feet_ground_contact"] = SimpleNamespace(
    data=SimpleNamespace(force=torch.zeros(count, 4, 3))
  )
  scene.env_origins = torch.zeros(count, 3)
  command = torch.tensor([[0.5, 0.0, 0.0]] * count)
  env = SimpleNamespace(
    num_envs=count,
    device="cpu",
    scene=scene,
    common_step_counter=1,
    step_dt=0.02,
    max_episode_length_s=20.0,
    reset_terminated=torch.zeros(count, dtype=torch.bool),
    command_manager=SimpleNamespace(get_command=lambda _name: command),
  )
  return cfg, env, command


@pytest.mark.parametrize("name", course.NARROW_TERRAIN_NAMES)
@pytest.mark.parametrize("difficulty, width", [(0.0, 0.8), (1.0, 0.8)])
def test_narrow_geometry_matches_curriculum_and_competition(name, difficulty, width):
  cfg = Nazarite_HIM_Complex_Terrain_XZ1()
  terrain = cfg.scene.terrain.terrain_generator.sub_terrains[name]
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  output = terrain.function(difficulty, spec, np.random.default_rng(0))
  spec.compile()
  assert output.origin == pytest.approx([0.6, 4.0, 0.0])
  if name == "narrow_stairs":
    blocks = [g.geom for g in output.geometries[1:]]
    assert len(blocks) == 5
    assert [b.size[1] * 2 for b in blocks] == pytest.approx([width] * 5)
    height = 0.09 if difficulty == 0 else 0.36
    assert max(b.pos[2] + b.size[2] for b in blocks) == pytest.approx(height)
    assert blocks[2].size[0] * 2 == pytest.approx(0.8)
    assert blocks[0].size[0] * 2 == pytest.approx(0.32)
  else:
    mesh = spec.mesh(output.geometries[1].geom.meshname)
    vertices = np.asarray(mesh.uservert).reshape(-1, 3)
    assert np.ptp(vertices[:, 1]) == pytest.approx(width)
    assert vertices[:, 2].max() == pytest.approx(0.06 if difficulty == 0 else 0.36)
  # Runtime corridor metadata must match the compiled obstacle, not just its
  # configuration. In particular, ramp length changes with difficulty.
  _, env, _ = _env([name], level=0 if difficulty == 0 else 9)
  _, _, half_width, _, end = course.course_bounds(env)
  actual_end = (
    max(b.pos[0] + b.size[0] for b in blocks)
    if name == "narrow_stairs"
    else vertices[:, 0].max()
  )
  assert end[0].item() + output.origin[0] == pytest.approx(actual_end)
  assert half_width[0].item() * 2 == pytest.approx(width)


def test_crossing_and_bypass_checks_use_actual_shape_and_do_not_touch_walls():
  _, env, _ = _env(["narrow_stairs", "narrow_ramp", "high_wall", "flat"])
  selected, position, half_width, start, end = course.course_bounds(env)
  torch.testing.assert_close(selected, torch.tensor([True, True, False, False]))
  assert half_width[:2].tolist() == pytest.approx([0.4, 0.4])
  position = env.scene["robot"].data.root_link_pos_w
  position[:2, 0] = end[:2] + 0.36
  position[2:, 1] = 100.0
  complete = course.narrow_course_complete(env)
  assert complete.tolist() == [True, True, False, False]
  position[0, 1] = 0.35
  position[0, 0] = start[0] + 0.1
  assert course.left_narrow_course(env).tolist() == [True, False, False, False]
  filtered = course.narrow_success_filter(
    env, torch.arange(4), torch.ones(4, dtype=torch.bool)
  )
  assert filtered.tolist() == [False, True, True, True]


def test_narrow_curriculum_rejects_partial_traversal_and_lateral_bypass():
  cfg, env, _ = _env(["narrow_stairs"] * 3, level=8)
  _, _, _, _, end = course.course_bounds(env)
  env.scene["robot"].data.root_link_pos_w[:, 0] = end + torch.tensor([-0.1, 0.4, 0.4])
  env.scene["robot"].data.root_link_pos_w[2, 1] = 0.6
  params = dict(cfg.curriculum["terrain_levels"].params)
  params.update(adaptive_type_sampling=False, sparse_success_streak_length=1)
  curriculums.terrain_levels_delta(env, torch.arange(3), **params)
  assert env.scene.terrain.terrain_levels.tolist() == [8, 9, 8]


def test_hip_cost_has_weak_baseline_and_taper_but_wall_cost_is_unchanged():
  cfg, env, command = _env(["narrow_stairs"] * 5 + ["high_wall"] * 3)
  outward = torch.tensor([0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
  env.scene["robot"].data.joint_pos[:] = outward[:, None] * torch.tensor([1, -1, -1, 1])
  command[:, 2] = torch.tensor([0.0, 0.0, 0.15, 0.4, 0.6, 0.0, 0.15, 0.4])
  cost = cfg.rewards["hip_pos"].func(env, **cfg.rewards["hip_pos"].params)
  # Below the 0.15 rad allowance only the weak all-angle baseline applies.
  assert cost[0] == pytest.approx(4 * 0.25 * 0.1**2)
  assert cost[1] == pytest.approx(4 * (6 * 0.35**2 + 0.25 * 0.5**2))
  assert cost[2] == pytest.approx(cost[1].item())
  assert cost[3] == pytest.approx(cost[1].item() / 2)
  assert cost[4] == 0
  legacy = rewards.hip_joint_deviation_penalty(env, "twist")
  torch.testing.assert_close(cost[5:], legacy[5:])
  # Mirroring signs does not evade the inward-leg constraint.
  env.scene["robot"].data.joint_pos[0] = torch.tensor([-0.3, 0.3, 0.3, -0.3])
  assert course.xz1_hip_position_cost(env, "twist")[0] > 0


def test_support_edge_cost_counts_loaded_feet_only_on_the_narrow_obstacle():
  cfg, env, _ = _env(["narrow_stairs"] * 7 + ["high_wall", "flat"])
  _, _, _, start, end = course.course_bounds(env)
  # Translate the complete scene to verify world positions are converted to
  # tile-relative coordinates rather than assuming an origin at world zero.
  env.scene.env_origins[:] = torch.tensor([10.0, -5.0, 0.0])
  feet = torch.zeros(9, 4, 3)
  feet[:, :, 0] = start[:, None] + 0.5
  feet[:, 0, 1] = 0.396  # 0.05 m inside the reserved edge band.
  feet[0, 0, 1] = 0.34  # sphere + 3 cm safety margin fit inside 0.4 m.
  feet[2, 0, 1] *= -1  # Both lateral edges must behave identically.
  feet[5, :, 0] = start[5] - 0.01
  feet[6, :, 0] = end[6] + 0.01
  env.scene["robot"].data.site_pos_w = feet + env.scene.env_origins[:, None]
  found = torch.ones(9, 4)
  force = torch.zeros(9, 4, 3)
  force[:, :, 2] = 20.0
  found[3, 0] = 0  # Swing foot: never penalize the lateral swing trajectory.
  force[4, 0] = torch.tensor([20.0, 0.0, 0.0])  # Wall brush is not support.
  env.scene["feet_ground_contact"] = SimpleNamespace(
    data=SimpleNamespace(found=found, force=force)
  )
  params = dict(cfg.rewards["narrow_support_edge"].params)
  params["asset_cfg"] = SceneEntityCfg("robot", site_ids=list(range(4)))
  cost = course.narrow_support_edge_cost(env, **params)
  torch.testing.assert_close(
    cost,
    torch.tensor([0.0, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    atol=1e-5,
    rtol=1e-5,
  )
  assert cfg.rewards["narrow_support_edge"].weight == -0.2
  sensor = next(s for s in cfg.scene.sensors if s.name == "feet_ground_contact")
  assert sensor.preserve_order


def test_adaptive_sampling_reserves_competition_width_exposure(monkeypatch):
  cfg, env, _ = _env(
    [
      "flat",
      "pyramid_stairs",
      "random_grid",
      "trapezoid_ramp",
      "high_wall",
      "narrow_stairs",
      "narrow_ramp",
    ],
    level=4,
  )
  # EMA order follows the generator, independently of the environment order.
  env._delta_type_success_ema = torch.tensor([0.5, 0.6, 0.2, 0.5, 0.5, 1.0, 1.0])
  env._delta_type_episode_count = torch.full((7,), 100, dtype=torch.long)
  sampled = []

  def sample(probabilities, count, replacement):
    assert count == 7 and replacement
    sampled.append(probabilities.clone())
    return env.scene.terrain.terrain_types.clone()

  monkeypatch.setattr(torch, "multinomial", sample)
  params = dict(cfg.curriculum["terrain_levels"].params)
  params.update(adaptive_warmup_episodes=0, type_success_ema_alpha=0.0)
  result = curriculums.terrain_levels_delta(env, torch.arange(7), **params)
  assert result["narrow_stairs_sampling_probability"] >= 0.15 - 1e-6
  assert result["narrow_ramp_sampling_probability"] >= 0.10 - 1e-6
  assert sampled[0].sum().item() == pytest.approx(1.0)
  assert (
    result["random_grid_sampling_probability"]
    > result["pyramid_stairs_sampling_probability"]
  )
  # Logs must describe the actual distribution passed to multinomial.
  assert sampled[0][-2:].tolist() == pytest.approx([0.15, 0.10])


def test_narrow_commands_correct_drift_and_refresh_after_type_reassignment():
  cfg, env, _ = _env(["narrow_stairs", "narrow_ramp", "high_wall", "flat"])
  term = cfg.commands["twist"].build(env)
  term._resample_command(torch.arange(4))
  env.scene["robot"].data.root_link_pos_w[:2, 1] = torch.tensor([0.1, -0.1])
  env.scene["robot"].data.heading_w[:2] = torch.tensor([0.1, -0.1])
  term._update_command()
  assert (term.command[:2, 0] > 0).all()
  assert term.command[0, 1] < 0 < term.command[1, 1]
  assert term.command[0, 2] < 0 < term.command[1, 2]
  assert not term.is_standing_env[:2].any()
  # At aligned yaw the wall remains the pre-existing forward-only command.
  torch.testing.assert_close(term.command[2, 1:], torch.zeros(2))
  generator = env.scene.terrain.cfg.terrain_generator
  env.scene.terrain.terrain_types[3] = list(generator.sub_terrains).index("narrow_ramp")
  term._resample_command(torch.tensor([3]))
  term._update_command(torch.tensor([3]))
  assert 0.3 <= term.command[3, 0] <= 0.7
  assert not term.is_standing_env[3]


def test_narrow_reset_is_centered_and_does_not_change_other_reset_rules(monkeypatch):
  _, env, _ = _env(["narrow_stairs", "high_wall", "flat"])
  calls = []
  monkeypatch.setattr(
    course,
    "reset_root_state_uniform",
    lambda _env, ids, pose, vel: calls.append(("narrow", ids.tolist(), pose)),
  )
  monkeypatch.setattr(
    course,
    "reset_root_state_uniform_facing_high_wall",
    lambda _env, ids, pose, vel: calls.append(("regular", ids.tolist(), pose)),
  )
  pose = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)}
  course.reset_xz1_course(env, None, pose)
  assert calls[0] == ("regular", [1, 2], pose)
  assert calls[1][1] == [0]
  assert calls[1][2]["y"] == (-0.05, 0.05)
  assert calls[1][2]["yaw"] == (-0.05, 0.05)
  assert pose["yaw"] == (-3.14, 3.14)
