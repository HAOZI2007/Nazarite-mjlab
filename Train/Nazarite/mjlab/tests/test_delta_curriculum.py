from types import SimpleNamespace

import pytest
import torch
from nazarite.config.train_config.XZ1_env_cfgs.him_env_cfg import (
  Nazarite_HIM_Complex_Terrain_XZ1,
)
from nazarite.mdp.curriculums import terrain_levels_delta


class _Scene(dict):
  pass


def test_xz1_sparse_curriculum_reaches_and_retains_the_final_row() -> None:
  cfg = Nazarite_HIM_Complex_Terrain_XZ1()
  generator = cfg.scene.terrain.terrain_generator
  names = list(generator.sub_terrains)
  terrain = SimpleNamespace(
    terrain_levels=torch.tensor([8, 8]),
    terrain_types=torch.tensor(
      [names.index("high_wall"), names.index("trapezoid_ramp")]
    ),
    terrain_origins=torch.zeros(generator.num_rows, len(names), 3),
    env_origins=torch.zeros(2, 3),
    max_terrain_level=generator.num_rows,
    cfg=SimpleNamespace(terrain_generator=generator),
  )
  env = SimpleNamespace(
    num_envs=2,
    device="cpu",
    scene=_Scene(
      robot=SimpleNamespace(
        data=SimpleNamespace(root_link_pos_w=torch.tensor([[4.0, 0.0, 0.3]] * 2))
      )
    ),
    common_step_counter=1,
    max_episode_length_s=20.0,
    reset_terminated=torch.zeros(2, dtype=torch.bool),
    command_manager=SimpleNamespace(
      get_command=lambda _name: torch.tensor([[1.0, 0.0, 0.0]] * 2)
    ),
  )
  env.scene.terrain = terrain
  env.scene.env_origins = torch.zeros(2, 3)
  params = dict(cfg.curriculum["terrain_levels"].params)
  # Keep terrain types fixed to isolate promotion from adaptive sampling.
  params["adaptive_type_sampling"] = False
  streak = params["sparse_success_streak_length"]
  for _ in range(streak):
    terrain_levels_delta(env, torch.arange(2), **params)
  assert torch.equal(terrain.terrain_levels, torch.tensor([9, 9]))
  for _ in range(streak):
    terrain_levels_delta(env, torch.arange(2), **params)
  assert torch.equal(terrain.terrain_levels, torch.tensor([9, 9]))


def test_delta_adaptive_sampling_emphasizes_weaker_terrain() -> None:
  num_envs = 8
  terrain = SimpleNamespace(
    terrain_levels=torch.zeros(num_envs, dtype=torch.long),
    terrain_types=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
    terrain_origins=torch.zeros(10, 2, 3),
    env_origins=torch.zeros(num_envs, 3),
    max_terrain_level=10,
    cfg=SimpleNamespace(
      terrain_generator=SimpleNamespace(
        sub_terrains={
          "easy": SimpleNamespace(proportion=0.5),
          "hard": SimpleNamespace(proportion=0.5),
        }
      )
    ),
  )
  root_pos = torch.zeros(num_envs, 3)
  root_pos[:4, 0] = 1.0
  asset = SimpleNamespace(data=SimpleNamespace(root_link_pos_w=root_pos))
  scene = _Scene(robot=asset)
  scene.terrain = terrain
  scene.env_origins = torch.zeros(num_envs, 3)
  env = SimpleNamespace(
    num_envs=num_envs,
    device="cpu",
    scene=scene,
    common_step_counter=1,
    max_episode_length_s=10.0,
    reset_terminated=torch.zeros(num_envs, dtype=torch.bool),
    command_manager=SimpleNamespace(
      get_command=lambda _name: torch.full((num_envs, 3), 0.2)
    ),
  )

  result = terrain_levels_delta(
    env,
    torch.arange(num_envs),
    command_name="twist",
    success_distance_scale=0.5,
    min_success_distance=0.5,
    max_success_distance=1.0,
    success_streak_length=1,
    failure_streak_length=1,
    adaptive_type_sampling=True,
    adaptive_warmup_episodes=0,
    type_success_ema_alpha=0.5,
  )

  assert result["hard_success_ema"] < result["easy_success_ema"]
  assert result["hard_sampling_probability"] > result["easy_sampling_probability"]
  assert torch.isclose(
    result["easy_sampling_probability"] + result["hard_sampling_probability"],
    torch.tensor(1.0),
  )


def test_high_wall_curriculum_requires_root_center_beyond_far_face_and_margin() -> None:
  num_envs = 3
  terrain = SimpleNamespace(
    terrain_levels=torch.zeros(num_envs, dtype=torch.long),
    terrain_types=torch.zeros(num_envs, dtype=torch.long),
    terrain_origins=torch.zeros(4, 1, 3),
    env_origins=torch.zeros(num_envs, 3),
    max_terrain_level=4,
    cfg=SimpleNamespace(
      terrain_generator=SimpleNamespace(
        size=(8.0, 8.0),
        sub_terrains={
          "high_wall": SimpleNamespace(
            proportion=1.0, wall_thickness=0.08, spawn_x_fraction=0.25
          )
        },
      )
    ),
  )
  root_pos = torch.zeros(num_envs, 3)
  # Wall far edge is 2.04 m from spawn origin; with a 0.4 m margin,
  # the root center must reach x >= 2.44 m.
  root_pos[:, 0] = torch.tensor([2.20, 2.439, 2.44])
  env = SimpleNamespace(
    num_envs=num_envs,
    device="cpu",
    scene=_Scene(robot=SimpleNamespace(data=SimpleNamespace(root_link_pos_w=root_pos))),
    common_step_counter=1,
    max_episode_length_s=10.0,
    reset_terminated=torch.zeros(num_envs, dtype=torch.bool),
    command_manager=SimpleNamespace(
      get_command=lambda _name: torch.tensor([[1.0, 0.0, 0.0]]).repeat(num_envs, 1)
    ),
  )
  env.scene.terrain = terrain
  env.scene.env_origins = torch.zeros(num_envs, 3)

  terrain_levels_delta(
    env,
    torch.arange(num_envs),
    command_name="twist",
    sparse_terrain_names=("high_wall",),
    sparse_success_streak_length=1,
    sparse_failure_streak_length=3,
    high_wall_crossing_clearance=0.4,
  )

  assert torch.equal(terrain.terrain_levels, torch.tensor([0, 0, 1]))


def test_tire_curriculum_requires_root_center_to_clear_last_tire_row() -> None:
  num_envs = 3
  terrain = SimpleNamespace(
    terrain_levels=torch.zeros(num_envs, dtype=torch.long),
    terrain_types=torch.zeros(num_envs, dtype=torch.long),
    terrain_origins=torch.zeros(4, 1, 3),
    env_origins=torch.zeros(num_envs, 3),
    max_terrain_level=4,
    cfg=SimpleNamespace(
      terrain_generator=SimpleNamespace(
        size=(8.0, 8.0),
        sub_terrains={
          "tire_array": SimpleNamespace(
            proportion=1.0,
            tire_outer_diameter=0.55,
            tire_pitch=0.56,
            startup_zone_length=4.0,
            spawn_x_fraction=0.25,
          )
        },
      )
    ),
  )
  root_pos = torch.zeros(num_envs, 3)
  # Last tire edge is 5.955 m from the spawn reference; add the 5 cm margin.
  root_pos[:, 0] = torch.tensor([5.9, 6.004, 6.005])
  env = SimpleNamespace(
    num_envs=num_envs,
    device="cpu",
    scene=_Scene(robot=SimpleNamespace(data=SimpleNamespace(root_link_pos_w=root_pos))),
    common_step_counter=1,
    max_episode_length_s=10.0,
    reset_terminated=torch.zeros(num_envs, dtype=torch.bool),
    command_manager=SimpleNamespace(
      get_command=lambda _name: torch.tensor([[1.0, 0.0, 0.0]]).repeat(num_envs, 1)
    ),
  )
  env.scene.terrain = terrain
  env.scene.env_origins = torch.zeros(num_envs, 3)

  terrain_levels_delta(
    env,
    torch.arange(num_envs),
    command_name="twist",
    sparse_terrain_names=("tire_array",),
    sparse_success_streak_length=1,
    sparse_failure_streak_length=3,
    tire_array_crossing_clearance=0.05,
  )

  assert torch.equal(terrain.terrain_levels, torch.tensor([0, 0, 1]))


def test_flat_tiles_neither_promote_nor_demote_the_curriculum() -> None:
  num_envs = 4
  terrain = SimpleNamespace(
    terrain_levels=torch.tensor([5, 5, 5, 5]),
    # Envs 0 and 2 sit on flat ground, envs 1 and 3 on a stair patch.
    terrain_types=torch.tensor([0, 1, 0, 1]),
    terrain_origins=torch.zeros(10, 2, 3),
    env_origins=torch.zeros(num_envs, 3),
    max_terrain_level=10,
    cfg=SimpleNamespace(
      terrain_generator=SimpleNamespace(
        size=(8.0, 8.0),
        sub_terrains={
          "flat": SimpleNamespace(proportion=0.5),
          "pyramid_stairs": SimpleNamespace(proportion=0.5),
        },
      )
    ),
  )
  root_pos = torch.zeros(num_envs, 3)
  # Envs 0 and 1 clear the success distance, envs 2 and 3 make no progress.
  root_pos[:, 0] = torch.tensor([5.0, 5.0, 0.0, 0.0])
  env = SimpleNamespace(
    num_envs=num_envs,
    device="cpu",
    scene=_Scene(robot=SimpleNamespace(data=SimpleNamespace(root_link_pos_w=root_pos))),
    common_step_counter=1,
    max_episode_length_s=10.0,
    reset_terminated=torch.zeros(num_envs, dtype=torch.bool),
    command_manager=SimpleNamespace(
      get_command=lambda _name: torch.tensor([[1.0, 0.0, 0.0]]).repeat(num_envs, 1)
    ),
  )
  env.scene.terrain = terrain
  env.scene.env_origins = torch.zeros(num_envs, 3)

  result = terrain_levels_delta(
    env,
    torch.arange(num_envs),
    command_name="twist",
    success_distance_scale=0.5,
    min_success_distance=0.5,
    max_success_distance=1.0,
    success_streak_length=1,
    failure_streak_length=1,
    flat_terrain_name="flat",
  )

  # Stairs promote on success and demote on failure; flat tiles stay put.
  assert torch.equal(terrain.terrain_levels, torch.tensor([5, 6, 5, 4]))
  # Flat success still feeds the per-type statistics used by adaptive sampling.
  assert result["flat_success_ema"] > 0.0


def _two_skill_env(types, levels):
  count = len(types)
  terrain = SimpleNamespace(
    terrain_levels=torch.tensor(levels),
    terrain_types=torch.tensor(types),
    terrain_origins=torch.zeros(10, 2, 3),
    env_origins=torch.zeros(count, 3),
    max_terrain_level=10,
    cfg=SimpleNamespace(
      terrain_generator=SimpleNamespace(
        sub_terrains={
          "stairs": SimpleNamespace(proportion=0.5),
          "gap": SimpleNamespace(proportion=0.5),
        },
      )
    ),
  )
  scene = _Scene(
    robot=SimpleNamespace(
      data=SimpleNamespace(
        root_link_pos_w=torch.zeros(count, 3),
      )
    )
  )
  scene.terrain = terrain
  scene.env_origins = torch.zeros(count, 3)
  return SimpleNamespace(
    num_envs=count,
    device="cpu",
    scene=scene,
    common_step_counter=1,
    max_episode_length_s=20.0,
    reset_terminated=torch.zeros(count, dtype=torch.bool),
    command_manager=SimpleNamespace(
      get_command=lambda _: torch.tensor([[0.5, 0.0, 0.0]] * count)
    ),
  )


def _skill_params():
  return dict(
    command_name="twist",
    sparse_terrain_names=("stairs", "gap"),
    sparse_max_level=9,
    sparse_success_streak_length=3,
    sparse_failure_streak_length=4,
    randomize_completed_levels=True,
  )


def test_completed_sparse_courses_revisit_rows_without_changing_type(monkeypatch):
  env = _two_skill_env([0] * 10 + [1] * 10, [9] * 20)
  terrain = env.scene.terrain
  terrain.terrain_origins[..., 0] = torch.arange(10)[:, None] * 10.0
  env.scene["robot"].data.root_link_pos_w[:, 0] = 3.0
  calls = []

  def draw_rows(high, size, device):
    calls.append(high)
    return torch.arange(size[0], device=device) % high

  monkeypatch.setattr(torch, "randint", draw_rows)
  for _ in range(2):
    terrain_levels_delta(env, torch.arange(20), **_skill_params())
    assert terrain.terrain_levels.tolist() == [9] * 20
    assert not calls
  terrain_levels_delta(env, torch.arange(20), **_skill_params())
  assert calls == [10]
  assert terrain.terrain_levels.tolist() == list(range(10)) * 2
  assert terrain.terrain_types.tolist() == [0] * 10 + [1] * 10
  torch.testing.assert_close(
    terrain.env_origins,
    terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types],
  )
  ids = torch.arange(20)
  assert torch.equal(
    env._delta_terrain_state[ids, terrain.terrain_types, 0], terrain.terrain_levels
  )
  assert not env._delta_terrain_state[ids, terrain.terrain_types, 1:].any()
  # A replay at level zero starts a fresh streak, not an immediate promotion.
  env.scene["robot"].data.root_link_pos_w[:] = terrain.env_origins
  env.scene["robot"].data.root_link_pos_w[:, 0] += 3.0
  env.scene.env_origins = terrain.env_origins
  terrain_levels_delta(env, ids, **_skill_params())
  assert terrain.terrain_levels.tolist() == list(range(10)) * 2
  assert env._delta_success_streak.tolist() == [1] * 20


@pytest.mark.parametrize("cap", [0, 3, 9])
def test_completion_recycling_respects_caps_and_requires_success(monkeypatch, cap):
  env = _two_skill_env([0, 1], [cap, cap])
  env.scene["robot"].data.root_link_pos_w[0, 0] = 3.0
  calls = []

  def draw_low_row(high, size, device):
    calls.append(high)
    return torch.zeros(size, device=device, dtype=torch.long)

  monkeypatch.setattr(torch, "randint", draw_low_row)
  params = _skill_params() | dict(sparse_max_level=cap)
  for _ in range(4):
    terrain_levels_delta(env, torch.arange(2), **params)
  assert calls == [cap + 1]
  assert env.scene.terrain.terrain_levels.tolist() == [0, max(0, cap - 1)]
  assert env.scene.terrain.terrain_types.tolist() == [0, 1]


def test_fixed_stairs_success_cannot_promote_gap_and_gap_failures_demote():
  env = _two_skill_env([0, 1], [4, 4])
  env.scene["robot"].data.root_link_pos_w[0, 0] = 3.0
  for _ in range(4):
    terrain_levels_delta(env, torch.arange(2), **_skill_params())
  assert env.scene.terrain.terrain_types.tolist() == [0, 1]
  assert env.scene.terrain.terrain_levels.tolist() == [5, 3]


def test_adaptive_switch_does_not_lend_stairs_level_to_unvisited_gap(monkeypatch):
  env = _two_skill_env([0], [8])
  env.scene["robot"].data.root_link_pos_w[0, 0] = 3.0
  params = _skill_params() | dict(
    adaptive_type_sampling=True,
    adaptive_warmup_episodes=0,
  )
  for next_type in (0, 0, 1):
    monkeypatch.setattr(
      torch, "multinomial", lambda *a, chosen=next_type, **k: torch.tensor([chosen])
    )
    terrain_levels_delta(env, torch.tensor([0]), **params)
  assert env._delta_terrain_state[0, 0, 0] == 9
  assert env._delta_type_episode_count[1] == 0
  assert env.scene.terrain.terrain_types.tolist() == [1]
  assert env.scene.terrain.terrain_levels.tolist() == [0]
  assert env._delta_success_streak.tolist() == [0]


def test_gap_failure_streak_survives_interleaved_stairs_success(monkeypatch):
  env = _two_skill_env([1], [3])
  params = _skill_params() | dict(
    adaptive_type_sampling=True,
    adaptive_warmup_episodes=0,
  )
  for visit in range(4):
    for completed, next_type in ((1, 0), (0, 1)):
      env.scene["robot"].data.root_link_pos_w[0, 0] = 0 if completed == 1 else 3
      monkeypatch.setattr(
        torch, "multinomial", lambda *a, chosen=next_type, **k: torch.tensor([chosen])
      )
      terrain_levels_delta(env, torch.tensor([0]), **params)
    assert env.scene.terrain.terrain_levels.item() == (3 if visit < 3 else 2)
  assert env._delta_terrain_state[0, 0, 0] == 1
  assert env._delta_failure_streak.item() == 0
