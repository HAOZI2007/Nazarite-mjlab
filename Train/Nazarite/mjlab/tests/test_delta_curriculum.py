from types import SimpleNamespace

import torch
from nazarite.mdp.curriculums import terrain_levels_delta


class _Scene(dict):
  pass


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
