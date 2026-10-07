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
