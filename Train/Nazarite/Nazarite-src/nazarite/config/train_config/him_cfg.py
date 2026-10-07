"""HIMLoco policy and runner configuration for Nazarite's Go2 task."""

from dataclasses import dataclass, field
from typing import Any

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)
from nazarite.config.train_config.env_cfgs.him_complex_env_cfg import (
  Nazarite_HIM_Complex_Terrain_Go2,
)
from nazarite.rl.him_runner import HIMOnPolicyRunner


def Nazarite_HIM_Obstacle_Go2(play: bool = False):
  """Backward-compatible name for the standalone HIM complex task."""
  return Nazarite_HIM_Complex_Terrain_Go2(play=play)


@dataclass
class RslRlHimActorCfg(RslRlModelCfg):
  class_name: str = "rsl_rl.models.him_actor_model:HIMActorModel"
  hidden_dims: tuple[int, ...] = (512, 256, 128)
  distribution_cfg: dict[str, Any] | None = field(default_factory=lambda: {
    "class_name": "rsl_rl.modules.distribution:GaussianDistribution",
    "init_std": 1.0,
    "std_type": "scalar",
  })
  num_one_step_obs: int = 49
  history_size: int = 6
  history_term_dims: tuple[int, ...] = (3, 3, 3, 2, 2, 12, 12, 12)
  history_order: str = "frame_major_oldest_first"
  estimator_cfg: dict[str, Any] = field(default_factory=lambda: {
    "enc_hidden_dims": (128, 64, 16),
    "tar_hidden_dims": (128, 64),
    "num_prototype": 32,
    "temperature": 3.0,
    "sinkhorn_eps": 0.05,
    "sinkhorn_iters": 3,
  })
  observation_clip: float = 100.0
  action_clip: float = 10.0
  action_observation_slice: tuple[int, int] = (37, 49)


@dataclass
class RslRlHimAlgorithmCfg(RslRlPpoAlgorithmCfg):
  class_name: str = "rsl_rl.algorithms.him_ppo:HIMPPO"
  estimator_learning_rate: float = 1.0e-3
  estimator_max_grad_norm: float = 10.0
  estimator_obs_groups: tuple[str, ...] = ("critic",)
  # Critic layout: actor frame excluding command, plus privileged velocity.
  estimator_target_slices: tuple[tuple[int, int], ...] = ((0, 6), (9, 49), (49, 52))
  estimator_velocity_slice: tuple[int, int] = (49, 52)


@dataclass
class RslRlHimRunnerCfg(RslRlOnPolicyRunnerCfg):
  actor: RslRlHimActorCfg = field(default_factory=RslRlHimActorCfg)
  critic: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(hidden_dims=(512, 256, 128))
  )
  algorithm: RslRlHimAlgorithmCfg = field(default_factory=RslRlHimAlgorithmCfg)
  class_name: str = "nazarite.rl.him_runner:HIMOnPolicyRunner"
  experiment_name: str = "go2_him"
  num_steps_per_env: int = 100
  max_iterations: int = 100_000
  save_interval: int = 100
  clip_actions: float = 10.0
  numerics: dict[str, Any] = field(default_factory=lambda: {
    "raw_action_warn": 20.0,
    "raw_observation_abort": 1000.0,
    "history_steps": 8,
  })


def unitree_go2_him_runner_cfg() -> RslRlHimRunnerCfg:
  return RslRlHimRunnerCfg(
    actor=RslRlHimActorCfg(),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128)),
    algorithm=RslRlHimAlgorithmCfg(),
    obs_groups={"actor": ("actor",), "critic": ("critic",)},
    logger="tensorboard",
  )


__all__ = [
  "HIMOnPolicyRunner",
  "Nazarite_HIM_Obstacle_Go2",
  "RslRlHimActorCfg",
  "RslRlHimAlgorithmCfg",
  "RslRlHimRunnerCfg",
  "unitree_go2_him_runner_cfg",
]
