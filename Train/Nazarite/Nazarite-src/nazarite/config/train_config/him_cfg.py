"""HIMLoco configuration for Nazarite Go2 rough-terrain traversal.

The environment deliberately reuses Nazarite's validated obstacle/stairs terrain
generator, while the policy uses the HIM history encoder copied from the
reference legged_wbc_mjlab implementation.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from mjlab.rl import (
  RslRlBaseRunnerCfg,
  RslRlModelCfg,
  RslRlPpoAlgorithmCfg,
)
from mjlab.managers.observation_manager import ObservationTermCfg
from rsl_rl.runners import HIMOnPolicyRunner

from nazarite.config.train_config.env_cfgs.go2_rough_env_cfgs import Nazarite_Velocity_Rough_Go2
from nazarite.mdp import observations as custom_observations


def Nazarite_HIM_Obstacle_Go2(play: bool = False):
  """Go2 HIM task on stairs, slopes, waves and discrete obstacles."""
  cfg = Nazarite_Velocity_Rough_Go2(play=play)
  # HIMLoco's phase signal is part of the temporal actor history.  Without it
  # the policy is asked to follow a periodic gait target whose phase is hidden.
  phase_term = ObservationTermCfg(
    func=custom_observations.phase,
    params={"period": 0.6, "command_name": "twist"},
  )
  cfg.observations["actor"].terms["phase"] = phase_term
  critic_terms = cfg.observations["critic"].terms
  # Keep phase immediately after the six proprioceptive/command terms so the
  # estimator target slice remains the first one-step observation (47 dims).
  cfg.observations["critic"].terms = {
    **{name: term for name, term in critic_terms.items() if name not in {
      "base_lin_vel", "base_height", "foot_height", "foot_air_time",
      "foot_contact", "foot_contact_forces",
    }},
    "phase": deepcopy(phase_term),
    **{name: term for name, term in critic_terms.items() if name in {
      "base_lin_vel", "base_height", "foot_height", "foot_air_time",
      "foot_contact", "foot_contact_forces",
    }},
  }
  # HIM consumes a fixed six-frame proprioceptive history.  The current
  # Actor terms are [ang_vel(3), gravity(3), q(12), dq(12), action(12),
  # command(3), phase(2)] = 47 features per frame, matching HIMLoco.
  for term in cfg.observations["actor"].terms.values():
    term.history_length = 6
  for term in cfg.observations["critic"].terms.values():
    term.history_length = 1
  cfg.observations["actor"].flatten_history_dim = True
  return cfg


@dataclass
class RslRlHimActorCfg(RslRlModelCfg):
  class_name: str = "rsl_rl.models.him_actor_model:HIMActorModel"
  hidden_dims: tuple[int, ...] = (512, 256, 128)
  distribution_cfg: dict[str, Any] | None = field(default_factory=lambda: {
    "class_name": "rsl_rl.modules.distribution:GaussianDistribution",
    "init_std": 1.0,
    "std_type": "scalar",
  })
  num_one_step_obs: int = 47
  history_size: int = 6
  history_term_dims: tuple[int, ...] = (3, 3, 12, 12, 12, 3, 2)
  history_order: str = "term_major_oldest_first"
  estimator_cfg: dict[str, Any] = field(default_factory=lambda: {
    "enc_hidden_dims": (128, 64, 16),
    "tar_hidden_dims": (128, 64),
    "num_prototype": 32,
    "temperature": 3.0,
    "sinkhorn_eps": 0.05,
    "sinkhorn_iters": 3,
  })


@dataclass
class RslRlHimAlgorithmCfg(RslRlPpoAlgorithmCfg):
  class_name: str = "rsl_rl.algorithms.him_ppo:HIMPPO"
  # The previous estimator loss rose as terrain difficulty increased.  A
  # smaller step and tighter clipping prevent latent/velocity estimates from
  # chasing highly randomized obstacle contacts.
  estimator_learning_rate: float = 3.0e-4
  estimator_max_grad_norm: float = 5.0
  estimator_obs_groups: tuple[str, ...] = ("critic",)
  # Critic starts with the seven actor terms (47 dims), then privileged linear
  # velocity is appended by the base config.
  estimator_target_slices: tuple[tuple[int, int], ...] = ((0, 47),)
  estimator_velocity_slice: tuple[int, int] = (47, 50)


@dataclass
class RslRlHimRunnerCfg(RslRlBaseRunnerCfg):
  actor: RslRlHimActorCfg = field(default_factory=RslRlHimActorCfg)
  critic: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(hidden_dims=(512, 256, 128))
  )
  algorithm: RslRlHimAlgorithmCfg = field(default_factory=RslRlHimAlgorithmCfg)
  class_name: str = "HIMOnPolicyRunner"
  experiment_name: str = "go2_him_complex_terrain"
  num_steps_per_env: int = 24
  max_iterations: int = 15_000


def unitree_go2_him_runner_cfg() -> RslRlHimRunnerCfg:
  return RslRlHimRunnerCfg(
    actor=RslRlHimActorCfg(),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128)),
    algorithm=RslRlHimAlgorithmCfg(),
    obs_groups={"actor": ("actor",), "critic": ("critic",)},
  )


__all__ = [
  "Nazarite_HIM_Obstacle_Go2",
  "RslRlHimActorCfg",
  "RslRlHimAlgorithmCfg",
  "RslRlHimRunnerCfg",
  "unitree_go2_him_runner_cfg",
  "HIMOnPolicyRunner",
]
