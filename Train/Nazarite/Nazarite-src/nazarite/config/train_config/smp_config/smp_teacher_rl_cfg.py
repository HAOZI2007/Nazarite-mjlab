"""RSL-RL configuration for the closed-loop SMP reference teacher."""

from __future__ import annotations

from nazarite.config.train_config.rl_cfg import unitree_go2_normal_ppo_runner_cfg


def smp_teacher_go2_runner_cfg(
  experiment_name: str = "go2_smp_reference_teacher",
):
  """Return a conservative PPO setup for the first teacher smoke run."""
  cfg = unitree_go2_normal_ppo_runner_cfg(experiment_name=experiment_name)
  cfg.num_steps_per_env = 64
  cfg.max_iterations = 15_000
  cfg.algorithm.learning_rate = 3.0e-4
  cfg.algorithm.entropy_coef = 0.01
  cfg.wandb_tags = ("smp", "reference-teacher", "go2")
  return cfg
