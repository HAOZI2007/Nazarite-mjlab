"""PPO configuration for downstream tasks guided by the frozen SMP prior."""

from __future__ import annotations

from nazarite.config.train_config.rl_cfg import unitree_go2_normal_ppo_runner_cfg


def smp_forward_go2_runner_cfg():
    cfg = unitree_go2_normal_ppo_runner_cfg(experiment_name="go2_smp_forward_3ddogs_1x")
    cfg.actor.distribution_cfg = {
        "class_name": "GaussianDistribution",
        "init_std": 0.30,
        "std_type": "scalar",
        "learn_std": False,
    }
    cfg.algorithm.learning_rate = 1.0e-3
    cfg.algorithm.entropy_coef = 0.0
    cfg.num_steps_per_env = 24
    cfg.max_iterations = 30_000
    cfg.save_interval = 500
    cfg.wandb_project = "smp"
    cfg.wandb_tags = ("smp", "3ddogs", "go2", "forward", "1x")
    return cfg
