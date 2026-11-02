# SPDX-License-Identifier: Apache-2.0
# Adapted from mjlab_playground/getup/config/go1/rl_cfg.py.
"""Plain PPO for the XZ1 Playground getup port; no MCP/VAE auxiliary branch."""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def xz1_frnet_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=False,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "log",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="xz1_getup_flat",
        logger="tensorboard",
        save_interval=100,
        num_steps_per_env=24,
        max_iterations=100_000,
    )
