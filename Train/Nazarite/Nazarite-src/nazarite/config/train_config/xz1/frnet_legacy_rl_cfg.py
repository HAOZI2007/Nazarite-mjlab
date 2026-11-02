"""Legacy XZ1 FR-Net PPO configuration, retained for old checkpoints."""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from nazarite.config.train_config.go2.frnet.frnet_rl_cfg import (
    FRNetActorCfg,
    FRNetPpoAlgorithmCfg,
)


def xz1_frnet_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=FRNetActorCfg(
            hidden_dims=(512, 256, 128),
            obs_normalization=True,
            use_vae=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.8,
                "std_type": "log",
            },
        ),
        critic=RslRlModelCfg(hidden_dims=(512, 256, 128), obs_normalization=True),
        algorithm=FRNetPpoAlgorithmCfg(
            learning_rate=1.0e-4,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            mcp_loss_coef=0.25,
            mass_loss_coef=1.0,
            contact_loss_coef=1.0,
            reconstruction_loss_coef=1.0,
            vae_kl_loss_coef=0.01,
        ),
        obs_groups={"actor": ("actor",), "critic": ("critic",)},
        experiment_name="xz1_frnet_recovery_flat",
        logger="tensorboard",
        num_steps_per_env=24,
        max_iterations=15000,
        save_interval=100,
        clip_actions=5.0,
    )
