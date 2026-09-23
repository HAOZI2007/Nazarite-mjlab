from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def delta_go2_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """PPO configuration for DELTA on the rough-terrain Go2 task."""
    actor_model = {
        "class_name": "nazarite.delta.rsl_model:DeltaModel",
        "hidden_dims": (256, 128),
        "obs_normalization": False,
        "proprio_dim": 45,
        "map_height": 16,
        "map_width": 26,
    }
    critic_model = {
        **actor_model,
        # DELTA privileged critic:
        # 45 actor terms + base_lin_vel(3) + base_height(1) +
        # foot_height(4) + foot_air_time(4) + foot_contact(4) +
        # foot_contact_forces(12) = 73 dimensions. The critic's joint_pos term
        # replaces the actor term in-place and therefore does not add dimensions.
        "proprio_dim": 73,
    }
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            **actor_model,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "log",
            },
        ),
        critic=RslRlModelCfg(**critic_model),
        algorithm=RslRlPpoAlgorithmCfg(
            learning_rate=1.0e-4,
            num_learning_epochs=5,
            num_mini_batches=4,
            gamma=0.99,
            lam=0.95,
            clip_param=0.2,
            entropy_coef=0.005,
        ),
        experiment_name="go2_delta_prototype",
        num_steps_per_env=24,
        # Matches DELTA's 30,000-iteration training schedule.
        max_iterations=30_000,
        save_interval=50,
    )
