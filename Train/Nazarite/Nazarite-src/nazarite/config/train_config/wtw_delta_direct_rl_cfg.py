"""PPO configuration for the direct-action WTW+DELTA task."""

from pathlib import Path

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def _default_wtw_checkpoint() -> str | None:
    repo_root = Path(__file__).resolve().parents[4]
    candidates = (
        repo_root
        / "logs/rsl_rl/go2_flat_wtw_independent/2026-09-19_23-27-56/model_14999.pt",
        repo_root
        / "logs/rsl_rl/go2_flat_wtw_independent/2026-09-11_20-39-13/model_14950.pt",
    )
    return next((str(path) for path in candidates if path.is_file()), None)


def wtw_delta_direct_go2_runner_cfg(
    experiment_name: str = "go2_wtw_delta_direct",
) -> RslRlOnPolicyRunnerCfg:
    actor = RslRlModelCfg(
        class_name="nazarite.delta.direct_action_model:WtwDeltaDirectActionModel",
        hidden_dims=(512, 256, 128),
        obs_normalization=False,
        prior_group="wtw_proprio",
        delta_map_group="delta_map",
        delta_proprio_group="delta_proprio",
        map_height=16,
        map_width=26,
        map_channels=5,
        map_extent=(2.0, 1.2),
        map_center=(1.0, 0.0),
        forward_x_threshold=0.0,
        residual_hidden_dims=(256, 128),
        # Unused by direct action fusion, retained only for parent API
        # compatibility while loading the WTW prior.
        residual_scale=1.0,
        residual_gate_bias=0.0,
        wtw_checkpoint=_default_wtw_checkpoint(),
        freeze_wtw=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 0.20,
            "std_type": "log",
            "std_range": (0.08, 0.25),
        },
    )
    critic = RslRlModelCfg(
        class_name="nazarite.delta.critic_model:DeltaPrivilegedCriticModel",
        hidden_dims=(512, 256, 128),
        obs_normalization=False,
        distribution_cfg=None,
        delta_map_group="delta_privileged_map",
        delta_proprio_group="delta_proprio_critic",
        map_height=16,
        map_width=26,
        map_channels=5,
        map_extent=(2.0, 1.2),
        map_center=(1.0, 0.0),
        forward_x_threshold=0.0,
    )
    return RslRlOnPolicyRunnerCfg(
        actor=actor,
        critic=critic,
        obs_groups={
            "actor": ("wtw_proprio", "delta_proprio", "delta_map"),
            "critic": (
                "critic_privileged",
                "delta_proprio_critic",
                "delta_privileged_map",
            ),
        },
        algorithm=RslRlPpoAlgorithmCfg(
            learning_rate=2.0e-4,
            schedule="adaptive",
            num_learning_epochs=5,
            num_mini_batches=8,
            gamma=0.99,
            lam=0.95,
            clip_param=0.2,
            entropy_coef=0.001,
        ),
        experiment_name=experiment_name,
        num_steps_per_env=24,
        max_iterations=30_000,
        save_interval=50,
        clip_actions=5.0,
    )
