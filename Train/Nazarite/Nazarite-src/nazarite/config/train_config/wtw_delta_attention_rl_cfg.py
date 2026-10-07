"""PPO configuration for the WTW-history + DELTA attention task."""

from pathlib import Path
import os

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from .env_cfgs.wtw_delta_attention_env_cfg import (
    BEV_MAP_CENTER,
    BEV_MAP_EXTENT,
    BEV_VARIANTS,
)


def _default_wtw_checkpoint() -> str | None:
    repo_root = Path(__file__).resolve().parents[4]
    candidates = (
        repo_root
        / "logs/rsl_rl/go2_flat_wtw_independent/2026-09-19_23-27-56/model_14999.pt",
        repo_root
        / "logs/rsl_rl/go2_flat_wtw_independent/2026-09-11_20-39-13/model_14950.pt",
    )
    return next((str(path) for path in candidates if path.is_file()), None)


def wtw_delta_go2_runner_cfg(
    bev_variant: str | None = None,
    experiment_name: str = "go2_wtw_delta_attention",
) -> RslRlOnPolicyRunnerCfg:
    bev_variant = bev_variant or os.getenv("NAZARITE_WTW_DELTA_BEV", "25x40")
    if bev_variant not in BEV_VARIANTS:
        raise ValueError(f"Unknown BEV variant {bev_variant}")
    map_height, map_width = BEV_VARIANTS[bev_variant]
    actor = RslRlModelCfg(
        class_name="nazarite.delta.wtw_delta_attention_model:WtwDeltaAttentionModel",
        hidden_dims=(512, 256, 128),
        obs_normalization=False,
        prior_group="wtw_proprio",
        delta_map_group="delta_map",
        map_height=map_height,
        map_width=map_width,
        map_channels=3,
        map_extent=BEV_MAP_EXTENT,
        map_center=BEV_MAP_CENTER,
        residual_hidden_dims=(256, 128),
        wtw_checkpoint=_default_wtw_checkpoint(),
        # This task jointly fine-tunes the WTW history encoder and the DELTA
        # fusion head; the WTW checkpoint only supplies a stable initialization.
        freeze_wtw=False,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 0.20,
            "std_type": "log",
            # Let PPO learn the exploration scale without an artificial task-
            # specific upper/lower bound.  The initial value remains 0.20.
            "std_range": None,
        },
    )
    critic = RslRlModelCfg(
        class_name="nazarite.delta.critic_model:DeltaPrivilegedCriticModel",
        hidden_dims=(512, 256, 128),
        obs_normalization=False,
        distribution_cfg=None,
        delta_map_group="delta_privileged_map",
        map_height=map_height,
        map_width=map_width,
        map_channels=3,
        map_extent=BEV_MAP_EXTENT,
        map_center=BEV_MAP_CENTER,
    )
    return RslRlOnPolicyRunnerCfg(
        actor=actor,
        critic=critic,
        obs_groups={
            "actor": ("wtw_proprio", "delta_map"),
            "critic": ("critic_privileged", "delta_privileged_map"),
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
