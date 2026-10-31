"""Teacher-owned networks, PPO/HIM hyperparameters and training defaults."""

from dataclasses import dataclass
from pathlib import Path

from mjlab.rl import RslRlModelCfg
from nazarite.config.train_config.go2.him_rl_cfg import (
    RslRlHimActorCfg,
    RslRlHimAlgorithmCfg,
    RslRlHimRunnerCfg,
)
from nazarite.config.train_config.xz1.teacher_env_cfg import (
    XZ1_TEACHER_SCAN_HISTORY_LENGTH,
    XZ1_TEACHER_SCAN_UPDATE_PERIOD,
)


@dataclass
class XZ1TeacherActorCfg(RslRlHimActorCfg):
    class_name: str = "nazarite.rl.xz1_teacher:XZ1TerrainTeacherActor"
    scan_group: str = "teacher_scan"
    scan_history_length: int = 1
    scan_update_period: float = 0.02


@dataclass
class XZ1TeacherRunnerCfg(RslRlHimRunnerCfg):
    class_name: str = "nazarite.rl.xz1_teacher:XZ1TerrainTeacherRunner"
    initial_checkpoint: str | None = None


def xz1_teacher_runner_cfg(
    scan_history_length: int = 1, scan_update_period: float = 0.02
) -> XZ1TeacherRunnerCfg:
    """All effective teacher hyperparameters live here, not in HIM factories.

    One frame remains the default for legacy collection/student callers.
    The registered training task uses xz1_height_history_training_cfg below.
    """
    return XZ1TeacherRunnerCfg(
        actor=XZ1TeacherActorCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "rsl_rl.modules.distribution:GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
            num_one_step_obs=47,
            history_size=6,
            history_term_dims=(3, 3, 3, 2, 12, 12, 12),
            history_order="frame_major_oldest_first",
            estimator_cfg={
                "enc_hidden_dims": (128, 64, 16),
                "tar_hidden_dims": (128, 64),
                "num_prototype": 32,
                "temperature": 3.0,
                "sinkhorn_eps": 0.05,
                "sinkhorn_iters": 3,
            },
            observation_clip=100.0,
            action_clip=10.0,
            action_observation_slice=(35, 47),
            scan_group="teacher_scan",
            scan_history_length=scan_history_length,
            scan_update_period=scan_update_period,
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True
        ),
        algorithm=RslRlHimAlgorithmCfg(
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            entropy_coef=0.01,
            desired_kl=0.01,
            max_grad_norm=1.0,
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            normalize_advantage_per_mini_batch=False,
            optimizer="adam",
            estimator_learning_rate=1e-3,
            estimator_max_grad_norm=10.0,
            estimator_obs_groups=("estimator",),
            estimator_target_slices=((0, 6), (9, 47), (47, 50)),
            estimator_velocity_slice=(47, 50),
        ),
        obs_groups={"actor": ("actor",), "critic": ("critic",)},
        seed=42,
        num_steps_per_env=100,
        max_iterations=100_000,
        save_interval=100,
        clip_actions=10.0,
        numerics={
            "raw_action_warn": 20.0,
            "raw_observation_abort": 1000.0,
            "history_steps": 8,
        },
        logger="tensorboard",
        experiment_name="xz1_wmp_him_teacher",
        run_name="",
        resume=False,
        load_run=".*",
        load_checkpoint="model_.*.pt",
        initial_checkpoint=None,
    )


def xz1_height_history_training_cfg() -> XZ1TeacherRunnerCfg:
    """One-command single-scan training, initialized from the XZ1 HIM policy."""
    cfg = xz1_teacher_runner_cfg(
        scan_history_length=XZ1_TEACHER_SCAN_HISTORY_LENGTH,
        scan_update_period=XZ1_TEACHER_SCAN_UPDATE_PERIOD,
    )
    cfg.initial_checkpoint = str(
        Path(__file__).resolve().parents[5]
        / "logs/rsl_rl/xz1_him_complex"
        / "2026-10-25_22-04-31/model_31300.pt"
    )
    cfg.max_iterations = 100_000
    cfg.run_name = "height_single_10hz_from_him"
    return cfg
