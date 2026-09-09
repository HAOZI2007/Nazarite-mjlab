"""Go2 forward locomotion trained with a frozen 3DDogs SMP prior."""

from __future__ import annotations

import math
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from nazarite.config.train_config.env_cfgs.go2_env_cfgs import (
    Nazarite_Velocity_Flat_Go2_No_WTW,
)
from nazarite.config.train_config.train_algorithm.smp.prior.events import (
    gsi_reset,
    initialize_smp_prior,
    refresh_gsi_pool,
    smp_gsi_eligible_fraction_metric,
)
from nazarite.config.train_config.train_algorithm.smp.prior.guidance import (
    smp_guidance_reward_metric,
    smp_raw_error_metric,
    smp_task_reward_metric,
    task_smp_product,
)
from nazarite.mdp import rewards as custom_rewards

_PROJECT_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_GO2_SMP_PRIOR = (
    _PROJECT_ROOT / "tools/smp_dataset/smp_prior/go2_3ddogs_1x/pretrained.pt"
)


def make_smp_forward_go2_env_cfg(
    play: bool = False,
    prior_checkpoint: str | None = None,
) -> ManagerBasedRlEnvCfg:
    """Create the first real SMP task: forward velocity × frozen motion prior."""
    cfg = Nazarite_Velocity_Flat_Go2_No_WTW(play=play)
    checkpoint = str(
        Path(prior_checkpoint).expanduser()
        if prior_checkpoint is not None
        else DEFAULT_GO2_SMP_PRIOR
    )

    cfg.commands["twist"] = UniformVelocityCommandCfg(
        entity_name="robot",
        resampling_time_range=(3.0, 8.0),
        heading_command=False,
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        rel_forward_envs=1.0,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.3, 2.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=None,
        ),
    )
    cfg.curriculum = {}
    # GSI can start from transient self-contacting poses, especially while the
    # prior is still under-trained.  The baseline's 35-contact budget is too
    # small for those states and makes MuJoCo-Warp abort before PPO can reject
    # them through the termination logic.
    cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 128)

    # Keep the Go2 locomotion stabilizers from the baseline.  The previous
    # version replaced the complete reward dictionary with only the gated
    # velocity term, which left no explicit air-time, foot-slip, posture,
    # torque, or action-smoothness signal.  That makes a low-lift, high-
    # frequency shuffling gait an easy local optimum.
    cfg.rewards.pop("track_linear_velocity", None)
    # Do not pull the policy back toward the nominal joint pose while learning
    # the motion-prior gait.  Keep stand_pose separately: it only acts as a
    # standing stabilizer and is not the variable-posture walking reward.
    cfg.rewards.pop("pose", None)
    cfg.rewards["track_angular_velocity"].weight = 0.2
    cfg.rewards["upright"].weight = 0.2
    cfg.rewards["base_height"].weight = 0.2
    cfg.rewards["air_time"].weight = 1.0
    cfg.rewards["task_smp_product"] = RewardTermCfg(
        func=task_smp_product,
        weight=5.0,
        params={
            "task_terms": (
                (
                    custom_rewards.track_linear_velocity,
                    1.0,
                    {
                        "command_name": "twist",
                        "std": math.sqrt(0.25),
                        "asset_cfg": SceneEntityCfg("robot"),
                    },
                ),
            ),
            "fixed_timesteps": (8, 15, 22),
            "smp_weight": 4.0,
        },
    )

    cfg.events["initialize_smp_prior"] = EventTermCfg(
        func=initialize_smp_prior,
        mode="startup",
        params={
            "checkpoint_path": checkpoint,
            "gsi_pool_size": 256 if play else 4096,
            "gsi_batch_size": 256 if play else 1024,
            "gsi_min_root_lin_vel_x": 0.0,
            "compile_model": not play,
            "compile_mode": "max-autotune" if not play else None,
        },
    )
    # These are appended after the generic base/joint reset terms, so GSI owns
    # the final reset state as required by the SMP algorithm.
    cfg.events["smp_gsi_reset"] = EventTermCfg(func=gsi_reset, mode="reset")
    if not play:
        cfg.events["refresh_smp_gsi_pool"] = EventTermCfg(
            func=refresh_gsi_pool,
            mode="step",
            params={"num_samples": 1024, "step_interval": 2400},
        )
    cfg.metrics.update(
        {
            "smp_raw_error": MetricsTermCfg(func=smp_raw_error_metric),
            "smp_guidance_reward": MetricsTermCfg(func=smp_guidance_reward_metric),
            "smp_task_reward": MetricsTermCfg(func=smp_task_reward_metric),
            "smp_gsi_eligible_fraction": MetricsTermCfg(
                func=smp_gsi_eligible_fraction_metric
            ),
        }
    )
    return cfg


def Nazarite_SMP_Forward_Go2(play: bool = False) -> ManagerBasedRlEnvCfg:
    return make_smp_forward_go2_env_cfg(play=play)
