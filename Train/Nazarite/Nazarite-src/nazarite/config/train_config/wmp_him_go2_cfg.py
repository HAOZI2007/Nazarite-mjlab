"""Runner configuration for the independent Go2 WMP + HIM task."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mjlab.rl import RslRlModelCfg
from nazarite.config.train_config.him_cfg import (
    RslRlHimActorCfg,
    RslRlHimAlgorithmCfg,
    RslRlHimRunnerCfg,
)


@dataclass
class RslRlWMPHIMActorCfg(RslRlHimActorCfg):
    class_name: str = "nazarite.rl.wmp.actor:WMPHIMActorModel"
    world_model_feature_group: str = "world_model_feature"
    world_model_feature_dim: int = 512
    visual_latent_dim: int = 32
    visual_hidden_dim: int = 128


@dataclass
class RslRlWMPHIMRunnerCfg(RslRlHimRunnerCfg):
    actor: RslRlHimActorCfg = field(default_factory=RslRlWMPHIMActorCfg)
    class_name: str = "nazarite.rl.wmp.runner:WMPHIMOnPolicyRunner"
    experiment_name: str = "go2_wmp_him_complex"
    num_steps_per_env: int = 64
    # Hard floor on the adaptive learning rate. rsl-rl's own clamp is 1e-5, and
    # the 2026-10-18 run hit it after ~2500 iterations: the policy stopped
    # updating and the crawl it had settled into became permanent.
    min_learning_rate: float = 1.0e-4
    world_model: dict[str, Any] = field(
        default_factory=lambda: {
            "image_height": 64,
            "image_width": 64,
            "image_channels": 1,
            "prop_dim": 33,
            "action_dim": 12,
            # The world model steps at the depth-camera rate, so five policy
            # actions are folded into one 60-dim transition (bytedance/WMP
            # runner.py does the same: `num_actions *= update_interval`).
            "action_chunk": 5,
            "cnn_depth": 32,
            "prop_hidden_dim": 1024,
            "prop_layers": 5,
            "deter_dim": 512,
            "stoch_groups": 32,
            "stoch_classes": 32,
            "rssm_hidden_dim": 512,
            "kl_free": 1.0,
            "dyn_scale": 0.5,
            "rep_scale": 0.1,
            "learning_rate": 1.0e-4,
            "grad_clip": 1000.0,
            "reward_loss_scale": 0.0,
            # Budget copied from bytedance/WMP (train_steps_per_iter 10,
            # batch_size 16, batch_length 64, train_start_steps 10000). At one
            # update of 8x32 per iteration the world model saw 40x less data than
            # the policy it was supposed to serve, so its feature stayed near
            # noise: `image_loss` plateaued at 0.02 and `kl_dyn_rep` sat exactly
            # on the free-bits ceiling of 2.0 for the whole 2026-10-17 run. The
            # capacity covers a 20 s episode at the 10 Hz world-model rate
            # (1000/5 + margin).
            "capacity_per_env": 256,
            "train_start_frames": 10000,
            # 10 is the reference's absolute number, but it collects with 4096
            # environments against our 1024, so copying it trains the world model
            # 4x harder per collected transition (0.78 of the buffer per iteration
            # vs their 0.195). Three updates restores that ratio and costs ~4.3 s
            # of learning time per iteration instead of 14.2 s; the 2026-10-19 run
            # had already reached an image loss of 0.011 after 600 iterations at
            # the higher budget, so the extra updates buy little.
            "updates_per_iteration": 3,
            "batch_size": 16,
            "batch_length": 64,
            # One burst of extra updates the first time the replay is warm, so
            # the policy stops consuming a stale feature as early as possible.
            "pretrain_updates": 100,
        }
    )
    # Adversarial motion prior, replacing the hand-tuned gait rewards. The
    # discriminator scores 30-dimensional proprioceptive transitions against the
    # bundled Go2 trot clips; its score becomes an additive style reward.
    amp: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "motion_dir": None,
            "hidden_dims": (1024, 512),
            # coefficient x clamp(1 - (d-1)^2 / 4, 0), added *without* a dt
            # factor, exactly like the reference. Our tracking term is ~0.028 per
            # step and the reference's is ~0.030 with a coefficient of 0.01, so
            # 0.02 buys twice the reference's share (~70% of tracking at d = 1).
            "style_coefficient": 0.02,
            # 256 = 128 reference + 128 policy transitions per discriminator step,
            # matching the reference's 1:1 mix.
            "batch_size": 256,
            "updates_per_iteration": 1,
            "learning_rate": 1.0e-4,
            # Gradient penalty on reference transitions only, as in the reference.
            "grad_penalty": 10.0,
            "seed": None,
        }
    )


def go2_wmp_him_runner_cfg(
    experiment_name: str = "go2_wmp_him_complex",
) -> RslRlWMPHIMRunnerCfg:
    actor = RslRlWMPHIMActorCfg(
        # 45, not 47: the gait clock (phase) is gone, because swing timing is the
        # discriminator's job now and a hard-coded 0.6 s phase fights it.
        num_one_step_obs=45,
        history_size=6,
        history_term_dims=(3, 3, 3, 12, 12, 12),
        action_observation_slice=(33, 45),
        obs_normalization=True,
    )
    algorithm = RslRlHimAlgorithmCfg(
        estimator_obs_groups=("estimator",),
        estimator_target_slices=((0, 6), (9, 45), (45, 48)),
        estimator_velocity_slice=(45, 48),
        # 0.01 let the action std grow monotonically from 0.88 to 1.30 in the
        # 2026-10-17_22-32-26 run (every earlier run settled at 0.50-0.55), i.e.
        # 0.33 rad of noise on every joint: the gait turns into a shuffle and the
        # value loss blows up to 0.54 trying to fit the resulting returns. 0.004
        # keeps exploration but stops the entropy bonus from overriding the
        # policy gradient.
        entropy_coef=0.005,
        # 0.01 was exceeded often enough to drive the learning rate to its floor
        # (1e-5) within ~2500 iterations of a 3592-iteration run. A wider band
        # keeps the rate in the useful range instead of trading all plasticity
        # for a KL bound the run never settled under.
        desired_kl=0.025,
    )
    return RslRlWMPHIMRunnerCfg(
        actor=actor,
        critic=RslRlModelCfg(hidden_dims=(512, 256, 128), obs_normalization=True),
        algorithm=algorithm,
        obs_groups={"actor": ("actor",), "critic": ("critic", "world_model_feature")},
        logger="tensorboard",
        experiment_name=experiment_name,
    )


__all__ = ["RslRlWMPHIMActorCfg", "RslRlWMPHIMRunnerCfg", "go2_wmp_him_runner_cfg"]
