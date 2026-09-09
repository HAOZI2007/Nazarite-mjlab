"""Online SMP guidance reward evaluated on physical Go2 trajectories."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from .features import MotionFeatureBuffer
from .runtime import DiffusionErrorNormalizer

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _update_motion_buffer(env: ManagerBasedRlEnv) -> MotionFeatureBuffer:
    robot = env.scene["robot"]
    buffer: MotionFeatureBuffer = env._smp_buffer  # type: ignore[attr-defined]
    site_ids: torch.Tensor = env._smp_foot_site_ids  # type: ignore[attr-defined]
    joint_ids: torch.Tensor = env._smp_joint_ids  # type: ignore[attr-defined]
    buffer.update(
        robot.data.root_link_pos_w - env.scene.env_origins,
        robot.data.root_link_quat_w,
        robot.data.root_link_lin_vel_w,
        robot.data.root_link_ang_vel_w,
        robot.data.site_pos_w[:, site_ids] - env.scene.env_origins[:, None],
        robot.data.joint_pos[:, joint_ids],
    )
    return buffer


def smp_guidance_reward(
    env: ManagerBasedRlEnv,
    fixed_timesteps: tuple[int, ...] = (8, 15, 22),
    weight: float = 4.0,
    normalize_error: bool = True,
) -> torch.Tensor:
    """Compute the frozen-prior SDS reward for each physical environment."""
    model, scheduler, q_low, q_high, _, _ = env._smp_bundle  # type: ignore[attr-defined]
    normalizer: DiffusionErrorNormalizer = env._smp_error_normalizer  # type: ignore[attr-defined]
    features = _update_motion_buffer(env).compute_features()
    normalized = 2.0 * (features - q_low) / (q_high - q_low + 1.0e-8) - 1.0
    total_error = torch.zeros(env.num_envs, device=env.device)
    total_raw_error = torch.zeros_like(total_error)
    with torch.no_grad():
        for timestep in fixed_timesteps:
            if not 0 <= timestep < scheduler.num_timesteps:
                raise ValueError(
                    f"SMP timestep {timestep} outside [0,{scheduler.num_timesteps})"
                )
            timestep_batch = torch.full(
                (env.num_envs,), timestep, dtype=torch.long, device=env.device
            )
            noise = torch.randn_like(normalized)
            noisy = scheduler.add_noise(normalized, noise, timestep_batch)
            predicted_noise = model(noisy, timestep_batch)
            error = torch.square(predicted_noise - noise).mean(dim=(-1, -2))
            total_raw_error += error
            total_error += (
                normalizer.update_and_normalize(timestep, error)
                if normalize_error
                else error
            )
    env._smp_raw_error = total_raw_error / len(fixed_timesteps)  # type: ignore[attr-defined]
    reward = torch.exp(-weight * total_error / len(fixed_timesteps))
    env._smp_guidance_reward = reward  # type: ignore[attr-defined]
    return reward


TaskRewardTerm = tuple[Callable[..., torch.Tensor], float, dict[str, Any]]


def task_smp_product(
    env: ManagerBasedRlEnv,
    task_terms: tuple[TaskRewardTerm, ...],
    fixed_timesteps: tuple[int, ...] = (8, 15, 22),
    smp_weight: float = 4.0,
) -> torch.Tensor:
    """Reference-reproduction reward: weighted task reward multiplied by SMP."""
    task_reward = torch.zeros(env.num_envs, device=env.device)
    for function, term_weight, parameters in task_terms:
        task_reward += term_weight * function(env, **parameters)
    env._smp_task_reward = task_reward  # type: ignore[attr-defined]
    return task_reward * smp_guidance_reward(
        env, fixed_timesteps=fixed_timesteps, weight=smp_weight
    )


def smp_raw_error_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env._smp_raw_error  # type: ignore[attr-defined]


def smp_guidance_reward_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env._smp_guidance_reward  # type: ignore[attr-defined]


def smp_task_reward_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env._smp_task_reward  # type: ignore[attr-defined]
