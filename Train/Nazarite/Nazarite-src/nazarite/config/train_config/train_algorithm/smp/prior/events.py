"""SMP startup and generative-state-initialization events for Go2."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch

from mjlab.utils.lab_api.math import quat_apply, quat_mul, yaw_quat

from .feature_to_state import rotation_6d_to_quaternion, split_features
from .features import (
    GO2_SMP_FEATURE_DIM,
    GO2_SMP_FOOT_SITE_NAMES,
    GO2_SMP_JOINT_NAMES,
    MotionFeatureBuffer,
)
from .model import DiffusionDenoiser
from .runtime import DiffusionErrorNormalizer, load_prior, sample_prior

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _maybe_compile_model(
    model: DiffusionDenoiser,
    compile_model: bool,
    compile_mode: str | None,
) -> DiffusionDenoiser:
    if not compile_model:
        return model
    torch.set_float32_matmul_precision("high")
    try:
        import torch._inductor.config as inductor_config

        inductor_config.shape_padding = False
    except ImportError:
        pass
    if compile_mode is not None:
        return cast(
            DiffusionDenoiser,
            torch.compile(model, fullgraph=True, mode=compile_mode),
        )
    return cast(DiffusionDenoiser, torch.compile(model, fullgraph=True))


@torch.no_grad()
def initialize_smp_prior(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None = None,
    checkpoint_path: str = "",
    gsi_pool_size: int = 1024,
    gsi_batch_size: int = 256,
    gsi_min_root_lin_vel_x: float | None = None,
    compile_model: bool = False,
    compile_mode: str | None = None,
) -> None:
    """Load the frozen prior, allocate online history, and generate a GSI pool."""
    del env_ids
    if not checkpoint_path:
        raise ValueError("SMP checkpoint_path must not be empty")
    if gsi_pool_size <= 0 or gsi_batch_size <= 0:
        raise ValueError("GSI pool and batch sizes must be positive")
    bundle = load_prior(checkpoint_path, env.device)
    model, scheduler, q_low, q_high, feature_dim, window_size = bundle
    if feature_dim != GO2_SMP_FEATURE_DIM:
        raise ValueError(f"prior feature dim must be {GO2_SMP_FEATURE_DIM}")
    model = _maybe_compile_model(model, compile_model, compile_mode)
    bundle = (model, scheduler, q_low, q_high, feature_dim, window_size)
    robot = env.scene["robot"]
    joint_ids, joint_names = robot.find_joints(GO2_SMP_JOINT_NAMES, preserve_order=True)
    site_ids, site_names = robot.find_sites(
        GO2_SMP_FOOT_SITE_NAMES, preserve_order=True
    )
    if tuple(joint_names) != GO2_SMP_JOINT_NAMES:
        raise ValueError(f"Go2 SMP joint ordering mismatch: {joint_names}")
    if tuple(site_names) != GO2_SMP_FOOT_SITE_NAMES:
        raise ValueError(f"Go2 SMP foot ordering mismatch: {site_names}")
    env._smp_bundle = bundle  # type: ignore[attr-defined]
    env._smp_joint_ids = torch.as_tensor(joint_ids, device=env.device)  # type: ignore[attr-defined]
    env._smp_foot_site_ids = torch.as_tensor(site_ids, device=env.device)  # type: ignore[attr-defined]
    env._smp_buffer = MotionFeatureBuffer(  # type: ignore[attr-defined]
        env.num_envs, window_size, env.device
    )
    env._smp_error_normalizer = DiffusionErrorNormalizer(  # type: ignore[attr-defined]
        scheduler.num_timesteps, env.device
    )
    chunks = []
    for start in range(0, gsi_pool_size, gsi_batch_size):
        count = min(gsi_batch_size, gsi_pool_size - start)
        chunks.append(sample_prior(model, scheduler, q_low, q_high, count))
    env._smp_gsi_pool = torch.cat(chunks)  # type: ignore[attr-defined]
    env._smp_gsi_head = 0  # type: ignore[attr-defined]
    env._smp_gsi_min_root_lin_vel_x = gsi_min_root_lin_vel_x  # type: ignore[attr-defined]
    _update_gsi_eligible_indices(env)
    if compile_model and env.num_envs != gsi_batch_size:
        dummy_window = torch.randn(
            env.num_envs, window_size, feature_dim, device=env.device
        )
        dummy_timestep = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        model(dummy_window, dummy_timestep)
    gsi_reset(env)


def gsi_eligible_indices(
    pool: torch.Tensor,
    min_root_lin_vel_x: float | None = None,
) -> torch.Tensor:
    """Return pool rows with a finite final local velocity above the threshold."""
    if pool.ndim != 3:
        raise ValueError(f"GSI pool must have shape [N, T, F], got {tuple(pool.shape)}")
    if len(pool) == 0:
        raise ValueError("GSI pool must not be empty")
    if min_root_lin_vel_x is not None and not math.isfinite(min_root_lin_vel_x):
        raise ValueError("GSI minimum root velocity must be finite or None")
    final_velocity_x = split_features(pool)["root_lin_vel"][:, -1, 0]
    eligible = torch.isfinite(final_velocity_x)
    if min_root_lin_vel_x is not None:
        eligible &= final_velocity_x >= min_root_lin_vel_x
    eligible_indices = torch.nonzero(eligible, as_tuple=False).squeeze(-1)
    if eligible_indices.numel() == 0:
        threshold_description = (
            "finite" if min_root_lin_vel_x is None else f">= {min_root_lin_vel_x:g} m/s"
        )
        raise RuntimeError(
            "GSI pool contains no windows with a finite final local root vx "
            f"{threshold_description}; regenerate the prior pool or relax the filter"
        )
    return eligible_indices


def _update_gsi_eligible_indices(env: ManagerBasedRlEnv) -> None:
    pool: torch.Tensor = env._smp_gsi_pool  # type: ignore[attr-defined]
    threshold: float | None = env._smp_gsi_min_root_lin_vel_x  # type: ignore[attr-defined]
    eligible_indices = gsi_eligible_indices(pool, threshold)
    env._smp_gsi_eligible_indices = eligible_indices  # type: ignore[attr-defined]
    env._smp_gsi_eligible_fraction = float(eligible_indices.numel() / len(pool))  # type: ignore[attr-defined]


def _prime_simulation_and_buffer(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    windows: torch.Tensor,
) -> None:
    count, window_size, _ = windows.shape
    parts = split_features(windows)
    root_pos_local = parts["root_pos"]
    root_rotation_6d = parts["root_rot_6d"]
    joint_pos = parts["joint_pos"]
    foot_pos_local = parts["foot_pos"].reshape(count, window_size, 4, 3)
    root_lin_vel_local = parts["root_lin_vel"]
    root_ang_vel_local = parts["root_ang_vel"]

    control_dt = float(env.cfg.sim.mujoco.timestep) * float(env.cfg.decimation)
    joint_vel = torch.zeros_like(joint_pos)
    if window_size > 1:
        joint_vel[:, 1:] = (joint_pos[:, 1:] - joint_pos[:, :-1]) / control_dt
        joint_vel[:, 0] = joint_vel[:, 1]

    robot = env.scene["robot"]
    default_root = robot.data.default_root_state[env_ids]
    default_pos = default_root[:, :3]
    default_quat = default_root[:, 3:7]
    anchor_yaw = yaw_quat(default_quat)
    yaw_window = anchor_yaw[:, None, :].expand(count, window_size, 4).reshape(-1, 4)

    local_xy = root_pos_local.clone()
    local_xy[..., 2] = 0.0
    root_pos = quat_apply(yaw_window, local_xy.reshape(-1, 3)).reshape(
        count, window_size, 3
    )
    root_pos[..., 0] += default_pos[:, None, 0]
    root_pos[..., 1] += default_pos[:, None, 1]
    root_pos[..., 2] = root_pos_local[..., 2]
    local_quat = rotation_6d_to_quaternion(root_rotation_6d.reshape(-1, 6))
    root_quat = quat_mul(yaw_window, local_quat).reshape(count, window_size, 4)
    root_lin_vel = quat_apply(yaw_window, root_lin_vel_local.reshape(-1, 3)).reshape(
        count, window_size, 3
    )
    root_ang_vel = quat_apply(yaw_window, root_ang_vel_local.reshape(-1, 3)).reshape(
        count, window_size, 3
    )
    yaw_feet = (
        anchor_yaw[:, None, None, :].expand(count, window_size, 4, 4).reshape(-1, 4)
    )
    foot_pos = (
        quat_apply(yaw_feet, foot_pos_local.reshape(-1, 3)).reshape(
            count, window_size, 4, 3
        )
        + root_pos[:, :, None]
    )

    root_state = torch.cat(
        (
            root_pos[:, -1] + env.scene.env_origins[env_ids],
            root_quat[:, -1],
            root_lin_vel[:, -1],
            root_ang_vel[:, -1],
        ),
        dim=-1,
    )
    robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    joint_ids: torch.Tensor = env._smp_joint_ids  # type: ignore[attr-defined]
    robot.write_joint_state_to_sim(
        joint_pos[:, -1],
        joint_vel[:, -1],
        joint_ids=joint_ids,
        env_ids=env_ids,
    )
    buffer: MotionFeatureBuffer = env._smp_buffer  # type: ignore[attr-defined]
    buffer.reset(
        env_ids,
        root_pos,
        root_quat,
        root_lin_vel,
        root_ang_vel,
        foot_pos,
        joint_pos,
    )


@torch.no_grad()
def gsi_reset(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None = None) -> None:
    """Reset selected environments from generated motion-prior windows."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if env_ids.numel() == 0:
        return
    pool: torch.Tensor = env._smp_gsi_pool  # type: ignore[attr-defined]
    eligible_indices: torch.Tensor = env._smp_gsi_eligible_indices  # type: ignore[attr-defined]
    choices = torch.randint(
        0, len(eligible_indices), (len(env_ids),), device=env.device
    )
    indices = eligible_indices[choices]
    _prime_simulation_and_buffer(env, env_ids, pool[indices])


@torch.no_grad()
def refresh_gsi_pool(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None = None,
    num_samples: int = 256,
    step_interval: int = 2400,
) -> None:
    """Periodically replace a FIFO slice of the generated initialization pool."""
    del env_ids
    step = int(env.common_step_counter)
    if step == 0 or step % step_interval:
        return
    pool: torch.Tensor = env._smp_gsi_pool  # type: ignore[attr-defined]
    if not 0 < num_samples <= len(pool):
        raise ValueError("num_samples must be within the GSI pool size")
    model, scheduler, q_low, q_high, _, _ = env._smp_bundle  # type: ignore[attr-defined]
    generated = sample_prior(model, scheduler, q_low, q_high, num_samples)
    head = int(env._smp_gsi_head)  # type: ignore[attr-defined]
    indices = (torch.arange(num_samples, device=env.device) + head) % len(pool)
    pool[indices] = generated
    env._smp_gsi_head = (head + num_samples) % len(pool)  # type: ignore[attr-defined]
    _update_gsi_eligible_indices(env)


def smp_gsi_eligible_fraction_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Report the fraction of generated GSI windows allowed by the filter."""
    return torch.full(
        (env.num_envs,),
        float(env._smp_gsi_eligible_fraction),  # type: ignore[attr-defined]
        device=env.device,
    )
