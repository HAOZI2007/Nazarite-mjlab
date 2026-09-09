"""Shared offline/online motion features for the Go2 SMP prior.

This follows the reference SMP implementation exactly at the representation
level: every window is expressed in the yaw-only frame of its final root pose.
Only morphology-dependent dimensions differ from G1 (12 Go2 joints and four
feet), giving a 39-dimensional feature vector per frame.
"""

from __future__ import annotations

import torch

from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_apply_inverse,
    quat_conjugate,
    quat_mul,
    yaw_quat,
)

GO2_SMP_JOINT_NAMES: tuple[str, ...] = tuple(
    f"{leg}_{kind}_joint"
    for leg in ("FL", "FR", "RL", "RR")
    for kind in ("hip", "thigh", "calf")
)
GO2_SMP_FOOT_SITE_NAMES: tuple[str, ...] = ("FL", "FR", "RL", "RR")

GO2_SMP_FPS = 50
GO2_SMP_WINDOW_SIZE = 10
GO2_SMP_FEATURE_DIMS: tuple[int, ...] = (3, 6, 12, 12, 3, 3)
GO2_SMP_FEATURE_NAMES: tuple[str, ...] = (
    "root_pos",
    "root_rot_6d",
    "joint_pos",
    "foot_pos",
    "root_lin_vel",
    "root_ang_vel",
)
GO2_SMP_FEATURE_DIM = sum(GO2_SMP_FEATURE_DIMS)


def tan_norm_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Convert wxyz quaternion to SMP's [rotation col0, col2] representation."""
    matrix = matrix_from_quat(quat)
    return torch.cat((matrix[..., :, 0], matrix[..., :, 2]), dim=-1)


def _validate_trajectory_shapes(
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    root_lin_vel_w: torch.Tensor,
    root_ang_vel_w: torch.Tensor,
    foot_pos_w: torch.Tensor,
    joint_pos: torch.Tensor,
) -> None:
    frame_count = root_pos_w.shape[0]
    expected = {
        "root_pos_w": (frame_count, 3),
        "root_quat_w": (frame_count, 4),
        "root_lin_vel_w": (frame_count, 3),
        "root_ang_vel_w": (frame_count, 3),
        "foot_pos_w": (frame_count, len(GO2_SMP_FOOT_SITE_NAMES), 3),
        "joint_pos": (frame_count, len(GO2_SMP_JOINT_NAMES)),
    }
    actual = {
        "root_pos_w": tuple(root_pos_w.shape),
        "root_quat_w": tuple(root_quat_w.shape),
        "root_lin_vel_w": tuple(root_lin_vel_w.shape),
        "root_ang_vel_w": tuple(root_ang_vel_w.shape),
        "foot_pos_w": tuple(foot_pos_w.shape),
        "joint_pos": tuple(joint_pos.shape),
    }
    invalid = [
        f"{name}: expected {expected[name]}, got {shape}"
        for name, shape in actual.items()
        if shape != expected[name]
    ]
    if invalid:
        raise ValueError("invalid Go2 SMP trajectory shapes: " + "; ".join(invalid))
    tensors = (
        root_pos_w,
        root_quat_w,
        root_lin_vel_w,
        root_ang_vel_w,
        foot_pos_w,
        joint_pos,
    )
    if any(not torch.isfinite(value).all() for value in tensors):
        raise ValueError("Go2 SMP trajectory contains non-finite values")


def compute_motion_windows(
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    root_lin_vel_w: torch.Tensor,
    root_ang_vel_w: torch.Tensor,
    foot_pos_w: torch.Tensor,
    joint_pos: torch.Tensor,
    window_size: int = GO2_SMP_WINDOW_SIZE,
    stride: int = 1,
) -> torch.Tensor:
    """Create final-frame yaw-anchored windows with shape ``[N, W, 39]``."""
    _validate_trajectory_shapes(
        root_pos_w, root_quat_w, root_lin_vel_w, root_ang_vel_w, foot_pos_w, joint_pos
    )
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    frame_count = root_pos_w.shape[0]
    if frame_count < window_size:
        return root_pos_w.new_empty((0, window_size, GO2_SMP_FEATURE_DIM))

    starts = torch.arange(
        0, frame_count - window_size + 1, stride, device=root_pos_w.device
    )
    offsets = torch.arange(window_size, device=root_pos_w.device)
    indices = starts[:, None] + offsets[None, :]
    num_windows = indices.shape[0]
    flat_indices = indices.reshape(-1)

    def gather(value: torch.Tensor) -> torch.Tensor:
        return value.index_select(0, flat_indices).reshape(
            num_windows, window_size, *value.shape[1:]
        )

    win_root_pos = gather(root_pos_w)
    win_root_quat = gather(root_quat_w)
    win_root_lin_vel = gather(root_lin_vel_w)
    win_root_ang_vel = gather(root_ang_vel_w)
    win_foot_pos = gather(foot_pos_w)
    win_joint_pos = gather(joint_pos)

    anchor_pos = win_root_pos[:, -1]
    anchor_yaw = yaw_quat(win_root_quat[:, -1])
    yaw_window = (
        anchor_yaw[:, None, :].expand(num_windows, window_size, 4).reshape(-1, 4)
    )
    heading_inverse = (
        quat_conjugate(anchor_yaw)[:, None, :]
        .expand(num_windows, window_size, 4)
        .reshape(-1, 4)
    )

    root_offset = win_root_pos - anchor_pos[:, None, :]
    root_pos_local = quat_apply_inverse(yaw_window, root_offset.reshape(-1, 3)).reshape(
        num_windows, window_size, 3
    )
    root_pos_local = root_pos_local.clone()
    root_pos_local[..., 2] = win_root_pos[..., 2]

    root_quat_local = quat_mul(heading_inverse, win_root_quat.reshape(-1, 4)).reshape(
        num_windows, window_size, 4
    )
    root_rot_6d = tan_norm_from_quat(root_quat_local)

    num_feet = len(GO2_SMP_FOOT_SITE_NAMES)
    foot_offset = win_foot_pos - win_root_pos[:, :, None, :]
    yaw_feet = (
        anchor_yaw[:, None, None, :]
        .expand(num_windows, window_size, num_feet, 4)
        .reshape(-1, 4)
    )
    foot_pos_local = quat_apply_inverse(yaw_feet, foot_offset.reshape(-1, 3)).reshape(
        num_windows, window_size, num_feet * 3
    )
    root_lin_vel_local = quat_apply_inverse(
        yaw_window, win_root_lin_vel.reshape(-1, 3)
    ).reshape(num_windows, window_size, 3)
    root_ang_vel_local = quat_apply_inverse(
        yaw_window, win_root_ang_vel.reshape(-1, 3)
    ).reshape(num_windows, window_size, 3)

    windows = torch.cat(
        (
            root_pos_local,
            root_rot_6d,
            win_joint_pos,
            foot_pos_local,
            root_lin_vel_local,
            root_ang_vel_local,
        ),
        dim=-1,
    )
    if windows.shape[-1] != GO2_SMP_FEATURE_DIM:
        raise RuntimeError(
            f"expected feature dim {GO2_SMP_FEATURE_DIM}, got {windows.shape[-1]}"
        )
    return windows


class MotionFeatureBuffer:
    """Rolling Go2 state buffer using the exact offline feature transform."""

    def __init__(self, num_envs: int, window_size: int, device: torch.device | str):
        self.num_envs = num_envs
        self.window_size = window_size
        self.device = torch.device(device)
        self.root_pos_w = torch.zeros(num_envs, window_size, 3, device=self.device)
        self.root_quat_w = torch.zeros(num_envs, window_size, 4, device=self.device)
        self.root_quat_w[..., 0] = 1.0
        self.root_lin_vel_w = torch.zeros(num_envs, window_size, 3, device=self.device)
        self.root_ang_vel_w = torch.zeros(num_envs, window_size, 3, device=self.device)
        self.foot_pos_w = torch.zeros(num_envs, window_size, 4, 3, device=self.device)
        self.joint_pos = torch.zeros(num_envs, window_size, 12, device=self.device)

    def reset(
        self,
        env_ids: torch.Tensor,
        root_pos_w: torch.Tensor,
        root_quat_w: torch.Tensor,
        root_lin_vel_w: torch.Tensor,
        root_ang_vel_w: torch.Tensor,
        foot_pos_w: torch.Tensor,
        joint_pos: torch.Tensor,
    ) -> None:
        self.root_pos_w[env_ids] = root_pos_w
        self.root_quat_w[env_ids] = root_quat_w
        self.root_lin_vel_w[env_ids] = root_lin_vel_w
        self.root_ang_vel_w[env_ids] = root_ang_vel_w
        self.foot_pos_w[env_ids] = foot_pos_w
        self.joint_pos[env_ids] = joint_pos

    def update(
        self,
        root_pos_w: torch.Tensor,
        root_quat_w: torch.Tensor,
        root_lin_vel_w: torch.Tensor,
        root_ang_vel_w: torch.Tensor,
        foot_pos_w: torch.Tensor,
        joint_pos: torch.Tensor,
    ) -> None:
        for value in (
            self.root_pos_w,
            self.root_quat_w,
            self.root_lin_vel_w,
            self.root_ang_vel_w,
            self.foot_pos_w,
            self.joint_pos,
        ):
            value[:, :-1] = value[:, 1:].clone()
        self.root_pos_w[:, -1] = root_pos_w
        self.root_quat_w[:, -1] = root_quat_w
        self.root_lin_vel_w[:, -1] = root_lin_vel_w
        self.root_ang_vel_w[:, -1] = root_ang_vel_w
        self.foot_pos_w[:, -1] = foot_pos_w
        self.joint_pos[:, -1] = joint_pos

    def compute_features(self) -> torch.Tensor:
        """Compute one ``[W,39]`` window per environment without copying history."""
        # Treat environments as the window batch.  This is the same transform as
        # compute_motion_windows, written directly because trajectories are already
        # windowed along dimension one.
        num_envs, window_size = self.num_envs, self.window_size
        anchor_pos = self.root_pos_w[:, -1]
        anchor_yaw = yaw_quat(self.root_quat_w[:, -1])
        yaw_window = (
            anchor_yaw[:, None, :].expand(num_envs, window_size, 4).reshape(-1, 4)
        )
        heading_inverse = (
            quat_conjugate(anchor_yaw)[:, None, :]
            .expand(num_envs, window_size, 4)
            .reshape(-1, 4)
        )
        root_offset = self.root_pos_w - anchor_pos[:, None]
        root_pos_local = quat_apply_inverse(
            yaw_window, root_offset.reshape(-1, 3)
        ).reshape(num_envs, window_size, 3)
        root_pos_local = root_pos_local.clone()
        root_pos_local[..., 2] = self.root_pos_w[..., 2]
        root_quat_local = quat_mul(
            heading_inverse, self.root_quat_w.reshape(-1, 4)
        ).reshape(num_envs, window_size, 4)
        root_rot_6d = tan_norm_from_quat(root_quat_local)
        foot_offset = self.foot_pos_w - self.root_pos_w[:, :, None]
        yaw_feet = (
            anchor_yaw[:, None, None, :]
            .expand(num_envs, window_size, 4, 4)
            .reshape(-1, 4)
        )
        foot_pos_local = quat_apply_inverse(
            yaw_feet, foot_offset.reshape(-1, 3)
        ).reshape(num_envs, window_size, 12)
        root_lin_vel_local = quat_apply_inverse(
            yaw_window, self.root_lin_vel_w.reshape(-1, 3)
        ).reshape(num_envs, window_size, 3)
        root_ang_vel_local = quat_apply_inverse(
            yaw_window, self.root_ang_vel_w.reshape(-1, 3)
        ).reshape(num_envs, window_size, 3)
        return torch.cat(
            (
                root_pos_local,
                root_rot_6d,
                self.joint_pos,
                foot_pos_local,
                root_lin_vel_local,
                root_ang_vel_local,
            ),
            dim=-1,
        )
