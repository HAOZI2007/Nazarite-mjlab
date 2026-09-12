from __future__ import annotations

import mujoco
import numpy as np

try:
    from .config import ACTION_DIM
    from .mujoco_io import MuJoCoIO
except ImportError:  # pragma: no cover
    from config import ACTION_DIM
    from mujoco_io import MuJoCoIO


def projected_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
    """Express world gravity [0, 0, -1] in the robot body frame."""
    rotation = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, np.asarray(quat_wxyz, dtype=np.float64))
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    return (rotation.reshape(3, 3).T @ gravity_world).astype(np.float32)


def build_actor_terms(
    io: MuJoCoIO,
    command: np.ndarray,
    last_action: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return the six baseline terms for term-major history construction."""
    command = np.asarray(command, dtype=np.float32)
    last_action = np.asarray(last_action, dtype=np.float32)
    if command.shape != (3,):
        raise ValueError(f"command must have shape (3,), got {command.shape}")
    if last_action.shape != (ACTION_DIM,):
        raise ValueError(
            f"last_action must have shape {(ACTION_DIM,)}, got {last_action.shape}"
        )
    return {
        "base_ang_vel": io.get_sensor("imu_ang_vel").astype(np.float32),
        "projected_gravity": projected_gravity(io.data.qpos[3:7]),
        "joint_pos": (io.get_joint_pos() - io.default_q).astype(np.float32),
        "joint_vel": io.get_joint_vel().astype(np.float32),
        "actions": last_action,
        "command": command,
    }
