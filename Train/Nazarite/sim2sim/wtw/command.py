"""Command limits matching the selected WTW training distribution."""

from __future__ import annotations

import numpy as np

from .config import COMMAND_MAX, COMMAND_MIN


def restrict_command(command: np.ndarray) -> np.ndarray:
    """Clip [vx, vy, yaw] to the selected WTW training distribution."""
    command = np.asarray(command, dtype=np.float32)
    if command.shape != (3,):
        raise ValueError(f"command must have shape (3,), got {command.shape}")
    return np.clip(command, COMMAND_MIN, COMMAND_MAX).astype(np.float32)
