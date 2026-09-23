"""Deployment contract for the selected WTW + DELTA checkpoint."""

import numpy as np

from ..config import ACTION_DIM, ROOT

POLICY = (
    ROOT
    / "logs"
    / "rsl_rl"
    / "go2_wtw_delta_residual"
    / "2026-09-22_18-53-22"
    / "model_5650.pt"
)

WTW_OBS_DIM = 498
DELTA_PROPRIO_DIM = 61
MAP_HEIGHT = 16
MAP_WIDTH = 26
MAP_CHANNELS = 4
DELTA_MAP_DIM = MAP_HEIGHT * MAP_WIDTH * MAP_CHANNELS
DELTA_OBS_DIM = WTW_OBS_DIM + DELTA_PROPRIO_DIM + DELTA_MAP_DIM

MAP_X_RANGE = (0.0, 2.5)
MAP_Y_RANGE = (-0.8, 0.8)
MAP_Z_SCALE = 0.8
MAP_RAY_HEIGHT = 3.0

RESIDUAL_JOINT_SCALES = (
    0.35, 0.80, 1.00,
    0.35, 0.80, 1.00,
    0.35, 0.80, 1.00,
    0.35, 0.80, 1.00,
)

COMMAND_MIN = np.array([-1.0, -0.5, -1.0], dtype=np.float32)
COMMAND_MAX = np.array([1.0, 0.5, 1.0], dtype=np.float32)

# The six instantaneous terms are the same terms used by WTW, followed by the
# 8D behavior vector and the 8D phase reference.
DELTA_PROPRIO_NAMES = (
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
    "command",
    "behavior",
    "phase",
)

if DELTA_PROPRIO_DIM != 3 + 3 + 12 + 12 + 12 + 3 + 8 + 8:
    raise RuntimeError("DELTA proprioception contract is inconsistent")
if DELTA_OBS_DIM != 2223:
    raise RuntimeError("Expected the WTW + DELTA checkpoint input to be 2223D")
if ACTION_DIM != 12:
    raise RuntimeError("Go2 sim2sim expects 12 actions")
