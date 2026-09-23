"""Deployment contract for the selected Walk-These-Ways policy."""

import numpy as np

from ..config import ACTOR_TERM_NAMES, ROOT

POLICY = (
    ROOT
    / "logs"
    / "rsl_rl"
    / "go2_flat_wtw_independent"
    / "2026-09-19_23-27-56"
    / "2026-09-19_23-27-56.onnx"
)

OBS_DIM = 498
OBSERVATION_NAMES = [*ACTOR_TERM_NAMES, "behavior", "phase"]
HIP_EFFORT = 23.7
CALF_EFFORT = 45.43

# Trot is fixed; the remaining behavior values are exposed to the gamepad.
BEHAVIOR_NOMINAL = np.array(
    [0.5, 0.0, 0.0, 3.0, 0.0, 0.0, 0.25, 0.0675],
    dtype=np.float32,
)
BEHAVIOR_MIN = np.array(
    [0.5, 0.0, 0.0, 2.0, -0.025, -0.035, 0.21, 0.055],
    dtype=np.float32,
)
BEHAVIOR_MAX = np.array(
    [0.5, 0.0, 0.0, 4.0, 0.025, 0.035, 0.29, 0.08],
    dtype=np.float32,
)
BEHAVIOR = BEHAVIOR_NOMINAL.copy()
COMMAND_MIN = np.array([-1.0, -0.5, -1.0], dtype=np.float32)
COMMAND_MAX = np.array([1.0, 0.5, 1.0], dtype=np.float32)
PHASE_COMMAND_THRESHOLD = 0.05
