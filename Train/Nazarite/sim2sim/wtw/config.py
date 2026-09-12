"""Deployment contract for the selected Walk-These-Ways policy."""

import numpy as np

from ..config import ACTOR_TERM_NAMES, ROOT

POLICY = (
    ROOT
    / "logs"
    / "rsl_rl"
    / "go2_flat_wtw_independent"
    / "2026-09-11_20-39-13"
    / "model_14950.pt"
)

OBS_DIM = 498
OBSERVATION_NAMES = [*ACTOR_TERM_NAMES, "behavior", "phase"]

# Exact fixed behavior used by the selected run's env.yaml.
# The training run sampled frequency in [2, 3] and used trot only; 2.5 Hz is
# the center of that range for deterministic deployment.
BEHAVIOR = np.array(
    [0.5, 0.0, 0.0, 2.5, 0.0, 0.0, 0.25, 0.06],
    dtype=np.float32,
)
COMMAND_MIN = np.array([-2.0, -1.0, -1.0], dtype=np.float32)
COMMAND_MAX = np.array([2.0, 1.0, 1.0], dtype=np.float32)
PHASE_COMMAND_THRESHOLD = 0.05
