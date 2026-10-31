"""Observation construction for the 45D baseline policy."""

from __future__ import annotations

import numpy as np

from sim2sim.go2.baseline.config import OBS_DIM, OBSERVATION_NAMES
from sim2sim.shared.mujoco_io import MuJoCoIO
from sim2sim.shared.observation import build_actor_terms


def build_observation(
    io: MuJoCoIO,
    command: np.ndarray,
    last_action: np.ndarray,
) -> np.ndarray:
    """Build the term-major 45D actor input used during baseline training."""
    terms = build_actor_terms(io, command, last_action)
    obs = np.concatenate([terms[name] for name in OBSERVATION_NAMES]).astype(np.float32)
    if obs.shape != (OBS_DIM,):
        raise RuntimeError(f"Baseline observation must be {OBS_DIM}D, got {obs.shape}")
    if not np.all(np.isfinite(obs)):
        raise FloatingPointError("Baseline observation contains NaN or Inf")
    return obs
