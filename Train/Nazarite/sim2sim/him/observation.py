"""Build current-first frame histories for a HIM ONNX policy."""

from __future__ import annotations

import numpy as np

from ..mujoco_io import MuJoCoIO
from ..observation import build_actor_terms
from .config import FRAME_DIM, HISTORY_SIZE, OBS_DIM, OBSERVATION_NAMES, PHASE_PERIOD


class HIMObservationBuilder:
    """Maintain six complete frames in current-first export order."""

    def __init__(self) -> None:
        self.history = np.zeros((HISTORY_SIZE, FRAME_DIM), dtype=np.float32)
        self.phase = 0.0

    def reset(self) -> None:
        self.history.fill(0.0)
        self.phase = 0.0

    def build(
        self,
        io: MuJoCoIO,
        command: np.ndarray,
        last_action: np.ndarray,
        behavior: np.ndarray,
    ) -> np.ndarray:
        terms = build_actor_terms(io, command, last_action)
        behavior = np.asarray(behavior, dtype=np.float32)
        if behavior.shape != (2,):
            raise ValueError(f"behavior must have shape (2,), got {behavior.shape}")
        terms["behavior"] = behavior
        phase = np.zeros(2, dtype=np.float32)
        if np.linalg.norm(command) >= 0.1:
            phase[:] = [np.sin(2.0 * np.pi * self.phase), np.cos(2.0 * np.pi * self.phase)]
        terms["phase"] = phase
        frame = np.concatenate([terms[name] for name in OBSERVATION_NAMES]).astype(np.float32)
        if frame.shape != (FRAME_DIM,) or not np.all(np.isfinite(frame)):
            raise FloatingPointError("HIM frame is invalid")
        if not np.any(self.history):
            self.history[:] = frame
        else:
            self.history[1:] = self.history[:-1].copy()
            self.history[0] = frame
        return self.history.reshape(OBS_DIM).copy()

    def advance(self, dt: float) -> None:
        self.phase = (self.phase + float(dt) / PHASE_PERIOD) % 1.0
