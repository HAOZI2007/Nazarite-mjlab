"""Direct-task observation builder with checkpoint-dependent DELTA history."""

from __future__ import annotations

from collections import deque

import numpy as np

from ..mujoco_io import MuJoCoIO
from ..observation import build_actor_terms
from ..wtw_delta_residual.config import DELTA_PROPRIO_NAMES
from ..wtw_delta_residual.observation import WtwDeltaObservationBuilder


class DirectObservationBuilder(WtwDeltaObservationBuilder):
    """Build WTW input plus the 1-frame or 5-frame Direct DELTA proprioception."""

    def __init__(self, delta_history_length: int = 1):
        super().__init__()
        if delta_history_length not in (1, 5):
            raise ValueError("Direct DELTA history must be 1 or 5 frames")
        self.delta_history_length = int(delta_history_length)
        self.delta_histories = {
            name: deque(maxlen=self.delta_history_length)
            for name in DELTA_PROPRIO_NAMES
        }

    def reset(self) -> None:
        super().reset()
        for history in self.delta_histories.values():
            history.clear()

    def build_inputs(
        self,
        io: MuJoCoIO,
        command: np.ndarray,
        last_action: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        wtw_proprio = self.build(io, command, last_action)
        terms = build_actor_terms(io, command, last_action)
        terms["behavior"] = self.behavior.copy()
        terms["phase"] = self.phase_reference()
        flattened: list[np.ndarray] = []
        for name in DELTA_PROPRIO_NAMES:
            frame = np.asarray(terms[name], dtype=np.float32)
            history = self.delta_histories[name]
            if not history:
                for _ in range(history.maxlen or 0):
                    history.append(frame.copy())
            else:
                history.append(frame.copy())
            flattened.append(np.stack(history, axis=0).reshape(-1))
        delta_proprio = np.concatenate(flattened).astype(np.float32)
        if not np.all(np.isfinite(delta_proprio)):
            raise FloatingPointError("Direct DELTA proprioception contains NaN or Inf")
        return wtw_proprio, delta_proprio
