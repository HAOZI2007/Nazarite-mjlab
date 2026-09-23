"""Walk-These-Ways observation history and phase generation."""

from __future__ import annotations

from collections import deque
from typing import ClassVar

import numpy as np

from ..mujoco_io import MuJoCoIO
from ..observation import build_actor_terms
from .config import (
    BEHAVIOR,
    BEHAVIOR_NOMINAL,
    OBS_DIM,
    OBSERVATION_NAMES,
    PHASE_COMMAND_THRESHOLD,
)


class WTWObservationBuilder:
    """Reproduce the selected WTW policy's 498D term-major actor input.

    The six proprioceptive/command terms use 10 frames, behavior uses 5 frames,
    and phase is the current 8D sin/cos reference. Histories are chronological
    (oldest to newest), matching mjlab's CircularBuffer.buffer property.
    """

    HISTORY_LENGTHS: ClassVar[dict[str, int]] = {
        "base_ang_vel": 10,
        "projected_gravity": 10,
        "joint_pos": 10,
        "joint_vel": 10,
        "actions": 10,
        "command": 10,
        "behavior": 5,
    }

    def __init__(self) -> None:
        self.behavior = BEHAVIOR.copy()
        self.base_phase = 0.0
        self.histories = {
            name: deque(maxlen=length)
            for name, length in self.HISTORY_LENGTHS.items()
        }

    def reset(self) -> None:
        self.base_phase = 0.0
        self.behavior[:] = BEHAVIOR_NOMINAL
        for history in self.histories.values():
            history.clear()

    def build(
        self,
        io: MuJoCoIO,
        command: np.ndarray,
        last_action: np.ndarray,
    ) -> np.ndarray:
        terms = build_actor_terms(io, command, last_action)
        terms["behavior"] = self.behavior

        flattened: list[np.ndarray] = []
        for name in OBSERVATION_NAMES[:-1]:
            frame = np.asarray(terms[name], dtype=np.float32)
            history = self.histories[name]
            if not history:
                # mjlab backfills every slot with the first post-reset frame.
                for _ in range(history.maxlen or 0):
                    history.append(frame.copy())
            else:
                history.append(frame.copy())
            flattened.append(np.stack(history, axis=0).reshape(-1))

        flattened.append(self.phase_reference())
        obs = np.concatenate(flattened).astype(np.float32)
        if obs.shape != (OBS_DIM,):
            raise RuntimeError(f"WTW observation must be {OBS_DIM}D, got {obs.shape}")
        if not np.all(np.isfinite(obs)):
            raise FloatingPointError("WTW observation contains NaN or Inf")
        return obs

    def advance_phase(self, command: np.ndarray, dt: float) -> None:
        command = np.asarray(command, dtype=np.float32)
        magnitude = float(np.linalg.norm(command[:2]) + abs(command[2]))
        if magnitude > PHASE_COMMAND_THRESHOLD:
            frequency = float(self.behavior[3])
            self.base_phase = (self.base_phase + frequency * dt) % 1.0

    def phase_reference(self) -> np.ndarray:
        # Convert theta to [FL, FR, RL, RR] offsets using the training code's
        # convention. Trot remains fixed at theta=[0.5, 0, 0].
        theta1, theta2, theta3 = self.behavior[:3]
        offsets = np.array(
            [theta1 + theta3, theta2 + theta3, theta2, theta1],
            dtype=np.float32,
        )
        phase = (self.base_phase + offsets) % 1.0
        angle = 2.0 * np.pi * phase
        sin_phase = np.sin(angle).astype(np.float32)
        cos_phase = np.cos(angle).astype(np.float32)
        return np.concatenate([sin_phase, cos_phase])
