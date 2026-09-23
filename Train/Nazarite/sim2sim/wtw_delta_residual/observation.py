"""WTW history and DELTA proprioception construction."""

from __future__ import annotations

import numpy as np

from ..mujoco_io import MuJoCoIO
from ..observation import build_actor_terms
from ..wtw.observation import WTWObservationBuilder
from .config import DELTA_PROPRIO_DIM, DELTA_PROPRIO_NAMES


class WtwDeltaObservationBuilder(WTWObservationBuilder):
    """Build the three inputs expected by the composite checkpoint."""

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
        delta_proprio = np.concatenate(
            [np.asarray(terms[name], dtype=np.float32) for name in DELTA_PROPRIO_NAMES]
        ).astype(np.float32)
        if delta_proprio.shape != (DELTA_PROPRIO_DIM,):
            raise RuntimeError(
                f"DELTA proprioception must be {DELTA_PROPRIO_DIM}D, "
                f"got {delta_proprio.shape}"
            )
        if not np.all(np.isfinite(delta_proprio)):
            raise FloatingPointError("DELTA proprioception contains NaN or Inf")
        return wtw_proprio, delta_proprio

