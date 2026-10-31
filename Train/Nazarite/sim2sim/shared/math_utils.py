from __future__ import annotations

import numpy as np


def action_to_target(
    raw_action: np.ndarray,
    default_q: np.ndarray,
    action_scale: np.ndarray,
) -> np.ndarray:
    """Convert normalized actor output to Go2 joint position targets."""
    raw_action = np.asarray(raw_action, dtype=np.float32)
    return np.asarray(default_q, dtype=np.float32) + raw_action * action_scale
