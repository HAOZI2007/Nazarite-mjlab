"""Deployment contract for the selected baseline policy."""

from sim2sim.shared.config import ACTOR_TERM_NAMES, ROOT

POLICY = (
    ROOT
    / "logs"
    / "rsl_rl"
    / "go2_flat_baseline"
    / "2026-09-10_07-58-04"
    / "2026-09-10_07-58-04.onnx"
)

OBS_DIM = 45
OBSERVATION_NAMES = ACTOR_TERM_NAMES.copy()
