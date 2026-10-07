"""Deployment contract for the Nazarite Go2 HIM policy."""

from ..config import ACTION_SCALE, ROOT

POLICY = ROOT / "logs" / "rsl_rl" / "go2_him" / "2026-09-30_22-45-59" / "policy.onnx"
OBS_DIM = 49 * 6
FRAME_DIM = 49
HISTORY_SIZE = 6
OBSERVATION_NAMES = [
    "base_ang_vel", "projected_gravity", "command", "behavior", "phase",
    "joint_pos", "joint_vel", "actions",
]
PHASE_PERIOD = 0.6
ACTION_DIM = 12
ACTION_SCALE = ACTION_SCALE.copy()
HIP_EFFORT = 23.7
CALF_EFFORT = 45.43
