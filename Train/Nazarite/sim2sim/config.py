from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

GO2_XML = (
    ROOT / "MJCF-Manager" / "Robots" / "GO2" / "xmls" / "go2.xml"
)

# Match the exported ONNX metadata and mjlab's natural joint order exactly.
JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]

# 与当前 Nazarite GO2_INIT_STATE 对齐
DEFAULT_Q = np.array([
    0.0, 0.8, -1.5,
    0.0, 0.8, -1.5,
    0.0, 1.0, -1.5,
    0.0, 1.0, -1.5,
], dtype=np.float32)

PHYSICS_DT = 0.002
DECIMATION = 10
CONTROL_DT = PHYSICS_DT * DECIMATION

# Match the reference Go2 Trot sim2sim controller.
STIFFNESS_HIP = 20.0
STIFFNESS_CALF = 20.0
DAMPING_HIP = 0.5
DAMPING_CALF = 0.5
KP_SCALE = 1.0
KD_SCALE = 1.0

ACTION_SCALE = np.full(12, 0.25, dtype=np.float32)

ACTION_DIM = 12
ACTOR_TERM_NAMES = [
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
    "command",
]
