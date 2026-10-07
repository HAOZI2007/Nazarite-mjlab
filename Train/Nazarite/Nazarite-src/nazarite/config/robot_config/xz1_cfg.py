"""XZ1 12-DoF quadruped asset configuration."""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
XZ1_XML = _PROJECT_ROOT / "MJCF-Manager" / "Robots" / "XZ1" / "xmls" / "xz1.xml"
assert XZ1_XML.is_file(), f"XZ1 XML not found: {XZ1_XML}"

# The reduction is output-side : motor-side = 3:38.  Therefore the
# motor-side/output-side speed ratio is 38/3.
XZ1_GEAR_RATIO = 38.0 / 3.0
XZ1_ROTOR_INERTIA = 0.000024796937
XZ1_OUTPUT_INERTIA = 0.025392063
XZ1_REFLECTED_ROTOR_INERTIA = XZ1_ROTOR_INERTIA * XZ1_GEAR_RATIO**2
# MuJoCo armature is expressed at the joint/output side.
XZ1_ARMATURE = XZ1_OUTPUT_INERTIA + XZ1_REFLECTED_ROTOR_INERTIA

XZ1_HIP_JOINT_PATTERNS = (r".*_hip_joint", r".*_thigh_joint")
XZ1_CALF_JOINT_PATTERNS = (r".*_calf_joint",)

XZ1_FOOT_GEOMS = (
    "FL_foot_link_primitive_1",
    "FR_foot_link_primitive_1",
    "RL_foot_link_primitive_1",
    "RR_foot_link_primitive_1",
)
XZ1_FOOT_BODIES = (
    "FL_foot_link",
    "FR_foot_link",
    "RL_foot_link",
    "RR_foot_link",
)
XZ1_HIP_BODIES = (
    "FL_hip_link",
    "FR_hip_link",
    "RL_hip_link",
    "RR_hip_link",
)
XZ1_THIGH_BODIES = (
    "FL_thigh_link",
    "FR_thigh_link",
    "RL_thigh_link",
    "RR_thigh_link",
)
XZ1_CALF_BODIES = (
    "FL_calf_link",
    "FR_calf_link",
    "RL_calf_link",
    "RR_calf_link",
)
XZ1_BASE_BODY = "base_link"

XZ1_FOOT_SITES = ("FL", "FR", "RL", "RR")

# Output-side actuator limit used by the Go2-style position actuator.
XZ1_MOTOR_TORQUE_LIMIT = 31.7
XZ1_HIP_THIGH_STIFFNESS = 35.0
XZ1_HIP_THIGH_DAMPING = 1.2
XZ1_CALF_STIFFNESS = 30.0
XZ1_CALF_DAMPING = 1.0


def get_spec() -> mujoco.MjSpec:
    """Load the XZ1 MJCF and let mjlab create its configured actuators."""
    spec = mujoco.MjSpec.from_file(str(XZ1_XML))
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    return spec


XZ1_HIP_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=XZ1_HIP_JOINT_PATTERNS,
    stiffness=XZ1_HIP_THIGH_STIFFNESS,
    damping=XZ1_HIP_THIGH_DAMPING,
    effort_limit=XZ1_MOTOR_TORQUE_LIMIT,
    armature=XZ1_ARMATURE,
    delay_min_lag=0,
    delay_max_lag=9,
    delay_update_period=10,
    delay_per_env_phase=False,
)
XZ1_CALF_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=XZ1_CALF_JOINT_PATTERNS,
    stiffness=XZ1_CALF_STIFFNESS,
    damping=XZ1_CALF_DAMPING,
    effort_limit=XZ1_MOTOR_TORQUE_LIMIT,
    armature=XZ1_ARMATURE,
    delay_min_lag=0,
    delay_max_lag=9,
    delay_update_period=10,
    delay_per_env_phase=False,
)

ARTICULATION_CFG = EntityArticulationInfoCfg(
    actuators=(XZ1_HIP_ACTUATOR_CFG, XZ1_CALF_ACTUATOR_CFG),
    soft_joint_pos_limit_factor=0.95,
)

XZ1_ACTION_SCALE = {
    r".*_hip_joint": 0.25,
    r".*_thigh_joint": 0.5,
    r".*_calf_joint": 0.5,
}

XZ1_INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.30),
    rot=(1.0, 0.0, 0.0, 0.0),
    lin_vel=(0.0, 0.0, 0.0),
    ang_vel=(0.0, 0.0, 0.0),
    joint_pos={
        r".*_hip_joint": 0.0,
        # Calibrated for root z=0.30 m: all four Go2-compatible foot spheres
        # sit within about 1 mm of the ground plane at reset.
        r".*_thigh_joint": 1.05,
        r".*_calf_joint": -1.75,
    },
    joint_vel={r".*": 0.0},
)


def get_xz1_cfg() -> EntityCfg:
    """Return the XZ1 asset configuration without changing any task."""
    return EntityCfg(
        init_state=XZ1_INIT_STATE,
        collisions=(),
        spec_fn=get_spec,
        articulation=ARTICULATION_CFG,
    )
