"""XZ1 12-DoF quadruped asset configuration."""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
XZ1_XML = _PROJECT_ROOT / "MJCF-Manager" / "Robots" / "XZ1" / "xmls" / "xz1.xml"
assert XZ1_XML.is_file(), f"XZ1 XML not found: {XZ1_XML}"

# Output-side joint inertia.  MuJoCo's armature is already expressed at the
# joint/output side, so the motor rotor inertia is not reflected and added here.
XZ1_OUTPUT_INERTIA = 0.025392063
XZ1_ARMATURE = XZ1_OUTPUT_INERTIA

XZ1_HIP_JOINT_PATTERNS = (r".*_hip_joint", r".*_thigh_joint")
XZ1_SHANK_JOINT_PATTERNS = (r".*_shank_joint",)

XZ1_FOOT_GEOMS = (
    "FL_foot_link_primitive_1",
    "FR_foot_link_primitive_1",
    "RR_foot_link_primitive_1",
    "RL_foot_link_primitive_1",
)
XZ1_FOOT_BODIES = (
    "FL_foot_link",
    "FR_foot_link",
    "RR_foot_link",
    "RL_foot_link",
)
XZ1_HIP_BODIES = (
    "FL_hip_link",
    "FR_hip_link",
    "RR_hip_link",
    "RL_hip_link",
)
XZ1_THIGH_BODIES = (
    "FL_thigh_link",
    "FR_thigh_link",
    "RR_thigh_link",
    "RL_thigh_link",
)
XZ1_SHANK_BODIES = (
    "FL_shank_link",
    "FR_shank_link",
    "RR_shank_link",
    "RL_shank_link",
)
XZ1_BASE_BODY = "base_link"

# Keep sensor/site columns aligned with the MJCF traversal order: FL, FR, RR, RL.
XZ1_FOOT_SITES = ("FL", "FR", "RR", "RL")

# Output-side actuator limit used by the Go2-style position actuator.
XZ1_MOTOR_TORQUE_LIMIT = 31.7
XZ1_HIP_THIGH_STIFFNESS = 32.0
XZ1_HIP_THIGH_DAMPING = 1.9
XZ1_SHANK_STIFFNESS = 35.0
XZ1_SHANK_DAMPING = 1.8


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
XZ1_SHANK_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=XZ1_SHANK_JOINT_PATTERNS,
    stiffness=XZ1_SHANK_STIFFNESS,
    damping=XZ1_SHANK_DAMPING,
    effort_limit=XZ1_MOTOR_TORQUE_LIMIT,
    armature=XZ1_ARMATURE,
    delay_min_lag=0,
    delay_max_lag=9,
    delay_update_period=10,
    delay_per_env_phase=False,
)

ARTICULATION_CFG = EntityArticulationInfoCfg(
    actuators=(XZ1_HIP_ACTUATOR_CFG, XZ1_SHANK_ACTUATOR_CFG),
    soft_joint_pos_limit_factor=0.95,
)

# Radians of joint target per unit action. All three match the Go2 reference:
# at 0.5 the thigh/shank explored twice the joint range per unit action, which
# (with kp 32/35) tripled the torque authority of an action and made the front
# thighs swing to vertical and stub on step edges.
XZ1_ACTION_SCALE = {
    r".*_hip_joint": 0.25,
    r".*_thigh_joint": 0.25,
    r".*_shank_joint": 0.25,
}

XZ1_INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.30),
    rot=(1.0, 0.0, 0.0, 0.0),
    lin_vel=(0.0, 0.0, 0.0),
    ang_vel=(0.0, 0.0, 0.0),
    joint_pos={
        r".*_hip_joint": 0.0,
        # Measured static equilibrium of the shipped position servo (hip and
        # thigh kp 32 / kd 1.9, shank kp 35 / kd 1.8): spawned exactly in this
        # pose and holding the stance targets, with all four foot spheres on the
        # ground, the robot rests at 0.3000 m with the trunk at -0.01 deg pitch
        # and +0.01 deg roll, the largest holding torque being 6.8 N m.  Hip
        # targets stay at zero because the hip axes are vertical: their settled
        # splay (+-1.4 deg) comes from the foot friction cone, not from load.
        # The knee angle is re-settled whenever the foot contact geometry moves:
        # the 0.024 m foot sphere sits 3.5 mm closer to the foot origin than the
        # 0.0275 m sphere it replaced, so the knee was extended by 0.0158 rad to
        # keep the 0.30 m spawn height.  Joint frictionloss (0.2 N m) carries
        # part of the trunk load without servo error and is included here.
        r".*_thigh_joint": 0.83,
        r".*_shank_joint": -1.7404,
 
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
