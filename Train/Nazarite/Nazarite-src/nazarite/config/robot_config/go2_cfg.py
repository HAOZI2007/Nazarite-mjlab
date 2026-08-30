from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
GO2_XML = _PROJECT_ROOT / "MJCF-Manager" / "Robots" / "GO2" / "xmls" / "go2.xml"
assert GO2_XML.is_file(), f"Go2 XML not found: {GO2_XML}"

#腿部关节组合
GO2_HIP_JOINT_PATTERNS = (
    r".*_hip_joint",
    r".*_thigh_joint",
    
)
GO2_CALF_JOINT_PATTERNS = (
    r".*_calf_joint",
)
# 足端名称.
GO2_FOOT_GEOMS = ("FL", "FR", "RL", "RR")
GO2_FOOT_SITES = ("FL", "FR", "RL", "RR")
GO2_FOOT_BODIES = (
    "FL_foot",
    "FR_foot",
    "RL_foot",
    "RR_foot",
)
# 髋部身体名称，用于检测髋部与地面的异常接触。
GO2_HIP_BODIES = (
    "FL_hip",
    "FR_hip",
    "RL_hip",
    "RR_hip",
)
GO2_THIGH_BODIES = (
    "FL_thigh",
    "FR_thigh",
    "RL_thigh",
    "RR_thigh",
)
GO2_CALF_BODIES = (
    "FL_calf",
    "FR_calf",
    "RL_calf",
    "RR_calf",
)
GO2_BASE_BODY = "base_link"

#获取 MJCF XML
def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(GO2_XML))
    actuators_to_delete = list(spec.actuators)
    for act in actuators_to_delete:
        spec.delete(act)
    return spec

# 执行器参数, 与 mjlab 的 Go2 配置保持一致.
STIFFNESS_HIP = 15.89524265323492
DAMPING_HIP = 1.0119225759919113
ARMATURE_HIP = 0.004026312
STIFFNESS_CALF = 35.76429596977857
DAMPING_CALF = 2.2768257959818006
ARMATURE_CALF = 0.009059202

# 腿部执行器配置.
GO2_HIP_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=GO2_HIP_JOINT_PATTERNS,
    stiffness=STIFFNESS_HIP,
    damping=DAMPING_HIP,
    effort_limit=23.7,
    armature=ARMATURE_HIP,
)
GO2_CALF_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=GO2_CALF_JOINT_PATTERNS,
    stiffness=STIFFNESS_CALF,
    damping=DAMPING_CALF,
    effort_limit=35.55,
    armature=ARMATURE_CALF,
)
ARTICULATION_CFG = EntityArticulationInfoCfg(
    actuators=(GO2_HIP_ACTUATOR_CFG, GO2_CALF_ACTUATOR_CFG),
    soft_joint_pos_limit_factor=0.95,
)

# 按执行器分组的 action scale, 与 mjlab Go2 保持一致.
GO2_ACTION_SCALE = {
    r".*_hip_joint": 0.25 * 23.7 / STIFFNESS_HIP,
    r".*_thigh_joint": 0.25 * 23.7 / STIFFNESS_HIP,
    r".*_calf_joint": 0.25 * 35.55 / STIFFNESS_CALF,
}

# Go2 的初始姿态.
GO2_INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.32),

    rot=(1.0, 0.0, 0.0, 0.0),

    lin_vel=(0.0, 0.0, 0.0),
    ang_vel=(0.0, 0.0, 0.0),

    joint_pos={
        r"F.*thigh_joint": 0.8,
        r"R.*thigh_joint": 1.0,
        r".*calf_joint": -1.5,
    },

    joint_vel={
        r".*": 0.0,
    },
)

def get_go2_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=GO2_INIT_STATE,
        collisions=(),
        spec_fn=get_spec,
        articulation=ARTICULATION_CFG,
    )
