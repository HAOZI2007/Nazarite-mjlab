from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sensor import (
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

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
#足部传感器
GO2_FOOT_GEOMS = ("FL", "FR", "RL", "RR")
GO2_FOOT_SITES = ("FL", "FR", "RL", "RR")
GO2_FOOT_BODIES = (
    "FL_foot",
    "FR_foot",
    "RL_foot",
    "RR_foot",
)

#获取 MJCF XML
def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(GO2_XML))
    actuators_to_delete = list(spec.actuators)
    for act in actuators_to_delete:
        spec.delete(act)
    return spec

#执行器增益参数
STIFFNESS_LEG = 40.0
DAMPING_LEG = 1.0

#腿部执行器配置
GO2_HIP_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=GO2_HIP_JOINT_PATTERNS,
    stiffness=STIFFNESS_LEG,
    damping=DAMPING_LEG,
    effort_limit=23.7,
)
GO2_CALF_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=GO2_CALF_JOINT_PATTERNS,
    stiffness=STIFFNESS_LEG,
    damping=DAMPING_LEG,
    effort_limit=45.43,
)
ARTICULATION_CFG = EntityArticulationInfoCfg(
    actuators=(GO2_HIP_ACTUATOR_CFG, GO2_CALF_ACTUATOR_CFG),
    soft_joint_pos_limit_factor=0.95,
)

#足端接触传感器
GO2_FEET_GROUND_CFG = ContactSensorCfg(
    name="feet_ground_contact",

    primary=ContactMatch(
        mode="geom",
        pattern=GO2_FOOT_GEOMS,
        entity="robot",
    ),

    secondary=ContactMatch(
        mode="body",
        pattern="terrain",
    ),

    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
)
#足端高度传感器
GO2_FEET_HEIGHT_CFG = TerrainHeightSensorCfg(
    name="foot_height_scan",

    frame=(
        ObjRef(type="site", name="FL", entity="robot"),
        ObjRef(type="site", name="FR", entity="robot"),
        ObjRef(type="site", name="RL", entity="robot"),
        ObjRef(type="site", name="RR", entity="robot"),
    ),

    ray_alignment="yaw",
    max_distance=1.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),

    pattern=RingPatternCfg.single_ring(
        radius=0.04,
        num_samples=4,
    ),
)
#GO2的初始姿态
GO2_INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.27),

    rot=(1.0, 0.0, 0.0, 0.0),

    lin_vel=(0.0, 0.0, 0.0),
    ang_vel=(0.0, 0.0, 0.0),

    joint_pos={
        r".*_hip_joint": 0.0,
        r".*_thigh_joint": 0.9,
        r".*_calf_joint": -1.8,
    },

    joint_vel={
        r".*": 0.0,
    },
)

def get_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=GO2_INIT_STATE,
        collisions=(),
        spec_fn=get_spec,
        articulation=ARTICULATION_CFG,
    )
