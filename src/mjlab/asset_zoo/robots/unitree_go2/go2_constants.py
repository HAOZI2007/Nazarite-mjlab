"""Unitree Go2 constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator, reflected_inertia
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

# 确保这里的路径指向你的 go2.xml
GO2_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "unitree_go2" / "xmls" / "go2.xml"
)
assert GO2_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(GO2_XML))


##
# Actuator config (电机参数).
##

# 转子惯量 (保持与原厂相近的量级，维持仿真稳定)
ROTOR_INERTIA = 0.000111842

# 减速比
HIP_GEAR_RATIO = 6
KNEE_GEAR_RATIO = HIP_GEAR_RATIO * 1.5

# Go2 的电机性能限制
HIP_ACTUATOR = ElectricActuator(
  reflected_inertia=reflected_inertia(ROTOR_INERTIA, HIP_GEAR_RATIO),
  velocity_limit=30.0,  # Go2 最大关节速度
  effort_limit=23.7,    # Go2 髋/大腿关节峰值扭矩
)
KNEE_ACTUATOR = ElectricActuator(
  reflected_inertia=reflected_inertia(ROTOR_INERTIA, KNEE_GEAR_RATIO),
  velocity_limit=20.0,
  effort_limit=35.55,   # 小腿关节通过连杆/减速拥有更高等效扭矩
)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_HIP = HIP_ACTUATOR.reflected_inertia * NATURAL_FREQ**2
DAMPING_HIP = 2 * DAMPING_RATIO * HIP_ACTUATOR.reflected_inertia * NATURAL_FREQ

STIFFNESS_KNEE = KNEE_ACTUATOR.reflected_inertia * NATURAL_FREQ**2
DAMPING_KNEE = 2 * DAMPING_RATIO * KNEE_ACTUATOR.reflected_inertia * NATURAL_FREQ

GO2_HIP_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_joint", ".*_thigh_joint"),
  stiffness=STIFFNESS_HIP,
  damping=DAMPING_HIP,
  effort_limit=HIP_ACTUATOR.effort_limit,
  armature=HIP_ACTUATOR.reflected_inertia,
)
GO2_KNEE_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_calf_joint",),
  stiffness=STIFFNESS_KNEE,
  damping=DAMPING_KNEE,
  effort_limit=KNEE_ACTUATOR.effort_limit,
  armature=KNEE_ACTUATOR.reflected_inertia,
)

##
# Keyframes (初始站立姿态).
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.32),  # Go2 初始高度比 Go1 略高
  joint_pos={
    ".*thigh_joint": 0.8,  # 大腿稍微前伸
    ".*calf_joint": -1.5,  # 小腿往后收
    ".*R_hip_joint": 0.1,  
    ".*L_hip_joint": -0.1, 
  },
  joint_vel={".*": 0.0},
)

##
# Collision config (碰撞检测).
##

# 完美匹配 go2.xml 里的极简脚部命名 ("FL", "FR", "RL", "RR")
_foot_regex = "^[FR][LR]$"

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(_foot_regex,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
  solimp=(0.9, 0.95, 0.023),
)

FULL_COLLISION = CollisionCfg(
  # 只针对脚部覆盖高摩擦力参数，身体其他碰撞体保留 xml 默认设置
  geom_names_expr=(_foot_regex,),
  solref=(0.01, 1),
  condim={_foot_regex: 6},
  priority={_foot_regex: 1},
  friction={_foot_regex: (1, 5e-3, 5e-4)},
)

##
# Final config (导出配置).
##

GO2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    GO2_HIP_ACTUATOR_CFG,
    GO2_KNEE_ACTUATOR_CFG,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_go2_robot_cfg() -> EntityCfg:
  """获取全新的 Go2 机器人配置实例"""
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=GO2_ARTICULATION,
  )


GO2_ACTION_SCALE: dict[str, float] = {}
for a in GO2_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    GO2_ACTION_SCALE[n] = 0.25 * e / s


# 这个代码块可以让你直接运行这个文件来预览模型！
if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_go2_robot_cfg())
  viewer.launch(robot.spec.compile())