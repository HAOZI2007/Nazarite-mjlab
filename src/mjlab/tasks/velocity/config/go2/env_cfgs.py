"""Unitree Go2 velocity environment configurations."""

import math
from typing import Literal

from mjlab.asset_zoo.robots import (
  GO2_ACTION_SCALE,
  get_go2_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
#爬楼梯
from dataclasses import replace
from mjlab.terrains.config import STAIRS_TERRAINS_CFG

TerrainType = Literal["rough", "obstacles"]


def unitree_go2_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 rough terrain velocity configuration."""
  # 获取通用的速度追踪 MDP 骨架 （包含了通用的 Obs， Reward ， Terminates逻辑）
  cfg = make_velocity_env_cfg()

  # Scene 层（对应底层实体资产）
  cfg.scene.entities = {"robot": get_go2_robot_cfg()}

    # ==========================================
  # 修改智能体（并行环境）数量
  # ==========================================
  # 训练时通常开 4096 或 8192，具体取决于你的显存大小 (VRAM)
  cfg.scene.num_envs = 1024 
  # 环境之间的间隔，防止不同环境里的狗互相穿模看到彼此
  cfg.scene.env_spacing = 1.0

  # 设定底层 mujoco 仿真后段的特定参数
  #连续碰撞检测 (Continuous Collision Detection) 的迭代次数。狗在跑动时脚速极快，加上乱石堆的几何边缘锐利，普通碰撞检测极易“一脚踩穿地面（穿模）”。提高此参数能极大提高物理仿真的保真度。
  cfg.sim.mujoco.ccd_iterations = 500
  #阻抗比 (Impedance ratio)。它决定了接触面的“硬度”。设为 10 可以让地面和脚底的接触更加坚硬，防止接触面像弹簧一样软绵绵的。
  cfg.sim.mujoco.impratio = 10
  #使用椭圆摩擦锥。相比于默认的金字塔摩擦锥，椭圆摩擦锥计算更精确，物理表现更真实（尤其在脚尖滑动时）。
  cfg.sim.mujoco.cone = "elliptic"
  #接触传感器的最大匹配数。复杂地形会导致极大量的接触点（比如脚趾卡在两块石头中间），必须调大内存分配，否则仿真会报错。
  cfg.sim.contact_sensor_maxmatch = 500

  #核心修复：显式限制底层物理引擎的内存分配上限，防止 OOM
  cfg.sim.nconmax = 100                 # 限制单只狗的最大接触点内存池（默认 heuristic 会分配极大）
  cfg.sim.njmax = 500                   # 限制最大约束数量内存池
  cfg.sim.contact_sensor_maxmatch = 64  # 缩小传感器的接触匹配矩阵
  cfg.sim.mujoco.ccd_iterations = 200   # 将连续碰撞迭代从 500 适当下调至 200
  
    # ==========================================
    # 3. 构造不对称架构 (Asymmetric Actor-Critic)
    # ==========================================
  if "height_scan" in cfg.observations["actor"].terms:
    del cfg.observations["actor"].terms["height_scan"]

  # Managers 层（Obs / Sensor）
  # 将通用的射线传感器绑定到GO2的 “trunk”（躯干上）
  for sensor in cfg.scene.sensors or ():
    #terrain_scan: 高程扫描仪（相当于深度相机）。这里强制把它的坐标系绑在 trunk (躯干) 上，这样机器人获取的始终是“以自我为中心”的身体前下方的地形高程数据。
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "base_link"

  # GO2脚部名称定义（根据你的xml文件决定）
  foot_names = ("FR", "FL", "RR", "RL")
  site_names = ("FR", "FL", "RR", "RL")
  # 直接使用真实的脚底碰撞体名称，即 "FR", "FL", "RR", "RL"
  geom_names = ("FR", "FL", "RR", "RL")

  # Wire foot height scan to per-foot sites.
  #foot_height_scan: 绑在四个脚底板（FR, FL, RR, RL）上的探测器。
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
      )
      #RingPatternCfg: 在脚底板周围 0.04 米的半径打 4 个激光点，专用来精确测量“脚底板距离地面还有多高”，用于计算跨越障碍物时的抬腿高度（Foot clearance）。
      sensor.pattern = RingPatternCfg.single_ring(radius=0.02, num_samples=4)

  #触觉神经网络 (接触传感器)，框架使用了极其精细的接触分类，用于奖励和惩罚判定：
  #feet_ground_cfg (脚底接触): 匹配四个脚底的碰撞体 (*_foot_collision) 和地形 (terrain)。用于判断是否踩实地面、计算腾空时间 (track_air_time=True)。
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  #self_collision_cfg (自碰撞): primary 和 secondary 都是机器人本体。如果大腿卡进了躯干，就会触发这个传感器。
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  # ==============================================================
  # 1. 大腿触地传感器 (使用 mode="body" 匹配)
  # ==============================================================
  thigh_body_names = tuple(f"{leg}_thigh" for leg in foot_names)
  thigh_ground_cfg = ContactSensorCfg(
    name="thigh_ground_touch",
    primary=ContactMatch(
      mode="body",                # 👈 核心改动：从 geom 改为 body
      entity="robot",
      pattern=thigh_body_names,   # 👈 直接匹配 Body 的名字 (如 FL_thigh)
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  # ==============================================================
  # 2. 小腿触地传感器 (使用 mode="body" 匹配)
  # ==============================================================
  calf_body_names = tuple(f"{leg}_calf" for leg in foot_names)
  shank_ground_cfg = ContactSensorCfg(
    name="shank_ground_touch",
    primary=ContactMatch(
      mode="body",                # 👈 核心改动：从 geom 改为 body
      entity="robot",
      pattern=calf_body_names,    # 👈 直接匹配 Body 的名字 (如 FL_calf)
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  # ==============================================================
  # 3. 躯干/头部触地传感器 (使用 mode="body" 匹配)
  # ==============================================================
  trunk_head_ground_cfg = ContactSensorCfg(
    name="trunk_ground_touch",
    primary=ContactMatch(
      mode="body",                # 👈 核心改动：从 geom 改为 body
      entity="robot",
      pattern=("base_link",),     # 👈 Go2 的主躯干叫 base_link
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )


# 🌟 核心修复 1：先剔除基础配置中自带的同名旧传感器，防止新传感器被屏蔽
  new_sensor_names = {
      "feet_ground_contact", 
      "self_collision", 
      "thigh_ground_touch", 
      "shank_ground_touch", 
      "trunk_ground_touch"
  }
  cfg.scene.sensors = tuple(
      s for s in (cfg.scene.sensors or ()) if s.name not in new_sensor_names
  )

  # 🌟 然后再把我们精确配置好的 4 通道传感器加进去
  cfg.scene.sensors = cfg.scene.sensors + (
    feet_ground_cfg,
    self_collision_cfg,
    thigh_ground_cfg,
    shank_ground_cfg,
    trunk_head_ground_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  # Managers 层， Action
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  #scale: 神经网络输出的动作区间通常是 [-1, 1]。通过乘上这个 scale（通常是个很小的值如 0.25），将其转化为真实的关节目标角度增量。这叫做动作缩放，是 RL 训练易于收敛的关键技巧。
  joint_pos_action.scale = GO2_ACTION_SCALE #使用GO2的关节缩放比例

  # viewer: 在你打开 GUI 窗口看仿真时，镜头会自动锁定并追踪 Go2 的 trunk（躯干）。
  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0


  # Managers层 Events（领域随机化）
  #这里的mode = "startup"说明摩擦力随机化只会在环境重置/启动时发生一次
  #如果mode = “interval”并且间隔很短，那么就会使得智能体受不合理噪声影响过大
  # Replace the base foot_friction with per-axis friction events for condim 6.
  #默认摩擦力通常只有一个系数。但为了让 Go2 更鲁棒，作者将摩擦力拆解成了滑动摩擦 (slide)、自旋摩擦 (spin)、滚动摩擦 (roll) 三个维度独立随机化，这被称为 condim 6 精细接触模型。
  del cfg.events["foot_friction"]
  cfg.events["foot_friction_slide"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=geom_names),
      "operation": "abs",
      "axes": [0],
      "ranges": (0.3, 1.5),
      "shared_random": True,
    },
  )
  cfg.events["foot_friction_spin"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=geom_names),
      "operation": "abs",
      "distribution": "log_uniform",
      "axes": [1],
      "ranges": (1e-4, 2e-2),
      "shared_random": True,
    },
  )
  cfg.events["foot_friction_roll"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=geom_names),
      "operation": "abs",
      "distribution": "log_uniform",
      "axes": [2],
      "ranges": (1e-5, 5e-3),
      "shared_random": True,
    },
  )
  # 质心偏移随机化（base_com: 随机偏移躯干（trunk）的质心位置。现实中电池位置可能没放正，这能让策略学会动态平衡。）
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)


  #Managers层 (Rewards & Terminations)

  #针对GO2的尺寸调整姿态奖励的标准差（std）
  #调整站立的
  #pose (姿态奖励): 期望机器人在不同速度下保持合理的默认姿态。
  #std (Standard Deviation): 高斯奖励函数的方差。0.05 非常小，意味着当要求机器人站立 (standing) 时，关节稍有偏差就会严重扣分（要求极严）；而跑动 (running) 时 std 放大到 0.6，允许机器人关节大幅度自由挥舞。
  #weight = 0.0: 把 body_ang_vel (躯干角速度惩罚)、air_time (腾空时间奖励) 的权重设为 0。意思是：在复杂地形上，为了翻越障碍，躯干晃动是合理的，不要惩罚它；同时不要强行鼓励腾空跳跃，先走稳再说。
  #weight = -0.1 (碰撞惩罚): 接入前面定义的接触传感器。大腿碰地 (shank_collision)、头碰地 (trunk_head_collision) 都会一直扣分（$-0.1$ 每步），逼迫网络学会在石头堆里高抬腿。
  cfg.rewards["pose"].weight = 1.5  # 👈 加大姿态奖励的诱惑力
  cfg.rewards["pose"].params["std_standing"] = {
    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.2,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.3,
  }
  #调整行走的
  cfg.rewards["pose"].params["std_walking"] = {
    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.5,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.8,
  }
  #调整奔跑的
  cfg.rewards["pose"].params["std_running"] = {
    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.8,
    r".*(FR|FL|RR|RL)_calf_joint.*": 1.2,
  }

  # 针对 GO2 躯干的惩罚配置
  cfg.rewards["upright"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["upright"].params["terrain_sensor_names"] = ("terrain_scan",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)


# ✅ 修复 2：恢复原始循环，绝不能把 foot_swing_height 加进来！
  for reward_name in ["foot_clearance", "foot_slip"]:
      if reward_name in cfg.rewards:
          cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

    
  if "foot_slip" in cfg.rewards:
    cfg.rewards["foot_slip"].weight = -0.02
  cfg.rewards["body_ang_vel"].weight = -0.02
  cfg.rewards["angular_momentum"].weight = 0.0
  cfg.rewards["air_time"].weight = 0.35

  # 🌟【新增】把平顺性惩罚也加到跑动基础配置里！
  # 严禁它为了追求速度和腾空而产生物理学不允许的鬼畜抖动
  smoothness_penalties = {
      "action_rate_l2": -0.01,  # 惩罚过快的动作突变
      "joint_acc_l2": -0.0001, # 惩罚过高的关节加速度
      "torques_l2": -0.0002,   # 惩罚过载的电机扭矩
      "dof_vel": -0.001        # 惩罚超速的关节转速
  }
  for penalty_name, weight in smoothness_penalties.items():
      if penalty_name in cfg.rewards:
          cfg.rewards[penalty_name].weight = weight
      elif penalty_name.replace("_l2", "") in cfg.rewards:
          cfg.rewards[penalty_name.replace("_l2", "")].weight = weight

  # Per-body-group collision penalties.
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": self_collision_cfg.name},
  )
  cfg.rewards["shank_collision"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": shank_ground_cfg.name},
  )
  cfg.rewards["trunk_head_collision"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": trunk_head_ground_cfg.name},
  )

  # On rough terrain the quadruped tilts significantly; don't terminate on
  # orientation alone. Let out_of_terrain_bounds handle resets.
  #终止条件 (Managers - Terminations)决定什么时候这局游戏直接宣告失败 (Game Over)。
  #pop("fell_over"): 这是一个非常聪明的改动！默认模板里，身体倾斜超过一定角度（比如 70 度）就判摔倒。但在 Rough Terrain 上爬陡坡或踩大坑时，身体不可避免会剧烈倾斜。如果按老规矩，它永远爬不上坡。所以删掉纯角度判定。
  cfg.terminations.pop("fell_over", None)
  #illegal_contact: 新的判定规则：只要你的大腿 (thigh) 碰到了地面，对不起，这就是真摔倒了，立刻结束回合。
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": thigh_ground_cfg.name},
  )
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  # Apply play mode overrides.
  #游玩/部署模式覆盖 (Play Mode Overrides)
  if play:
    
    # Effectively infinite episode length.
    cfg.scene.num_envs = 1  # 👈 Play 模式下强制只生成 1 只智能体，方便观察
    # 🌟 修改这里：禁用随机抽样，使其完全听从 UI 滑块的绝对指挥
    twist_cmd.ranges.lin_vel_x = (0.0, 0.0)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
    # 🌟 将重采样时间设为无穷大，防止环境在后台偷偷改变你的指令
    twist_cmd.resampling_time_range = (1e9, 1e9)

    #enable_corruption = False: 关掉给传感器强行加的训练噪声（我们想看策略的真实表现）。
    cfg.observations["actor"].enable_corruption = False
    #pop("push_robot"): 关掉训练时的“无形之手”（随机推力）。
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    #curriculum = False / border_width = 10.0: 关掉地形难度递增，直接生成一个小块的固定地形用于观赏。
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_go2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 flat terrain velocity configuration."""
  cfg = unitree_go2_rough_env_cfg(play=play)

  # ==========================================
  # 修改智能体（并行环境）数量
  # ==========================================
  # 训练时通常开 4096 或 8192，具体取决于你的显存大小 (VRAM)
  cfg.scene.num_envs = 8192 
  # 环境之间的间隔，防止不同环境里的狗互相穿模看到彼此
  cfg.scene.env_spacing = 2.0

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensors and collision sensors not needed on flat.
  # 1. 修复：把触地传感器加回来！让裁判睁开眼睛！
  remove_sensors = {
    "terrain_scan",
    "self_collision",
    #"thigh_ground_touch",
    #"shank_ground_touch",
    #"trunk_ground_touch",
  }
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name not in remove_sensors
  )
  cfg.observations["actor"].terms.pop("height_scan", None)
  cfg.observations["critic"].terms.pop("height_scan", None)
  cfg.rewards["upright"].params.pop("terrain_sensor_names", None)

  # Remove granular collision rewards (not useful on flat ground).
  for key in ("self_collisions", "shank_collision", "trunk_head_collision"):
    cfg.rewards.pop(key, None)

  # On flat terrain fell_over is sufficient; thigh contact implies fallen.
  # 2. 修复：保留非法接触（趴地）的终止条件！
  # cfg.terminations.pop("illegal_contact", None)  <-- 把这一行直接【删掉】或注释掉！
  #cfg.terminations.pop("illegal_contact", None)
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation,
    params={"limit_angle": math.radians(70.0)},
  )

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  if play:
    cfg.scene.num_envs = 1  # 👈 Play 模式下强制只生成 1 只智能体，方便观察
    # 🌟 修改这里：禁用随机抽样，使其完全听从 UI 滑块的绝对指挥
    twist_cmd.ranges.lin_vel_x = (0.0, 0.0)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
    # 🌟 将重采样时间设为无穷大，防止环境在后台偷偷改变你的指令
    twist_cmd.resampling_time_range = (1e9, 1e9)
  else:
     # 👈 新增：在训练模式下，强迫模型最高训练到 2.5m/s，让 2.0m/s 落在舒适区内
    twist_cmd.ranges.lin_vel_x = (-1.0, 5.0)   
    # 可选：适当增加一点 Y 和 Yaw 的训练范围，增强鲁棒性
    twist_cmd.ranges.lin_vel_y = (-2.0, 2.0)
    twist_cmd.ranges.ang_vel_z = (-2.0, 2.0)
  return cfg


def unitree_go2_stand_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 flat terrain STANDING configuration."""
  
  # 1. 继承刚才精简好的平地配置
  cfg = unitree_go2_flat_env_cfg(play=play)

  # ==========================================
  # 核心修改 1：命令降维 (Commands)
  # ==========================================
  # 站立的本质就是：目标线速度和角速度永远为 0
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_x = (0.0, 0.0)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
  twist_cmd.ranges.heading = (0.0, 0.0)

  # ==========================================
  # 核心修改 2：课程关闭 (Curriculum)
  # ==========================================
  # 必须关掉速度课程！否则训练到 5000 步时，系统会自动把速度目标提升到 1.5m/s，狗就跑出去了
  cfg.curriculum.pop("command_vel", None)

  # ==========================================
  # 核心修改 3：奖励重塑 (Rewards)
  # ==========================================
  remove_rewards = ["foot_clearance", "foot_swing_height", "air_time", "soft_landing"]
  for rew in remove_rewards:
    cfg.rewards.pop(rew, None)

  if "foot_slip" in cfg.rewards:
    cfg.rewards["foot_slip"].weight = -0.05

  # 🌟 修复 1：提高姿态奖励的权重，并【放宽】严苛的高斯方差
  if "pose" in cfg.rewards:
    cfg.rewards["pose"].weight = 3.0  # 👈 加大姿态奖励的诱惑力
    cfg.rewards["pose"].params["std_standing"] = {
      r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.2, # 👈 从 0.05 放宽到 0.2，允许一点点误差
      r".*(FR|FL|RR|RL)_calf_joint.*": 0.3,        # 👈 从 0.1 放宽到 0.3
    }
  
  # 🌟 修复 2：严惩“把腿伸直劈叉/收到底”的平躺行为
  if "dof_pos_limits" in cfg.rewards:
      cfg.rewards["dof_pos_limits"].weight = -1.0  # 👈 给一个极大的负分，绝对不允许关节碰到底线


# 【修改】给平顺性惩罚一个很小的权重，防止策略学出“高频鬼畜抖动”导致物理爆炸
  smoothness_penalties = {
      "action_rate_l2": -0.01,  # 惩罚网络输出动作的突变
      "joint_acc_l2": -0.00001, # 惩罚物理关节的极端加速度
      "torques_l2": -0.00001,   # 惩罚过大的电机扭矩
      "dof_vel": -0.0001        # 惩罚过快的关节速度
  }
  
  for penalty_name, weight in smoothness_penalties.items():
      # 兼容不同的命名习惯
      if penalty_name in cfg.rewards:
          cfg.rewards[penalty_name].weight = weight
      # 尝试去掉 _l2 后缀的兼容
      elif penalty_name.replace("_l2", "") in cfg.rewards:
          cfg.rewards[penalty_name.replace("_l2", "")].weight = weight

  # ==========================================
  # 核心修改 4：抗扰动训练 (Events) - 进阶平衡
  # ==========================================
  if not play:
    # 站立训练后期，为了防止它学成一个“僵尸站”（一碰就倒），
    # 我们需要重新引入“随机推力”。让它学会在原地被推后恢复平衡。
    cfg.events["push_robot"] = EventTermCfg(
      func=envs_mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(2.0, 4.0), # 每 2~4 秒推一次
      params={
        "velocity_range": {
          "x": (-0.3, 0.3),  # 推力不要太大，先从 0.3m/s 冲量开始
          "y": (-0.3, 0.3),
          "z": (0.0, 0.0),
          "yaw": (-0.2, 0.2),
        },
      },
    )
  
  return cfg

def unitree_go2_stairs_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Unitree Go2 Blind Stairs configuration."""
    cfg = unitree_go2_rough_env_cfg(play=play)

    # ==========================================
    # 修改智能体（并行环境）数量
    # ==========================================
    cfg.scene.num_envs = 1024 
    cfg.scene.env_spacing = 0.0

    # 🌟 核心修复：显式限制底层物理引擎的内存分配上限，防止 OOM
    cfg.sim.nconmax = 100                 # 限制单只狗的最大接触点内存池（默认 heuristic 会分配极大）
    cfg.sim.njmax = 500                   # 限制最大约束数量内存池
    cfg.sim.contact_sensor_maxmatch = 64  # 缩小传感器的接触匹配矩阵
    cfg.sim.mujoco.ccd_iterations = 200   # 将连续碰撞迭代从 500 适当下调至 100

    # ==========================================
    # 2. 地形课程配置 (Curriculum)
    # ==========================================
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_generator = replace(STAIRS_TERRAINS_CFG)
    
    # ✅ 修复 1：必须关掉速度课程（防止覆盖指令），但保留地形课程（terrain_levels）让狗能向上爬！
    cfg.curriculum.pop("command_vel", None)

    # ==========================================
    # 3. 构造不对称架构 (Asymmetric Actor-Critic)
    # ==========================================
    if "height_scan" in cfg.observations["actor"].terms:
        del cfg.observations["actor"].terms["height_scan"]
    
    # ✅ 修复 2：删除了 cfg.observations["actor"].history_length = 15，因为底层类不支持。
    # 历史堆叠将留到 train.py 中通过 HistoryWrapper 实现。

    # ==========================================
    # 4. 奖励重塑 (Reward Reshaping)
    # ==========================================
    if "upright" in cfg.rewards:
        cfg.rewards["upright"].weight = 0.3

    if "body_ang_vel" in cfg.rewards:
        cfg.rewards["body_ang_vel"].weight = -0.01

    if "foot_clearance" in cfg.rewards:
        cfg.rewards["foot_clearance"].weight = -3.0

    # ==========================================
    # 5. 指令降维 (Commands)
    # ==========================================
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    
    if play:
        cfg.scene.num_envs = 1
        twist_cmd.ranges.lin_vel_x = (0.0, 0.0)
        twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
        twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
        twist_cmd.resampling_time_range = (1e9, 1e9)
    else:
        if cfg.scene.terrain.terrain_generator is not None:
            cfg.scene.terrain.terrain_generator.curriculum = True
            
        twist_cmd.ranges.lin_vel_x = (0.4, 1.2)  
        twist_cmd.ranges.lin_vel_y = (-0.1, 0.1) 
        twist_cmd.ranges.ang_vel_z = (-0.2, 0.2) 

    return cfg