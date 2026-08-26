import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from nazarite.config.robot_config.go2_cfg import (
  GO2_ACTION_SCALE,
  GO2_BASE_BODY,
  GO2_CALF_BODIES,
  GO2_FOOT_GEOMS,
  GO2_FOOT_SITES,
  GO2_THIGH_BODIES,
  get_go2_cfg,
)
from nazarite.config.train_config.base_env_cfg import make_base_env_cfg
from nazarite.mdp import rewards as custom_rewards


def Nazarite_Velocity_Flat_Go2(
  play: bool = False,
  enable_wtw: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the Nazarite Go2 flat-ground Grid Adaptive velocity task.

  默认返回不带 WTW 的 baseline；WTW 对照任务通过显式设置
  ``enable_wtw=True`` 创建。
  """
  cfg = make_base_env_cfg(
    enable_wtw=enable_wtw,
  )

  # ==========================================
  # 修改智能体（并行环境）数量
  # ==========================================
  # 训练时通常开 4096 或 8192，具体取决于你的显存大小 (VRAM)
  cfg.scene.num_envs = 4096
  # 环境之间的间隔，防止不同环境里的狗互相穿模看到彼此
  cfg.scene.env_spacing = 1.0

  # The base configuration is intentionally robot-agnostic. Bind Go2 here,
  # where the concrete trainable task is created.
  cfg.scene.entities = {"robot": get_go2_cfg()}

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = GO2_ACTION_SCALE

  ##
  # Sensors
  ##
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=GO2_FOOT_GEOMS,
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  foot_height_cfg = TerrainHeightSensorCfg(
    name="foot_height_scan",
    frame=tuple(
      ObjRef(type="site", name=site_name, entity="robot")
      for site_name in GO2_FOOT_SITES
    ),
    ray_alignment="yaw",
    max_distance=1.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),
    pattern=RingPatternCfg.single_ring(radius=0.03, num_samples=4),
  )

  terrain_match = ContactMatch(mode="body", pattern="terrain")
  thigh_ground_cfg = ContactSensorCfg(
    name="thigh_ground_touch",
    primary=ContactMatch(
      mode="body",
      pattern=GO2_THIGH_BODIES,
      entity="robot",
    ),
    secondary=terrain_match,
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  shank_ground_cfg = ContactSensorCfg(
    name="shank_ground_touch",
    primary=ContactMatch(
      mode="body",
      pattern=GO2_CALF_BODIES,
      entity="robot",
    ),
    secondary=terrain_match,
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  trunk_ground_cfg = ContactSensorCfg(
    name="trunk_ground_touch",
    primary=ContactMatch(
      mode="body",
      pattern=(GO2_BASE_BODY,),
      entity="robot",
    ),
    secondary=terrain_match,
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )

  cfg.scene.sensors = (
    foot_height_cfg,
    feet_ground_cfg,
    thigh_ground_cfg,
    shank_ground_cfg,
    trunk_ground_cfg,
  )

  # Fill the Go2-specific entity selectors left empty in the reusable base.
  ##
  # Events
  ##
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = GO2_FOOT_GEOMS
  cfg.events["base_com"].params["asset_cfg"].body_names = (GO2_BASE_BODY,)

  ##
  # Rewards
  ##
  if not enable_wtw:
    cfg.rewards["track_linear_velocity"].weight = 2.5
    cfg.rewards["track_angular_velocity"].weight = 2.0

  cfg.rewards["upright"].params["asset_cfg"].body_names = (GO2_BASE_BODY,)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (GO2_BASE_BODY,)
  cfg.rewards["body_ang_vel"].weight = -0.02
  cfg.rewards["base_height"] = RewardTermCfg(
    func=custom_rewards.base_height_reward,
    weight=1.0,
    params={
      "target_height": 0.32,
      "std": 0.04,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  if enable_wtw:
    # WTW 任务中不再使用 baseline 的固定 base_height 奖励。
    cfg.rewards.pop("base_height")
    # 独立奖励中的 stance width 和 Raibert 项都需要四个足端 site。
    cfg.rewards["wtw_stance_width"].params[
      "asset_cfg"
    ].site_names = GO2_FOOT_SITES
    cfg.rewards["wtw_raibert_foot_position"].params[
      "asset_cfg"
    ].site_names = GO2_FOOT_SITES
    # 速度任务也作为独立项交给 RewardManager 累加。
    cfg.rewards["track_linear_velocity"].weight = 2.0
    cfg.rewards["track_angular_velocity"].weight = 2.0
  # 官方 WTW 训练没有正向 default-pose 奖励；该项会把腿锁在默认
  # 姿态附近，和摆动/支撑行为目标竞争，容易形成高速小碎步。
  cfg.rewards["pose"].weight = 0.0 if enable_wtw else 1.0
  cfg.rewards["pose"].params["std_standing"] = {
    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.2,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.3,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.5,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.8,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.8,
    r".*(FR|FL|RR|RL)_calf_joint.*": 1.5,
  }
  # Match mjlab's flat Go2 penalty magnitudes. The previous joint acceleration
  # and joint-limit weights were too large for the old high-gain actuators and
  # dominated the positive tracking rewards during early exploration.
  cfg.rewards["dof_pos_limits"].weight = -0.2
  cfg.rewards["joint_acc_l2"].weight = -2.5e-7
  cfg.rewards["action_rate_l2"].weight = -0.005
  if not enable_wtw:
    cfg.rewards["air_time"].weight = 0.15
  cfg.rewards["soft_landing"].weight = -1.0e-5
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = GO2_FOOT_SITES
  cfg.rewards["foot_slip"].weight = -0.05

  cfg.viewer.body_name = GO2_BASE_BODY

  # Explicitly keep the standard timeout and orientation failure conditions.
  ##
  # Terminations
  ##
  cfg.terminations["time_out"] = TerminationTermCfg(
    func=mdp.time_out,
    time_out=True,
  )
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation,
    params={"limit_angle": math.radians(70.0)},
  )
  # Keep trunk_ground_touch as a diagnostic sensor, but do not terminate on
  # brief trunk contacts caused by pushes. Match mjlab's fall criterion by
  # terminating when a thigh contacts the terrain.
  cfg.terminations.pop("trunk_ground_touch", None)
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": "thigh_ground_touch"},
  )

  ##
  # Play
  ##
  if play:
    cfg.scene.num_envs = 1
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    # WTW 的 play 保留与训练一致的随机推力，便于直接检查抗扰动能力。
    # baseline 的 play 仍关闭推力，保持原有的纯动作观察方式。
    if not enable_wtw:
      cfg.events.pop("push_robot", None)
    cfg.curriculum = {}

  return cfg


def Nazarite_Velocity_Flat_Go2_WTW(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the Grid Adaptive + official-style independent WTW task.

  使用 Grid Adaptive 和 WTW 行为命令；速度任务、摆动相、支撑相及其他
  行为项拆成独立 reward term，便于在 TensorBoard 中分别观察各项贡献。
  """
  return Nazarite_Velocity_Flat_Go2(
    play=play,
    enable_wtw=True,
  )


def Nazarite_Velocity_Flat_Go2_No_WTW(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the Grid Adaptive Go2 baseline without WTW."""
  return Nazarite_Velocity_Flat_Go2(play=play, enable_wtw=False)
