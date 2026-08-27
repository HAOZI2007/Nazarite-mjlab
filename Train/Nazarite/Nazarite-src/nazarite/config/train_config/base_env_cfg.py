import math
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig
from nazarite.mdp import rewards as custom_rewards
from nazarite.mdp import wtw as custom_wtw
from nazarite.mdp import wtw_rewards as custom_wtw_rewards
from nazarite.mdp.commands import GridAdaptiveVelocityCommandCfg
from nazarite.mdp.wtw import WTWBehaviorCommandCfg


# 基础强化学习训练环境配置.
def make_base_env_cfg(
  enable_wtw: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the reusable flat-ground velocity tracking configuration.

  ``twist`` 始终使用 Grid Adaptive；``enable_wtw`` 只控制是否加入
  WTW 的 behavior/phase 观测、行为命令和行为辅助奖励。
  """

  ##
  # Sensors
  ##

  ##
  # Observations
  ##

  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"biased": True},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    # WTW 行为参数：步态时序、步频、身体姿态和摆腿高度。
    "behavior": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "behavior"},
    ),
    # 四足 sin/cos timing reference，顺序为 [FL, FR, RL, RR]。
    # 该 term 下面会显式设置 history_length=0，只输入当前 phase。
    "phase": ObservationTermCfg(
      func=custom_wtw.wtw_phase_reference,
      params={"command_name": "behavior"},
    ),
  }

  # baseline 任务只保留普通本体观测和 Grid Adaptive 速度命令。
  if not enable_wtw:
    actor_terms.pop("behavior")
    actor_terms.pop("phase")

  if enable_wtw:
    # 取消组级 history_length 后，才能为不同 observation term 设置不同历史。
    # actor 的普通观测保留 10 帧；behavior 只保留 5 帧；phase 使用当前帧。
    for term in actor_terms.values():
      term.history_length = 10
    actor_terms["behavior"].history_length = 5
    actor_terms["phase"].history_length = 0

  # 深拷贝，避免后面给 critic 设置 3 帧历史时修改 actor 的配置对象。
  critic_terms = deepcopy(actor_terms)
  critic_terms.update({
    # Critic sees the true (unbiased) joint positions as privileged information.
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    # Critic receives privileged body linear velocity, matching mjlab's
    # asymmetric velocity policy configuration. The actor remains blind to it.
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
    ),
    "base_height": ObservationTermCfg(
      func=custom_rewards.safe_base_height,
    ),
    "foot_height": ObservationTermCfg(
      func=custom_rewards.safe_foot_height,
      params={"sensor_name": "foot_height_scan"},
    ),
    "foot_air_time": ObservationTermCfg(
      func=custom_rewards.safe_foot_air_time,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact": ObservationTermCfg(
      func=custom_rewards.safe_foot_contact,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact_forces": ObservationTermCfg(
      func=custom_rewards.safe_foot_contact_forces,
      params={"sensor_name": "feet_ground_contact"},
    ),
  })

  if enable_wtw:
    # critic 的普通和特权观测使用 3 帧；behavior 仍保留 5 帧；phase 不堆叠。
    # 这一步放在 update() 之后，确保新加入的特权项也不会意外变成 0 帧。
    for term in critic_terms.values():
      term.history_length = 3
    critic_terms["behavior"].history_length = 5
    critic_terms["phase"].history_length = 0

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
      # 使用各个 term 自己的 history_length；组级设置不能覆盖 phase=0、behavior=5。
      history_length=None,
      flatten_history_dim=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      # 使用逐项历史配置：普通项为 3，behavior 为 5，phase 为 0。
      history_length=None,
      flatten_history_dim=True,
    ),
  }

  ##
  # Metrics
  ##

  metrics = {
    "mean_action_acc": MetricsTermCfg(
      func=mdp.mean_action_acc,
    ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.5,  # Override per-robot.
      use_default_offset=True,
    )
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    "twist": GridAdaptiveVelocityCommandCfg(
      entity_name="robot",
      # 一个 episode 使用一个速度网格，便于把成功/失败归因到当前 cell。
      resampling_time_range=(10.0, 20.0),
      # 保留 10% 零速度站立任务；该任务不参与速度网格成功率统计。
      rel_standing_envs=0.3,
      rel_heading_envs=0.0,
      rel_forward_envs=0.0,
      heading_command=False,
      grid_num_x=9,
      grid_num_yaw=7,
      # 初始 cell 覆盖接近零速度的区域；课程成功后向四周扩展。
      initial_cell=(3, 3),
      min_cell_visits=100,
      success_window_size=100,
      max_new_cells_per_update=1,
      success_rate_threshold=0.8,
      velocity_error_threshold=0.35,
      yaw_error_threshold=0.35,
      debug_vis=True,
      ranges=GridAdaptiveVelocityCommandCfg.Ranges(
        # 这是课程最终覆盖范围，而不是每个 episode 的采样范围。
        lin_vel_x=(-2.0, 2.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-0.7, 0.7),
        heading=None,
      ),
    ),
    # WTW 行为在一个 episode 内保持，避免步态和速度同时频繁跳变。
    "behavior": WTWBehaviorCommandCfg(
      entity_name="robot",
      resampling_time_range=(30.0, 30.0),
      frequency_range=(2.0, 2.0),
      body_height_range=(0.32, 0.32),
      body_pitch_range=(0.0, 0.0),
      stance_width_range=(0.21, 0.21),
      foot_swing_height_range=(0.07, 0.07),
      # 第一阶段先固定 trot，待速度追踪和接触时序稳定后再开放其他步态。
      gait_names=("trot",),
      debug_vis=False,
    ),
  }

  if not enable_wtw:
    # 两个任务都保留 twist，因此 Grid Adaptive 的采样和课程逻辑完全一致。
    commands.pop("behavior")

  ##
  # Events
  ##

  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (0.01, 0.05),
          "yaw": (-3.14, 3.14),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.4, 0.4),
          "roll": (-0.52, 0.52),
          "pitch": (-0.52, 0.52),
          "yaw": (-0.78, 0.78),
        },
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),
        "operation": "abs",
        "ranges": (0.3, 1.2),
        "shared_random": True,  # All foot geoms share the same friction.
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.015, 0.015),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),
        "operation": "add",
        "ranges": {
          0: (-0.025, 0.025),
          1: (-0.025, 0.025),
          2: (-0.03, 0.03),
        },
      },
    ),
  }

  ##
  # Rewards
  ##

  rewards = {
    "track_linear_velocity": RewardTermCfg(
      func=custom_rewards.track_linear_velocity,
      weight=2.5,
      params={
        "command_name": "twist",
        "std": math.sqrt(0.25),
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "track_angular_velocity": RewardTermCfg(
      func=custom_rewards.track_angular_velocity,
      weight=2.0,
      params={
        "command_name": "twist",
        "std": math.sqrt(0.5),
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "upright": RewardTermCfg(
      func=custom_rewards.upright,
      weight=1.0,
      params={
        "std": math.sqrt(0.2),
        "asset_cfg": SceneEntityCfg("robot", body_names=()),
      },
    ),
    "pose": RewardTermCfg(
      func=mdp.variable_posture,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "command_name": "twist",
        "std_standing": {},
        "std_walking": {},
        "std_running": {},
        "walking_threshold": 0.05,
        "running_threshold": 1.5,
      },
    ),
    "body_ang_vel": RewardTermCfg(
      func=custom_rewards.body_angular_velocity_penalty,
      weight=0.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),
      },
    ),
    "dof_pos_limits": RewardTermCfg(
      func=custom_rewards.joint_pos_limits,
      weight=-1.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "joint_acc_l2": RewardTermCfg(
      func=custom_rewards.joint_acc_l2,
      weight=-1.0e-5,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "joint_torques_l2": RewardTermCfg(
      func=custom_rewards.joint_torques_l2,
      weight=-1.0e-4,
      params={"asset_cfg": SceneEntityCfg("robot", actuator_names=".*")},
    ),
    "action_rate_l2": RewardTermCfg(
      func=custom_rewards.action_rate_l2,
      weight=-0.05,
    ),
    "air_time": RewardTermCfg(
      func=custom_rewards.feet_air_time,
      weight=2.0,
      params={
        "sensor_name": "feet_ground_contact",
        "threshold": 0.1,
        "command_name": "twist",
        "command_threshold": 0.05,
      },
    ),
    "prolonged_air_time": RewardTermCfg(
      func=custom_rewards.prolonged_air_time,
      weight=-0.5,
      params={
        "sensor_name": "feet_ground_contact",
        "max_air_time": 0.3,
      },
    ),
    "stance_contact": RewardTermCfg(
      func=custom_rewards.feet_stance_contact,
      weight=-0.5,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.05,
        "force_threshold": 5.0,
      },
    ),
    "foot_slip": RewardTermCfg(
      func=custom_rewards.feet_slip,
      weight=-0.05,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),
      },
    ),
    "soft_landing": RewardTermCfg(
      func=custom_rewards.soft_landing,
      weight=-1.0e-4,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.05,
      },
    ),
    "stand_pose": RewardTermCfg(
      func=custom_rewards.zero_command_pose_penalty,
      weight=-1.0,
      params={
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    # WTW 的零速度站立约束：同时抑制机体线速度、角速度和关节速度。
    # 函数返回正的 cost，因此这里使用负权重；行走指令下该项自动为 0。
    "zero_command_stillness": RewardTermCfg(
      func=custom_rewards.zero_command_stillness,
      weight=0.0,
      params={
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "linear_velocity_weight": 1.0,
        "angular_velocity_weight": 1.0,
        "joint_velocity_weight": 0.05,
      },
    ),
    # 官方式独立奖励：摆动相要求足端接触力尽可能小。
    # 函数返回正的代价，因此使用负权重。
    "wtw_swing_phase_force": RewardTermCfg(
      func=custom_wtw_rewards.wtw_swing_phase_contact_cost,
      weight=-4.0,
      params={
        "sensor_name": "feet_ground_contact",
        "behavior_command_name": "behavior",
        "command_name": "twist",
        "command_threshold": 0.05,
        "smoothing": 0.15,
        "force_std": 100.0,
      },
    ),
    # 官方式独立奖励：支撑相要求足端水平速度尽可能小。
    "wtw_stance_phase_velocity": RewardTermCfg(
      func=custom_wtw_rewards.wtw_stance_phase_velocity_cost,
      weight=-4.0,
      params={
        "sensor_name": "feet_ground_contact",
        "behavior_command_name": "behavior",
        "command_name": "twist",
        "command_threshold": 0.05,
        "smoothing": 0.15,
        "velocity_std": 10.0,
      },
    ),
    "wtw_body_height": RewardTermCfg(
      func=custom_wtw_rewards.wtw_body_height,
      weight=0.10,
      params={
        "behavior_command_name": "behavior",
        "asset_cfg": SceneEntityCfg("robot"),
        "std": 0.035,
      },
    ),
    "wtw_body_pitch": RewardTermCfg(
      func=custom_wtw_rewards.wtw_body_pitch,
      weight=0.03,
      params={
        "behavior_command_name": "behavior",
        "asset_cfg": SceneEntityCfg("robot"),
        "std": 0.08,
        "command_name": "twist",
        "command_threshold": 0.05,
      },
    ),
    "wtw_stance_width": RewardTermCfg(
      func=custom_wtw_rewards.wtw_stance_width,
      weight=0.03,
      params={
        "asset_cfg": SceneEntityCfg("robot", site_names=()),
        "behavior_command_name": "behavior",
        "command_name": "twist",
        "command_threshold": 0.05,
        "std": 0.035,
      },
    ),
    "wtw_foot_swing_height": RewardTermCfg(
      func=custom_wtw_rewards.wtw_foot_swing_height,
      weight=0.10,
      params={
        "sensor_name": "feet_ground_contact",
        "height_sensor_name": "foot_height_scan",
        "behavior_command_name": "behavior",
        "command_name": "twist",
        "command_threshold": 0.05,
        "std": 0.04,
        "smoothing": 0.15,
      },
    ),
    "wtw_raibert_foot_position": RewardTermCfg(
      func=custom_wtw_rewards.wtw_raibert_foot_position,
      # Raibert 项只做小幅辅助，避免压过速度追踪和接触时序。
      weight=0.03,
      params={
        "asset_cfg": SceneEntityCfg("robot", site_names=()),
        "behavior_command_name": "behavior",
        "command_name": "twist",
        "command_threshold": 0.05,
        "std": 0.08,
        "smoothing": 0.15,
      },
    ),
  }

  if not enable_wtw:
    # baseline 不使用任何 WTW 行为辅助奖励，避免对照实验混入隐性行为约束。
    for reward_name in (
      "zero_command_stillness",
      "wtw_swing_phase_force",
      "wtw_stance_phase_velocity",
      "wtw_body_height",
      "wtw_body_pitch",
      "wtw_stance_width",
      "wtw_foot_swing_height",
      "wtw_raibert_foot_position",
    ):
      rewards.pop(reward_name)

  else:
    # 这些奖励没有使用 WTW phase，容易与行为参数给出的摆动/支撑节奏竞争。
    for reward_name in ("air_time", "prolonged_air_time", "stance_contact"):
      rewards.pop(reward_name)

    # WTW 任务始终采用独立奖励：速度任务和各个行为项分别交给
    # RewardManager 累加，便于单独调参和观察每一项的贡献。
    rewards["wtw_swing_phase_force"].weight = -4.0
    rewards["wtw_swing_phase_force"].params["force_std"] = 100.0
    rewards["wtw_stance_phase_velocity"].weight = -4.0
    rewards["wtw_stance_phase_velocity"].params["velocity_std"] = 10.0

    rewards["wtw_body_height"].weight = 0.80
    rewards["wtw_body_pitch"].weight = 0.10
    rewards["wtw_stance_width"].weight = 0.03
    rewards["wtw_foot_swing_height"].weight = 0.60
    rewards["wtw_raibert_foot_position"].weight = 0.20

  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
  }

  ##
  # Curriculum
  ##

  # 速度课程已经由 GridAdaptiveVelocityCommand 管理。
  # 不再同时启用固定时间阶段课程，避免两个课程系统同时修改速度范围。
  curriculum = {}

  ##
  # Assemble and return
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      sensors=(),
      entities={},
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    metrics=metrics,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=1500,
      mujoco=MujocoCfg(
        timestep=0.002,
        iterations=10,
        ls_iterations=30,
      ),
    ),
    decimation=10,
    episode_length_s=30.0,
  )
