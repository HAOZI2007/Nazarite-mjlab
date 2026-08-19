import math

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
from nazarite.mdp.commands import GridAdaptiveVelocityCommandCfg


# 基础强化学习训练环境配置.
def make_base_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create the reusable base velocity tracking configuration."""

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
  }

  critic_terms = {
    **actor_terms,
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
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
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
      resampling_time_range=(8.0, 12.0),
      # 保留 10% 零速度站立任务；该任务不参与速度网格成功率统计。
      rel_standing_envs=0.1,
      rel_heading_envs=0.0,
      rel_forward_envs=0.0,
      heading_command=False,
      grid_num_x=9,
      grid_num_yaw=7,
      # 初始 cell 覆盖接近零速度的区域；课程成功后向四周扩展。
      initial_cell=(3, 3),
      min_cell_visits=50,
      success_window_size=100,
      max_new_cells_per_update=4,
      success_rate_threshold=0.8,
      velocity_error_threshold=0.35,
      yaw_error_threshold=0.35,
      debug_vis=True,
      ranges=GridAdaptiveVelocityCommandCfg.Ranges(
        # 这是课程最终覆盖范围，而不是每个 episode 的采样范围。
        lin_vel_x=(-2.0, 3.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-0.7, 0.7),
        heading=None,
      ),
    )
  }

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
  }

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
