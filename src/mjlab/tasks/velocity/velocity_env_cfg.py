"""Velocity task configuration.

This module provides a factory function to create a base velocity task config.
Robot-specific configurations call the factory and customize as needed.
"""

import math
from dataclasses import dataclass, replace
from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig


@dataclass(frozen=True)
class VelocityContactSensorSpec:
  """Robot-specific contact sensor description for velocity tasks."""

  name: str
  primary: ContactMatch
  secondary: ContactMatch | None = None
  fields: tuple[str, ...] = ("found", "force")
  reduce: Literal["none", "mindist", "maxforce", "netforce"] = "none"
  num_slots: int = 1
  track_air_time: bool = False
  history_length: int = 0

  def build(self) -> ContactSensorCfg:
    """Build a fresh contact sensor configuration."""
    return ContactSensorCfg(
      name=self.name,
      primary=self.primary,
      secondary=self.secondary,
      fields=self.fields,
      reduce=self.reduce,
      num_slots=self.num_slots,
      track_air_time=self.track_air_time,
      history_length=self.history_length,
    )


@dataclass(frozen=True)
class VelocityRobotSpec:
  """Robot-specific names and scales shared by velocity task variants."""

  body_name: str
  foot_site_names: tuple[str, ...]
  foot_geom_names: tuple[str, ...]
  foot_contact: ContactMatch
  self_collision: ContactMatch
  action_scale: float | dict[str, float]
  foot_scan_radius: float
  foot_scan_samples: int
  extra_contact_sensors: tuple[VelocityContactSensorSpec, ...] = ()
  inertial_body_names: tuple[str, ...] = ()


def _configure_robot_specific_terms(
  cfg: ManagerBasedRlEnvCfg,
  robot: VelocityRobotSpec,
) -> None:
  """Apply robot-specific sensor, reward, action, and viewer settings."""
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = robot.body_name
    elif sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=name, entity="robot") for name in robot.foot_site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(
        radius=robot.foot_scan_radius,
        num_samples=robot.foot_scan_samples,
      )

  terrain_match = ContactMatch(mode="body", pattern="terrain")
  feet_ground = ContactSensorCfg(
    name="feet_ground_contact",
    primary=robot.foot_contact,
    secondary=terrain_match,
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision = ContactSensorCfg(
    name="self_collision",
    primary=robot.self_collision,
    secondary=robot.self_collision,
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground,
    self_collision,
    *(sensor.build() for sensor in robot.extra_contact_sensors),
  )

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = robot.action_scale
  cfg.viewer.body_name = robot.body_name

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = robot.foot_geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = (robot.body_name,)
  cfg.rewards["upright"].params["asset_cfg"].body_names = (robot.body_name,)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (robot.body_name,)
  for reward_name in ("foot_clearance", "foot_slip"):
    cfg.rewards[reward_name].params["asset_cfg"].site_names = robot.foot_site_names


def _configure_staged_velocity_curriculum(
  cfg: ManagerBasedRlEnvCfg,
  robot: VelocityRobotSpec,
) -> None:
  """Install the staged flat-ground velocity curriculum."""
  if not robot.inertial_body_names:
    raise ValueError(
      "Staged velocity curriculum requires inertial_body_names in the robot spec."
    )

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.resampling_time_range = (5.0, 5.0)
  twist_cmd.rel_standing_envs = 0.15
  twist_cmd.velocity_buckets = (
    (-1.0, -0.5, 0.10),
    (0.0, 0.0, 0.15),
    (0.5, 0.5, 0.20),
    (1.0, 1.0, 0.25),
    (1.5, 1.5, 0.15),
    (2.0, 2.0, 0.15),
  )
  twist_cmd.high_speed_metric_threshold = 1.8
  twist_cmd.direct_start_speed_threshold = 1.8
  twist_cmd.direct_start_window_s = 2.0
  twist_cmd.direct_start_tracking_tolerance = 0.35
  twist_cmd.direct_start_success_ratio = 0.7
  twist_cmd.ranges.lin_vel_x = (-1.0, 3.0)
  twist_cmd.ranges.lin_vel_y = (-2.0, 2.0)
  twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)

  cfg.events["push_robot"].func = mdp.staged_push_robot
  cfg.events["body_mass"].mode = "reset"
  cfg.events["body_mass"].func = mdp.staged_pseudo_inertia
  cfg.events["body_mass"].params["asset_cfg"].body_names = robot.inertial_body_names
  cfg.events["base_mass_offset"] = EventTermCfg(
    mode="reset",
    func=mdp.staged_body_mass,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(robot.body_name,)),
      "operation": "add",
      "ranges": (-1.0, 2.0),
    },
  )
  cfg.curriculum["command_vel"] = CurriculumTermCfg(
    func=mdp.StagedVelocityCommand,
    params={
      "command_name": "twist",
      "stages": [
        {
          "lin_vel_x": (-1.0, 1.0),
          "lin_vel_y": (-0.3, 0.3),
          "ang_vel_z": (-0.5, 0.5),
          "randomization_scale": 0.35,
        },
        {
          "lin_vel_x": (-1.0, 1.5),
          "lin_vel_y": (-0.4, 0.4),
          "ang_vel_z": (-0.6, 0.6),
          "randomization_scale": 0.60,
        },
        {
          "lin_vel_x": (-1.0, 2.0),
          "lin_vel_y": (-0.5, 0.5),
          "ang_vel_z": (-0.7, 0.7),
          "randomization_scale": 1.0,
        },
      ],
      "min_phase_episodes": 16384,
      "evaluation_window_episodes": 8192,
      "required_windows": 2,
      "survival_threshold": 0.95,
      "track_threshold": 0.75,
      "high_speed_track_threshold": 0.65,
      "direct_start_threshold": 1.8,
      "direct_start_success_threshold": 0.65,
      # Do not advance a stage while numerical protection is firing. During a
      # randomized phase, the curriculum backs off and ramps back up gradually.
      "max_invalid_rate": 0.0,
      "max_action_outlier_rate": 0.0,
      "max_state_limit_rate": 0.0,
      "max_nan_rate": 0.0,
    },
  )


def make_velocity_env_cfg(
  asymmetric: bool = True,
  robot: VelocityRobotSpec | None = None,
  staged_velocity_curriculum: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create base velocity tracking task configuration.

  Args:
    asymmetric: If True (default), ``height_scan`` is only included in the
      critic observation group — the actor is blind to terrain and must
      infer it from proprioception.  If False, ``height_scan`` is included
      in both actor and critic (symmetric architecture).
    robot: Optional robot-specific names and scales. When provided, the
      factory creates common contact sensors and applies robot-specific
      action, reward, and viewer settings.
    staged_velocity_curriculum: Add the staged flat-ground speed curriculum.
  """

  ##
  # Sensors
  ##

  terrain_scan = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="", entity="robot"),  # Set per-robot.
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
    max_distance=5.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),  # Terrain only.
    debug_vis=True,
  )

  foot_height_scan = TerrainHeightSensorCfg(
    name="foot_height_scan",
    frame=(),  # Set per-robot: frame and pattern.
    ray_alignment="yaw",
    max_distance=1.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),  # Terrain only.
    debug_vis=True,
    viz=TerrainHeightSensorCfg.VizCfg(
      show_rays=True,
      hit_color=(1.0, 0.0, 1.0, 0.8),  # Magenta rays.
      hit_sphere_color=(1.0, 0.0, 1.0, 1.0),
    ),
  )

  ##
  # Observations
  ##

  # ── 本体感知项 (proprioceptive) — Actor 与 Critic 共享 ────────
  proprioceptive_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
      delay_min_lag=1,
      delay_max_lag=3,
      delay_hold_prob=0.9,
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
      delay_min_lag=1,
      delay_max_lag=3,
      delay_hold_prob=0.9,
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
      delay_min_lag=1,
      delay_max_lag=3,
      delay_hold_prob=0.9,
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
    ),
  }

  # ── 地形感知项 (exteroceptive) — 仅限非对称模式下 Critic 可见 ──
  # Actor 在非对称模式下不包含 height_scan，迫使其通过本体感知推断地形。
  height_scan_noisy = ObservationTermCfg(
    func=envs_mdp.height_scan,
    params={"sensor_name": "terrain_scan"},
    noise=Unoise(n_min=-0.1, n_max=0.1),
    scale=1 / terrain_scan.max_distance,
  )
  height_scan_clean = ObservationTermCfg(
    func=envs_mdp.height_scan,
    params={"sensor_name": "terrain_scan"},
    scale=1 / terrain_scan.max_distance,
  )

  # ── 特权观测项 (privileged) — 仅 Critic 可见 ──────────────────
  # 线速度：实际机体无法直接感知（无精确里程计），只提供给 Critic
  # 以学习更准确的价值函数。
  privileged_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
    ),
    "foot_height": ObservationTermCfg(
      func=mdp.foot_height,
      params={"sensor_name": "foot_height_scan"},
    ),
    "foot_air_time": ObservationTermCfg(
      func=mdp.foot_air_time,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact_forces": ObservationTermCfg(
      func=mdp.foot_contact_forces,
      params={"sensor_name": "feet_ground_contact"},
    ),
  }

  # ── 组装 Actor / Critic 观测组 ──────────────────────────────
  if asymmetric:
    actor_terms = proprioceptive_terms
  else:
    actor_terms = {**proprioceptive_terms, "height_scan": height_scan_noisy}

  critic_terms = {
    **proprioceptive_terms,
    "height_scan": height_scan_clean,
    **privileged_terms,
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=10,
      history_ordering="time",
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=3,
      history_ordering="time",
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
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 8.0),
      rel_standing_envs=0.1,
      rel_heading_envs=0.3,
      rel_forward_envs=0.2,
      heading_command=True,
      heading_control_stiffness=0.5,
      debug_vis=True,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.0, 1.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-0.5, 0.5),
        heading=(-math.pi, math.pi),
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
      interval_range_s=(1.0, 5.0),
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
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
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
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "operation": "add",
        "ranges": {
          0: (-0.025, 0.025),
          1: (-0.025, 0.025),
          2: (-0.03, 0.03),
        },
      },
    ),
    # Keep mass and inertia physically consistent while covering payload and
    # manufacturing variation. Robot-specific configs may narrow body_names.
    "body_mass": EventTermCfg(
      mode="startup",
      func=dr.pseudo_inertia,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
        "alpha_range": (math.log(math.sqrt(0.9)), math.log(math.sqrt(1.1))),
      },
    ),
    "joint_damping": EventTermCfg(
      mode="startup",
      func=dr.joint_damping,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "operation": "scale",
        "ranges": (0.9, 1.1),
      },
    ),
    "pd_gains": EventTermCfg(
      mode="startup",
      func=dr.pd_gains,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "operation": "scale",
        "kp_range": (0.9, 1.1),
        "kd_range": (0.9, 1.1),
      },
    ),
    "effort_limits": EventTermCfg(
      mode="startup",
      func=dr.effort_limits,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "operation": "scale",
        "effort_limit_range": (0.9, 1.1),
      },
    ),
  }

  ##
  # Rewards
  ##

  rewards = {
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=2.5,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=2.0,
      params={"command_name": "twist", "std": math.sqrt(0.5)},
    ),
    "upright": RewardTermCfg(
      func=mdp.upright,
      weight=1.0,
      params={
        "std": math.sqrt(0.2),
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
      },
    ),
    "pose": RewardTermCfg(
      func=mdp.variable_posture,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "command_name": "twist",
        "std_standing": {},  # Set per-robot.
        "std_walking": {},  # Set per-robot.
        "std_running": {},  # Set per-robot.
        "walking_threshold": 0.05,
        "running_threshold": 1.5,
      },
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=0.0,  # Override per-robot
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot.
    ),
    "angular_momentum": RewardTermCfg(
      func=mdp.angular_momentum_penalty,
      weight=0.0,  # Override per-robot
      params={"sensor_name": "robot/root_angmom"},
    ),
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-0.2),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
    "joint_acc_l2": RewardTermCfg(
      func=envs_mdp.joint_acc_l2,
      weight=-0.0001,
    ),
    "torques_l2": RewardTermCfg(
      func=envs_mdp.joint_torques_l2,
      weight=-0.0002,
    ),
    "dof_vel": RewardTermCfg(
      func=envs_mdp.joint_vel_l2,
      weight=-0.001,
    ),
    "air_time": RewardTermCfg(
      func=mdp.feet_air_time,
      weight=0.0,  # Override per-robot.
      params={
        "sensor_name": "feet_ground_contact",
        "threshold": 0.1,
        "command_name": "twist",
        "command_threshold": 0.5,
      },
    ),
    "stance_contact": RewardTermCfg(
      func=mdp.feet_stance_contact,
      weight=0.0,  # Override per-robot.
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.05,
        "force_threshold": 5.0,
      },
    ),
    "prolonged_air_time": RewardTermCfg(
      func=mdp.prolonged_air_time,
      weight=0.0,  # Override per-robot.
      params={
        "sensor_name": "feet_ground_contact",
        "max_air_time": 0.3,
      },
    ),
    "foot_clearance": RewardTermCfg(
      func=mdp.feet_clearance,
      weight=-0.2,
      params={
        "target_height": 0.1,
        "height_sensor_name": "foot_height_scan",
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "foot_swing_height": RewardTermCfg(
      func=mdp.feet_swing_height,
      weight=-0.2,
      params={
        "sensor_name": "feet_ground_contact",
        "height_sensor_name": "foot_height_scan",
        "target_height": 0.1,
        "command_name": "twist",
        "command_threshold": 0.05,
      },
    ),
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-0.1,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "soft_landing": RewardTermCfg(
      func=mdp.soft_landing,
      weight=-1e-5,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.05,
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
    "out_of_terrain_bounds": TerminationTermCfg(
      func=mdp.out_of_terrain_bounds,
      time_out=True,
    ),
    # Numerical safety terms are part of the shared velocity-task skeleton so
    # flat and rough variants both isolate unstable environments.
    "state_limit": TerminationTermCfg(
      func=envs_mdp.state_limit,
      params={
        "max_joint_vel": 60.0,
        "max_joint_acc": 5000.0,
        "max_root_lin_vel": 10.0,
        "max_root_ang_vel": 30.0,
      },
    ),
    "nan_detection": TerminationTermCfg(func=envs_mdp.nan_detection),
  }

  ##
  # Curriculum
  ##

  curriculum = {
    "terrain_levels": CurriculumTermCfg(
      func=mdp.terrain_levels_vel,
      params={"command_name": "twist"},
    ),
    "command_vel": CurriculumTermCfg(
      func=mdp.commands_vel,
      params={
        "command_name": "twist",
        "velocity_stages": [
          {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
          {"step": 5000 * 24, "lin_vel_x": (-1.5, 2.0), "ang_vel_z": (-0.7, 0.7)},
          {"step": 10000 * 24, "lin_vel_x": (-2.0, 3.0)},
        ],
      },
    ),
  }

  ##
  # Assemble and return
  ##

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=replace(ROUGH_TERRAINS_CFG),
        max_init_terrain_level=5,
      ),
      sensors=(terrain_scan, foot_height_scan),
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
      body_name="",  # Set per-robot.
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=1500,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=20.0,
    strict_reward_checks=True,
    action_safety_enabled=True,
    action_safety_max_abs=5.0,
  )

  if robot is not None:
    _configure_robot_specific_terms(cfg, robot)
    if staged_velocity_curriculum:
      _configure_staged_velocity_curriculum(cfg, robot)

  return cfg
