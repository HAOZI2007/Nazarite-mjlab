"""Isolated WTW+DELTA terrain-adaptation environment."""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.terrains import TerrainEntityCfg
from nazarite.config.train_config.env_cfgs.delta_go2_env_cfgs import (
  DELTA_DEPTH_CAMERA,
  DELTA_TERRAINS_CFG,
)
from nazarite.mdp.commands import (
  ScheduledVelocityStageCfg,
  TerrainConditionedVelocityCommandCfg,
)
from nazarite.config.train_config.env_cfgs.go2_env_cfgs import (
  Nazarite_Velocity_Flat_Go2_WTW,
)
from nazarite.delta.terrains import (
  BoxApproachSteppingStonesTerrainCfg,
  BoxStripedGapsTerrainCfg,
)
from nazarite.mdp import curriculums as custom_curriculums
from nazarite.mdp import rewards as custom_rewards
from nazarite.mdp.delta_observations import delta_depth_image

_DELTA_PROPRIO_NAMES = (
  "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions", "command",
  # Terrain corrections must be phase-aware: without these terms DELTA cannot
  # tell which legs are swinging or what body/swing height WTW is targeting.
  "behavior", "phase",
)


def _term_group(terms: dict[str, ObservationTermCfg], names: tuple[str, ...], history: int) -> dict[str, ObservationTermCfg]:
  selected = {name: deepcopy(terms[name]) for name in names}
  for term in selected.values():
    term.history_length = history
    term.flatten_history_dim = True
  return selected


def Nazarite_Wtw_Delta_Go2(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create a separate WTW-prior/DELTA-residual task.

  The original WTW and DELTA factories are only used as deep-copy sources;
  their returned configurations are never mutated after this function returns.
  """
  cfg = Nazarite_Velocity_Flat_Go2_WTW(play=play)
  cfg = deepcopy(cfg)
  # WTW supplies the locomotion prior and its velocity tracking is pretrained.
  # DELTA first learns low-speed terrain corrections, then progressively opens
  # lateral/yaw coverage. Stepping stones remain positive-x only below.
  cfg.commands["twist"] = TerrainConditionedVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(4.0, 6.0),
    rel_standing_envs=0.1,
    rel_heading_envs=0.0,
    rel_world_envs=0.0,
    rel_forward_envs=0.0,
    heading_command=False,
    ranges=TerrainConditionedVelocityCommandCfg.Ranges(
      lin_vel_x=(-1.0, 1.0),
      lin_vel_y=(-0.5, 0.5),
      ang_vel_z=(-0.5, 0.5),
      heading=None,
    ),
    # common_step_counter is in control steps; with 24 rollout steps this is
    # approximately 3k, 8k and 15k PPO iterations respectively.
    stages=(
      ScheduledVelocityStageCfg(
        step=0,
        lin_vel_x=(-0.30, 0.30),
        lin_vel_y=(0.0, 0.0),
        ang_vel_z=(-0.15, 0.15),
      ),
      ScheduledVelocityStageCfg(
        step=72_000,
        lin_vel_x=(-0.50, 0.50),
        lin_vel_y=(0.0, 0.0),
        ang_vel_z=(-0.25, 0.25),
      ),
      ScheduledVelocityStageCfg(
        step=192_000,
        lin_vel_x=(-0.80, 0.80),
        lin_vel_y=(-0.35, 0.35),
        ang_vel_z=(-0.35, 0.35),
      ),
      ScheduledVelocityStageCfg(
        step=360_000,
        lin_vel_x=(-1.00, 1.00),
        lin_vel_y=(-0.50, 0.50),
        ang_vel_z=(-0.50, 0.50),
      ),
    ),
    forward_only_terrain_names=("stepping_stones",),
    forward_only_x_min=0.10,
  )
  if not play:
    # The depth camera and DELTA patches are more expensive than the original
    # WTW observation; keep this new task's batch size independent of WTW.
    cfg.scene.num_envs = 256
  if not play:
    cfg.episode_length_s = 10.0

  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=deepcopy(DELTA_TERRAINS_CFG),
    max_init_terrain_level=0,
  )
  terrain_generator = cfg.scene.terrain.terrain_generator
  assert terrain_generator is not None
  terrain_generator.sub_terrains["gaps"] = BoxStripedGapsTerrainCfg(
    proportion=0.125,
    # Stage-0 gaps are intentionally easy; the terrain curriculum widens them
    # only after repeated successful traversals.
    strip_width_range=(0.95, 0.75),
    gap_width_range=(0.04, 0.10),
    center_platform_width=0.9,
    border_width=0.25,
    floor_depth=1.0,
  )
  terrain_generator.sub_terrains["stepping_stones"] = BoxApproachSteppingStonesTerrainCfg(
    proportion=0.125,
    stone_size_range=(0.80, 1.00),
    stone_distance_range=(0.05, 0.12),
    stone_height=0.06,
    stone_height_variation=0.02,
    stone_size_variation=0.05,
    displacement_range=0.02,
    floor_depth=2.0,
    platform_width=0.0,
    approach_platform_length=1.20,
    approach_platform_width=1.40,
    approach_gap=0.18,
    border_width=0.25,
  )
  reset_pose_range = cfg.events["reset_base"].params["pose_range"]
  reset_pose_range["x"] = (-0.20, 0.20)
  reset_pose_range["y"] = (-0.30, 0.30)
  reset_pose_range["yaw"] = (-0.15, 0.15)
  cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + (deepcopy(DELTA_DEPTH_CAMERA),)

  # Keep malformed or unusually large policy outputs from corrupting PPO
  # returns. This override is local to WTW+DELTA; the source WTW task remains
  # unchanged.
  cfg.rewards["action_acc_l2"].func = custom_rewards.action_acc_l2
  # Sparse terrain needs room for body-height and foot-placement adaptation.
  # These are local deep-copied terms; the original WTW task keeps its stronger
  # flat-ground gait-shaping weights unchanged.
  cfg.rewards["wtw_body_height"].weight = 12.0
  cfg.rewards["wtw_foot_clearance"].weight = -12.0
  cfg.rewards["wtw_raibert_foot_position"].weight = -4.0
  cfg.rewards["delta_swing_clearance"] = RewardTermCfg(
    func=custom_rewards.delta_swing_clearance_cost,
    # Normalize the adaptive deficit inside the reward; this weight is strong
    # enough to teach obstacle clearance without competing with WTW's gait
    # terms on flat ground.
    weight=-4.0,
    params={
      "height_sensor_name": "foot_height_scan",
      "contact_sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "command_threshold": 0.05,
      "minimum_clearance": 0.055,
      "obstacle_gain": 0.45,
      "max_obstacle_extra": 0.12,
      "forward_x_min": -0.25,
      "min_map_valid_ratio": 0.35,
      "min_obstacle_relief": 0.025,
      "clearance_std": 0.04,
    },
  )

  wtw_actor_terms = deepcopy(cfg.observations["actor"].terms)
  wtw_critic_terms = deepcopy(cfg.observations["critic"].terms)
  delta_proprio_terms = _term_group(wtw_actor_terms, _DELTA_PROPRIO_NAMES, 0)
  delta_proprio_critic_terms = _term_group(wtw_critic_terms, _DELTA_PROPRIO_NAMES, 0)
  delta_map_term = {
    "delta_map": ObservationTermCfg(
      func=delta_depth_image,
      params={
        "sensor_name": "delta_depth_camera",
        "map_height": 16,
        "map_width": 26,
        "project_to_bev": True,
        "map_channels": 4,
        # Concentrate the 16x26 grid on the near field so D435 points occupy
        # more cells instead of being diluted across a mostly empty map.
        "x_range_m": (0.0, 2.5),
        "y_range_m": (-0.8, 0.8),
      },
    ),
  }

  # Separate observation groups let the composite model preserve WTW's full
  # history while feeding DELTA one current proprioception/behavior/phase
  # frame. The model derives this dimension from the observation manager.
  cfg.observations = {
    "wtw_proprio": ObservationGroupCfg(
      terms=wtw_actor_terms, concatenate_terms=True,
      enable_corruption=not play, history_length=None, flatten_history_dim=True,
    ),
    "critic_privileged": ObservationGroupCfg(
      terms=wtw_critic_terms, concatenate_terms=True,
      enable_corruption=False, history_length=None, flatten_history_dim=True,
    ),
    "delta_proprio": ObservationGroupCfg(
      terms=delta_proprio_terms, concatenate_terms=True,
      enable_corruption=not play, history_length=None, flatten_history_dim=True,
    ),
    "delta_proprio_critic": ObservationGroupCfg(
      terms=delta_proprio_critic_terms, concatenate_terms=True,
      enable_corruption=False, history_length=None, flatten_history_dim=True,
    ),
    "delta_map": ObservationGroupCfg(
      terms=delta_map_term, concatenate_terms=True,
      enable_corruption=False, history_length=None, flatten_history_dim=True,
    ),
  }

  # Reuse the paper-style stage curriculum only in this new task.
  cfg.curriculum = {
    "terrain_levels": CurriculumTermCfg(
      func=custom_curriculums.terrain_levels_delta,
      params={
        "command_name": "twist",
        "success_distance_scale": 0.8,
        "min_success_distance": 0.8,
        "max_success_distance": 1.6,
        "success_streak_length": 2,
        "failure_streak_length": 3,
        "sparse_terrain_names": ("gaps", "stepping_stones", "grid_stones"),
        "sparse_success_distance_scale": 0.5,
        "sparse_min_success_distance": 0.5,
        "sparse_max_success_distance": 1.2,
        "sparse_success_streak_length": 3,
        "sparse_failure_streak_length": 4,
        "sparse_max_level": 4,
      },
    ),
  }
  if play:
    cfg.scene.num_envs = 1
    cfg.curriculum = {}
  return cfg
