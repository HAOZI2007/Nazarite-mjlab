"""Go2 fall-recovery environments used for FR-Net training and evaluation."""

from __future__ import annotations

import math
from dataclasses import replace

from mjlab.envs import mdp as env_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
  ContactSensorCfg,
  ObjRef,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.mdp import terminations as velocity_terminations
from mjlab.terrains import TerrainEntityCfg, TerrainGeneratorCfg
from mjlab.terrains.config import (
  box_random_grid,
  flat,
  pyramid_stairs,
  random_spread_boxes,
)
from nazarite.config.robot_config.go2_cfg import GO2_BASE_BODY
from nazarite.config.train_config.env_cfgs.go2_env_cfgs import (
  Nazarite_Velocity_Flat_Go2,
)
from nazarite.mdp import rewards as common_rewards
from nazarite.mdp.fr_net import (
  frnet_curriculums,
  frnet_events,
  frnet_observations,
  frnet_rewards,
  frnet_terminations,
)

# Fall recovery intentionally begins with the trunk and legs on the ground.
# Heightfields are a poor fit for this reset distribution: a fallen Go2 can
# overlap more than MuJoCo's per-heightfield collision-candidate limit in one
# step. Use primitive boxes only, with a flat central platform at every spawn
# origin, so recovery remains challenging without heightfield overflows.
_FRNET_RECOVERY_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=12.0,
  num_rows=8,
  num_cols=4,
  curriculum=False,
  sub_terrains={
    "flat": flat(proportion=0.40),
    "gentle_pyramid_stairs": pyramid_stairs(
      proportion=0.25,
      step_height_range=(0.02, 0.06),
      step_width=0.40,
      platform_width=3.0,
      border_width=1.0,
    ),
    "low_random_grid": box_random_grid(
      proportion=0.20,
      grid_width=0.80,
      grid_height_range=(0.02, 0.08),
      platform_width=2.0,
      border_width=0.40,
      merge_similar_heights=True,
      height_merge_threshold=0.04,
      max_merge_distance=3,
    ),
    "low_scattered_boxes": random_spread_boxes(
      proportion=0.15,
      num_boxes=20,
      box_width_range=(0.25, 0.60),
      box_length_range=(0.25, 0.80),
      box_height_range=(0.03, 0.12),
      platform_width=2.0,
      border_width=0.40,
    ),
  },
  add_lights=True,
)


def _configure_recovery_contact_history(cfg) -> None:
  """Cover a full policy period when producing body-contact supervision."""
  for sensor_cfg in cfg.scene.sensors or ():
    if isinstance(sensor_cfg, ContactSensorCfg) and sensor_cfg.name in {
      "hip_ground_touch",
      "thigh_ground_touch",
      "shank_ground_touch",
      "trunk_ground_touch",
    }:
      sensor_cfg.history_length = cfg.decimation


def _add_base_height_scan(cfg) -> None:
  """Add a local terrain probe for the terrain-relative base-height reward."""
  assert cfg.scene.sensors is not None
  cfg.scene.sensors = (
    *cfg.scene.sensors,
    TerrainHeightSensorCfg(
      name="base_height_scan",
      frame=ObjRef(type="body", name=GO2_BASE_BODY, entity="robot"),
      ray_alignment="yaw",
      max_distance=1.5,
      exclude_parent_body=True,
      include_geom_groups=(0,),
      pattern=RingPatternCfg.single_ring(radius=0.03, num_samples=4),
      reduction="min",
    ),
  )


def Nazarite_FRNet_Recovery_Go2(
  play: bool = False,
  challenging_terrain: bool = False,
):
  """Create an FR-Net Go2 recovery task.

  The task starts from randomized fallen poses. It deliberately removes the
  velocity command and gait rewards from the existing Go2 task: recovery is
  optimized directly, while mass/contact labels remain privileged supervision.
  """
  cfg = Nazarite_Velocity_Flat_Go2(play=False, enable_wtw=False)
  cfg.scene.num_envs = 1024
  cfg.scene.env_spacing = 1.0
  cfg.episode_length_s = 8.0

  # Recovery has no velocity command. Remove the source and every velocity-only
  # observation/reward term rather than leaving a zero command as a shortcut.
  cfg.commands = {}
  actor_terms, critic_terms, auxiliary_terms = (
    frnet_observations.make_frnet_observation_terms()
  )
  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=None,
      flatten_history_dim=True,
      nan_policy="sanitize",
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=None,
      flatten_history_dim=True,
      nan_policy="sanitize",
    ),
    # RolloutStorage automatically keeps this group. FRNetPPO reads it only
    # during update; actor/critic obs_groups intentionally exclude it.
    "frnet_aux": ObservationGroupCfg(
      terms=auxiliary_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=None,
      flatten_history_dim=True,
      nan_policy="error",
    ),
  }
  _configure_recovery_contact_history(cfg)
  _add_base_height_scan(cfg)

  cfg.events.pop("push_robot", None)
  cfg.events["reset_base"] = EventTermCfg(
    mode="reset",
    func=frnet_events.reset_fallen_root_state,
    params={"asset_cfg": SceneEntityCfg("robot")},
  )
  cfg.events["reset_robot_joints"].params.update(
    {
      "position_range": (-0.15, 0.15),
      "velocity_range": (-0.5, 0.5),
    }
  )

  # pseudo_inertia scales both mass and inertia. For alpha, mass_ratio=e^(2a),
  # so this corresponds to an independent per-link range of [0.8, 1.2].
  # The Go2 *_foot bodies are massless frames for the foot geom/site, not
  # inertial links. Excluding them is required because pseudo-inertia needs a
  # strictly positive-definite inertia matrix.
  alpha_range = (0.5 * math.log(0.8), 0.5 * math.log(1.2))
  cfg.events["frnet_leg_inertia"] = EventTermCfg(
    mode="reset",
    func=dr.pseudo_inertia,
    params={
      "alpha_range": alpha_range,
      "asset_cfg": SceneEntityCfg(
        "robot",
        body_names=(
          "FL_hip",
          "FL_thigh",
          "FL_calf",
          "FR_hip",
          "FR_thigh",
          "FR_calf",
          "RL_hip",
          "RL_thigh",
          "RL_calf",
          "RR_hip",
          "RR_thigh",
          "RR_calf",
        ),
        preserve_order=True,
      ),
    },
  )

  all_joint_cfg = SceneEntityCfg("robot", joint_names=(".*",))
  all_actuator_cfg = SceneEntityCfg("robot", actuator_names=".*")
  cfg.rewards = {
    "upright": RewardTermCfg(
      func=frnet_rewards.upright_gaussian,
      weight=6.0,
      params={"asset_cfg": SceneEntityCfg("robot")},
    ),
    "orientation_cost": RewardTermCfg(
      func=common_rewards.flat_orientation_l2,
      weight=-0.5,
      params={"asset_cfg": SceneEntityCfg("robot")},
    ),
    "base_height": RewardTermCfg(
      func=frnet_rewards.base_height_relative_gaussian,
      weight=1.0,
      params={
        "target_height": 0.32,
        "height_sensor_name": "base_height_scan",
      },
    ),
    "foot_contacts": RewardTermCfg(
      func=frnet_rewards.foot_contact_count,
      weight=0.1,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "stand_pose": RewardTermCfg(
      func=frnet_rewards.recovered_stand_pose,
      weight=4.0,
      params={"asset_cfg": all_joint_cfg},
    ),
    "body_ang_vel": RewardTermCfg(
      func=common_rewards.body_angular_velocity_penalty,
      weight=-0.05,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("base_link",))},
    ),
    "joint_torques": RewardTermCfg(
      func=common_rewards.joint_torques_l2,
      weight=-2.0e-4,
      params={"asset_cfg": all_actuator_cfg},
    ),
    "joint_acc": RewardTermCfg(
      func=common_rewards.joint_acc_l2,
      weight=-1.0e-6,
      params={"asset_cfg": all_joint_cfg},
    ),
    "joint_limits": RewardTermCfg(
      func=common_rewards.joint_pos_limits,
      weight=-5.0,
      params={"asset_cfg": all_joint_cfg},
    ),
    "action_rate": RewardTermCfg(
      func=common_rewards.action_rate_l2,
      weight=-0.02,
    ),
  }
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=velocity_mdp.time_out, time_out=True),
    "invalid_state": TerminationTermCfg(func=env_mdp.nan_detection),
    "penetrated_ground": TerminationTermCfg(
      func=frnet_terminations.base_below_safe_height,
      params={
        "minimum_height": 0.02,
        "height_sensor_name": "base_height_scan",
      },
    ),
  }
  if not play:
    # A successful recovery is a short, stable four-foot stand rather than
    # merely rotating the trunk upright. The termination is evaluated before
    # rewards, allowing recovery_complete to emit a one-time terminal bonus.
    cfg.terminations["recovery_success"] = TerminationTermCfg(
      func=frnet_terminations.StableRecoveryTermination,
      params={
        "hold_duration_s": 0.3,
        "upright_gravity_z_max": -0.85,
        "min_relative_height": 0.27,
        "min_foot_contacts": 3,
        "contact_force_threshold": 5.0,
        "max_angular_velocity": 3.0,
        "max_joint_position_error": 0.35,
        "foot_sensor_name": "feet_ground_contact",
        "height_sensor_name": "base_height_scan",
        "asset_cfg": all_joint_cfg,
      },
    )
    cfg.rewards["recovery_complete"] = RewardTermCfg(
      func=frnet_rewards.stable_recovery_completion_bonus,
      # The function compensates for RewardManager's dt scaling, so this is
      # an actual one-off return bonus of 20 rather than 20 * step_dt.
      weight=20.0,
      params={"termination_name": "recovery_success"},
    )

  if challenging_terrain:
    # This task has many simultaneous contacts during the first recovery
    # frames. Keep its batch and solver buffers practical for an 8 GiB GPU.
    cfg.scene.num_envs = 256
    terrain_cfg = TerrainEntityCfg(
      terrain_type="generator",
      terrain_generator=_FRNET_RECOVERY_TERRAINS_CFG,
      # Seed both level 0 and level 1.  This exposes the policy to gentle
      # terrain variation from the first rollout, while higher rows remain
      # success-gated by the reset curriculum.
      max_init_terrain_level=1,
    )
    cfg.scene.terrain = terrain_cfg
    cfg.sim.nconmax = 128
    cfg.sim.njmax = 2_000
    cfg.sim.contact_sensor_maxmatch = 128
    cfg.terminations["out_of_terrain_bounds"] = TerminationTermCfg(
      func=velocity_terminations.out_of_terrain_bounds,
    )
    cfg.events["reset_base"].params.update(
      {
        # The origin is a flat center platform.  Offsetting within 1.8 m keeps
        # a 2.2 m margin to the edge of each 8 m tile, but places many falls on
        # the surrounding low stairs, grid, and box surfaces.
        "xy_offset_range": (-1.8, 1.8),
        # Local terrain can sit above the center-platform origin.  Begin high
        # enough that a randomly oriented trunk or leg cannot start embedded in
        # a low obstacle, then let gravity create the actual fallen contact.
        "height_offset_range": (0.30, 0.42),
        "minimum_root_height_above_origin": 0.65,
      }
    )
    if not play:
      cfg.rewards["stability"] = RewardTermCfg(
        func=frnet_rewards.upright_stability_support,
        weight=2.0,
        params={
          "upright_threshold": -0.8,
          "contact_force_threshold": 5.0,
          "angular_velocity_std": 1.5,
          "foot_sensor_name": "feet_ground_contact",
          "asset_cfg": SceneEntityCfg("robot"),
        },
      )
      # Preserve every individual cost, then make the *combined* Terrain
      # recovery reward non-negative.  The correction function must stay last:
      # it reads the weighted contributions produced by all terms above.
      cfg.rewards["nonnegative_total_correction"] = RewardTermCfg(
        func=frnet_rewards.nonnegative_total_reward_correction,
        weight=1.0,
        params={"source_term_names": tuple(cfg.rewards)},
      )
      terrain_generator = terrain_cfg.terrain_generator
      if terrain_generator is None:
        raise RuntimeError("Terrain recovery training requires a terrain generator")
      terrain_generator.curriculum = True
      cfg.curriculum = {
        "frnet_terrain_levels": CurriculumTermCfg(
          func=frnet_curriculums.terrain_levels_from_recovery_success,
          params={
            "success_termination_name": "recovery_success",
            "low_level_exploration_probability": 0.25,
          },
        ),
      }

  if play:
    if challenging_terrain:
      # Use one world per terrain type.  The native viewer renders every batch
      # world, so a larger batch would put several robots at the same level-0
      # spawn origin and make them appear as a single many-legged robot.
      # Environment indices then map directly to flat, stairs, grid, boxes.
      terrain_cfg = cfg.scene.terrain
      if terrain_cfg is None:
        raise RuntimeError("Terrain recovery play requires a terrain entity")
      terrain_generator = terrain_cfg.terrain_generator
      if terrain_generator is None:
        raise RuntimeError("Terrain recovery play requires a terrain generator")
      terrain_cfg.terrain_generator = replace(
        terrain_generator,
        curriculum=True,
      )
    cfg.scene.num_envs = 4 if challenging_terrain else 1
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    # A fallen orientation is expected during recovery evaluation.
    cfg.terminations.pop("penetrated_ground", None)
    cfg.curriculum = {}

  return cfg


def Nazarite_FRNet_Recovery_Terrain_Go2(play: bool = False):
  """Create the challenging-terrain variant used for FR-Net evaluation."""
  return Nazarite_FRNet_Recovery_Go2(play=play, challenging_terrain=True)
