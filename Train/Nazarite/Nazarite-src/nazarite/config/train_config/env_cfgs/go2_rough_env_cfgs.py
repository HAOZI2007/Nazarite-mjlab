"""Go2 velocity task on procedural rough and obstacle terrain."""

from copy import deepcopy
import math

from mjlab.envs.mdp import dr
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.terrains import TerrainEntityCfg, TerrainGeneratorCfg
from mjlab.terrains.config import (
  discrete_obstacles,
  flat,
  hf_pyramid_slope,
  pyramid_stairs,
  pyramid_stairs_inv,
  random_rough,
  wave_terrain,
)
from mjlab.tasks.velocity import mdp as velocity_mdp

from nazarite.config.train_config.env_cfgs.go2_env_cfgs import Nazarite_Velocity_Flat_Go2
from nazarite.config.robot_config.go2_cfg import (
  GO2_BASE_BODY,
  GO2_CALF_BODIES,
  GO2_FOOT_BODIES,
  GO2_FOOT_SITES,
  GO2_THIGH_BODIES,
)
from nazarite.mdp import curriculums as custom_curriculums
from nazarite.mdp import rewards as custom_rewards


CTS_MOE_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0), border_width=20.0, num_rows=10, num_cols=20, curriculum=True,
  sub_terrains={
    "wave": wave_terrain(proportion=0.05),
    "slope": hf_pyramid_slope(proportion=0.20, slope_range=(0.0, 0.7)),
    "rough": random_rough(proportion=0.05),
    "stairs_up": pyramid_stairs(proportion=0.25, step_height_range=(0.02, 0.20)),
    "stairs_down": pyramid_stairs_inv(proportion=0.10, step_height_range=(0.02, 0.20)),
    "obstacles": discrete_obstacles(proportion=0.20, obstacle_height_range=(0.05, 0.25)),
    "flat": flat(proportion=0.15),
  },
  add_lights=True,
)


def Nazarite_Velocity_Rough_Go2(play: bool = False):
  """Create the Nazarite rough-terrain Go2 velocity task."""
  cfg = Nazarite_Velocity_Flat_Go2(play=play, enable_wtw=False)
  cfg.scene.num_envs = 1024
  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator", terrain_generator=deepcopy(CTS_MOE_TERRAINS_CFG), max_init_terrain_level=0,
  )
  cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
    func=custom_curriculums.terrain_levels_vel_strict, params={"command_name": "twist"},
  )

  # HIMLoco-style dynamics randomization.  The existing limb-mass event is
  # widened to the reference link_mass_range; pseudo-inertia additionally
  # keeps mass, COM and inertia physically consistent for every environment.
  # pseudo_inertia is the physically consistent replacement for independent
  # body_mass edits.  Keeping both would apply two unrelated mass samples and
  # can make the first MuJoCo step numerically unstable.
  cfg.events.pop("base_mass", None)
  cfg.events.pop("link_mass", None)
  cfg.events["effort_limits"] = EventTermCfg(
    func=dr.effort_limits,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot", actuator_names=".*"),
      "operation": "scale",
      "effort_limit_range": (0.9, 1.1),
    },
  )
  cfg.events["pseudo_inertia"] = EventTermCfg(
    func=dr.pseudo_inertia,
    mode="startup",
    params={
      # The Go2 MJCF contains massless hip connector bodies.  Exclude them:
      # pseudo-inertia requires strictly positive body mass and inertia.
      "asset_cfg": SceneEntityCfg(
        "robot",
        body_names=(GO2_BASE_BODY,) + GO2_THIGH_BODIES + GO2_CALF_BODIES + GO2_FOOT_BODIES,
      ),
      "alpha_range": (0.5 * math.log(0.8), 0.5 * math.log(1.2)),
      "distribution": "uniform",
    },
  )

  student_terms = deepcopy(cfg.observations["actor"].terms)
  for term in student_terms.values():
    term.history_length = 5
  cfg.observations["student_history"] = ObservationGroupCfg(
    terms=student_terms, concatenate_terms=True, enable_corruption=True,
    history_length=None, flatten_history_dim=True,
  )
  cfg.observations["teacher"] = deepcopy(cfg.observations["critic"])
  cfg.observations["teacher"].enable_corruption = False

  cfg.rewards.pop("track_linear_velocity", None)
  cfg.rewards.pop("track_angular_velocity", None)
  cfg.rewards["track_velocity_x"].weight = 1.0
  cfg.rewards["track_velocity_y"].weight = 0.5
  cfg.rewards["track_yaw_velocity"].weight = 0.5
  # Replace the three generic foot terms with the phase-conditioned
  # HIMLoco formulation.  This separates swing clearance, stance contact,
  # slip, and touchdown impact instead of making them compete globally.
  cfg.rewards.pop("air_time", None)
  cfg.rewards.pop("prolonged_air_time", None)
  cfg.rewards.pop("stance_contact", None)
  cfg.rewards["foot_gait"] = RewardTermCfg(
    func=custom_rewards.feet_gait,
    weight=0.01,
    params={
      "period": 0.6,
      "offset": [0.0, 0.5, 0.5, 0.0],
      "threshold": 0.56,
      "command_threshold": 0.1,
      "command_name": "twist",
      "sensor_name": "feet_ground_contact",
    },
  )
  cfg.rewards["foot_clearance"] = RewardTermCfg(
    func=velocity_mdp.feet_clearance,
    weight=-0.01,
    params={
      "target_height": 0.08,
      "height_sensor_name": "foot_height_scan",
      "command_name": "twist",
      "command_threshold": 0.1,
      "asset_cfg": SceneEntityCfg("robot", site_names=GO2_FOOT_SITES),
    },
  )
  cfg.rewards["foot_slip"].weight = -0.05
  # Keep touchdown smoothing as a secondary term; the previous magnitude
  # dominated the foot-related rewards without improving obstacle traversal.
  cfg.rewards["soft_landing"].weight = -2.5e-4
  cfg.terminations["terrain_edge_reached"] = TerminationTermCfg(
    func=velocity_mdp.terrain_edge_reached, time_out=True,
  )
  cfg.sim.nconmax = 160
  cfg.sim.njmax = 2_000
  cfg.sim.contact_sensor_maxmatch = 160
  if play:
    cfg.scene.num_envs = 1
    cfg.terminations.pop("terrain_edge_reached", None)
  return cfg
