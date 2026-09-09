"""Go2 closed-loop reference-tracking teacher configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import (
  base_ang_vel,
  base_lin_vel,
  joint_pos_rel,
  joint_vel_rel,
  last_action,
  projected_gravity,
)
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig
from nazarite.config.robot_config.go2_cfg import (
  GO2_ACTION_SCALE,
  GO2_BASE_BODY,
  GO2_FOOT_GEOMS,
  get_go2_cfg,
)
from nazarite.config.train_config.train_algorithm.smp.teacher import (
  observations as teacher_obs,
)
from nazarite.config.train_config.train_algorithm.smp.teacher import (
  reference as teacher_reference,
)
from nazarite.config.train_config.train_algorithm.smp.teacher import (
  rewards as teacher_rewards,
)
from nazarite.config.train_config.train_algorithm.smp.teacher import (
  terminations as teacher_terminations,
)
from nazarite.mdp import rewards as custom_rewards

# File location: <Nazarite>/Nazarite-src/nazarite/config/train_config/smp_config/...
_PROJECT_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_REFERENCE_FILE = (
  _PROJECT_ROOT
  / "output/smp_reference_dataset_anchor035_actual_controller"
  / "accepted_manifest_preprocessed.json"
)


def make_smp_teacher_go2_env_cfg(
  reference_file: str | None = None,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the closed-loop Go2 teacher task.

  The teacher sees the current robot state and the current reference state. Its
  action is still the normal 12-D Nazarite action and is converted by
  ``GO2_ACTION_SCALE`` into position targets by mjlab.
  """
  reference_path = str(Path(reference_file).expanduser()) if reference_file else str(DEFAULT_REFERENCE_FILE)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=GO2_FOOT_GEOMS, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  actor_terms = {
    "base_lin_vel": ObservationTermCfg(func=base_lin_vel),
    "base_ang_vel": ObservationTermCfg(func=base_ang_vel),
    "projected_gravity": ObservationTermCfg(func=projected_gravity),
    "joint_pos": ObservationTermCfg(
      func=joint_pos_rel,
      params={"biased": True, "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "joint_vel": ObservationTermCfg(
      func=joint_vel_rel,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "actions": ObservationTermCfg(func=last_action),
    "reference_joint_pos": ObservationTermCfg(func=teacher_obs.reference_joint_pos_rel),
    "reference_joint_vel": ObservationTermCfg(func=teacher_obs.reference_joint_vel),
    "reference_root_pos": ObservationTermCfg(func=teacher_obs.reference_root_pos_b),
    "reference_root_lin_vel": ObservationTermCfg(func=teacher_obs.reference_root_lin_vel_b),
    "reference_foot_target": ObservationTermCfg(func=teacher_obs.reference_foot_target_base),
    "robot_foot_pos": ObservationTermCfg(func=teacher_obs.robot_foot_pos_base),
    "reference_contact": ObservationTermCfg(func=teacher_obs.reference_contact),
    "reference_phase": ObservationTermCfg(func=teacher_obs.reference_phase),
  }
  critic_terms = deepcopy(actor_terms)

  rewards = {
    "reference_joint_position": RewardTermCfg(
      func=teacher_rewards.reference_joint_position_tracking,
      weight=4.0,
      params={"std": 0.25},
    ),
    "reference_joint_velocity": RewardTermCfg(
      func=teacher_rewards.reference_joint_velocity_tracking,
      weight=1.0,
      params={"std": 2.0},
    ),
    "reference_foot_position": RewardTermCfg(
      func=teacher_rewards.reference_foot_position_tracking,
      weight=3.0,
      params={"std": 0.08},
    ),
    "reference_root_velocity": RewardTermCfg(
      func=teacher_rewards.reference_root_velocity_tracking,
      weight=2.0,
      params={"std": 0.5},
    ),
    "reference_contact": RewardTermCfg(
      func=teacher_rewards.reference_contact_tracking,
      weight=0.5,
      params={"sensor_name": "feet_ground_contact", "force_threshold": 5.0},
    ),
    "upright": RewardTermCfg(
      func=teacher_rewards.upright_tracking,
      weight=1.0,
      params={"std": 0.25},
    ),
    "joint_torque_l2": RewardTermCfg(
      func=custom_rewards.joint_torques_l2,
      weight=-1.0e-4,
      params={"asset_cfg": SceneEntityCfg("robot", actuator_names=".*")},
    ),
    "joint_acceleration_l2": RewardTermCfg(
      func=custom_rewards.joint_acc_l2,
      weight=-2.5e-7,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "action_rate_l2": RewardTermCfg(
      func=custom_rewards.action_rate_l2,
      weight=-0.005,
    ),
  }

  terminations = {
    "reference_finished": TerminationTermCfg(
      func=teacher_terminations.reference_finished,
      params={"command_name": "reference"},
      time_out=True,
    ),
    "fell_over": TerminationTermCfg(
      func=teacher_terminations.bad_orientation,
      params={"limit_angle": 1.2},
    ),
    "low_base": TerminationTermCfg(
      func=teacher_terminations.root_height_below_minimum,
      params={"minimum_height": 0.16},
    ),
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      sensors=(feet_ground_cfg,),
      entities={"robot": get_go2_cfg()},
      num_envs=1024,
      env_spacing=1.0,
      extent=2.0,
    ),
    observations={
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
    },
    actions={
      "joint_pos": JointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale=GO2_ACTION_SCALE,
        use_default_offset=True,
      )
    },
    commands={
      "reference": teacher_reference.ReferenceMotionCommandCfg(
        entity_name="robot",
        motion_file=reference_path,
        randomize_clip=not play,
        randomize_start_phase=not play,
        resampling_time_range=(1.0e9, 1.0e9),
      )
    },
    events={},
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name=GO2_BASE_BODY,
      distance=3.0,
      elevation=-10.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(
      # Reference resets can briefly create multiple simultaneous contacts;
      # leave headroom for the contact sensor and avoid solver overflow.
      nconmax=128,
      njmax=1500,
      mujoco=MujocoCfg(timestep=0.002, iterations=10, ls_iterations=30),
    ),
    decimation=10,
    episode_length_s=30.0,
  )

  if play:
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 1.0e9
    cfg.observations["actor"].enable_corruption = False

  return cfg


def Nazarite_SMP_Teacher_Go2(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Registered task factory for the first closed-loop SMP teacher."""
  return make_smp_teacher_go2_env_cfg(play=play)
