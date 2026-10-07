"""Standalone 47-D HIM task for the XZ1 quadruped.

XZ1 owns this task's sensors, actions, rewards, events, and curriculum. It
does not call the Go2 HIM task factory, so XZ1 tuning cannot affect Go2.
"""

import math
from copy import deepcopy

import mjlab.terrains as terrain_gen
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
    BuiltinSensorCfg,
    ContactMatch,
    ContactSensorCfg,
    GridPatternCfg,
    ObjRef,
    RayCastSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig
from nazarite.config.robot_config.xz1_cfg import (
    XZ1_ACTION_SCALE,
    XZ1_BASE_BODY,
    XZ1_CALF_BODIES,
    XZ1_FOOT_GEOMS,
    XZ1_FOOT_SITES,
    XZ1_HIP_BODIES,
    XZ1_THIGH_BODIES,
    get_xz1_cfg,
)
from nazarite.mdp import observations as custom_observations
from nazarite.mdp import rewards as custom_rewards
from nazarite.mdp.him_numerics import configure_him_observations

# XZ1-local HIM knobs.  Keep these here instead of mutating the Go2 task.
XZ1_HIM_NUM_ENVS = 512
XZ1_HIM_EPISODE_LENGTH_S = 20.0
XZ1_HIM_SELF_COLLISION_FORCE_THRESHOLD = 1.0
XZ1_HIM_STUMBLE_HORIZONTAL_RATIO = 5.0
XZ1_HIM_GAIT_PERIOD = 0.6
XZ1_HIM_GAIT_OFFSETS = (0.0, 0.5, 0.5, 0.0)
XZ1_HIM_GAIT_STANCE_FRACTION = 0.56
XZ1_HIM_GAIT_COMMAND_THRESHOLD = 0.1
# MuJoCo's hfield narrow-phase caps contacts generated for one geom pair at
# 50.  XZ1's collision boxes/cylinders are larger than one 0.1 m cell, so use
# a coarser hfield grid to keep each pair below that cap.
XZ1_HIM_HFIELD_HORIZONTAL_SCALE = 0.20
XZ1_HIM_HFIELD_DOWNSAMPLED_SCALE = 0.40


XZ1_HIM_COMPLEX_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    num_rows=8,
    num_cols=8,
    border_width=25.0,
    sub_terrains={
        "stairs_up": terrain_gen.BoxPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.03, 0.12),
            step_width=0.35,
            platform_width=1.0,
            border_width=0.25,
        ),
        "stairs_down": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.03, 0.12),
            step_width=0.35,
            platform_width=1.0,
            border_width=0.25,
        ),
        "discrete_grid": terrain_gen.BoxRandomGridTerrainCfg(
            proportion=0.20,
            grid_width=0.40,
            grid_height_range=(0.02, 0.08),
            platform_width=1.0,
            fill_gaps=True,
            merge_similar_heights=True,
            border_width=0.25,
        ),
        "gravel": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15,
            noise_range=(0.005, 0.035),
            noise_step=0.005,
            horizontal_scale=XZ1_HIM_HFIELD_HORIZONTAL_SCALE,
            vertical_scale=0.005,
            downsampled_scale=XZ1_HIM_HFIELD_DOWNSAMPLED_SCALE,
            border_width=0.25,
            scale_with_difficulty=True,
        ),
        "waves": terrain_gen.HfWaveTerrainCfg(
            proportion=0.15,
            amplitude_range=(0.02, 0.10),
            num_waves=3,
            horizontal_scale=XZ1_HIM_HFIELD_HORIZONTAL_SCALE,
            vertical_scale=0.005,
            border_width=0.25,
        ),
    },
)


def Nazarite_HIM_Complex_Terrain_XZ1(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the independent XZ1 HIM complex-terrain environment."""
    terrain_scan = RayCastSensorCfg(
        name="terrain_scan",
        frame=ObjRef(type="body", name=XZ1_BASE_BODY, entity="robot"),
        ray_alignment="yaw",
        pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
        max_distance=5.0,
        exclude_parent_body=True,
    )
    feet_ground = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=XZ1_FOOT_GEOMS, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    nonfoot_ground = ContactSensorCfg(
        name="nonfoot_ground_touch",
        primary=ContactMatch(mode="body", pattern=(XZ1_BASE_BODY,), entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    leg_ground = ContactSensorCfg(
        name="leg_ground_contact",
        primary=ContactMatch(
            mode="body",
            pattern=XZ1_HIP_BODIES + XZ1_THIGH_BODIES + XZ1_CALF_BODIES,
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=1,
    )
    self_collision = ContactSensorCfg(
        name="self_collision_contact",
        primary=ContactMatch(mode="geom", pattern=r".*_primitive_1$", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern=XZ1_BASE_BODY, entity="robot"),
        fields=("found", "force"),
        reduce="maxforce",
        num_slots=1,
        history_length=1,
        secondary_policy="error",
    )
    root_angmom = BuiltinSensorCfg(
        name="root_angmom",
        sensor_type="subtreeangmom",
        obj=ObjRef(type="body", name=XZ1_BASE_BODY, entity="robot"),
    )

    actor_terms = {
        "base_ang_vel": ObservationTermCfg(
            func=envs_mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "projected_gravity": ObservationTermCfg(
            func=envs_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
        ),
        "command": ObservationTermCfg(
            func=envs_mdp.generated_commands, params={"command_name": "twist"}
        ),
        "phase": ObservationTermCfg(
            func=custom_observations.phase,
            params={"period": XZ1_HIM_GAIT_PERIOD, "command_name": "twist"},
        ),
        "joint_pos": ObservationTermCfg(
            func=envs_mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        ),
        "joint_vel": ObservationTermCfg(
            func=envs_mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5)
        ),
        "actions": ObservationTermCfg(func=envs_mdp.last_action),
    }
    critic_terms = {
        **deepcopy(actor_terms),
        "base_lin_vel": ObservationTermCfg(
            func=envs_mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        ),
        "base_com": ObservationTermCfg(
            func=custom_observations.base_com,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=(XZ1_BASE_BODY,))},
        ),
        "foot_contact": ObservationTermCfg(
            func=custom_observations.foot_contact,
            params={"sensor_name": "feet_ground_contact"},
        ),
        "height_scan": ObservationTermCfg(
            func=envs_mdp.height_scan,
            params={"sensor_name": "terrain_scan"},
            scale=1.0 / terrain_scan.max_distance,
        ),
    }
    for term_name in ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel"):
        actor_terms[term_name].delay_min_lag = 0
        actor_terms[term_name].delay_max_lag = 2
        actor_terms[term_name].delay_hold_prob = 0.3
        actor_terms[term_name].delay_update_period = 10
    for term in critic_terms.values():
        term.delay_min_lag = 0
        term.delay_max_lag = 0
        term.delay_hold_prob = 0.0
        term.delay_update_period = 0
    observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=True,
            history_length=6,
            flatten_history_dim=False,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
            history_length=1,
            flatten_history_dim=True,
        ),
    }
    configure_him_observations(
        type("_HimObservationCfg", (), {"observations": observations})()
    )

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=XZ1_ACTION_SCALE,
            use_default_offset=True,
        )
    }
    commands: dict[str, CommandTermCfg] = {
        "twist": UniformVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(8.0, 12.0),
            rel_standing_envs=0.05,
            rel_forward_envs=0.1,
            rel_heading_envs=0.0,
            heading_command=False,
            debug_vis=True,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0),
                lin_vel_y=(-1.0, 1.0),
                ang_vel_z=(-1.0, 1.0),
                heading=None,
            ),
        )
    }
    events = {
        "reset_base": EventTermCfg(
            func=envs_mdp.reset_root_state_uniform,
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
            func=envs_mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
        "push_robot": EventTermCfg(
            func=envs_mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(5.0, 6.0),
            params={
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.4, 0.4),
                    "roll": (-0.52, 0.52),
                    "pitch": (-0.52, 0.52),
                    "yaw": (-0.78, 0.78),
                }
            },
        ),
        "foot_friction": EventTermCfg(
            func=dr.geom_friction,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=XZ1_FOOT_GEOMS),
                "operation": "abs",
                "ranges": (0.3, 1.6),
                "shared_random": True,
            },
        ),
        "encoder_bias": EventTermCfg(
            func=dr.encoder_bias,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "bias_range": (-0.015, 0.015),
            },
        ),
        "base_com": EventTermCfg(
            func=dr.body_com_offset,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=(XZ1_BASE_BODY,)),
                "operation": "add",
                "ranges": {0: (-0.05, 0.05), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
            },
        ),
        "pd_gains": EventTermCfg(
            func=dr.pd_gains,
            mode="startup",
            params={
                "kp_range": (0.9, 1.1),
                "kd_range": (0.9, 1.1),
                "asset_cfg": SceneEntityCfg("robot"),
                "operation": "scale",
            },
        ),
        "effort_limits": EventTermCfg(
            func=dr.effort_limits,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "operation": "scale",
                "effort_limit_range": (0.9, 1.1),
            },
        ),
        "pseudo_inertia": EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "alpha_range": (0.5 * math.log(0.8), 0.5 * math.log(1.2)),
                "distribution": "uniform",
            },
        ),
    }
    # Reward audit: each active term owns a distinct signal.  In particular,
    # body angular velocity and angular momentum use different measurements;
    # action rate and smoothness penalize first- and second-order changes;
    # foot clearance, slip, soft landing, and stumble cover height, tangential
    # motion, vertical impact, and horizontal impact respectively.  Terrain
    # contact terms are split by body region, so no Go2 reward is reused here.
    rewards = {
        "track_velocity_x": RewardTermCfg(
            func=custom_rewards.track_velocity_x,
            weight=0.5,
            params={
                "command_name": "twist",
                "std": math.sqrt(0.25),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "track_velocity_y": RewardTermCfg(
            func=custom_rewards.track_velocity_y,
            weight=0.5,
            params={
                "command_name": "twist",
                "std": math.sqrt(0.25),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "track_yaw_velocity": RewardTermCfg(
            func=custom_rewards.him_track_angular_velocity,
            weight=1.0,
            params={
                "command_name": "twist",
                "std": math.sqrt(0.25),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "lin_vel_z": RewardTermCfg(
            func=custom_rewards.lin_vel_z_l2,
            weight=-2.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "body_orientation_l2": RewardTermCfg(
            func=custom_rewards.body_orientation_l2,
            weight=-0.2,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=(XZ1_BASE_BODY,))},
        ),
        "body_ang_vel": RewardTermCfg(
            func=velocity_mdp.body_angular_velocity_penalty,
            weight=-0.05,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=(XZ1_BASE_BODY,))},
        ),
        "angular_momentum": RewardTermCfg(
            func=velocity_mdp.angular_momentum_penalty,
            weight=-0.025,
            params={"sensor_name": "robot/root_angmom"},
        ),
        "is_terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-10.0),
        "joint_acc_l2": RewardTermCfg(func=envs_mdp.joint_acc_l2, weight=-2.5e-7),
        "joint_pos_limits": RewardTermCfg(func=envs_mdp.joint_pos_limits, weight=-1.0),
        "action_rate_l2": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.01),
        "smoothness": RewardTermCfg(func=envs_mdp.action_acc_l2, weight=-0.01),
        "joint_torques_l2": RewardTermCfg(
            func=envs_mdp.joint_torques_l2, weight=-2.0e-5
        ),
        "pose": RewardTermCfg(
            func=velocity_mdp.variable_posture,
            weight=0.5,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "command_name": "twist",
                "std_standing": {
                    r".*_hip_joint": 0.05,
                    r".*_thigh_joint": 0.10,
                    r".*_calf_joint": 0.15,
                },
                "std_walking": {
                    r".*_hip_joint": 0.15,
                    r".*_thigh_joint": 0.35,
                    r".*_calf_joint": 0.35,
                },
                "std_running": {
                    r".*_hip_joint": 0.15,
                    r".*_thigh_joint": 0.35,
                    r".*_calf_joint": 0.40,
                },
                "walking_threshold": 0.1,
                "running_threshold": 1.0,
            },
        ),
        "hip_pos": RewardTermCfg(
            func=custom_rewards.hip_joint_deviation_penalty,
            weight=-0.1,
            params={"command_name": "twist"},
        ),
        "foot_gait": RewardTermCfg(
            func=custom_rewards.feet_gait,
            weight=0.5,
            params={
                "period": XZ1_HIM_GAIT_PERIOD,
                "offset": list(XZ1_HIM_GAIT_OFFSETS),
                "threshold": XZ1_HIM_GAIT_STANCE_FRACTION,
                "command_threshold": XZ1_HIM_GAIT_COMMAND_THRESHOLD,
                "command_name": "twist",
                "sensor_name": "feet_ground_contact",
            },
        ),
        "feet_air_time": RewardTermCfg(
            func=custom_rewards.him_feet_air_time,
            weight=1.0,
            params={
                "sensor_name": "feet_ground_contact",
                "gait_name": "foot_gait",
                "command_name": "twist",
                "command_threshold": XZ1_HIM_GAIT_COMMAND_THRESHOLD,
            },
        ),
        "foot_clearance": RewardTermCfg(
            func=custom_rewards.terrain_relative_feet_clearance,
            weight=-1.0,
            params={
                "target_height": 0.08,
                "max_height": 0.25,
                "gait_name": "foot_gait",
                "scan_sensor_name": "terrain_scan",
                "command_name": "twist",
                "command_threshold": XZ1_HIM_GAIT_COMMAND_THRESHOLD,
                "lookahead_distance": 0.25,
                "path_half_width": 0.10,
                "foot_radius": 0.022,
                "asset_cfg": SceneEntityCfg("robot", site_names=XZ1_FOOT_SITES),
            },
        ),
        "foot_slip": RewardTermCfg(
            func=velocity_mdp.feet_slip,
            weight=-0.5,
            params={
                "sensor_name": "feet_ground_contact",
                "command_name": "twist",
                "command_threshold": 0.1,
                "asset_cfg": SceneEntityCfg("robot", site_names=XZ1_FOOT_SITES),
            },
        ),
        "soft_landing": RewardTermCfg(
            func=velocity_mdp.soft_landing,
            weight=-1e-3,
            params={
                "sensor_name": "feet_ground_contact",
                "command_name": "twist",
                "command_threshold": 0.1,
            },
        ),
        "stand_still": RewardTermCfg(
            func=custom_rewards.stand_still,
            weight=-1.0,
            params={
                "command_name": "twist",
                "command_threshold": 0.1,
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            },
        ),
        "nonfoot_ground_touch": RewardTermCfg(
            func=velocity_mdp.illegal_contact,
            weight=-3.0,
            params={"sensor_name": "nonfoot_ground_touch", "force_threshold": 1.0},
        ),
        "leg_collision": RewardTermCfg(
            func=velocity_mdp.illegal_contact,
            weight=-2.0,
            params={"sensor_name": "leg_ground_contact", "force_threshold": 1.0},
        ),
        "self_collision": RewardTermCfg(
            func=custom_rewards.current_self_collision_cost,
            weight=-10.0,
            params={
                "sensor_name": "self_collision_contact",
                "force_threshold": XZ1_HIM_SELF_COLLISION_FORCE_THRESHOLD,
            },
        ),
        "stumble": RewardTermCfg(
            func=custom_rewards.feet_stumble,
            weight=-0.1,
            params={
                "sensor_name": "feet_ground_contact",
                "horizontal_ratio": XZ1_HIM_STUMBLE_HORIZONTAL_RATIO,
            },
        ),
    }
    terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=envs_mdp.bad_orientation, params={"limit_angle": math.radians(70.0)}
        ),
    }
    curriculum = {
        "terrain_levels": CurriculumTermCfg(
            func=velocity_mdp.terrain_levels_vel, params={"command_name": "twist"}
        ),
        "command_vel": CurriculumTermCfg(
            func=velocity_mdp.commands_vel,
            params={
                "command_name": "twist",
                "velocity_stages": [
                    {
                        "step": 0,
                        "lin_vel_x": (-1.0, 1.0),
                        "lin_vel_y": (-1.0, 1.0),
                        "ang_vel_z": (-1.0, 1.0),
                    }
                ],
            },
        ),
    }
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(
                terrain_type="generator",
                terrain_generator=deepcopy(XZ1_HIM_COMPLEX_TERRAINS_CFG),
                max_init_terrain_level=1,
            ),
            sensors=(
                terrain_scan,
                root_angmom,
                feet_ground,
                nonfoot_ground,
                leg_ground,
                self_collision,
            ),
            entities={"robot": get_xz1_cfg()},
            num_envs=XZ1_HIM_NUM_ENVS,
            extent=2.0,
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum=curriculum,
        metrics={"mean_action_acc": MetricsTermCfg(func=envs_mdp.mean_action_acc)},
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name=XZ1_BASE_BODY,
            distance=3.0,
            elevation=-5.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            nconmax=256,
            njmax=1500,
            contact_sensor_maxmatch=500,
            mujoco=MujocoCfg(timestep=0.002, iterations=10, ls_iterations=20),
        ),
        decimation=10,
        episode_length_s=XZ1_HIM_EPISODE_LENGTH_S,
        auto_reset=play,
    )
    if play:
        terrain_cfg = cfg.scene.terrain
        assert terrain_cfg is not None
        terrain_generator = terrain_cfg.terrain_generator
        assert terrain_generator is not None
        cfg.scene.num_envs = 1
        cfg.episode_length_s = int(1e9)
        cfg.auto_reset = True
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}
        cfg.events["randomize_terrain"] = EventTermCfg(
            func=envs_mdp.randomize_terrain, mode="reset", params={}
        )
        terrain_generator.curriculum = False
        terrain_generator.num_cols = 5
        terrain_generator.num_rows = 5
        terrain_generator.border_width = 10.0
    return cfg


__all__ = ["XZ1_HIM_COMPLEX_TERRAINS_CFG", "Nazarite_HIM_Complex_Terrain_XZ1"]
