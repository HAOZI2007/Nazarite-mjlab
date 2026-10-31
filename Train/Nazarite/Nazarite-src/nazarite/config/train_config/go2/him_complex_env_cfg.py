"""Standalone Go2 HIM complex-terrain task."""

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
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig
from nazarite.config.robot_config.go2_cfg import (
    GO2_ACTION_SCALE,
    GO2_BASE_BODY,
    GO2_FOOT_GEOMS,
    GO2_FOOT_SITES,
    get_go2_cfg,
)
from nazarite.config.train_config.xz1.him_terrain import (
    BoxHighWallTerrainCfg,
    BoxTireArrayTerrainCfg,
)
from nazarite.mdp.him.numerics import configure_him_observations
from nazarite.mdp.him.terminal_observations import configure_estimator_observations
from nazarite.mdp.shared import curriculums as custom_curriculums
from nazarite.mdp.shared import events as custom_events
from nazarite.mdp.shared import observations as custom_observations
from nazarite.mdp.shared import rewards as custom_rewards
from nazarite.mdp.shared.commands import TerrainConditionedVelocityCommandCfg

HIM_LINK_MASS_RANGE = (0.8, 1.2)
HIM_EFFORT_LIMIT_RANGE = (0.9, 1.1)

# Keep Go2-HIM's terrain geometry and terrain curriculum inputs identical to
# XZ1-HIM. The stair border leaves five stair rings around the center platform.
HIM_STAIR_LEVELS = 6
HIM_STAIR_WIDTH = 0.35
HIM_STAIR_PLATFORM_WIDTH = 2.5
HIM_STAIR_BORDER_WIDTH = (
    8.0 - HIM_STAIR_PLATFORM_WIDTH - 2 * (HIM_STAIR_LEVELS - 1) * HIM_STAIR_WIDTH
) / 2
HIM_COMPLEX_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    num_rows=8,
    num_cols=3,
    border_width=25.0,
    sub_terrains={
        "stairs_up": terrain_gen.BoxPyramidStairsTerrainCfg(
            proportion=0.35,
            step_height_range=(0.03, 0.12),
            step_width=HIM_STAIR_WIDTH,
            platform_width=HIM_STAIR_PLATFORM_WIDTH - 1e-6,
            border_width=HIM_STAIR_BORDER_WIDTH,
        ),
        "stairs_down": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
            proportion=0.35,
            step_height_range=(0.03, 0.12),
            step_width=HIM_STAIR_WIDTH,
            platform_width=HIM_STAIR_PLATFORM_WIDTH - 1e-6,
            border_width=HIM_STAIR_BORDER_WIDTH,
        ),
        "tire_array": BoxTireArrayTerrainCfg(
            proportion=0.15,
            tire_outer_diameter=0.55,
            tire_inner_diameter=0.33,
            tire_width=0.165,
            tire_pitch=0.56,
            tire_height=0.165,
            startup_zone_length=4.0,
            spawn_x_fraction=0.25,
        ),
        "high_wall": BoxHighWallTerrainCfg(
            proportion=0.15,
            wall_height_range=(0.05, 0.30),
            wall_thickness=0.08,
        ),
    },
)


def Nazarite_HIM_Complex_Terrain_Go2(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the standalone reference-compatible Go2 HIM environment."""
    terrain_scan = RayCastSensorCfg(
        name="terrain_scan",
        frame=ObjRef(type="body", name=GO2_BASE_BODY, entity="robot"),
        ray_alignment="yaw",
        pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
        max_distance=5.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
    )
    feet_height = TerrainHeightSensorCfg(
        name="feet_terrain_height",
        frame=tuple(
            ObjRef(type="site", name=name, entity="robot") for name in GO2_FOOT_SITES
        ),
        pattern=RingPatternCfg.single_ring(
            radius=0.02, num_samples=4, include_center=True
        ),
        ray_alignment="world",
        max_distance=1.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
        reduction="min",
    )
    feet_ground = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=GO2_FOOT_GEOMS, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    # Match only explicitly named hip/thigh/calf collision geoms. Go2's foot
    # collision geoms live on the calf bodies, so body-level matching would
    # incorrectly count normal foot support as a leg collision.
    nonfoot_ground = ContactSensorCfg(
        name="nonfoot_ground_touch",
        primary=ContactMatch(mode="body", pattern=(GO2_BASE_BODY,), entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    leg_ground = ContactSensorCfg(
        name="leg_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(FL|FR|RL|RR)_(hip|thigh|calf)_collision_[0-9]+$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=1,
    )
    root_angmom = BuiltinSensorCfg(
        name="root_angmom",
        sensor_type="subtreeangmom",
        obj=ObjRef(type="body", name=GO2_BASE_BODY, entity="robot"),
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
            params={"period": 0.6, "command_name": "twist"},
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
            params={"asset_cfg": SceneEntityCfg("robot", body_names=(GO2_BASE_BODY,))},
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

    # The reference Go2-HIM task randomizes actuator latency, but does not add
    # observation latency. Keep actor, critic, and estimator observations current.
    for term in (*actor_terms.values(), *critic_terms.values()):
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
    # Keep raw-value diagnostics and the same clipping contract used by exports.
    numerical_cfg = type("_HimObservationCfg", (), {"observations": observations})()
    configure_him_observations(numerical_cfg)
    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=GO2_ACTION_SCALE,
            use_default_offset=True,
        ),
    }
    commands: dict[str, CommandTermCfg] = {
        "twist": TerrainConditionedVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(8.0, 12.0),
            rel_standing_envs=0.05,
            rel_forward_envs=0.1,
            rel_heading_envs=0.0,
            heading_command=False,
            debug_vis=True,
            forward_only_terrain_names=("tire_array", "high_wall"),
            forward_only_x_min=0.3,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0),
                lin_vel_y=(-1.0, 1.0),
                ang_vel_z=(-1.0, 1.0),
                heading=None,
            ),
        ),
    }
    events = {
        "reset_base": EventTermCfg(
            func=custom_events.reset_root_state_uniform_facing_high_wall,
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
                "asset_cfg": SceneEntityCfg("robot", geom_names=GO2_FOOT_GEOMS),
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
                "asset_cfg": SceneEntityCfg("robot", body_names=(GO2_BASE_BODY,)),
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
                "effort_limit_range": HIM_EFFORT_LIMIT_RANGE,
            },
        ),
        "pseudo_inertia": EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "alpha_range": (
                    0.5 * math.log(HIM_LINK_MASS_RANGE[0]),
                    0.5 * math.log(HIM_LINK_MASS_RANGE[1]),
                ),
                "distribution": "uniform",
            },
        ),
    }
    rewards = {
        "track_linear_velocity": RewardTermCfg(
            func=custom_rewards.him_track_linear_velocity,
            weight=1.0,
            params={
                "command_name": "twist",
                "std": math.sqrt(0.25),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "track_angular_velocity": RewardTermCfg(
            func=custom_rewards.him_track_angular_velocity,
            weight=1.0,
            params={
                "command_name": "twist",
                "std": math.sqrt(0.5),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "body_orientation_l2": RewardTermCfg(
            func=custom_rewards.body_orientation_l2,
            weight=-0.2,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=(GO2_BASE_BODY,))},
        ),
        "pose": RewardTermCfg(
            func=velocity_mdp.variable_posture,
            weight=0.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "command_name": "twist",
                "std_standing": {},
                "std_walking": {},
                "std_running": {},
                "walking_threshold": 0.1,
                "running_threshold": 1.5,
            },
        ),
        "body_ang_vel": RewardTermCfg(
            func=velocity_mdp.body_angular_velocity_penalty,
            weight=-0.05,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=(GO2_BASE_BODY,))},
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
        "hip_pos": RewardTermCfg(
            func=custom_rewards.hip_joint_deviation_penalty,
            weight=-0.1,
            params={"command_name": "twist"},
        ),
        "foot_gait": RewardTermCfg(
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
        ),
        "foot_clearance": RewardTermCfg(
            func=velocity_mdp.feet_clearance,
            weight=-0.01,
            params={
                "target_height": 0.10,
                "height_sensor_name": "feet_terrain_height",
                "command_name": "twist",
                "command_threshold": 0.1,
                "asset_cfg": SceneEntityCfg("robot", site_names=GO2_FOOT_SITES),
            },
        ),
        "foot_slip": RewardTermCfg(
            func=velocity_mdp.feet_slip,
            weight=-0.05,
            params={
                "sensor_name": "feet_ground_contact",
                "command_name": "twist",
                "command_threshold": 0.1,
                "asset_cfg": SceneEntityCfg("robot", site_names=GO2_FOOT_SITES),
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
            weight=-0.5,
            params={
                "command_name": "twist",
                "command_threshold": 0.1,
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            },
        ),
        "leg_collision": RewardTermCfg(
            func=velocity_mdp.illegal_contact,
            weight=-1.0,
            params={"sensor_name": "leg_ground_contact", "force_threshold": 1.0},
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
            func=custom_curriculums.terrain_levels_delta,
            params={
                "command_name": "twist",
                "success_distance_scale": 0.8,
                "min_success_distance": 0.8,
                "max_success_distance": 3.2,
                "success_streak_length": 2,
                "failure_streak_length": 3,
                "sparse_terrain_names": ("stairs_down", "tire_array", "high_wall"),
                "sparse_success_distance_scale": 0.5,
                "sparse_min_success_distance": 0.8,
                "sparse_max_success_distance": 2.0,
                "sparse_success_streak_length": 3,
                "sparse_failure_streak_length": 4,
                "high_wall_terrain_name": "high_wall",
                "high_wall_crossing_clearance": 0.4,
                "tire_array_terrain_name": "tire_array",
                "tire_array_crossing_clearance": 0.05,
                "adaptive_type_sampling": True,
                "adaptive_sampling_floor": 0.25,
                "adaptive_sampling_temperature": 1.0,
                "adaptive_warmup_episodes": 20,
                "type_success_ema_alpha": 0.05,
            },
        ),
        "command_vel": CurriculumTermCfg(
            func=velocity_mdp.commands_vel,
            params={
                "command_name": "twist",
                "velocity_stages": [
                    {
                        "step": 0,
                        "lin_vel_x": (-1.0, 1.0),
                        "lin_vel_y": (-0.5, 0.5),
                        "ang_vel_z": (-0.5, 0.5),
                    }
                ],
            },
        ),
    }
    # Match the reference Go2 actuator communication/action delay. The delay is
    # expressed in 2 ms physics steps and resampled once per 10-step control cycle.
    robot_cfg = deepcopy(get_go2_cfg())
    assert robot_cfg.articulation is not None
    for actuator in robot_cfg.articulation.actuators:
        actuator.delay_min_lag = 0
        actuator.delay_max_lag = 4
        actuator.delay_hold_prob = 0.5
        actuator.delay_update_period = 10
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(
                terrain_type="generator",
                terrain_generator=deepcopy(HIM_COMPLEX_TERRAINS_CFG),
                max_init_terrain_level=1,
            ),
            sensors=(
                terrain_scan,
                root_angmom,
                feet_ground,
                feet_height,
                nonfoot_ground,
                leg_ground,
            ),
            entities={"robot": robot_cfg},
            num_envs=1024,
            extent=1.0,
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
            body_name=GO2_BASE_BODY,
            distance=3.0,
            elevation=-5.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            nconmax=128,
            njmax=500,
            contact_sensor_maxmatch=300,
            mujoco=MujocoCfg(
                timestep=0.002, solver="cg", iterations=10, ls_iterations=20
            ),
        ),
        decimation=10,
        episode_length_s=20.0,
        auto_reset=True,
    )
    configure_estimator_observations(cfg, play=play)
    if play:
        cfg.scene.num_envs = 1
        cfg.episode_length_s = int(1e9)
        cfg.auto_reset = True
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}
        cfg.events["randomize_terrain"] = EventTermCfg(
            func=envs_mdp.randomize_terrain, mode="reset", params={}
        )
        terrain_cfg = cfg.scene.terrain
        assert terrain_cfg is not None and terrain_cfg.terrain_generator is not None
        terrain_cfg.terrain_generator.curriculum = False
        terrain_cfg.terrain_generator.num_cols = 5
        terrain_cfg.terrain_generator.num_rows = 5
        terrain_cfg.terrain_generator.border_width = 10.0
    return cfg


__all__ = [
    "HIM_COMPLEX_TERRAINS_CFG",
    "HIM_EFFORT_LIMIT_RANGE",
    "HIM_LINK_MASS_RANGE",
    "Nazarite_HIM_Complex_Terrain_Go2",
]
