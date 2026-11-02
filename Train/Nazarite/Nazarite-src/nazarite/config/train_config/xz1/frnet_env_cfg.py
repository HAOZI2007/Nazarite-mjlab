# SPDX-License-Identifier: Apache-2.0
# Adapted from mjlab_playground/getup (Copyright 2025, The mjlab Developers).
"""XZ1 port of Playground Go1 getup, under the historical FR-Net task ID."""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.envs.mdp import dr
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
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.utils.spec_config import CollisionCfg
from mjlab.viewer import ViewerConfig
from nazarite.config.robot_config.xz1_cfg import (
    XZ1_BASE_BODY,
    XZ1_FOOT_GEOMS,
    get_xz1_cfg,
)
from nazarite.mdp.xz1 import getup
from nazarite.mdp.xz1.frnet import StableRecoveryMetric

XZ1_GETUP_HEIGHT = 0.30
XZ1_GETUP_SETTLE_STEPS = 25


def Nazarite_FRNet_Recovery_XZ1(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Plain PPO baseline: random drops, relative actions, time-based smoothing."""
    robot = deepcopy(get_xz1_cfg())
    assert robot.articulation is not None
    for actuator in robot.articulation.actuators:
        actuator.delay_min_lag = 0
        actuator.delay_max_lag = 0
    foot_regex = r"^[FR][LR]_foot_link_primitive_1$"
    robot.collisions = (
        CollisionCfg(
            geom_names_expr=(".*_primitive_.*",),
            contype=1,
            conaffinity=1,
            condim={foot_regex: 6, ".*": 3},
            priority=1,
            friction={foot_regex: (1.0, 0.005, 0.0005), ".*": (0.6,)},
            solref=(0.01, 1),
        ),
    )
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
            noise=Unoise(n_min=-0.03, n_max=0.03),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }
    critic_terms = deepcopy(actor_terms)
    critic_terms["joint_pos"] = ObservationTermCfg(func=mdp.joint_pos_rel)
    critic_terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.builtin_sensor,
        params={"sensor_name": "robot/imu_lin_vel"},
    )
    observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=not play,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }
    sensors = (
        ContactSensorCfg(
            name="self_collision",
            primary=ContactMatch(mode="subtree", pattern=XZ1_BASE_BODY, entity="robot"),
            secondary=ContactMatch(
                mode="subtree", pattern=XZ1_BASE_BODY, entity="robot"
            ),
            fields=("found", "force"),
            reduce="none",
            num_slots=1,
            history_length=10,
        ),
        ContactSensorCfg(
            name="feet_ground_contact",
            primary=ContactMatch(mode="geom", pattern=XZ1_FOOT_GEOMS, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
        ),
        TerrainHeightSensorCfg(
            name="base_height_scan",
            frame=ObjRef(type="body", name=XZ1_BASE_BODY, entity="robot"),
            ray_alignment="yaw",
            max_distance=2.0,
            exclude_parent_body=True,
            include_geom_groups=(0,),
            reduction="min",
            pattern=RingPatternCfg.single_ring(radius=0.03, num_samples=4),
        ),
    )
    events = {
        "reset_base": EventTermCfg(
            func=getup.RandomDropReset,
            mode="reset",
            params={
                "fall_probability": 1.0 if play else 0.6,
                "fall_height": 0.5,
                "velocity_range": 0.5,
                "fall_case": "mixed",
            },
        ),
        "encoder_bias": EventTermCfg(
            func=dr.encoder_bias,
            mode="startup",
            params={"bias_range": (-0.015, 0.015)},
        ),
        "base_com": EventTermCfg(
            func=dr.body_com_offset,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=(XZ1_BASE_BODY,)),
                "operation": "add",
                "ranges": {0: (-0.025, 0.025), 1: (-0.025, 0.025), 2: (-0.03, 0.03)},
            },
        ),
        "geom_friction_slide": EventTermCfg(
            func=dr.geom_friction,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=(".*_primitive_.*",)),
                "operation": "abs",
                "axes": [0],
                "ranges": (0.3, 1.5),
                "shared_random": True,
            },
        ),
    }
    for name, axis, limits in (
        ("foot_friction_spin", 1, (1e-4, 2e-2)),
        ("foot_friction_roll", 2, (1e-5, 5e-3)),
    ):
        events[name] = EventTermCfg(
            func=dr.geom_friction,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=XZ1_FOOT_GEOMS),
                "operation": "abs",
                "distribution": "log_uniform",
                "axes": [axis],
                "ranges": limits,
                "shared_random": True,
            },
        )
    rewards = {
        "orientation": RewardTermCfg(func=getup.orientation_reward, weight=1.0),
        "torso_height": RewardTermCfg(
            func=getup.height_reward,
            weight=1.0,
            params={"desired_height": XZ1_GETUP_HEIGHT},
        ),
        "posture": RewardTermCfg(
            func=getup.GatedPostureReward,
            weight=1.0,
            params={
                "std": {"hip": 0.05, "thigh": 0.1, "shank": 0.15},
                "orientation_threshold": 0.01,
            },
        ),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
        "joint_vel_l2": RewardTermCfg(func=mdp.joint_vel_l2, weight=0.0),
        "joint_torques_l2": RewardTermCfg(func=mdp.joint_torques_l2, weight=-2e-6),
        "feet_support": RewardTermCfg(
            func=getup.feet_support_reward,
            weight=0.2,
            params={"settle_steps": XZ1_GETUP_SETTLE_STEPS},
        ),
        "synchronized_landing": RewardTermCfg(
            func=getup.SynchronizedLandingReward,
            weight=0.1,
            params={
                "landing_window_s": 0.12,
                "min_air_time": 0.04,
                "settle_steps": XZ1_GETUP_SETTLE_STEPS,
            },
        ),
        "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
        "self_collisions": RewardTermCfg(
            func=getup.self_collision_penalty,
            weight=-0.1,
            params={"sensor_name": "self_collision"},
        ),
    }
    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "energy": TerminationTermCfg(
            func=getup.energy_termination,
            params={
                "threshold": 400.0 if play else float("inf"),
                "settle_steps": XZ1_GETUP_SETTLE_STEPS,
            },
        ),
        "invalid_state": TerminationTermCfg(func=mdp.nan_detection),
    }
    curriculum = {}
    for name, reward, weights in (
        ("action_rate_weight", "action_rate_l2", (-0.01, -0.05, -0.08, -0.1)),
        ("joint_vel_weight", "joint_vel_l2", (0.0, -0.005, -0.008, -0.01)),
        (
            "joint_torques_weight",
            "joint_torques_l2",
            (-2e-6, -5e-6, -8e-6, -1e-5),
        ),
    ):
        curriculum[name] = CurriculumTermCfg(
            func=mdp.reward_curriculum,
            params={
                "reward_name": reward,
                "stages": [
                    {"step": i * 24, "weight": weight}
                    for i, weight in zip((0, 500, 900, 1200), weights)
                ],
            },
        )
    curriculum["energy_threshold"] = CurriculumTermCfg(
        func=mdp.termination_curriculum,
        params={
            "termination_name": "energy",
            "stages": [
                {"step": i * 24, "params": {"threshold": power}}
                for i, power in ((500, 1000.0), (900, 700.0), (1200, 400.0))
            ],
        },
    )
    if play:
        rewards["action_rate_l2"].weight = -0.1
        rewards["joint_vel_l2"].weight = -0.01
        rewards["joint_torques_l2"].weight = -1e-5
        curriculum = {}
    strict = {
        "hold_duration_s": 0.8,
        "upright_gravity_z_max": -0.90,
        "min_relative_height": 0.25,
        "min_foot_contacts": 3,
        "contact_force_threshold": 5.0,
        "max_angular_velocity": 1.5,
        "max_joint_position_error": 0.35,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
    }
    metrics = {
        "getup_success": MetricsTermCfg(
            func=getup.GetupSuccess,
            reduce="last",
            params={"desired_height": XZ1_GETUP_HEIGHT},
        ),
        "recovery_success": MetricsTermCfg(
            func=StableRecoveryMetric, reduce="max", params=strict
        ),
        "recovery_final_stable": MetricsTermCfg(
            func=StableRecoveryMetric, reduce="last", params=strict
        ),
        "joint_speed_rms": MetricsTermCfg(func=getup.joint_speed_rms),
        "joint_torque_rms": MetricsTermCfg(func=getup.joint_torque_rms),
        "action_delta_rms": MetricsTermCfg(func=getup.action_delta_rms),
        "base_angular_speed": MetricsTermCfg(func=getup.base_angular_speed),
        "peak_power_w": MetricsTermCfg(func=getup.mechanical_power, reduce="max"),
    }
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": robot},
            sensors=sensors,
            num_envs=1 if play else 1024,
            env_spacing=2.5,
        ),
        observations=observations,
        actions={
            "joint_pos": getup.SettleRelativeJointPositionActionCfg(
                entity_name="robot",
                actuator_names=(".*",),
                scale=0.6,
                settle_steps=XZ1_GETUP_SETTLE_STEPS,
            )
        },
        commands={},
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum=curriculum,
        metrics=metrics,
        # Keep XZ1's stable 2 ms physics step; both ports control at 50 Hz.
        sim=SimulationCfg(
            nconmax=128,
            njmax=1000,
            contact_sensor_maxmatch=256,
            mujoco=MujocoCfg(
                timestep=0.002,
                iterations=10,
                ls_iterations=30,
                impratio=10,
                cone="elliptic",
            ),
        ),
        decimation=10,
        episode_length_s=6.0,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name=XZ1_BASE_BODY,
            distance=2.5,
            elevation=-20.0,
            azimuth=135.0,
        ),
    )
