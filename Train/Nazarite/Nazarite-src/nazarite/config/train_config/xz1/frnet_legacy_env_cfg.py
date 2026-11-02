"""Legacy paper-reward FR-Net configuration, retained for old checkpoints."""

import math
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.envs.mdp import dr
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    GridPatternCfg,
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig
from nazarite.config.robot_config.xz1_cfg import (
    XZ1_BASE_BODY,
    XZ1_FOOT_GEOMS,
    XZ1_FOOT_SITES,
    XZ1_HIP_BODIES,
    XZ1_SHANK_BODIES,
    XZ1_THIGH_BODIES,
    get_xz1_cfg,
)
from nazarite.mdp.fr_net import frnet_observations, frnet_terminations, paper_rewards
from nazarite.mdp.xz1 import frnet
from nazarite.mdp.xz1.frnet_curriculum import RecoveryCurriculum

XZ1_FRNET_PHYSICS_DT = 0.002
XZ1_FRNET_DECIMATION = 10
XZ1_FRNET_MIN_HEIGHT = 0.25
# Table I leaves epsilon unspecified; use 0.15 explicitly for both Eq. (8)
# kernels, matching the reference repository's upright Gaussian default.
XZ1_FRNET_REWARD_EPSILON = 0.15
XZ1_FRNET_EPISODE_STEPS = 350
# Paper-style component groups, not Go2's legacy four-leg targets. The four
# separate XZ1 foot bodies retain nominal inertia; no massless frames are scaled.
XZ1_FRNET_MASS_GROUPS = (
    (XZ1_BASE_BODY,),
    XZ1_HIP_BODIES,
    XZ1_THIGH_BODIES,
    XZ1_SHANK_BODIES,
)


def Nazarite_FRNet_Recovery_XZ1(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Train on a plane with success-gated pose and regularization stages."""
    robot = deepcopy(get_xz1_cfg())
    # A short, fixed baseline latency helps isolate recovery learning. This is
    # local to FR-Net; HIM's actuator/observation latency settings are separate.
    assert robot.articulation is not None
    for actuator in robot.articulation.actuators:
        actuator.delay_min_lag = 0
        actuator.delay_max_lag = 0

    ground = ContactMatch(mode="body", pattern="terrain")
    sensors = [
        ContactSensorCfg(
            name="feet_ground_contact",
            primary=ContactMatch(mode="geom", pattern=XZ1_FOOT_GEOMS, entity="robot"),
            secondary=ground,
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
        ),
        TerrainHeightSensorCfg(
            name="foot_height_scan",
            frame=tuple(
                ObjRef(type="site", name=name, entity="robot")
                for name in XZ1_FOOT_SITES
            ),
            ray_alignment="yaw",
            max_distance=1.5,
            exclude_parent_body=True,
            include_geom_groups=(0,),
            pattern=RingPatternCfg.single_ring(radius=0.02, num_samples=4),
        ),
        TerrainHeightSensorCfg(
            name="base_height_scan",
            frame=ObjRef(type="body", name=XZ1_BASE_BODY, entity="robot"),
            ray_alignment="yaw",
            max_distance=1.5,
            exclude_parent_body=True,
            include_geom_groups=(0,),
            pattern=RingPatternCfg.single_ring(radius=0.03, num_samples=4),
            reduction="min",
        ),
        TerrainHeightSensorCfg(
            name="frnet_terrain_height_map",
            frame=ObjRef(type="body", name=XZ1_BASE_BODY, entity="robot"),
            ray_alignment="yaw",
            max_distance=2.0,
            exclude_parent_body=True,
            include_geom_groups=(0,),
            # 17 x 11 = 187 privileged scan dots (paper observation dimension).
            pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
            reduction="none",
        ),
    ]
    for name, bodies in (
        ("trunk_ground_touch", (XZ1_BASE_BODY,)),
        ("hip_ground_touch", XZ1_HIP_BODIES),
        ("thigh_ground_touch", XZ1_THIGH_BODIES),
        ("shank_ground_touch", XZ1_SHANK_BODIES),
    ):
        sensors.append(
            ContactSensorCfg(
                name=name,
                primary=ContactMatch(mode="body", pattern=bodies, entity="robot"),
                secondary=ground,
                fields=("found", "force"),
                reduce="netforce",
                num_slots=1,
                history_length=XZ1_FRNET_DECIMATION,
            )
        )

    actor, critic, aux = frnet_observations.make_frnet_observation_terms(
        XZ1_FRNET_MASS_GROUPS
    )
    next_proprio = deepcopy(actor)
    for term in next_proprio.values():
        term.history_length = 0
    observations = {
        "actor": ObservationGroupCfg(terms=actor, enable_corruption=not play),
        "critic": ObservationGroupCfg(terms=critic, enable_corruption=False),
        "frnet_aux": ObservationGroupCfg(terms=aux, enable_corruption=False),
        "frnet_proprio": ObservationGroupCfg(
            terms=next_proprio, enable_corruption=False
        ),
    }
    for group in observations.values():
        group.concatenate_terms = True
        group.history_length = None
        group.flatten_history_dim = True
        group.nan_policy = "error"

    all_joints = SceneEntityCfg("robot", joint_names=(".*",))
    success_params = {
        "hold_duration_s": 0.8,
        "upright_gravity_z_max": -0.90,
        "min_relative_height": XZ1_FRNET_MIN_HEIGHT,
        "min_foot_contacts": 3,
        "contact_force_threshold": 5.0,
        "max_angular_velocity": 1.5,
        "max_joint_position_error": 0.35,
        "asset_cfg": all_joints,
    }
    rewards = {
        # FR-Net Table I, in paper order. Keep the signed additive objective;
        # no height shaping or terminal bonus. Curriculum scales regularizers
        # during training and restores these final paper weights at level 5.
        "base_orientation": RewardTermCfg(
            func=paper_rewards.base_orientation, weight=-0.5
        ),
        "upright": RewardTermCfg(
            func=paper_rewards.upright_orientation,
            weight=6.0,
            params={"epsilon": XZ1_FRNET_REWARD_EPSILON},
        ),
        "stand_pose": RewardTermCfg(
            func=paper_rewards.target_posture,
            weight=4.0,
            params={"epsilon": XZ1_FRNET_REWARD_EPSILON, "asset_cfg": all_joints},
        ),
        "foot_contacts": RewardTermCfg(
            func=paper_rewards.feet_contact,
            weight=0.3,
            params={"sensor_name": "feet_ground_contact"},
        ),
        "body_contact": RewardTermCfg(func=paper_rewards.body_contact, weight=-0.2),
        "safety_force": RewardTermCfg(
            func=paper_rewards.safety_force,
            weight=-1.0e-2,
            params={"sensor_name": "shank_ground_touch"},
        ),
        "body_bias": RewardTermCfg(func=paper_rewards.body_bias, weight=-0.1),
        "joint_limits": RewardTermCfg(
            func=paper_rewards.position_limits,
            weight=-1.0,
            params={"asset_cfg": all_joints},
        ),
        "joint_velocity_limit": RewardTermCfg(
            func=paper_rewards.angular_velocity_limit,
            weight=-0.1,
            params={"threshold": 0.8, "asset_cfg": all_joints},
        ),
        "joint_acc": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-6),
        "joint_vel": RewardTermCfg(func=mdp.joint_vel_l2, weight=-1.0e-2),
        "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
        "joint_torques": RewardTermCfg(func=mdp.joint_torques_l2, weight=-5.0e-4),
    }
    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "invalid_state": TerminationTermCfg(func=mdp.nan_detection),
        "penetrated_ground": TerminationTermCfg(
            func=frnet_terminations.base_below_safe_height
        ),
    }
    events = {
        "reset_base": EventTermCfg(
            func=frnet.FlatRecoveryReset,
            mode="reset",
            params={
                "standing_probability": 0.05 if not play else 0.0,
                "semi_fallen_probability": 0.15 if not play else 0.0,
                "fall_case": "mixed",
                "use_curriculum": not play,
                "rehearsal_probability": 0.20,
            },
        ),
        "frnet_component_inertia": EventTermCfg(
            func=dr.pseudo_inertia,
            mode="reset",
            params={
                "alpha_range": (0.5 * math.log(0.85), 0.5 * math.log(1.15)),
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=sum(XZ1_FRNET_MASS_GROUPS, ()),
                    preserve_order=True,
                ),
            },
        ),
        "ground_friction": EventTermCfg(
            func=dr.geom_friction,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=(".*_primitive_.*",)),
                "operation": "abs",
                "ranges": (0.5, 1.2),
                "shared_random": True,
            },
        ),
        "pd_gains": EventTermCfg(
            func=dr.pd_gains,
            mode="startup",
            params={
                "kp_range": (0.9, 1.1),
                "kd_range": (0.9, 1.1),
                "operation": "scale",
            },
        ),
        "encoder_bias": EventTermCfg(
            func=dr.encoder_bias,
            mode="startup",
            params={"bias_range": (-0.01, 0.01)},
        ),
    }
    condition_params = {
        key: value for key, value in success_params.items() if key != "hold_duration_s"
    }
    metrics = {
        f"recovery_{condition}_pass": MetricsTermCfg(
            func=frnet_terminations.recovery_condition,
            params={"condition": condition, **condition_params},
        )
        for condition in (
            "upright",
            "height",
            "contacts",
            "angular_velocity",
            "joint_pose",
            "strict",
        )
    }
    metrics.update(
        {
            f"reset_{case}_fraction": MetricsTermCfg(
                func=frnet.reset_fraction, params={"case": case}
            )
            for case in frnet.FALL_CASES
        }
    )
    metrics["recovery_success"] = MetricsTermCfg(
        func=frnet.StableRecoveryMetric,
        params={**success_params, "track_progress": True},
        reduce="max",
    )
    metrics["fallen_recovery_success"] = MetricsTermCfg(
        func=frnet.StableRecoveryMetric,
        params={**success_params, "fallen_only": True},
        reduce="max",
    )
    metrics["recovery_final_stable"] = MetricsTermCfg(
        func=frnet.StableRecoveryMetric, params=success_params, reduce="last"
    )

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=1 if play else 1024,
            env_spacing=2.5,
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": robot},
            sensors=tuple(sensors),
        ),
        observations=observations,
        actions={
            "joint_pos": frnet.RecoveryJointPositionActionCfg(
                entity_name="robot",
                actuator_names=(".*",),
                scale=0.7,
                max_target_delta=0.15,
                settle_steps=6,
                ramp_steps=6,
            )
        },
        commands={},
        events=events,
        rewards=rewards,
        terminations=terminations,
        metrics=metrics,
        curriculum={}
        if play
        else {
            "recovery": CurriculumTermCfg(
                func=RecoveryCurriculum,
                params={
                    "min_episodes_per_case": 128,
                    "min_stage_steps": 12000,
                    "promotion_success_rate": 0.70,
                    "promotion_final_stable_rate": 0.60,
                },
            )
        },
        sim=SimulationCfg(
            nconmax=128,
            njmax=1000,
            contact_sensor_maxmatch=128,
            mujoco=MujocoCfg(
                timestep=XZ1_FRNET_PHYSICS_DT, iterations=10, ls_iterations=30
            ),
        ),
        decimation=XZ1_FRNET_DECIMATION,
        # Multiply the control period first, as max_episode_length divides by
        # that same period before ceil(). Left-associating 350 * .002 * 10
        # produces 7.000000000000001 and accidentally schedules 351 steps.
        episode_length_s=(
            (XZ1_FRNET_PHYSICS_DT * XZ1_FRNET_DECIMATION) * XZ1_FRNET_EPISODE_STEPS
        ),
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name=XZ1_BASE_BODY,
            distance=2.5,
            elevation=-20.0,
            azimuth=135.0,
        ),
    )
