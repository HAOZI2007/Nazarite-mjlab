"""Independent Go2 WMP + HIM complex-terrain task."""

from __future__ import annotations

from copy import deepcopy
from typing import cast

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
    CameraSensorCfg,
    ContactSensorCfg,
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp as velocity_mdp
from nazarite.config.robot_config.go2_cfg import GO2_BASE_BODY, GO2_FOOT_SITES
from nazarite.config.train_config.env_cfgs.him_complex_env_cfg import (
    Nazarite_HIM_Complex_Terrain_Go2,
)
from nazarite.config.train_config.env_cfgs.wmp_terrain import (
    WMP_FLAT_TERRAINS_CFG,
    WMP_FORWARD_TERRAIN_NAMES,
    WMP_NUM_ROWS,
    WMP_OMNI_TERRAIN_NAMES,
    WMP_TERRAIN_NAMES,
    WMP_TERRAINS_CFG,
)
from nazarite.mdp import rewards as custom_rewards
from nazarite.mdp import (
    wmp_curriculums,
    wmp_events,
    wmp_metrics,
    wmp_observations,
    wmp_rewards,
    wmp_terminations,
)
from nazarite.mdp.commands import TerrainConditionedVelocityCommandCfg
from nazarite.rl.wmp.world_model import WMPWorldModelCfg

WMP_FEATURE_DIM = 512
WMP_REFERENCE_INIT_PROBABILITY = 0.0
"""Share of episodes that start mid-stride from a reference clip (0 disables)."""
WMP_DEPTH_CAMERA_NAME = "wmp_depth_camera"
WMP_TRACKING_CLIP_TERRAINS = ("gap", "high_wall", "single_beam")
WMP_FEET_EDGE_TERRAINS = ("gap", "high_wall", "plum_blossom", "single_beam")
WMP_CHEAT_TERRAINS = (
    "low_frame",
    "high_wall",
    "plum_blossom",
    "gap",
    "tire_array",
    "single_beam",
)
# Tiles with a ceiling low enough that crouching is mandatory: the beam of the
# low frame sits at 0.42-0.60 m while the standing Go2 is about 0.44 m tall.
WMP_LOW_CEILING_TERRAINS = ("low_frame",)


def Nazarite_WMP_HIM_Complex_Terrain_Go2(
    play: bool = False, flat: bool = False
) -> ManagerBasedRlEnvCfg:
    """Build Go2 WMP with a dedicated progressive 2027 obstacle course.

    Args:
      play: Build the single-environment playback variant.
      flat: Build stage one of the two-stage schedule: the same robot, reward
        set and world-model pipeline on a plain start deck. Stage one answers
        "can it walk at the commanded speed with an upright, periodic gait"
        before the obstacle course is allowed to confound that question. Load
        its checkpoint into the terrain stage rather than training it from
        scratch.
    """
    cfg = Nazarite_HIM_Complex_Terrain_Go2(play=play)
    terrain_cfg = cfg.scene.terrain
    assert terrain_cfg is not None
    terrain_cfg.terrain_generator = deepcopy(
        WMP_FLAT_TERRAINS_CFG if flat else WMP_TERRAINS_CFG
    )
    if play:
        # Keep the ordered type-by-column grid and its easy-to-hard row gradient.
        terrain_cfg.terrain_generator.curriculum = True
        terrain_cfg.terrain_generator.num_rows = WMP_NUM_ROWS
        terrain_cfg.terrain_generator.num_cols = len(WMP_TERRAIN_NAMES) + len(
            WMP_OMNI_TERRAIN_NAMES
        )
        terrain_cfg.terrain_generator.difficulty_range = (0.0, 0.9)
    else:
        # Training starts at the easiest row and advances only via traversal.
        terrain_cfg.max_init_terrain_level = 0
        # The HIM base config ships its own terrain curriculum; the WMP course
        # replaces it, and a single flat deck has nothing to advance at all.
        cfg.curriculum.pop("terrain_levels", None)
        if not flat:
            # A single flat deck has nothing to advance, and the gait gate below
            # reads obstacle-tile statistics that do not exist there.
            cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
                func=wmp_curriculums.terrain_levels_wmp_reference,
                params={
                    "command_name": "twist",
                    # Promote on distance *and* on a walkable gait: distance
                    # alone let the 2026-10-18 run reach level 4.9 with a
                    # nose-down crawl that crossed seven of the nine families.
                    "min_gait_match": 0.62,
                    "max_gait_tilt": 0.35,
                    "min_base_clearance": 0.25,
                    "crouch_terrain_names": WMP_LOW_CEILING_TERRAINS,
                    # The omni pad never advances and never demotes: it is
                    # practice ground, and a level change would teleport it onto
                    # a different row of the same flat deck for no reason.
                    "omni_terrain_names": WMP_OMNI_TERRAIN_NAMES,
                },
            )
        cfg.curriculum["forward_velocity"] = CurriculumTermCfg(
            func=wmp_curriculums.forward_velocity_wmp_reference,
            params={
                "command_name": "twist",
                "reward_name": "track_linear_velocity",
                # Same mask problem as the command sampler: the stage-one deck
                # has one terrain type, so the obstacle names would match nothing
                # and the curriculum refuses to start.
                "terrain_names": (("flat",) if flat else WMP_FORWARD_TERRAIN_NAMES),
            },
        )
    cfg.events["reset_base"].func = wmp_events.reset_root_state_uniform_facing_course
    # Keep resets near the center of each open 8 x 8 m terrain tile, and do not
    # add an airborne z offset on top of the nominal 0.32 m standing height.
    reset_pose_range = deepcopy(cfg.events["reset_base"].params["pose_range"])
    reset_pose_range["y"] = (-0.2, 0.2)
    reset_pose_range["z"] = (0.0, 0.0)
    cfg.events["reset_base"].params["pose_range"] = reset_pose_range
    # Spawn with a little forward momentum. From rest the first trench sits
    # 1.1-2.1 m ahead of the spawn point, and a policy that has to accelerate from
    # zero spends most of the episode not touching the obstacle at all. Keep it
    # modest: teleporting the trunk to 0.8 m/s while the feet are still planted
    # teaches the robot to pitch nose-down and let the body fall forward onto the
    # front legs, which is exactly the gait the 2026-10-17_22-32-26 run settled
    # into.
    cfg.events["reset_base"].params["velocity_range"] = {"x": (0.1, 0.35)}
    # Reference state initialisation. Registered after ``reset_base`` on purpose:
    # events run in registration order, so this overwrites joints and velocities
    # that the course reset has just set. Off by default, because it changes what
    # the policy sees at the start of every episode and stage one is the A/B
    # baseline for it.
    cfg.events["reference_state_init"] = EventTermCfg(
        func=wmp_events.reference_state_initialization,
        mode="reset",
        params={
            "probability": WMP_REFERENCE_INIT_PROBABILITY,
            "motion_dir": None,
        },
    )
    # The shared config annotates the command map with the generic base type,
    # while this entry is concretely a TerrainConditionedVelocityCommandCfg.
    twist_cfg = cast(TerrainConditionedVelocityCommandCfg, cfg.commands["twist"])
    # Stage one is omni-directional on purpose: it is the only place the actor
    # ever sees lin_vel_y and ang_vel_z. An earlier version marked the flat deck
    # forward-only as well, so both channels were zero for the whole schedule and
    # the deployed policy could only walk straight ahead.
    twist_cfg.forward_only_terrain_names = () if flat else WMP_FORWARD_TERRAIN_NAMES
    twist_cfg.rel_forward_envs = 0.0
    # Every non-forward environment (the whole flat deck, and the omni pad inside
    # the obstacle course) drops to a zero command this often, which is the only
    # place stand_still can earn anything.
    twist_cfg.rel_standing_envs = 0.15
    # Source WMP forward terrains sample [0, 0.5] initially and expand the upper
    # limit to 1.0 via the reward curriculum. The port additionally zeroed every
    # sample at or below 0.2 m/s, which wasted ~40 % of the episodes as "stand
    # still" on the tiles that need the most practice. Sample from 0.15 m/s
    # upwards and drop the zeroing instead.
    twist_cfg.forward_only_x_min = 0.15
    twist_cfg.forward_only_x_max = 0.5
    twist_cfg.forward_only_zero_below = 0.0
    # Heading mode, as in the reference implementation: each resample picks a
    # heading target and the base update turns that error into a bounded yaw rate,
    # so the robot corrects drift instead of being clamped to zero yaw. The
    # forward-only obstacle columns always target +x (see the command class).
    twist_cfg.heading_command = True
    twist_cfg.ranges.heading = (-3.14, 3.14)
    twist_cfg.ranges.ang_vel_z = (-1.0, 1.0)
    twist_cfg.ranges.lin_vel_x = (-0.6, 0.8)
    # Half the non-forward environments (flat deck in stage one, omni pad in stage
    # two) hold a heading target while the other half still take an explicit yaw
    # rate command, so both interfaces stay trained.
    twist_cfg.rel_heading_envs = 0.5

    # Match the source AMP domain-randomization ranges using mjlab events.
    # The source randomizes every robot collision geom, so drop HIM's foot-only
    # friction event instead of randomizing the same geoms twice.
    cfg.events.pop("foot_friction", None)
    cfg.events["robot_friction"] = EventTermCfg(
        func=dr.geom_friction,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=(".*",)),
            "operation": "abs",
            "ranges": (0.5, 2.0),
            "shared_random": True,
        },
    )
    cfg.events["base_added_mass"] = EventTermCfg(
        func=dr.body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=(GO2_BASE_BODY,)),
            "operation": "add",
            "ranges": (0.0, 3.0),
        },
    )
    cfg.events.pop("pseudo_inertia", None)
    cfg.events["link_mass"] = EventTermCfg(
        func=dr.body_mass,
        mode="startup",
        params={
            # The source randomizes each non-root link's mass independently.
            # The root is handled separately as an added payload above.
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=(rf"^(?!{GO2_BASE_BODY}$).+",)
            ),
            "operation": "scale",
            "ranges": (0.8, 1.2),
        },
    )
    cfg.events["pd_gains"] = EventTermCfg(
        func=dr.pd_gains,
        mode="startup",
        params={
            "kp_range": (0.8, 1.2),
            "kd_range": (0.8, 1.2),
            "asset_cfg": SceneEntityCfg("robot"),
            "operation": "scale",
        },
    )
    # Per-substep torque scaling in the source is approximated by per-world
    # actuator strength limits; the policy gains are independently randomized.
    cfg.events["effort_limits"] = EventTermCfg(
        func=dr.effort_limits,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "operation": "scale",
            "effort_limit_range": (0.8, 1.2),
        },
    )
    if play:
        cfg.events.pop("push_robot", None)
    else:
        cfg.events["push_robot"] = EventTermCfg(
            func=wmp_events.push_root_velocity_xy,
            mode="interval",
            interval_range_s=(15.0, 15.0),
            params={"max_velocity": 1.0},
        )
    cfg.events["depth_camera_mount"] = EventTermCfg(
        func=wmp_events.randomize_depth_camera_mount,
        mode="startup",
        params={"camera_name": WMP_DEPTH_CAMERA_NAME},
    )

    robot_cfg = cfg.scene.entities["robot"]
    assert robot_cfg.articulation is not None
    physics_dt = cfg.sim.mujoco.timestep
    max_action_lag = round(0.01 / physics_dt)
    for actuator in robot_cfg.articulation.actuators:
        actuator.delay_min_lag = 0
        actuator.delay_max_lag = max_action_lag
        actuator.delay_hold_prob = 0.0
        actuator.delay_update_period = cfg.decimation
        actuator.delay_per_env_phase = False

    cfg.terminations = {
        "time_out": TerminationTermCfg(
            func=wmp_terminations.reference_time_out, time_out=True
        ),
        "base_contact": TerminationTermCfg(
            func=wmp_terminations.base_contact,
            params={
                "sensor_name": "nonfoot_ground_touch",
                "force_threshold": 1.0,
                # Low-frame contact is allowed when the Go2 remains upright;
                # only a low/inverted base contact is terminal.
                "require_fall_state": True,
                "min_base_height": 0.18,
            },
        ),
        "velocity_violation": TerminationTermCfg(
            func=wmp_terminations.velocity_violation,
            params={
                "command_name": "twist",
                "terrain_level_threshold": 3,
                "velocity_error_threshold": 1.5,
            },
        ),
        "stall_timeout": TerminationTermCfg(
            # A hard floor under the speed command. Until the 2026-10-18 run the
            # only pressure to keep up was the tracking reward, and surviving at
            # a third of the command scored better than trying to walk: the
            # curriculum reached level 4.9 with a crawl that crossed seven of
            # the nine terrain families.
            func=wmp_terminations.stall_timeout,
            params={
                "command_name": "twist",
                "speed_fraction": 0.4,
                "command_threshold": 0.1,
                "time_limit": 2.0,
                # Stage one has a single difficulty row, so the level gate that
                # keeps the termination away from the easy rows stays at zero.
                "min_terrain_level": 0 if flat else 2,
            },
        ),
        "fall": TerminationTermCfg(
            func=wmp_terminations.fall,
            params={"downward_velocity_threshold": -3.0},
        ),
        "forward_course_bounds": TerminationTermCfg(
            func=wmp_terminations.forward_course_bounds,
            params={
                "terrain_names": WMP_FORWARD_TERRAIN_NAMES,
                # Each track is 7.20 m wide with a 0.40 m void on either side, so
                # 0.60 leaves 0.20 m of deck before the edge: wide enough that a
                # one-frame overshoot does not end the episode, narrow enough that
                # a policy cannot walk beside the obstacles instead of over them.
                "lateral_margin": 0.60,
            },
        ),
    }
    for sensor in cfg.scene.sensors:
        if not isinstance(sensor, ContactSensorCfg):
            continue
        if sensor.name == "nonfoot_ground_touch":
            # Capture all physics substeps in the policy step for brief impacts.
            # Net force matches the source rigid-body contact-force tensor.
            sensor.reduce = "netforce"
            sensor.history_length = cfg.decimation
        elif sensor.name == "leg_ground_contact":
            # The source collision reward penalizes thigh/calf contact, not hips.
            sensor.primary.pattern = r"^(FL|FR|RL|RR)_(thigh|calf)_collision_[0-9]+$"

    camera = CameraSensorCfg(
        name=WMP_DEPTH_CAMERA_NAME,
        parent_body=f"robot/{GO2_BASE_BODY}",
        pos=(0.28, 0.0, 0.12),
        # Forward-facing optical axis pitched down by about 20 degrees.
        quat=(0.579228, 0.405580, -0.405580, -0.579228),
        width=64,
        height=64,
        fovy=58.0,
        data_types=("depth",),
        # Match the reference Warp camera, which raycasts the terrain only.
        enabled_geom_groups=(0,),
        use_shadows=False,
        use_textures=False,
        # Render at 10 Hz on a 50 Hz policy, i.e. every fifth policy step. This
        # is the reference setting ("update_interval=5 works without retraining,
        # 8 is worse") and it removes ~80 % of the per-step BVH ray casting that
        # dominates collection time.
        update_interval=5,
    )
    support_scan = TerrainHeightSensorCfg(
        name="wmp_feet_support",
        frame=tuple(
            ObjRef(type="site", name=name, entity="robot") for name in GO2_FOOT_SITES
        ),
        pattern=RingPatternCfg.single_ring(
            radius=0.10, num_samples=8, include_center=True
        ),
        ray_alignment="world",
        max_distance=1.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
        reduction="none",
    )
    base_clearance = TerrainHeightSensorCfg(
        name="wmp_base_clearance",
        frame=(ObjRef(type="body", name=GO2_BASE_BODY, entity="robot"),),
        # A small ring right under the trunk. The mean keeps a step edge from
        # reading as a crouch, which a min reduction would do.
        pattern=RingPatternCfg.single_ring(
            radius=0.05, num_samples=4, include_center=True
        ),
        ray_alignment="world",
        max_distance=1.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
        reduction="mean",
    )
    cfg.scene.sensors = (*cfg.scene.sensors, camera, support_scan, base_clearance)
    cfg.observations["world_model_feature"] = ObservationGroupCfg(
        terms={
            "rssm_deter": ObservationTermCfg(
                func=wmp_observations.empty_world_model_feature,
                params={"feature_dim": WMP_FEATURE_DIM},
            )
        },
        concatenate_terms=True,
        enable_corruption=False,
        history_length=0,
    )
    # The discriminator's own view of the robot: absolute, noise-free, history-free,
    # and deliberately not part of the actor's or the critic's observation. It is
    # scored by the motion prior and turned into the style reward, so it has to be
    # the same 30 numbers the clip loader produces.
    cfg.observations["amp"] = ObservationGroupCfg(
        terms={
            "state": ObservationTermCfg(func=wmp_observations.amp_state),
        },
        concatenate_terms=True,
        enable_corruption=False,
        history_length=0,
    )

    # Reward set aligned with the reference recipe. The manual gait-shaping
    # terms are gone: measured on 2026-10-19_22-53-44, the four terms that paid
    # for lifting a foot over an obstacle were worth 1.5 % of the reward while the
    # six smoothness/energy terms were worth 17.7 %. Tuning those knobs changed
    # the objective by less than the measurement noise. Style is now the
    # discriminator's job.
    cfg.rewards = {
        "track_linear_velocity": RewardTermCfg(
            func=wmp_rewards.tracking_linear_velocity,
            weight=1.5,
            params={
                "command_name": "twist",
                "sigma": 0.25,
                "asset_cfg": SceneEntityCfg("robot"),
                "terrain_names": WMP_TRACKING_CLIP_TERRAINS,
            },
        ),
        "track_angular_velocity": RewardTermCfg(
            func=wmp_rewards.tracking_angular_velocity,
            # bytedance/WMP tracks yaw at 0.5 against 1.5 for linear velocity.
            # At parity the two terms made an untrained policy prefer spinning.
            weight=0.5,
            params={
                "command_name": "twist",
                "sigma": 0.25,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "joint_torques_l2": RewardTermCfg(
            func=envs_mdp.joint_torques_l2,
            weight=-1.0e-4,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "joint_acc_l2": RewardTermCfg(
            func=envs_mdp.joint_acc_l2,
            weight=-2.5e-7,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "action_rate_l2": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.03),
        "leg_collision": RewardTermCfg(
            func=velocity_mdp.illegal_contact,
            # At -0.05 a thigh scraping the ground cost 0.001 per step, which is
            # why the crouched gait of the 2026-10-17 runs could drag its shins
            # across the terrain for free.
            weight=-1.0,
            params={"sensor_name": "leg_ground_contact", "force_threshold": 0.1},
        ),
        "joint_pos_limits": RewardTermCfg(
            func=wmp_rewards.joint_position_limits,
            weight=-10.0,
            params={"asset_cfg": SceneEntityCfg("robot"), "soft_limit_factor": 0.9},
        ),
        "joint_velocity_limits": RewardTermCfg(
            func=wmp_rewards.joint_velocity_limits,
            weight=-10.0,
            params={"asset_cfg": SceneEntityCfg("robot"), "velocity_limit": 27.0},
        ),
        "lin_vel_z": RewardTermCfg(
            func=wmp_rewards.lin_vel_z_l2,
            # Same as the reference. Leaping needs vertical base velocity, but the
            # 2026-10-17_22-32-26 run's base bob (0.5 weight) is exactly what this
            # should charge.
            weight=-1.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "dof_error": RewardTermCfg(
            func=wmp_rewards.dof_error,
            # Replaces the tolerance-banded posture reward. A single smooth basin
            # around the default stand, so a hind leg held straight back (the
            # crawl's "kickstand") is charged every step instead of only once it
            # leaves a band.
            weight=-0.04,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "is_terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-10.0),
        "stand_still": RewardTermCfg(
            func=custom_rewards.stand_still,
            weight=-0.5,
            params={
                "command_name": "twist",
                "command_threshold": 0.1,
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            },
        ),
        "feet_air_time": RewardTermCfg(
            func=wmp_rewards.feet_air_time,
            # The reference's term verbatim: 0.5 s threshold, two-sided, so a
            # normal 0.26 s swing is charged 0.24 per contact. Clamping this to
            # zero (the 2026-10-20 run) left swing timing to the discriminator
            # alone, and the discriminator had 1% of the objective, so nothing
            # shaped the gait at all.
            weight=0.5,
            params={
                "sensor_name": "feet_ground_contact",
                "threshold": 0.5,
                "command_name": "twist",
                "command_threshold": 0.1,
            },
        ),
        "feet_stumble": RewardTermCfg(
            func=wmp_rewards.feet_stumble,
            weight=-0.1,
            params={"sensor_name": "feet_ground_contact", "horizontal_ratio": 4.0},
        ),
        "feet_edge": RewardTermCfg(
            func=wmp_rewards.feet_edge,
            weight=-1.0,
            params={
                "support_sensor_name": "wmp_feet_support",
                "contact_sensor_name": "feet_ground_contact",
                "command_name": "twist",
                "command_threshold": 0.1,
                "terrain_names": WMP_FEET_EDGE_TERRAINS,
                "curriculum_start_iteration": None if play else 4000,
                "curriculum_end_iteration": 10000,
                "curriculum_start_scale": 0.1,
                "curriculum_end_scale": 1.0,
            },
        ),
        "cheat": RewardTermCfg(
            func=wmp_rewards.cheat,
            weight=-1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "command_name": "twist",
                "terrain_names": WMP_CHEAT_TERRAINS,
            },
        ),
        "stuck": RewardTermCfg(
            func=wmp_rewards.stuck,
            weight=-1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "command_name": "twist",
                "speed_threshold": 0.12,
                "grace_steps": 25,
            },
        ),
    }

    # Gait diagnostics. Cheap, and without them the only way to tell a trot from
    # the nose-down crawl that these terms were added to remove was to watch the
    # playback video. The curriculum gate reads the same accumulators.
    cfg.metrics.update(
        {
            "gait_diagonal_match": MetricsTermCfg(
                func=wmp_metrics.gait_diagonal_match,
                params={
                    "period": 0.6,
                    "offset": [0.0, 0.5, 0.5, 0.0],
                    "threshold": 0.56,
                    "command_threshold": 0.1,
                    "command_name": "twist",
                },
            ),
            "gait_base_clearance": MetricsTermCfg(
                func=wmp_metrics.gait_base_clearance,
                params={"sensor_name": "wmp_base_clearance"},
            ),
            "gait_base_tilt": MetricsTermCfg(func=wmp_metrics.gait_base_tilt),
            "gait_hind_swing_ratio": MetricsTermCfg(
                func=wmp_metrics.gait_hind_swing_ratio,
                params={"sensor_name": "feet_ground_contact"},
            ),
            "gait_front_hind_swing_gap": MetricsTermCfg(
                func=wmp_metrics.gait_front_hind_swing_gap,
                params={"sensor_name": "feet_ground_contact"},
            ),
        }
    )

    # No gait clock. Swing timing is the discriminator's job from here on, and a
    # hard-coded 0.6 s phase would fight whatever timing the reference clips ask
    # for. The estimator group is popped too, because it is a copy of the critic
    # terms and its width sets the HIM target slices.
    for group in ("actor", "critic", "estimator"):
        cfg.observations[group].terms.pop("phase", None)

    # Respect the HIM task's delay-free estimator observations and dimensions.
    if play:
        cfg.scene.num_envs = 1
    return cfg


def wmp_go2_world_model_cfg() -> WMPWorldModelCfg:
    """Return reference-sized WMP model dimensions for Go2."""
    return WMPWorldModelCfg(action_dim=12, prop_dim=33)


__all__ = [
    "WMP_DEPTH_CAMERA_NAME",
    "WMP_FEATURE_DIM",
    "WMP_REFERENCE_INIT_PROBABILITY",
    "Nazarite_WMP_HIM_Complex_Terrain_Go2",
    "wmp_go2_world_model_cfg",
]
