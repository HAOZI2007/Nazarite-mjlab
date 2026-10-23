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
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.lab_api.string import resolve_matching_names_values
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig
from nazarite.config.robot_config.xz1_cfg import (
    XZ1_ACTION_SCALE,
    XZ1_BASE_BODY,
    XZ1_FOOT_GEOMS,
    XZ1_FOOT_SITES,
    XZ1_INIT_STATE,
    get_xz1_cfg,
)
from nazarite.mdp import curriculums as custom_curriculums
from nazarite.mdp import events as custom_events
from nazarite.mdp import observations as custom_observations
from nazarite.mdp import rewards as custom_rewards
from nazarite.mdp import terminations as custom_terminations
from nazarite.mdp.commands import TerrainConditionedVelocityCommandCfg
from nazarite.mdp.him_numerics import configure_him_observations

from .him_terrain import BoxHighWallTerrainCfg, BoxTrapezoidRampTerrainCfg
from .terminal_observations import configure_estimator_observations

# XZ1-local HIM knobs.  Keep these here instead of mutating the Go2 task.
XZ1_HIM_NUM_ENVS = 800
# Matches the reference Go2 HIM runs (2026-09-28_20-59-53 / 2026-09-29_19-00-44).
XZ1_HIM_EPISODE_LENGTH_S = 20.0
XZ1_HIM_GAIT_PERIOD = 0.6
# MJCF foot order is FL, FR, RR, RL; diagonal trot pairs are FL+RR / FR+RL.
XZ1_HIM_GAIT_OFFSETS = (0.0, 0.5, 0.0, 0.5)
# A 0.25 m foot lift is more than half the standing height, so a swing needs
# roughly 0.3 s of the 0.6 s period (up, across, down).  At 0.56 the reward
# expected the foot back on the ground after 0.264 s, which a 0.25 m lift cannot
# meet; 0.50 gives the swing half the period.
XZ1_HIM_GAIT_STANCE_FRACTION = 0.50
XZ1_HIM_GAIT_COMMAND_THRESHOLD = 0.1
# Reference Go2 HIM physics: 2 ms MuJoCo steps, ten substeps per 20 ms
# control step.  Keeping the substep count at the Go2 value makes the
# actuator/observation latency ranges below equal in milliseconds, not only
# in control steps.
XZ1_HIM_PHYSICS_DT = 0.002
XZ1_HIM_DECIMATION = 10
# A leg brush against a stair or tire during one or two substeps is normal
# while climbing.  Charge only contacts that survive two whole control steps
# (40 ms), which is long enough to exclude stubbing and short enough to catch
# kneeling on a tread or dragging a shank through a wall.
XZ1_HIM_LEG_CONTACT_SUSTAIN_CONTROL_STEPS = 2
XZ1_HIM_LEG_CONTACT_FORCE_THRESHOLD = 1.0
XZ1_HIM_LEG_CONTACT_HISTORY = (
    XZ1_HIM_DECIMATION * XZ1_HIM_LEG_CONTACT_SUSTAIN_CONTROL_STEPS
)
# Reference Go2 HIM sensor latency, in control steps: up to two steps (40 ms)
# with a 30% chance of holding the previous lag, resampled every ten steps.
# A trunk that rests on the terrain means the robot has fallen onto its belly
# and is dragging itself forward, which the 70 degree orientation limit does not
# catch.  Five newtons is a real weight transfer rather than a scrape; holding
# it for a quarter of a control step (50 ms = 25 substeps) rejects single
# substep penetration spikes and CCD blips instead of ending episodes early.
XZ1_HIM_BASE_CONTACT_FORCE_THRESHOLD = 5.0
XZ1_HIM_BASE_CONTACT_HOLD_S = 0.05
XZ1_HIM_BASE_CONTACT_HISTORY = max(
    1, round(XZ1_HIM_BASE_CONTACT_HOLD_S / XZ1_HIM_PHYSICS_DT)
)
# A belly slide drags the trunk across box edges, so the load chatters.  Count
# loaded substeps inside the 50 ms history instead of requiring one unbroken
# run: ten substeps is 20 ms of trunk load.
XZ1_HIM_BASE_CONTACT_LOAD_S = 0.02
XZ1_HIM_BASE_CONTACT_LOAD_STEPS = max(
    1, round(XZ1_HIM_BASE_CONTACT_LOAD_S / XZ1_HIM_PHYSICS_DT)
)

XZ1_HIM_OBS_DELAY_MAX_LAG = 2
XZ1_HIM_OBS_DELAY_HOLD_PROB = 0.3
XZ1_HIM_OBS_DELAY_UPDATE_PERIOD = 10
# XZ1_INIT_STATE is the physically loaded stance: 0.30 m base height, level
# trunk, every foot on the ground.  A MuJoCo position servo still needs a
# steady-state error to carry that load, so its *target* has to sit past the
# pose it holds.  With kp 32 (hip, thigh) / 35 (shank) the holding torque is
# 6.9 N m at most, split over the four legs as below: +0.018 rad at the thigh
# and +0.18-0.20 rad at the knee.  The hips take no entry: their axes are
# vertical, so their settled splay comes from the foot friction cone rather
# than from gravity, and targeting it would only creep.  Feeding target = init
# pose + these offsets is what makes the robot rest on XZ1_INIT_STATE, so every
# episode starts, and stands, at 0.300 m and level instead of sinking 7 cm into
# a 5.4 deg nose-up crouch.  Calibrated by iterating the servo fixed point in
# pure MuJoCo, spawning in the training initial pose (.scratch/xz1_pd).
XZ1_HIM_STANCE_PRECOMPENSATION = {
    "FL_hip_joint": 0.0,
    "FR_hip_joint": 0.0,
    "RR_hip_joint": 0.0,
    "RL_hip_joint": 0.0,
    "FL_thigh_joint": 0.0183,
    "FR_thigh_joint": 0.0181,
    "RR_thigh_joint": 0.0178,
    "RL_thigh_joint": 0.0180,
    "FL_shank_joint": 0.1764,
    "FR_shank_joint": 0.1780,
    "RR_shank_joint": 0.1977,
    "RL_shank_joint": 0.1954,
}
XZ1_HIM_JOINT_NAMES = tuple(
    f"{leg}_{part}_joint"
    for leg in ("FL", "FR", "RR", "RL")
    for part in ("hip", "thigh", "shank")
)


def _standing_servo_targets() -> dict[str, float]:
    """Absolute servo position targets that hold XZ1_INIT_STATE exactly."""
    init_joint_pos = XZ1_INIT_STATE.joint_pos
    if init_joint_pos is None:
        raise ValueError("XZ1_INIT_STATE.joint_pos must list the twelve leg joints.")
    _, names, base = resolve_matching_names_values(init_joint_pos, XZ1_HIM_JOINT_NAMES)
    return {
        name: value + XZ1_HIM_STANCE_PRECOMPENSATION[name]
        for name, value in zip(names, base, strict=True)
    }


XZ1_HIM_STANCE_TARGET = _standing_servo_targets()
# Training terrain set: flat, pyramid stairs, a discrete random-height grid, a
# solid trapezoidal ramp and the high wall. Difficulty climbs along ten rows, so
# the last row reaches the 300 mm wall (the competition rule height).
XZ1_HIM_TERRAIN_PATCH = (8.0, 8.0)
XZ1_HIM_TERRAIN_ROWS = 10
# Reference layout: five stair rings plus the central sixth level. Keep
# XZ1's tread width and height range, with flat space around the staircase.
XZ1_HIM_STAIR_LEVELS = 6
XZ1_HIM_STAIR_WIDTH = 0.35
XZ1_HIM_STAIR_PLATFORM_WIDTH = 2.5
XZ1_HIM_STAIR_BORDER_WIDTH = (
    XZ1_HIM_TERRAIN_PATCH[0] - XZ1_HIM_STAIR_PLATFORM_WIDTH
    - 2 * (XZ1_HIM_STAIR_LEVELS - 1) * XZ1_HIM_STAIR_WIDTH
) / 2
# Grid cells are 0.4 m squares whose height is sampled in [-bound, +bound]; the
# bound grows to 100 mm and similar neighbours are merged to keep the geom count
# low. The continuous slab below the cells turns the gaps into a walkable floor.
XZ1_HIM_GRID_CELL = 0.4
XZ1_HIM_GRID_MAX_HEIGHT = 0.10
XZ1_HIM_COMPLEX_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=XZ1_HIM_TERRAIN_PATCH,
    num_rows=XZ1_HIM_TERRAIN_ROWS,
    border_width=25.0,
    sub_terrains={
        "flat": terrain_gen.BoxFlatTerrainCfg(
            proportion=0.10,
            size=XZ1_HIM_TERRAIN_PATCH,
        ),
        "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
            proportion=0.30,
            size=XZ1_HIM_TERRAIN_PATCH,
            step_height_range=(0.03, 0.12),
            step_width=XZ1_HIM_STAIR_WIDTH,
            platform_width=XZ1_HIM_STAIR_PLATFORM_WIDTH - 1e-6,
            border_width=XZ1_HIM_STAIR_BORDER_WIDTH,
        ),
        "random_grid": terrain_gen.BoxRandomGridTerrainCfg(
            proportion=0.25,
            size=XZ1_HIM_TERRAIN_PATCH,
            grid_width=XZ1_HIM_GRID_CELL,
            grid_height_range=(0.0, XZ1_HIM_GRID_MAX_HEIGHT),
            platform_width=1.0,
            fill_gaps=True,
            merge_similar_heights=True,
            height_merge_threshold=0.05,
            max_merge_distance=3,
            border_width=0.25,
        ),
        "trapezoid_ramp": BoxTrapezoidRampTerrainCfg(
            proportion=0.20,
            size=XZ1_HIM_TERRAIN_PATCH,
            start_x=1.6,
            height_range=(0.06, 0.30),
            slope_angle_range=(10.0, 30.0),
            deck_length=0.9,
            spawn_x_fraction=0.075,
        ),
        "high_wall": BoxHighWallTerrainCfg(
            proportion=0.15,
            size=XZ1_HIM_TERRAIN_PATCH,
            wall_height_range=(0.05, 0.30),
            wall_thickness=0.08,
        ),
    },
)


def Nazarite_HIM_Complex_Terrain_XZ1(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the independent XZ1 HIM complex-terrain environment."""
    # Only geom group 0 is raycast: the generated terrain lives there, while the
    # robot's visual meshes are group 1 and its collision primitives group 3.
    # `exclude_parent_body` is not enough on its own -- MuJoCo excludes exactly
    # the frame's own body, not its subtree, so the legs and feet stay in range
    # of the rays unless the group filter keeps them out. Scanning the robot's
    # own meshes is both wrong (the heightmap sees the legs) and expensive.
    terrain_scan = RayCastSensorCfg(
        name="terrain_scan",
        frame=ObjRef(type="body", name=XZ1_BASE_BODY, entity="robot"),
        ray_alignment="yaw",
        pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
        max_distance=5.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
    )
    feet_height = TerrainHeightSensorCfg(
        name="feet_terrain_height",
        frame=tuple(
            ObjRef(type="site", name=name, entity="robot") for name in XZ1_FOOT_SITES
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
        primary=ContactMatch(mode="geom", pattern=XZ1_FOOT_GEOMS, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    leg_ground = ContactSensorCfg(
        name="leg_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(FL|FR|RR|RL)_(thigh|shank)_link_primitive_.*$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=XZ1_HIM_LEG_CONTACT_HISTORY,
    )
    rear_thigh_ground = ContactSensorCfg(
        name="rear_thigh_ground_contact",
        primary=ContactMatch(
            mode="body",
            pattern=("RR_thigh_link", "RL_thigh_link"),
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=1,
    )
    base_ground = ContactSensorCfg(
        name="base_ground_contact",
        primary=ContactMatch(mode="body", pattern=XZ1_BASE_BODY, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        history_length=XZ1_HIM_BASE_CONTACT_HISTORY,
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
    # Mirror the reference Go2 HIM sensor pipeline: only the four terms the
    # estimator must bridge carry latency, every environment samples its own
    # lag, and all environments are phase-staggered over one 20 ms control
    # step.  Lags are counted in control steps, so the range is 0-40 ms.
    for term_name in ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel"):
        actor_terms[term_name].delay_min_lag = 0
        actor_terms[term_name].delay_max_lag = XZ1_HIM_OBS_DELAY_MAX_LAG
        actor_terms[term_name].delay_hold_prob = XZ1_HIM_OBS_DELAY_HOLD_PROB
        actor_terms[term_name].delay_update_period = XZ1_HIM_OBS_DELAY_UPDATE_PERIOD
        actor_terms[term_name].delay_per_env = True
        actor_terms[term_name].delay_per_env_phase = True
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
            # The zero action must hold the standing pose, and holding it costs
            # a steady-state servo error, so the target is the pre-compensated
            # table rather than the bare init pose.
            use_default_offset=False,
            offset=XZ1_HIM_STANCE_TARGET,
        )
    }
    commands: dict[str, CommandTermCfg] = {
        "twist": TerrainConditionedVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(8.0, 12.0),
            rel_standing_envs=0.05,
            rel_forward_envs=0.1,
            # Heading mode replaces the constant yaw-rate sample with a
            # proportional law on a sampled heading target:
            #   wz_cmd = clip(0.5 * wrap_to_pi(heading_target - heading), +-1)
            # (mjlab velocity_command.py:115-125).  The linear ranges are
            # untouched, so the command space still covers backwards and
            # sideways motion; only the yaw reference changes from a rate to
            # an angle.  80% of the envs follow the law, which keeps a fifth
            # of the batch on plain yaw-rate tracking so the deployed policy
            # stays drivable by a rate command.
            #
            # Why: with an independent yaw rate the robot can turn and walk in
            # circles, so net displacement from the spawn point stops tracking
            # real progress (measured: 88% of straight-line vx=1 episodes
            # covered 15 m, but only 20% with the training sampler, and 33.7%
            # of the steps moved backwards).  The terrain curriculum is scored
            # on that displacement, so it was promotion by luck.  The reference
            # legged_gym curriculum avoids exactly this by deriving the yaw
            # rate from a heading error every step
            # (legged_robot.py:384-387); 0.5 is its control stiffness.
            rel_heading_envs=0.8,
            heading_command=True,
            heading_control_stiffness=0.5,
            debug_vis=True,
            # The wall spans the full patch width, so the policy has to face it
            # and push through instead of sidestepping. The ramp is symmetric in
            # x, so it needs no forward-only command constraint.
            forward_only_terrain_names=("high_wall",),
            forward_only_x_min=0.3,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0),
                lin_vel_y=(-1.0, 1.0),
                ang_vel_z=(-1.0, 1.0),
                heading=(-math.pi, math.pi),
            ),
        )
    }
    events = {
        "reset_base": EventTermCfg(
            func=custom_events.reset_root_state_uniform_facing_high_wall,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    # XZ1_INIT_STATE is the loaded stance, so this is a plain
                    # 1-5 cm terrain clearance lift on top of 0.30 m.
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
    # Reward set ports the reference Go2 HIM recipe used by
    # logs/rsl_rl/go2_him/2026-09-28_20-59-53 and 2026-09-29_19-00-44: same
    # nineteen terms in the same order with the same weights and parameters,
    # substituting only XZ1 bodies, sites, and geoms.  Two intentional
    # differences:
    #   * ``pose`` keeps the reference weight of exactly 0.0, so the term is
    #     declared for audit visibility but contributes no gradient.
    #   * ``leg_collision`` no longer reuses the ``illegal_contact``
    #     termination as a reward.  The reference value is charged when a
    #     thigh or shank stays on the terrain for two whole control steps, so
    #     unavoidable brushes while climbing are free and kneeling or dragging
    #     a shank is not.
    rewards = {
        "track_linear_velocity": RewardTermCfg(
            func=custom_rewards.him_track_linear_velocity,
            weight=1.5,
            params={
                "command_name": "twist",
                "std": math.sqrt(0.25),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "track_angular_velocity": RewardTermCfg(
            func=custom_rewards.him_track_angular_velocity,
            weight=0.8,
            params={
                "command_name": "twist",
                "std": math.sqrt(0.5),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        # The asset is symmetric (legs within 1 g, CoM 1.2 mm off centre, the
        # servo alone settles at +0.004 deg roll) yet the policy levies a load
        # dependent left-high bias: +1.5 deg standing, +4.3 deg at 0.8 m/s,
        # against +0.8 deg for the Go2 reference. At -0.2 this term is worth
        # ~1e-5 per step for a 4 deg lean, i.e. nothing, so the bias is free.
        "body_orientation_l2": RewardTermCfg(
            func=custom_rewards.body_orientation_l2,
            weight=-0.5,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=(XZ1_BASE_BODY,))},
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
        # These two scale with the square of the action scale, so they are only
        # meaningful relative to it. They were tripled (-0.03) to compensate a
        # thigh/shank action scale of 0.5 rad; that scale is now 0.25 rad like
        # the Go2 reference, so the reference weights apply unchanged.
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
            weight=0.1,
            params={
                "period": XZ1_HIM_GAIT_PERIOD,
                # MJCF foot order FL, FR, RR, RL; this reproduces the
                # reference diagonal pairs with XZ1's own ordering.
                "offset": list(XZ1_HIM_GAIT_OFFSETS),
                "threshold": XZ1_HIM_GAIT_STANCE_FRACTION,
                "command_threshold": XZ1_HIM_GAIT_COMMAND_THRESHOLD,
                "command_name": "twist",
                "sensor_name": "feet_ground_contact",
            },
        ),
        "foot_clearance": RewardTermCfg(
            func=velocity_mdp.feet_clearance,
            weight=-0.01,
            params={
                # Site height above the terrain. The 0.08 reference was tuned
                # for a 22 mm foot radius (58 mm of sole clearance); the sphere
                # is now 24 mm, so the same clearance needs 0.086.  Raised to
                # 0.12 because the measured swing apex is 0.13-0.16 m, i.e. the
                # leg can already lift more than the wall ever asked for, it
                # just was not asked to do it at the right moment.
                "target_height": 0.25,
                "height_sensor_name": "feet_terrain_height",
                "command_name": "twist",
                "command_threshold": XZ1_HIM_GAIT_COMMAND_THRESHOLD,
                "asset_cfg": SceneEntityCfg("robot", site_names=XZ1_FOOT_SITES),
            },
        ),
        "foot_slip": RewardTermCfg(
            func=velocity_mdp.feet_slip,
            weight=-0.05,
            params={
                "sensor_name": "feet_ground_contact",
                "command_name": "twist",
                "command_threshold": XZ1_HIM_GAIT_COMMAND_THRESHOLD,
                "asset_cfg": SceneEntityCfg("robot", site_names=XZ1_FOOT_SITES),
            },
        ),
        "soft_landing": RewardTermCfg(
            func=velocity_mdp.soft_landing,
            weight=-1e-3,
            params={
                "sensor_name": "feet_ground_contact",
                "command_name": "twist",
                "command_threshold": XZ1_HIM_GAIT_COMMAND_THRESHOLD,
            },
        ),
        "stand_still": RewardTermCfg(
            func=custom_rewards.stand_still,
            weight=-1.0,
            params={
                "command_name": "twist",
                "command_threshold": XZ1_HIM_GAIT_COMMAND_THRESHOLD,
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            },
        ),
        "leg_collision": RewardTermCfg(
            func=custom_rewards.sustained_leg_terrain_contact,
            weight=-1.0,
            params={
                "sensor_name": "leg_ground_contact",
                "force_threshold": XZ1_HIM_LEG_CONTACT_FORCE_THRESHOLD,
                "window_length": XZ1_HIM_LEG_CONTACT_HISTORY,
            },
        ),
        # Added on top of the reference set: the Gaussian tracking reward is
        # convex in the velocity error, so rocking back and forth through the
        # command pays more than tracking it steadily when the terrain prevents
        # steady motion.  Measured on the 1200 iteration XZ1 policy over
        # generated terrain: one environment moved backwards at up to 1.48 m/s
        # for 23.7% of the steps and still earned 0.462 tracking reward, while
        # steady motion at its mean speed (0.345 m/s) would have earned 0.180.
        # At weight -2.0 the measured retrograde motion costs 0.33 per step on
        # that terrain and nothing at all on flat ground, where the policy never
        # goes negative.
        "reverse_velocity": RewardTermCfg(
            func=custom_rewards.reverse_velocity_penalty,
            weight=-2.0,
            params={
                "command_name": "twist",
                "command_threshold": 0.1,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        # The three terms below are ported from the UW-HIMLoco dog stack
        # (legged_gym/envs/dog/dog_config.py:120-161).  Each one prices a
        # behaviour that the measured XZ1 policy exhibits but no existing term
        # constrains:
        #
        # * prolonged_air_time charges a foot that stays airborne past a whole
        #   gait period, i.e. hovering instead of stepping.  This is the direct
        #   counter to the stalled-at-obstacle pose: 72% of the steps at the
        #   wall were spent with both front feet in the air at 0.13 m while the
        #   trunk reared to +11 deg and the base stopped, and that pose still
        #   collected 0.98 of angular tracking reward per step.  The UW-HIMLoco
        #   counterpart is feet_air_time (+1.0), which pays at landing for a
        #   long swing; measured on the XZ1 gait (0.26 s swing) it contributed
        #   0.0005/step at their 0.5 s threshold and 0.0048/step at 0.2 s, and
        #   it would pay for the hover rather than charge for it.
        # * feet_stumble charges a contact whose horizontal force dominates the
        #   vertical one, i.e. a foot wedged against a step face.  Leg-terrain
        #   contact already covered 8.5% of the env-steps with peaks near 200 N.
        # * hip_mirror_symmetry charges left/right hip asymmetry only; the
        #   measured stance is 82% wider than the Go2 reference and the standing
        #   roll bias was +1.5 deg with DR disabled.
        "prolonged_air_time": RewardTermCfg(
            func=custom_rewards.prolonged_air_time,
            weight=-0.5,
            params={
                "sensor_name": "feet_ground_contact",
                # A nominal swing is (1 - 0.50) * 0.6 s = 0.30 s and a 0.25 m
                # lift is slower than that on a real step, so the cap is set at
                # two periods' worth of swing (0.60 s) before a foot that never
                # lands starts paying.
                "max_air_time": 0.60,
            },
        ),
        "feet_stumble": RewardTermCfg(
            func=custom_rewards.feet_stumble_reference,
            weight=-0.5,
            params={
                "sensor_name": "feet_ground_contact",
                "horizontal_ratio": 5.0,
            },
        ),
        "hip_mirror_symmetry": RewardTermCfg(
            func=custom_rewards.hip_mirror_symmetry,
            weight=-0.2,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
    }
    terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        # Belly on the terrain: the 70 degree orientation limit lets a robot
        # that has fallen forward keep crawling on its trunk, and that pose was
        # measured as the destination of the stalled-at-obstacle behaviour.
        "base_contact": TerminationTermCfg(
            func=custom_terminations.sustained_base_contact,
            params={
                "sensor_name": "base_ground_contact",
                "force_threshold": XZ1_HIM_BASE_CONTACT_FORCE_THRESHOLD,
                "min_loaded_steps": XZ1_HIM_BASE_CONTACT_LOAD_STEPS,
            },
        ),
        "fell_over": TerminationTermCfg(
            func=envs_mdp.bad_orientation, params={"limit_angle": math.radians(70.0)}
        ),
    }
    curriculum = {
        "terrain_levels": CurriculumTermCfg(
            # The native distance rule can require more distance than an 8 m
            # XZ1 tile provides at high commands.  Use the type-aware DELTA
            # curriculum with hysteresis and a capped success distance.
            #
            # The original 0.8/3.2 m rule promoted every env that walked a
            # few metres, so the mean level ran to 6.2 in 2.2k iterations
            # while the wall stayed at 0.22 m (0.14 successful episodes).
            # Three consecutive successes at 2.4-4.8 m then over-corrected:
            # the mean level sat at 2.33 for 3.7k iterations even though 88%
            # of straight-line driven episodes covered 15 m, because the test
            # reads the net displacement from the spawn point while the
            # sampled commands carry an independent yaw rate (33.7% of steps
            # move backwards), so frontier envs were demoted by luck.
            #
            # Two changes follow the legged_gym reference curriculum in
            # My_unitree_go2_gym-main (legged_robot.py:476-496): the success
            # distance is capped at the 4.0 m half tile, and promotion needs
            # two successes while demotion needs three failures, moving the
            # streak equilibrium from p^3=(1-p)^2 (p=0.57) to p^2=(1-p)^3
            # (p=0.43).  Capping the sparse terrains still keeps the wall and
            # the ramp inside what a 0.25 m foot lift can clear.
            #
            # The structural issue this does not fix is the criterion
            # itself: net displacement says little about progress when the
            # command turns.  The reference avoids that by deriving the yaw
            # rate from a heading error every step instead of sampling it
            # independently (legged_robot.py:384-387).
            func=custom_curriculums.terrain_levels_delta,
            params={
                "command_name": "twist",
                "success_distance_scale": 0.35,
                "min_success_distance": 2.4,
                "max_success_distance": 4.0,
                "success_streak_length": 2,
                "failure_streak_length": 3,
                "sparse_terrain_names": ("trapezoid_ramp", "high_wall"),
                "sparse_success_distance_scale": 0.5,
                "sparse_min_success_distance": 0.8,
                "sparse_max_success_distance": 2.0,
                "sparse_success_streak_length": 3,
                "sparse_failure_streak_length": 4,
                "sparse_max_level": 4,
                "flat_terrain_name": "flat",
                "high_wall_terrain_name": "high_wall",
                "high_wall_crossing_clearance": 0.4,
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
                        "lin_vel_y": (-1.0, 1.0),
                        "ang_vel_z": (-1.0, 1.0),
                    }
                ],
            },
        ),
    }
    # Actuator command latency follows the reference Go2 HIM run: at most nine
    # 2 ms physics steps (18 ms), held constant for one 20 ms control interval
    # and phase-staggered per environment, without mutating other XZ1 tasks.
    robot_cfg = deepcopy(get_xz1_cfg())
    assert robot_cfg.articulation is not None
    for actuator in robot_cfg.articulation.actuators:
        actuator.delay_min_lag = 0
        actuator.delay_max_lag = XZ1_HIM_DECIMATION - 1
        actuator.delay_hold_prob = 0.0
        actuator.delay_update_period = XZ1_HIM_DECIMATION
        actuator.delay_per_env_phase = True

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(
                terrain_type="generator",
                terrain_generator=deepcopy(XZ1_HIM_COMPLEX_TERRAINS_CFG),
                max_init_terrain_level=1,
            ),
            sensors=(
                terrain_scan,
                feet_height,
                root_angmom,
                feet_ground,
                leg_ground,
                rear_thigh_ground,
                base_ground,
            ),
            entities={"robot": robot_cfg},
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
        metrics={
            "mean_action_acc": MetricsTermCfg(func=envs_mdp.mean_action_acc),
            "rr_thigh_action_abs": MetricsTermCfg(
                func=custom_rewards.xz1_joint_action_abs,
                params={"asset_cfg": SceneEntityCfg("robot", joint_names=("RR_thigh_joint",))},
            ),
            "rl_thigh_action_abs": MetricsTermCfg(
                func=custom_rewards.xz1_joint_action_abs,
                params={"asset_cfg": SceneEntityCfg("robot", joint_names=("RL_thigh_joint",))},
            ),
            "rr_thigh_target_error": MetricsTermCfg(
                func=custom_rewards.xz1_joint_target_error,
                params={"asset_cfg": SceneEntityCfg("robot", joint_names=("RR_thigh_joint",))},
            ),
            "rl_thigh_target_error": MetricsTermCfg(
                func=custom_rewards.xz1_joint_target_error,
                params={"asset_cfg": SceneEntityCfg("robot", joint_names=("RL_thigh_joint",))},
            ),
            "rr_thigh_torque_ratio": MetricsTermCfg(
                func=custom_rewards.xz1_joint_torque_ratio,
                reduce="max",
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=("RR_thigh_joint",)),
                    "effort_limit": 31.7,
                },
            ),
            "rl_thigh_torque_ratio": MetricsTermCfg(
                func=custom_rewards.xz1_joint_torque_ratio,
                reduce="max",
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=("RL_thigh_joint",)),
                    "effort_limit": 31.7,
                },
            ),
            "rr_thigh_limit_proximity": MetricsTermCfg(
                func=custom_rewards.xz1_joint_limit_proximity,
                reduce="max",
                params={"asset_cfg": SceneEntityCfg("robot", joint_names=("RR_thigh_joint",))},
            ),
            "rl_thigh_limit_proximity": MetricsTermCfg(
                func=custom_rewards.xz1_joint_limit_proximity,
                reduce="max",
                params={"asset_cfg": SceneEntityCfg("robot", joint_names=("RL_thigh_joint",))},
            ),
            "rear_thigh_ground_contact": MetricsTermCfg(
                func=velocity_mdp.illegal_contact,
                params={
                    "sensor_name": "rear_thigh_ground_contact",
                    "force_threshold": 1.0,
                },
            ),
        },
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name=XZ1_BASE_BODY,
            distance=3.0,
            elevation=-5.0,
            azimuth=90.0,
        ),
        # Solver and contact budgets follow the reference Go2 HIM run
        # (logs/rsl_rl/go2_him/2026-09-28_20-59-53): Newton with a 1e-8
        # tolerance and 50 CCD iterations, 256 contacts, 1500 constraint rows and
        # 500 contact-sensor matches. The HIM contact sensors track six contact
        # pairs with history, so the smaller 64/500/128 budget of the earlier
        # XZ1 task was too tight.
        sim=SimulationCfg(
            nconmax=256,
            njmax=1500,
            contact_sensor_maxmatch=500,
            mujoco=MujocoCfg(
                timestep=XZ1_HIM_PHYSICS_DT,
                solver="newton",
                iterations=10,
                tolerance=1e-8,
                ls_iterations=20,
                ls_tolerance=0.01,
                ccd_iterations=50,
            ),
        ),
        decimation=XZ1_HIM_DECIMATION,
        episode_length_s=XZ1_HIM_EPISODE_LENGTH_S,
        auto_reset=True,
    )
    configure_estimator_observations(cfg, play=play)
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
