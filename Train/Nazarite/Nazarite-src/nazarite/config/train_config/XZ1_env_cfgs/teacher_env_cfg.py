"""Independent XZ1 elevation teacher environment configuration.

Edit terrain, commands, rewards, observations, randomization and physics here.
Only robot assets and MDP implementations are shared with the foundation task.
"""

import math
from copy import deepcopy
from dataclasses import dataclass

import torch

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
from mjlab.managers.recorder_manager import RecorderTermCfg
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
from nazarite.mdp import observations as custom_observations
from nazarite.mdp import rewards as custom_rewards
from nazarite.mdp import terminations as custom_terminations
from nazarite.mdp.him_numerics import configure_him_observations
from nazarite.mdp.xz1_diagnostics import XZ1CourseDiagnostics

from . import him_course
from .him_obstacles import (
    XZ1GapTerrainCfg,
    XZ1LowFrameTerrainCfg,
    XZ1PlumBlossomTerrainCfg,
    XZ1SingleBeamTerrainCfg,
    XZ1TireArrayTerrainCfg,
)
from .him_terrain import (
    BoxHighWallTerrainCfg,
    BoxNarrowStairsTerrainCfg,
    BoxTrapezoidRampTerrainCfg,
)
from .terminal_observations import configure_estimator_observations

# Teacher-owned environment parameters; changes do not modify the base HIM task.
XZ1_TEACHER_SCAN_HISTORY_LENGTH = 1  # Latest scan only; no terrain history stack.
XZ1_TEACHER_SCAN_UPDATE_PERIOD = 0.1  # Seconds; independent of 50 Hz control/HIM.
XZ1_TEACHER_NUM_ENVS = 800
# Matches the reference Go2 HIM runs (2026-09-28_20-59-53 / 2026-09-29_19-00-44).
XZ1_TEACHER_EPISODE_LENGTH_S = 20.0
XZ1_TEACHER_GAIT_PERIOD = 0.6
# MJCF foot order is FL, FR, RR, RL; diagonal trot pairs are FL+RR / FR+RL.
XZ1_TEACHER_GAIT_OFFSETS = (0.0, 0.5, 0.0, 0.5)
# A 0.25 m foot lift is more than half the standing height, so a swing needs
# roughly 0.3 s of the 0.6 s period (up, across, down).  At 0.56 the reward
# expected the foot back on the ground after 0.264 s, which a 0.25 m lift cannot
# meet; 0.50 gives the swing half the period.
XZ1_TEACHER_GAIT_STANCE_FRACTION = 0.50
XZ1_TEACHER_GAIT_COMMAND_THRESHOLD = 0.1
# Reference Go2 HIM physics: 2 ms MuJoCo steps, ten substeps per 20 ms
# control step.  Keeping the substep count at the Go2 value makes the
# actuator/observation latency ranges below equal in milliseconds, not only
# in control steps.
XZ1_TEACHER_PHYSICS_DT = 0.002
XZ1_TEACHER_DECIMATION = 10
# A leg brush against a stair or tire during one or two substeps is normal
# while climbing.  Charge only contacts that survive two whole control steps
# (40 ms), which is long enough to exclude stubbing and short enough to catch
# kneeling on a tread or dragging a shank through a wall.
XZ1_TEACHER_LEG_CONTACT_SUSTAIN_CONTROL_STEPS = 2
XZ1_TEACHER_LEG_CONTACT_FORCE_THRESHOLD = 1.0
XZ1_TEACHER_LEG_CONTACT_HISTORY = (
    XZ1_TEACHER_DECIMATION * XZ1_TEACHER_LEG_CONTACT_SUSTAIN_CONTROL_STEPS
)
# Reference Go2 HIM sensor latency, in control steps: up to two steps (40 ms)
# with a 30% chance of holding the previous lag, resampled every ten steps.
# A trunk that rests on the terrain means the robot has fallen onto its belly
# and is dragging itself forward, which the 70 degree orientation limit does not
# catch. Count predominantly vertical support, not horizontal wall impacts.
XZ1_TEACHER_BASE_CONTACT_FORCE_THRESHOLD = 5.0
XZ1_TEACHER_BASE_CONTACT_MIN_VERTICAL_FORCE_RATIO = 0.7
XZ1_TEACHER_BASE_CONTACT_HOLD_S = 0.20
XZ1_TEACHER_BASE_CONTACT_HISTORY = max(
    1, round(XZ1_TEACHER_BASE_CONTACT_HOLD_S / XZ1_TEACHER_PHYSICS_DT)
)
# A belly slide drags the trunk across box edges, so the load chatters.  Count
# loaded substeps inside the 200 ms history instead of requiring one unbroken
# run: 75 substeps is 150 ms of trunk support, allowing short obstacle scrapes.
XZ1_TEACHER_BASE_CONTACT_LOAD_S = 0.15
XZ1_TEACHER_BASE_CONTACT_LOAD_STEPS = max(
    1, round(XZ1_TEACHER_BASE_CONTACT_LOAD_S / XZ1_TEACHER_PHYSICS_DT)
)

XZ1_TEACHER_OBS_DELAY_MAX_LAG = 2
XZ1_TEACHER_OBS_DELAY_HOLD_PROB = 0.3
XZ1_TEACHER_OBS_DELAY_UPDATE_PERIOD = 10
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
XZ1_TEACHER_STANCE_PRECOMPENSATION = {
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
XZ1_TEACHER_JOINT_NAMES = tuple(
    f"{leg}_{part}_joint"
    for leg in ("FL", "FR", "RR", "RL")
    for part in ("hip", "thigh", "shank")
)


def _standing_servo_targets() -> dict[str, float]:
    """Absolute servo position targets that hold XZ1_INIT_STATE exactly."""
    init_joint_pos = XZ1_INIT_STATE.joint_pos
    if init_joint_pos is None:
        raise ValueError("XZ1_INIT_STATE.joint_pos must list the twelve leg joints.")
    _, names, base = resolve_matching_names_values(
        init_joint_pos, XZ1_TEACHER_JOINT_NAMES
    )
    return {
        name: value + XZ1_TEACHER_STANCE_PRECOMPENSATION[name]
        for name, value in zip(names, base, strict=True)
    }


XZ1_TEACHER_STANCE_TARGET = _standing_servo_targets()
# Nine obstacle terrains owned by this teacher task.
# Difficulty climbs along ten rows; the last row reaches the 300 mm wall.
XZ1_TEACHER_TERRAIN_PATCH = (8.0, 8.0)
XZ1_TEACHER_TERRAIN_ROWS = 10
# Keep competition-width exposure even after narrow terrain becomes easy.
XZ1_TEACHER_NARROW_WIDTH_RANGE = (0.8, 0.8)
XZ1_TEACHER_NARROW_SAMPLING_MINIMUMS = {"narrow_stairs": 0.15, "narrow_ramp": 0.10}
XZ1_TEACHER_HIP_OUTWARD_ALLOWANCE = 0.15
XZ1_TEACHER_HIP_INWARD_ALLOWANCE = 0.15
XZ1_TEACHER_HIP_SOFT_GAIN = 6.0
XZ1_TEACHER_HIP_BASE_GAIN = 0.25
XZ1_TEACHER_SUPPORT_EDGE_MARGIN = 0.03
XZ1_TEACHER_SUPPORT_EDGE_WEIGHT = -0.20
XZ1_TEACHER_ACTION_RATE_WEIGHT = -0.015
XZ1_TEACHER_POSE_WEIGHT = 0.20
# Pose is measured from XZ1_INIT_STATE, independently of servo precompensation.
XZ1_TEACHER_POSE_STD_STANDING = {
    ".*_hip_joint": 0.10,
    ".*_thigh_joint": 0.20,
    ".*_shank_joint": 0.30,
}
XZ1_TEACHER_POSE_STD_WALKING = {
    ".*_hip_joint": 0.20,
    ".*_thigh_joint": 0.60,
    ".*_shank_joint": 0.80,
}
XZ1_TEACHER_POSE_STD_RUNNING = {
    ".*_hip_joint": 0.25,
    ".*_thigh_joint": 0.70,
    ".*_shank_joint": 0.90,
}
XZ1_TEACHER_POSE_TERRAIN_TOLERANCE = {
    ".*_hip_joint": 0.04,
    ".*_thigh_joint": 0.20,
    ".*_shank_joint": 0.30,
}
XZ1_TEACHER_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=XZ1_TEACHER_TERRAIN_PATCH,
    num_rows=XZ1_TEACHER_TERRAIN_ROWS,
    border_width=25.0,
    sub_terrains={
        "flat": terrain_gen.BoxFlatTerrainCfg(
            proportion=0.10,
            size=XZ1_TEACHER_TERRAIN_PATCH,
        ),
        "high_wall": BoxHighWallTerrainCfg(
            proportion=0.10,
            size=XZ1_TEACHER_TERRAIN_PATCH,
            wall_height_range=(0.05, 0.30),
            wall_thickness=0.08,
            spawn_x_fraction=0.25,
        ),
        "narrow_stairs": BoxNarrowStairsTerrainCfg(
            proportion=0.15,
            size=XZ1_TEACHER_TERRAIN_PATCH,
            width_range=XZ1_TEACHER_NARROW_WIDTH_RANGE,
            start_x=1.6,
            step_height_range=(0.03, 0.12),
            step_depth=0.32,
            num_steps=3,
            deck_length=0.8,
            spawn_x_fraction=0.075,
        ),
        "narrow_ramp": BoxTrapezoidRampTerrainCfg(
            proportion=0.10,
            size=XZ1_TEACHER_TERRAIN_PATCH,
            width_range=XZ1_TEACHER_NARROW_WIDTH_RANGE,
            height_range=(0.06, 0.36),
            slope_angle_range=(10.0, 20.0),
            deck_length=0.8,
            start_x=1.6,
            spawn_x_fraction=0.075,
        ),
        "plum_blossom": XZ1PlumBlossomTerrainCfg(
            proportion=0.15,
            size=XZ1_TEACHER_TERRAIN_PATCH,
            spawn_x_fraction=0.125,
            start_x=1.4,
            stone_size_range=(0.50, 0.20),
            height_range=(0.05, 0.20),
            row_spacing=0.40,
            stone_pitch=0.60,
            row_stagger=0.30,
            platform_length=0.80,
            width=0.80,
        ),
        "single_beam": XZ1SingleBeamTerrainCfg(
            proportion=0.15,
            size=XZ1_TEACHER_TERRAIN_PATCH,
            spawn_x_fraction=0.125,
            start_x=1.6,
            length=3.0,
            width_range=(0.80, 0.40),
            height_range=(0.05, 0.10),
            platform_width=0.80,
        ),
        "low_frame": XZ1LowFrameTerrainCfg(
            proportion=0.10,
            size=XZ1_TEACHER_TERRAIN_PATCH,
            spawn_x_fraction=0.125,
            start_x=2.0,
            length=0.80,
            width=0.80,
            clearance_range=(0.60, 0.29),
            beam_thickness=0.04,
        ),
        "tire_array": XZ1TireArrayTerrainCfg(
            proportion=0.10,
            size=XZ1_TEACHER_TERRAIN_PATCH,
            spawn_x_fraction=0.125,
            start_x=1.8,
            outer_diameter=0.55,
            inner_diameter=0.33,
            height_range=(0.03, 0.165),
            pitch=0.56,
            segments=12,
        ),
        "gap": XZ1GapTerrainCfg(
            proportion=0.05,
            size=XZ1_TEACHER_TERRAIN_PATCH,
            spawn_x_fraction=0.125,
            start_x=1.8,
            gap_width_range=(0.02, 0.36),
            pit_depth_range=(0.15, 0.36),
            num_gaps=3,
            landing_length=0.80,
        ),
    },
)


SCAN_X = (-0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2)
SCAN_Y = (-0.75, -0.6, -0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.45, 0.6, 0.75)


@dataclass
class ParkourScanPatternCfg(GridPatternCfg):
    """Reference asymmetric 12x11 grid, x-major (not mjlab's xy ordering)."""

    def generate_rays(self, mj_model, device):
        del mj_model
        x = torch.tensor(SCAN_X, device=device)
        y = torch.tensor(SCAN_Y, device=device)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        offsets = torch.zeros(132, 3, device=device)
        offsets[:, 0], offsets[:, 1] = xx.flatten(), yy.flatten()
        # Cast above the tallest course surface, not from inside a wall whose
        # top may be higher than the trunk. Height output still uses frame Z.
        offsets[:, 2] = 1.0
        directions = torch.zeros_like(offsets)
        directions[:, 2] = -1.0
        return offsets, directions


def parkour_height_scan(env):
    """Reference clip(base_z - 0.3 - terrain_z, -1, 1), no renormalizing."""
    return envs_mdp.height_scan(
        env, sensor_name="teacher_terrain_scan", offset=0.3, miss_value=1.0
    ).clamp(-1.0, 1.0)


class SampledParkourHeightScan:
    """Hold the entire scan history between acquisitions, including base Z.

    Store raw scans so PPO still recomputes all embeddings with its current
    encoder. The manager must not add its own control-rate history on top.
    """

    def __init__(self, cfg, env):
        self.history_length = cfg.params["history_length"]
        self.update_steps = _scan_update_steps(cfg.params["update_period"], env.step_dt)
        self.history = torch.zeros(
            env.num_envs, self.history_length, 132, device=env.device
        )
        self.last_sample_step = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )

    def reset(self, env_ids=None):
        self.last_sample_step[slice(None) if env_ids is None else env_ids] = -1

    def __call__(self, env, history_length, update_period):
        del history_length, update_period
        step = env.common_step_counter
        fresh = self.last_sample_step < 0
        due = fresh | (step - self.last_sample_step >= self.update_steps)
        ids = due.nonzero(as_tuple=False).flatten()
        if ids.numel():
            scan = parkour_height_scan(env)
            # Clone before the shift: advanced indexing and overlapping views
            # must not overwrite older frames or another environment's state.
            self.history[ids, :-1] = self.history[ids, 1:].clone()
            self.history[ids, -1] = scan[ids]
            first = fresh.nonzero(as_tuple=False).flatten()
            self.history[first] = scan[first, None].expand(-1, self.history_length, -1)
            self.last_sample_step[ids] = step
        return self.history[:, 0] if self.history_length == 1 else self.history


def _scan_update_steps(update_period: float, step_dt: float) -> int:
    steps = round(update_period / step_dt)
    if steps < 1 or not math.isclose(steps * step_dt, update_period, abs_tol=1e-9):
        raise ValueError("Scan update period must be a positive multiple of control dt")
    return steps


def Nazarite_WMP_HIM_Teacher_XZ1(
    play: bool = False,
    *,
    scan_history_length: int = 1,
    scan_update_period: float = 0.02,
):
    """Add an oracle actor input without changing XZ1 HIM or its 244-D critic."""
    if scan_history_length < 1:
        raise ValueError("scan_history_length must be positive")
    cfg = _build_xz1_teacher_env(play)
    _scan_update_steps(scan_update_period, cfg.sim.mujoco.timestep * cfg.decimation)
    cfg.scene.sensors = (
        *cfg.scene.sensors,
        RayCastSensorCfg(
            name="teacher_terrain_scan",
            frame=ObjRef(type="body", name=XZ1_BASE_BODY, entity="robot"),
            pattern=ParkourScanPatternCfg(),
            ray_alignment="yaw",
            max_distance=5.0,
            include_geom_groups=(0,),
            exclude_parent_body=True,
        ),
    )
    cfg.observations["teacher_scan"] = ObservationGroupCfg(
        terms={"height_scan": ObservationTermCfg(func=parkour_height_scan)},
        concatenate_terms=True,
        enable_corruption=False,
        history_length=scan_history_length if scan_history_length > 1 else None,
        flatten_history_dim=False,
    )
    if scan_update_period != 0.02:
        cfg.observations["teacher_scan"] = ObservationGroupCfg(
            terms={
                "height_scan": ObservationTermCfg(
                    func=SampledParkourHeightScan,
                    params={
                        "history_length": scan_history_length,
                        "update_period": scan_update_period,
                    },
                )
            },
            concatenate_terms=True,
            enable_corruption=False,
        )
    if not play:
        # After terminal estimator capture, before curriculum/reset overwrite.
        cfg.recorders["course_diagnostics"] = RecorderTermCfg(func=XZ1CourseDiagnostics)
    return cfg


def Nazarite_WMP_HIM_Eval_XZ1(
    play: bool = False,
    *,
    terrain: str = "narrow_stairs",
    difficulty: float = 0.0,
    speed: float = 0.4,
    seed: int = 42,
    reset_x: float = 0.0,
    reset_y: float = 0.0,
    reset_yaw: float = 0.0,
    command_mode: str = "velocity",
    scan_history_length: int = 1,
    scan_update_period: float = 0.02,
):
    """Fixed nominal benchmark, with no curriculum or oracle command steering.

    The evaluator, not training rewards/terrain levels, counts completed
    crossings. This task intentionally disables noise, delays and randomization.
    """
    if not 0.0 <= difficulty <= 1.0 or not 0.0 < speed <= 1.0:
        raise ValueError("Evaluation requires difficulty in [0,1], speed in (0,1]")
    if min(reset_x, reset_y, reset_yaw) < 0:
        raise ValueError("Reset jitter must be nonnegative")
    if command_mode not in ("velocity", "course"):
        raise ValueError("command_mode must be velocity or course")
    cfg = Nazarite_WMP_HIM_Teacher_XZ1(
        scan_history_length=scan_history_length, scan_update_period=scan_update_period
    )
    assert cfg.scene.terrain is not None
    generator = cfg.scene.terrain.terrain_generator
    assert generator is not None
    if terrain not in generator.sub_terrains:
        raise ValueError(f"Unknown XZ1 terrain: {terrain}")
    generator.sub_terrains = {terrain: generator.sub_terrains[terrain]}
    generator.sub_terrains[terrain].proportion = 1.0
    generator.num_rows = generator.num_cols = 1
    generator.curriculum = True
    generator.difficulty_range = (difficulty, difficulty)
    generator.seed = seed
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.scene.num_envs = 1
    cfg.seed = seed
    cfg.auto_reset = play
    cfg.episode_length_s = 20.0
    cfg.curriculum = {}
    cfg.commands["twist"] = UniformVelocityCommandCfg(
        entity_name="robot",
        resampling_time_range=(1000.0, 1000.0),
        heading_command=False,
        rel_heading_envs=0.0,
        rel_standing_envs=0.0,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(speed, speed),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
    )
    if command_mode == "course" and terrain in him_course.COURSE_TERRAIN_NAMES:
        cfg.commands["twist"] = him_course.XZ1CourseVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(1000.0, 1000.0),
            heading_command=False,
            rel_heading_envs=0.0,
            rel_standing_envs=0.0,
            narrow_speed_range=(speed, speed),
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(speed, speed),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
            ),
        )
    cfg.events = {
        name: cfg.events[name] for name in ("reset_base", "reset_robot_joints")
    }
    cfg.events["reset_base"].func = envs_mdp.reset_root_state_uniform
    cfg.events["reset_base"].params = {
        "pose_range": {
            "x": (-reset_x, reset_x),
            "y": (-reset_y, reset_y),
            "z": (0.01, 0.01),
            "yaw": (-reset_yaw, reset_yaw),
        },
        "velocity_range": {},
    }
    cfg.terminations.pop("narrow_course_complete")
    for group in cfg.observations.values():
        group.enable_corruption = False
        for term in group.terms.values():
            term.delay_min_lag = term.delay_max_lag = 0
            term.delay_hold_prob = 0.0
            term.delay_update_period = 0
    articulation = cfg.scene.entities["robot"].articulation
    assert articulation is not None
    for actuator in articulation.actuators:
        actuator.delay_min_lag = actuator.delay_max_lag = 0
        actuator.delay_hold_prob = 0.0
        actuator.delay_update_period = 0
    return cfg


def _build_xz1_teacher_env(play: bool) -> ManagerBasedRlEnvCfg:
    """Construct teacher settings without calling the foundation HIM factory."""
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
        preserve_order=True,
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
        history_length=XZ1_TEACHER_LEG_CONTACT_HISTORY,
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
        history_length=XZ1_TEACHER_BASE_CONTACT_HISTORY,
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
            params={"period": XZ1_TEACHER_GAIT_PERIOD, "command_name": "twist"},
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
        actor_terms[term_name].delay_max_lag = XZ1_TEACHER_OBS_DELAY_MAX_LAG
        actor_terms[term_name].delay_hold_prob = XZ1_TEACHER_OBS_DELAY_HOLD_PROB
        actor_terms[term_name].delay_update_period = XZ1_TEACHER_OBS_DELAY_UPDATE_PERIOD
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
            offset=XZ1_TEACHER_STANCE_TARGET,
        )
    }
    commands: dict[str, CommandTermCfg] = {
        "twist": him_course.XZ1CourseVelocityCommandCfg(
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
            rel_heading_envs=0.5,
            heading_command=True,
            heading_control_stiffness=0.5,
            debug_vis=True,
            # Wide terrains retain their command distribution. Narrow courses
            # use +x travel with bounded heading/centerline corrections.
            forward_only_terrain_names=("high_wall",),
            forward_only_x_min=0.3,
            narrow_speed_range=(0.3, 0.7),
            lateral_gain=0.8,
            lateral_limit=0.15,
            yaw_gain=0.8,
            yaw_limit=0.3,
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
            func=him_course.reset_xz1_course,
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
    # XZ1-specific tuning built on the reference HIM recipe. Pose tracking is
    # hip-focused, relaxes on rough terrain, and is disabled on high walls.
    # Leg collision charges sustained terrain contact over two control steps;
    # transient brushes while climbing remain free.
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
            func=him_course.XZ1TerrainPosture,
            weight=XZ1_TEACHER_POSE_WEIGHT,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "command_name": "twist",
                "std_standing": XZ1_TEACHER_POSE_STD_STANDING.copy(),
                "std_walking": XZ1_TEACHER_POSE_STD_WALKING.copy(),
                "std_running": XZ1_TEACHER_POSE_STD_RUNNING.copy(),
                "terrain_tolerance": XZ1_TEACHER_POSE_TERRAIN_TOLERANCE.copy(),
                "walking_threshold": 0.1,
                "running_threshold": 1.5,
                "scan_sensor_name": "terrain_scan",
                "terrain_radius": 0.35,
                "terrain_height_std": (0.01, 0.08),
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
        # Penalize raw action differences with the actuator action scale held
        # at 0.25. Increase first-order smoothing mildly; retain the existing
        # second-order weight to avoid changing both penalties at once.
        "action_rate_l2": RewardTermCfg(
            func=envs_mdp.action_rate_l2,
            weight=XZ1_TEACHER_ACTION_RATE_WEIGHT,
        ),
        "smoothness": RewardTermCfg(func=envs_mdp.action_acc_l2, weight=-0.02),
        "joint_torques_l2": RewardTermCfg(
            func=envs_mdp.joint_torques_l2, weight=-2.0e-5
        ),
        "hip_pos": RewardTermCfg(
            func=him_course.xz1_hip_position_cost,
            weight=-0.05,
            params={
                "command_name": "twist",
                "outward_allowance": XZ1_TEACHER_HIP_OUTWARD_ALLOWANCE,
                "inward_allowance": XZ1_TEACHER_HIP_INWARD_ALLOWANCE,
                "soft_gain": XZ1_TEACHER_HIP_SOFT_GAIN,
                "base_gain": XZ1_TEACHER_HIP_BASE_GAIN,
                "lateral_full": 0.2,
                "lateral_free": 0.6,
                "yaw_full": 0.2,
                "yaw_free": 0.6,
            },
        ),
        "narrow_support_edge": RewardTermCfg(
            func=him_course.narrow_support_edge_cost,
            weight=XZ1_TEACHER_SUPPORT_EDGE_WEIGHT,
            params={
                "sensor_name": "feet_ground_contact",
                "asset_cfg": SceneEntityCfg(
                    "robot", site_names=XZ1_FOOT_SITES, preserve_order=True
                ),
                "edge_margin": XZ1_TEACHER_SUPPORT_EDGE_MARGIN,
                "foot_radius": 0.024,
                "force_threshold": 5.0,
                "distance_scale": 0.10,
            },
        ),
        "foot_gait": RewardTermCfg(
            func=custom_rewards.feet_gait,
            weight=0.1,
            params={
                "period": XZ1_TEACHER_GAIT_PERIOD,
                # MJCF foot order FL, FR, RR, RL; this reproduces the
                # reference diagonal pairs with XZ1's own ordering.
                "offset": list(XZ1_TEACHER_GAIT_OFFSETS),
                "threshold": XZ1_TEACHER_GAIT_STANCE_FRACTION,
                "command_threshold": XZ1_TEACHER_GAIT_COMMAND_THRESHOLD,
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
                "command_threshold": XZ1_TEACHER_GAIT_COMMAND_THRESHOLD,
                "asset_cfg": SceneEntityCfg("robot", site_names=XZ1_FOOT_SITES),
            },
        ),
        "foot_slip": RewardTermCfg(
            func=velocity_mdp.feet_slip,
            weight=-0.05,
            params={
                "sensor_name": "feet_ground_contact",
                "command_name": "twist",
                "command_threshold": XZ1_TEACHER_GAIT_COMMAND_THRESHOLD,
                "asset_cfg": SceneEntityCfg("robot", site_names=XZ1_FOOT_SITES),
            },
        ),
        "soft_landing": RewardTermCfg(
            func=velocity_mdp.soft_landing,
            weight=-1e-3,
            params={
                "sensor_name": "feet_ground_contact",
                "command_name": "twist",
                "command_threshold": XZ1_TEACHER_GAIT_COMMAND_THRESHOLD,
            },
        ),
        "stand_still": RewardTermCfg(
            func=custom_rewards.stand_still,
            weight=-1.0,
            params={
                "command_name": "twist",
                "command_threshold": XZ1_TEACHER_GAIT_COMMAND_THRESHOLD,
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            },
        ),
        "leg_collision": RewardTermCfg(
            func=custom_rewards.sustained_leg_terrain_contact,
            weight=-1.0,
            params={
                "sensor_name": "leg_ground_contact",
                "force_threshold": XZ1_TEACHER_LEG_CONTACT_FORCE_THRESHOLD,
                "window_length": XZ1_TEACHER_LEG_CONTACT_HISTORY,
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
            weight=-1.0,
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
                "force_threshold": XZ1_TEACHER_BASE_CONTACT_FORCE_THRESHOLD,
                "min_loaded_steps": XZ1_TEACHER_BASE_CONTACT_LOAD_STEPS,
                "min_vertical_force_ratio": (
                    XZ1_TEACHER_BASE_CONTACT_MIN_VERTICAL_FORCE_RATIO
                ),
            },
        ),
        "fell_over": TerminationTermCfg(
            func=envs_mdp.bad_orientation, params={"limit_angle": math.radians(70.0)}
        ),
        "left_narrow_course": TerminationTermCfg(func=him_course.left_narrow_course),
        "obstacle_ground_contact": TerminationTermCfg(
            func=him_course.ObstacleGroundContact,
        ),
        # A completed crossing is a successful timeout, not a fall penalty.
        "narrow_course_complete": TerminationTermCfg(
            func=him_course.narrow_course_complete,
            time_out=True,
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
            # (p=0.43). Sparse terrains can progress to the last row: a 0.30 m
            # wall and a competition-profile ramp (0.36 m high, 20 degrees).
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
                "sparse_terrain_names": ("high_wall", *him_course.COURSE_TERRAIN_NAMES),
                "sparse_success_distance_scale": 0.5,
                "sparse_min_success_distance": 0.8,
                "sparse_max_success_distance": 2.0,
                "sparse_success_streak_length": 3,
                "sparse_failure_streak_length": 4,
                "sparse_max_level": XZ1_TEACHER_TERRAIN_ROWS - 1,
                "flat_terrain_name": "flat",
                "high_wall_terrain_name": "high_wall",
                "high_wall_crossing_clearance": 0.4,
                # The shared legacy branch describes the old half-tile tire
                # lattice. The XZ1 course filter below uses the actual 2x4 array.
                "tire_array_terrain_name": "",
                # WMP courses follow reference fixed terrain columns. An easy
                # course must never lend its row or streak to a different skill.
                "adaptive_type_sampling": False,
                "randomize_completed_levels": True,
                "adaptive_sampling_floor": 0.25,
                "adaptive_sampling_temperature": 1.0,
                "adaptive_warmup_episodes": 20,
                "type_success_ema_alpha": 0.05,
                "success_filter": him_course.narrow_success_filter,
                "type_sampling_minimums": dict(XZ1_TEACHER_NARROW_SAMPLING_MINIMUMS),
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
        actuator.delay_max_lag = XZ1_TEACHER_DECIMATION - 1
        actuator.delay_hold_prob = 0.0
        actuator.delay_update_period = XZ1_TEACHER_DECIMATION
        actuator.delay_per_env_phase = True

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(
                terrain_type="generator",
                terrain_generator=deepcopy(XZ1_TEACHER_TERRAINS_CFG),
                max_init_terrain_level=0,
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
            num_envs=XZ1_TEACHER_NUM_ENVS,
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
            "hip_outward_peak": MetricsTermCfg(
                func=him_course.hip_outward_peak,
                reduce="max",
            ),
            "rr_thigh_action_abs": MetricsTermCfg(
                func=custom_rewards.xz1_joint_action_abs,
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot", joint_names=("RR_thigh_joint",)
                    )
                },
            ),
            "rl_thigh_action_abs": MetricsTermCfg(
                func=custom_rewards.xz1_joint_action_abs,
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot", joint_names=("RL_thigh_joint",)
                    )
                },
            ),
            "rr_thigh_target_error": MetricsTermCfg(
                func=custom_rewards.xz1_joint_target_error,
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot", joint_names=("RR_thigh_joint",)
                    )
                },
            ),
            "rl_thigh_target_error": MetricsTermCfg(
                func=custom_rewards.xz1_joint_target_error,
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot", joint_names=("RL_thigh_joint",)
                    )
                },
            ),
            "rr_thigh_torque_ratio": MetricsTermCfg(
                func=custom_rewards.xz1_joint_torque_ratio,
                reduce="max",
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot", joint_names=("RR_thigh_joint",)
                    ),
                    "effort_limit": 31.7,
                },
            ),
            "rl_thigh_torque_ratio": MetricsTermCfg(
                func=custom_rewards.xz1_joint_torque_ratio,
                reduce="max",
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot", joint_names=("RL_thigh_joint",)
                    ),
                    "effort_limit": 31.7,
                },
            ),
            "rr_thigh_limit_proximity": MetricsTermCfg(
                func=custom_rewards.xz1_joint_limit_proximity,
                reduce="max",
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot", joint_names=("RR_thigh_joint",)
                    )
                },
            ),
            "rl_thigh_limit_proximity": MetricsTermCfg(
                func=custom_rewards.xz1_joint_limit_proximity,
                reduce="max",
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot", joint_names=("RL_thigh_joint",)
                    )
                },
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
                timestep=XZ1_TEACHER_PHYSICS_DT,
                solver="newton",
                iterations=10,
                tolerance=1e-8,
                ls_iterations=20,
                ls_tolerance=0.01,
                ccd_iterations=50,
            ),
        ),
        decimation=XZ1_TEACHER_DECIMATION,
        episode_length_s=XZ1_TEACHER_EPISODE_LENGTH_S,
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
        # Select the tile BEFORE the type-dependent pose reset. Keep columns
        # tied to terrain types and rows to known difficulty, even in play;
        # randomizing geometry per cell would invalidate course boundary checks.
        cfg.events = {
            "randomize_terrain": EventTermCfg(
                func=envs_mdp.randomize_terrain, mode="reset", params={}
            ),
            **cfg.events,
        }
        terrain_generator.curriculum = True
        terrain_generator.num_cols = len(terrain_generator.sub_terrains)
        terrain_generator.num_rows = 5
        terrain_generator.border_width = 10.0
    return cfg
