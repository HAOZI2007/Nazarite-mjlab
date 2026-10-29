"""XZ1 privileged teacher using Extreme Parkour's exact scan coordinates."""

from dataclasses import dataclass

import torch

from mjlab.envs import mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.recorder_manager import RecorderTermCfg
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from nazarite.config.robot_config.xz1_cfg import XZ1_BASE_BODY
from nazarite.mdp.xz1_diagnostics import XZ1CourseDiagnostics

from .him_course import COURSE_TERRAIN_NAMES, XZ1CourseVelocityCommandCfg
from .him_env_cfg import XZ1_WMP_COMPLEX_TERRAINS_CFG, _build_xz1_env

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
    return mdp.height_scan(
        env, sensor_name="teacher_terrain_scan", offset=0.3, miss_value=1.0
    ).clamp(-1.0, 1.0)


def Nazarite_WMP_HIM_Teacher_XZ1(play: bool = False):
    """Add an oracle actor input without changing XZ1 HIM or its 244-D critic."""
    cfg = _build_xz1_env(play, XZ1_WMP_COMPLEX_TERRAINS_CFG, obstacle_courses=True)
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
    cfg = Nazarite_WMP_HIM_Teacher_XZ1()
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
    if command_mode == "course" and terrain in COURSE_TERRAIN_NAMES:
        cfg.commands["twist"] = XZ1CourseVelocityCommandCfg(
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
    cfg.events["reset_base"].func = mdp.reset_root_state_uniform
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
