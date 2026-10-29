"""Fixed-seed, fixed-difficulty XZ1 teacher/HIM crossing benchmark."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from nazarite.config.robot_config.xz1_cfg import XZ1_FOOT_SITES
from nazarite.config.train_config.XZ1_env_cfgs.him_course import COURSE_TERRAIN_NAMES
from nazarite.config.train_config.XZ1_env_cfgs.him_env_cfg import XZ1_HIM_TERRAIN_ROWS
from nazarite.config.train_config.XZ1_env_cfgs.him_obstacles import (
    OBSTACLE_TERRAIN_NAMES,
    XZ1_TERRAIN_NAMES,
)
from nazarite.config.train_config.XZ1_env_cfgs.him_rl_cfg import xz1_him_runner_cfg
from nazarite.config.train_config.XZ1_env_cfgs.teacher_env_cfg import (
    Nazarite_WMP_HIM_Eval_XZ1,
)
from nazarite.config.train_config.XZ1_env_cfgs.teacher_rl_cfg import (
    xz1_teacher_runner_cfg,
)
from nazarite.rl.him_runner import HIMOnPolicyRunner
from nazarite.rl.xz1_student import (
    load_student_policy,
    student_eval_cfg,
)
from nazarite.rl.xz1_teacher import XZ1TerrainTeacherRunner

TERRAINS = XZ1_TERRAIN_NAMES


class EpisodeAllocation:
    """Preassign episode quotas; fast failures cannot replace slow trials."""

    def __init__(self, episodes: int, num_envs: int):
        if min(episodes, num_envs) < 1:
            raise ValueError("Episode and environment counts must be positive")
        self.quotas = [
            episodes // num_envs + int(index < episodes % num_envs)
            for index in range(num_envs)
        ]
        self.completed = [0] * num_envs

    def finish(self, index: int):
        if self.completed[index] >= self.quotas[index]:
            return None
        episode_index = self.completed[index]
        self.completed[index] += 1
        return episode_index


def evaluation_difficulties(args):
    """Map exact training rows to geometry; retain the old difficulty interface."""
    if args.all_levels or args.levels is not None:
        levels = range(XZ1_HIM_TERRAIN_ROWS) if args.all_levels else args.levels
        if any(level < 0 or level >= XZ1_HIM_TERRAIN_ROWS for level in levels):
            raise ValueError(f"levels must be in [0,{XZ1_HIM_TERRAIN_ROWS - 1}]")
        return [(level, level / (XZ1_HIM_TERRAIN_ROWS - 1)) for level in levels]
    difficulties = args.difficulties if args.difficulties is not None else [0, 0.5, 1]
    if any(not 0 <= value <= 1 for value in difficulties):
        raise ValueError("difficulties must be in [0,1]")
    return [(None, value) for value in difficulties]


def course_goal(generator, terrain: str, difficulty: float):
    """Return far obstacle edge relative to spawn, plus valid lateral bound."""
    cfg = generator.sub_terrains[terrain]
    width = generator.size[1]
    if terrain in OBSTACLE_TERRAIN_NAMES:
        _, end, width = cfg.course_profile(difficulty)
        end -= generator.size[0] * cfg.spawn_x_fraction
    elif terrain == "high_wall":
        end = generator.size[0] * (0.5 - cfg.spawn_x_fraction)
        end += cfg.wall_thickness / 2
    elif terrain == "narrow_stairs":
        end = (
            cfg.start_x
            + 2 * (cfg.num_steps - 1) * cfg.step_depth
            + cfg.deck_length
            - generator.size[0] * cfg.spawn_x_fraction
        )
        width = cfg.width_range[0] + difficulty * (
            cfg.width_range[1] - cfg.width_range[0]
        )
    elif terrain == "narrow_ramp":
        height = cfg.height_range[0] + difficulty * (
            cfg.height_range[1] - cfg.height_range[0]
        )
        angle = cfg.slope_angle_range[0] + difficulty * (
            cfg.slope_angle_range[1] - cfg.slope_angle_range[0]
        )
        end = (
            cfg.start_x
            + 2 * height / math.tan(math.radians(angle))
            + cfg.deck_length
            - generator.size[0] * cfg.spawn_x_fraction
        )
        if cfg.width_range is not None:
            width = cfg.width_range[0] + difficulty * (
                cfg.width_range[1] - cfg.width_range[0]
            )
    else:
        # Flat is a fixed-distance traversal rather than an isolated obstacle.
        end = 2.5
    return end, width / 2 - 0.12


def landing_half_width(generator, terrain: str, difficulty: float):
    """The bridge's far platform is wider than its narrow crossing surface."""
    if terrain == "single_beam":
        return generator.sub_terrains[terrain].platform_width / 2
    return course_goal(generator, terrain, difficulty)[1] + 0.12


def crossing_route_limit(root, feet, goal_x, half_width, platform_half_width):
    """Keep the crossing corridor until the trunk AND all feet clear its end."""
    cleared = (feet[..., 0].amin(-1) >= goal_x + 0.05) & (root[:, 0] >= goal_x + 0.35)
    return torch.where(cleared, platform_half_width - 0.12, half_width)


def landing_mask(
    root,
    feet,
    gravity,
    support,
    goal_x,
    half_width,
    minimum_foot_height=None,
    platform_half_width=None,
):
    """All feet past the far edge, upright with two loaded feet on landing."""
    root_limit = half_width
    feet_limit = half_width + 0.10
    if platform_half_width is not None:
        root_limit = platform_half_width - 0.12
        feet_limit = platform_half_width - 0.02
    landed = (
        (feet[..., 0].amin(-1) >= goal_x + 0.05)
        & (root[:, 0] >= goal_x + 0.35)
        & (root[:, 1].abs() <= root_limit)
        & (feet[..., 1].abs().amax(-1) <= feet_limit)
        & (-gravity[:, 2] >= math.cos(math.radians(45)))
        & (support.sum(-1) >= 2)
    )
    if minimum_foot_height is not None:
        landed &= feet[..., 2].amin(-1) > minimum_foot_height
    return landed


def evaluate_case(args, terrain: str, difficulty: float):
    cfg_factory = Nazarite_WMP_HIM_Eval_XZ1
    if args.policy == "student":
        from functools import partial

        cfg_factory = partial(student_eval_cfg, args.checkpoint)
    cfg = cfg_factory(
        terrain=terrain,
        difficulty=difficulty,
        speed=args.speed,
        seed=args.seed,
        reset_x=getattr(args, "reset_x", 0.0),
        reset_y=getattr(args, "reset_y", 0.0),
        reset_yaw=getattr(args, "reset_yaw", 0.0),
        command_mode=getattr(args, "command_mode", "velocity"),
    )
    cfg.scene.num_envs = args.num_envs
    cfg.episode_length_s = args.episode_seconds
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    records = []
    allocation = EpisodeAllocation(args.episodes, args.num_envs)
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=10.0)
        policy: Any
        if args.policy == "student":
            initial_obs = wrapped.get_observations()
            policy = load_student_policy(
                env,
                initial_obs,
                args.checkpoint,
                args.device,
                getattr(args, "vision", "normal"),
            )
        else:
            if args.policy == "teacher":
                runner_cfg, runner_type = (
                    xz1_teacher_runner_cfg(),
                    XZ1TerrainTeacherRunner,
                )
            else:
                runner_cfg, runner_type = xz1_him_runner_cfg(), HIMOnPolicyRunner
            runner = runner_type(wrapped, asdict(runner_cfg), device=args.device)
            runner.load(
                str(args.checkpoint),
                load_cfg={
                    "actor": True,
                    "critic": False,
                    "optimizer": False,
                    "estimator": True,
                    "iteration": False,
                },
            )
            policy = runner.get_inference_policy(device=args.device)
        # Loading training counters cannot affect this benchmark's phase/reset.
        env.common_step_counter = 0
        obs, _ = env.reset(seed=args.seed)
        obs = TensorDict(obs, batch_size=[env.num_envs])
        assert cfg.scene.terrain is not None
        generator = cfg.scene.terrain.terrain_generator
        assert generator is not None
        goal, half_width = course_goal(generator, terrain, difficulty)
        platform_half_width = landing_half_width(generator, terrain, difficulty)
        feet_ids, _ = env.scene["robot"].find_sites(XZ1_FOOT_SITES, preserve_order=True)
        hold = torch.zeros(env.num_envs, dtype=torch.long, device=args.device)
        errors = torch.zeros(env.num_envs, device=args.device)
        counts = torch.zeros_like(hold)
        crossed = torch.zeros_like(hold, dtype=torch.bool)
        longest_hold = torch.zeros_like(hold)
        lateral_peak = torch.zeros_like(errors)
        hold_steps = math.ceil(0.2 / env.step_dt)
        with torch.inference_mode():
            while len(records) < args.episodes:
                actions = policy(obs)
                if not torch.isfinite(actions).all():
                    raise RuntimeError("Non-finite evaluation actions")
                if args.policy == "student":
                    policy.advance(actions)
                obs, rewards, dones, _ = wrapped.step(actions)
                if not torch.isfinite(rewards).all():
                    raise RuntimeError("Non-finite evaluation rewards")
                robot = env.scene["robot"].data
                root = robot.root_link_pos_w - env.scene.env_origins
                feet = robot.site_pos_w[:, feet_ids] - env.scene.env_origins[:, None]
                force = env.scene["feet_ground_contact"].data.force
                assert force is not None
                support = force[..., 2].reshape(env.num_envs, 4).abs() > 5.0
                route_limit = crossing_route_limit(
                    root, feet, goal, half_width, platform_half_width
                )
                off_route = root[:, 1].abs() > route_limit
                crossed |= feet[..., 0].amin(-1) >= goal + 0.05
                lateral_peak = torch.maximum(lateral_peak, root[:, 1].abs())
                landed = landing_mask(
                    root,
                    feet,
                    robot.projected_gravity_b,
                    support,
                    goal,
                    half_width,
                    minimum_foot_height=(
                        -0.015 if terrain in OBSTACLE_TERRAIN_NAMES else None
                    ),
                    platform_half_width=platform_half_width,
                )
                hold = torch.where(landed, hold + 1, 0)
                longest_hold = torch.maximum(longest_hold, hold)
                # Failure always wins over an overlapping success/timeout.
                success = (
                    (hold >= hold_steps)
                    & ~off_route
                    & ~env.termination_manager.terminated
                )
                counts += 1
                errors += (robot.root_link_lin_vel_b[:, 0] - args.speed).abs()
                finished = (dones.bool() | success | off_route).nonzero().flatten()
                for index in finished.tolist():
                    episode_index = allocation.finish(index)
                    if episode_index is None:
                        continue
                    reasons = [
                        name
                        for name in env.termination_manager.active_terms
                        if env.termination_manager.get_term(name)[index]
                    ]
                    if off_route[index]:
                        reasons.append("off_evaluation_route")
                    records.append(
                        {
                            "env_id": index,
                            "episode_index": episode_index,
                            "success": bool(success[index]),
                            "termination": reasons,
                            "seconds": int(counts[index]) * env.step_dt,
                            "forward_distance_m": float(root[index, 0]),
                            "velocity_mae_m_s": float(errors[index] / counts[index]),
                            "crossed_obstacle": bool(crossed[index]),
                            "max_stable_landing_seconds": (
                                int(longest_hold[index]) * env.step_dt
                            ),
                            "max_lateral_deviation_m": float(lateral_peak[index]),
                        }
                    )
                if finished.numel():
                    reset_obs, _ = env.reset(env_ids=finished)
                    obs = TensorDict(reset_obs, batch_size=[env.num_envs])
                    policy.reset(
                        dones=torch.isin(
                            torch.arange(env.num_envs, device=args.device), finished
                        )
                    )
                    hold[finished] = errors[finished] = counts[finished] = 0
                    crossed[finished] = False
                    longest_hold[finished] = 0
                    lateral_peak[finished] = 0
        records.sort(key=lambda row: (row["env_id"], row["episode_index"]))
        failures = Counter(
            name for row in records if not row["success"] for name in row["termination"]
        )
        return {
            "terrain": terrain,
            "vision": getattr(args, "vision", "normal"),
            "difficulty": difficulty,
            "seed": args.seed,
            "speed_m_s": args.speed,
            "command_mode": getattr(args, "command_mode", "velocity"),
            "privileged_course_steering": (
                getattr(args, "command_mode", "velocity") == "course"
                and terrain in COURSE_TERRAIN_NAMES
            ),
            "success_rate": sum(row["success"] for row in records) / len(records),
            "sampling": "per_environment_quota_v1",
            "episode_quotas": allocation.quotas,
            "failure_causes": dict(failures),
            "episodes": records,
            "far_edge_from_spawn_m": goal,
            "root_lateral_limit_m": half_width,
            "landing_platform_half_width_m": platform_half_width,
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--policy", choices=("him", "teacher", "student"), required=True
    )
    parser.add_argument(
        "--vision", choices=("normal", "far", "shuffled"), default="normal"
    )
    parser.add_argument(
        "--terrains",
        nargs="+",
        choices=TERRAINS,
        default=list(TERRAINS),
    )
    difficulty_group = parser.add_mutually_exclusive_group()
    difficulty_group.add_argument("--difficulties", type=float, nargs="+")
    difficulty_group.add_argument("--levels", type=int, nargs="+")
    difficulty_group.add_argument("--all-levels", action="store_true")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--speed", type=float, default=0.4)
    parser.add_argument("--speeds", type=float, nargs="+")
    parser.add_argument("--reset-x", type=float, default=0.0)
    parser.add_argument("--reset-y", type=float, default=0.0)
    parser.add_argument("--reset-yaw", type=float, default=0.0)
    parser.add_argument(
        "--command-mode", choices=("velocity", "course"), default="velocity"
    )
    parser.add_argument("--episode-seconds", type=float, default=20.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1 or args.num_envs < 1 or args.episode_seconds <= 0:
        parser.error("episodes, num-envs and episode-seconds must be positive")
    try:
        difficulty_cases = evaluation_difficulties(args)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.checkpoint.is_file():
        parser.error("checkpoint does not exist")
    if args.output.exists():
        parser.error("output already exists; select a new filename")
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "policy": args.policy,
        "vision": args.vision,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "sampling": "per_environment_quota_v1",
        "training_num_levels": XZ1_HIM_TERRAIN_ROWS,
        "speed_m_s": args.speed,
        "nominal_no_noise_no_delays_no_randomization": True,
        "deterministic_repeats_not_independent_robustness_trials": (
            args.reset_x == args.reset_y == args.reset_yaw == 0
        ),
        "reset_jitter": {
            "x": args.reset_x,
            "y": args.reset_y,
            "yaw_rad": args.reset_yaw,
        },
        "command_mode": args.command_mode,
        "seeds": args.seeds or [args.seed],
        "speeds_m_s": args.speeds or [args.speed],
        "success_definition": "all feet past edge, 2 loaded feet, upright, "
        "0.2s stable landing, failure takes precedence",
        "cases": [],
    }
    for seed in result["seeds"]:
        for speed in result["speeds_m_s"]:
            args.seed, args.speed = seed, speed
            for terrain in args.terrains:
                for level, difficulty in difficulty_cases:
                    case = evaluate_case(args, terrain, difficulty)
                    case["training_level"] = level
                    result["cases"].append(case)
                    print(
                        f"{terrain} d={difficulty} seed={seed} vx={speed}: "
                        f"{case['success_rate']:.1%}",
                        flush=True,
                    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(f"Evaluation written to {args.output}")


if __name__ == "__main__":
    main()
