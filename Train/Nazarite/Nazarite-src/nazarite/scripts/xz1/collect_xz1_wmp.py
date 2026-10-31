"""Record synchronized 10 Hz depth sequences under a frozen XZ1 scan teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from nazarite.config.robot_config.xz1_cfg import XZ1_FOOT_SITES
from nazarite.config.train_config.xz1.collect_env_cfg import (
    Nazarite_WMP_HIM_Collect_XZ1,
    XZ1DepthCfg,
    attach_xz1_depth,
)
from nazarite.config.train_config.xz1.him_course import XZ1CourseVelocityCommandCfg
from nazarite.config.train_config.xz1.him_env_cfg import XZ1_HIM_TERRAIN_ROWS
from nazarite.config.train_config.xz1.him_obstacles import OBSTACLE_TERRAIN_NAMES
from nazarite.config.train_config.xz1.teacher_env_cfg import Nazarite_WMP_HIM_Eval_XZ1
from nazarite.config.train_config.xz1.teacher_rl_cfg import xz1_teacher_runner_cfg
from nazarite.rl.wmp.xz1_dataset import (
    EpisodeWriter,
    sanitize_xz1_depth,
    xz1_wm_prop,
)
from nazarite.rl.xz1.teacher import XZ1TerrainTeacherActor, XZ1TerrainTeacherRunner
from nazarite.scripts.xz1.evaluate_xz1_wmp import (
    TERRAINS,
    EpisodeAllocation,
    course_goal,
    crossing_route_limit,
    evaluation_difficulties,
    landing_half_width,
    landing_mask,
)


def collection_plan(args):
    """Cover exact rows by default; flat has only one distinct geometry."""
    if not args.all_levels and args.levels is None and args.difficulties is None:
        points = [
            (level, level / (XZ1_HIM_TERRAIN_ROWS - 1))
            for level in range(XZ1_HIM_TERRAIN_ROWS)
        ]
    else:
        points = evaluation_difficulties(args)
    speed = args.speed if args.speed is not None else 0.4
    speeds = (args.speeds or [speed]) if args.nominal else [None]
    result = []
    for seed in args.seeds:
        for terrain in args.terrains:
            for level, difficulty in [(0, 0.0)] if terrain == "flat" else points:
                for speed in speeds:
                    result.append(
                        {
                            "case_id": len(result),
                            "terrain": terrain,
                            "training_level": level,
                            "difficulty": difficulty,
                            "seed": seed,
                            "speed_m_s": speed,
                            "episodes": args.episodes_per_case,
                            "num_envs": min(args.num_envs, args.episodes_per_case),
                        }
                    )
    return result


def write_collection_summary(writer):
    """Audit committed data, including partial cases after interruption."""
    episodes = writer.manifest["episodes"]
    cases = []
    for plan in writer.manifest["metadata"]["collection_plan"]:
        rows = [row for row in episodes if row["case_id"] == plan["case_id"]]
        allocation = EpisodeAllocation(plan["episodes"], plan["num_envs"])
        counts = Counter(row["env_id"] for row in rows)
        per_env = [counts[index] for index in range(plan["num_envs"])]
        cases.append(
            {
                **plan,
                "completed": len(rows),
                "complete": per_env == allocation.quotas,
                "episode_quotas": allocation.quotas,
                "episodes_by_env": per_env,
                "success": sum(row["success"] for row in rows),
                "timeout_failure": sum(row["timeout_failure"] for row in rows),
                "success_rate": sum(row["success"] for row in rows) / len(rows)
                if rows
                else None,
                "timeout_rate": sum(row["timeout_failure"] for row in rows) / len(rows)
                if rows
                else None,
                "failure_reasons": dict(
                    Counter(
                        reason
                        for row in rows
                        if not row["success"]
                        for reason in row["termination"]
                    )
                ),
                "frames": sum(row["frames"] for row in rows),
                "eligible_32_frames": sum(row["frames"] >= 32 for row in rows),
                "eligible_64_frames": sum(row["frames"] >= 64 for row in rows),
            }
        )
    by_level = {}
    for case in cases:
        key = f"{case['terrain']}/difficulty_{case['difficulty']:.9g}"
        group = by_level.setdefault(
            key,
            {
                "terrain": case["terrain"],
                "training_level": case["training_level"],
                "difficulty": case["difficulty"],
                "episodes": 0,
                "success": 0,
                "timeout_failure": 0,
                "eligible_32_frames": 0,
                "eligible_64_frames": 0,
            },
        )
        group["episodes"] += case["completed"]
        for name in (
            "success",
            "timeout_failure",
            "eligible_32_frames",
            "eligible_64_frames",
        ):
            group[name] += case[name]
    for group in by_level.values():
        count = group["episodes"]
        group["success_rate"] = group["success"] / count if count else None
        group["timeout_rate"] = group["timeout_failure"] / count if count else None
    summary = {
        "status": writer.manifest["status"],
        "planned_episodes": sum(case["episodes"] for case in cases),
        "completed_episodes": len(episodes),
        "completed_cases": sum(case["complete"] for case in cases),
        "by_terrain_level": list(by_level.values()),
        "splits": {
            split: {
                "episodes": sum(row["split"] == split for row in episodes),
                "eligible_32_frames": sum(
                    row["split"] == split and row["frames"] >= 32 for row in episodes
                ),
                "eligible_64_frames": sum(
                    row["split"] == split and row["frames"] >= 64 for row in episodes
                ),
            }
            for split in ("train", "validation")
        },
        "cases": cases,
    }
    temporary = writer.directory / "collection_summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    temporary.replace(writer.directory / "collection_summary.json")
    return summary


class CollectionTraversal:
    """Track full crossings and stable landings before any explicit reset."""

    def __init__(self, env, terrain, difficulty):
        self.env, self.terrain = env, terrain
        generator = env.scene.terrain.cfg.terrain_generator
        self.goal, self.half_width = course_goal(generator, terrain, difficulty)
        self.platform_half_width = landing_half_width(generator, terrain, difficulty)
        self.feet_ids, _ = env.scene["robot"].find_sites(
            XZ1_FOOT_SITES, preserve_order=True
        )
        self.hold = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.longest = torch.zeros_like(self.hold)
        self.crossed = torch.zeros_like(self.hold, dtype=torch.bool)

    def update(self):
        env = self.env
        robot = env.scene["robot"].data
        root = robot.root_link_pos_w - env.scene.env_origins
        feet = robot.site_pos_w[:, self.feet_ids] - env.scene.env_origins[:, None]
        force = env.scene["feet_ground_contact"].data.force
        support = force[..., 2].reshape(env.num_envs, 4).abs() > 5.0
        if self.terrain == "flat":
            # Training flat commands include all directions, not just +x.
            cleared = torch.linalg.vector_norm(root[:, :2], dim=-1) >= 2.4
            landed = cleared & (support.sum(-1) >= 2)
            landed &= -robot.projected_gravity_b[:, 2] >= math.cos(math.radians(45))
            off_route = torch.zeros_like(cleared)
        else:
            cleared = feet[..., 0].amin(-1) >= self.goal + 0.05
            landed = landing_mask(
                root,
                feet,
                robot.projected_gravity_b,
                support,
                self.goal,
                self.half_width,
                minimum_foot_height=(
                    -0.015 if self.terrain in OBSTACLE_TERRAIN_NAMES else None
                ),
                platform_half_width=self.platform_half_width,
            )
            route_limit = crossing_route_limit(
                root, feet, self.goal, self.half_width, self.platform_half_width
            )
            off_route = root[:, 1].abs() > route_limit
        self.crossed |= cleared
        self.hold = torch.where(landed & ~off_route, self.hold + 1, 0)
        self.longest = torch.maximum(self.longest, self.hold)
        success = self.hold >= math.ceil(0.2 / env.step_dt)
        success &= ~env.termination_manager.terminated & ~off_route
        return success, off_route

    def reset(self, ids):
        self.hold[ids] = self.longest[ids] = 0
        self.crossed[ids] = False


def collection_cfg(args, terrain: str, difficulty: float, seed: int):
    camera = XZ1DepthCfg(pos=tuple(args.camera_pos), fovy=args.camera_fovy)
    if args.nominal:
        cfg = attach_xz1_depth(
            Nazarite_WMP_HIM_Eval_XZ1(
                terrain=terrain,
                difficulty=difficulty,
                seed=seed,
                speed=args.speed,
                command_mode=args.command_mode,
            ),
            camera,
        )
    else:
        cfg = Nazarite_WMP_HIM_Collect_XZ1()
        # Replace the provisional camera with explicitly selected extrinsics.
        cfg.scene.sensors = tuple(
            sensor for sensor in cfg.scene.sensors if sensor.name != "xz1_depth_camera"
        )
        attach_xz1_depth(cfg, camera)
        assert cfg.scene.terrain is not None
        generator = cfg.scene.terrain.terrain_generator
        assert generator is not None
        generator.sub_terrains = {terrain: generator.sub_terrains[terrain]}
        generator.sub_terrains[terrain].proportion = 1.0
        generator.num_rows = generator.num_cols = 1
        generator.curriculum = True
        generator.difficulty_range = (difficulty, difficulty)
        generator.seed = seed
        command = cfg.commands["twist"]
        if not isinstance(command, XZ1CourseVelocityCommandCfg):
            raise TypeError("Unexpected XZ1 teacher command configuration")
        command.forward_only_terrain_names = tuple(
            name for name in command.forward_only_terrain_names if name == terrain
        )
        cfg.scene.terrain.max_init_terrain_level = 0
        # Fixed cases cover the requested difficulties without waiting for a
        # new low-level curriculum. Teacher rewards/events/commands are retained.
        cfg.curriculum = {}
    cfg.seed = seed
    cfg.scene.num_envs = min(args.num_envs, args.episodes_per_case)
    cfg.episode_length_s = args.episode_seconds
    cfg.auto_reset = False
    # The collector owns completion, allowing the same 0.2 s stable landing
    # check in nominal and randomized profiles. Physical failures stay enabled.
    cfg.terminations.pop("narrow_course_complete", None)
    return cfg, camera


def collect_case(
    args,
    writer,
    terrain: str,
    difficulty: float,
    seed: int,
    training_level=None,
    case_id=0,
):
    cfg, camera = collection_cfg(args, terrain, difficulty, seed)
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        if abs(env.step_dt - 0.02) > 1e-8:
            raise ValueError("XZ1 collection contract requires a 50 Hz policy")
        wrapped = RslRlVecEnvWrapper(env, clip_actions=10.0)
        runner = XZ1TerrainTeacherRunner(
            wrapped, asdict(xz1_teacher_runner_cfg()), device=args.device
        )
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
        actor = runner.alg.actor
        if not isinstance(actor, XZ1TerrainTeacherActor):
            raise TypeError("Collection requires the XZ1 privileged terrain teacher")
        env.common_step_counter = 0
        obs_dict, _ = env.reset(seed=seed)
        obs = TensorDict(obs_dict, batch_size=[env.num_envs])
        histories = torch.zeros(env.num_envs, 5, 12, device=args.device)
        ticks = torch.zeros(env.num_envs, dtype=torch.long, device=args.device)
        frames: list[list[dict[str, torch.Tensor]]] = [[] for _ in range(env.num_envs)]
        allocation = EpisodeAllocation(args.episodes_per_case, env.num_envs)
        traversal = CollectionTraversal(env, terrain, difficulty)
        active = torch.ones(env.num_envs, dtype=torch.bool, device=args.device)
        command_sum = torch.zeros(env.num_envs, 3, device=args.device)
        command_min = torch.full_like(command_sum, float("inf"))
        command_max = torch.full_like(command_sum, -float("inf"))
        steps = 0

        def teacher_actions():
            commands = policy(obs)
            if not torch.isfinite(commands).all():
                raise RuntimeError("Teacher produced non-finite commands")
            return commands.clamp(-10, 10)

        def capture(ids):
            # sim.sense may skip rendering during reset or drift in phase after
            # partial resets. Explicitly acquire current depth at this timestamp.
            # It does not advance physics, HIM history, commands or delay buffers.
            env.sim.sense(force_render=True)
            depth = env.scene["xz1_depth_camera"].data.depth
            if depth is None or depth.shape != (env.num_envs, 64, 64, 1):
                raise RuntimeError("XZ1 camera returned an invalid depth frame")
            depth = sanitize_xz1_depth(depth, camera.far_clip)
            prop = xz1_wm_prop(obs["estimator"])
            labels = teacher_actions()
            latent = actor.scan_encoder(obs["teacher_scan"])
            for index in ids.tolist():
                first = not frames[index]
                row = {
                    "depth_m": depth[index].to(torch.float16),
                    "prop": prop[index],
                    "actor_history": obs["actor"][index],
                    "teacher_scan": obs["teacher_scan"][index],
                    "teacher_latent": latent[index],
                    "teacher_action": labels[index],
                    "action": histories[index].flatten(),
                    "is_first": torch.tensor(first),
                    "time_s": ticks[index].double() * env.step_dt,
                    "global_time_s": torch.tensor(
                        steps * env.step_dt, dtype=torch.float64
                    ),
                }
                frames[index].append(
                    {name: value.detach().cpu().clone() for name, value in row.items()}
                )

        completed = 0
        with torch.inference_mode():
            capture(torch.arange(env.num_envs, device=args.device))
            while completed < args.episodes_per_case:
                commands = obs["estimator"][:, 6:9]
                command_sum += commands
                command_min = torch.minimum(command_min, commands)
                command_max = torch.maximum(command_max, commands)
                actions = teacher_actions()
                histories = torch.roll(histories, -1, dims=1)
                histories[:, -1] = actions
                obs, rewards, dones, _ = wrapped.step(actions)
                if not torch.isfinite(rewards).all():
                    raise RuntimeError("Non-finite simulation rewards")
                ticks += 1
                steps += 1
                success, off_route = traversal.update()
                due = ((ticks % camera.interval == 0) & active).nonzero().flatten()
                if due.numel():
                    capture(due)
                finished = (
                    (dones.bool() | ((success | off_route) & active))
                    .nonzero()
                    .flatten()
                )
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
                        reasons.append("off_collection_route")
                    passed = bool(success[index])
                    timeout = (
                        "time_out" in reasons
                        and not passed
                        and not bool(env.termination_manager.terminated[index])
                        and not bool(off_route[index])
                    )
                    writer.add(
                        frames[index],
                        {
                            "case_id": case_id,
                            "terrain": terrain,
                            "training_level": training_level,
                            "difficulty": difficulty,
                            "seed": seed,
                            "command_profile": (
                                args.command_mode
                                if args.nominal
                                else "training_commands"
                            ),
                            "speed_m_s": args.speed if args.nominal else None,
                            "command_mean_b": (
                                command_sum[index] / ticks[index]
                            ).tolist(),
                            "command_min_b": command_min[index].tolist(),
                            "command_max_b": command_max[index].tolist(),
                            "split": "validation"
                            if seed in args.validation_seeds
                            else "train",
                            "env_id": index,
                            "episode": episode_index,
                            "success": passed,
                            "timeout_failure": timeout,
                            "time_limit_reached": "time_out" in reasons,
                            "outcome": "success"
                            if passed
                            else ("timeout" if timeout else "failure"),
                            "crossed_obstacle": bool(traversal.crossed[index]),
                            "max_stable_landing_seconds": (
                                int(traversal.longest[index]) * env.step_dt
                            ),
                            "termination": reasons,
                            "completion": "stable_traversal" if passed else None,
                            "terminal_time_s": int(ticks[index]) * env.step_dt,
                            "terminal_frame_sampled": bool(
                                ticks[index] % camera.interval == 0
                            ),
                        },
                    )
                    frames[index] = []
                    active[index] = (
                        allocation.completed[index] < allocation.quotas[index]
                    )
                    completed += 1
                if finished.numel() and completed < args.episodes_per_case:
                    reset_obs, _ = env.reset(env_ids=finished)
                    obs = TensorDict(reset_obs, batch_size=[env.num_envs])
                    reset_mask = torch.zeros_like(active)
                    reset_mask[finished] = True
                    policy.reset(dones=reset_mask)
                    histories[finished] = 0
                    ticks[finished] = 0
                    command_sum[finished] = 0
                    command_min[finished] = float("inf")
                    command_max[finished] = -float("inf")
                    traversal.reset(finished)
                    pending = finished[active[finished]]
                    if pending.numel():
                        capture(pending)
        assert allocation.completed == allocation.quotas
        print(
            f"Collected {terrain} d={difficulty} seed={seed}: "
            f"{completed} episodes, {steps} control steps",
            flush=True,
        )
    finally:
        env.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--terrains", nargs="+", choices=TERRAINS, default=list(TERRAINS)
    )
    levels = parser.add_mutually_exclusive_group()
    levels.add_argument("--all-levels", action="store_true")
    levels.add_argument("--levels", type=int, nargs="+")
    levels.add_argument("--difficulties", type=float, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--validation-seeds", type=int, nargs="+", default=[43])
    parser.add_argument("--episodes-per-case", type=int, default=8)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--episode-seconds", type=float, default=20.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--nominal",
        action="store_true",
        help="Disable training randomization; fixed forward command",
    )
    speeds = parser.add_mutually_exclusive_group()
    speeds.add_argument("--speed", type=float)
    speeds.add_argument("--speeds", type=float, nargs="+")
    parser.add_argument(
        "--command-mode",
        choices=("velocity", "course"),
        default="velocity",
        help="Only applies to --nominal",
    )
    parser.add_argument("--camera-pos", type=float, nargs=3, default=[0.20, 0, 0.08])
    parser.add_argument("--camera-fovy", type=float, default=58.0)
    args = parser.parse_args(argv)
    if not args.checkpoint.is_file() or args.output.exists():
        parser.error("Need an existing teacher checkpoint and a new output directory")
    if (
        min(args.num_envs, args.episodes_per_case) < 1
        or not math.isfinite(args.episode_seconds)
        or args.episode_seconds < 0.1
    ):
        parser.error("Environment/episode counts must be positive; duration >=0.1s")
    validation = set(args.validation_seeds)
    if not validation < set(args.seeds) or not validation:
        parser.error("Validation seeds must be a nonempty proper subset of --seeds")
    if args.difficulties is not None and any(
        not 0 <= d <= 1 for d in args.difficulties
    ):
        parser.error("Difficulties must be in [0,1]")
    if args.levels is not None and any(
        not 0 <= level < XZ1_HIM_TERRAIN_ROWS for level in args.levels
    ):
        parser.error(f"Levels must be in [0,{XZ1_HIM_TERRAIN_ROWS - 1}]")
    if not args.nominal and (args.speed is not None or args.speeds is not None):
        parser.error(
            "--speed/--speeds only apply to --nominal; training samples speeds"
        )
    speed = args.speed if args.speed is not None else 0.4
    if any(not 0 < value <= 1 for value in (args.speeds or [speed])):
        parser.error("Nominal speeds must be in (0,1]")
    if not args.nominal and args.command_mode != "velocity":
        parser.error("--command-mode only applies to --nominal")
    for name in (
        "terrains",
        "seeds",
        "validation_seeds",
        "levels",
        "difficulties",
        "speeds",
    ):
        values = getattr(args, name)
        if values is not None and len(values) != len(set(values)):
            parser.error(f"Duplicate --{name.replace('_', '-')} entries")
    if not 0 < args.camera_fovy < 180 or not all(
        math.isfinite(value) for value in args.camera_pos
    ):
        parser.error("Camera FOV must be in (0,180) and position must be finite")
    return args


def main():
    args = parse_args()
    validation = set(args.validation_seeds)
    plan = collection_plan(args)
    torch.manual_seed(args.seeds[0])
    camera = XZ1DepthCfg(pos=tuple(args.camera_pos), fovy=args.camera_fovy)
    writer = EpisodeWriter(
        args.output,
        {
            "teacher_checkpoint": str(args.checkpoint.resolve()),
            "teacher_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "camera": asdict(camera),
            "profile": "nominal" if args.nominal else "teacher_training",
            "command_mode": args.command_mode if args.nominal else "training_commands",
            "collection_plan": plan,
            "sampling": "per_environment_quota_v1",
            "outcome_version": "stable_traversal_platform_bounds_v1",
            "success_definition": "full crossing, upright, 2 loaded feet, 0.2s stable; "
            "flat uses 2.4m displacement in any direction; physical failure wins",
            "validation_seeds": sorted(validation),
            "terminal_between_frames": "terminal metadata; no partial action chunk",
            "unfinished_parallel_episodes": "all assigned quotas finish",
            "camera_acquisition": "fresh at sample/reset; may render extra frames",
            "timestamp_origin": "global_time_s: case clock; time_s: episode clock",
        },
    )
    print(
        f"Collection plan: {len(plan)} cases, "
        f"{sum(case['episodes'] for case in plan)} complete episodes; "
        "fixed per-environment quotas; depth 64x64 at 10 Hz",
        flush=True,
    )
    try:
        write_collection_summary(writer)
        for case in plan:
            case_args = argparse.Namespace(**{**vars(args), "speed": case["speed_m_s"]})
            print(
                f"Starting case {case['case_id'] + 1}/{len(plan)}: {case}", flush=True
            )
            collect_case(
                case_args,
                writer,
                case["terrain"],
                case["difficulty"],
                case["seed"],
                training_level=case["training_level"],
                case_id=case["case_id"],
            )
            summary = write_collection_summary(writer)
            latest = summary["cases"][case["case_id"]]
            if not latest["complete"]:
                raise RuntimeError("Collection did not fulfill every environment quota")
            print(
                f"Case quality: success={latest['success']}/{latest['completed']}, "
                f"timeout={latest['timeout_failure']}, "
                f"eligible sequences: 32 frames={latest['eligible_32_frames']}, "
                f"64 frames={latest['eligible_64_frames']}",
                flush=True,
            )
    except BaseException:
        writer.finish("interrupted")
        write_collection_summary(writer)
        raise
    writer.finish()
    write_collection_summary(writer)
    print(f"Dataset written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
