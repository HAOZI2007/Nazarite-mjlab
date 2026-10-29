"""Online XZ1 scan-teacher -> frozen-WMP visual-student action distillation."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from nazarite.config.train_config.XZ1_env_cfgs.collect_env_cfg import attach_xz1_depth
from nazarite.config.train_config.XZ1_env_cfgs.teacher_env_cfg import (
    Nazarite_WMP_HIM_Teacher_XZ1,
)
from nazarite.rl.wmp.xz1_dataset import CONTRACT
from nazarite.rl.xz1_student import (
    STUDENT_FORMAT,
    XZ1VisualPolicy,
    action_distillation_step,
    actor_kwargs,
    build_student,
    file_sha256,
    load_frozen_world_model,
    load_teacher,
    warm_start_student,
)


def teacher_probability(iteration, warmup, mix):
    if iteration < warmup:
        return 1.0
    return max(0.0, 1.0 - (iteration - warmup + 1) / mix) if mix else 0.0


CURRICULUM_BUFFERS = (
    "_delta_success_streak",
    "_delta_failure_streak",
    "_delta_type_success_ema",
    "_delta_type_episode_count",
)


def curriculum_state(env):
    terrain = env.scene.terrain
    return {
        "terrain_names": list(terrain.cfg.terrain_generator.sub_terrains),
        "common_step_counter": env.common_step_counter,
        "levels": terrain.terrain_levels.clone(),
        "types": terrain.terrain_types.clone(),
        "buffers": {
            k: getattr(env, k).clone() for k in CURRICULUM_BUFFERS if hasattr(env, k)
        },
    }


def restore_curriculum(env, state):
    terrain = env.scene.terrain
    names = list(terrain.cfg.terrain_generator.sub_terrains)
    saved_names = state.get("terrain_names")
    saved_counts = state["buffers"].get("_delta_type_episode_count")
    if (saved_names is not None and saved_names != names) or (
        saved_names is None
        and (saved_counts is None or saved_counts.numel() != len(names))
    ):
        raise ValueError(
            "Student resume terrain layout differs; start a new distillation run "
            "with a teacher and WM trained on the current terrains"
        )
    terrain.terrain_levels.copy_(state["levels"])
    terrain.terrain_types.copy_(state["types"])
    terrain.env_origins.copy_(terrain.terrain_origins[state["levels"], state["types"]])
    # Reset physical episodes without scoring the new spawn as a failed episode.
    env.common_step_counter = 0
    raw, _ = env.reset()
    for name, value in state["buffers"].items():
        setattr(env, name, value.clone())
    env.common_step_counter = state["common_step_counter"]
    return TensorDict(raw, batch_size=[env.num_envs])


def run(args):
    torch.manual_seed(args.seed)
    wm, camera = load_frozen_world_model(
        args.world_model, args.device, file_sha256(args.teacher)
    )
    cfg = attach_xz1_depth(Nazarite_WMP_HIM_Teacher_XZ1(), camera)
    cfg.auto_reset = False
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    args.output.mkdir(parents=True, exist_ok=False)
    env = ManagerBasedRlEnv(cfg, device=args.device)
    writer = SummaryWriter(str(args.output))
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=10.0)
        raw, _ = env.reset(seed=args.seed)
        obs = TensorDict(raw, batch_size=[env.num_envs])
        teacher = load_teacher(obs, args.teacher, args.device)
        student = build_student(obs, args.device)
        warm_start_student(student, teacher)
        optimizer = torch.optim.Adam(
            [p for p in student.parameters() if p.requires_grad], lr=args.learning_rate
        )
        metadata = {
            "format": STUDENT_FORMAT,
            "contract": CONTRACT,
            "actor_kwargs": actor_kwargs(),
            "camera": asdict(camera),
            "teacher_path": str(args.teacher.resolve()),
            "teacher_sha256": file_sha256(args.teacher),
            "world_model_path": str(args.world_model.resolve()),
            "world_model_sha256": file_sha256(args.world_model),
            "settings": {
                k: str(v) if isinstance(v, Path) else v
                for k, v in vars(args).items()
                if k not in ("output", "resume", "iterations", "save_interval")
            },
        }
        start = 0
        if args.resume:
            source = torch.load(
                args.resume, map_location=args.device, weights_only=True
            )
            for key in (
                "format",
                "contract",
                "actor_kwargs",
                "teacher_sha256",
                "world_model_sha256",
                "settings",
            ):
                if source[key] != metadata[key]:
                    raise ValueError(f"Student resume contract differs: {key}")
            student.load_state_dict(source["actor_state_dict"], strict=True)
            optimizer.load_state_dict(source["optimizer"])
            start = source["iteration"]
            obs = restore_curriculum(env, source["curriculum"])
            torch.set_rng_state(source["cpu_rng"].cpu())
            if args.device.startswith("cuda"):
                torch.cuda.set_rng_state(source["cuda_rng"].cpu(), args.device)
            # Physical environments and RSSM episodes restart on resume. Do not
            # claim bit-exact continuation of simulator/curriculum state.
        if args.iterations <= start:
            raise ValueError("iterations must exceed resumed iteration")
        (args.output / "config.json").write_text(json.dumps(metadata, indent=2))
        policy = XZ1VisualPolicy(env, student, wm, camera)
        returns = torch.zeros(env.num_envs, device=args.device)
        lengths = torch.zeros_like(returns)
        for iteration in range(start, args.iterations):
            begin = time.monotonic()
            buffers, labels, episode_returns, episode_lengths = [], [], [], []
            failures = {}
            probability = teacher_probability(
                iteration, args.teacher_warmup, args.teacher_mix_iterations
            )
            for _ in range(args.steps_per_iteration):
                with torch.no_grad():
                    visual_obs = policy.observations(obs)
                    target = teacher(obs).clamp(-10, 10)
                    predicted = student(visual_obs).clamp(-10, 10)
                    use_teacher = (
                        torch.rand(env.num_envs, 1, device=args.device) < probability
                    )
                    executed = torch.where(use_teacher, target, predicted)
                    buffers.append(visual_obs)
                    labels.append(target.clone())
                    policy.advance(executed)
                    obs, reward, dones, _ = wrapped.step(executed)
                    if not torch.isfinite(reward).all():
                        raise RuntimeError("Non-finite simulation rewards")
                    returns += reward
                    lengths += 1
                    ids = dones.nonzero().flatten()
                    if ids.numel():
                        episode_returns.extend(returns[ids].tolist())
                        episode_lengths.extend(lengths[ids].tolist())
                        for name in env.termination_manager.active_terms:
                            count = int(
                                env.termination_manager.get_term(name)[ids].sum()
                            )
                            failures[name] = failures.get(name, 0) + count
                        raw, _ = env.reset(env_ids=ids)
                        obs = TensorDict(raw, batch_size=[env.num_envs])
                        policy.reset(dones.bool())
                        returns[ids] = lengths[ids] = 0
            loss, grad = action_distillation_step(
                student,
                optimizer,
                torch.cat(buffers, dim=0),
                torch.cat(labels),
            )
            step = iteration + 1
            metrics = {
                "Distill/action_l2": loss,
                "Distill/grad_norm": grad,
                "Distill/teacher_probability": probability,
                "Perf/control_samples_per_second": env.num_envs
                * args.steps_per_iteration
                / (time.monotonic() - begin),
            }
            if episode_returns:
                metrics["Episode/reward"] = sum(episode_returns) / len(episode_returns)
                metrics["Episode/length_steps"] = sum(episode_lengths) / len(
                    episode_lengths
                )
            metrics.update({f"Termination/{k}": v for k, v in failures.items()})
            terrain = env.scene.terrain
            if terrain is not None and hasattr(terrain, "terrain_levels"):
                metrics["Curriculum/terrain_level"] = float(
                    terrain.terrain_levels.float().mean()
                )
            for name, value in metrics.items():
                writer.add_scalar(name, value, step)
            if step % 10 == 0 or step == args.iterations:
                print(
                    f"Student {step}: action_l2={loss:.5f}, teacher_p={probability:.3f}",
                    flush=True,
                )
            if step % args.save_interval == 0 or step == args.iterations:
                temporary = args.output / f"student_{step}.pt.tmp"
                torch.save(
                    {
                        **metadata,
                        "iteration": step,
                        "actor_state_dict": student.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "curriculum": curriculum_state(env),
                        "cpu_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state(args.device)
                        if args.device.startswith("cuda")
                        else None,
                    },
                    temporary,
                )
                temporary.replace(args.output / f"student_{step}.pt")
                writer.flush()
    finally:
        writer.close()
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--world-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps-per-iteration", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--save-interval", type=int, default=100)
    # Reference learn_vision has num_pretrain_iter=0 (student executes from start).
    # Optional teacher execution warmup/mixing is an explicit project adaptation.
    parser.add_argument("--teacher-warmup", type=int, default=0)
    parser.add_argument("--teacher-mix-iterations", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if (
        min(
            args.iterations, args.num_envs, args.steps_per_iteration, args.save_interval
        )
        < 1
    ):
        parser.error(
            "iterations, num-envs, steps-per-iteration, save-interval must be positive"
        )
    if (
        args.learning_rate <= 0
        or min(args.teacher_warmup, args.teacher_mix_iterations) < 0
    ):
        parser.error("Invalid learning rate or teacher execution schedule")
    run(args)


if __name__ == "__main__":
    main()
