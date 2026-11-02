"""Evaluate XZ1 getup or legacy FR-Net, including standardized fall starts."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from nazarite.mdp.fr_net.frnet_terminations import stable_recovery_mask
from nazarite.mdp.xz1.getup import RandomDropReset

TASK_ID = "Nazarite-FRNet-Recovery-XZ1"


@dataclass(frozen=True)
class EvaluateConfig:
    checkpoint_file: Path
    task: str = TASK_ID
    num_envs: int = 64
    horizon_s: float = 10.0
    stable_duration_s: float = 1.0
    seed: int = 42
    device: str | None = None
    output_file: Path | None = None
    include_random_drops: bool = True


@torch.inference_mode()
def evaluate(cfg: EvaluateConfig) -> dict:
    """Keep one environment alive and evaluate each initial-pose family separately."""
    if not cfg.checkpoint_file.is_file():
        raise FileNotFoundError(cfg.checkpoint_file)
    if cfg.num_envs < 1 or not (0 < cfg.stable_duration_s <= cfg.horizon_s):
        raise ValueError(
            "Require positive num_envs and 0 < stable_duration_s <= horizon_s"
        )
    env_cfg = load_env_cfg(cfg.task, play=True)
    agent_cfg = load_rl_cfg(cfg.task)
    env_cfg.seed = cfg.seed
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.auto_reset = False
    env_cfg.episode_length_s = cfg.horizon_s
    env_cfg.terminations = {}
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    raw_env = ManagerBasedRlEnv(env_cfg, device=device)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    try:
        runner_cls = load_runner_cls(cfg.task)
        assert runner_cls is not None
        runner = runner_cls(env, asdict(agent_cfg), device=device)
        runner.load(
            str(cfg.checkpoint_file), load_cfg={"actor": True}, map_location=device
        )
        policy = runner.get_inference_policy(device=device)
        results = {}
        required = math.ceil(cfg.stable_duration_s / raw_env.step_dt)
        reset_cfg = raw_env.event_manager.get_term_cfg("reset_base")
        cases = ("left_side", "right_side", "back", "belly")
        if cfg.include_random_drops and isinstance(reset_cfg.func, RandomDropReset):
            cases += ("random_drop",)
        for case in cases:
            reset_cfg.params["fall_case"] = "mixed" if case == "random_drop" else case
            raw_env.reset()
            obs = env.get_observations()
            stable_steps = torch.zeros(cfg.num_envs, dtype=torch.long, device=device)
            first_time = torch.full((cfg.num_envs,), float("nan"), device=device)
            initial_xy = raw_env.scene["robot"].data.root_link_pos_w[:, :2].clone()
            max_displacement = torch.zeros(cfg.num_envs, device=device)
            max_ang_vel = torch.zeros_like(max_displacement)
            finite = torch.ones(cfg.num_envs, dtype=torch.bool, device=device)
            standing_samples = torch.zeros_like(max_displacement)
            standing_joint_speed_sq = torch.zeros_like(max_displacement)
            standing_ang_speed_sq = torch.zeros_like(max_displacement)
            standing_action_delta_sq = torch.zeros_like(max_displacement)
            for step in range(math.ceil(cfg.horizon_s / raw_env.step_dt)):
                with torch.inference_mode():
                    obs, _, _, _ = env.step(policy(obs))
                    robot = raw_env.scene["robot"]
                    finite &= torch.isfinite(robot.data.root_link_pos_w).all(-1)
                    finite &= torch.isfinite(robot.data.joint_pos).all(-1)
                    stable = (
                        stable_recovery_mask(
                            raw_env,
                            upright_gravity_z_max=-0.90,
                            min_relative_height=0.25,
                            max_angular_velocity=1.5,
                            max_joint_position_error=0.35,
                        )
                        & finite
                    )
                    stable_steps = torch.where(stable, stable_steps + 1, 0)
                    newly = (stable_steps >= required) & torch.isnan(first_time)
                    first_time[newly] = (step + 1) * raw_env.step_dt
                    displacement = torch.linalg.vector_norm(
                        robot.data.root_link_pos_w[:, :2] - initial_xy, dim=-1
                    )
                    angular_velocity = torch.linalg.vector_norm(
                        robot.data.root_link_ang_vel_b, dim=-1
                    )
                    max_displacement = torch.maximum(
                        max_displacement,
                        torch.nan_to_num(displacement, nan=float("inf")),
                    )
                    max_ang_vel = torch.maximum(
                        max_ang_vel,
                        torch.nan_to_num(angular_velocity, nan=float("inf")),
                    )
                    # Measure jitter while upright and raised, even if it fails
                    # the stricter contact/posture stability criteria.
                    standing = (
                        (robot.data.projected_gravity_b[:, 2] <= -0.90)
                        & (raw_env.scene["base_height_scan"].data.heights[:, 0] >= 0.25)
                        & finite
                    ).float()
                    standing_samples += standing
                    standing_joint_speed_sq += (
                        robot.data.joint_vel.square().mean(-1) * standing
                    )
                    standing_ang_speed_sq += angular_velocity.square() * standing
                    action_manager = raw_env.action_manager
                    standing_action_delta_sq += (
                        action_manager.action - action_manager.prev_action
                    ).square().mean(-1) * standing
            success = torch.isfinite(first_time) & finite
            times = first_time[success]
            results[case] = {
                "trials": cfg.num_envs,
                "successes": int(success.sum()),
                "success_rate": success.float().mean().item(),
                "final_stable_rate": ((stable_steps >= required) & finite)
                .float()
                .mean()
                .item(),
                "mean_time_to_stable_s": times.mean().item() if len(times) else None,
                "total_success_time_s": times.sum().item(),
                "invalid_state_rate": (~finite).float().mean().item(),
                "mean_max_xy_displacement_m": max_displacement[finite].mean().item()
                if finite.any()
                else None,
                "mean_max_angular_velocity_rad_s": max_ang_vel[finite].mean().item()
                if finite.any()
                else None,
            }
            count = standing_samples.sum()
            for name, total in (
                ("standing_joint_speed_rms_rad_s", standing_joint_speed_sq),
                ("standing_base_ang_speed_rms_rad_s", standing_ang_speed_sq),
                ("standing_action_delta_rms", standing_action_delta_sq),
            ):
                results[case][name] = (
                    (total.sum() / count).sqrt().item() if count > 0 else None
                )
            results[case]["standing_sample_count"] = int(count)
            print(
                f"{case:>12}: success={results[case]['success_rate']:.1%}, final_stable={results[case]['final_stable_rate']:.1%}"
            )
        successes = sum(result["successes"] for result in results.values())
        total_time = sum(result["total_success_time_s"] for result in results.values())
        metrics = {
            "task": cfg.task,
            "checkpoint": str(cfg.checkpoint_file),
            "seed": cfg.seed,
            "horizon_s": cfg.horizon_s,
            "stable_duration_s": cfg.stable_duration_s,
            "falls": results,
            "overall": {
                "success_rate": successes / (len(cases) * cfg.num_envs),
                "final_stable_rate": sum(
                    r["final_stable_rate"] for r in results.values()
                )
                / len(cases),
                "mean_time_to_stable_s": total_time / successes if successes else None,
            },
        }
        output = cfg.output_file or cfg.checkpoint_file.parent / "recovery_eval.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(f"Metrics saved to {output}")
        return metrics
    finally:
        env.close()


def main() -> None:
    evaluate(tyro.cli(EvaluateConfig))


if __name__ == "__main__":
    main()
