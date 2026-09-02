"""Evaluate FR-Net recovery success from fixed fallen initializations."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.sensor import TerrainHeightSensor
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class EvaluateConfig:
  """Settings for deterministic FR-Net fall-recovery evaluation."""

  checkpoint_file: str
  """Path to the local RSL-RL checkpoint to evaluate."""

  num_envs: int = 64
  """Number of randomized trials per fall category."""

  horizon_s: float = 5.0
  """Maximum simulated time allowed for one recovery trial."""

  stable_duration_s: float = 1.0
  """Continuous stable-standing duration required for a successful recovery."""

  min_base_height: float = 0.27
  """Minimum base-link height in metres for stable standing."""

  upright_gravity_z_max: float = -0.85
  """Maximum projected-gravity Z value; -1.0 is perfectly upright."""

  min_foot_contacts: int = 3
  """Minimum number of feet with ground force above the contact threshold."""

  contact_force_threshold: float = 5.0
  """Per-foot ground-contact threshold in newtons."""

  seed: int = 42
  device: str | None = None
  output_file: str | None = None
  """Optional JSON file receiving the aggregate and per-fall metrics."""


@dataclass(frozen=True)
class _FallCase:
  name: str
  roll: float
  pitch_magnitude: float
  pitch_sign: float


_FALL_CASES = (
  _FallCase("left_side", math.pi / 2.0, 0.0, 1.0),
  _FallCase("right_side", -math.pi / 2.0, 0.0, 1.0),
  _FallCase("back", 0.0, math.pi / 2.0, 1.0),
  _FallCase("belly", 0.0, math.pi / 2.0, -1.0),
)


def _stable_standing_mask(
  env: ManagerBasedRlEnv,
  cfg: EvaluateConfig,
) -> torch.Tensor:
  """Return environments currently upright, elevated, finite, and supported."""
  robot = env.scene["robot"]
  sensor = env.scene["feet_ground_contact"]
  force = sensor.data.force
  if force is None:
    raise RuntimeError("FR-Net evaluation requires feet_ground_contact force data")

  foot_contacts = torch.linalg.vector_norm(force, dim=-1) >= cfg.contact_force_threshold
  enough_contacts = foot_contacts.sum(dim=-1) >= cfg.min_foot_contacts
  base_pos = robot.data.root_link_pos_w
  finite_state = torch.isfinite(base_pos).all(dim=-1)
  upright = robot.data.projected_gravity_b[:, 2] <= cfg.upright_gravity_z_max
  height_sensor = env.scene["base_height_scan"]
  if not isinstance(height_sensor, TerrainHeightSensor):
    raise TypeError("FR-Net evaluation requires the base_height_scan sensor")
  elevated = height_sensor.data.heights[:, 0] >= cfg.min_base_height
  return finite_state & upright & elevated & enough_contacts


def _configure_case(
  env_cfg, case: _FallCase, cfg: EvaluateConfig, case_index: int
) -> None:
  """Configure one fixed fall family while retaining random yaw and dynamics."""
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed + case_index
  env_cfg.episode_length_s = cfg.horizon_s
  env_cfg.auto_reset = False
  env_cfg.terminations = {}
  env_cfg.observations["actor"].enable_corruption = False
  env_cfg.events.pop("push_robot", None)
  reset_params = env_cfg.events["reset_base"].params
  reset_params.update(
    {
      "roll_range": (case.roll, case.roll),
      "fallen_pitch_range": (case.pitch_magnitude, case.pitch_magnitude),
      "pitch_sign": case.pitch_sign,
    }
  )


def _evaluate_case(
  task_id: str,
  case: _FallCase,
  cfg: EvaluateConfig,
  device: str,
  case_index: int,
) -> dict[str, float | None]:
  """Evaluate one fall category and return vectorized trial statistics."""
  env_cfg = load_env_cfg(task_id, play=False)
  agent_cfg = load_rl_cfg(task_id)
  _configure_case(env_cfg, case, cfg, case_index)

  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
  try:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(cfg.checkpoint_file, load_cfg={"actor": True}, map_location=device)
    policy = runner.get_inference_policy(device=device)

    required_stable_steps = max(1, math.ceil(cfg.stable_duration_s / raw_env.step_dt))
    max_steps = math.ceil(cfg.horizon_s / raw_env.step_dt)
    stable_steps = torch.zeros(cfg.num_envs, dtype=torch.long, device=device)
    recovery_time_s = torch.full((cfg.num_envs,), float("nan"), device=device)
    obs = env.get_observations()

    for step in range(max_steps):
      with torch.inference_mode():
        actions = policy(obs)
      obs, _, _, _ = env.step(actions)

      stable = _stable_standing_mask(raw_env, cfg)
      stable_steps = torch.where(
        stable, stable_steps + 1, torch.zeros_like(stable_steps)
      )
      newly_recovered = stable_steps.eq(required_stable_steps) & torch.isnan(
        recovery_time_s
      )
      recovery_time_s[newly_recovered] = (step + 1) * raw_env.step_dt

    success = torch.isfinite(recovery_time_s)
    final_stable = stable_steps >= required_stable_steps
    mean_recovery_time_s = (
      recovery_time_s[success].mean().item() if success.any() else None
    )
    return {
      "success_rate": success.float().mean().item(),
      "final_stable_rate": final_stable.float().mean().item(),
      "mean_time_to_stable_s": mean_recovery_time_s,
      "trials": float(cfg.num_envs),
    }
  finally:
    env.close()


def _format_time(value: float | None) -> str:
  """Format an unavailable recovery time without emitting a JSON NaN."""
  return f"{value:.3f}s" if value is not None else "n/a"


def run_evaluate(task_id: str, cfg: EvaluateConfig) -> dict[str, object]:
  """Evaluate a checkpoint over left/right-side, back, and belly falls."""
  if cfg.num_envs < 1:
    raise ValueError("num_envs must be positive")
  if cfg.horizon_s <= 0.0 or cfg.stable_duration_s <= 0.0:
    raise ValueError("horizon_s and stable_duration_s must be positive")
  if cfg.min_foot_contacts not in (1, 2, 3, 4):
    raise ValueError("min_foot_contacts must be between 1 and 4")
  checkpoint = Path(cfg.checkpoint_file)
  if not checkpoint.is_file():
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")

  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  results = {
    case.name: _evaluate_case(task_id, case, cfg, device, index)
    for index, case in enumerate(_FALL_CASES)
  }
  per_case = list(results.values())
  success_rates = [metrics["success_rate"] for metrics in per_case]
  final_stable_rates = [metrics["final_stable_rate"] for metrics in per_case]
  recovery_times = [
    value
    for metrics in per_case
    if (value := metrics["mean_time_to_stable_s"]) is not None
  ]
  overall = {
    "mean_success_rate": sum(success_rates) / len(success_rates),
    "mean_final_stable_rate": sum(final_stable_rates) / len(final_stable_rates),
    "mean_time_to_stable_s": sum(recovery_times) / len(recovery_times)
    if recovery_times
    else None,
  }
  metrics: dict[str, object] = {
    "checkpoint": str(checkpoint),
    "num_trials_per_fall": cfg.num_envs,
    "horizon_s": cfg.horizon_s,
    "stable_duration_s": cfg.stable_duration_s,
    "falls": results,
    "overall": overall,
  }

  print("\nFR-Net Recovery Evaluation")
  for name, result in results.items():
    print(
      f"  {name:>10}: success={result['success_rate']:.1%}, "
      f"final_stable={result['final_stable_rate']:.1%}, "
      f"time={_format_time(result['mean_time_to_stable_s'])}"
    )
  print(
    f"  {'overall':>10}: success={overall['mean_success_rate']:.1%}, "
    f"final_stable={overall['mean_final_stable_rate']:.1%}, "
    f"time={_format_time(overall['mean_time_to_stable_s'])}"
  )

  if cfg.output_file:
    output_path = Path(cfg.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
      json.dump(metrics, file, indent=2)
    print(f"[INFO] Metrics saved to {output_path}")
  return metrics


def main() -> None:
  """Parse CLI arguments and evaluate one registered FR-Net task."""
  import nazarite  # noqa: F401

  task_id, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(
      ("Nazarite-FRNet-Recovery-Go2", "Nazarite-FRNet-Recovery-Terrain-Go2")
    ),
    add_help=False,
    return_unknown_args=True,
  )
  cfg = tyro.cli(
    EvaluateConfig,
    args=remaining_args,
    prog=sys.argv[0] + f" {task_id}",
  )
  run_evaluate(task_id, cfg)


if __name__ == "__main__":
  main()
