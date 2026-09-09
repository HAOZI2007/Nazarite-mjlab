"""Compare two retarget quality JSON reports and print a compact A/B table."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any


METRICS: tuple[tuple[str, str], ...] = (
  ("valid_fraction", "higher"),
  ("foot_error_rmse_m", "lower"),
  ("foot_error_p95_m", "lower"),
  ("solver_residual_p95_m", "lower"),
  ("contact_foot_speed_mean_m_s", "lower"),
  ("contact_foot_speed_p95_m_s", "lower"),
  ("joint_speed_max_rad_s", "lower"),
  ("joint_acceleration_max_rad_s2", "lower"),
  ("joint_limit_violation_frames", "lower"),
  ("ground_penetration_frames", "lower"),
  ("self_collision_frames", "lower"),
)


def _load(path: Path) -> dict[str, Any]:
  with path.open("r", encoding="utf-8") as stream:
    value = json.load(stream)
  if not isinstance(value, dict):
    raise ValueError(f"quality report must be a JSON object: {path}")
  return value


def _winner(baseline: float, candidate: float, direction: str) -> str:
  if abs(candidate - baseline) <= 1.0e-12:
    return "tie"
  if direction == "higher":
    return "candidate" if candidate > baseline else "baseline"
  return "candidate" if candidate < baseline else "baseline"


def compare(baseline_path: Path, candidate_path: Path) -> dict[str, Any]:
  baseline = _load(baseline_path)
  candidate = _load(candidate_path)
  rows: list[dict[str, Any]] = []
  for metric, direction in METRICS:
    if metric not in baseline or metric not in candidate:
      continue
    baseline_value = float(baseline[metric])
    candidate_value = float(candidate[metric])
    rows.append({
      "metric": metric,
      "baseline": baseline_value,
      "candidate": candidate_value,
      "delta_candidate_minus_baseline": candidate_value - baseline_value,
      "preferred": _winner(baseline_value, candidate_value, direction),
      "direction": direction,
    })
  return {
    "baseline": str(baseline_path),
    "candidate": str(candidate_path),
    "rows": rows,
  }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--baseline", type=Path, required=True)
  parser.add_argument("--candidate", type=Path, required=True)
  parser.add_argument("--output", type=Path)
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
  args = _parse_args(argv)
  report = compare(args.baseline.expanduser().resolve(), args.candidate.expanduser().resolve())
  if args.output:
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
      json.dump(report, stream, indent=2)
      stream.write("\n")
  print(f"{'metric':38s} {'baseline':>14s} {'candidate':>14s} {'preferred':>10s}")
  print("-" * 82)
  for row in report["rows"]:
    print(
      f"{row['metric']:38s} {row['baseline']:14.6g} {row['candidate']:14.6g} {row['preferred']:>10s}"
    )
  if args.output:
    print(f"\nWrote: {args.output.expanduser().resolve()}")


if __name__ == "__main__":
  main()

