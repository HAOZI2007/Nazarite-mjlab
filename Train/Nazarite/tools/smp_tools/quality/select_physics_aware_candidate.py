"""Select the most executable retarget candidate from quality/replay reports.

Candidate specification format::

    --candidate strength035=quality.json|physics_tracking_summary.json

The selector prioritizes physical validity and joint tracking, then uses
geometric contact/foot metrics as tie breakers.  It does not copy or mutate
any motion file; it only writes a decision report.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
  with path.open("r", encoding="utf-8") as stream:
    value = json.load(stream)
  if not isinstance(value, dict):
    raise ValueError(f"expected JSON object: {path}")
  return value


def _parse_candidate(spec: str) -> tuple[str, Path, Path]:
  try:
    name, paths = spec.split("=", 1)
    quality_path, physics_path = paths.split("|", 1)
  except ValueError as exc:
    raise ValueError(
      "candidate must use NAME=QUALITY_JSON|PHYSICS_JSON"
    ) from exc
  if not name:
    raise ValueError("candidate name must not be empty")
  return name, Path(quality_path), Path(physics_path)


def _score(
  quality: dict[str, Any],
  physics: dict[str, Any],
  min_physics_valid_fraction: float,
) -> tuple[float, dict[str, float], bool]:
  physics_valid = float(physics.get("physics_valid_fraction", 0.0))
  max_joint_error = float(physics.get("maximum_joint_error_rad", 999.0))
  foot_error = float(physics.get("mean_foot_target_error_m", 999.0))
  geometric_valid = float(quality.get("valid_fraction", 0.0))
  contact_slide = float(quality.get("contact_foot_speed_mean_m_s", 999.0))
  # Physical executability dominates.  The metric scales are intentionally
  # explicit so the selected candidate can be audited in the JSON report.
  components = {
    "physical_valid": 2.0 * physics_valid,
    "joint_tracking": -1.5 * max_joint_error,
    "foot_tracking": -2.0 * foot_error,
    "geometric_valid": 0.25 * geometric_valid,
    "contact_sliding": -0.5 * contact_slide,
  }
  score = float(sum(components.values()))
  feasible = physics_valid >= min_physics_valid_fraction
  return score, components, feasible


def select(candidates: list[str], min_physics_valid_fraction: float) -> dict[str, Any]:
  rows: list[dict[str, Any]] = []
  for specification in candidates:
    name, quality_path, physics_path = _parse_candidate(specification)
    quality_path = quality_path.expanduser().resolve()
    physics_path = physics_path.expanduser().resolve()
    quality = _read(quality_path)
    physics = _read(physics_path)
    score, components, feasible = _score(quality, physics, min_physics_valid_fraction)
    rows.append({
      "name": name,
      "quality_report": str(quality_path),
      "physics_report": str(physics_path),
      "score": score,
      "score_components": components,
      "feasible": feasible,
      "physics_valid_fraction": physics.get("physics_valid_fraction"),
      "maximum_joint_error_rad": physics.get("maximum_joint_error_rad"),
      "mean_foot_target_error_m": physics.get("mean_foot_target_error_m"),
      "geometric_valid_fraction": quality.get("valid_fraction"),
      "contact_foot_speed_mean_m_s": quality.get("contact_foot_speed_mean_m_s"),
    })
  feasible_rows = [row for row in rows if row["feasible"]]
  pool = feasible_rows if feasible_rows else rows
  if not pool:
    raise ValueError("at least one candidate is required")
  best = max(pool, key=lambda row: float(row["score"]))
  return {
    "minimum_physics_valid_fraction": min_physics_valid_fraction,
    "selected": best,
    "candidates": sorted(rows, key=lambda row: float(row["score"]), reverse=True),
    "selection_used_feasibility_filter": bool(feasible_rows),
  }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--candidate",
    action="append",
    required=True,
    help="NAME=QUALITY_JSON|PHYSICS_JSON; repeat for each candidate",
  )
  parser.add_argument("--min-physics-valid-fraction", type=float, default=0.75)
  parser.add_argument("--output", type=Path, required=True)
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
  args = _parse_args(argv)
  if not 0.0 <= args.min_physics_valid_fraction <= 1.0:
    raise ValueError("--min-physics-valid-fraction must be in [0, 1]")
  report = select(args.candidate, args.min_physics_valid_fraction)
  output = args.output.expanduser().resolve()
  output.parent.mkdir(parents=True, exist_ok=True)
  with output.open("w", encoding="utf-8") as stream:
    json.dump(report, stream, indent=2)
    stream.write("\n")
  print(json.dumps(report, indent=2))
  print(f"\nWrote: {output}")


if __name__ == "__main__":
  main()

