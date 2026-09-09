"""Build and physically filter a batch of 3DDogs-to-Go2 references.

The input is the CSV emitted by ``scan_3ddogs.py``.  Each selected valid run is
converted to the canonical MuJoCo frame, retargeted with the parallel
GQMR-style solver, evaluated, and optionally replayed with Nazarite's native
actuators.  The original 3DDogs files are read only.

Example::

    uv run python tools/smp_tools/dataset/build_reference_dataset.py \
      --manifest output/3ddogs_scan_0p5/selected_clips.csv \
      --output-dir output/smp_reference_dataset_anchor035 \
      --run-physics
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

try:
  from tools.smp_tools.data.inspect_3ddogs import extract_retarget_inputs, load_optical_trial
except ModuleNotFoundError:
  sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
  from tools.smp_tools.data.inspect_3ddogs import extract_retarget_inputs, load_optical_trial


AXIS_MAP_NEG_X_Z_Y = np.array(
  ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
  dtype=np.float64,
)


def _read_manifest(path: Path) -> list[dict[str, str]]:
  with path.open("r", newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
  required = {"source_path", "trial_name", "start_index", "end_index_exclusive"}
  if not rows or not required.issubset(rows[0]):
    raise ValueError(f"manifest must contain {sorted(required)}: {path}")
  return rows


def _clip_name(row: dict[str, str]) -> str:
  trial = row["trial_name"]
  start = int(row["start_index"])
  end = int(row["end_index_exclusive"])
  return f"{trial}_s{start:04d}_e{end:04d}"


def _write_canonical(row: dict[str, str], output_path: Path) -> dict[str, Any]:
  source_path = Path(row["source_path"]).expanduser().resolve()
  start = int(row["start_index"])
  end = int(row["end_index_exclusive"])
  trial = load_optical_trial(source_path)
  extracted = extract_retarget_inputs(trial)
  clip = slice(start, end)
  if start < 0 or end > trial.frame_numbers.size or end <= start:
    raise ValueError(f"invalid clip range [{start}, {end}) for {source_path}")
  if not np.all(extracted["valid_mask"][clip]):
    raise ValueError("manifest clip contains invalid frames")

  rotation = AXIS_MAP_NEG_X_Z_Y
  transform = lambda values: values @ rotation.T
  root_pos = transform(extracted["root_pos_raw"][clip])
  paw_pos = transform(extracted["paw_pos_raw"][clip])
  root_forward = transform(extracted["root_forward_raw"][clip])
  root_left = transform(extracted["root_left_raw"][clip])
  root_up = transform(extracted["root_up_raw"][clip])
  output_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    output_path,
    frame_numbers=extracted["frame_numbers"][clip],
    root_pos_mujoco=root_pos,
    root_forward_mujoco=root_forward,
    root_left_mujoco=root_left,
    root_up_mujoco=root_up,
    paw_pos_mujoco=paw_pos,
    fps=np.asarray(trial.fps, dtype=np.float64),
    paw_order=np.asarray(("FL", "FR", "RL", "RR")),
    axis_map=np.asarray("neg_x_z_y"),
  )
  return {
    "source": str(source_path),
    "fps": float(trial.fps),
    "start_index": start,
    "end_index_exclusive": end,
    "frames": end - start,
    "duration_s": float((end - start) / trial.fps),
    "axis_map": "neg_x_z_y",
  }


def _run_command(arguments: list[str]) -> None:
  result = subprocess.run(arguments, check=False, capture_output=True, text=True)
  if result.returncode != 0:
    raise RuntimeError(
      f"command failed ({result.returncode}): {' '.join(arguments)}\n{result.stdout}\n{result.stderr}"
    )


def _load_json(path: Path) -> dict[str, Any]:
  with path.open("r", encoding="utf-8") as stream:
    return json.load(stream)


def _process_one(
  row: dict[str, str],
  output_root: Path,
  project_root: Path,
  anchor_strength: float,
  run_physics: bool,
  kp_scale: float,
  kd_scale: float,
  control_decimation: int,
  max_action_saturation_fraction: float,
  min_physics_valid_fraction: float,
  max_joint_error_rad: float,
) -> dict[str, Any]:
  clip_name = _clip_name(row)
  clip_root = output_root / clip_name
  canonical_path = clip_root / "canonical" / "retarget_inputs_mujoco.npz"
  retarget_root = clip_root / "retarget"
  quality_path = retarget_root / "quality_report.json"
  physics_path = retarget_root / "physics_tracking_summary.json"
  canonical_summary = _write_canonical(row, canonical_path)
  high_quality_script = project_root / "tools/smp_tools/retargeting/high_quality_retarget.py"
  evaluate_script = project_root / "tools/smp_tools/quality/evaluate_retarget.py"
  physics_script = project_root / "tools/smp_tools/physics/track_go2_reference_physics.py"
  _run_command([
    sys.executable,
    str(high_quality_script),
    "--input", str(canonical_path),
    "--output-dir", str(retarget_root),
    "--contact-anchor-strength", str(anchor_strength),
  ])
  geometric_path = retarget_root / "go2_reference_geometric.npz"
  _run_command([
    sys.executable,
    str(evaluate_script),
    "--input", str(geometric_path),
    "--output", str(quality_path),
  ])
  quality = _load_json(quality_path)
  physics: dict[str, Any] | None = None
  if run_physics:
    _run_command([
      sys.executable,
      str(physics_script),
      "--input", str(geometric_path),
      "--output-dir", str(retarget_root),
      "--kp-scale", str(kp_scale),
      "--kd-scale", str(kd_scale),
      "--control-decimation", str(control_decimation),
      "--max-action-saturation-fraction", str(max_action_saturation_fraction),
    ])
    physics = _load_json(physics_path)
  accepted = bool(
    quality.get("valid_fraction", 0.0) >= 0.95
    and quality.get("joint_limit_violation_frames", 1) == 0
    and (
      not run_physics
      or (
        physics is not None
        and physics.get("physics_valid_fraction", 0.0) >= min_physics_valid_fraction
        and physics.get("maximum_joint_error_rad", float("inf")) <= max_joint_error_rad
        and physics.get("action_feasibility", {}).get(
          "equivalent_action_saturation_fraction", float("inf")
        ) <= max_action_saturation_fraction
      )
    )
  )
  return {
    "clip_name": clip_name,
    "accepted": accepted,
    "canonical_npz": str(canonical_path),
    "geometric_npz": str(geometric_path),
    "quality_report": str(quality_path),
    "physics_report": str(physics_path) if run_physics else None,
    "anchor_strength": anchor_strength,
    "canonical": canonical_summary,
    "quality": quality,
    "physics": physics,
  }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--anchor-strength", type=float, default=0.35)
  parser.add_argument("--run-physics", action="store_true")
  parser.add_argument(
    "--kp-scale", type=float, default=1.0,
    help="Additional diagnostic multiplier; 1.0 matches go2_cfg.py training",
  )
  parser.add_argument(
    "--kd-scale", type=float, default=1.0,
    help="Additional diagnostic multiplier; 1.0 matches go2_cfg.py training",
  )
  parser.add_argument(
    "--control-decimation", type=int, default=10,
    help="Must match the teacher environment; 10 is the current Nazarite value",
  )
  parser.add_argument(
    "--max-action-saturation-fraction", type=float, default=1.0,
    help="Diagnostic only; current smp RL config does not clip policy actions",
  )
  parser.add_argument(
    "--min-physics-valid-fraction", type=float, default=0.25,
    help="Open-loop stability floor for the actual low-gain controller",
  )
  parser.add_argument(
    "--max-joint-error-rad", type=float, default=1.0,
    help="Open-loop diagnostic ceiling; teacher PPO will close the remaining error",
  )
  parser.add_argument("--max-clips", type=int)
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
  args = _parse_args(argv)
  if not 0.0 <= args.anchor_strength <= 1.0:
    raise ValueError("--anchor-strength must be in [0, 1]")
  if args.max_clips is not None and args.max_clips <= 0:
    raise ValueError("--max-clips must be positive")
  manifest = args.manifest.expanduser().resolve()
  output_root = args.output_dir.expanduser().resolve()
  project_root = Path(__file__).resolve().parents[3]
  rows = _read_manifest(manifest)
  if args.max_clips is not None:
    rows = rows[: args.max_clips]
  output_root.mkdir(parents=True, exist_ok=True)
  results: list[dict[str, Any]] = []
  failures: list[dict[str, str]] = []
  for index, row in enumerate(rows, start=1):
    try:
      print(f"[{index}/{len(rows)}] processing {_clip_name(row)}", flush=True)
      results.append(_process_one(
        row=row,
        output_root=output_root,
        project_root=project_root,
        anchor_strength=args.anchor_strength,
        run_physics=args.run_physics,
        kp_scale=args.kp_scale,
        kd_scale=args.kd_scale,
        control_decimation=args.control_decimation,
        max_action_saturation_fraction=args.max_action_saturation_fraction,
        min_physics_valid_fraction=args.min_physics_valid_fraction,
        max_joint_error_rad=args.max_joint_error_rad,
      ))
    except (OSError, ValueError, RuntimeError) as exc:
      failures.append({"clip_name": _clip_name(row), "error": str(exc)})

  accepted = [result for result in results if result["accepted"]]
  report = {
    "manifest": str(manifest),
    "output_dir": str(output_root),
    "anchor_strength": args.anchor_strength,
    "run_physics": args.run_physics,
    "kp_scale": args.kp_scale,
    "kd_scale": args.kd_scale,
    "control_decimation": args.control_decimation,
    "max_action_saturation_fraction": args.max_action_saturation_fraction,
    "min_physics_valid_fraction": args.min_physics_valid_fraction,
    "max_joint_error_rad": args.max_joint_error_rad,
    "requested_clips": len(rows),
    "processed_clips": len(results),
    "accepted_clips": len(accepted),
    "failed_clips": len(failures),
    "total_accepted_duration_s": float(sum(result["canonical"]["duration_s"] for result in accepted)),
    # Keep every processed result for threshold audits.  The accepted manifest
    # remains intentionally small, while dataset_report.json must explain why
    # a candidate was rejected and support a later threshold-only selection
    # without rerunning MuJoCo.
    "processed": results,
    "accepted": accepted,
    "failures": failures,
  }
  report_path = output_root / "dataset_report.json"
  with report_path.open("w", encoding="utf-8") as stream:
    json.dump(report, stream, indent=2)
    stream.write("\n")
  manifest_path = output_root / "accepted_manifest.json"
  with manifest_path.open("w", encoding="utf-8") as stream:
    json.dump(accepted, stream, indent=2)
    stream.write("\n")
  print(json.dumps({key: report[key] for key in (
    "requested_clips", "processed_clips", "accepted_clips", "failed_clips", "total_accepted_duration_s"
  )}, indent=2))
  print(f"\nWrote: {report_path}")
  print(f"Wrote: {manifest_path}")


if __name__ == "__main__":
  main()
