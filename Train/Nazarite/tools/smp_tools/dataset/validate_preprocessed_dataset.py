"""Validate preprocessed Go2 references with the actual Nazarite controller.

The geometric dataset filter runs before time stretching and smoothing.  This
tool closes that gap by replaying every preprocessed NPZ with the same
controller and decimation used by the SMP teacher, then writing a validated
manifest for training.

Example::

    .venv/bin/python tools/smp_tools/dataset/validate_preprocessed_dataset.py \
      --manifest output/smp_reference_dataset_rate075_actual_controller/accepted_manifest_preprocessed.json \
      --output-dir output/smp_reference_dataset_rate075_actual_controller/validation_0p75 \
      --max-action-saturation-fraction 0.20 \
      --min-physics-valid-fraction 0.55 \
      --max-joint-error-rad 0.90
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _read_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
  with path.open("r", encoding="utf-8") as stream:
    value = json.load(stream)
  if isinstance(value, dict):
    clips = value.get("clips")
    if not isinstance(clips, list):
      raise ValueError(f"manifest dictionary must contain a 'clips' list: {path}")
    return value, clips
  if isinstance(value, list):
    return {"source_manifest": str(path)}, value
  raise ValueError(f"manifest must be a JSON list or dictionary: {path}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--kp-scale", type=float, default=1.0)
  parser.add_argument("--kd-scale", type=float, default=1.0)
  parser.add_argument("--control-decimation", type=int, default=10)
  parser.add_argument("--max-action-saturation-fraction", type=float, default=0.20)
  parser.add_argument("--min-physics-valid-fraction", type=float, default=0.55)
  parser.add_argument("--max-joint-error-rad", type=float, default=0.90)
  parser.add_argument("--max-clips", type=int)
  return parser.parse_args(argv)


def _run_one(
  entry: dict[str, Any],
  output_dir: Path,
  physics_script: Path,
  args: argparse.Namespace,
) -> dict[str, Any]:
  clip_name = str(entry.get("clip_name", "unknown"))
  input_path = Path(str(entry["preprocessed_npz"])).expanduser().resolve()
  if not input_path.is_file():
    raise FileNotFoundError(input_path)
  clip_output = output_dir / clip_name
  command = [
    sys.executable,
    str(physics_script),
    "--input", str(input_path),
    "--output-dir", str(clip_output),
    "--kp-scale", str(args.kp_scale),
    "--kd-scale", str(args.kd_scale),
    "--control-decimation", str(args.control_decimation),
    "--max-action-saturation-fraction", str(args.max_action_saturation_fraction),
  ]
  result = subprocess.run(command, check=False, capture_output=True, text=True)
  if result.returncode != 0:
    raise RuntimeError(
      f"physics replay failed for {clip_name} ({result.returncode})\n"
      f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
  summary_path = clip_output / "physics_tracking_summary.json"
  with summary_path.open("r", encoding="utf-8") as stream:
    physics = json.load(stream)
  saturation = physics.get("action_feasibility", {}).get(
    "equivalent_action_saturation_fraction", float("inf")
  )
  accepted = bool(
    physics.get("physics_valid_fraction", 0.0) >= args.min_physics_valid_fraction
    and physics.get("maximum_joint_error_rad", float("inf")) <= args.max_joint_error_rad
    and saturation <= args.max_action_saturation_fraction
  )
  enriched = dict(entry)
  enriched.update({
    "preprocessed_physics_report": str(summary_path),
    "preprocessed_physics_rollout": str(clip_output / "go2_physics_rollout.npz"),
    "preprocessed_physics": physics,
    "preprocessed_physics_accepted": accepted,
  })
  return enriched


def main(argv: Sequence[str] | None = None) -> None:
  args = _parse_args(argv)
  if args.kp_scale <= 0.0 or args.kd_scale <= 0.0 or args.control_decimation <= 0:
    raise ValueError("controller scales and decimation must be positive")
  if not 0.0 <= args.max_action_saturation_fraction <= 1.0:
    raise ValueError("--max-action-saturation-fraction must be in [0, 1]")
  if not 0.0 <= args.min_physics_valid_fraction <= 1.0:
    raise ValueError("--min-physics-valid-fraction must be in [0, 1]")
  if args.max_joint_error_rad <= 0.0:
    raise ValueError("--max-joint-error-rad must be positive")
  manifest_path = args.manifest.expanduser().resolve()
  source_manifest, entries = _read_manifest(manifest_path)
  if args.max_clips is not None:
    if args.max_clips <= 0:
      raise ValueError("--max-clips must be positive")
    entries = entries[: args.max_clips]
  output_dir = args.output_dir.expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  physics_script = Path(__file__).resolve().parents[3] / "tools/smp_tools/physics/track_go2_reference_physics.py"
  processed: list[dict[str, Any]] = []
  failures: list[dict[str, str]] = []
  for index, entry in enumerate(entries, start=1):
    clip_name = str(entry.get("clip_name", "unknown"))
    print(f"[{index}/{len(entries)}] validating {clip_name}", flush=True)
    try:
      processed.append(_run_one(entry, output_dir, physics_script, args))
    except (OSError, KeyError, ValueError, RuntimeError) as exc:
      failures.append({"clip_name": clip_name, "error": str(exc)})

  accepted = [entry for entry in processed if entry["preprocessed_physics_accepted"]]
  report = {
    "source_manifest": str(manifest_path),
    "source_manifest_metadata": source_manifest,
    "kp_scale": args.kp_scale,
    "kd_scale": args.kd_scale,
    "control_decimation": args.control_decimation,
    "max_action_saturation_fraction": args.max_action_saturation_fraction,
    "min_physics_valid_fraction": args.min_physics_valid_fraction,
    "max_joint_error_rad": args.max_joint_error_rad,
    "requested_clips": len(entries),
    "processed_clips": len(processed),
    "accepted_clips": len(accepted),
    "failed_clips": len(failures),
    "total_accepted_duration_s": float(sum(
      float(entry["preprocessed_physics"].get("simulation_frames", 1)) * 0.002
      for entry in accepted
    )),
    "processed": processed,
    "accepted": accepted,
    "failures": failures,
  }
  report_path = output_dir / "preprocessed_validation_report.json"
  with report_path.open("w", encoding="utf-8") as stream:
    json.dump(report, stream, indent=2)
    stream.write("\n")
  manifest_out = output_dir / "validated_manifest.json"
  with manifest_out.open("w", encoding="utf-8") as stream:
    json.dump({
      "source_manifest": str(manifest_path),
      "playback_rate": source_manifest.get("playback_rate"),
      "smoothing_window": source_manifest.get("smoothing_window"),
      "validation_report": str(report_path),
      "clips": accepted,
    }, stream, indent=2)
    stream.write("\n")
  print(json.dumps({key: report[key] for key in (
    "requested_clips", "processed_clips", "accepted_clips", "failed_clips",
  )}, indent=2))
  print(f"\nWrote: {report_path}")
  print(f"Wrote: {manifest_out}")


if __name__ == "__main__":
  main()
