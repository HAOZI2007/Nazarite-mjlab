"""Classify 3DDogs clips by anatomical forward velocity and transitions.

This is a read-only data audit.  It consumes the CSV produced by
``scan_3ddogs.py`` and writes lightweight manifests; it does not retarget or
modify the source recordings.  Velocity is measured in the dog's anatomical
forward/left/up basis, so a global optical-axis choice does not decide whether
the dog is moving forward or backward relative to its body.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.smp_tools.data.inspect_3ddogs import (
        extract_retarget_inputs,
        load_optical_trial,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from tools.smp_tools.data.inspect_3ddogs import (
        extract_retarget_inputs,
        load_optical_trial,
    )

AXIS_MAP_NEG_X_Z_Y = np.array(
    ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    dtype=np.float64,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--deadband",
        type=float,
        default=0.3,
        help="Absolute local vx below this value is low speed (default: 0.3 m/s).",
    )
    parser.add_argument(
        "--pure-fraction",
        type=float,
        default=0.9,
        help="Fraction required for a forward/backward dominant clip.",
    )
    parser.add_argument(
        "--transition-fraction",
        type=float,
        default=0.1,
        help="Minimum fraction for low-speed or direction transition labels.",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=5,
        help="Shorter runs are reported as too_short, not as behavior clips.",
    )
    return parser.parse_args(argv)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"source_path", "trial_name", "start_index", "end_index_exclusive"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"manifest must contain {sorted(required)}: {path}")
    return rows


def _unit(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norm <= 1.0e-9):
        raise ValueError("clip contains a degenerate anatomical direction")
    return values / norm


def _classify(
    vx: np.ndarray,
    deadband: float,
    pure_fraction: float,
    transition_fraction: float,
) -> tuple[str, list[str], dict[str, Any]]:
    low = np.abs(vx) <= deadband
    forward = vx > deadband
    backward = vx < -deadband
    direction_change = bool(np.any(forward[1:] & backward[:-1]) or np.any(backward[1:] & forward[:-1]))
    # A crossing through the deadband catches start/stop clips even when the
    # clip never contains both strictly positive and strictly negative motion.
    speed_transition = bool(
        np.any(low[1:] != low[:-1])
        and np.mean(low) >= transition_fraction
    )
    forward_fraction = float(np.mean(forward))
    backward_fraction = float(np.mean(backward))
    low_fraction = float(np.mean(low))

    labels: list[str] = []
    if forward_fraction >= pure_fraction:
        primary = "forward"
        labels.append("forward")
    elif backward_fraction >= pure_fraction:
        primary = "backward"
        labels.append("backward")
    elif low_fraction >= max(transition_fraction, 0.5):
        primary = "low_speed_or_standing"
        labels.append("low_speed_or_standing")
    elif direction_change:
        primary = "direction_transition"
        labels.append("direction_transition")
    elif forward_fraction > backward_fraction:
        primary = "forward_mixed"
        labels.append("forward_mixed")
    else:
        primary = "backward_mixed"
        labels.append("backward_mixed")

    if direction_change:
        labels.append("direction_transition")
    if speed_transition:
        labels.append("start_stop_or_speed_transition")
    if low_fraction >= transition_fraction:
        labels.append("contains_low_speed")

    metrics = {
        "forward_fraction": forward_fraction,
        "backward_fraction": backward_fraction,
        "low_speed_fraction": low_fraction,
        "direction_change": direction_change,
        "speed_transition": speed_transition,
    }
    return primary, sorted(set(labels)), metrics


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "source_path",
        "trial_name",
        "start_index",
        "end_index_exclusive",
        "primary_behavior",
        "labels",
        "duration_s",
        "vx_median_mps",
        "vx_min_mps",
        "vx_max_mps",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.deadband < 0.0:
        raise ValueError("--deadband must be non-negative")
    if not 0.5 <= args.pure_fraction <= 1.0:
        raise ValueError("--pure-fraction must be in [0.5, 1.0]")
    if not 0.0 <= args.transition_fraction <= 1.0:
        raise ValueError("--transition-fraction must be in [0, 1]")
    if args.min_frames < 2:
        raise ValueError("--min-frames must be at least 2")

    rows = _read_rows(args.manifest.expanduser().resolve())
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    classified: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for row in rows:
        source = Path(row["source_path"]).expanduser().resolve()
        try:
            trial = load_optical_trial(source)
            extracted = extract_retarget_inputs(trial)
            start = int(row["start_index"])
            end = int(row["end_index_exclusive"])
            if start < 0 or end <= start or end > len(trial.frame_numbers):
                raise ValueError(f"invalid clip range [{start}, {end})")
            clip = slice(start, end)
            if not np.all(extracted["valid_mask"][clip]):
                raise ValueError("clip contains invalid required markers")

            root = extracted["root_pos_raw"][clip] @ AXIS_MAP_NEG_X_Z_Y.T
            forward = extracted["root_forward_raw"][clip] @ AXIS_MAP_NEG_X_Z_Y.T
            forward_unit = _unit(forward)
            # root velocity projected onto anatomical forward is invariant to
            # the arbitrary global heading of the recording.
            if len(root) < 2:
                root_velocity = np.zeros_like(root)
            else:
                root_velocity = np.gradient(
                    root, 1.0 / trial.fps, axis=0, edge_order=1
                )
            vx = np.sum(root_velocity * forward_unit, axis=-1)
            if len(vx) < args.min_frames:
                primary = "too_short"
                labels = ["too_short"]
                metrics = {
                    "forward_fraction": float(np.mean(vx > args.deadband)),
                    "backward_fraction": float(np.mean(vx < -args.deadband)),
                    "low_speed_fraction": float(np.mean(np.abs(vx) <= args.deadband)),
                    "direction_change": False,
                    "speed_transition": False,
                }
            else:
                primary, labels, metrics = _classify(
                    vx,
                    args.deadband,
                    args.pure_fraction,
                    args.transition_fraction,
                )
            classified.append(
                {
                    "source_path": str(source),
                    "trial_name": row["trial_name"],
                    "start_index": start,
                    "end_index_exclusive": end,
                    "start_frame_num": int(trial.frame_numbers[start]),
                    "end_frame_num": int(trial.frame_numbers[end - 1]),
                    "frames": end - start,
                    "duration_s": float((end - start) / trial.fps),
                    "fps": float(trial.fps),
                    "primary_behavior": primary,
                    "labels": labels,
                    "vx_median_mps": float(np.median(vx)),
                    "vx_min_mps": float(np.min(vx)),
                    "vx_max_mps": float(np.max(vx)),
                    "vx_mean_mps": float(np.mean(vx)),
                    "velocity_metrics": metrics,
                }
            )
        except (OSError, ValueError) as exc:
            failures.append({"trial_name": row.get("trial_name", "unknown"), "error": str(exc)})

    counts: dict[str, int] = {}
    label_counts: dict[str, int] = {}
    for item in classified:
        primary = item["primary_behavior"]
        counts[primary] = counts.get(primary, 0) + 1
        for label in item["labels"]:
            label_counts[label] = label_counts.get(label, 0) + 1

    report = {
        "manifest": str(args.manifest.expanduser().resolve()),
        "output_dir": str(output_dir),
        "axis_map": "neg_x_z_y",
        "deadband_mps": args.deadband,
        "pure_fraction": args.pure_fraction,
        "transition_fraction": args.transition_fraction,
        "min_frames": args.min_frames,
        "requested_clips": len(rows),
        "classified_clips": len(classified),
        "failed_clips": len(failures),
        "primary_behavior_counts": counts,
        "label_counts": label_counts,
        "clips": classified,
        "failures": failures,
        "note": (
            "Classification uses anatomical local root vx. It is a behavior audit, "
            "not a substitute for visual and MuJoCo physical acceptance."
        ),
    }
    manifest_path = output_dir / "direction_behavior_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")

    _write_csv(output_dir / "all_classified_clips.csv", classified)
    for category in (
        "forward",
        "backward",
        "low_speed_or_standing",
        "direction_transition",
        "forward_mixed",
        "backward_mixed",
        "too_short",
    ):
        category_rows = [item for item in classified if item["primary_behavior"] == category]
        _write_csv(output_dir / f"{category}_clips.csv", category_rows)

    print(json.dumps(
        {
            "requested_clips": len(rows),
            "classified_clips": len(classified),
            "failed_clips": len(failures),
            "primary_behavior_counts": counts,
            "label_counts": label_counts,
        },
        indent=2,
    ))
    print(f"Wrote: {manifest_path}")


if __name__ == "__main__":
    main()
