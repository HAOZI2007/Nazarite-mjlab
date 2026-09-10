"""Audit forward/backward velocity coverage in SMP motion windows.

The tool is read-only.  It reports the local root velocity distribution used by
the prior, including per-clip direction classification.  The velocity feature
starts after root_pos, root_rot_6d, joint_pos and foot_pos: indices 33:36 are
``[vx, vy, vz]`` for the current 39-dimensional Go2 representation.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from nazarite.config.train_config.train_algorithm.smp.prior.features import (
    GO2_SMP_FEATURE_DIM,
)

VELOCITY_START = 3 + 6 + 12 + 12
VELOCITY_DIM = 3


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--deadband",
        type=float,
        default=0.3,
        help="Absolute vx below this value is counted as low-speed (default: 0.3).",
    )
    parser.add_argument(
        "--pure-fraction",
        type=float,
        default=0.9,
        help="Fraction required to classify a clip as purely forward/backward.",
    )
    return parser.parse_args(argv)


def _summary(values: np.ndarray, deadband: float) -> dict[str, Any]:
    if values.ndim != 2 or values.shape[1] != VELOCITY_DIM:
        raise ValueError(f"expected velocity array [N,3], got {values.shape}")
    vx = values[:, 0]
    bins = {
        "backward_fast": vx < -deadband,
        "low_speed": np.abs(vx) <= deadband,
        "forward_slow": (vx > deadband) & (vx < 1.0),
        "forward_medium": (vx >= 1.0) & (vx < 2.0),
        "forward_fast": vx >= 2.0,
    }
    return {
        "samples": int(len(values)),
        "vx": {
            "minimum": float(vx.min()),
            "maximum": float(vx.max()),
            "mean": float(vx.mean()),
            "median": float(np.median(vx)),
            "q01": float(np.quantile(vx, 0.01)),
            "q99": float(np.quantile(vx, 0.99)),
        },
        "vy": {
            "minimum": float(values[:, 1].min()),
            "maximum": float(values[:, 1].max()),
            "mean": float(values[:, 1].mean()),
        },
        "vz": {
            "minimum": float(values[:, 2].min()),
            "maximum": float(values[:, 2].max()),
            "mean": float(values[:, 2].mean()),
        },
        "fractions": {
            name: float(np.mean(mask)) for name, mask in bins.items()
        },
    }


def _classify(vx: np.ndarray, pure_fraction: float) -> str:
    forward = float(np.mean(vx > 0.0))
    backward = float(np.mean(vx < 0.0))
    if forward >= pure_fraction:
        return "forward_dominant"
    if backward >= pure_fraction:
        return "backward_dominant"
    return "mixed_or_low_speed"


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.deadband < 0.0:
        raise ValueError("--deadband must be non-negative")
    if not 0.5 <= args.pure_fraction <= 1.0:
        raise ValueError("--pure-fraction must be in [0.5, 1.0]")

    input_dir = args.input_dir.expanduser().resolve()
    files = sorted(input_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no NPZ files found in {input_dir}")

    all_velocity: list[np.ndarray] = []
    file_reports: list[dict[str, Any]] = []
    expected_window_size: int | None = None
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            if "windows" not in archive.files:
                continue
            windows = np.asarray(archive["windows"], dtype=np.float32)
        if windows.ndim != 3 or windows.shape[-1] != GO2_SMP_FEATURE_DIM:
            raise ValueError(f"{path}: invalid windows shape {windows.shape}")
        if expected_window_size is None:
            expected_window_size = int(windows.shape[1])
        elif windows.shape[1] != expected_window_size:
            raise ValueError(f"{path}: inconsistent window size")
        if not np.isfinite(windows).all():
            raise ValueError(f"{path}: windows contain NaN or Inf")

        velocity = windows[:, :, VELOCITY_START : VELOCITY_START + VELOCITY_DIM]
        velocity_flat = velocity.reshape(-1, VELOCITY_DIM)
        vx = velocity_flat[:, 0]
        report = _summary(velocity_flat, args.deadband)
        report.update(
            {
                "file": str(path),
                "windows": int(len(windows)),
                "classification": _classify(vx, args.pure_fraction),
            }
        )
        file_reports.append(report)
        all_velocity.append(velocity_flat)

    if not all_velocity:
        raise FileNotFoundError(f"no motion-window NPZ files found in {input_dir}")

    velocity = np.concatenate(all_velocity, axis=0)
    vx = velocity[:, 0]
    global_summary = _summary(velocity, args.deadband)
    report = {
        "valid": True,
        "input_dir": str(input_dir),
        "files": len(file_reports),
        "windows": int(sum(item["windows"] for item in file_reports)),
        "window_size": expected_window_size,
        "feature_dim": GO2_SMP_FEATURE_DIM,
        "velocity_feature_slice": [VELOCITY_START, VELOCITY_START + VELOCITY_DIM],
        "deadband_mps": args.deadband,
        "pure_fraction": args.pure_fraction,
        "global": global_summary,
        "clip_class_counts": {
            classification: sum(
                item["classification"] == classification for item in file_reports
            )
            for classification in (
                "forward_dominant",
                "backward_dominant",
                "mixed_or_low_speed",
            )
        },
        "clips": file_reports,
        "recommendation": {
            "supports_bidirectional_motion": bool(
                np.mean(vx > args.deadband) > 0.1
                and np.mean(vx < -args.deadband) > 0.1
            ),
            "has_low_speed_or_standing_data": bool(
                np.mean(np.abs(vx) <= args.deadband) > 0.01
            ),
            "warning": (
                "The dataset contains both directions but little low-speed data. "
                "A command range including zero should add standing, start, stop, "
                "and transition motions."
                if np.mean(vx > args.deadband) > 0.1
                and np.mean(vx < -args.deadband) > 0.1
                else "The dataset is predominantly one-directional; add or collect "
                "the opposite direction before bidirectional training."
            ),
        },
    }

    output = args.output.expanduser().resolve() if args.output else None
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")

    print(json.dumps(
        {
            "valid": report["valid"],
            "files": report["files"],
            "windows": report["windows"],
            "global": report["global"],
            "clip_class_counts": report["clip_class_counts"],
            "recommendation": report["recommendation"],
        },
        indent=2,
    ))
    if output:
        print(f"Wrote: {output}")


if __name__ == "__main__":
    main()
