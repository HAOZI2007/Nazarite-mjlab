"""Read-only consistency audit for a Go2 SMP motion-window dataset."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from nazarite.config.train_config.train_algorithm.smp.prior.features import (
    GO2_SMP_FEATURE_DIM,
    GO2_SMP_FEATURE_DIMS,
    GO2_SMP_FEATURE_NAMES,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    files = sorted(args.input_dir.expanduser().resolve().glob("*.npz"))
    arrays = []
    file_reports = []
    expected_window = None
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            if "windows" not in archive.files:
                continue
            windows = np.asarray(archive["windows"], dtype=np.float32)
            speed = (
                float(np.asarray(archive["speed_scale"]).item())
                if "speed_scale" in archive.files
                else None
            )
            fps = (
                float(np.asarray(archive["fps"]).item())
                if "fps" in archive.files
                else None
            )
        if windows.ndim != 3 or windows.shape[-1] != GO2_SMP_FEATURE_DIM:
            raise ValueError(f"{path}: invalid windows shape {windows.shape}")
        if expected_window is None:
            expected_window = windows.shape[1]
        elif windows.shape[1] != expected_window:
            raise ValueError(f"{path}: inconsistent window size")
        if not np.isfinite(windows).all():
            raise ValueError(f"{path}: windows contain NaN or Inf")
        arrays.append(windows)
        file_reports.append(
            {
                "file": str(path),
                "windows": len(windows),
                "fps": fps,
                "speed_scale": speed,
            }
        )
    if not arrays:
        raise FileNotFoundError(f"no valid window NPZ files in {args.input_dir}")
    data = np.concatenate(arrays)
    flat = data.reshape(-1, GO2_SMP_FEATURE_DIM)
    report = {
        "valid": True,
        "files": len(file_reports),
        "windows": len(data),
        "window_size": int(data.shape[1]),
        "feature_dim": int(data.shape[2]),
        "feature_names": list(GO2_SMP_FEATURE_NAMES),
        "feature_dims": list(GO2_SMP_FEATURE_DIMS),
        "minimum": flat.min(axis=0).tolist(),
        "maximum": flat.max(axis=0).tolist(),
        "mean": flat.mean(axis=0).tolist(),
        "std": flat.std(axis=0).tolist(),
        "files_detail": file_reports,
    }
    if args.norm_stats:
        with np.load(
            args.norm_stats.expanduser().resolve(), allow_pickle=False
        ) as archive:
            q_low = np.asarray(archive["q_low"], dtype=np.float32)
            q_high = np.asarray(archive["q_high"], dtype=np.float32)
        if q_low.shape != (GO2_SMP_FEATURE_DIM,) or q_high.shape != q_low.shape:
            raise ValueError("normalization stats have incompatible shape")
        normalized = 2.0 * (flat - q_low) / (q_high - q_low) - 1.0
        report["normalization"] = {
            "finite": bool(np.isfinite(normalized).all()),
            "below_minus_one_fraction": float(np.mean(normalized < -1.0)),
            "above_plus_one_fraction": float(np.mean(normalized > 1.0)),
            "max_abs": float(np.abs(normalized).max()),
        }
    output = args.output.expanduser().resolve() if args.output else None
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "valid",
                    "files",
                    "windows",
                    "window_size",
                    "feature_dim",
                )
            },
            indent=2,
        )
    )
    if "normalization" in report:
        print(json.dumps(report["normalization"], indent=2))
    if output:
        print(f"Wrote: {output}")


if __name__ == "__main__":
    main()
