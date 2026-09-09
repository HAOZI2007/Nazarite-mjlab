"""Compute robust per-feature normalization statistics for Go2 SMP windows."""

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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--q-low", type=float, default=0.01)
    parser.add_argument("--q-high", type=float, default=0.99)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if not 0.0 <= args.q_low < args.q_high <= 1.0:
        raise ValueError("quantiles must satisfy 0 <= q_low < q_high <= 1")
    files = sorted(args.input_dir.expanduser().resolve().glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no NPZ files in {args.input_dir}")
    frames: list[np.ndarray] = []
    file_summaries = []
    window_shape: tuple[int, int] | None = None
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            if "windows" not in archive.files:
                continue
            windows = np.asarray(archive["windows"], dtype=np.float32)
        if windows.ndim != 3 or windows.shape[-1] != GO2_SMP_FEATURE_DIM:
            raise ValueError(
                f"{path}: expected [N,W,{GO2_SMP_FEATURE_DIM}], got {windows.shape}"
            )
        if not np.isfinite(windows).all():
            raise ValueError(f"{path}: non-finite windows")
        shape = (windows.shape[1], windows.shape[2])
        if window_shape is not None and shape != window_shape:
            raise ValueError(
                f"{path}: inconsistent window shape {shape} != {window_shape}"
            )
        window_shape = shape
        frames.append(windows.reshape(-1, GO2_SMP_FEATURE_DIM))
        file_summaries.append({"file": str(path), "windows": int(windows.shape[0])})
    if not frames or window_shape is None:
        raise ValueError(f"no window arrays found in {args.input_dir}")

    all_frames = np.concatenate(frames, axis=0).astype(np.float64)
    q_low = np.quantile(all_frames, args.q_low, axis=0).astype(np.float32)
    q_high = np.quantile(all_frames, args.q_high, axis=0).astype(np.float32)
    span = q_high - q_low
    tiny = span < 1.0e-6
    q_high[tiny] = q_low[tiny] + 1.0
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        q_low=q_low,
        q_high=q_high,
        quantile_low=np.asarray(args.q_low, dtype=np.float32),
        quantile_high=np.asarray(args.q_high, dtype=np.float32),
        feature_names=np.asarray(GO2_SMP_FEATURE_NAMES),
        feature_dims=np.asarray(GO2_SMP_FEATURE_DIMS, dtype=np.int32),
    )
    report = {
        "input_dir": str(args.input_dir.expanduser().resolve()),
        "output": str(output),
        "files": len(file_summaries),
        "windows": int(sum(item["windows"] for item in file_summaries)),
        "flattened_frames": len(all_frames),
        "window_size": window_shape[0],
        "feature_dim": window_shape[1],
        "q_low": args.q_low,
        "q_high": args.q_high,
        "near_constant_features": np.flatnonzero(tiny).tolist(),
        "minimum_span": float(span.min()),
        "maximum_span": float(span.max()),
        "files_detail": file_summaries,
    }
    report_path = output.with_suffix(".json")
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "files",
                    "windows",
                    "flattened_frames",
                    "window_size",
                    "feature_dim",
                    "near_constant_features",
                )
            },
            indent=2,
        )
    )
    print(f"Wrote: {output}")
    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()
