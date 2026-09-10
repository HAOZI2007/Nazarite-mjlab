"""Decode a video and run an installed GQMR pose-backend plugin.

This is the first, deliberately narrow boundary of the video-to-SMP pipeline.
It writes generic GQMR keypoints only; it does not triangulate, retarget, run
physics validation, or add anything to ``tools/smp_dataset``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from gqmr.plugins.runner import run_pose_backend_plugin
from gqmr.pose import discover_pose_backends
from gqmr.sources.files import save_generic_keypoints_npz
from gqmr.sources.video import read_video_frames

FIXTURE_BACKENDS = {"dog-pose-fixture"}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input video")
    parser.add_argument("--output", type=Path, required=True, help="GQMR keypoint NPZ")
    parser.add_argument("--backend", required=True, help="Installed gqmr.pose_backends entry point")
    parser.add_argument(
        "--backend-config-json",
        default="{}",
        help="Strict JSON object passed to backend.load()",
    )
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--end-seconds", type=float)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument(
        "--allow-fixture-backend",
        action="store_true",
        help="Allow test-only fixture output; never use it for training",
    )
    return parser.parse_args(argv)


def _load_config(value: str) -> dict[str, Any]:
    try:
        config = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"--backend-config-json is invalid JSON: {error}") from error
    if not isinstance(config, dict):
        raise TypeError("--backend-config-json must contain a JSON object")
    return config


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"video does not exist: {source}")
    if output.suffix.lower() != ".npz":
        raise ValueError("--output must use the .npz suffix")
    if args.backend in FIXTURE_BACKENDS and not args.allow_fixture_backend:
        raise ValueError(
            "test fixture backend is disabled by default; pass "
            "--allow-fixture-backend only for interface smoke tests"
        )

    backends = discover_pose_backends()
    if args.backend not in backends:
        available = ", ".join(sorted(backends)) or "<none>"
        raise ValueError(
            f"pose backend {args.backend!r} is not installed; available: {available}"
        )
    info = backends[args.backend]().describe()
    config = _load_config(args.backend_config_json)
    video = read_video_frames(
        source,
        start_seconds=args.start_seconds,
        end_seconds=args.end_seconds,
        max_frames=args.max_frames,
    )
    result = run_pose_backend_plugin(
        args.backend,
        config,
        video,
        timeout=args.timeout_seconds,
    )
    if len(result.timestamps) != len(video.timestamps):
        raise ValueError(
            "pose backend changed the frame count; explicit timestamp alignment "
            "must be implemented before this output is accepted"
        )
    timestamp_error = np.max(np.abs(result.timestamps - video.timestamps))
    if not np.isfinite(timestamp_error) or timestamp_error > 1e-9:
        raise ValueError(
            f"pose/video timestamp mismatch: maximum error {timestamp_error:.9g} s"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    save_generic_keypoints_npz(output, result)
    source_is_fixture = bool(result.metadata.get("source_is_fixture", False))
    summary = {
        "input": str(source),
        "output": str(output),
        "backend": args.backend,
        "backend_package": info.package,
        "frames": len(result.timestamps),
        "duration_seconds": float(result.timestamps[-1] - result.timestamps[0]),
        "dimensions": result.dimensions,
        "keypoints": len(result.keypoint_names),
        "instances": len(result.instance_ids),
        "coordinate_frame": result.coordinate_frame,
        "valid_observation_fraction": float(np.mean(result.valid_mask)),
        "source_is_fixture": source_is_fixture,
        "training_eligible": bool(result.metadata.get("training_eligible", False)),
        "next_required_step": (
            "replace fixture with a real pose backend"
            if source_is_fixture
            else "audit pose quality; 2D output still requires calibrated 3D reconstruction"
        ),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote: {output}")


if __name__ == "__main__":
    main()
