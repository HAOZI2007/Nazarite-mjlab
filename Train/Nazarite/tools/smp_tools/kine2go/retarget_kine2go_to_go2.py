"""Retarget Kine2Go animal paw trajectories to Nazarite Go2 with bounded IK.

Kine2Go clips may contain animal or another robot's joint angles.  Those angles
must not be copied directly into the Go2 model.  This adapter discards the
source joint angles for retargeting, derives a canonical body frame from the
source root quaternion, and uses the source four-paw trajectories as task-space
targets for the existing contact-aware Go2 IK implementation.

The source inspected NPZ is produced by ``inspect_kine2go.py``.  The output is
the same geometric Go2 reference schema used by the SMP pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.smp_tools.retargeting.high_quality_retarget import main as retarget_main

SMP_LEG_ORDER = ("FL", "FR", "RL", "RR")


def _load_source(path: Path) -> dict[str, np.ndarray]:
    required = {
        "root_pos_mujoco",
        "root_quat_wxyz",
        "source_foot_pos_world",
        "fps",
        "smp_foot_order",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"Kine2Go NPZ lacks arrays: {', '.join(missing)}")
        source = {key: archive[key].copy() for key in required}
    frames = source["root_pos_mujoco"].shape[0]
    if source["root_pos_mujoco"].shape != (frames, 3):
        raise ValueError("root_pos_mujoco must have shape [frames, 3]")
    if source["root_quat_wxyz"].shape != (frames, 4):
        raise ValueError("root_quat_wxyz must have shape [frames, 4]")
    if source["source_foot_pos_world"].shape != (frames, 4, 3):
        raise ValueError("source_foot_pos_world must have shape [frames, 4, 3]")
    if tuple(str(x) for x in source["smp_foot_order"].tolist()) != SMP_LEG_ORDER:
        raise ValueError("source foot order must be FL, FR, RL, RR")
    if not all(np.isfinite(value).all() for value in source.values() if value.dtype.kind in "fiu"):
        raise ValueError("Kine2Go source contains non-finite values")
    return source


def _body_vectors(source: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Build forward/left vectors with physical scale for high_quality_retarget."""
    quaternions = source["root_quat_wxyz"].astype(np.float64)
    rotations = np.empty((len(quaternions), 3, 3), dtype=np.float64)
    for index, quaternion in enumerate(quaternions):
        norm = np.linalg.norm(quaternion)
        if norm <= 1.0e-9:
            raise ValueError("root quaternion has zero norm")
        mujoco.mju_quat2Mat(rotations[index].reshape(-1), quaternion / norm)

    feet = source["source_foot_pos_world"].astype(np.float64)
    front_center = 0.5 * (feet[:, 0] + feet[:, 1])
    rear_center = 0.5 * (feet[:, 2] + feet[:, 3])
    left_center = 0.5 * (feet[:, 0] + feet[:, 2])
    right_center = 0.5 * (feet[:, 1] + feet[:, 3])
    body_length = float(np.median(np.linalg.norm(front_center - rear_center, axis=1)))
    body_width = float(np.median(np.linalg.norm(left_center - right_center, axis=1)))
    if body_length <= 1.0e-4 or body_width <= 1.0e-4:
        raise ValueError("cannot estimate non-zero body length and width from paws")
    return rotations[:, :, 0] * body_length, rotations[:, :, 1] * body_width


def build_canonical(source: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    forward, left = _body_vectors(source)
    frames = source["root_pos_mujoco"].shape[0]
    return {
        "frame_numbers": np.arange(frames, dtype=np.int64),
        "root_pos_mujoco": source["root_pos_mujoco"],
        "root_forward_mujoco": forward,
        "root_left_mujoco": left,
        "paw_pos_mujoco": source["source_foot_pos_world"],
        "fps": source["fps"],
        "paw_order": np.asarray(SMP_LEG_ORDER),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--target-median-window",
        type=int,
        default=1,
        help="Odd-frame median filter for paw targets; 1 disables it",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    source = _load_source(input_path)
    canonical = build_canonical(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = output_dir / "kine2go_canonical_taskspace.npz"
    np.savez_compressed(canonical_path, allow_pickle=False, **canonical)
    print(json.dumps({
        "source": str(input_path),
        "frames": int(canonical["paw_pos_mujoco"].shape[0]),
        "fps": float(np.asarray(canonical["fps"]).item()),
        "canonical": str(canonical_path),
    }, indent=2))
    # Reuse the already-tested contact-aware DLS solver.  Its CLI writes the
    # final Go2 geometric reference next to the canonical intermediate file.
    retarget_args = [
        "--input", str(canonical_path),
        "--output-dir", str(output_dir / "go2_retarget"),
        "--target-median-window", str(args.target_median_window),
    ]
    if args.plot:
        retarget_args.append("--plot")
    retarget_main(retarget_args)


if __name__ == "__main__":
    main()
