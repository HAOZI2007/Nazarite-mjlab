"""Adapt an inspected Kine2Go clip to Nazarite's Go2 MuJoCo geometry.

The Kine2Go reference already contains Go2 joint angles, so this stage does
not solve a new cross-species IK problem.  It removes an optional leading
airborne-settling transient, translates the trajectory to a local horizontal
origin, applies one constant vertical ground alignment, and recomputes feet
with the exact Nazarite MuJoCo model.  No dynamics or actuator tracking is run
here.

Input is the safe NPZ written by ``inspect_kine2go.py``.  Output follows the
same geometric-reference schema used by the existing replay, quality, motion
preprocessing, and physics-tracking tools.

Example:

    cd Train/Nazarite
    uv run python tools/smp_tools/kine2go/adapt_kine2go_to_go2.py \
      --input output/kine2go_inspect/solo8_crawl_slow/reconstructed_motion.npz \
      --output-dir output/kine2go_adapted/solo8_crawl_slow
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

# File location: <Nazarite>/tools/smp_tools/kine2go/adapt_kine2go_to_go2.py
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.smp_tools.retargeting.inspect_go2_kinematics import (
    DEFAULT_GO2_XML,
    SMP_LEG_ORDER,
    _joint_names,
    _object_id,
)

EXPECTED_JOINT_ORDER = _joint_names()


def _load_inspected_motion(path: Path) -> dict[str, np.ndarray]:
    required = {
        "qpos",
        "fps",
        "root_pos_mujoco",
        "root_quat_wxyz",
        "joint_pos",
        "source_foot_pos_world",
        "smp_joint_order",
        "smp_foot_order",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"inspected Kine2Go NPZ lacks arrays: {', '.join(missing)}")
        motion = {key: archive[key].copy() for key in archive.files}

    qpos = motion["qpos"]
    if qpos.ndim != 2 or qpos.shape[0] < 3:
        raise ValueError("qpos must have at least three [frames, nq] rows")
    frame_count = qpos.shape[0]
    expected_shapes = {
        "root_pos_mujoco": (frame_count, 3),
        "root_quat_wxyz": (frame_count, 4),
        "joint_pos": (frame_count, 12),
        "source_foot_pos_world": (frame_count, 4, 3),
    }
    for key, shape in expected_shapes.items():
        if motion[key].shape != shape:
            raise ValueError(f"{key} must have shape {shape}, got {motion[key].shape}")
    if not all(
        np.isfinite(value).all()
        for value in motion.values()
        if np.issubdtype(value.dtype, np.number)
    ):
        raise ValueError("inspected Kine2Go NPZ contains non-finite values")

    joint_order = tuple(str(value) for value in motion["smp_joint_order"].tolist())
    foot_order = tuple(str(value) for value in motion["smp_foot_order"].tolist())
    if joint_order != EXPECTED_JOINT_ORDER:
        raise ValueError(
            f"expected canonical joint order {EXPECTED_JOINT_ORDER}, got {joint_order}"
        )
    if foot_order != SMP_LEG_ORDER:
        raise ValueError(f"expected foot order {SMP_LEG_ORDER}, got {foot_order}")
    return motion


def _finite_difference(values: np.ndarray, fps: float) -> np.ndarray:
    return np.gradient(values, 1.0 / fps, axis=0, edge_order=2)


def _valid_runs(mask: np.ndarray, minimum_length: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(np.asarray(mask, dtype=np.bool_)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= minimum_length:
                runs.append((start, index))
            start = None
    if start is not None and len(mask) - start >= minimum_length:
        runs.append((start, len(mask)))
    return runs


def _estimate_contacts(
    foot_pos: np.ndarray,
    fps: float,
    height_threshold: float,
    speed_threshold: float,
    minimum_contact_frames: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    foot_speed = np.linalg.norm(_finite_difference(foot_pos, fps), axis=-1)
    ground_height = float(np.percentile(foot_pos[..., 2], 10.0))
    height_score = np.clip(
        1.0 - (foot_pos[..., 2] - ground_height) / height_threshold, 0.0, 1.0
    )
    speed_score = np.clip(1.0 - foot_speed / speed_threshold, 0.0, 1.0)
    probability = height_score * speed_score
    candidates = (height_score >= 0.5) & (speed_score >= 0.5)
    contact = np.zeros_like(candidates, dtype=np.bool_)
    for foot_index in range(candidates.shape[1]):
        for start, end in _valid_runs(
            candidates[:, foot_index], minimum_contact_frames
        ):
            contact[start:end, foot_index] = True
    return contact, probability, ground_height


def _settled_start_frame(
    source_foot_pos: np.ndarray,
    contact: np.ndarray,
    ground_height: float,
    height_threshold: float,
    minimum_grounded_feet: int,
    minimum_settled_frames: int,
) -> int:
    initial_min_height = float(np.min(source_foot_pos[0, :, 2]))
    initially_airborne = initial_min_height > ground_height + height_threshold
    if not initially_airborne:
        return 0
    settled = np.count_nonzero(contact, axis=1) >= minimum_grounded_feet
    runs = _valid_runs(settled, minimum_settled_frames)
    return runs[0][0] if runs else 0


def _free_qpos_address(model: mujoco.MjModel) -> int:
    free_joints = [
        joint_id
        for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
    ]
    if len(free_joints) != 1:
        raise ValueError(f"expected exactly one Go2 free joint, found {len(free_joints)}")
    return int(model.jnt_qposadr[free_joints[0]])


def _joint_ids_and_addresses(
    model: mujoco.MjModel,
) -> tuple[np.ndarray, np.ndarray]:
    joint_ids = np.asarray(
        [
            _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in EXPECTED_JOINT_ORDER
        ],
        dtype=np.intp,
    )
    return joint_ids, np.asarray(model.jnt_qposadr[joint_ids], dtype=np.intp)


def _forward_feet(model: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
    site_ids = np.asarray(
        [
            _object_id(model, mujoco.mjtObj.mjOBJ_SITE, leg)
            for leg in SMP_LEG_ORDER
        ],
        dtype=np.intp,
    )
    data = mujoco.MjData(model)
    feet = np.empty((len(qpos), 4, 3), dtype=np.float64)
    for frame_index, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        feet[frame_index] = data.site_xpos[site_ids]
    if not np.isfinite(feet).all():
        raise ValueError("MuJoCo forward kinematics produced non-finite feet")
    return feet


def _rotation_matrices(quaternions_wxyz: np.ndarray) -> np.ndarray:
    result = np.empty((len(quaternions_wxyz), 3, 3), dtype=np.float64)
    for frame_index, quaternion in enumerate(quaternions_wxyz):
        mujoco.mju_quat2Mat(result[frame_index].reshape(-1), quaternion)
    return result


def adapt_motion(
    motion: dict[str, np.ndarray],
    model: mujoco.MjModel,
    ground_clearance: float,
    trim_leading_settle: bool,
    contact_height_threshold: float,
    contact_speed_threshold: float,
    minimum_contact_frames: int,
    minimum_grounded_feet: int,
    minimum_settled_frames: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    fps = float(np.asarray(motion["fps"]).item())
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps!r}")
    if motion["qpos"].shape[1] != model.nq:
        raise ValueError(
            f"input qpos has nq={motion['qpos'].shape[1]}, MuJoCo model has nq={model.nq}"
        )

    source_foot_full = np.asarray(
        motion["source_foot_pos_world"], dtype=np.float64
    )
    contact_full, probability_full, source_ground_height = _estimate_contacts(
        source_foot_full,
        fps,
        contact_height_threshold,
        contact_speed_threshold,
        minimum_contact_frames,
    )
    settle_start = _settled_start_frame(
        source_foot_full,
        contact_full,
        source_ground_height,
        contact_height_threshold,
        minimum_grounded_feet,
        minimum_settled_frames,
    )
    start_frame = settle_start if trim_leading_settle else 0
    if len(motion["qpos"]) - start_frame < 3:
        raise ValueError("leading-settle trimming leaves fewer than three frames")

    qpos = np.asarray(motion["qpos"][start_frame:], dtype=np.float64).copy()
    root_pos = np.asarray(
        motion["root_pos_mujoco"][start_frame:], dtype=np.float64
    ).copy()
    root_quat = np.asarray(
        motion["root_quat_wxyz"][start_frame:], dtype=np.float64
    ).copy()
    source_foot = source_foot_full[start_frame:].copy()
    contact = contact_full[start_frame:].copy()
    contact_probability = probability_full[start_frame:].copy()

    free_address = _free_qpos_address(model)
    horizontal_shift = -root_pos[0, :2]
    root_pos[:, :2] += horizontal_shift
    qpos[:, free_address : free_address + 2] = root_pos[:, :2]
    source_foot[:, :, :2] += horizontal_shift

    feet_before_ground_alignment = _forward_feet(model, qpos)
    lowest_foot_height = float(np.min(feet_before_ground_alignment[..., 2]))
    vertical_shift = ground_clearance - lowest_foot_height
    root_pos[:, 2] += vertical_shift
    qpos[:, free_address + 2] = root_pos[:, 2]
    source_foot[:, :, 2] += vertical_shift
    foot_position = _forward_feet(model, qpos)

    joint_ids, joint_addresses = _joint_ids_and_addresses(model)
    joint_pos = qpos[:, joint_addresses]
    joint_limits = model.jnt_range[joint_ids]
    joint_valid = np.all(
        (joint_pos >= joint_limits[:, 0]) & (joint_pos <= joint_limits[:, 1]), axis=1
    )
    fk_valid = np.isfinite(foot_position).all(axis=(1, 2))
    valid = joint_valid & fk_valid
    statuses = np.where(valid, "SUCCESS", "INVALID").astype("U16")
    root_rotation = _rotation_matrices(root_quat)
    foot_target_base = (foot_position - root_pos[:, None, :]) @ root_rotation
    zero_foot_error = np.zeros_like(foot_position)

    output = {
        "source_frame_numbers": np.arange(
            start_frame, start_frame + len(qpos), dtype=np.int64
        ),
        "fps": np.asarray(fps, dtype=np.float64),
        "smp_leg_order": np.asarray(SMP_LEG_ORDER),
        "smp_joint_order": np.asarray(EXPECTED_JOINT_ORDER),
        "root_pos_mujoco": root_pos,
        "root_quat_wxyz": root_quat,
        "root_rotation_mujoco": root_rotation,
        "joint_pos_rad": joint_pos,
        "qpos": qpos,
        # Kine2Go already supplies robot joint angles.  The corresponding
        # Nazarite FK feet are therefore both the target and solved positions.
        "foot_target_world_m": foot_position,
        "foot_target_base_m": foot_target_base,
        "foot_position_world_m": foot_position,
        "foot_error_world_m": zero_foot_error,
        "ik_iterations": np.zeros(len(qpos), dtype=np.int64),
        "ik_success": valid,
        "solver_status": statuses,
        "solver_residual_m": np.zeros(len(qpos), dtype=np.float64),
        "contact": contact,
        "contact_probability": contact_probability,
        "contact_anchor_error_world_m": np.zeros_like(foot_position),
        "source_foot_position_aligned_world_m": source_foot,
    }
    source_vs_fk = np.linalg.norm(source_foot - foot_position, axis=2)
    summary: dict[str, Any] = {
        "method": "preserve_kine2go_joint_angles_and_recompute_nazarite_fk",
        "input_frames": len(motion["qpos"]),
        "output_frames": len(qpos),
        "fps": fps,
        "output_duration_s": (len(qpos) - 1) / fps,
        "leading_settle": {
            "enabled": trim_leading_settle,
            "detected_start_frame": settle_start,
            "applied_start_frame": start_frame,
            "trimmed_frames": start_frame,
            "source_ground_height_m": source_ground_height,
            "minimum_grounded_feet": minimum_grounded_feet,
            "minimum_settled_frames": minimum_settled_frames,
        },
        "alignment": {
            "horizontal_shift_xy_m": horizontal_shift.tolist(),
            "ground_clearance_m": ground_clearance,
            "lowest_fk_foot_before_m": lowest_foot_height,
            "constant_vertical_shift_m": vertical_shift,
            "lowest_fk_foot_after_m": float(np.min(foot_position[..., 2])),
            "root_height_min_after_m": float(np.min(root_pos[:, 2])),
            "root_height_max_after_m": float(np.max(root_pos[:, 2])),
        },
        "validity": {
            "joint_limit_valid_fraction": float(np.mean(joint_valid)),
            "fk_valid_fraction": float(np.mean(fk_valid)),
            "valid_fraction": float(np.mean(valid)),
            "ground_penetration_frames": int(
                np.count_nonzero(np.any(foot_position[..., 2] < -1.0e-7, axis=1))
            ),
        },
        "contacts": {
            "height_threshold_m": contact_height_threshold,
            "speed_threshold_m_s": contact_speed_threshold,
            "minimum_contact_frames": minimum_contact_frames,
            "fraction_by_leg": {
                leg: float(value)
                for leg, value in zip(SMP_LEG_ORDER, np.mean(contact, axis=0))
            },
        },
        "source_vs_nazarite_fk_foot_error_m": {
            "mean": float(np.mean(source_vs_fk)),
            "median": float(np.median(source_vs_fk)),
            "p95": float(np.percentile(source_vs_fk, 95.0)),
            "max": float(np.max(source_vs_fk)),
        },
    }
    return output, summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--xml", type=Path, default=DEFAULT_GO2_XML)
    parser.add_argument("--ground-clearance", type=float, default=0.005)
    parser.add_argument("--keep-leading-settle", action="store_true")
    parser.add_argument("--contact-height-threshold", type=float, default=0.03)
    parser.add_argument("--contact-speed-threshold", type=float, default=0.5)
    parser.add_argument("--minimum-contact-frames", type=int, default=3)
    parser.add_argument("--minimum-grounded-feet", type=int, default=2)
    parser.add_argument("--minimum-settled-frames", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    positive_values = {
        "--contact-height-threshold": args.contact_height_threshold,
        "--contact-speed-threshold": args.contact_speed_threshold,
        "--minimum-contact-frames": args.minimum_contact_frames,
        "--minimum-grounded-feet": args.minimum_grounded_feet,
        "--minimum-settled-frames": args.minimum_settled_frames,
    }
    if any(value <= 0 for value in positive_values.values()):
        raise ValueError(f"contact/settling parameters must be positive: {positive_values}")
    if not 1 <= args.minimum_grounded_feet <= 4:
        raise ValueError("--minimum-grounded-feet must be between 1 and 4")
    if not np.isfinite(args.ground_clearance) or args.ground_clearance < 0.0:
        raise ValueError("--ground-clearance must be a finite non-negative value")

    input_path = args.input.expanduser().resolve()
    xml_path = args.xml.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not xml_path.is_file():
        raise FileNotFoundError(xml_path)

    output, summary = adapt_motion(
        motion=_load_inspected_motion(input_path),
        model=mujoco.MjModel.from_xml_path(str(xml_path)),
        ground_clearance=args.ground_clearance,
        trim_leading_settle=not args.keep_leading_settle,
        contact_height_threshold=args.contact_height_threshold,
        contact_speed_threshold=args.contact_speed_threshold,
        minimum_contact_frames=args.minimum_contact_frames,
        minimum_grounded_feet=args.minimum_grounded_feet,
        minimum_settled_frames=args.minimum_settled_frames,
    )
    summary.update(
        {
            "source": str(input_path),
            "go2_xml": str(xml_path),
            "output": str(output_dir / "go2_reference_geometric.npz"),
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "go2_reference_geometric.npz"
    np.savez_compressed(output_path, allow_pickle=False, **output)
    summary_path = output_dir / "adapt_summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote: {output_path}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
