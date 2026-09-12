"""Prepare a Kine2Go geometric reference for 1x Nazarite physics tracking.

This stage keeps the source FPS and duration unchanged.  It applies a short
zero-phase weighted moving average to the twelve joint trajectories, clips
them to the Nazarite MuJoCo joint limits, recomputes exact Go2 FK, and trims a
late self-collision onset when it occurs inside a configurable tail window.
Contacts are then re-estimated from the recomputed FK feet instead of copying
Kine2Go foot labels from a different Go2 model.

Example:

    cd Train/Nazarite
    uv run python tools/smp_tools/kine2go/prepare_kine2go_reference.py \
      --input output/kine2go_adapted/solo8_crawl_slow/go2_reference_geometric.npz \
      --output-dir output/kine2go_prepared/solo8_crawl_slow
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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.smp_tools.kine2go.adapt_kine2go_to_go2 import (
    _estimate_contacts,
    _forward_feet,
    _free_qpos_address,
    _joint_ids_and_addresses,
    _rotation_matrices,
)
from tools.smp_tools.retargeting.inspect_go2_kinematics import (
    DEFAULT_GO2_XML,
    SMP_LEG_ORDER,
    _joint_names,
)


def _load_geometric_reference(path: Path) -> dict[str, np.ndarray]:
    required = {
        "fps",
        "qpos",
        "root_pos_mujoco",
        "root_quat_wxyz",
        "smp_leg_order",
        "smp_joint_order",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"geometric reference lacks arrays: {', '.join(missing)}")
        reference = {key: archive[key].copy() for key in archive.files}
    frame_count = reference["qpos"].shape[0]
    if reference["qpos"].ndim != 2 or frame_count < 3:
        raise ValueError("qpos must have at least three [frames, nq] rows")
    if reference["root_pos_mujoco"].shape != (frame_count, 3):
        raise ValueError("root_pos_mujoco must have shape [frames, 3]")
    if reference["root_quat_wxyz"].shape != (frame_count, 4):
        raise ValueError("root_quat_wxyz must have shape [frames, 4]")
    if tuple(str(x) for x in reference["smp_leg_order"].tolist()) != SMP_LEG_ORDER:
        raise ValueError(f"expected leg order {SMP_LEG_ORDER}")
    if tuple(str(x) for x in reference["smp_joint_order"].tolist()) != _joint_names():
        raise ValueError(f"expected joint order {_joint_names()}")
    if not all(
        np.isfinite(array).all()
        for array in reference.values()
        if np.issubdtype(array.dtype, np.number)
    ):
        raise ValueError("geometric reference contains non-finite values")
    return reference


def _weighted_zero_phase_smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Symmetric triangular FIR filtering with edge-value padding."""
    if window == 1:
        return values.astype(np.float64, copy=True)
    radius = window // 2
    weights = np.concatenate(
        (
            np.arange(1, radius + 2, dtype=np.float64),
            np.arange(radius, 0, -1, dtype=np.float64),
        )
    )
    weights /= np.sum(weights)
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    result = np.empty_like(values, dtype=np.float64)
    for dimension in range(values.shape[1]):
        result[:, dimension] = np.convolve(
            padded[:, dimension], weights, mode="valid"
        )
    return result


def _self_collision_indices(model: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
    data = mujoco.MjData(model)
    collision = np.zeros(len(qpos), dtype=np.bool_)
    for frame_index, pose in enumerate(qpos):
        data.qpos[:] = pose
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        mujoco.mj_collision(model, data)
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            body_a = int(model.geom_bodyid[contact.geom1])
            body_b = int(model.geom_bodyid[contact.geom2])
            if body_a != 0 and body_b != 0 and body_a != body_b:
                collision[frame_index] = True
                break
    return np.flatnonzero(collision)


def _maximum_derivatives(
    joint_position: np.ndarray, fps: float
) -> tuple[float, float]:
    if len(joint_position) < 2:
        return 0.0, 0.0
    velocity = np.gradient(joint_position, 1.0 / fps, axis=0, edge_order=2)
    acceleration = np.gradient(velocity, 1.0 / fps, axis=0, edge_order=2)
    return float(np.max(np.abs(velocity))), float(np.max(np.abs(acceleration)))


def prepare_reference(
    reference: dict[str, np.ndarray],
    model: mujoco.MjModel,
    smoothing_window: int,
    tail_collision_window: int,
    ground_clearance: float,
    contact_height_threshold: float,
    contact_speed_threshold: float,
    minimum_contact_frames: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    fps = float(np.asarray(reference["fps"]).item())
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps!r}")
    qpos = np.asarray(reference["qpos"], dtype=np.float64).copy()
    if qpos.shape[1] != model.nq:
        raise ValueError(f"qpos nq={qpos.shape[1]} does not match model nq={model.nq}")
    joint_ids, joint_addresses = _joint_ids_and_addresses(model)
    joint_limits = model.jnt_range[joint_ids]
    original_joint_position = qpos[:, joint_addresses].copy()
    smoothed_joint_position = _weighted_zero_phase_smooth(
        original_joint_position, smoothing_window
    )
    smoothed_joint_position = np.clip(
        smoothed_joint_position, joint_limits[:, 0], joint_limits[:, 1]
    )
    qpos[:, joint_addresses] = smoothed_joint_position

    root_pos = np.asarray(reference["root_pos_mujoco"], dtype=np.float64).copy()
    root_quat = np.asarray(reference["root_quat_wxyz"], dtype=np.float64).copy()
    collision_before = _self_collision_indices(model, qpos)
    tail_start = max(0, len(qpos) - tail_collision_window)
    tail_collisions = collision_before[collision_before >= tail_start]
    trim_end = int(tail_collisions[0]) if tail_collisions.size else len(qpos)
    if trim_end < 3:
        raise ValueError("tail collision trimming leaves fewer than three frames")

    qpos = qpos[:trim_end].copy()
    root_pos = root_pos[:trim_end].copy()
    root_quat = root_quat[:trim_end].copy()
    smoothed_joint_position = smoothed_joint_position[:trim_end].copy()
    source_frame_numbers = np.asarray(
        reference.get("source_frame_numbers", np.arange(len(reference["qpos"]))),
        dtype=np.int64,
    )[:trim_end]

    foot_position = _forward_feet(model, qpos)
    vertical_shift = ground_clearance - float(np.min(foot_position[..., 2]))
    free_address = _free_qpos_address(model)
    root_pos[:, 2] += vertical_shift
    qpos[:, free_address + 2] = root_pos[:, 2]
    foot_position = _forward_feet(model, qpos)
    collisions_after = _self_collision_indices(model, qpos)

    contact, contact_probability, estimated_ground_height = _estimate_contacts(
        foot_position,
        fps,
        contact_height_threshold,
        contact_speed_threshold,
        minimum_contact_frames,
    )
    root_rotation = _rotation_matrices(root_quat)
    foot_target_base = (foot_position - root_pos[:, None, :]) @ root_rotation
    joint_valid = np.all(
        (smoothed_joint_position >= joint_limits[:, 0])
        & (smoothed_joint_position <= joint_limits[:, 1]),
        axis=1,
    )
    collision_valid = np.ones(len(qpos), dtype=np.bool_)
    collision_valid[collisions_after] = False
    valid = joint_valid & collision_valid
    status = np.where(valid, "SUCCESS", "SELF_COLLISION").astype("U16")
    zeros_foot = np.zeros_like(foot_position)
    zeros_frame = np.zeros(len(qpos), dtype=np.float64)

    output = {
        "source_frame_numbers": source_frame_numbers,
        "fps": np.asarray(fps, dtype=np.float64),
        "smp_leg_order": np.asarray(SMP_LEG_ORDER),
        "smp_joint_order": np.asarray(_joint_names()),
        "root_pos_mujoco": root_pos,
        "root_quat_wxyz": root_quat,
        "root_rotation_mujoco": root_rotation,
        "joint_pos_rad": smoothed_joint_position,
        "qpos": qpos,
        "foot_target_world_m": foot_position,
        "foot_target_base_m": foot_target_base,
        "foot_position_world_m": foot_position,
        "foot_error_world_m": zeros_foot,
        "ik_iterations": np.zeros(len(qpos), dtype=np.int64),
        "ik_success": valid,
        "solver_status": status,
        "solver_residual_m": zeros_frame,
        "contact": contact,
        "contact_probability": contact_probability,
        "contact_anchor_error_world_m": zeros_foot,
    }
    speed_before, acceleration_before = _maximum_derivatives(
        original_joint_position, fps
    )
    speed_after, acceleration_after = _maximum_derivatives(
        smoothed_joint_position, fps
    )
    summary: dict[str, Any] = {
        "method": "1x_triangular_zero_phase_smoothing_then_nazarite_fk",
        "playback_rate": 1.0,
        "fps": fps,
        "input_frames": len(reference["qpos"]),
        "output_frames": len(qpos),
        "output_duration_s": (len(qpos) - 1) / fps,
        "smoothing": {
            "window_frames": smoothing_window,
            "window_duration_s": smoothing_window / fps,
            "max_joint_speed_before_rad_s": speed_before,
            "max_joint_speed_after_rad_s": speed_after,
            "max_joint_acceleration_before_rad_s2": acceleration_before,
            "max_joint_acceleration_after_rad_s2": acceleration_after,
        },
        "tail_collision_trim": {
            "search_window_frames": tail_collision_window,
            "collisions_before_indices": collision_before.tolist(),
            "trim_end_exclusive": trim_end,
            "trimmed_frames": len(reference["qpos"]) - trim_end,
            "collisions_after_indices": collisions_after.tolist(),
        },
        "ground_alignment": {
            "ground_clearance_m": ground_clearance,
            "constant_vertical_shift_m": vertical_shift,
            "minimum_foot_height_after_m": float(np.min(foot_position[..., 2])),
            "estimated_contact_ground_height_m": estimated_ground_height,
        },
        "contact": {
            "source": "Nazarite MuJoCo FK after smoothing",
            "height_threshold_m": contact_height_threshold,
            "speed_threshold_m_s": contact_speed_threshold,
            "minimum_contact_frames": minimum_contact_frames,
            "fraction_by_leg": {
                leg: float(value)
                for leg, value in zip(SMP_LEG_ORDER, np.mean(contact, axis=0))
            },
        },
        "valid_fraction": float(np.mean(valid)),
    }
    return output, summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--xml", type=Path, default=DEFAULT_GO2_XML)
    parser.add_argument("--smoothing-window", type=int, default=5)
    parser.add_argument("--tail-collision-window", type=int, default=12)
    parser.add_argument("--ground-clearance", type=float, default=0.005)
    parser.add_argument("--contact-height-threshold", type=float, default=0.03)
    parser.add_argument("--contact-speed-threshold", type=float, default=0.5)
    parser.add_argument("--minimum-contact-frames", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.smoothing_window < 1 or args.smoothing_window % 2 == 0:
        raise ValueError("--smoothing-window must be a positive odd integer")
    if args.tail_collision_window <= 0:
        raise ValueError("--tail-collision-window must be positive")
    if not np.isfinite(args.ground_clearance) or args.ground_clearance < 0.0:
        raise ValueError("--ground-clearance must be finite and non-negative")
    if (
        args.contact_height_threshold <= 0.0
        or args.contact_speed_threshold <= 0.0
        or args.minimum_contact_frames <= 0
    ):
        raise ValueError("contact thresholds and minimum frames must be positive")

    input_path = args.input.expanduser().resolve()
    xml_path = args.xml.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not xml_path.is_file():
        raise FileNotFoundError(xml_path)
    reference = _load_geometric_reference(input_path)
    output, summary = prepare_reference(
        reference=reference,
        model=mujoco.MjModel.from_xml_path(str(xml_path)),
        smoothing_window=args.smoothing_window,
        tail_collision_window=args.tail_collision_window,
        ground_clearance=args.ground_clearance,
        contact_height_threshold=args.contact_height_threshold,
        contact_speed_threshold=args.contact_speed_threshold,
        minimum_contact_frames=args.minimum_contact_frames,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "go2_reference_prepared.npz"
    np.savez_compressed(output_path, allow_pickle=False, **output)
    summary.update(
        {
            "source": str(input_path),
            "go2_xml": str(xml_path),
            "output": str(output_path),
        }
    )
    summary_path = output_dir / "prepare_summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote: {output_path}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
