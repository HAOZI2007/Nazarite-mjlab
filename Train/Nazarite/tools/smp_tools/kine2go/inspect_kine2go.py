"""Audit and render one Kine2Go reference clip without modifying the dataset.

Kine2Go stores one reference frame as 61 float32 values.  This tool validates
that schema, reconstructs the zero/missing velocity channels from positions,
maps the reference joints to Nazarite's canonical ``FL, FR, RL, RR`` order,
and replays the resulting qpos through Nazarite's Go2 MuJoCo model.

The current Kine2Go dataset card documents a leg-major joint order, while the
released Solo8 clip inspected during integration is joint-type-major.  Auto
mode evaluates both layouts against the MuJoCo joint limits and source foot
positions instead of silently trusting either convention.

Example:

    cd Train/Nazarite
    uv run python tools/smp_tools/kine2go/inspect_kine2go.py \
      --input-dir tools/smp_dataset/kine2go/raw/data/solo8_crawl_slow \
      --output-dir output/kine2go_inspect/solo8_crawl_slow
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# MuJoCo needs its offscreen backend selected before importing the module.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

# File location: <Nazarite>/tools/smp_tools/kine2go/inspect_kine2go.py
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.smp_tools.physics.replay_go2_reference import render_reference
from tools.smp_tools.retargeting.inspect_go2_kinematics import DEFAULT_GO2_XML

EXPECTED_FEATURE_DIM = 61
SMP_LEG_ORDER = ("FL", "FR", "RL", "RR")
JOINT_KINDS = ("hip", "thigh", "calf")
SMP_JOINT_ORDER = tuple(
    f"{leg}_{joint_kind}_joint"
    for leg in SMP_LEG_ORDER
    for joint_kind in JOINT_KINDS
)
SMP_FOOT_SITES = SMP_LEG_ORDER

# This is the order currently stated on the Kine2Go dataset card.
DOCUMENTED_JOINT_ORDER = tuple(
    f"{leg}_{joint_kind}_joint"
    for leg in ("FR", "FL", "RR", "RL")
    for joint_kind in JOINT_KINDS
)

# Released Solo8 values are grouped as four hips, four thighs, four calves.
# The candidate is never selected merely by its name: auto mode validates it.
GENESIS_TYPE_MAJOR_JOINT_ORDER = tuple(
    f"{leg}_{joint_kind}_joint"
    for joint_kind in JOINT_KINDS
    for leg in ("FR", "FL", "RR", "RL")
)

# Kine2Go feet are FL, RL, FR, RR; SMP consistently uses FL, FR, RL, RR.
KINE2GO_FOOT_ORDER = ("FL", "RL", "FR", "RR")
FOOT_REORDER = np.asarray(
    [KINE2GO_FOOT_ORDER.index(leg) for leg in SMP_LEG_ORDER], dtype=np.intp
)


def _normalize_quaternions(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError("base quaternion contains a zero-norm frame")
    result /= norms
    for index in range(1, len(result)):
        if np.dot(result[index - 1], result[index]) < 0.0:
            result[index] *= -1.0
    return result


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    scalar = left[..., :1] * right[..., :1] - np.sum(
        left[..., 1:] * right[..., 1:], axis=-1, keepdims=True
    )
    vector = (
        left[..., :1] * right[..., 1:]
        + right[..., :1] * left[..., 1:]
        + np.cross(left[..., 1:], right[..., 1:])
    )
    return np.concatenate((scalar, vector), axis=-1)


def _relative_rotvec(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    previous_inverse = previous.copy()
    previous_inverse[..., 1:] *= -1.0
    relative = _normalize_quaternions(_quat_multiply(current, previous_inverse))
    relative[relative[:, 0] < 0.0] *= -1.0
    vector_norm = np.linalg.norm(relative[:, 1:], axis=1)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(relative[:, 0], 0.0, 1.0))
    scale = np.divide(
        angle,
        vector_norm,
        out=np.full_like(angle, 2.0),
        where=vector_norm > 1.0e-9,
    )
    return relative[:, 1:] * scale[:, None]


def _angular_velocity_world(quaternions: np.ndarray, fps: float) -> np.ndarray:
    count = len(quaternions)
    result = np.zeros((count, 3), dtype=np.float64)
    if count < 2:
        return result
    result[0] = _relative_rotvec(quaternions[1:2], quaternions[0:1])[0] * fps
    result[-1] = _relative_rotvec(quaternions[-1:], quaternions[-2:-1])[0] * fps
    if count > 2:
        result[1:-1] = _relative_rotvec(quaternions[2:], quaternions[:-2]) * (
            0.5 * fps
        )
    return result


def _object_id(
    model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str
) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"Go2 model has no {object_type.name} named {name!r}")
    return object_id


def _joint_qpos_addresses(model: mujoco.MjModel) -> np.ndarray:
    return np.asarray(
        [
            int(
                model.jnt_qposadr[
                    _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
            )
            for name in SMP_JOINT_ORDER
        ],
        dtype=np.intp,
    )


def _joint_ranges(model: mujoco.MjModel) -> np.ndarray:
    return np.asarray(
        [
            model.jnt_range[_object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in SMP_JOINT_ORDER
        ],
        dtype=np.float64,
    )


def _free_qpos_address(model: mujoco.MjModel) -> int:
    free_joint_ids = [
        joint_id
        for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
    ]
    if len(free_joint_ids) != 1:
        raise ValueError(f"expected exactly one Go2 free joint, found {len(free_joint_ids)}")
    return int(model.jnt_qposadr[free_joint_ids[0]])


def _canonical_joint_positions(
    raw_joint_positions: np.ndarray, source_order: tuple[str, ...]
) -> np.ndarray:
    if len(set(source_order)) != 12 or set(source_order) != set(SMP_JOINT_ORDER):
        raise ValueError("joint-order candidate must contain the 12 canonical joints once")
    indices = [source_order.index(name) for name in SMP_JOINT_ORDER]
    return raw_joint_positions[:, indices]


def _build_qpos(
    model: mujoco.MjModel,
    base_pos: np.ndarray,
    base_quat: np.ndarray,
    joint_pos: np.ndarray,
) -> np.ndarray:
    qpos = np.repeat(model.qpos0[None, :], len(base_pos), axis=0)
    free_address = _free_qpos_address(model)
    qpos[:, free_address : free_address + 3] = base_pos
    qpos[:, free_address + 3 : free_address + 7] = base_quat
    qpos[:, _joint_qpos_addresses(model)] = joint_pos
    return qpos


def _forward_kinematics(
    model: mujoco.MjModel, qpos: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    foot_site_ids = np.asarray(
        [
            _object_id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in SMP_FOOT_SITES
        ],
        dtype=np.intp,
    )
    data = mujoco.MjData(model)
    foot_pos = np.empty((len(qpos), 4, 3), dtype=np.float64)
    valid = np.ones(len(qpos), dtype=np.bool_)
    for frame_index, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        foot_pos[frame_index] = data.site_xpos[foot_site_ids]
        valid[frame_index] = np.isfinite(data.qpos).all() and np.isfinite(
            foot_pos[frame_index]
        ).all()
    return foot_pos, valid


def _evaluate_order_candidate(
    model: mujoco.MjModel,
    raw_joint_pos: np.ndarray,
    source_order: tuple[str, ...],
    base_pos: np.ndarray,
    base_quat: np.ndarray,
    source_foot_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    joint_pos = _canonical_joint_positions(raw_joint_pos, source_order)
    qpos = _build_qpos(model, base_pos, base_quat, joint_pos)
    fk_foot_pos, fk_valid = _forward_kinematics(model, qpos)
    ranges = _joint_ranges(model)
    joint_element_valid = (joint_pos >= ranges[:, 0]) & (joint_pos <= ranges[:, 1])
    joint_frame_valid = np.all(joint_element_valid, axis=1)
    foot_error = np.linalg.norm(fk_foot_pos - source_foot_pos, axis=2)
    report = {
        "source_joint_order": list(source_order),
        "canonical_reorder_indices": [
            source_order.index(name) for name in SMP_JOINT_ORDER
        ],
        "joint_limit_element_valid_fraction": float(np.mean(joint_element_valid)),
        "joint_limit_frame_valid_fraction": float(np.mean(joint_frame_valid)),
        "fk_valid_fraction": float(np.mean(fk_valid)),
        "source_vs_mujoco_foot_error_m": {
            "mean": float(np.mean(foot_error)),
            "median": float(np.median(foot_error)),
            "p95": float(np.quantile(foot_error, 0.95)),
            "max": float(np.max(foot_error)),
            "per_foot_mean": {
                leg: float(value)
                for leg, value in zip(SMP_LEG_ORDER, np.mean(foot_error, axis=0))
            },
        },
    }
    return joint_pos, qpos, fk_foot_pos, report


def _select_joint_order(
    model: mujoco.MjModel,
    raw_joint_pos: np.ndarray,
    base_pos: np.ndarray,
    base_quat: np.ndarray,
    source_foot_pos: np.ndarray,
    requested: str,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    candidates = {
        "documented_leg_major": DOCUMENTED_JOINT_ORDER,
        "genesis_type_major": GENESIS_TYPE_MAJOR_JOINT_ORDER,
    }
    evaluated: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]] = {}
    for name, source_order in candidates.items():
        evaluated[name] = _evaluate_order_candidate(
            model,
            raw_joint_pos,
            source_order,
            base_pos,
            base_quat,
            source_foot_pos,
        )

    if requested == "auto":
        selected = min(
            candidates,
            key=lambda name: (
                -evaluated[name][3]["joint_limit_frame_valid_fraction"],
                -evaluated[name][3]["joint_limit_element_valid_fraction"],
                evaluated[name][3]["source_vs_mujoco_foot_error_m"]["median"],
            ),
        )
    else:
        selected = requested
    joint_pos, qpos, fk_foot_pos, _ = evaluated[selected]
    selection_report = {
        "requested": requested,
        "selected": selected,
        "selection_rule": (
            "maximize joint-limit-valid frames/elements, then minimize median "
            "source-vs-MuJoCo foot error"
        ),
        "candidates": {name: values[3] for name, values in evaluated.items()},
    }
    return selected, joint_pos, qpos, fk_foot_pos, selection_report


def _load_clip(input_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    motion_path = input_dir / "motion.npy"
    metadata_path = input_dir / "clip.json"
    if not motion_path.is_file():
        raise FileNotFoundError(motion_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    motion = np.load(motion_path, allow_pickle=False)
    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    if not isinstance(metadata, dict):
        raise TypeError(f"clip metadata must be a JSON object: {metadata_path}")
    if motion.ndim != 2 or motion.shape[1] != EXPECTED_FEATURE_DIM:
        raise ValueError(
            f"motion.npy must have shape [T,{EXPECTED_FEATURE_DIM}], got {motion.shape}"
        )
    if len(motion) < 3:
        raise ValueError("motion.npy needs at least three frames for velocity reconstruction")
    if not np.issubdtype(motion.dtype, np.floating):
        raise TypeError(f"motion.npy must use a floating dtype, got {motion.dtype}")
    if not np.isfinite(motion).all():
        raise ValueError("motion.npy contains non-finite values")
    return motion, metadata


def _plot_trajectories(
    output_path: Path,
    fps: float,
    base_pos: np.ndarray,
    base_lin_vel: np.ndarray,
    source_foot_pos: np.ndarray,
    fk_foot_pos: np.ndarray,
    min_base_height: float,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time = np.arange(len(base_pos), dtype=np.float64) / fps
    horizontal_speed = np.linalg.norm(base_lin_vel[:, :2], axis=1)
    foot_error = np.linalg.norm(fk_foot_pos - source_foot_pos, axis=2)
    colors = {"FL": "tab:blue", "FR": "tab:orange", "RL": "tab:green", "RR": "tab:red"}

    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for axis_index, axis_name in enumerate(("x", "y", "z")):
        axes[0, 0].plot(time, base_pos[:, axis_index], label=axis_name)
    axes[0, 0].axhline(
        min_base_height, color="tab:red", linestyle=":", label="minimum base height"
    )
    axes[0, 0].set_title("Base position (world frame)")
    axes[0, 0].set_ylabel("position (m)")
    axes[0, 0].legend(ncols=2)

    for axis_index, axis_name in enumerate(("vx", "vy", "vz")):
        axes[0, 1].plot(time, base_lin_vel[:, axis_index], label=axis_name)
    axes[0, 1].plot(time, horizontal_speed, color="black", linewidth=1.3, label="|vxy|")
    axes[0, 1].set_title("Reconstructed base velocity")
    axes[0, 1].set_ylabel("velocity (m/s)")
    axes[0, 1].legend(ncols=2)

    for foot_index, leg in enumerate(SMP_LEG_ORDER):
        color = colors[leg]
        axes[1, 0].plot(
            time,
            source_foot_pos[:, foot_index, 2],
            color=color,
            label=f"{leg} source",
        )
        axes[1, 0].plot(
            time,
            fk_foot_pos[:, foot_index, 2],
            color=color,
            linestyle="--",
            alpha=0.8,
            label=f"{leg} MuJoCo FK",
        )
        axes[1, 1].plot(time, foot_error[:, foot_index], color=color, label=leg)
    axes[1, 0].set_title("Foot height: source (solid) vs MuJoCo FK (dashed)")
    axes[1, 0].set_ylabel("world z (m)")
    axes[1, 0].legend(ncols=2, fontsize=8)
    axes[1, 1].set_title("Source-vs-MuJoCo foot-position difference")
    axes[1, 1].set_ylabel("3D error (m)")
    axes[1, 1].legend(ncols=2)

    for axis in axes.flat:
        axis.set_xlabel("time (s)")
        axis.grid(alpha=0.3)
    figure.suptitle("Kine2Go reference audit")
    figure.tight_layout()
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def inspect_clip(
    input_dir: Path,
    output_dir: Path,
    xml_path: Path,
    min_base_height: float,
    joint_order: str,
    render_stride: int,
    width: int,
    height: int,
    render: bool,
    plot: bool,
) -> dict[str, Any]:
    motion, metadata = _load_clip(input_dir)
    fps = float(metadata.get("fps", 0.0))
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"clip.json must contain a positive fps, got {fps!r}")

    raw_dof_pos = np.asarray(motion[:, 0:18], dtype=np.float64)
    stored_dof_vel = np.asarray(motion[:, 18:36], dtype=np.float64)
    source_foot_pos = np.asarray(motion[:, 36:48], dtype=np.float64).reshape(-1, 4, 3)
    source_foot_pos = source_foot_pos[:, FOOT_REORDER]
    base_pos = np.asarray(motion[:, 48:51], dtype=np.float64)
    raw_base_quat = np.asarray(motion[:, 51:55], dtype=np.float64)
    stored_base_lin_vel = np.asarray(motion[:, 55:58], dtype=np.float64)
    stored_base_ang_vel = np.asarray(motion[:, 58:61], dtype=np.float64)
    base_quat = _normalize_quaternions(raw_base_quat)

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    selected_order, joint_pos, qpos, fk_foot_pos, order_report = _select_joint_order(
        model,
        raw_dof_pos[:, 6:18],
        base_pos,
        base_quat,
        source_foot_pos,
        joint_order,
    )

    dt = 1.0 / fps
    joint_vel = np.gradient(joint_pos, dt, axis=0, edge_order=2)
    base_lin_vel = np.gradient(base_pos, dt, axis=0, edge_order=2)
    base_ang_vel = _angular_velocity_world(base_quat, fps)

    ranges = _joint_ranges(model)
    joint_limit_valid = np.all(
        (joint_pos >= ranges[:, 0]) & (joint_pos <= ranges[:, 1]), axis=1
    )
    quaternion_norm = np.linalg.norm(raw_base_quat, axis=1)
    quaternion_valid = np.abs(quaternion_norm - 1.0) <= 1.0e-3
    root_height_valid = base_pos[:, 2] >= min_base_height
    fk_valid = np.isfinite(fk_foot_pos).all(axis=(1, 2))
    frame_valid = joint_limit_valid & quaternion_valid & root_height_valid & fk_valid
    foot_error = np.linalg.norm(fk_foot_pos - source_foot_pos, axis=2)

    metadata_frames = metadata.get("n_frames")
    metadata_duration = metadata.get("duration_s")
    actual_duration = len(motion) / fps
    duplicated_base_error = np.max(np.abs(raw_dof_pos[:, :3] - base_pos))
    warnings: list[str] = []
    if metadata_frames is not None and int(metadata_frames) != len(motion):
        warnings.append(
            f"clip.json n_frames={metadata_frames} but motion.npy has {len(motion)} frames"
        )
    if metadata_duration is not None and not np.isclose(
        float(metadata_duration), actual_duration, atol=1.0 / fps
    ):
        warnings.append(
            f"clip.json duration_s={metadata_duration} but motion.npy/fps gives "
            f"{actual_duration:.6f} s"
        )
    if not np.any(stored_dof_vel):
        warnings.append("stored dofs_velocity is entirely zero; reconstructed values are used")
    if not np.any(stored_base_lin_vel):
        warnings.append("stored base_lin_vel is entirely zero; reconstructed values are used")
    if not np.any(stored_base_ang_vel):
        warnings.append("stored base_ang_vel is entirely zero; reconstructed values are used")
    if selected_order != "documented_leg_major":
        warnings.append(
            "auto mapping rejected the dataset-card joint order and selected "
            f"{selected_order}"
        )
    if not np.all(root_height_valid):
        warnings.append(
            f"{np.count_nonzero(~root_height_valid)} frames are below the "
            f"{min_base_height:.3f} m base-height audit threshold"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "reconstructed_motion.npz",
        qpos=qpos.astype(np.float32),
        fps=np.asarray(fps, dtype=np.float32),
        root_pos_mujoco=base_pos.astype(np.float32),
        root_quat_wxyz=base_quat.astype(np.float32),
        root_lin_vel_world=base_lin_vel.astype(np.float32),
        root_ang_vel_world=base_ang_vel.astype(np.float32),
        joint_pos=joint_pos.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        source_foot_pos_world=source_foot_pos.astype(np.float32),
        mujoco_foot_pos_world=fk_foot_pos.astype(np.float32),
        frame_valid=frame_valid,
        ik_success=frame_valid,
        smp_joint_order=np.asarray(SMP_JOINT_ORDER),
        smp_foot_order=np.asarray(SMP_LEG_ORDER),
        selected_source_joint_order=np.asarray(
            order_report["candidates"][selected_order]["source_joint_order"]
        ),
    )

    if plot:
        _plot_trajectories(
            output_dir / "trajectories.png",
            fps,
            base_pos,
            base_lin_vel,
            source_foot_pos,
            fk_foot_pos,
            min_base_height,
        )
    rendered_frames = 0
    if render:
        rendered_frames = render_reference(
            model=model,
            reference={
                "qpos": qpos,
                "fps": np.asarray(fps),
                "root_pos_mujoco": base_pos,
                "ik_success": frame_valid,
            },
            output_path=output_dir / "kinematic_replay.gif",
            output_fps=max(1, round(fps / render_stride)),
            width=width,
            height=height,
            stride=render_stride,
            follow_camera=True,
        )

    summary: dict[str, Any] = {
        "source": {
            "input_dir": str(input_dir),
            "motion": str(input_dir / "motion.npy"),
            "metadata": str(input_dir / "clip.json"),
            "go2_xml": str(xml_path),
        },
        "metadata": metadata,
        "motion": {
            "shape": list(motion.shape),
            "dtype": str(motion.dtype),
            "finite": bool(np.isfinite(motion).all()),
            "actual_frames": len(motion),
            "metadata_frames": metadata_frames,
            "metadata_frame_count_matches": (
                metadata_frames is None or int(metadata_frames) == len(motion)
            ),
            "fps": fps,
            "actual_duration_s": actual_duration,
            "metadata_duration_s": metadata_duration,
            "duplicated_base_position_max_error": float(duplicated_base_error),
        },
        "coordinate_convention": {
            "source": "Kine2Go world frame: x forward, y left, z up",
            "mujoco": "Nazarite Go2 world frame: x forward, y left, z up",
            "axis_remap_applied": False,
            "quaternion": "wxyz",
        },
        "joint_order_selection": order_report,
        "velocity_reconstruction": {
            "method": "second-order numpy gradient; quaternion relative rotation vector",
            "angular_velocity_frame": "world",
            "stored_dof_velocity_nonzero_fraction": float(
                np.count_nonzero(stored_dof_vel) / stored_dof_vel.size
            ),
            "stored_base_linear_velocity_nonzero_fraction": float(
                np.count_nonzero(stored_base_lin_vel) / stored_base_lin_vel.size
            ),
            "stored_base_angular_velocity_nonzero_fraction": float(
                np.count_nonzero(stored_base_ang_vel) / stored_base_ang_vel.size
            ),
            "derived_base_velocity_mean_xyz_m_s": np.mean(
                base_lin_vel, axis=0
            ).tolist(),
            "derived_horizontal_speed_mean_m_s": float(
                np.mean(np.linalg.norm(base_lin_vel[:, :2], axis=1))
            ),
            "derived_horizontal_speed_max_m_s": float(
                np.max(np.linalg.norm(base_lin_vel[:, :2], axis=1))
            ),
        },
        "audit": {
            "min_base_height_threshold_m": min_base_height,
            "base_height_min_m": float(np.min(base_pos[:, 2])),
            "base_height_max_m": float(np.max(base_pos[:, 2])),
            "root_height_valid_fraction": float(np.mean(root_height_valid)),
            "joint_limit_valid_fraction": float(np.mean(joint_limit_valid)),
            "quaternion_valid_fraction": float(np.mean(quaternion_valid)),
            "fk_valid_fraction": float(np.mean(fk_valid)),
            "all_checks_valid_fraction": float(np.mean(frame_valid)),
            "invalid_frame_indices": np.flatnonzero(~frame_valid).tolist(),
            "source_vs_mujoco_foot_error_m": {
                "mean": float(np.mean(foot_error)),
                "median": float(np.median(foot_error)),
                "p95": float(np.quantile(foot_error, 0.95)),
                "max": float(np.max(foot_error)),
            },
        },
        "outputs": {
            "reconstructed_motion": str(output_dir / "reconstructed_motion.npz"),
            "summary": str(output_dir / "summary.json"),
            "trajectories": str(output_dir / "trajectories.png") if plot else None,
            "kinematic_replay": (
                str(output_dir / "kinematic_replay.gif") if render else None
            ),
            "rendered_frames": rendered_frames,
        },
        "warnings": warnings,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing motion.npy and clip.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--xml", type=Path, default=DEFAULT_GO2_XML)
    parser.add_argument(
        "--joint-order",
        choices=("auto", "documented_leg_major", "genesis_type_major"),
        default="auto",
    )
    parser.add_argument(
        "--min-base-height",
        type=float,
        default=0.18,
        help="Audit threshold only; frames are not deleted",
    )
    parser.add_argument("--render-stride", type=int, default=2)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.render_stride <= 0 or args.width <= 0 or args.height <= 0:
        raise ValueError("render stride, width, and height must be positive")
    if not np.isfinite(args.min_base_height):
        raise ValueError("--min-base-height must be finite")

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    xml_path = args.xml.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)
    if not xml_path.is_file():
        raise FileNotFoundError(xml_path)

    summary = inspect_clip(
        input_dir=input_dir,
        output_dir=output_dir,
        xml_path=xml_path,
        min_base_height=args.min_base_height,
        joint_order=args.joint_order,
        render_stride=args.render_stride,
        width=args.width,
        height=args.height,
        render=not args.no_render,
        plot=not args.no_plot,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    for path in summary["outputs"].values():
        if isinstance(path, str):
            print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
