"""Convert 1x retargeted Go2 motions into canonical SMP motion windows.

Input motions keep their physical duration: e.g. 60 Hz 3DDogs references are
resampled to the 50 Hz Go2 policy rate, never time-stretched.  Foot positions
come from MuJoCo forward kinematics of the retargeted Go2 qpos, matching the
online robot state that the frozen prior will score during PPO.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch

from nazarite.config.train_config.train_algorithm.smp.prior.features import (
    GO2_SMP_FEATURE_DIM,
    GO2_SMP_FEATURE_DIMS,
    GO2_SMP_FEATURE_NAMES,
    GO2_SMP_FOOT_SITE_NAMES,
    GO2_SMP_FPS,
    GO2_SMP_JOINT_NAMES,
    GO2_SMP_WINDOW_SIZE,
    compute_motion_windows,
)

# File location: <Nazarite>/tools/smp_tools/prior/build_motion_windows.py
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GO2_XML = PROJECT_ROOT / "MJCF-Manager/Robots/GO2/xmls/go2.xml"


def _read_entries(path: Path, entries_key: str | None) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if isinstance(value, list):
        entries = value
    elif isinstance(value, dict):
        if entries_key is not None:
            entries = value.get(entries_key)
            if not isinstance(entries, list):
                raise ValueError(f"manifest has no list at key {entries_key!r}: {path}")
        else:
            entries = next(
                (
                    value[key]
                    for key in ("clips", "processed", "accepted")
                    if isinstance(value.get(key), list)
                ),
                None,
            )
            if entries is None:
                raise ValueError(f"cannot find clips/processed/accepted list in {path}")
    else:
        raise TypeError(f"manifest must contain a JSON list or dictionary: {path}")
    if not all(isinstance(entry, dict) for entry in entries):
        raise ValueError(f"manifest entries must be dictionaries: {path}")
    return entries


def _normalize_quaternions(values: np.ndarray) -> np.ndarray:
    result = values.astype(np.float64, copy=True)
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError("motion contains a zero-norm root quaternion")
    result /= norms
    for index in range(1, len(result)):
        if np.dot(result[index - 1], result[index]) < 0.0:
            result[index] *= -1.0
    return result


def _interpolate(values: np.ndarray, source_positions: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(values.shape[0], -1)
    source_frames = np.arange(values.shape[0], dtype=np.float64)
    output = np.empty((len(source_positions), flat.shape[1]), dtype=np.float64)
    for dimension in range(flat.shape[1]):
        output[:, dimension] = np.interp(
            source_positions, source_frames, flat[:, dimension]
        )
    return output.reshape(len(source_positions), *values.shape[1:])


def _interpolate_quaternions(
    quaternions: np.ndarray, source_positions: np.ndarray
) -> np.ndarray:
    return _normalize_quaternions(
        _interpolate(_normalize_quaternions(quaternions), source_positions)
    )


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
    negative = relative[:, 0] < 0.0
    relative[negative] *= -1.0
    vector_norm = np.linalg.norm(relative[:, 1:], axis=1)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(relative[:, 0], 0.0, 1.0))
    scale = np.divide(
        angle, vector_norm, out=np.full_like(angle, 2.0), where=vector_norm > 1.0e-9
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
        result[1:-1] = _relative_rotvec(quaternions[2:], quaternions[:-2]) * (0.5 * fps)
    return result


def _joint_qpos_addresses(model: mujoco.MjModel) -> np.ndarray:
    addresses = []
    for name in GO2_SMP_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Go2 model has no joint {name!r}")
        addresses.append(int(model.jnt_qposadr[joint_id]))
    return np.asarray(addresses, dtype=np.intp)


def _foot_site_ids(model: mujoco.MjModel) -> np.ndarray:
    ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        for name in GO2_SMP_FOOT_SITE_NAMES
    ]
    if any(site_id < 0 for site_id in ids):
        raise ValueError(f"Go2 model lacks SMP foot sites {GO2_SMP_FOOT_SITE_NAMES}")
    return np.asarray(ids, dtype=np.intp)


def _load_reference(path: Path) -> dict[str, np.ndarray]:
    required = {"fps", "qpos", "root_pos_mujoco", "root_quat_wxyz"}
    with np.load(path, allow_pickle=False) as archive:
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{path} lacks arrays: {sorted(missing)}")
        return {key: archive[key].copy() for key in archive.files}


def _build_clip(
    path: Path,
    model: mujoco.MjModel,
    output_fps: int,
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    reference = _load_reference(path)
    qpos = np.asarray(reference["qpos"], dtype=np.float64)
    root_pos = np.asarray(reference["root_pos_mujoco"], dtype=np.float64)
    root_quat = np.asarray(reference["root_quat_wxyz"], dtype=np.float64)
    input_fps = float(np.asarray(reference["fps"]).item())
    if qpos.ndim != 2 or qpos.shape[1] != model.nq or qpos.shape[0] < 2:
        raise ValueError(f"{path}: qpos must have shape [T,{model.nq}] with T >= 2")
    if root_pos.shape != (len(qpos), 3) or root_quat.shape != (len(qpos), 4):
        raise ValueError(f"{path}: root arrays do not match qpos frames")
    if input_fps <= 0.0:
        raise ValueError(f"{path}: fps must be positive")

    duration = (len(qpos) - 1) / input_fps
    output_frames = round(duration * output_fps) + 1
    source_positions = np.linspace(0.0, len(qpos) - 1, output_frames)
    qpos_output = _interpolate(qpos, source_positions)
    root_pos_output = _interpolate(root_pos, source_positions)
    root_quat_output = _interpolate_quaternions(root_quat, source_positions)
    qpos_output[:, :3] = root_pos_output
    qpos_output[:, 3:7] = root_quat_output

    joint_addresses = _joint_qpos_addresses(model)
    joint_pos = qpos_output[:, joint_addresses]
    foot_ids = _foot_site_ids(model)
    data = mujoco.MjData(model)
    foot_pos = np.empty((output_frames, 4, 3), dtype=np.float64)
    for frame, pose in enumerate(qpos_output):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        foot_pos[frame] = data.site_xpos[foot_ids]

    root_lin_vel = np.gradient(root_pos_output, 1.0 / output_fps, axis=0, edge_order=2)
    root_ang_vel = _angular_velocity_world(root_quat_output, output_fps)
    windows = compute_motion_windows(
        root_pos_w=torch.from_numpy(root_pos_output).float(),
        root_quat_w=torch.from_numpy(root_quat_output).float(),
        root_lin_vel_w=torch.from_numpy(root_lin_vel).float(),
        root_ang_vel_w=torch.from_numpy(root_ang_vel).float(),
        foot_pos_w=torch.from_numpy(foot_pos).float(),
        joint_pos=torch.from_numpy(joint_pos).float(),
        window_size=window_size,
        stride=stride,
    ).numpy()
    if not np.isfinite(windows).all():
        raise ValueError(f"{path}: generated windows contain non-finite values")
    summary = {
        "source": str(path),
        "speed_scale": 1.0,
        "input_fps": input_fps,
        "output_fps": output_fps,
        "input_frames": len(qpos),
        "output_frames": output_frames,
        "input_duration_s": duration,
        "output_duration_s": (output_frames - 1) / output_fps,
        "windows": int(windows.shape[0]),
        "window_size": window_size,
        "stride": stride,
        "feature_dim": int(windows.shape[-1]),
        "max_root_speed_m_s": float(np.linalg.norm(root_lin_vel, axis=1).max()),
        "max_root_angular_speed_rad_s": float(
            np.linalg.norm(root_ang_vel, axis=1).max()
        ),
    }
    return windows.astype(np.float32), summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--entries-key", choices=("clips", "processed", "accepted"))
    parser.add_argument("--source-field", default="geometric_npz")
    parser.add_argument("--xml", type=Path, default=DEFAULT_GO2_XML)
    parser.add_argument("--output-fps", type=int, default=GO2_SMP_FPS)
    parser.add_argument("--window-size", type=int, default=GO2_SMP_WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-clips", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.output_fps <= 0 or args.window_size <= 0 or args.stride <= 0:
        raise ValueError("fps, window size, and stride must be positive")
    manifest = args.manifest.expanduser().resolve()
    entries = _read_entries(manifest, args.entries_key)
    if args.max_clips is not None:
        if args.max_clips <= 0:
            raise ValueError("--max-clips must be positive")
        entries = entries[: args.max_clips]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(args.xml.expanduser().resolve()))
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    total_windows = 0
    for index, entry in enumerate(entries, start=1):
        name = str(entry.get("clip_name", f"clip_{index:04d}"))
        print(f"[{index}/{len(entries)}] {name}", flush=True)
        try:
            source = Path(str(entry[args.source_field])).expanduser().resolve()
            windows, summary = _build_clip(
                source, model, args.output_fps, args.window_size, args.stride
            )
            if len(windows) == 0:
                raise ValueError(f"clip is shorter than window_size={args.window_size}")
            output_path = output_dir / f"{name}.npz"
            np.savez_compressed(
                output_path,
                windows=windows,
                fps=np.asarray(args.output_fps, dtype=np.float32),
                speed_scale=np.asarray(1.0, dtype=np.float32),
                window_size=np.asarray(args.window_size, dtype=np.int32),
                stride=np.asarray(args.stride, dtype=np.int32),
                feature_dims=np.asarray(GO2_SMP_FEATURE_DIMS, dtype=np.int32),
                feature_names=np.asarray(GO2_SMP_FEATURE_NAMES),
                joint_names=np.asarray(GO2_SMP_JOINT_NAMES),
                foot_site_names=np.asarray(GO2_SMP_FOOT_SITE_NAMES),
            )
            summary.update({"clip_name": name, "output": str(output_path)})
            total_windows += len(windows)
            results.append(summary)
        except (OSError, KeyError, ValueError, RuntimeError) as error:
            failures.append({"clip_name": name, "error": str(error)})

    report = {
        "source_manifest": str(manifest),
        "source_entries_key": args.entries_key,
        "source_field": args.source_field,
        "speed_scale": 1.0,
        "output_fps": args.output_fps,
        "window_size": args.window_size,
        "stride": args.stride,
        "feature_dim": GO2_SMP_FEATURE_DIM,
        "feature_names": list(GO2_SMP_FEATURE_NAMES),
        "feature_dims": list(GO2_SMP_FEATURE_DIMS),
        "requested_clips": len(entries),
        "processed_clips": len(results),
        "failed_clips": len(failures),
        "total_windows": total_windows,
        "clips": results,
        "failures": failures,
    }
    report_path = output_dir / "motion_window_report.json"
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "processed_clips",
                    "failed_clips",
                    "total_windows",
                    "feature_dim",
                )
            },
            indent=2,
        )
    )
    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()
