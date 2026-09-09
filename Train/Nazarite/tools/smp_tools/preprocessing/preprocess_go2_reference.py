"""Make a geometric Go2 reference easier for Nazarite's real actuators to track.

This tool never edits the original IK output.  It time-stretches the joint
trajectory (0.5 means half-speed), applies a short zero-phase moving-average
filter, and exports feet relative to the reference base frame.  The latter is
the appropriate error frame when the physics rollout has a free base rather
than a root-position controller.

Example:

    cd Train/Nazarite
    uv run python tools/smp_tools/preprocessing/preprocess_go2_reference.py \\
      --input output/go2_retarget/d29_t1_a/go2_reference_geometric.npz \\
      --output-dir output/go2_reference_preprocessed/d29_t1_a_half_speed
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

try:
  from tools.smp_tools.retargeting.inspect_go2_kinematics import DEFAULT_GO2_XML
except ModuleNotFoundError:
  sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
  from tools.smp_tools.retargeting.inspect_go2_kinematics import DEFAULT_GO2_XML


def _load_reference(path: Path) -> dict[str, np.ndarray]:
  required_keys = {
    "fps",
    "qpos",
    "smp_leg_order",
    "smp_joint_order",
    "root_pos_mujoco",
    "root_quat_wxyz",
    "foot_target_world_m",
  }
  with np.load(path, allow_pickle=False) as archive:
    missing = sorted(required_keys - set(archive.files))
    if missing:
      raise ValueError(f"input reference lacks required arrays: {', '.join(missing)}")
    return {key: archive[key].copy() for key in archive.files}


def _interpolate(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
  """Linear interpolation along the leading frame dimension."""
  source_positions = np.arange(values.shape[0], dtype=np.float64)
  flat = values.reshape(values.shape[0], -1)
  interpolated = np.empty((positions.size, flat.shape[1]), dtype=np.float64)
  for dimension in range(flat.shape[1]):
    interpolated[:, dimension] = np.interp(positions, source_positions, flat[:, dimension])
  return interpolated.reshape((positions.size, *values.shape[1:]))


def _interpolate_quaternions(quaternions: np.ndarray, positions: np.ndarray) -> np.ndarray:
  """Use normalized linear interpolation after making quaternion signs continuous."""
  continuous = quaternions.astype(np.float64, copy=True)
  for frame_index in range(1, continuous.shape[0]):
    if np.dot(continuous[frame_index - 1], continuous[frame_index]) < 0.0:
      continuous[frame_index] *= -1.0
  interpolated = _interpolate(continuous, positions)
  norms = np.linalg.norm(interpolated, axis=1, keepdims=True)
  if np.any(norms <= np.finfo(np.float64).eps):
    raise ValueError("reference contains a zero-norm root quaternion")
  return interpolated / norms


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
  """Centered moving-average smoothing that preserves endpoint values."""
  if window == 1:
    return values.copy()
  pad = window // 2
  padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
  kernel = np.full(window, 1.0 / window, dtype=np.float64)
  result = np.empty_like(values)
  for joint_index in range(values.shape[1]):
    result[:, joint_index] = np.convolve(padded[:, joint_index], kernel, mode="valid")
  result[0] = values[0]
  result[-1] = values[-1]
  return result


def _joint_addresses(model: mujoco.MjModel, joint_names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
  qpos_addresses: list[int] = []
  joint_ids: list[int] = []
  for joint_name in joint_names:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0 or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
      raise ValueError(f"Go2 XML lacks expected hinge joint {joint_name!r}")
    joint_ids.append(joint_id)
    qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
  return np.asarray(qpos_addresses, dtype=np.intp), np.asarray(joint_ids, dtype=np.intp)


def _rotation_matrices(quaternions_wxyz: np.ndarray) -> np.ndarray:
  matrices = np.empty((quaternions_wxyz.shape[0], 3, 3), dtype=np.float64)
  for frame_index, quaternion in enumerate(quaternions_wxyz):
    mujoco.mju_quat2Mat(matrices[frame_index].reshape(-1), quaternion)
  return matrices


def preprocess_reference(
  reference: dict[str, np.ndarray],
  model: mujoco.MjModel,
  playback_rate: float,
  smoothing_window: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
  """Return a time-stretched, smoothed reference and its provenance report."""
  qpos = reference["qpos"].astype(np.float64, copy=False)
  frame_count = qpos.shape[0]
  if qpos.ndim != 2 or qpos.shape[1] != model.nq or frame_count < 2:
    raise ValueError("reference qpos must have at least two [frames, model.nq] rows")
  fps = float(np.asarray(reference["fps"]).item())
  if fps <= 0.0:
    raise ValueError("reference fps must be positive")
  joint_names = tuple(str(name) for name in reference["smp_joint_order"].tolist())
  qpos_addresses, joint_ids = _joint_addresses(model, joint_names)
  if reference["root_pos_mujoco"].shape != (frame_count, 3) or reference["root_quat_wxyz"].shape != (
    frame_count,
    4,
  ):
    raise ValueError("root position and quaternion must match the reference frame count")
  if reference["foot_target_world_m"].shape != (frame_count, 4, 3):
    raise ValueError("foot_target_world_m must have shape [frames, 4, 3]")

  output_frame_count = int(round((frame_count - 1) / playback_rate)) + 1
  source_positions = np.linspace(0.0, frame_count - 1, output_frame_count, dtype=np.float64)
  output_qpos = _interpolate(qpos, source_positions)
  root_position = _interpolate(reference["root_pos_mujoco"], source_positions)
  root_quaternion = _interpolate_quaternions(reference["root_quat_wxyz"], source_positions)
  foot_target_world = _interpolate(reference["foot_target_world_m"], source_positions)

  original_joint_position = output_qpos[:, qpos_addresses].copy()
  smoothed_joint_position = _smooth(original_joint_position, smoothing_window)
  joint_limits = model.jnt_range[joint_ids]
  smoothed_joint_position = np.clip(smoothed_joint_position, joint_limits[:, 0], joint_limits[:, 1])
  output_qpos[:, qpos_addresses] = smoothed_joint_position

  root_rotation = _rotation_matrices(root_quaternion)
  # Row-vector form: local = world_vector @ world_from_base_rotation.
  foot_target_base = (foot_target_world - root_position[:, None, :]) @ root_rotation
  output: dict[str, np.ndarray] = {
    "fps": np.asarray(fps, dtype=np.float64),
    "qpos": output_qpos,
    "smp_leg_order": reference["smp_leg_order"],
    "smp_joint_order": reference["smp_joint_order"],
    "root_pos_mujoco": root_position,
    "root_quat_wxyz": root_quaternion,
    "root_rotation_mujoco": root_rotation,
    "foot_target_world_m": foot_target_world,
    "foot_target_base_m": foot_target_base,
    "source_reference_position": source_positions,
    "ik_success": (
      reference["ik_success"][np.rint(source_positions).astype(np.intp)].astype(bool, copy=True)
      if "ik_success" in reference
      else np.ones(output_frame_count, dtype=bool)
    ),
  }
  if "source_frame_numbers" in reference:
    output["source_frame_numbers"] = np.rint(
      _interpolate(reference["source_frame_numbers"], source_positions)
    ).astype(np.int64)

  original_velocity = np.diff(original_joint_position, axis=0) * fps
  processed_velocity = np.diff(smoothed_joint_position, axis=0) * fps
  summary: dict[str, Any] = {
    "input_frames": frame_count,
    "output_frames": output_frame_count,
    "input_fps": fps,
    "output_fps": fps,
    "playback_rate": playback_rate,
    "input_duration_s": (frame_count - 1) / fps,
    "output_duration_s": (output_frame_count - 1) / fps,
    "smoothing_window_frames": smoothing_window,
    "max_joint_speed_before_smoothing_rad_s": float(np.max(np.abs(original_velocity))),
    "max_joint_speed_after_smoothing_rad_s": float(np.max(np.abs(processed_velocity))),
    "foot_target_frame": "base",
    "smp_joint_order": list(joint_names),
  }
  return output, summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", type=Path, required=True, help="Geometric Go2 reference NPZ")
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--xml", type=Path, default=DEFAULT_GO2_XML)
  parser.add_argument(
    "--playback-rate",
    type=float,
    default=0.5,
    help="Reference speed relative to source; 0.5 is half-speed (default)",
  )
  parser.add_argument(
    "--smoothing-window",
    type=int,
    default=5,
    help="Odd moving-average width at output FPS; 1 disables smoothing",
  )
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
  args = _parse_args(argv)
  if not 0.0 < args.playback_rate <= 1.0:
    raise ValueError("--playback-rate must be in (0, 1]")
  if args.smoothing_window < 1 or args.smoothing_window % 2 == 0:
    raise ValueError("--smoothing-window must be a positive odd integer")
  input_path = args.input.expanduser().resolve()
  xml_path = args.xml.expanduser().resolve()
  if not input_path.is_file() or not xml_path.is_file():
    raise FileNotFoundError("input reference and Go2 XML must exist")
  output_dir = args.output_dir.expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  output, summary = preprocess_reference(
    reference=_load_reference(input_path),
    model=mujoco.MjModel.from_xml_path(str(xml_path)),
    playback_rate=args.playback_rate,
    smoothing_window=args.smoothing_window,
  )
  summary.update({"source": str(input_path), "go2_xml": str(xml_path)})
  output_path = output_dir / "go2_reference_preprocessed.npz"
  np.savez_compressed(output_path, allow_pickle=False, **output)
  with (output_dir / "preprocess_summary.json").open("w", encoding="utf-8") as stream:
    json.dump(summary, stream, indent=2)
    stream.write("\n")
  print(json.dumps(summary, indent=2))
  print(f"\nWrote: {output_path}")
  print(f"Wrote: {output_dir / 'preprocess_summary.json'}")


if __name__ == "__main__":
  main()
