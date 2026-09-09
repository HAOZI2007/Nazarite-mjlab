"""Compute GQMR-style quality metrics for a Go2 geometric reference."""

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
  from tools.smp_tools.retargeting.inspect_go2_kinematics import (
    DEFAULT_GO2_XML,
    SMP_LEG_ORDER,
    _joint_names,
    _object_id,
  )
except ModuleNotFoundError:
  sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
  from tools.smp_tools.retargeting.inspect_go2_kinematics import (
    DEFAULT_GO2_XML,
    SMP_LEG_ORDER,
    _joint_names,
    _object_id,
  )


def _load_motion(path: Path) -> dict[str, np.ndarray]:
  required = {"fps", "qpos", "foot_target_world_m", "foot_position_world_m"}
  with np.load(path, allow_pickle=False) as archive:
    missing = sorted(required - set(archive.files))
    if missing:
      raise ValueError(f"reference NPZ lacks required arrays: {', '.join(missing)}")
    return {key: archive[key].copy() for key in archive.files}


def _p95(values: np.ndarray) -> float:
  return float(np.percentile(values, 95.0)) if values.size else 0.0


def _collision_count(model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray, stride: int) -> tuple[int, list[int]]:
  collision_frames = 0
  collision_indices: list[int] = []
  for frame_index in range(0, qpos.shape[0], max(stride, 1)):
    data.qpos[:] = qpos[frame_index]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    mujoco.mj_collision(model, data)
    self_contacts = 0
    for contact_index in range(data.ncon):
      contact = data.contact[contact_index]
      body_a = int(model.geom_bodyid[contact.geom1])
      body_b = int(model.geom_bodyid[contact.geom2])
      if body_a != 0 and body_b != 0 and body_a != body_b:
        self_contacts += 1
    if self_contacts:
      collision_frames += 1
      collision_indices.append(frame_index)
  return collision_frames, collision_indices


def evaluate_motion(input_path: Path, xml_path: Path, collision_stride: int = 1) -> dict[str, Any]:
  motion = _load_motion(input_path)
  model = mujoco.MjModel.from_xml_path(str(xml_path))
  qpos = np.asarray(motion["qpos"], dtype=np.float64)
  foot_target = np.asarray(motion["foot_target_world_m"], dtype=np.float64)
  foot_position = np.asarray(motion["foot_position_world_m"], dtype=np.float64)
  fps = float(np.asarray(motion["fps"]).item())
  if qpos.ndim != 2 or qpos.shape[1] != model.nq:
    raise ValueError(f"qpos must have shape [frames, {model.nq}]")
  if foot_target.shape != foot_position.shape or foot_target.ndim != 3 or foot_target.shape[1:] != (4, 3):
    raise ValueError("foot target and solved positions must have shape [frames, 4, 3]")

  joint_ids = np.asarray([
    _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in _joint_names()
  ], dtype=np.intp)
  joint_qpos_addresses = np.asarray([model.jnt_qposadr[joint_id] for joint_id in joint_ids], dtype=np.intp)
  joint_position = qpos[:, joint_qpos_addresses]
  joint_bounds = model.jnt_range[joint_ids]
  lower_violation = np.maximum(joint_bounds[None, :, 0] - joint_position, 0.0)
  upper_violation = np.maximum(joint_position - joint_bounds[None, :, 1], 0.0)
  joint_violation = np.maximum(lower_violation, upper_violation)
  foot_error = np.linalg.norm(foot_target - foot_position, axis=-1)
  joint_speed = np.linalg.norm(np.diff(joint_position, axis=0), axis=-1) * fps if qpos.shape[0] > 1 else np.zeros((0, 12))
  joint_acceleration = np.diff(joint_speed, axis=0) * fps if joint_speed.shape[0] > 1 else np.zeros((0, 12))
  foot_speed = np.linalg.norm(np.diff(foot_position, axis=0), axis=-1) * fps if foot_position.shape[0] > 1 else np.zeros((0, 4))
  if "contact" in motion:
    contact = np.asarray(motion["contact"], dtype=bool)
  else:
    contact = foot_target[..., 2] <= (np.percentile(foot_target[..., 2], 10.0) + 0.025)
  contact_speed = foot_speed[contact[1:]] if foot_speed.size else np.zeros(0)
  model_data = mujoco.MjData(model)
  collision_frames, collision_indices = _collision_count(model, model_data, qpos, collision_stride)
  ground_penetration = np.maximum(-foot_position[..., 2], 0.0)
  solver_residual = np.asarray(motion.get("solver_residual_m", np.max(foot_error, axis=1)), dtype=np.float64)
  if "solver_status" in motion:
    statuses = np.asarray(motion["solver_status"]).astype(str)
  elif "ik_success" in motion:
    # Baseline retarget outputs predate explicit solver statuses.  Preserve
    # their success information so the A/B report remains meaningful.
    statuses = np.where(np.asarray(motion["ik_success"], dtype=bool), "SUCCESS", "UNKNOWN")
  else:
    statuses = np.full(qpos.shape[0], "UNKNOWN", dtype="U16")
  status_counts = {status: int(np.count_nonzero(statuses == status)) for status in np.unique(statuses)}
  valid_status = np.isin(statuses, ("SUCCESS", "REPAIRED"))
  return {
    "source": str(input_path),
    "go2_xml": str(xml_path),
    "fps": fps,
    "frames": int(qpos.shape[0]),
    "duration_s": float((qpos.shape[0] - 1) / fps),
    "valid_fraction": float(np.mean(valid_status)) if statuses.size else 0.0,
    "solver_status_counts": status_counts,
    "foot_error_rmse_m": float(np.sqrt(np.mean(np.square(foot_error)))),
    "foot_error_mean_m": float(np.mean(foot_error)),
    "foot_error_p95_m": _p95(foot_error.reshape(-1)),
    "solver_residual_rmse_m": float(np.sqrt(np.mean(np.square(solver_residual))),),
    "solver_residual_p95_m": _p95(solver_residual),
    "joint_limit_violation_frames": int(np.count_nonzero(np.any(joint_violation > 1.0e-7, axis=1))),
    "joint_limit_violation_max_rad": float(np.max(joint_violation)),
    "joint_speed_max_rad_s": float(np.max(np.abs(joint_speed))) if joint_speed.size else 0.0,
    "joint_acceleration_max_rad_s2": float(np.max(np.abs(joint_acceleration))) if joint_acceleration.size else 0.0,
    "contact_fraction_by_leg": np.mean(contact, axis=0).tolist(),
    "contact_foot_speed_mean_m_s": float(np.mean(contact_speed)) if contact_speed.size else 0.0,
    "contact_foot_speed_p95_m_s": _p95(contact_speed),
    "ground_penetration_frames": int(np.count_nonzero(np.any(ground_penetration > 1.0e-5, axis=1))),
    "ground_penetration_max_m": float(np.max(ground_penetration)),
    "self_collision_frames": collision_frames,
    "self_collision_frame_indices": collision_indices,
    "collision_check_stride": int(collision_stride),
    "max_finite": bool(np.isfinite(qpos).all() and np.isfinite(foot_target).all() and np.isfinite(foot_position).all()),
    "leg_order": list(SMP_LEG_ORDER),
  }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", type=Path, required=True)
  parser.add_argument("--xml", type=Path, default=DEFAULT_GO2_XML)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--collision-stride", type=int, default=1)
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
  args = _parse_args(argv)
  report = evaluate_motion(
    args.input.expanduser().resolve(),
    args.xml.expanduser().resolve(),
    args.collision_stride,
  )
  output = args.output.expanduser().resolve()
  output.parent.mkdir(parents=True, exist_ok=True)
  with output.open("w", encoding="utf-8") as stream:
    json.dump(report, stream, indent=2)
    stream.write("\n")
  print(json.dumps(report, indent=2))
  print(f"\nWrote: {output}")


if __name__ == "__main__":
  main()
