"""Audit speed, acceleration, and inverse-dynamics torque demand of a Go2 reference.

This is a read-only pre-filter for geometric retargeting results.  It does not
simulate contact and does not replace the physics tracker; it identifies
whether a reference asks the Go2 joints for excessive speed, acceleration, or
inverse-dynamics torque.
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


LEG_ORDER = ("FL", "FR", "RL", "RR")


def _load_reference(path: Path) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
  required = {"qpos", "fps", "smp_joint_order"}
  with np.load(path, allow_pickle=False) as archive:
    missing = sorted(required - set(archive.files))
    if missing:
      raise ValueError(f"reference NPZ lacks arrays: {', '.join(missing)}")
    reference = {key: archive[key].copy() for key in required}
  if reference["qpos"].ndim != 2 or reference["qpos"].shape[0] < 3:
    raise ValueError("qpos must contain at least three frames")
  if not np.isfinite(reference["qpos"]).all():
    raise ValueError("qpos contains NaN or Inf")
  names = tuple(str(value) for value in reference["smp_joint_order"].tolist())
  if len(names) != 12 or len(set(names)) != 12:
    raise ValueError("smp_joint_order must contain 12 distinct joints")
  return reference, names


def _joint_dof_addresses(model: mujoco.MjModel, names: tuple[str, ...]) -> np.ndarray:
  addresses = []
  for name in names:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0 or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
      raise ValueError(f"invalid Go2 hinge joint: {name}")
    addresses.append(int(model.jnt_dofadr[joint_id]))
  return np.asarray(addresses, dtype=np.intp)


def _reference_velocities(model: mujoco.MjModel, qpos: np.ndarray, fps: float) -> np.ndarray:
  qvel = np.empty((qpos.shape[0], model.nv), dtype=np.float64)
  dt = 1.0 / fps
  for index in range(qpos.shape[0] - 1):
    mujoco.mj_differentiatePos(model, qvel[index], dt, qpos[index], qpos[index + 1])
  qvel[-1] = qvel[-2]
  return qvel


def _percentiles(values: np.ndarray) -> dict[str, float]:
  absolute = np.abs(values)
  return {
    "max": float(np.max(absolute)),
    "p95": float(np.percentile(absolute, 95.0)),
    "p99": float(np.percentile(absolute, 99.0)),
    "mean": float(np.mean(absolute)),
  }


def audit(path: Path, xml_path: Path, effort_limits: dict[str, float]) -> dict[str, Any]:
  reference, joint_names = _load_reference(path)
  model = mujoco.MjModel.from_xml_path(str(xml_path))
  qpos = reference["qpos"].astype(np.float64)
  fps = float(np.asarray(reference["fps"]).item())
  if fps <= 0.0:
    raise ValueError("fps must be positive")
  if qpos.shape[1] != model.nq:
    raise ValueError(f"reference nq={qpos.shape[1]} does not match model nq={model.nq}")

  dof_addresses = _joint_dof_addresses(model, joint_names)
  qvel = _reference_velocities(model, qpos, fps)
  qacc = np.gradient(qvel, 1.0 / fps, axis=0, edge_order=2)
  joint_velocity = qvel[:, dof_addresses]
  joint_acceleration = qacc[:, dof_addresses]

  inverse_torque = np.empty_like(joint_velocity)
  data = mujoco.MjData(model)
  for index in range(qpos.shape[0]):
    data.qpos[:] = qpos[index]
    data.qvel[:] = qvel[index]
    data.qacc[:] = qacc[index]
    mujoco.mj_inverse(model, data)
    inverse_torque[index] = data.qfrc_inverse[dof_addresses]

  limits = np.asarray([
    effort_limits["calf"] if name.endswith("_calf_joint") else effort_limits["hip_thigh"]
    for name in joint_names
  ])
  torque_ratio = np.abs(inverse_torque) / limits[None]
  joint_reports = []
  for index, name in enumerate(joint_names):
    joint_reports.append({
      "joint": name,
      "speed_rad_s": _percentiles(joint_velocity[:, index]),
      "acceleration_rad_s2": _percentiles(joint_acceleration[:, index]),
      "inverse_dynamics_torque_nm": _percentiles(inverse_torque[:, index]),
      "effort_limit_nm": float(limits[index]),
      "torque_ratio_max": float(np.max(torque_ratio[:, index])),
      "torque_over_limit_fraction": float(np.mean(torque_ratio[:, index] > 1.0)),
    })
  joint_reports.sort(key=lambda item: item["torque_ratio_max"], reverse=True)
  return {
    "valid": True,
    "source": str(path),
    "go2_xml": str(xml_path),
    "frames": int(qpos.shape[0]),
    "fps": fps,
    "duration_s": float((qpos.shape[0] - 1) / fps),
    "limits_nm": effort_limits,
    "global": {
      "joint_speed_rad_s": _percentiles(joint_velocity),
      "joint_acceleration_rad_s2": _percentiles(joint_acceleration),
      "inverse_dynamics_torque_nm": _percentiles(inverse_torque),
      "max_torque_ratio": float(np.max(torque_ratio)),
      "torque_over_limit_fraction": float(np.mean(torque_ratio > 1.0)),
    },
    "joint_ranking_by_torque_ratio": joint_reports,
  }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--xml", type=Path, default=DEFAULT_GO2_XML)
  parser.add_argument("--hip-thigh-effort", type=float, default=23.7)
  parser.add_argument("--calf-effort", type=float, default=45.43)
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
  args = _parse_args(argv)
  if args.hip_thigh_effort <= 0.0 or args.calf_effort <= 0.0:
    raise ValueError("effort limits must be positive")
  report = audit(
    args.input.expanduser().resolve(),
    args.xml.expanduser().resolve(),
    {"hip_thigh": args.hip_thigh_effort, "calf": args.calf_effort},
  )
  output = args.output.expanduser().resolve()
  output.parent.mkdir(parents=True, exist_ok=True)
  with output.open("w", encoding="utf-8") as stream:
    json.dump(report, stream, indent=2)
    stream.write("\n")
  print(json.dumps({
    "valid": report["valid"],
    "frames": report["frames"],
    "global": report["global"],
    "worst_joints": report["joint_ranking_by_torque_ratio"][:4],
  }, indent=2))
  print(f"Wrote: {output}")


if __name__ == "__main__":
  main()
