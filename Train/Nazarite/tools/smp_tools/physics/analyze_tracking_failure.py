"""Diagnose why a geometric Go2 reference fails physics tracking.

The geometric retargeter is sampled at the 3DDogs frame rate, while the
physics tracker simulates at MuJoCo's faster control rate.  This tool aligns
them through ``source_reference_frame`` and reports which joint first exceeds
the tracking-error limit used by the physics filter.

Example:

    cd Train/Nazarite
    uv run python tools/smp_tools/physics/analyze_tracking_failure.py \\
      --geometric-input output/go2_retarget/d29_t1_a/go2_reference_geometric.npz \\
      --physics-input output/go2_physics_tracking/d29_t1_a_train_actuators/go2_physics_rollout.npz \\
      --output-dir output/go2_tracking_analysis/d29_t1_a_train_actuators --plot
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

try:
  from tools.smp_tools.retargeting.inspect_go2_kinematics import DEFAULT_GO2_XML
except ModuleNotFoundError:
  # Support direct execution from the repository root.
  sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
  from tools.smp_tools.retargeting.inspect_go2_kinematics import DEFAULT_GO2_XML


def _load_npz(path: Path, required_keys: set[str], label: str) -> dict[str, np.ndarray]:
  with np.load(path, allow_pickle=False) as archive:
    missing = sorted(required_keys - set(archive.files))
    if missing:
      raise ValueError(f"{label} NPZ lacks required arrays: {', '.join(missing)}")
    result = {key: archive[key].copy() for key in required_keys}
  return result


def _joint_addresses(model: mujoco.MjModel, joint_names: tuple[str, ...]) -> np.ndarray:
  addresses: list[int] = []
  for joint_name in joint_names:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
      raise ValueError(f"Go2 XML has no joint named {joint_name!r}")
    if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
      raise ValueError(f"expected a hinge joint for {joint_name!r}")
    addresses.append(int(model.jnt_qposadr[joint_id]))
  return np.asarray(addresses, dtype=np.intp)


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
  padded = np.concatenate(([False], mask, [False]))
  changes = np.flatnonzero(padded[1:] != padded[:-1])
  return [(int(start), int(end)) for start, end in changes.reshape(-1, 2)]


def analyze_tracking(
  geometric: dict[str, np.ndarray],
  physics: dict[str, np.ndarray],
  model: mujoco.MjModel,
  error_limit_rad: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
  """Align reference and rollout, then return a JSON-safe report and arrays."""
  joint_names = tuple(str(name) for name in geometric["smp_joint_order"].tolist())
  if len(joint_names) != 12 or len(set(joint_names)) != 12:
    raise ValueError("geometric smp_joint_order must contain 12 distinct Go2 joint names")
  joint_qpos_addresses = _joint_addresses(model, joint_names)

  reference_qpos = geometric["qpos"]
  physics_qpos = physics["qpos"]
  source_frame = physics["source_reference_frame"].astype(np.intp, copy=False)
  physics_frames = physics_qpos.shape[0]
  if (
    reference_qpos.ndim != 2
    or physics_qpos.ndim != 2
    or reference_qpos.shape[1] != model.nq
    or physics_qpos.shape[1] != model.nq
  ):
    raise ValueError("geometric and physics qpos must be [frames, model.nq]")
  if source_frame.shape != (physics_frames,) or np.any(source_frame < 0) or np.any(
    source_frame >= reference_qpos.shape[0]
  ):
    raise ValueError("source_reference_frame is outside the geometric reference range")

  time_s = physics["time_s"]
  valid = physics["physics_valid"].astype(bool, copy=False)
  base_position = physics["root_pos_mujoco"]
  base_up_dot = physics["base_up_dot"]
  foot_error = physics["foot_target_error_m"]
  if (
    time_s.shape != (physics_frames,)
    or valid.shape != (physics_frames,)
    or base_position.shape != (physics_frames, 3)
    or base_up_dot.shape != (physics_frames,)
    or foot_error.shape != (physics_frames, 4)
  ):
    raise ValueError("physics rollout arrays have incompatible frame dimensions")

  desired_joint_position = reference_qpos[source_frame][:, joint_qpos_addresses]
  actual_joint_position = physics_qpos[:, joint_qpos_addresses]
  joint_error = desired_joint_position - actual_joint_position
  abs_joint_error = np.abs(joint_error)
  max_joint_error = np.max(abs_joint_error, axis=1)
  error_exceeds_limit = max_joint_error > error_limit_rad

  first_error_index = int(np.flatnonzero(error_exceeds_limit)[0]) if np.any(error_exceeds_limit) else None
  first_invalid_index = int(np.flatnonzero(~valid)[0]) if np.any(~valid) else None
  joint_reports: list[dict[str, Any]] = []
  for joint_index, joint_name in enumerate(joint_names):
    per_joint_error = abs_joint_error[:, joint_index]
    violations = per_joint_error > error_limit_rad
    joint_reports.append(
      {
        "joint": joint_name,
        "max_abs_error_rad": float(np.max(per_joint_error)),
        "rms_error_rad": float(np.sqrt(np.mean(np.square(joint_error[:, joint_index])))),
        "p95_abs_error_rad": float(np.percentile(per_joint_error, 95.0)),
        "fraction_over_limit": float(np.mean(violations)),
        "first_over_limit_time_s": (
          float(time_s[np.flatnonzero(violations)[0]]) if np.any(violations) else None
        ),
      }
    )
  joint_reports.sort(key=lambda item: float(item["max_abs_error_rad"]), reverse=True)

  valid_runs = [
    {
      "start_time_s": float(time_s[start]),
      "end_time_s": float(time_s[end - 1]),
      "duration_s": float(time_s[end - 1] - time_s[start]),
    }
    for start, end in _contiguous_runs(valid)
  ]
  first_error: dict[str, Any] | None = None
  if first_error_index is not None:
    dominant_joint_index = int(np.argmax(abs_joint_error[first_error_index]))
    first_error = {
      "time_s": float(time_s[first_error_index]),
      "source_reference_frame": int(source_frame[first_error_index]),
      "dominant_joint": joint_names[dominant_joint_index],
      "dominant_joint_error_rad": float(joint_error[first_error_index, dominant_joint_index]),
      "max_abs_error_rad": float(max_joint_error[first_error_index]),
    }

  report: dict[str, Any] = {
    "smp_joint_order": list(joint_names),
    "geometric_frames": int(reference_qpos.shape[0]),
    "geometric_fps": float(np.asarray(geometric["fps"]).item()),
    "physics_frames": int(physics_frames),
    "physics_fps": float(np.asarray(physics["fps"]).item()),
    "error_limit_rad": error_limit_rad,
    "first_joint_error_over_limit": first_error,
    "first_physics_invalid_time_s": (
      float(time_s[first_invalid_index]) if first_invalid_index is not None else None
    ),
    "longest_valid_duration_s": max((run["duration_s"] for run in valid_runs), default=0.0),
    "valid_runs": valid_runs,
    "max_abs_joint_error_rad": float(np.max(max_joint_error)),
    "mean_foot_target_error_m": float(np.mean(foot_error)),
    "max_foot_target_error_m": float(np.max(foot_error)),
    "minimum_base_height_m": float(np.min(base_position[:, 2])),
    "minimum_base_up_dot": float(np.min(base_up_dot)),
    "joint_error_ranking": joint_reports,
  }
  arrays = {
    "time_s": time_s,
    "desired_joint_position": desired_joint_position,
    "actual_joint_position": actual_joint_position,
    "joint_error_rad": joint_error,
    "max_abs_joint_error_rad": max_joint_error,
    "physics_valid": valid,
    "base_height_m": base_position[:, 2],
    "base_up_dot": base_up_dot,
    "foot_target_error_m": foot_error,
  }
  return report, arrays


def _plot_analysis(
  output_path: Path,
  report: dict[str, Any],
  arrays: dict[str, np.ndarray],
) -> None:
  os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".matplotlib"))
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  time_s = arrays["time_s"]
  ranking = report["joint_error_ranking"]
  assert isinstance(ranking, list)
  top_joint_names = [str(item["joint"]) for item in ranking[:3]]
  source_joint_names = [str(name) for name in report["smp_joint_order"]]

  figure, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
  for joint_name in top_joint_names:
    joint_index = source_joint_names.index(joint_name)
    axes[0].plot(time_s, arrays["desired_joint_position"][:, joint_index], label=f"{joint_name} desired")
    axes[0].plot(
      time_s,
      arrays["actual_joint_position"][:, joint_index],
      linestyle="--",
      label=f"{joint_name} actual",
    )
    axes[1].plot(time_s, np.abs(arrays["joint_error_rad"][:, joint_index]), label=joint_name)
  axes[0].set_ylabel("joint angle (rad)")
  axes[0].set_title("Three worst-tracked joints: desired vs physical")
  axes[0].legend(ncol=2, fontsize=8)
  axes[0].grid(alpha=0.3)
  axes[1].axhline(float(report["error_limit_rad"]), color="tab:red", linestyle=":", label="error limit")
  axes[1].set_ylabel("absolute error (rad)")
  axes[1].set_title("Joint tracking error")
  axes[1].legend(ncol=4, fontsize=8)
  axes[1].grid(alpha=0.3)
  axes[2].plot(time_s, arrays["base_height_m"], label="base height (m)")
  axes[2].plot(time_s, arrays["base_up_dot"], label="base up dot")
  axes[2].set_ylabel("height / alignment")
  axes[2].set_title("Body stability")
  axes[2].legend()
  axes[2].grid(alpha=0.3)
  axes[3].plot(time_s, arrays["foot_target_error_m"])
  axes[3].step(time_s, arrays["physics_valid"].astype(int), where="post", color="black", label="physics valid")
  axes[3].set_xlabel("simulation time (s)")
  axes[3].set_ylabel("foot error (m) / valid")
  axes[3].set_title("Foot-target error and accepted samples")
  axes[3].legend(["FL", "FR", "RL", "RR", "physics valid"], ncol=5, fontsize=8)
  axes[3].grid(alpha=0.3)
  figure.suptitle("Go2 geometric-reference to physics-tracking diagnosis")
  figure.tight_layout()
  figure.savefig(output_path, dpi=180)
  plt.close(figure)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--geometric-input", type=Path, required=True)
  parser.add_argument("--physics-input", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--xml", type=Path, default=DEFAULT_GO2_XML)
  parser.add_argument("--error-limit-rad", type=float, default=0.45)
  parser.add_argument("--plot", action="store_true", help="Write tracking_diagnosis.png")
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
  args = _parse_args(argv)
  if args.error_limit_rad <= 0.0:
    raise ValueError("--error-limit-rad must be positive")
  geometric_path = args.geometric_input.expanduser().resolve()
  physics_path = args.physics_input.expanduser().resolve()
  xml_path = args.xml.expanduser().resolve()
  if not geometric_path.is_file() or not physics_path.is_file() or not xml_path.is_file():
    raise FileNotFoundError("geometric input, physics input, and XML must all exist")
  output_dir = args.output_dir.expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  geometric = _load_npz(geometric_path, {"qpos", "fps", "smp_joint_order"}, "geometric")
  physics = _load_npz(
    physics_path,
    {
      "qpos", "fps", "time_s", "source_reference_frame", "physics_valid",
      "root_pos_mujoco", "base_up_dot", "foot_target_error_m",
    },
    "physics",
  )
  report, arrays = analyze_tracking(
    geometric=geometric,
    physics=physics,
    model=mujoco.MjModel.from_xml_path(str(xml_path)),
    error_limit_rad=args.error_limit_rad,
  )
  report.update(
    {
      "geometric_input": str(geometric_path),
      "physics_input": str(physics_path),
      "go2_xml": str(xml_path),
    }
  )
  with (output_dir / "tracking_failure_report.json").open("w", encoding="utf-8") as stream:
    json.dump(report, stream, indent=2)
    stream.write("\n")
  if args.plot:
    _plot_analysis(output_dir / "tracking_diagnosis.png", report, arrays)
  print(json.dumps(report, indent=2))
  print(f"\nWrote: {output_dir / 'tracking_failure_report.json'}")
  if args.plot:
    print(f"Wrote: {output_dir / 'tracking_diagnosis.png'}")


if __name__ == "__main__":
  main()
