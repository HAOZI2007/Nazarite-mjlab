"""Retarget canonical 3DDogs paws to Go2 with MuJoCo damped-least-squares IK.

This first retargeting stage is geometric only: it maps the dog root and paw
motion to a Go2-sized task-space reference, solves the 12 leg joints frame by
frame, and reports reachability.  It intentionally does not run a physics
controller; successful geometric references are the input to that later stage.

Example:

    cd Train/Nazarite
    uv run python tools/smp_tools/retargeting/retarget_3ddogs_to_go2.py \\
      --input output/3ddogs_coordinate_validation/d29_t1_a/retarget_inputs_mujoco.npz \\
      --output-dir output/go2_retarget/d29_t1_a --plot
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np

if TYPE_CHECKING:
  from tools.smp_tools.retargeting.inspect_go2_kinematics import (
    DEFAULT_GO2_XML,
    NAZARITE_BASE_POS,
    NAZARITE_JOINT_POS,
    SMP_LEG_ORDER,
    _joint_names,
    _object_id,
    _set_nazarite_default_pose,
  )
else:
  from inspect_go2_kinematics import (
    DEFAULT_GO2_XML,
    NAZARITE_BASE_POS,
    NAZARITE_JOINT_POS,
    SMP_LEG_ORDER,
    _joint_names,
    _object_id,
    _set_nazarite_default_pose,
  )


def _normalize(vectors: np.ndarray) -> np.ndarray:
  lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
  if np.any(lengths <= 1e-9):
    raise ValueError("cannot build root orientation from zero-length marker vectors")
  return vectors / lengths


def _root_rotations(forward: np.ndarray, left: np.ndarray) -> np.ndarray:
  """Build world-from-body rotation matrices with columns forward, left, up."""
  forward_unit = _normalize(forward)
  left_orthogonal = left - np.sum(left * forward_unit, axis=-1, keepdims=True) * forward_unit
  left_unit = _normalize(left_orthogonal)
  up_unit = _normalize(np.cross(forward_unit, left_unit))
  return np.stack((forward_unit, left_unit, up_unit), axis=-1)


def _matrix_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
  """Convert one right-handed rotation matrix to MuJoCo's wxyz quaternion."""
  trace = float(np.trace(rotation))
  if trace > 0.0:
    scale = 2.0 * np.sqrt(trace + 1.0)
    quaternion = np.array(
      (
        0.25 * scale,
        (rotation[2, 1] - rotation[1, 2]) / scale,
        (rotation[0, 2] - rotation[2, 0]) / scale,
        (rotation[1, 0] - rotation[0, 1]) / scale,
      )
    )
  elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
    scale = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
    quaternion = np.array(
      (
        (rotation[2, 1] - rotation[1, 2]) / scale,
        0.25 * scale,
        (rotation[0, 1] + rotation[1, 0]) / scale,
        (rotation[0, 2] + rotation[2, 0]) / scale,
      )
    )
  elif rotation[1, 1] > rotation[2, 2]:
    scale = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
    quaternion = np.array(
      (
        (rotation[0, 2] - rotation[2, 0]) / scale,
        (rotation[0, 1] + rotation[1, 0]) / scale,
        0.25 * scale,
        (rotation[1, 2] + rotation[2, 1]) / scale,
      )
    )
  else:
    scale = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
    quaternion = np.array(
      (
        (rotation[1, 0] - rotation[0, 1]) / scale,
        (rotation[0, 2] + rotation[2, 0]) / scale,
        (rotation[1, 2] + rotation[2, 1]) / scale,
        0.25 * scale,
      )
    )
  return quaternion / np.linalg.norm(quaternion)


def _load_canonical_motion(input_path: Path) -> dict[str, np.ndarray]:
  required_keys = {
    "frame_numbers",
    "root_pos_mujoco",
    "root_forward_mujoco",
    "root_left_mujoco",
    "paw_pos_mujoco",
    "fps",
    "paw_order",
  }
  with np.load(input_path, allow_pickle=False) as archive:
    missing = sorted(required_keys - set(archive.files))
    if missing:
      raise ValueError(f"canonical NPZ lacks required arrays: {', '.join(missing)}")
    motion = {key: archive[key].copy() for key in required_keys}
  if motion["paw_pos_mujoco"].ndim != 3 or motion["paw_pos_mujoco"].shape[1:] != (4, 3):
    raise ValueError("paw_pos_mujoco must have shape [frames, 4, 3]")
  if motion["root_pos_mujoco"].shape != (motion["paw_pos_mujoco"].shape[0], 3):
    raise ValueError("root_pos_mujoco must have shape [frames, 3]")
  paw_order = tuple(str(value) for value in motion["paw_order"].tolist())
  if paw_order != SMP_LEG_ORDER:
    raise ValueError(f"expected paw order {SMP_LEG_ORDER}, got {paw_order}")
  if not all(np.isfinite(value).all() for value in motion.values() if value.dtype.kind in "fiu"):
    raise ValueError("canonical NPZ contains non-finite values")
  return motion


def _go2_defaults(
  model: mujoco.MjModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Return default qpos, joint addresses/bounds, plus hip and foot offsets."""
  data = mujoco.MjData(model)
  _set_nazarite_default_pose(model, data)
  joint_names = _joint_names()
  joint_ids = [
    _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    for joint_name in joint_names
  ]
  joint_qpos_addresses = np.asarray(
    [model.jnt_qposadr[joint_id] for joint_id in joint_ids], dtype=np.intp
  )
  joint_dof_addresses = np.asarray(
    [model.jnt_dofadr[joint_id] for joint_id in joint_ids], dtype=np.intp
  )
  joint_bounds = np.asarray([model.jnt_range[joint_id] for joint_id in joint_ids])
  base_id = _object_id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
  base_position = data.xpos[base_id]
  hip_offsets = np.stack(
    [
      data.xpos[_object_id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_hip")]
      - base_position
      for leg in SMP_LEG_ORDER
    ]
  )
  foot_offsets = np.stack(
    [
      data.site_xpos[_object_id(model, mujoco.mjtObj.mjOBJ_SITE, leg)] - base_position
      for leg in SMP_LEG_ORDER
    ]
  )
  return (
    data.qpos.copy(),
    joint_qpos_addresses,
    joint_dof_addresses,
    joint_bounds,
    hip_offsets,
    foot_offsets,
  )


def _free_qpos_address(model: mujoco.MjModel) -> int:
  free_joint_ids = [
    joint_id
    for joint_id in range(model.njnt)
    if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
  ]
  if len(free_joint_ids) != 1:
    raise ValueError(f"expected one Go2 free joint, found {len(free_joint_ids)}")
  return int(model.jnt_qposadr[free_joint_ids[0]])


def _set_frame_qpos(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  template_qpos: np.ndarray,
  free_qpos_address: int,
  joint_qpos_addresses: np.ndarray,
  root_position: np.ndarray,
  root_quaternion: np.ndarray,
  joint_position: np.ndarray,
) -> None:
  data.qpos[:] = template_qpos
  data.qpos[free_qpos_address : free_qpos_address + 3] = root_position
  data.qpos[free_qpos_address + 3 : free_qpos_address + 7] = root_quaternion
  data.qpos[joint_qpos_addresses] = joint_position
  mujoco.mj_forward(model, data)


def _solve_foot_ik(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  template_qpos: np.ndarray,
  free_qpos_address: int,
  joint_qpos_addresses: np.ndarray,
  joint_dof_addresses: np.ndarray,
  joint_bounds: np.ndarray,
  foot_site_ids: np.ndarray,
  root_position: np.ndarray,
  root_quaternion: np.ndarray,
  target_foot_positions: np.ndarray,
  initial_joint_position: np.ndarray,
  damping: float,
  max_iterations: int,
  tolerance: float,
) -> tuple[np.ndarray, np.ndarray, int]:
  """Solve all four foot targets with a bounded damped-least-squares update."""
  joint_position = np.clip(initial_joint_position, joint_bounds[:, 0], joint_bounds[:, 1])
  jacobian_position = np.empty((3, model.nv), dtype=np.float64)
  jacobian_rotation = np.empty((3, model.nv), dtype=np.float64)
  for iteration in range(max_iterations):
    _set_frame_qpos(
      model,
      data,
      template_qpos,
      free_qpos_address,
      joint_qpos_addresses,
      root_position,
      root_quaternion,
      joint_position,
    )
    solved_foot_positions = data.site_xpos[foot_site_ids].copy()
    position_error = target_foot_positions - solved_foot_positions
    if float(np.max(np.linalg.norm(position_error, axis=1))) <= tolerance:
      return joint_position, position_error, iteration

    jacobian = np.empty((12, 12), dtype=np.float64)
    for foot_index, site_id in enumerate(foot_site_ids):
      mujoco.mj_jacSite(model, data, jacobian_position, jacobian_rotation, int(site_id))
      jacobian[3 * foot_index : 3 * foot_index + 3] = jacobian_position[
        :, joint_dof_addresses
      ]
    error_vector = position_error.reshape(-1)
    regularized = jacobian @ jacobian.T + damping**2 * np.eye(jacobian.shape[0])
    delta = jacobian.T @ np.linalg.solve(regularized, error_vector)
    joint_position = np.clip(
      joint_position + delta, joint_bounds[:, 0], joint_bounds[:, 1]
    )

  _set_frame_qpos(
    model,
    data,
    template_qpos,
    free_qpos_address,
    joint_qpos_addresses,
    root_position,
    root_quaternion,
    joint_position,
  )
  position_error = target_foot_positions - data.site_xpos[foot_site_ids]
  return joint_position, position_error, max_iterations


def _retarget_targets(
  motion: dict[str, np.ndarray], foot_offsets: np.ndarray, motion_scale: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Map dog root/paw motion into a Go2-sized root and four foot targets."""
  root_position = motion["root_pos_mujoco"]
  rotations_dog = _root_rotations(motion["root_forward_mujoco"], motion["root_left_mujoco"])
  rotations_go2 = rotations_dog @ rotations_dog[0].T
  root_quaternion = np.stack(
    [_matrix_to_quaternion_wxyz(rotation) for rotation in rotations_go2]
  )
  target_root_position = (
    NAZARITE_BASE_POS + motion_scale * (root_position - root_position[0])
  )
  paws_relative_body = np.einsum(
    "tji,tlj->tli", rotations_dog, motion["paw_pos_mujoco"] - root_position[:, None]
  )
  neutral_paws_relative_body = np.median(paws_relative_body, axis=0)
  target_paws_relative_body = foot_offsets[None] + motion_scale * (
    paws_relative_body - neutral_paws_relative_body[None]
  )
  target_foot_positions = target_root_position[:, None] + np.einsum(
    "tij,tlj->tli", rotations_go2, target_paws_relative_body
  )
  return target_root_position, root_quaternion, target_foot_positions, rotations_go2


def _automatic_motion_scale(motion: dict[str, np.ndarray], hip_offsets: np.ndarray) -> float:
  dog_body_length = float(np.median(np.linalg.norm(motion["root_forward_mujoco"], axis=1)))
  go2_body_length = float(
    0.5 * (hip_offsets[0, 0] + hip_offsets[1, 0])
    - 0.5 * (hip_offsets[2, 0] + hip_offsets[3, 0])
  )
  if dog_body_length <= 1e-6:
    raise ValueError("cannot derive scale: dog trunk length is zero")
  return go2_body_length / dog_body_length


def _plot_retargeting(
  output_path: Path,
  targets: np.ndarray,
  solved: np.ndarray,
) -> None:
  """Plot target versus solved Go2 foot paths in forward-height view."""
  os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".matplotlib"))
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  figure, axes = plt.subplots(figsize=(8, 5))
  colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")
  for foot_index, leg in enumerate(SMP_LEG_ORDER):
    axes.plot(
      targets[:, foot_index, 0],
      targets[:, foot_index, 2],
      color=colors[foot_index],
      linestyle="--",
      label=f"{leg} target",
    )
    axes.plot(
      solved[:, foot_index, 0],
      solved[:, foot_index, 2],
      color=colors[foot_index],
      label=f"{leg} IK",
    )
  axes.set_xlabel("x / forward (m)")
  axes.set_ylabel("z / up (m)")
  axes.set_title("Go2 retargeting: desired (dashed) versus IK feet (solid)")
  axes.grid(alpha=0.3)
  axes.legend(ncol=2, fontsize=8)
  figure.tight_layout()
  figure.savefig(output_path, dpi=180)
  plt.close(figure)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", type=Path, required=True, help="Canonical 3DDogs MuJoCo NPZ")
  parser.add_argument("--output-dir", type=Path, required=True, help="Derived Go2 reference directory")
  parser.add_argument("--xml", type=Path, default=DEFAULT_GO2_XML, help="Go2 MuJoCo XML")
  parser.add_argument(
    "--motion-scale",
    type=float,
    help="Scale dog deviations and root travel; default derives Go2/dog torso scale",
  )
  parser.add_argument("--damping", type=float, default=0.02, help="DLS IK damping")
  parser.add_argument("--max-iterations", type=int, default=80, help="Maximum IK updates/frame")
  parser.add_argument(
    "--success-tolerance-m", type=float, default=0.015, help="Maximum foot error for success"
  )
  parser.add_argument("--plot", action="store_true", help="Save target-versus-IK trajectory PNG")
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
  args = _parse_args(argv)
  if args.motion_scale is not None and args.motion_scale <= 0.0:
    raise ValueError("--motion-scale must be positive")
  if args.damping <= 0.0 or args.max_iterations <= 0 or args.success_tolerance_m <= 0.0:
    raise ValueError("damping, max-iterations, and success-tolerance-m must be positive")

  input_path: Path = args.input.expanduser().resolve()
  xml_path: Path = args.xml.expanduser().resolve()
  if not input_path.is_file():
    raise FileNotFoundError(input_path)
  if not xml_path.is_file():
    raise FileNotFoundError(xml_path)
  output_dir: Path = args.output_dir.expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

  motion = _load_canonical_motion(input_path)
  model = mujoco.MjModel.from_xml_path(str(xml_path))
  (
    template_qpos,
    joint_qpos_addresses,
    joint_dof_addresses,
    joint_bounds,
    hip_offsets,
    foot_offsets,
  ) = _go2_defaults(model)
  motion_scale = args.motion_scale or _automatic_motion_scale(motion, hip_offsets)
  root_position, root_quaternion, foot_targets, root_rotation = _retarget_targets(
    motion, foot_offsets, motion_scale
  )
  free_qpos_address = _free_qpos_address(model)
  foot_site_ids = np.asarray(
    [_object_id(model, mujoco.mjtObj.mjOBJ_SITE, leg) for leg in SMP_LEG_ORDER],
    dtype=np.intp,
  )
  data = mujoco.MjData(model)
  initial_joint_position = np.asarray([NAZARITE_JOINT_POS[name] for name in _joint_names()])
  frame_count = root_position.shape[0]
  joint_position = np.empty((frame_count, 12), dtype=np.float64)
  foot_position = np.empty((frame_count, 4, 3), dtype=np.float64)
  foot_error = np.empty((frame_count, 4, 3), dtype=np.float64)
  iterations = np.empty(frame_count, dtype=np.int64)
  qpos = np.empty((frame_count, model.nq), dtype=np.float64)
  previous_joint_position = initial_joint_position
  for frame_index in range(frame_count):
    solved_joint_position, error, iteration_count = _solve_foot_ik(
      model=model,
      data=data,
      template_qpos=template_qpos,
      free_qpos_address=free_qpos_address,
      joint_qpos_addresses=joint_qpos_addresses,
      joint_dof_addresses=joint_dof_addresses,
      joint_bounds=joint_bounds,
      foot_site_ids=foot_site_ids,
      root_position=root_position[frame_index],
      root_quaternion=root_quaternion[frame_index],
      target_foot_positions=foot_targets[frame_index],
      initial_joint_position=previous_joint_position,
      damping=args.damping,
      max_iterations=args.max_iterations,
      tolerance=args.success_tolerance_m,
    )
    _set_frame_qpos(
      model,
      data,
      template_qpos,
      free_qpos_address,
      joint_qpos_addresses,
      root_position[frame_index],
      root_quaternion[frame_index],
      solved_joint_position,
    )
    joint_position[frame_index] = solved_joint_position
    foot_position[frame_index] = data.site_xpos[foot_site_ids]
    foot_error[frame_index] = error
    iterations[frame_index] = iteration_count
    qpos[frame_index] = data.qpos
    previous_joint_position = solved_joint_position

  foot_error_norm = np.linalg.norm(foot_error, axis=-1)
  frame_success = np.max(foot_error_norm, axis=1) <= args.success_tolerance_m
  summary: dict[str, Any] = {
    "source": str(input_path),
    "go2_xml": str(xml_path),
    "fps": float(np.asarray(motion["fps"]).item()),
    "frames": frame_count,
    "duration_s": frame_count / float(np.asarray(motion["fps"]).item()),
    "smp_leg_order": list(SMP_LEG_ORDER),
    "smp_joint_order": list(_joint_names()),
    "motion_scale": motion_scale,
    "damping": args.damping,
    "max_iterations": args.max_iterations,
    "success_tolerance_m": args.success_tolerance_m,
    "successful_frames": int(np.count_nonzero(frame_success)),
    "success_fraction": float(np.mean(frame_success)),
    "max_foot_error_m": float(np.max(foot_error_norm)),
    "mean_foot_error_m": float(np.mean(foot_error_norm)),
    "mean_ik_iterations": float(np.mean(iterations)),
  }
  np.savez_compressed(
    output_dir / "go2_reference_geometric.npz",
    allow_pickle=False,
    source_frame_numbers=motion["frame_numbers"],
    fps=motion["fps"],
    smp_leg_order=np.asarray(SMP_LEG_ORDER),
    smp_joint_order=np.asarray(_joint_names()),
    root_pos_mujoco=root_position,
    root_quat_wxyz=root_quaternion,
    root_rotation_mujoco=root_rotation,
    joint_pos_rad=joint_position,
    qpos=qpos,
    foot_target_world_m=foot_targets,
    foot_target_base_m=np.einsum(
      "tli,tij->tlj",
      foot_targets - root_position[:, None, :],
      root_rotation,
    ),
    foot_position_world_m=foot_position,
    foot_error_world_m=foot_error,
    ik_iterations=iterations,
    ik_success=frame_success,
  )
  with (output_dir / "retarget_summary.json").open("w", encoding="utf-8") as stream:
    json.dump(summary, stream, indent=2)
    stream.write("\n")
  if args.plot:
    _plot_retargeting(output_dir / "foot_target_vs_ik.png", foot_targets, foot_position)

  print(json.dumps(summary, indent=2))
  print(f"\nWrote: {output_dir / 'go2_reference_geometric.npz'}")
  print(f"Wrote: {output_dir / 'retarget_summary.json'}")
  if args.plot:
    print(f"Wrote: {output_dir / 'foot_target_vs_ik.png'}")


if __name__ == "__main__":
  main()
