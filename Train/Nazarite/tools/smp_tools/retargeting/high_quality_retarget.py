"""Contact-aware, GQMR-style 3DDogs to Go2 retargeting.

This is deliberately a parallel implementation of the original geometric
retargeter.  It keeps the same Nazarite NPZ schema while adding the parts of
GQMR that are most useful for animal motion: contact-aware foot anchors,
bounded DLS updates, median-window refinement, and explicit solver status.

Example::

    uv run python tools/smp_tools/retargeting/high_quality_retarget.py \
      --input output/3ddogs_coordinate_validation/d29_t1_a/retarget_inputs_mujoco.npz \
      --output-dir output/go2_retarget_gqmr_style/d29_t1_a
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
  from tools.smp_tools.retargeting.inspect_go2_kinematics import (
    DEFAULT_GO2_XML,
    NAZARITE_BASE_POS,
    NAZARITE_JOINT_POS,
    SMP_LEG_ORDER,
    _joint_names,
    _object_id,
    _set_nazarite_default_pose,
  )
  from tools.smp_tools.retargeting.retarget_3ddogs_to_go2 import (
    _free_qpos_address,
    _go2_defaults,
    _load_canonical_motion,
    _automatic_motion_scale,
    _retarget_targets,
    _set_frame_qpos,
  )
except ModuleNotFoundError:
  # Also support running this file directly from tools/smp_tools/retargeting.
  sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
  from tools.smp_tools.retargeting.inspect_go2_kinematics import (
    DEFAULT_GO2_XML,
    NAZARITE_BASE_POS,
    NAZARITE_JOINT_POS,
    SMP_LEG_ORDER,
    _joint_names,
    _object_id,
    _set_nazarite_default_pose,
  )
  from tools.smp_tools.retargeting.retarget_3ddogs_to_go2 import (
    _free_qpos_address,
    _go2_defaults,
    _load_canonical_motion,
    _automatic_motion_scale,
    _retarget_targets,
    _set_frame_qpos,
  )


def _finite_difference(values: np.ndarray, fps: float) -> np.ndarray:
  if values.shape[0] < 2:
    return np.zeros_like(values, dtype=np.float64)
  return np.gradient(values, 1.0 / fps, axis=0)


def _valid_contact_runs(mask: np.ndarray, minimum_length: int) -> list[tuple[int, int]]:
  """Return contiguous true runs that are long enough to be stance phases."""
  runs: list[tuple[int, int]] = []
  start: int | None = None
  for index, value in enumerate(np.asarray(mask, dtype=bool)):
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
  paw_positions: np.ndarray,
  fps: float,
  height_threshold: float,
  speed_threshold: float,
  minimum_length: int,
) -> tuple[np.ndarray, np.ndarray, float]:
  """Estimate contact probabilities from paw height and speed.

  The input is in the canonical MuJoCo frame.  The ground estimate is the
  lower 10th percentile of all paw heights, which is robust to a moving root
  and avoids assuming that the source clip was recorded at z=0.
  """
  paw_speed = np.linalg.norm(_finite_difference(paw_positions, fps), axis=-1)
  ground_height = float(np.percentile(paw_positions[..., 2], 10.0))
  height_score = np.clip(
    1.0 - (paw_positions[..., 2] - ground_height) / max(height_threshold, 1.0e-6),
    0.0,
    1.0,
  )
  speed_score = np.clip(
    1.0 - paw_speed / max(speed_threshold, 1.0e-6),
    0.0,
    1.0,
  )
  probability = height_score * speed_score
  candidate = (height_score >= 0.5) & (speed_score >= 0.5)
  contact = np.zeros_like(candidate, dtype=bool)
  for leg_index in range(candidate.shape[1]):
    for start, end in _valid_contact_runs(candidate[:, leg_index], minimum_length):
      contact[start:end, leg_index] = True
  return contact, probability.astype(np.float64), ground_height


def _apply_contact_anchors(
  foot_targets: np.ndarray,
  contact: np.ndarray,
  anchor_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
  """Apply a soft robust anchor to each contiguous stance phase.

  ``anchor_strength=1`` reproduces hard contact locking.  Lower values retain
  some measured paw motion, which is useful when the optical contact estimate
  is correct but the exact anchor is not dynamically reachable by Go2.
  """
  anchored = foot_targets.copy()
  anchor_error = np.zeros_like(foot_targets)
  for leg_index in range(foot_targets.shape[1]):
    for start, end in _valid_contact_runs(contact[:, leg_index], minimum_length=1):
      anchor = np.median(foot_targets[start:end, leg_index], axis=0)
      anchor_error[start:end, leg_index] = foot_targets[start:end, leg_index] - anchor
      anchored[start:end, leg_index] = (
        foot_targets[start:end, leg_index]
        + anchor_strength * (anchor - foot_targets[start:end, leg_index])
      )
  return anchored, anchor_error


def _median_filter(values: np.ndarray, window: int) -> np.ndarray:
  if window <= 1:
    return values.copy()
  radius = window // 2
  padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
  result = np.empty_like(values)
  for frame_index in range(values.shape[0]):
    result[frame_index] = np.median(
      padded[frame_index : frame_index + window], axis=0
    )
  return result


def _solve_foot_ik_bounded(
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
  max_joint_step: float,
  unreachable_error: float,
) -> tuple[np.ndarray, np.ndarray, int, str]:
  """Solve one frame with DLS, step clipping, and explicit status."""
  joint_position = np.clip(initial_joint_position, joint_bounds[:, 0], joint_bounds[:, 1])
  jacobian_position = np.empty((3, model.nv), dtype=np.float64)
  jacobian_rotation = np.empty((3, model.nv), dtype=np.float64)
  position_error = np.zeros((4, 3), dtype=np.float64)
  for iteration in range(1, max_iterations + 1):
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
    max_error = float(np.max(np.linalg.norm(position_error, axis=1)))
    if max_error <= tolerance:
      return joint_position, position_error, iteration, "SUCCESS"

    jacobian = np.empty((12, 12), dtype=np.float64)
    for foot_index, site_id in enumerate(foot_site_ids):
      mujoco.mj_jacSite(model, data, jacobian_position, jacobian_rotation, int(site_id))
      jacobian[3 * foot_index : 3 * foot_index + 3] = jacobian_position[:, joint_dof_addresses]
    error_vector = position_error.reshape(-1)
    regularized = jacobian @ jacobian.T + damping**2 * np.eye(12)
    delta = jacobian.T @ np.linalg.solve(regularized, error_vector)
    delta = np.clip(delta, -max_joint_step, max_joint_step)
    joint_position = np.clip(
      joint_position + delta,
      joint_bounds[:, 0],
      joint_bounds[:, 1],
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
  max_error = float(np.max(np.linalg.norm(position_error, axis=1)))
  status = "UNREACHABLE" if max_error > unreachable_error else "MAX_ITER"
  return joint_position, position_error, max_iterations, status


def _solve_sequence(
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
  foot_targets: np.ndarray,
  initial_joint_position: np.ndarray,
  damping: float,
  max_iterations: int,
  tolerance: float,
  max_joint_step: float,
  unreachable_error: float,
  previous_solution: np.ndarray | None = None,
  initial_positions: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  frame_count = root_position.shape[0]
  joint_position = np.empty((frame_count, 12), dtype=np.float64)
  foot_position = np.empty((frame_count, 4, 3), dtype=np.float64)
  foot_error = np.empty((frame_count, 4, 3), dtype=np.float64)
  iterations = np.empty(frame_count, dtype=np.int64)
  statuses = np.empty(frame_count, dtype="U16")
  previous = initial_joint_position if previous_solution is None else previous_solution
  for frame_index in range(frame_count):
    frame_initial = (
      initial_positions[frame_index]
      if initial_positions is not None
      else previous
    )
    solved, error, iteration_count, status = _solve_foot_ik_bounded(
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
      initial_joint_position=frame_initial,
      damping=damping,
      max_iterations=max_iterations,
      tolerance=tolerance,
      max_joint_step=max_joint_step,
      unreachable_error=unreachable_error,
    )
    _set_frame_qpos(
      model,
      data,
      template_qpos,
      free_qpos_address,
      joint_qpos_addresses,
      root_position[frame_index],
      root_quaternion[frame_index],
      solved,
    )
    joint_position[frame_index] = solved
    foot_position[frame_index] = data.site_xpos[foot_site_ids]
    foot_error[frame_index] = error
    iterations[frame_index] = iteration_count
    statuses[frame_index] = status
    previous = solved
  return joint_position, foot_position, foot_error, iterations, statuses


def _repair_failed_frames(
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
  anchored_targets: np.ndarray,
  raw_targets: np.ndarray,
  joint_position: np.ndarray,
  foot_position: np.ndarray,
  foot_error: np.ndarray,
  iterations: np.ndarray,
  statuses: np.ndarray,
  damping: float,
  max_iterations: int,
  tolerance: float,
  max_joint_step: float,
  unreachable_error: float,
) -> int:
  """Retry non-converged frames from a temporal-neighbourhood seed.

  A failed frame is first solved against a halfway target between the
  contact-anchored and raw target.  This avoids allowing one impossible
  anchor to create a large joint discontinuity.  Only a successful retry is
  marked ``REPAIRED``; unresolved frames remain explicit ``MAX_ITER`` or
  ``UNREACHABLE`` states.
  """
  repaired_count = 0
  failed_indices = np.flatnonzero(statuses != "SUCCESS")
  for frame_index in failed_indices:
    if frame_index == 0:
      seed = joint_position[1]
    elif frame_index == len(joint_position) - 1:
      seed = joint_position[-2]
    else:
      seed = np.median(
        np.stack((joint_position[frame_index - 1], joint_position[frame_index + 1])),
        axis=0,
      )
    retry_target = 0.5 * (anchored_targets[frame_index] + raw_targets[frame_index])
    solved, error, retry_iterations, retry_status = _solve_foot_ik_bounded(
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
      target_foot_positions=retry_target,
      initial_joint_position=seed,
      damping=damping * 1.25,
      max_iterations=max(max_iterations * 2, 160),
      tolerance=tolerance,
      max_joint_step=max_joint_step * 1.25,
      unreachable_error=unreachable_error,
    )
    if retry_status != "SUCCESS":
      continue
    _set_frame_qpos(
      model,
      data,
      template_qpos,
      free_qpos_address,
      joint_qpos_addresses,
      root_position[frame_index],
      root_quaternion[frame_index],
      solved,
    )
    joint_position[frame_index] = solved
    foot_position[frame_index] = data.site_xpos[foot_site_ids]
    # Report residual against the final anchored target, which is what the
    # teacher will eventually be asked to track.
    foot_error[frame_index] = anchored_targets[frame_index] - foot_position[frame_index]
    iterations[frame_index] = retry_iterations
    statuses[frame_index] = "REPAIRED"
    repaired_count += 1
  return repaired_count


def _plot_retargeting(output_path: Path, targets: np.ndarray, solved: np.ndarray, contact: np.ndarray) -> None:
  import os

  os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".matplotlib"))
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  figure, axes = plt.subplots(figsize=(8, 5))
  colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")
  for foot_index, leg in enumerate(SMP_LEG_ORDER):
    axes.plot(targets[:, foot_index, 0], targets[:, foot_index, 2], "--", color=colors[foot_index], label=f"{leg} target")
    axes.plot(solved[:, foot_index, 0], solved[:, foot_index, 2], color=colors[foot_index], label=f"{leg} IK")
    contact_indices = np.flatnonzero(contact[:, foot_index])
    if contact_indices.size:
      axes.scatter(
        solved[contact_indices, foot_index, 0],
        solved[contact_indices, foot_index, 2],
        color=colors[foot_index],
        s=8,
        alpha=0.45,
      )
  axes.set_xlabel("x / forward (m)")
  axes.set_ylabel("z / up (m)")
  axes.set_title("GQMR-style retargeting: target versus IK feet")
  axes.grid(alpha=0.3)
  axes.legend(ncol=2, fontsize=8)
  figure.tight_layout()
  figure.savefig(output_path, dpi=180)
  plt.close(figure)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", type=Path, required=True, help="Canonical 3DDogs MuJoCo NPZ")
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--xml", type=Path, default=DEFAULT_GO2_XML)
  parser.add_argument("--motion-scale", type=float)
  parser.add_argument("--damping", type=float, default=0.02)
  parser.add_argument("--max-iterations", type=int, default=100)
  parser.add_argument("--success-tolerance-m", type=float, default=0.010)
  parser.add_argument("--unreachable-error-m", type=float, default=0.040)
  parser.add_argument("--max-joint-step-rad", type=float, default=0.12)
  parser.add_argument("--refinement-passes", type=int, default=2)
  parser.add_argument("--median-window", type=int, default=5)
  parser.add_argument("--contact-height-threshold-m", type=float, default=0.025)
  parser.add_argument("--contact-speed-threshold-mps", type=float, default=0.25)
  parser.add_argument("--minimum-contact-frames", type=int, default=3)
  parser.add_argument(
    "--contact-anchor-strength",
    type=float,
    default=0.35,
    help="Soft contact anchor strength in [0, 1]; 1 is hard locking",
  )
  parser.add_argument(
    "--ground-clearance-m",
    type=float,
    default=0.005,
    help="Lift the whole retargeted motion so the lowest foot clears the plane",
  )
  parser.add_argument("--plot", action="store_true")
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
  args = _parse_args(argv)
  if args.motion_scale is not None and args.motion_scale <= 0.0:
    raise ValueError("--motion-scale must be positive")
  if args.damping <= 0.0 or args.max_iterations <= 0 or args.refinement_passes < 0:
    raise ValueError("damping, max-iterations, and refinement-passes must be valid")
  if args.median_window < 1 or args.median_window % 2 == 0:
    raise ValueError("--median-window must be a positive odd integer")
  if args.ground_clearance_m < 0.0:
    raise ValueError("--ground-clearance-m must be non-negative")
  if not 0.0 <= args.contact_anchor_strength <= 1.0:
    raise ValueError("--contact-anchor-strength must be in [0, 1]")
  input_path = args.input.expanduser().resolve()
  xml_path = args.xml.expanduser().resolve()
  output_dir = args.output_dir.expanduser().resolve()
  if not input_path.is_file() or not xml_path.is_file():
    raise FileNotFoundError("input canonical NPZ and Go2 XML must exist")
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
  fps = float(np.asarray(motion["fps"]).item())
  motion_scale = args.motion_scale or _automatic_motion_scale(motion, hip_offsets)
  root_position, root_quaternion, raw_targets, root_rotation = _retarget_targets(
    motion, foot_offsets, motion_scale
  )
  # The source optical clip is not guaranteed to use z=0 as the floor.  A
  # common vertical translation preserves the motion while preventing the
  # generated Go2 feet from starting below Nazarite's training plane.
  ground_lift = max(0.0, args.ground_clearance_m - float(np.min(raw_targets[..., 2])))
  root_position = root_position.copy()
  raw_targets = raw_targets.copy()
  root_position[:, 2] += ground_lift
  raw_targets[..., 2] += ground_lift
  contact, contact_probability, ground_height = _estimate_contacts(
    motion["paw_pos_mujoco"],
    fps,
    args.contact_height_threshold_m,
    args.contact_speed_threshold_mps,
    args.minimum_contact_frames,
  )
  foot_targets, anchor_error = _apply_contact_anchors(
    raw_targets,
    contact,
    args.contact_anchor_strength,
  )

  free_qpos_address = _free_qpos_address(model)
  foot_site_ids = np.asarray(
    [_object_id(model, mujoco.mjtObj.mjOBJ_SITE, leg) for leg in SMP_LEG_ORDER],
    dtype=np.intp,
  )
  data = mujoco.MjData(model)
  initial_joint_position = np.asarray([NAZARITE_JOINT_POS[name] for name in _joint_names()])
  joint_position, foot_position, foot_error, iterations, statuses = _solve_sequence(
    model,
    data,
    template_qpos,
    free_qpos_address,
    joint_qpos_addresses,
    joint_dof_addresses,
    joint_bounds,
    foot_site_ids,
    root_position,
    root_quaternion,
    foot_targets,
    initial_joint_position,
    args.damping,
    args.max_iterations,
    args.success_tolerance_m,
    args.max_joint_step_rad,
    args.unreachable_error_m,
  )

  for _ in range(args.refinement_passes):
    refined_initial = _median_filter(joint_position, args.median_window)
    joint_position, foot_position, foot_error, iterations, statuses = _solve_sequence(
      model,
      data,
      template_qpos,
      free_qpos_address,
      joint_qpos_addresses,
      joint_dof_addresses,
      joint_bounds,
      foot_site_ids,
      root_position,
      root_quaternion,
      foot_targets,
      refined_initial[0],
      args.damping,
      args.max_iterations,
      args.success_tolerance_m,
      args.max_joint_step_rad,
      args.unreachable_error_m,
      previous_solution=refined_initial[0],
      initial_positions=refined_initial,
    )

  repaired_frames = _repair_failed_frames(
    model=model,
    data=data,
    template_qpos=template_qpos,
    free_qpos_address=free_qpos_address,
    joint_qpos_addresses=joint_qpos_addresses,
    joint_dof_addresses=joint_dof_addresses,
    joint_bounds=joint_bounds,
    foot_site_ids=foot_site_ids,
    root_position=root_position,
    root_quaternion=root_quaternion,
    anchored_targets=foot_targets,
    raw_targets=raw_targets,
    joint_position=joint_position,
    foot_position=foot_position,
    foot_error=foot_error,
    iterations=iterations,
    statuses=statuses,
    damping=args.damping,
    max_iterations=args.max_iterations,
    tolerance=args.success_tolerance_m,
    max_joint_step=args.max_joint_step_rad,
    unreachable_error=args.unreachable_error_m,
  )

  qpos = np.empty((root_position.shape[0], model.nq), dtype=np.float64)
  for frame_index in range(root_position.shape[0]):
    _set_frame_qpos(
      model,
      data,
      template_qpos,
      free_qpos_address,
      joint_qpos_addresses,
      root_position[frame_index],
      root_quaternion[frame_index],
      joint_position[frame_index],
    )
    qpos[frame_index] = data.qpos

  foot_error_norm = np.linalg.norm(foot_error, axis=-1)
  foot_target_base = np.einsum(
    "tli,tij->tlj",
    foot_targets - root_position[:, None, :],
    root_rotation,
  )
  valid_status = np.isin(statuses, ("SUCCESS", "REPAIRED"))
  status_counts = {str(status): int(np.count_nonzero(statuses == status)) for status in np.unique(statuses)}
  summary: dict[str, Any] = {
    "source": str(input_path),
    "go2_xml": str(xml_path),
    "method": "gqmr_style_contact_aware_dls",
    "fps": fps,
    "frames": int(root_position.shape[0]),
    "duration_s": float((root_position.shape[0] - 1) / fps),
    "motion_scale": float(motion_scale),
    "contact_ground_height_m": ground_height,
    "ground_lift_m": ground_lift,
    "contact_fraction_by_leg": np.mean(contact, axis=0).tolist(),
    "contact_anchor_raw_error_mean_m": float(np.mean(np.linalg.norm(anchor_error, axis=-1))),
    "contact_anchor_strength": args.contact_anchor_strength,
    "damping": args.damping,
    "max_iterations": args.max_iterations,
    "success_tolerance_m": args.success_tolerance_m,
    "unreachable_error_m": args.unreachable_error_m,
    "max_joint_step_rad": args.max_joint_step_rad,
    "refinement_passes": args.refinement_passes,
    "median_window": args.median_window,
    "successful_frames": int(np.count_nonzero(valid_status)),
    "repaired_frames": repaired_frames,
    "success_fraction": float(np.mean(valid_status)),
    "max_foot_error_m": float(np.max(foot_error_norm)),
    "mean_foot_error_m": float(np.mean(foot_error_norm)),
    "foot_error_p95_m": float(np.percentile(foot_error_norm, 95.0)),
    "mean_ik_iterations": float(np.mean(iterations)),
    "solver_status_counts": status_counts,
  }
  np.savez_compressed(
    output_dir / "go2_reference_geometric.npz",
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
    foot_target_base_m=foot_target_base,
    foot_position_world_m=foot_position,
    foot_error_world_m=foot_error,
    ik_iterations=iterations,
    ik_success=valid_status,
    solver_status=statuses,
    solver_residual_m=np.max(foot_error_norm, axis=1),
    contact=contact,
    contact_probability=contact_probability,
    contact_anchor_error_world_m=anchor_error,
  )
  with (output_dir / "retarget_summary.json").open("w", encoding="utf-8") as stream:
    json.dump(summary, stream, indent=2)
    stream.write("\n")
  if args.plot:
    _plot_retargeting(output_dir / "foot_target_vs_ik.png", foot_targets, foot_position, contact)
  print(json.dumps(summary, indent=2))
  print(f"\nWrote: {output_dir / 'go2_reference_geometric.npz'}")
  print(f"Wrote: {output_dir / 'retarget_summary.json'}")
  if args.plot:
    print(f"Wrote: {output_dir / 'foot_target_vs_ik.png'}")


if __name__ == "__main__":
  main()
