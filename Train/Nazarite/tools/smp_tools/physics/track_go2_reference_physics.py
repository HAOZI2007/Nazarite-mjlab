"""Track a geometric Go2 reference with Nazarite's native position actuators.

This is an untrained physics-tracking baseline.  It creates a ground plane in
memory, recreates the Go2 position actuators configured by Nazarite, and saves the
actual simulated state plus a validity mask for later motion-prior filtering.

Example:

    cd Train/Nazarite
    uv run python tools/smp_tools/physics/track_go2_reference_physics.py \\
      --input output/go2_retarget/d29_t1_a/go2_reference_geometric.npz \\
      --output-dir output/go2_physics_tracking/d29_t1_a
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

from nazarite.config.robot_config.go2_cfg import (
  ARMATURE_CALF,
  ARMATURE_HIP,
  DAMPING_CALF,
  DAMPING_HIP,
  GO2_ACTION_SCALE,
  GO2_ACTUATOR_KD_SCALE,
  GO2_ACTUATOR_KP_SCALE,
  GO2_CALF_ACTUATOR_CFG,
  GO2_HIP_ACTUATOR_CFG,
  STIFFNESS_CALF,
  STIFFNESS_HIP,
)

try:
  from tools.smp_tools.retargeting.inspect_go2_kinematics import (
    DEFAULT_GO2_XML,
    NAZARITE_JOINT_POS,
  )
except ModuleNotFoundError:
  # Support direct execution from the repository root.
  sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
  from tools.smp_tools.retargeting.inspect_go2_kinematics import (
    DEFAULT_GO2_XML,
    NAZARITE_JOINT_POS,
  )

def _load_reference(input_path: Path) -> dict[str, np.ndarray]:
  required_keys = {
    "qpos",
    "fps",
    "root_pos_mujoco",
    "root_quat_wxyz",
    "foot_target_world_m",
    "smp_joint_order",
  }
  with np.load(input_path, allow_pickle=False) as archive:
    missing = sorted(required_keys - set(archive.files))
    if missing:
      raise ValueError(f"reference NPZ lacks required arrays: {', '.join(missing)}")
    reference = {key: archive[key].copy() for key in required_keys}
    if "foot_target_base_m" in archive.files:
      reference["foot_target_base_m"] = archive["foot_target_base_m"].copy()
  if reference["qpos"].ndim != 2 or reference["qpos"].shape[0] < 2:
    raise ValueError("reference qpos must have at least two [frames, nq] rows")
  if not np.isfinite(reference["qpos"]).all():
    raise ValueError("reference qpos contains non-finite values")
  if "foot_target_base_m" in reference and reference["foot_target_base_m"].shape != (
    reference["qpos"].shape[0],
    4,
    3,
  ):
    raise ValueError("foot_target_base_m must have shape [reference_frames, 4, 3]")
  if reference["root_pos_mujoco"].shape != (reference["qpos"].shape[0], 3):
    raise ValueError("root_pos_mujoco must have shape [reference_frames, 3]")
  if reference["root_quat_wxyz"].shape != (reference["qpos"].shape[0], 4):
    raise ValueError("root_quat_wxyz must have shape [reference_frames, 4]")
  if "foot_target_base_m" not in reference:
    root_rotation = np.empty((reference["qpos"].shape[0], 3, 3), dtype=np.float64)
    for frame_index, quaternion in enumerate(reference["root_quat_wxyz"]):
      mujoco.mju_quat2Mat(root_rotation[frame_index].reshape(-1), quaternion)
    reference["foot_target_base_m"] = (
      reference["foot_target_world_m"] - reference["root_pos_mujoco"][:, None, :]
    ) @ root_rotation
  return reference


def _initial_reference_velocity(model: mujoco.MjModel, reference: dict[str, np.ndarray]) -> np.ndarray:
  """Estimate qvel at the first frame using MuJoCo's manifold-aware helper."""
  qvel = np.zeros(model.nv, dtype=np.float64)
  reference_fps = float(np.asarray(reference["fps"]).item())
  mujoco.mj_differentiatePos(
    model,
    qvel,
    1.0 / reference_fps,
    reference["qpos"][0],
    reference["qpos"][1],
  )
  return qvel


def _add_nazarite_position_actuator(
  spec: mujoco.MjSpec,
  joint_name: str,
  stiffness: float,
  damping: float,
  effort_limit: float,
  armature: float,
) -> None:
  """Recreate mjlab's BuiltinPositionActuatorCfg for one Go2 joint."""
  actuator = spec.add_actuator(name=joint_name, target=joint_name)
  actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
  actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
  actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
  actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
  actuator.gainprm[0] = stiffness
  actuator.biasprm[1] = -stiffness
  actuator.biasprm[2] = -damping
  actuator.inheritrange = 0.0
  actuator.ctrllimited = False
  actuator.forcelimited = True
  actuator.forcerange[:] = np.array((-effort_limit, effort_limit))
  joint = spec.joint(joint_name)
  delta = effort_limit / stiffness
  actuator.ctrlrange[:] = np.array((joint.range[0] - delta, joint.range[1] + delta))
  joint.armature = armature


def _model_with_ground(xml_path: Path, kp_scale: float, kd_scale: float) -> mujoco.MjModel:
  """Compile Go2 with the *training* actuators and a plane, in memory.

  ``get_go2_cfg()`` removes XML actuators and mjlab compiles the two actuator
  groups below into native MuJoCo position actuators.  Read those configuration
  objects directly instead of duplicating their numerical values here, so an
  offline SMP rollout remains aligned with the trainable environment.
  """
  spec = mujoco.MjSpec.from_file(str(xml_path))
  # This is exactly what go2_cfg.get_spec() does before mjlab adds its two
  # BuiltinPositionActuatorCfg groups.
  for actuator in list(spec.actuators):
    spec.delete(actuator)
  for leg in ("FL", "FR", "RL", "RR"):
    for joint_kind in ("hip", "thigh", "calf"):
      actuator_cfg = GO2_CALF_ACTUATOR_CFG if joint_kind == "calf" else GO2_HIP_ACTUATOR_CFG
      if (
        actuator_cfg.stiffness is None
        or actuator_cfg.damping is None
        or actuator_cfg.effort_limit is None
        or actuator_cfg.armature is None
      ):
        raise ValueError("Go2 training actuator configuration must define Kp, Kd, effort, armature")
      _add_nazarite_position_actuator(
        spec=spec,
        joint_name=f"{leg}_{joint_kind}_joint",
        stiffness=float(actuator_cfg.stiffness) * kp_scale,
        damping=float(actuator_cfg.damping) * kd_scale,
        effort_limit=float(actuator_cfg.effort_limit),
        armature=float(actuator_cfg.armature),
      )
  ground = spec.worldbody.add_geom()
  ground.name = "smp_ground"
  ground.type = mujoco.mjtGeom.mjGEOM_PLANE
  ground.size = np.array((10.0, 10.0, 0.1))
  ground.pos = np.array((0.0, 0.0, 0.0))
  ground.friction = np.array((1.0, 0.005, 0.0001))
  ground.contype = 1
  ground.conaffinity = 1
  return spec.compile()


def _actuator_layout(model: mujoco.MjModel) -> np.ndarray:
  """Return qpos addresses in the actual injected-actuator order."""
  joint_ids = np.asarray(model.actuator_trnid[:, 0], dtype=np.intp)
  if joint_ids.size != 12:
    raise ValueError(f"expected 12 Go2 actuators, found {joint_ids.size}")
  joint_names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(joint_id))
    for joint_id in joint_ids
  ]
  if any(name is None for name in joint_names):
    raise ValueError("Go2 model has an unnamed actuator transmission joint")
  if set(joint_names) != {
    f"{leg}_{kind}_joint"
    for leg in ("FL", "FR", "RL", "RR")
    for kind in ("hip", "thigh", "calf")
  }:
    raise ValueError("Go2 actuator joints do not match the expected 12 leg joints")
  return np.asarray([model.jnt_qposadr[joint_id] for joint_id in joint_ids], dtype=np.intp)


def _actuator_action_parameters(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
  """Return default joint positions and GO2_ACTION_SCALE in actuator order."""
  defaults: list[float] = []
  scales: list[float] = []
  hip_scale = float(GO2_ACTION_SCALE[r".*_hip_joint"])
  calf_scale = float(GO2_ACTION_SCALE[r".*_calf_joint"])
  for actuator_id in range(model.nu):
    joint_id = int(model.actuator_trnid[actuator_id, 0])
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    if joint_name is None or joint_name not in NAZARITE_JOINT_POS:
      raise ValueError(f"unknown Go2 actuator joint: {joint_name!r}")
    defaults.append(float(NAZARITE_JOINT_POS[joint_name]))
    scales.append(calf_scale if joint_name.endswith("_calf_joint") else hip_scale)
  return np.asarray(defaults), np.asarray(scales)


def _actuator_delay_parameters(
  model: mujoco.MjModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Return per-actuator delay bounds and fused delay-group identifiers.

  mjlab fuses position actuators with the same delay configuration.  The
  current Go2 config therefore has one shared random lag for hip/thigh and
  another shared lag for calf at each physics step, rather than one lag per
  motor.
  """
  min_lags: list[int] = []
  max_lags: list[int] = []
  group_ids: list[int] = []
  for actuator_id in range(model.nu):
    joint_id = int(model.actuator_trnid[actuator_id, 0])
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    if joint_name is None:
      raise ValueError("Go2 actuator has no transmission joint name")
    cfg = GO2_CALF_ACTUATOR_CFG if joint_name.endswith("_calf_joint") else GO2_HIP_ACTUATOR_CFG
    min_lags.append(int(cfg.delay_min_lag))
    max_lags.append(int(cfg.delay_max_lag))
    group_ids.append(1 if joint_name.endswith("_calf_joint") else 0)
  return (
    np.asarray(min_lags, dtype=np.int64),
    np.asarray(max_lags, dtype=np.int64),
    np.asarray(group_ids, dtype=np.int64),
  )


def _reference_state(
  reference_qpos: np.ndarray,
  reference_fps: float,
  qpos_addresses: np.ndarray,
  time_s: float,
) -> tuple[np.ndarray, int]:
  """Linearly interpolate the native position-actuator joint targets."""
  phase = min(time_s * reference_fps, reference_qpos.shape[0] - 1)
  lower = int(np.floor(phase))
  upper = min(lower + 1, reference_qpos.shape[0] - 1)
  alpha = phase - lower
  lower_position = reference_qpos[lower, qpos_addresses]
  upper_position = reference_qpos[upper, qpos_addresses]
  position = (1.0 - alpha) * lower_position + alpha * upper_position
  return position, lower


def _foot_contacts(
  model: mujoco.MjModel, data: mujoco.MjData, ground_geom_id: int
) -> np.ndarray:
  """Return FL/FR/RL/RR contact booleans for the current physics step."""
  foot_geom_ids = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, leg)
    for leg in ("FL", "FR", "RL", "RR")
  ]
  contacts = np.zeros(4, dtype=bool)
  for contact_index in range(data.ncon):
    contact = data.contact[contact_index]
    for foot_index, foot_geom_id in enumerate(foot_geom_ids):
      if {contact.geom1, contact.geom2} == {ground_geom_id, foot_geom_id}:
        contacts[foot_index] = True
  return contacts


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
  padded = np.concatenate(([False], mask, [False]))
  changes = np.flatnonzero(padded[1:] != padded[:-1])
  return [(int(start), int(end)) for start, end in changes.reshape(-1, 2)]


def track_reference(
  model: mujoco.MjModel,
  reference: dict[str, np.ndarray],
  min_base_height: float,
  min_up_dot: float,
  max_joint_error: float,
  control_decimation: int,
  delay_min_lags: np.ndarray | None = None,
  delay_max_lags: np.ndarray | None = None,
  delay_group_ids: np.ndarray | None = None,
  delay_seed: int = 0,
  initialize_reference_velocity: bool = False,
) -> dict[str, np.ndarray]:
  """Simulate the whole reference and return actual states and filter signals."""
  if reference["qpos"].shape[1] != model.nq:
    raise ValueError(
      f"reference nq {reference['qpos'].shape[1]} does not match model nq {model.nq}"
    )
  reference_fps = float(np.asarray(reference["fps"]).item())
  duration_s = (reference["qpos"].shape[0] - 1) / reference_fps
  qpos_addresses = _actuator_layout(model)
  default_joint_position, action_scale = _actuator_action_parameters(model)
  base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
  ground_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "smp_ground")
  foot_site_ids = np.asarray(
    [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, leg) for leg in ("FL", "FR", "RL", "RR")],
    dtype=np.intp,
  )
  if base_id < 0 or ground_geom_id < 0 or np.any(foot_site_ids < 0):
    raise ValueError("Go2 model is missing base, ground, or foot sites")

  data = mujoco.MjData(model)
  data.qpos[:] = reference["qpos"][0]
  data.qvel[:] = (
    _initial_reference_velocity(model, reference)
    if initialize_reference_velocity
    else 0.0
  )
  mujoco.mj_forward(model, data)
  time_step = model.opt.timestep
  step_count = int(np.ceil(duration_s / time_step)) + 1
  time_s = np.empty(step_count, dtype=np.float64)
  qpos = np.empty((step_count, model.nq), dtype=np.float64)
  qvel = np.empty((step_count, model.nv), dtype=np.float64)
  ctrl_position_target = np.empty((step_count, model.nu), dtype=np.float64)
  actuator_force = np.empty((step_count, model.nu), dtype=np.float64)
  desired_joint_position = np.empty((step_count, model.nu), dtype=np.float64)
  joint_error = np.empty((step_count, model.nu), dtype=np.float64)
  base_position = np.empty((step_count, 3), dtype=np.float64)
  base_up_dot = np.empty(step_count, dtype=np.float64)
  foot_position = np.empty((step_count, 4, 3), dtype=np.float64)
  foot_position_base = np.empty((step_count, 4, 3), dtype=np.float64)
  foot_target_error = np.empty((step_count, 4), dtype=np.float64)
  foot_contacts = np.empty((step_count, 4), dtype=bool)
  equivalent_action = np.empty((step_count, model.nu), dtype=np.float64)
  applied_position_target = np.empty((step_count, model.nu), dtype=np.float64)
  delay_lag = np.zeros((step_count, model.nu), dtype=np.int64)
  valid = np.empty(step_count, dtype=bool)
  source_frame = np.empty(step_count, dtype=np.int64)

  if delay_min_lags is None or delay_max_lags is None or delay_group_ids is None:
    delay_min_lags = np.zeros(model.nu, dtype=np.int64)
    delay_max_lags = np.zeros(model.nu, dtype=np.int64)
    delay_group_ids = np.arange(model.nu, dtype=np.int64)
  if not (
    delay_min_lags.shape == (model.nu,)
    and delay_max_lags.shape == (model.nu,)
    and delay_group_ids.shape == (model.nu,)
  ):
    raise ValueError("delay parameter arrays must have shape [model.nu]")
  if np.any(delay_min_lags < 0) or np.any(delay_max_lags < delay_min_lags):
    raise ValueError("invalid actuator delay bounds")
  delay_rng = np.random.default_rng(delay_seed)
  command_history: list[np.ndarray] = []

  desired_q, reference_index = _reference_state(
    reference["qpos"], reference_fps, qpos_addresses, 0.0
  )
  for step_index in range(step_count):
    current_time = min(step_index * time_step, duration_s)
    if step_index % control_decimation == 0:
      control_time = min(
        (step_index // control_decimation) * control_decimation * time_step,
        duration_s,
      )
      desired_q, reference_index = _reference_state(
        reference["qpos"], reference_fps, qpos_addresses, control_time
      )
    actual_q = data.qpos[qpos_addresses]
    command_history.append(desired_q.copy())
    applied_q = desired_q.copy()
    for group_id in np.unique(delay_group_ids):
      actuator_mask = delay_group_ids == group_id
      min_lag = int(delay_min_lags[actuator_mask][0])
      max_lag = int(delay_max_lags[actuator_mask][0])
      if not np.all(delay_min_lags[actuator_mask] == min_lag) or not np.all(
        delay_max_lags[actuator_mask] == max_lag
      ):
        raise ValueError("actuators in one delay group must share delay bounds")
      lag = int(delay_rng.integers(min_lag, max_lag + 1))
      history_index = max(0, len(command_history) - 1 - lag)
      applied_q[actuator_mask] = command_history[history_index][actuator_mask]
      delay_lag[step_index, actuator_mask] = lag
    data.ctrl[:] = applied_q
    mujoco.mj_forward(model, data)

    base_rotation = data.xmat[base_id].reshape(3, 3)
    base_up = float(base_rotation[2, 2])
    current_joint_error = desired_q - actual_q
    current_equivalent_action = (desired_q - default_joint_position) / action_scale
    current_foot_positions = data.site_xpos[foot_site_ids].copy()
    current_foot_base = (current_foot_positions - data.xpos[base_id]) @ base_rotation
    if "foot_target_base_m" in reference:
      target_foot = reference["foot_target_base_m"][reference_index]
      current_foot_target_error = np.linalg.norm(target_foot - current_foot_base, axis=1)
    else:
      target_foot = reference["foot_target_world_m"][reference_index]
      current_foot_target_error = np.linalg.norm(target_foot - current_foot_positions, axis=1)
    current_valid = bool(
      data.xpos[base_id, 2] >= min_base_height
      and base_up >= min_up_dot
      and np.max(np.abs(current_joint_error)) <= max_joint_error
    )

    time_s[step_index] = current_time
    qpos[step_index] = data.qpos
    qvel[step_index] = data.qvel
    ctrl_position_target[step_index] = data.ctrl
    actuator_force[step_index] = data.actuator_force
    desired_joint_position[step_index] = desired_q
    applied_position_target[step_index] = applied_q
    joint_error[step_index] = current_joint_error
    base_position[step_index] = data.xpos[base_id]
    base_up_dot[step_index] = base_up
    foot_position[step_index] = current_foot_positions
    foot_position_base[step_index] = current_foot_base
    foot_target_error[step_index] = current_foot_target_error
    foot_contacts[step_index] = _foot_contacts(model, data, ground_geom_id)
    equivalent_action[step_index] = current_equivalent_action
    valid[step_index] = current_valid
    source_frame[step_index] = reference_index
    if step_index + 1 < step_count:
      mujoco.mj_step(model, data)

  return {
    "time_s": time_s,
    "qpos": qpos,
    "qvel": qvel,
    "ctrl_position_target": ctrl_position_target,
    "actuator_force": actuator_force,
    "reference_joint_position": desired_joint_position,
    "applied_position_target": applied_position_target,
    "joint_error_rad": joint_error,
    "root_pos_mujoco": base_position,
    "base_up_dot": base_up_dot,
    "foot_position_world_m": foot_position,
    "foot_position_base_m": foot_position_base,
    "foot_target_error_m": foot_target_error,
    "foot_ground_contact": foot_contacts,
    "equivalent_action": equivalent_action,
    "actuator_delay_lag": delay_lag,
    "physics_valid": valid,
    "source_reference_frame": source_frame,
    "fps": np.asarray(1.0 / time_step, dtype=np.float64),
    # `replay_go2_reference.py` accepts this alias for an immediate GIF check.
    "ik_success": valid,
  }


def _plot_tracking(output_path: Path, rollout: dict[str, np.ndarray]) -> None:
  """Save the core stability and tracking diagnostics for a human audit."""
  try:
    os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
  except ImportError as exc:
    raise RuntimeError("--plot requires matplotlib in the Nazarite environment") from exc

  time_s = rollout["time_s"]
  figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
  axes[0].plot(time_s, rollout["root_pos_mujoco"][:, 2], label="base height")
  axes[0].plot(time_s, rollout["base_up_dot"], label="base up dot")
  axes[0].set_ylabel("height / alignment")
  axes[0].legend()
  axes[0].grid(alpha=0.3)
  axes[1].plot(time_s, np.max(np.abs(rollout["joint_error_rad"]), axis=1))
  axes[1].set_ylabel("max joint error (rad)")
  axes[1].grid(alpha=0.3)
  axes[2].step(time_s, rollout["physics_valid"].astype(int), where="post", label="valid")
  for foot_index, foot_name in enumerate(("FL", "FR", "RL", "RR")):
    axes[2].step(
      time_s,
      rollout["foot_ground_contact"][:, foot_index].astype(int),
      where="post",
      alpha=0.65,
      label=f"{foot_name} contact",
    )
  axes[2].set_xlabel("time (s)")
  axes[2].set_ylabel("boolean")
  axes[2].set_ylim(-0.1, 1.1)
  axes[2].legend(ncol=3, fontsize=8)
  axes[2].grid(alpha=0.3)
  figure.suptitle("Go2 PD physics tracking diagnostics")
  figure.tight_layout()
  figure.savefig(output_path, dpi=180)
  plt.close(figure)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", type=Path, required=True, help="Geometric Go2 reference NPZ")
  parser.add_argument("--output-dir", type=Path, required=True, help="Physics rollout output")
  parser.add_argument("--xml", type=Path, default=DEFAULT_GO2_XML, help="Go2 MuJoCo XML")
  parser.add_argument(
    "--kp-scale", type=float, default=1.0,
    help="Additional diagnostic multiplier on go2_cfg Kp; keep 1.0 for training parity",
  )
  parser.add_argument(
    "--kd-scale", type=float, default=1.0,
    help="Additional diagnostic multiplier on go2_cfg Kd; keep 1.0 for training parity",
  )
  parser.add_argument("--min-base-height", type=float, default=0.18)
  parser.add_argument("--min-up-dot", type=float, default=0.5)
  parser.add_argument("--max-joint-error-rad", type=float, default=0.45)
  parser.add_argument("--min-success-duration-s", type=float, default=0.5)
  parser.add_argument(
    "--control-decimation", type=int, default=10,
    help="Policy-to-simulation decimation; 10 matches SMP teacher env",
  )
  parser.add_argument(
    "--delay-seed", type=int, default=0,
    help="Seed for reproducing the stochastic go2_cfg actuator delay",
  )
  parser.add_argument(
    "--ignore-actuator-delay", action="store_true",
    help="Disable go2_cfg delay for an isolated no-delay comparison",
  )
  parser.add_argument(
    "--initialize-reference-velocity", action="store_true",
    help="Initialize qvel from the first two reference frames instead of zero",
  )
  parser.add_argument(
    "--max-action-saturation-fraction", type=float, default=1.0,
    help="Diagnostic threshold only; current smp RL config has clip_actions=None",
  )
  parser.add_argument("--plot", action="store_true", help="Save tracking diagnostics PNG")
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
  args = _parse_args(argv)
  if (
    args.kp_scale <= 0.0
    or args.kd_scale <= 0.0
    or args.min_base_height <= 0.0
    or not -1.0 <= args.min_up_dot <= 1.0
    or args.max_joint_error_rad <= 0.0
    or args.min_success_duration_s <= 0.0
    or args.control_decimation <= 0
    or not 0.0 <= args.max_action_saturation_fraction <= 1.0
  ):
    raise ValueError("invalid PD gain scale or validity threshold")
  input_path: Path = args.input.expanduser().resolve()
  xml_path: Path = args.xml.expanduser().resolve()
  if not input_path.is_file():
    raise FileNotFoundError(input_path)
  if not xml_path.is_file():
    raise FileNotFoundError(xml_path)
  output_dir: Path = args.output_dir.expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

  reference = _load_reference(input_path)
  model = _model_with_ground(xml_path, args.kp_scale, args.kd_scale)
  delay_min_lags, delay_max_lags, delay_group_ids = _actuator_delay_parameters(model)
  if args.ignore_actuator_delay:
    delay_min_lags = np.zeros_like(delay_min_lags)
    delay_max_lags = np.zeros_like(delay_max_lags)
  rollout = track_reference(
    model=model,
    reference=reference,
    min_base_height=args.min_base_height,
    min_up_dot=args.min_up_dot,
    max_joint_error=args.max_joint_error_rad,
    control_decimation=args.control_decimation,
    delay_min_lags=delay_min_lags,
    delay_max_lags=delay_max_lags,
    delay_group_ids=delay_group_ids,
    delay_seed=args.delay_seed,
    initialize_reference_velocity=args.initialize_reference_velocity,
  )
  physics_fps = float(np.asarray(rollout["fps"]).item())
  min_run_frames = int(np.ceil(args.min_success_duration_s * physics_fps))
  accepted_runs = [
    {"start_index": start, "end_index_exclusive": end, "duration_s": (end - start) / physics_fps}
    for start, end in _contiguous_runs(rollout["physics_valid"])
    if end - start >= min_run_frames
  ]
  summary: dict[str, Any] = {
    "source": str(input_path),
    "go2_xml": str(xml_path),
    "simulation_fps": physics_fps,
    "simulation_frames": int(rollout["qpos"].shape[0]),
    "control_decimation": args.control_decimation,
    "policy_frequency_hz": float(1.0 / (model.opt.timestep * args.control_decimation)),
    "actuator_mode": "Nazarite BuiltinPositionActuatorCfg equivalent",
    "controller_source": "nazarite.config.robot_config.go2_cfg",
    "controller_parameters": {
      "hip_thigh": {
        "nominal_stiffness": STIFFNESS_HIP,
        "nominal_damping": DAMPING_HIP,
        "stiffness": float(GO2_HIP_ACTUATOR_CFG.stiffness) * args.kp_scale,
        "damping": float(GO2_HIP_ACTUATOR_CFG.damping) * args.kd_scale,
        "effort_limit": float(GO2_HIP_ACTUATOR_CFG.effort_limit),
        "armature": ARMATURE_HIP,
      },
      "calf": {
        "nominal_stiffness": STIFFNESS_CALF,
        "nominal_damping": DAMPING_CALF,
        "stiffness": float(GO2_CALF_ACTUATOR_CFG.stiffness) * args.kp_scale,
        "damping": float(GO2_CALF_ACTUATOR_CFG.damping) * args.kd_scale,
        "effort_limit": float(GO2_CALF_ACTUATOR_CFG.effort_limit),
        "armature": ARMATURE_CALF,
      },
      "action_scale": {str(key): float(value) for key, value in GO2_ACTION_SCALE.items()},
      "delay": {
        "enabled": not args.ignore_actuator_delay,
        "seed": args.delay_seed,
        "hip_thigh_min_lag_physics_steps": int(delay_min_lags[0]),
        "hip_thigh_max_lag_physics_steps": int(delay_max_lags[0]),
        "calf_min_lag_physics_steps": int(delay_min_lags[np.flatnonzero(delay_group_ids == 1)[0]]),
        "calf_max_lag_physics_steps": int(delay_max_lags[np.flatnonzero(delay_group_ids == 1)[0]]),
      },
      "go2_cfg_kp_scale": GO2_ACTUATOR_KP_SCALE,
      "go2_cfg_kd_scale": GO2_ACTUATOR_KD_SCALE,
      "additional_kp_scale": args.kp_scale,
      "additional_kd_scale": args.kd_scale,
    },
    "action_feasibility": {
      "max_action_saturation_fraction": args.max_action_saturation_fraction,
      "equivalent_action_max_abs": float(np.max(np.abs(rollout["equivalent_action"]))),
      "equivalent_action_saturation_fraction": float(
        np.mean(np.abs(rollout["equivalent_action"]) > 1.0)
      ),
    },
    "foot_target_error_frame": (
      "base" if "foot_target_base_m" in reference else "world"
    ),
    "kp_scale": args.kp_scale,
    "kd_scale": args.kd_scale,
    "delay_enabled": not args.ignore_actuator_delay,
    "initial_velocity_mode": (
      "reference_finite_difference"
      if args.initialize_reference_velocity
      else "zero"
    ),
    "validity_thresholds": {
      "min_base_height_m": args.min_base_height,
      "min_up_dot": args.min_up_dot,
      "max_joint_error_rad": args.max_joint_error_rad,
      "min_success_duration_s": args.min_success_duration_s,
    },
    "physics_valid_fraction": float(np.mean(rollout["physics_valid"])),
    "minimum_base_height_m": float(np.min(rollout["root_pos_mujoco"][:, 2])),
    "minimum_base_up_dot": float(np.min(rollout["base_up_dot"])),
    "maximum_joint_error_rad": float(np.max(np.abs(rollout["joint_error_rad"]))),
    "mean_foot_target_error_m": float(np.mean(rollout["foot_target_error_m"])),
    "p95_foot_target_error_m": float(np.percentile(rollout["foot_target_error_m"], 95.0)),
    "accepted_runs": accepted_runs,
  }
  np.savez_compressed(output_dir / "go2_physics_rollout.npz", **rollout)
  with (output_dir / "physics_tracking_summary.json").open("w", encoding="utf-8") as stream:
    json.dump(summary, stream, indent=2)
    stream.write("\n")
  if args.plot:
    _plot_tracking(output_dir / "physics_tracking_diagnostics.png", rollout)

  print(json.dumps(summary, indent=2))
  print(f"\nWrote: {output_dir / 'go2_physics_rollout.npz'}")
  print(f"Wrote: {output_dir / 'physics_tracking_summary.json'}")
  if args.plot:
    print(f"Wrote: {output_dir / 'physics_tracking_diagnostics.png'}")


if __name__ == "__main__":
  main()
