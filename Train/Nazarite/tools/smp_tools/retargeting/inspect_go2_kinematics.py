"""Audit the exact Go2 MuJoCo kinematics used by Nazarite before SMP IK.

The report records the joint address/order/limits, actuator order, hip and
foot positions, and the default Nazarite training stance.  The SMP retargeter
must use the explicit ``FL, FR, RL, RR`` leg order written by this tool rather
than assuming MuJoCo actuator order.

Example:

    cd Train/Nazarite
    uv run python tools/smp_tools/retargeting/inspect_go2_kinematics.py \\
      --output-dir output/go2_kinematics --plot
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import mujoco
import numpy as np

# File location: <Nazarite>/tools/smp_tools/retargeting/inspect_go2_kinematics.py
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GO2_XML = PROJECT_ROOT / "MJCF-Manager" / "Robots" / "GO2" / "xmls" / "go2.xml"

# This order matches PAW_ORDER in the 3DDogs readers and remains the sole SMP
# convention, independently of the model's actuator declaration order.
SMP_LEG_ORDER: tuple[str, ...] = ("FL", "FR", "RL", "RR")
JOINT_KINDS: tuple[str, ...] = ("hip", "thigh", "calf")

# These values mirror GO2_INIT_STATE in nazarite/config/robot_config/go2_cfg.py.
# Keeping them locally avoids importing the top-level training package (which
# registers every task) for a lightweight offline diagnostic.
NAZARITE_BASE_POS = np.array((0.0, 0.0, 0.32), dtype=np.float64)
NAZARITE_BASE_QUAT = np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
NAZARITE_JOINT_POS: dict[str, float] = {
  "FL_hip_joint": 0.0,
  "FL_thigh_joint": 0.8,
  "FL_calf_joint": -1.5,
  "FR_hip_joint": 0.0,
  "FR_thigh_joint": 0.8,
  "FR_calf_joint": -1.5,
  "RL_hip_joint": 0.0,
  "RL_thigh_joint": 1.0,
  "RL_calf_joint": -1.5,
  "RR_hip_joint": 0.0,
  "RR_thigh_joint": 1.0,
  "RR_calf_joint": -1.5,
}


def _name(model: mujoco.MjModel, object_type: mujoco.mjtObj, object_id: int) -> str:
  value = mujoco.mj_id2name(model, object_type, object_id)
  if value is None:
    raise ValueError(f"unnamed MuJoCo object: type={object_type}, id={object_id}")
  return value


def _object_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
  object_id = mujoco.mj_name2id(model, object_type, name)
  if object_id < 0:
    raise ValueError(f"MuJoCo model has no object {name!r} of type {object_type}")
  return object_id


def _joint_names() -> tuple[str, ...]:
  return tuple(f"{leg}_{kind}_joint" for leg in SMP_LEG_ORDER for kind in JOINT_KINDS)


def _set_nazarite_default_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
  """Apply Nazarite's training spawn pose before forward kinematics."""
  data.qpos[:] = model.qpos0
  free_joint_ids = [
    joint_id
    for joint_id in range(model.njnt)
    if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
  ]
  if len(free_joint_ids) != 1:
    raise ValueError(f"expected one Go2 free joint, found {len(free_joint_ids)}")
  free_qpos_adr = int(model.jnt_qposadr[free_joint_ids[0]])
  data.qpos[free_qpos_adr : free_qpos_adr + 3] = NAZARITE_BASE_POS
  data.qpos[free_qpos_adr + 3 : free_qpos_adr + 7] = NAZARITE_BASE_QUAT
  for joint_name, position in NAZARITE_JOINT_POS.items():
    joint_id = _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    data.qpos[int(model.jnt_qposadr[joint_id])] = position
  mujoco.mj_forward(model, data)


def _joint_report(model: mujoco.MjModel, joint_name: str) -> dict[str, object]:
  joint_id = _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
  return {
    "name": joint_name,
    "joint_id": joint_id,
    "qpos_address": int(model.jnt_qposadr[joint_id]),
    "dof_address": int(model.jnt_dofadr[joint_id]),
    "range_rad": model.jnt_range[joint_id].tolist(),
    "limited": bool(model.jnt_limited[joint_id]),
  }


def _actuator_report(model: mujoco.MjModel) -> list[dict[str, object]]:
  report: list[dict[str, object]] = []
  for actuator_id in range(model.nu):
    transmission_joint_id = int(model.actuator_trnid[actuator_id, 0])
    report.append(
      {
        "actuator_id": actuator_id,
        "name": _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id),
        "joint": _name(model, mujoco.mjtObj.mjOBJ_JOINT, transmission_joint_id),
        "control_range": model.actuator_ctrlrange[actuator_id].tolist(),
      }
    )
  return report


def build_report(model: mujoco.MjModel, data: mujoco.MjData, xml_path: Path) -> dict[str, object]:
  """Extract all identifiers and pose quantities needed by the future IK tool."""
  base_id = _object_id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
  base_position = data.xpos[base_id].copy()
  legs: dict[str, dict[str, object]] = {}
  for leg in SMP_LEG_ORDER:
    hip_body_name = f"{leg}_hip"
    foot_site_name = leg
    hip_body_id = _object_id(model, mujoco.mjtObj.mjOBJ_BODY, hip_body_name)
    foot_site_id = _object_id(model, mujoco.mjtObj.mjOBJ_SITE, foot_site_name)
    hip_position = data.xpos[hip_body_id].copy()
    foot_position = data.site_xpos[foot_site_id].copy()
    legs[leg] = {
      "joint_names": [f"{leg}_{kind}_joint" for kind in JOINT_KINDS],
      "hip_body": hip_body_name,
      "hip_position_base_frame_m": (hip_position - base_position).tolist(),
      "foot_site": foot_site_name,
      "foot_position_base_frame_m": (foot_position - base_position).tolist(),
      "foot_position_world_m": foot_position.tolist(),
    }

  return {
    "xml_path": str(xml_path),
    "model_name": _name(model, mujoco.mjtObj.mjOBJ_BODY, base_id),
    "dimensions": {"nq": model.nq, "nv": model.nv, "njnt": model.njnt, "nu": model.nu},
    "coordinate_convention": {"x": "forward", "y": "left", "z": "up"},
    "smp_leg_order": list(SMP_LEG_ORDER),
    "smp_joint_order": list(_joint_names()),
    "nazarite_default_base_position_m": NAZARITE_BASE_POS.tolist(),
    "nazarite_default_base_quaternion_wxyz": NAZARITE_BASE_QUAT.tolist(),
    "nazarite_default_joint_positions_rad": NAZARITE_JOINT_POS,
    "joint_report": [_joint_report(model, name) for name in _joint_names()],
    "actuator_report_xml_order": _actuator_report(model),
    "legs": legs,
  }


def _plot_default_pose(output_path: Path, report: dict[str, object]) -> None:
  """Render a compact top/side diagnostic of hip and foot positions."""
  os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".matplotlib"))
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  legs = cast(dict[str, dict[str, Any]], report["legs"])
  figure, (top_axes, side_axes) = plt.subplots(1, 2, figsize=(10, 4.5))
  colors = {"FL": "tab:blue", "FR": "tab:orange", "RL": "tab:green", "RR": "tab:red"}
  for leg in SMP_LEG_ORDER:
    hip = np.asarray(legs[leg]["hip_position_base_frame_m"], dtype=np.float64)
    foot = np.asarray(legs[leg]["foot_position_base_frame_m"], dtype=np.float64)
    color = colors[leg]
    top_axes.plot((hip[0], foot[0]), (hip[1], foot[1]), color=color, linewidth=2)
    top_axes.scatter(hip[0], hip[1], color=color, marker="o")
    top_axes.scatter(foot[0], foot[1], color=color, marker="x")
    top_axes.annotate(leg, (foot[0], foot[1]), xytext=(4, 4), textcoords="offset points")
    side_axes.plot((hip[0], foot[0]), (hip[2], foot[2]), color=color, linewidth=2)
    side_axes.scatter(hip[0], hip[2], color=color, marker="o")
    side_axes.scatter(foot[0], foot[2], color=color, marker="x")
  for axes, y_label, title in (
    (top_axes, "y / left (m)", "Top view: hip-to-foot"),
    (side_axes, "z / up (m)", "Side view: hip-to-foot"),
  ):
    axes.axhline(0.0, color="0.7", linewidth=1)
    axes.axvline(0.0, color="0.7", linewidth=1)
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel("x / forward (m)")
    axes.set_ylabel(y_label)
    axes.set_title(title)
    axes.grid(alpha=0.3)
  figure.suptitle("Nazarite Go2 default stance: base-frame kinematics")
  figure.tight_layout()
  figure.savefig(output_path, dpi=180)
  plt.close(figure)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--xml", type=Path, default=DEFAULT_GO2_XML, help="Go2 MuJoCo XML")
  parser.add_argument("--output-dir", type=Path, required=True, help="Derived report directory")
  parser.add_argument("--plot", action="store_true", help="Save a default-pose diagnostic PNG")
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
  args = _parse_args(argv)
  xml_path: Path = args.xml.expanduser().resolve()
  if not xml_path.is_file():
    raise FileNotFoundError(xml_path)
  output_dir: Path = args.output_dir.expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

  model = mujoco.MjModel.from_xml_path(str(xml_path))
  data = mujoco.MjData(model)
  _set_nazarite_default_pose(model, data)
  report = build_report(model, data, xml_path)
  with (output_dir / "go2_kinematics_report.json").open("w", encoding="utf-8") as stream:
    json.dump(report, stream, indent=2)
    stream.write("\n")
  if args.plot:
    _plot_default_pose(output_dir / "go2_default_stance.png", report)

  print(json.dumps(report, indent=2))
  print(f"\nWrote: {output_dir / 'go2_kinematics_report.json'}")
  if args.plot:
    print(f"Wrote: {output_dir / 'go2_default_stance.png'}")


if __name__ == "__main__":
  main()
