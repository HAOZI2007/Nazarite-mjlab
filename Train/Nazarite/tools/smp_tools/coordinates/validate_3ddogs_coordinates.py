"""Validate and export one 3DDogs clip in the candidate MuJoCo coordinate frame.

The 3DDogs optical recordings are y-up.  This tool makes that conversion
explicit and exports only a fully valid selected clip; it never changes the
licensed source recording.  The default axis map was chosen from the longest
fully valid scan result (``d29_t1_a``): it maps that dog's travel direction to
MuJoCo +x, its anatomical left to MuJoCo +y, and height to MuJoCo +z.

Example:

    cd Train/Nazarite
    uv run python tools/smp_tools/coordinates/validate_3ddogs_coordinates.py \\
      --input /home/haozi/3DDogs/3DDogs2024_full/Data/Optical/ \\
Sync_Align_v2023_11_16b/optical_sync_align_d29_t1_a.txt \\
      --start-index 0 --end-index 132 \\
      --output-dir output/3ddogs_coordinate_validation/d29_t1_a --plot

Inspect the resulting ``coordinate_diagnostics.png`` before treating the
exported ``retarget_inputs_mujoco.npz`` as a canonical retargeting input.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

# Each row gives one MuJoCo coordinate in the raw 3DDogs (x, y, z) basis. All
# candidates map raw y-up to MuJoCo z-up and have determinant +1.
AXIS_MAPS: dict[str, np.ndarray] = {
  "neg_x_z_y": np.array(((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0))),
  "x_neg_z_y": np.array(((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0))),
  "z_x_y": np.array(((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
  "neg_z_neg_x_y": np.array(
    ((0.0, 0.0, -1.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
  ),
}

if TYPE_CHECKING:
  from tools.smp_tools.data.inspect_3ddogs import (
    PAW_ORDER,
    extract_retarget_inputs,
    load_optical_trial,
  )
else:
  # Keep direct-script use independent from the simulator training package.
  sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
  from tools.smp_tools.data.inspect_3ddogs import (
    PAW_ORDER,
    extract_retarget_inputs,
    load_optical_trial,
  )


def _transform(values: np.ndarray, rotation: np.ndarray) -> np.ndarray:
  """Apply a raw-to-MuJoCo rotation to values ending in an xyz dimension."""
  if values.shape[-1] != 3:
    raise ValueError(f"expected final xyz dimension, got {values.shape}")
  return values @ rotation.T


def _mean_unit_vector(vectors: np.ndarray) -> np.ndarray:
  """Return a robust average direction, ignoring zero-length frame vectors."""
  lengths = np.linalg.vector_norm(vectors, axis=-1)
  nonzero = lengths > 1e-9
  if not np.any(nonzero):
    raise ValueError("cannot measure direction: every vector has zero length")
  unit_vectors = vectors[nonzero] / lengths[nonzero, None]
  direction = unit_vectors.mean(axis=0)
  direction_norm = np.linalg.vector_norm(direction)
  if direction_norm <= 1e-9:
    raise ValueError("cannot measure direction: average direction is degenerate")
  return direction / direction_norm


def _diagnostics(
  root_pos: np.ndarray,
  paws: np.ndarray,
  forward: np.ndarray,
  left: np.ndarray,
  up: np.ndarray,
  rotation: np.ndarray,
) -> dict[str, object]:
  """Calculate checks that make a candidate axis map easy to audit."""
  root_travel = root_pos[-1] - root_pos[0]
  return {
    "rotation_matrix_raw_to_mujoco": rotation.tolist(),
    "rotation_determinant": float(np.linalg.det(rotation)),
    "rotation_is_orthonormal": bool(
      np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    ),
    "root_height_above_mean_paws_m": float(
      np.median(root_pos[:, 2] - paws[:, :, 2].mean(axis=1))
    ),
    "root_travel_m": root_travel.tolist(),
    "root_travel_length_m": float(np.linalg.vector_norm(root_travel)),
    "mean_anatomical_forward_unit": _mean_unit_vector(forward).tolist(),
    "mean_anatomical_left_unit": _mean_unit_vector(left).tolist(),
    "mean_anatomical_up_unit": _mean_unit_vector(up).tolist(),
    "interpretation": {
      "mujoco_axes": {"x": "forward", "y": "left", "z": "up"},
      "expected_for_this_clip": (
        "root height above paws should be positive; anatomical forward should "
        "mostly point +x; anatomical left should mostly point +y; anatomical "
        "up should mostly point +z"
      ),
    },
  }


def _plot(
  output_path: Path,
  raw_root: np.ndarray,
  raw_paws: np.ndarray,
  mapped_root: np.ndarray,
  mapped_paws: np.ndarray,
) -> None:
  """Render raw and candidate-frame trajectories for manual validation."""
  try:
    mpl_config_dir = output_path.parent / ".matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
  except ImportError as exc:
    raise RuntimeError("--plot requires matplotlib in the Nazarite environment") from exc

  figure = plt.figure(figsize=(14, 6))
  panels = (
    (figure.add_subplot(1, 2, 1, projection="3d"), raw_root, raw_paws, "raw 3DDogs"),
    (
      figure.add_subplot(1, 2, 2, projection="3d"),
      mapped_root,
      mapped_paws,
      "candidate MuJoCo frame",
    ),
  )
  for axes, root, paws, title in panels:
    axes.plot(root[:, 0], root[:, 1], root[:, 2], label="trunk centre", linewidth=2)
    for paw_index, paw_name in enumerate(PAW_ORDER):
      paw = paws[:, paw_index]
      axes.plot(paw[:, 0], paw[:, 1], paw[:, 2], label=paw_name, alpha=0.8)
    axes.scatter(*root[0], color="black", marker="o", label="start")
    axes.scatter(*root[-1], color="black", marker="x", label="end")
    axes.set_title(title)
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_zlabel("z")
    axes.legend(loc="best")
    axes.set_box_aspect((1.0, 1.0, 0.5))
  figure.suptitle("3DDogs coordinate-map validation: root + four paws")
  figure.tight_layout()
  figure.savefig(output_path, dpi=180)
  plt.close(figure)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", type=Path, required=True, help="Source optical .txt file")
  parser.add_argument("--start-index", type=int, required=True, help="Inclusive clip index")
  parser.add_argument(
    "--end-index", type=int, required=True, help="Exclusive clip index from the scan manifest"
  )
  parser.add_argument("--output-dir", type=Path, required=True, help="Derived output directory")
  parser.add_argument(
    "--axis-map",
    choices=tuple(AXIS_MAPS),
    default="neg_x_z_y",
    help="Candidate right-handed map; default matches d29_t1_a anatomy and travel",
  )
  parser.add_argument("--plot", action="store_true", help="Save raw/candidate trajectory plot")
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
  args = _parse_args(argv)
  if args.start_index < 0 or args.end_index <= args.start_index:
    raise ValueError("expected 0 <= --start-index < --end-index")

  input_path: Path = args.input.expanduser().resolve()
  if not input_path.is_file():
    raise FileNotFoundError(input_path)
  trial = load_optical_trial(input_path)
  if args.end_index > trial.frame_numbers.size:
    raise IndexError(
      f"--end-index {args.end_index} exceeds trial length {trial.frame_numbers.size}"
    )
  extracted = extract_retarget_inputs(trial)
  clip_slice = slice(args.start_index, args.end_index)
  if not np.all(extracted["valid_mask"][clip_slice]):
    raise ValueError("selected clip contains invalid frames; use a run from selected_clips.csv")

  rotation = AXIS_MAPS[args.axis_map]
  raw_root = extracted["root_pos_raw"][clip_slice]
  raw_paws = extracted["paw_pos_raw"][clip_slice]
  root_pos = _transform(raw_root, rotation)
  paw_pos = _transform(raw_paws, rotation)
  root_forward = _transform(extracted["root_forward_raw"][clip_slice], rotation)
  root_left = _transform(extracted["root_left_raw"][clip_slice], rotation)
  root_up = _transform(extracted["root_up_raw"][clip_slice], rotation)

  report: dict[str, object] = {
    "source": str(input_path),
    "fps": trial.fps,
    "axis_map": args.axis_map,
    "start_index": args.start_index,
    "end_index_exclusive": args.end_index,
    "start_frame_num": int(trial.frame_numbers[args.start_index]),
    "end_frame_num": int(trial.frame_numbers[args.end_index - 1]),
    "length_frames": args.end_index - args.start_index,
    "duration_s": (args.end_index - args.start_index) / trial.fps,
    "paw_order": list(PAW_ORDER),
    "diagnostics": _diagnostics(
      root_pos, paw_pos, root_forward, root_left, root_up, rotation
    ),
    "manual_acceptance_required": True,
  }

  output_dir: Path = args.output_dir.expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    output_dir / "retarget_inputs_mujoco.npz",
    allow_pickle=False,
    frame_numbers=extracted["frame_numbers"][clip_slice],
    root_pos_mujoco=root_pos,
    root_forward_mujoco=root_forward,
    root_left_mujoco=root_left,
    root_up_mujoco=root_up,
    paw_pos_mujoco=paw_pos,
    fps=np.asarray(trial.fps, dtype=np.float64),
    paw_order=np.asarray(PAW_ORDER),
    axis_map=np.asarray(args.axis_map),
  )
  with (output_dir / "coordinate_report.json").open("w", encoding="utf-8") as stream:
    json.dump(report, stream, indent=2)
    stream.write("\n")
  if args.plot:
    _plot(
      output_dir / "coordinate_diagnostics.png",
      raw_root,
      raw_paws,
      root_pos,
      paw_pos,
    )

  print(json.dumps(report, indent=2))
  print(f"\nWrote: {output_dir / 'retarget_inputs_mujoco.npz'}")
  print(f"Wrote: {output_dir / 'coordinate_report.json'}")
  if args.plot:
    print(f"Wrote: {output_dir / 'coordinate_diagnostics.png'}")


if __name__ == "__main__":
  main()
