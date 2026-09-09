"""Inspect a 3DDogs-Lab optical MoCap trial without changing the source data.

The 3DDogs optical files contain global 3D marker positions in the form::

    frame_num marker_0_x marker_0_y marker_0_z ...

This tool extracts the markers needed for task-space retargeting to Go2 and
writes a compact intermediate NPZ.  It deliberately preserves the source
coordinate frame: deciding the exact 3DDogs -> MuJoCo axis transform requires
visual validation of forward, left/right, and up directions first.

Example:

    cd Train/Nazarite
    uv run python tools/smp_tools/data/inspect_3ddogs.py \\
      --input /home/haozi/3DDogs/3DDogs2024_full/Data/Optical/ \\
Sync_Align_v2023_11_16b/optical_sync_align_d19_t1_a.txt \\
      --output-dir output/3ddogs_inspect/d19_t1_a --plot
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np

# The order is also the target order used by Go2 retargeting.  Keep it fixed in
# every later data format, retargeter, and feature encoder.
PAW_MARKERS: tuple[str, ...] = (
  "l_meta_carp",
  "r_meta_carp",
  "l_meta_tars",
  "r_meta_tars",
)
PAW_ORDER: tuple[str, ...] = ("FL", "FR", "RL", "RR")

# The body reference is intentionally simple and fully observable in 3DDogs.
# A later retargeter can use additional markers when constructing the Go2 root
# orientation, but these are the minimum required to audit the trial.
ROOT_MARKERS: tuple[str, ...] = (
  "withers",
  "sacrum",
  "l_shdr",
  "r_shdr",
  "l_iliac",
  "r_iliac",
)
REQUIRED_MARKERS = ROOT_MARKERS + PAW_MARKERS


@dataclass(frozen=True)
class MocapTrial:
  """Raw, source-coordinate data from one optical MoCap file."""

  fps: float
  frame_numbers: np.ndarray
  marker_names: tuple[str, ...]
  marker_positions: np.ndarray


class RetargetArchive(TypedDict):
  """Named arrays stored in the intermediate NPZ archive.

  A ``TypedDict`` is used here instead of unpacking the generic dictionary
  returned by ``extract_retarget_inputs`` directly.  NumPy reserves the
  ``allow_pickle`` keyword in ``savez_compressed``; without the explicit keys,
  Pylance conservatively assumes any unpacked array could be that boolean.
  """

  frame_numbers: np.ndarray
  marker_positions_raw: np.ndarray
  root_pos_raw: np.ndarray
  root_forward_raw: np.ndarray
  root_left_raw: np.ndarray
  root_up_raw: np.ndarray
  paw_pos_raw: np.ndarray
  valid_mask: np.ndarray
  fps: np.ndarray
  marker_names: np.ndarray
  paw_order: np.ndarray


def _parse_header(input_path: Path) -> tuple[float, tuple[str, ...]]:
  """Read the 3DDogs header and validate the marker-name line."""
  with input_path.open("r", encoding="utf-8") as stream:
    header = [stream.readline().strip() for _ in range(4)]

  if len(header) != 4 or not header[0].startswith("fps "):
    raise ValueError(f"{input_path} is not a supported 3DDogs optical file")
  try:
    fps = float(header[0].split(maxsplit=1)[1])
  except (IndexError, ValueError) as exc:
    raise ValueError(f"could not parse fps from {header[0]!r}") from exc

  fields = tuple(header[3].split())
  if not fields or fields[0] != "frame_num":
    raise ValueError(f"expected 'frame_num' at header line 4, got {header[3]!r}")
  marker_names = fields[1:]
  if len(set(marker_names)) != len(marker_names):
    raise ValueError("marker names must be unique")
  return fps, marker_names


def load_optical_trial(input_path: Path) -> MocapTrial:
  """Load one global `optical_sync_align_*.txt` file.

  `numpy.loadtxt` parses the literal `NaN` values emitted by the optical system
  as `np.nan`, which is exactly what the downstream validity mask needs.
  """
  fps, marker_names = _parse_header(input_path)
  raw = np.loadtxt(input_path, dtype=np.float64, skiprows=4)
  if raw.ndim == 1:
    raw = raw[None, :]

  expected_columns = 1 + 3 * len(marker_names)
  if raw.shape[1] != expected_columns:
    raise ValueError(
      f"{input_path.name}: got {raw.shape[1]} columns, expected {expected_columns} "
      f"for {len(marker_names)} markers"
    )

  return MocapTrial(
    fps=fps,
    frame_numbers=raw[:, 0].astype(np.int64),
    marker_names=marker_names,
    marker_positions=raw[:, 1:].reshape(-1, len(marker_names), 3),
  )


def _marker_indexes(marker_names: tuple[str, ...]) -> dict[str, int]:
  """Return required marker indexes or fail with a useful missing-name error."""
  index_by_name = {name: index for index, name in enumerate(marker_names)}
  missing = sorted(set(REQUIRED_MARKERS) - set(index_by_name))
  if missing:
    raise ValueError(f"trial lacks required 3DDogs markers: {', '.join(missing)}")
  return index_by_name


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
  """Return half-open index ranges for all true runs in a 1D boolean mask."""
  if mask.ndim != 1:
    raise ValueError(f"expected 1D validity mask, got {mask.shape}")
  padded = np.concatenate(([False], mask, [False]))
  changes = np.flatnonzero(padded[1:] != padded[:-1])
  return [(int(start), int(end)) for start, end in changes.reshape(-1, 2)]


def extract_retarget_inputs(trial: MocapTrial) -> dict[str, np.ndarray]:
  """Extract task-space data required by the future Go2 retargeter.

  Output coordinates are still in the original 3DDogs global frame.  The
  `valid_mask` requires every root and paw marker to be finite.  Invalid rows
  remain in the arrays so source frame numbers stay aligned and inspectable.
  """
  indexes = _marker_indexes(trial.marker_names)
  marker_pos = trial.marker_positions

  withers = marker_pos[:, indexes["withers"]]
  sacrum = marker_pos[:, indexes["sacrum"]]
  l_shoulder = marker_pos[:, indexes["l_shdr"]]
  r_shoulder = marker_pos[:, indexes["r_shdr"]]
  l_hip = marker_pos[:, indexes["l_iliac"]]
  r_hip = marker_pos[:, indexes["r_iliac"]]
  paws = np.stack([marker_pos[:, indexes[name]] for name in PAW_MARKERS], axis=1)

  required_indexes = [indexes[name] for name in REQUIRED_MARKERS]
  valid_mask = np.isfinite(marker_pos[:, required_indexes]).all(axis=(1, 2))

  # `root_pos` is the centre of the trunk.  The three direction vectors are
  # saved instead of a quaternion because the raw dataset is y-up while MuJoCo
  # is z-up; quaternion conversion belongs after the axis map is verified.
  root_pos = 0.5 * (withers + sacrum)
  forward = withers - sacrum
  left = 0.5 * ((l_shoulder - r_shoulder) + (l_hip - r_hip))
  up = np.cross(forward, left)

  return {
    "frame_numbers": trial.frame_numbers,
    "marker_positions_raw": marker_pos,
    "root_pos_raw": root_pos,
    "root_forward_raw": forward,
    "root_left_raw": left,
    "root_up_raw": up,
    "paw_pos_raw": paws,
    "valid_mask": valid_mask,
  }


def _summary(
  input_path: Path,
  trial: MocapTrial,
  extracted: dict[str, np.ndarray],
) -> dict[str, object]:
  valid_mask = extracted["valid_mask"]
  runs = _contiguous_runs(valid_mask)
  return {
    "source": str(input_path.resolve()),
    "fps": trial.fps,
    "num_frames": int(trial.frame_numbers.size),
    "marker_count": len(trial.marker_names),
    "marker_names": list(trial.marker_names),
    "paw_order": list(PAW_ORDER),
    "paw_markers": list(PAW_MARKERS),
    "root_markers": list(ROOT_MARKERS),
    "coordinate_frame": "3DDogs global optical frame; no axis remap applied",
    "valid_frames": int(valid_mask.sum()),
    "valid_fraction": float(valid_mask.mean()),
    "valid_runs": [
      {
        "start_index": start,
        "end_index_exclusive": end,
        "start_frame_num": int(trial.frame_numbers[start]),
        "end_frame_num": int(trial.frame_numbers[end - 1]),
        "length_frames": end - start,
        "duration_s": (end - start) / trial.fps,
      }
      for start, end in runs
    ],
  }


def _plot_trial(
  output_path: Path,
  extracted: dict[str, np.ndarray],
  max_frames: int,
) -> None:
  """Save a lightweight 3D trajectory plot; matplotlib is optional."""
  try:
    # The project may run in a sandboxed account whose default matplotlib
    # configuration directory is not writable. Keep the cache beside this
    # explicitly requested derived output instead of falling back to $HOME.
    mpl_config_dir = output_path.parent / ".matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
  except ImportError as exc:
    raise RuntimeError(
      "--plot requires matplotlib. Install it in the Nazarite environment first."
    ) from exc

  valid = extracted["valid_mask"]
  valid_indexes = np.flatnonzero(valid)
  if valid_indexes.size == 0:
    raise RuntimeError("cannot plot: the trial has no frames with all required markers")
  stride = max(1, int(np.ceil(valid_indexes.size / max_frames)))
  indexes = valid_indexes[::stride]

  root = extracted["root_pos_raw"][indexes]
  paws = extracted["paw_pos_raw"][indexes]
  figure = plt.figure(figsize=(9, 7))
  axes = figure.add_subplot(projection="3d")
  axes.plot(root[:, 0], root[:, 1], root[:, 2], label="trunk centre", linewidth=2)
  for paw_index, paw_name in enumerate(PAW_ORDER):
    trajectory = paws[:, paw_index]
    axes.plot(
      trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label=paw_name, alpha=0.8
    )
  axes.set_title("3DDogs raw global-frame trajectories (no MuJoCo axis mapping)")
  axes.set_xlabel("raw x")
  axes.set_ylabel("raw y")
  axes.set_zlabel("raw z")
  axes.legend(loc="best")
  figure.tight_layout()
  figure.savefig(output_path, dpi=180)
  plt.close(figure)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--input", type=Path, required=True, help="Path to optical_sync_align_*.txt"
  )
  parser.add_argument(
    "--output-dir", type=Path, required=True, help="Directory for derived NPZ/report"
  )
  parser.add_argument(
    "--plot", action="store_true", help="Also save raw-frame 3D trajectory PNG"
  )
  parser.add_argument(
    "--max-plot-frames", type=int, default=300, help="Maximum plotted valid frames"
  )
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  if args.max_plot_frames <= 0:
    raise ValueError("--max-plot-frames must be positive")
  input_path: Path = args.input.expanduser().resolve()
  if not input_path.is_file():
    raise FileNotFoundError(input_path)

  trial = load_optical_trial(input_path)
  extracted = extract_retarget_inputs(trial)
  summary = _summary(input_path, trial, extracted)

  output_dir: Path = args.output_dir.expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  archive: RetargetArchive = {
    "frame_numbers": extracted["frame_numbers"],
    "marker_positions_raw": extracted["marker_positions_raw"],
    "root_pos_raw": extracted["root_pos_raw"],
    "root_forward_raw": extracted["root_forward_raw"],
    "root_left_raw": extracted["root_left_raw"],
    "root_up_raw": extracted["root_up_raw"],
    "paw_pos_raw": extracted["paw_pos_raw"],
    "valid_mask": extracted["valid_mask"],
    "fps": np.asarray(trial.fps, dtype=np.float64),
    "marker_names": np.asarray(trial.marker_names),
    "paw_order": np.asarray(PAW_ORDER),
  }
  np.savez_compressed(
    output_dir / "retarget_inputs_raw.npz",
    allow_pickle=False,
    **archive,
  )
  with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
    json.dump(summary, stream, indent=2)
    stream.write("\n")
  if args.plot:
    _plot_trial(output_dir / "raw_trajectories.png", extracted, args.max_plot_frames)

  print(json.dumps(summary, indent=2))
  print(f"\nWrote: {output_dir / 'retarget_inputs_raw.npz'}")
  print(f"Wrote: {output_dir / 'summary.json'}")
  if args.plot:
    print(f"Wrote: {output_dir / 'raw_trajectories.png'}")


if __name__ == "__main__":
  main()
