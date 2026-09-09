"""Render a 3DDogs optical recording as a moving marker-and-bone animation.

This is a diagnostic view of the real optical MoCap data, not a fitted dog
mesh. It makes marker loss, limb ordering, and raw-to-MuJoCo axis choices
visible before Go2 inverse-kinematics retargeting.

Example:

    cd Train/Nazarite
    uv run python tools/smp_tools/data/visualize_3ddogs_mocap.py \\
      --input /home/haozi/3DDogs/3DDogs2024_full/Data/Optical/ \\
Sync_Align_v2023_11_16b/optical_sync_align_d29_t1_a.txt \\
      --start-index 0 --end-index 132 --coordinate-frame mujoco \\
      --output output/3ddogs_visualizations/d29_t1_a_mujoco.mp4
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

# Every edge is a physical limb/trunk connection in the optical marker model.
# The graph deliberately uses markers, rather than inventing joints that are
# absent from the dataset.
SKELETON_EDGES: tuple[tuple[str, str], ...] = (
  ("poll", "withers"),
  ("withers", "sacrum"),
  ("l_shdr", "r_shdr"),
  ("l_iliac", "r_iliac"),
  ("l_shdr", "l_iliac"),
  ("r_shdr", "r_iliac"),
  ("l_shdr", "l_elb"),
  ("l_elb", "l_dist_rad"),
  ("l_dist_rad", "l_meta_carp"),
  ("r_shdr", "r_elb"),
  ("r_elb", "r_dist_rad"),
  ("r_dist_rad", "r_meta_carp"),
  ("l_iliac", "l_grt_trc"),
  ("l_grt_trc", "l_stifle"),
  ("l_stifle", "l_hock"),
  ("l_hock", "l_meta_tars"),
  ("r_iliac", "r_grt_trc"),
  ("r_grt_trc", "r_stifle"),
  ("r_stifle", "r_hock"),
  ("r_hock", "r_meta_tars"),
)
SKELETON_MARKERS = tuple(sorted({marker for edge in SKELETON_EDGES for marker in edge}))

# This right-handed map was validated on d29_t1_a. It converts 3DDogs y-up to
# MuJoCo z-up and maps that dog's forward/left anatomy to MuJoCo +x/+y.
RAW_TO_MUJOCO = np.array(
  ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)), dtype=np.float64
)

if TYPE_CHECKING:
  from tools.smp_tools.data.inspect_3ddogs import MocapTrial, load_optical_trial
else:
  from inspect_3ddogs import MocapTrial, load_optical_trial


def _axis_limits(points: np.ndarray) -> tuple[tuple[float, float], ...]:
  """Return equal-scale xyz bounds with a modest diagnostic margin."""
  lower = np.nanmin(points, axis=(0, 1))
  upper = np.nanmax(points, axis=(0, 1))
  center = 0.5 * (lower + upper)
  span = max(float(np.max(upper - lower)), 0.1)
  half_span = 0.55 * span
  return tuple((float(value - half_span), float(value + half_span)) for value in center)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", type=Path, required=True, help="Source optical .txt file")
  parser.add_argument("--output", type=Path, required=True, help="Output .mp4 or .gif path")
  parser.add_argument("--start-index", type=int, default=0, help="Inclusive frame index")
  parser.add_argument("--end-index", type=int, help="Exclusive frame index; default: end")
  parser.add_argument(
    "--coordinate-frame",
    choices=("raw", "mujoco"),
    default="mujoco",
    help="Render source 3DDogs axes or the validated candidate MuJoCo axes",
  )
  parser.add_argument(
    "--camera-mode",
    choices=("follow", "world"),
    default="follow",
    help="Follow the trunk for a readable gait view, or keep world axes fixed",
  )
  parser.add_argument(
    "--output-fps",
    type=int,
    help="Animation FPS; default is the source recording FPS",
  )
  parser.add_argument("--elevation", type=float, default=18.0, help="3D camera elevation")
  parser.add_argument("--azimuth", type=float, default=-62.0, help="3D camera azimuth")
  return parser.parse_args(argv)


def _writer_for(output_path: Path, fps: int):
  """Choose a Matplotlib writer by extension and give actionable failures."""
  from matplotlib import animation

  extension = output_path.suffix.lower()
  if extension == ".mp4":
    if not animation.writers.is_available("ffmpeg"):
      raise RuntimeError("MP4 output needs ffmpeg; install it or choose a .gif output path")
    return animation.FFMpegWriter(fps=fps, bitrate=2800)
  if extension == ".gif":
    if not animation.writers.is_available("pillow"):
      raise RuntimeError("GIF output needs Pillow; install it in the Nazarite environment")
    return animation.PillowWriter(fps=fps)
  raise ValueError("--output must end in .mp4 or .gif")


def render_animation(
  trial: MocapTrial,
  start_index: int,
  end_index: int,
  output_path: Path,
  coordinate_frame: str,
  camera_mode: str,
  output_fps: int,
  elevation: float,
  azimuth: float,
) -> None:
  """Render the selected fully observed skeleton segment to MP4 or GIF."""
  marker_indexes = {name: index for index, name in enumerate(trial.marker_names)}
  missing_markers = sorted(set(SKELETON_MARKERS) - set(marker_indexes))
  if missing_markers:
    raise ValueError(f"trial lacks skeleton markers: {', '.join(missing_markers)}")

  clip = trial.marker_positions[start_index:end_index]
  skeleton_indexes = [marker_indexes[name] for name in SKELETON_MARKERS]
  if not np.isfinite(clip[:, skeleton_indexes]).any():
    raise ValueError(
      "selected clip has no finite skeleton markers; check the selected frame range"
    )
  if coordinate_frame == "mujoco":
    clip = clip @ RAW_TO_MUJOCO.T

  edge_indexes = [(marker_indexes[start], marker_indexes[end]) for start, end in SKELETON_EDGES]
  skeleton_clip = clip[:, skeleton_indexes]
  limits = _axis_limits(skeleton_clip)
  paw_indexes = [
    marker_indexes[marker]
    for marker in ("l_meta_carp", "r_meta_carp", "l_meta_tars", "r_meta_tars")
  ]
  ground_height = float(np.nanmin(clip[:, paw_indexes, 2]))
  root_positions = 0.5 * (
    clip[:, marker_indexes["withers"]] + clip[:, marker_indexes["sacrum"]]
  )
  local_skeleton = skeleton_clip - root_positions[:, None]
  follow_half_span = max(0.35, 1.25 * float(np.nanmax(np.abs(local_skeleton))))

  # Set the cache before importing pyplot so the script also works under
  # accounts where ~/.config is read-only.
  os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".matplotlib"))
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  figure = plt.figure(figsize=(8, 7))
  # Matplotlib's 3D artist stubs omit methods such as ``set_data_3d`` even
  # though they are present at runtime; keep this narrow plotting boundary
  # dynamically typed rather than weakening types in data processing.
  axes: Any = figure.add_subplot(projection="3d")
  if camera_mode == "world":
    axes.set_xlim(*limits[0])
    axes.set_ylim(*limits[1])
    axes.set_zlim(*limits[2])
  else:
    initial_center = root_positions[0]
    axes.set_xlim(initial_center[0] - follow_half_span, initial_center[0] + follow_half_span)
    axes.set_ylim(initial_center[1] - follow_half_span, initial_center[1] + follow_half_span)
    axes.set_zlim(initial_center[2] - follow_half_span, initial_center[2] + follow_half_span)
  axes.set_box_aspect((1.0, 1.0, 0.65))
  axes.view_init(elev=elevation, azim=azimuth)
  axes.set_xlabel("x (m)")
  axes.set_ylabel("y (m)")
  axes.set_zlabel("z (m)")
  axes.set_title("3DDogs marker skeleton")

  grid_x, grid_y = np.meshgrid(
    np.linspace(limits[0][0] - 0.5, limits[0][1] + 0.5, 2),
    np.linspace(limits[1][0] - 0.5, limits[1][1] + 0.5, 2),
  )
  axes.plot_surface(grid_x, grid_y, np.full_like(grid_x, ground_height), alpha=0.12)
  marker_scatter = axes.scatter([], [], [], s=22, color="tab:blue", label="optical marker")
  bones = [axes.plot([], [], [], color="tab:orange", linewidth=2.5)[0] for _ in edge_indexes]
  axes.legend(loc="upper right")

  def draw(frame_index: int):
    positions = clip[frame_index]
    if camera_mode == "follow":
      center = root_positions[frame_index]
      axes.set_xlim(center[0] - follow_half_span, center[0] + follow_half_span)
      axes.set_ylim(center[1] - follow_half_span, center[1] + follow_half_span)
      axes.set_zlim(center[2] - follow_half_span, center[2] + follow_half_span)
    marker_is_finite = np.isfinite(positions).all(axis=1)
    visible_positions = positions[marker_is_finite]
    marker_scatter._offsets3d = (
      visible_positions[:, 0],
      visible_positions[:, 1],
      visible_positions[:, 2],
    )
    for line, (start, end) in zip(bones, edge_indexes, strict=True):
      edge_positions = positions[[start, end]]
      if np.isfinite(edge_positions).all():
        line.set_data_3d(
          edge_positions[:, 0], edge_positions[:, 1], edge_positions[:, 2]
        )
      else:
        # Marker dropouts remain visible as temporary gaps instead of being
        # imputed; this is precisely what this diagnostic animation should show.
        line.set_data_3d([], [], [])
    source_frame = int(trial.frame_numbers[start_index + frame_index])
    visible_count = int(np.count_nonzero(marker_is_finite))
    axes.set_title(
      f"3DDogs marker skeleton | {coordinate_frame} | "
      f"frame {source_frame} | t={frame_index / trial.fps:.2f}s | "
      f"visible markers: {visible_count}/{positions.shape[0]}"
    )
    return [marker_scatter, *bones]

  from matplotlib import animation

  animation_object = animation.FuncAnimation(
    figure,
    draw,
    frames=clip.shape[0],
    interval=1000.0 / output_fps,
    blit=False,
  )
  output_path.parent.mkdir(parents=True, exist_ok=True)
  animation_object.save(output_path, writer=_writer_for(output_path, output_fps), dpi=150)
  plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
  args = _parse_args(argv)
  if args.start_index < 0:
    raise ValueError("--start-index must be non-negative")
  if args.output_fps is not None and args.output_fps <= 0.0:
    raise ValueError("--output-fps must be positive")

  input_path: Path = args.input.expanduser().resolve()
  if not input_path.is_file():
    raise FileNotFoundError(input_path)
  trial = load_optical_trial(input_path)
  end_index = args.end_index if args.end_index is not None else trial.frame_numbers.size
  if end_index <= args.start_index or end_index > trial.frame_numbers.size:
    raise ValueError(
      f"expected 0 <= start < end <= {trial.frame_numbers.size}, got "
      f"{args.start_index}:{end_index}"
    )

  output_path: Path = args.output.expanduser().resolve()
  output_path.parent.mkdir(parents=True, exist_ok=True)
  render_animation(
    trial=trial,
    start_index=args.start_index,
    end_index=end_index,
    output_path=output_path,
    coordinate_frame=args.coordinate_frame,
    camera_mode=args.camera_mode,
    output_fps=args.output_fps or round(trial.fps),
    elevation=args.elevation,
    azimuth=args.azimuth,
  )
  print(f"Wrote: {output_path}")
  print(f"Frames: {end_index - args.start_index}; output FPS: {args.output_fps or trial.fps}")


if __name__ == "__main__":
  main()
