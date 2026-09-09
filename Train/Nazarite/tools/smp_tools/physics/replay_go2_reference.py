"""Render a geometric Go2 reference trajectory without physics simulation.

The replayer assigns each exported ``qpos`` directly, calls MuJoCo forward
kinematics, and records the result.  It is the visual check between geometric
IK and the later PD-controlled physics-tracking stage.

Example:

    cd Train/Nazarite
    uv run python tools/smp_tools/physics/replay_go2_reference.py \\
      --input output/go2_retarget/d29_t1_a/go2_reference_geometric.npz \\
      --output output/go2_retarget/d29_t1_a/go2_geometric_replay.gif
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

# MuJoCo needs the rendering backend before its Python module is imported.
# EGL is available in the Nazarite development environment and supports
# headless/offscreen rendering without opening an interactive viewer window.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

if TYPE_CHECKING:
  from tools.smp_tools.retargeting.inspect_go2_kinematics import DEFAULT_GO2_XML
else:
  if str(Path(__file__).resolve().parents[3]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
  from tools.smp_tools.retargeting.inspect_go2_kinematics import DEFAULT_GO2_XML


def _load_reference(input_path: Path) -> dict[str, np.ndarray]:
  required_keys = {"qpos", "fps", "root_pos_mujoco", "ik_success"}
  with np.load(input_path, allow_pickle=False) as archive:
    missing = sorted(required_keys - set(archive.files))
    if missing:
      raise ValueError(f"reference NPZ lacks required arrays: {', '.join(missing)}")
    reference = {key: archive[key].copy() for key in required_keys}
  if reference["qpos"].ndim != 2:
    raise ValueError("qpos must have shape [frames, nq]")
  frame_count = reference["qpos"].shape[0]
  if reference["root_pos_mujoco"].shape != (frame_count, 3):
    raise ValueError("root_pos_mujoco must have shape [frames, 3]")
  if reference["ik_success"].shape != (frame_count,):
    raise ValueError("ik_success must have shape [frames]")
  if not np.isfinite(reference["qpos"]).all():
    raise ValueError("reference qpos contains non-finite values")
  return reference


def _camera_for_base(base_position: np.ndarray, follow: bool) -> mujoco.MjvCamera:
  camera = mujoco.MjvCamera()
  camera.type = mujoco.mjtCamera.mjCAMERA_FREE
  camera.azimuth = 132.0
  camera.elevation = -18.0
  camera.distance = 1.45
  if follow:
    camera.lookat[:] = base_position
  return camera


def _write_gif(frames: list[np.ndarray], output_path: Path, fps: int) -> None:
  try:
    from PIL import Image
  except ImportError as exc:
    raise RuntimeError("GIF output requires Pillow in the Nazarite environment") from exc
  if not frames:
    raise ValueError("cannot write an animation with zero frames")
  images = [Image.fromarray(frame) for frame in frames]
  duration_ms = round(1000.0 / fps)
  images[0].save(
    output_path,
    save_all=True,
    append_images=images[1:],
    duration=duration_ms,
    loop=0,
    disposal=2,
  )


def _write_mp4(frames: list[np.ndarray], output_path: Path, fps: int) -> None:
  """Encode MP4 through imageio only when its ffmpeg backend is available."""
  try:
    import imageio.v3 as imageio
  except ImportError as exc:
    raise RuntimeError(
      "MP4 output requires imageio with an ffmpeg backend; choose a .gif path instead"
    ) from exc
  try:
    imageio.imwrite(output_path, np.stack(frames), fps=fps)
  except Exception as exc:
    raise RuntimeError(
      "MP4 encoding failed. Install ffmpeg/imageio-ffmpeg or choose a .gif output path"
    ) from exc


def render_reference(
  model: mujoco.MjModel,
  reference: dict[str, np.ndarray],
  output_path: Path,
  output_fps: int,
  width: int,
  height: int,
  stride: int,
  follow_camera: bool,
) -> int:
  """Return the number of rendered frames after writing GIF or MP4 output."""
  if reference["qpos"].shape[1] != model.nq:
    raise ValueError(
      f"reference nq {reference['qpos'].shape[1]} does not match Go2 model nq {model.nq}"
    )
  base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
  if base_id < 0:
    raise ValueError("Go2 model has no base_link body")
  data = mujoco.MjData(model)
  frames: list[np.ndarray] = []
  renderer = mujoco.Renderer(model, width=width, height=height)
  try:
    world_camera = _camera_for_base(reference["root_pos_mujoco"][0], follow=False)
    if not follow_camera:
      world_camera.lookat[:] = np.mean(reference["root_pos_mujoco"], axis=0)
      world_camera.distance = max(1.45, 0.45 * np.ptp(reference["root_pos_mujoco"][:, 0]) + 1.1)
    for frame_index in range(0, reference["qpos"].shape[0], stride):
      data.qpos[:] = reference["qpos"][frame_index]
      mujoco.mj_forward(model, data)
      camera = (
        _camera_for_base(data.xpos[base_id], follow=True) if follow_camera else world_camera
      )
      renderer.update_scene(data, camera=camera)
      frames.append(renderer.render().copy())
  finally:
    renderer.close()

  if output_path.suffix.lower() == ".gif":
    _write_gif(frames, output_path, output_fps)
  elif output_path.suffix.lower() == ".mp4":
    _write_mp4(frames, output_path, output_fps)
  else:
    raise ValueError("--output must end in .gif or .mp4")
  return len(frames)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", type=Path, required=True, help="Geometric Go2 reference NPZ")
  parser.add_argument("--output", type=Path, required=True, help="Output .gif or .mp4")
  parser.add_argument("--xml", type=Path, default=DEFAULT_GO2_XML, help="Go2 MuJoCo XML")
  parser.add_argument(
    "--output-fps", type=int, help="Rendered FPS; default is source FPS divided by stride"
  )
  parser.add_argument("--stride", type=int, default=1, help="Render every Nth source frame")
  parser.add_argument("--width", type=int, default=640, help="Output width in pixels")
  parser.add_argument("--height", type=int, default=480, help="Output height in pixels")
  parser.add_argument(
    "--world-camera", action="store_true", help="Keep one fixed camera instead of following Go2"
  )
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
  args = _parse_args(argv)
  if args.stride <= 0 or args.width <= 0 or args.height <= 0:
    raise ValueError("stride, width, and height must be positive")
  if args.output_fps is not None and args.output_fps <= 0:
    raise ValueError("--output-fps must be positive")

  input_path: Path = args.input.expanduser().resolve()
  xml_path: Path = args.xml.expanduser().resolve()
  output_path: Path = args.output.expanduser().resolve()
  if not input_path.is_file():
    raise FileNotFoundError(input_path)
  if not xml_path.is_file():
    raise FileNotFoundError(xml_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)

  reference = _load_reference(input_path)
  source_fps = float(np.asarray(reference["fps"]).item())
  output_fps = args.output_fps or max(1, round(source_fps / args.stride))
  model = mujoco.MjModel.from_xml_path(str(xml_path))
  rendered_frames = render_reference(
    model=model,
    reference=reference,
    output_path=output_path,
    output_fps=output_fps,
    width=args.width,
    height=args.height,
    stride=args.stride,
    follow_camera=not args.world_camera,
  )
  summary = {
    "source": str(input_path),
    "go2_xml": str(xml_path),
    "output": str(output_path),
    "source_frames": int(reference["qpos"].shape[0]),
    "rendered_frames": rendered_frames,
    "source_fps": source_fps,
    "output_fps": output_fps,
    "stride": args.stride,
    "follow_camera": not args.world_camera,
    "input_valid_fraction": float(np.mean(reference["ik_success"])),
  }
  with (output_path.parent / "replay_summary.json").open("w", encoding="utf-8") as stream:
    json.dump(summary, stream, indent=2)
    stream.write("\n")
  print(json.dumps(summary, indent=2))
  print(f"Wrote: {output_path}")
  print(f"Wrote: {output_path.parent / 'replay_summary.json'}")


if __name__ == "__main__":
  main()
