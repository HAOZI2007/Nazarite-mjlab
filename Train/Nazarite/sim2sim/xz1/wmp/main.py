"""XZ1 HIM + WMP visual student in the Robocon2027 CPU MuJoCo arena."""

from __future__ import annotations

import argparse
import json
import threading
import time
from functools import partial
from pathlib import Path

import mujoco.viewer
import numpy as np
import torch

from sim2sim.shared.camera import FollowCameraConfig, follow_robot
from sim2sim.shared.config import CONTROL_DT, ROOT
from sim2sim.shared.gamepad import GamepadController, format_gamepads, list_gamepads
from sim2sim.shared.him.observation import HIMObservationBuilder
from sim2sim.xz1.him.main import SCENE, XZ1IO
from sim2sim.xz1.wmp.camera import WMPDepthCamera, install_camera
from sim2sim.xz1.wmp.policy import WMPStudentPolicy, load_bundle
from sim2sim.xz1.wmp.preview import WMPDepthPreview

DEFAULT_POLICY = ROOT / "logs/rsl_rl/xz1_wmp_student/stage4_nine_v1/student_3000.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument(
        "--world-model", type=Path, help="Relocated matching WM checkpoint"
    )
    parser.add_argument("--export-bundle", type=Path, help="Save actor+WM and exit")
    parser.add_argument("--scene", type=Path, default=SCENE)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamepad", action="store_true")
    parser.add_argument("--list-gamepads", action="store_true")
    parser.add_argument("--gamepad-device")
    parser.add_argument("--max-vx", type=float, default=1.0)
    parser.add_argument("--max-vy", type=float, default=1.0)
    parser.add_argument("--max-yaw", type=float, default=1.0)
    parser.add_argument("--deadzone", type=float, default=0.08)
    parser.add_argument("--axis-vx", default="ABS_Y")
    parser.add_argument("--axis-vy", default="ABS_X")
    parser.add_argument("--axis-yaw", default="ABS_RX")
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0, help="yaw rate, rad/s")
    parser.add_argument("--spawn", type=float, nargs=3, default=(0.0, 0.0, 0.30))
    parser.add_argument("--spawn-yaw", type=float, default=0.0, help="degrees")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--duration", type=float, default=0.0, help="sim seconds; 0 unlimited"
    )
    parser.add_argument("--no-camera-follow", action="store_true")
    parser.add_argument(
        "--no-depth-window",
        action="store_true",
        help="Hide the policy depth preview (always hidden in headless mode)",
    )
    parser.add_argument(
        "--record", type=Path, help="New JSON trace, with last depth .npy"
    )
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.duration < 0 or (
        args.headless and args.duration <= 0 and args.export_bundle is None
    ):
        parser.error("--headless requires --duration > 0")
    values = [args.vx, args.vy, args.yaw, *args.spawn, args.spawn_yaw, args.duration]
    if not np.isfinite(values).all():
        parser.error("Commands, spawn and duration must be finite")
    for path in (args.export_bundle, args.record):
        if path is not None and path.exists():
            parser.error(f"Output already exists: {path}")
    if args.record is not None and args.record.with_suffix(".depth.npy").exists():
        parser.error("Recorded depth output already exists")
    return args


def main() -> None:
    args = parse_args()
    if args.list_gamepads:
        devices = list_gamepads()
        print(
            format_gamepads(devices) if devices else "No compatible gamepad detected."
        )
        return
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    bundle = load_bundle(args.policy, args.world_model)
    if args.export_bundle is not None:
        args.export_bundle.parent.mkdir(parents=True, exist_ok=True)
        with args.export_bundle.open("xb") as stream:
            torch.save(bundle, stream)
        print(f"[XZ1 WMP] exported actor+WM: {args.export_bundle.resolve()}")
        return
    policy = WMPStudentPolicy(bundle, args.device)
    io = XZ1IO(
        args.scene,
        args.spawn,
        args.spawn_yaw,
        configure_robot=partial(install_camera, camera=bundle["camera"]),
    )
    depth = WMPDepthCamera(io.model, bundle["camera"])
    builder = HIMObservationBuilder()
    gamepad = None
    viewer = None
    depth_preview = None
    viewer_threads: set[threading.Thread] = set()
    reset_requested = False
    records = []
    inference_ms = []
    image_updates = 0
    resets = 0

    def on_key(key: int) -> None:
        nonlocal reset_requested
        if key == ord("R"):
            reset_requested = True

    print(f"[XZ1 WMP] policy: {args.policy}")
    print(f"[XZ1 WMP] scene: {args.scene}")
    print(f"[XZ1 WMP] camera: {bundle['camera']}")
    print(f"[XZ1 WMP] device={args.device}; actor50Hz / WM10Hz / physics500Hz")
    print("[XZ1 WMP] depth64x64 + prop33 + preceding action5x12 -> deter512")
    steps = 0
    started = time.perf_counter()
    try:
        if args.gamepad:
            gamepad = GamepadController(
                args.gamepad_device,
                max_vx=args.max_vx,
                max_vy=args.max_vy,
                max_yaw=args.max_yaw,
                deadzone=args.deadzone,
                axis_vx=args.axis_vx,
                axis_vy=args.axis_vy,
                axis_yaw=args.axis_yaw,
            )
            print(f"[XZ1 WMP] gamepad: {gamepad.name} ({gamepad.path})")
            print(f"[XZ1 WMP] {gamepad.axis_mapping}")
            print("[XZ1 WMP] left stick vx/vy; right X yaw; Start resets all history")
        if not args.headless:
            previous_threads = set(threading.enumerate())
            viewer = mujoco.viewer.launch_passive(
                io.model, io.data, key_callback=on_key
            )
            viewer_threads = set(threading.enumerate()) - previous_threads
            print("[XZ1 WMP] R: reset robot, phase, history, actions and RSSM")
            if not args.no_depth_window:
                depth_preview = WMPDepthPreview(
                    bundle["camera"]["far_clip"], partial(on_key, ord("R"))
                )
                print(
                    "[XZ1 WMP] depth preview: actual policy captures, near white / far black"
                )
        follow_config = FollowCameraConfig()
        deadline = time.perf_counter()
        while viewer is None or viewer.is_running():
            command = (
                gamepad.poll()
                if gamepad is not None
                else np.array([args.vx, args.vy, args.yaw], dtype=np.float32)
            )
            if reset_requested or (
                gamepad is not None and gamepad.consume_button("BTN_START")
            ):
                io.reset()
                builder.reset()
                policy.reset()
                reset_requested = False
                resets += 1
            command = np.clip(command, -1.0, 1.0)
            obs = builder.build(io, command, policy.last_action)
            before = time.perf_counter()
            update = policy.tick % 5 == 0
            action = policy.step(obs, lambda: depth.read(io.data))
            inference_ms.append((time.perf_counter() - before) * 1000)
            image_updates += int(update)
            io.send_target(io.action_offset + io.action_scale * action)
            io.step()
            policy.advance(action)
            builder.advance(CONTROL_DT)
            if not (
                np.isfinite(io.data.qpos).all() and np.isfinite(io.data.qvel).all()
            ):
                raise FloatingPointError("Non-finite simulation state")
            if args.record is not None:
                records.append(
                    {
                        "t": (steps + 1) * CONTROL_DT,
                        "reset": resets,
                        "command": command.tolist(),
                        "root": io.data.qpos[:7].tolist(),
                        "action": action.tolist(),
                        "wm_updated": update,
                        "feature_norm": float(policy.feature.norm()),
                    }
                )
            steps += 1
            if viewer is not None:
                if not args.no_camera_follow:
                    follow_robot(viewer, io.data.qpos[:3], follow_config)
                viewer.sync()
            if depth_preview is not None:
                if update and policy.last_depth is not None:
                    depth_preview.update(policy.last_depth)
                depth_preview.poll()
            if args.duration and steps * CONTROL_DT >= args.duration:
                break
            if viewer is not None:
                deadline += CONTROL_DT
                time.sleep(max(0.0, deadline - time.perf_counter()))
                deadline = max(deadline, time.perf_counter() - CONTROL_DT)
    finally:
        if depth_preview is not None:
            depth_preview.close()
        if viewer is not None:
            viewer.close()
            for thread in viewer_threads:
                thread.join(timeout=5.0)
        if gamepad is not None:
            gamepad.close()
    result = {
        "policy": str(args.policy.resolve()),
        "sources": bundle["sources"],
        "scene": str(args.scene.resolve()),
        "camera": bundle["camera"],
        "seed": args.seed,
        "simulated_s": steps * CONTROL_DT,
        "wall_s": time.perf_counter() - started,
        "control_steps": steps,
        "image_updates": image_updates,
        "resets": resets,
        "mean_inference_ms": float(np.mean(inference_ms)) if inference_ms else 0,
        "p95_inference_ms": float(np.percentile(inference_ms, 95))
        if inference_ms
        else 0,
        "final_base": io.data.qpos[:3].tolist(),
        "records": records,
    }
    if args.record is not None:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        with args.record.open("x") as stream:
            json.dump(result, stream, indent=2)
        if policy.last_depth is not None:
            with args.record.with_suffix(".depth.npy").open("xb") as stream:
                np.save(stream, policy.last_depth)
    print(
        "[XZ1 WMP] " + json.dumps({k: v for k, v in result.items() if k != "records"})
    )


if __name__ == "__main__":
    main()
