"""XZ1 privileged HIM terrain teacher in the Robocon2027 MuJoCo arena."""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import mujoco.viewer
import numpy as np
import torch

from .camera import FollowCameraConfig, follow_robot
from .config import CONTROL_DT, ROOT
from .gamepad import GamepadController, format_gamepads, list_gamepads
from .him.observation import HIMObservationBuilder
from .teacher import TeacherHeightScan, TeacherPolicy
from .xz1_him import SCENE, XZ1IO

RUN = (
    ROOT / "logs/rsl_rl/xz1_wmp_him_teacher/2026-10-30_10-01-18_height_single_10hz_from_him"
)
DEFAULT_POLICY = RUN / "model_13100.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--scene", type=Path, default=SCENE)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
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
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--no-camera-follow", action="store_true")
    parser.add_argument("--record", type=Path, help="New JSON state/action/scan trace")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.duration < 0 or (args.headless and args.duration <= 0):
        parser.error("--headless requires --duration > 0")
    if not np.isfinite(
        [args.vx, args.vy, args.yaw, *args.spawn, args.spawn_yaw, args.duration]
    ).all():
        parser.error("Commands, spawn and duration must be finite")
    if args.record is not None and args.record.exists():
        parser.error(f"Output already exists: {args.record}")
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
    policy = TeacherPolicy(args.policy, args.device)
    io = XZ1IO(args.scene, args.spawn, args.spawn_yaw)
    scan_sensor = TeacherHeightScan(io.model)
    builder = HIMObservationBuilder()
    viewer = None
    gamepad = None
    viewer_threads: set[threading.Thread] = set()
    reset_requested = False
    steps = resets = 0
    records = []
    inference_ms = []

    def on_key(key: int) -> None:
        nonlocal reset_requested
        if key == ord("R"):
            reset_requested = True

    print(f"[XZ1 teacher] policy: {args.policy}")
    print(f"[XZ1 teacher] scene: {args.scene}")
    print(
        f"[XZ1 teacher] actor50Hz / physics500Hz; HIM47x6 + scan12x11 x{policy.scan_history_length} frames"
    )
    print(
        f"[XZ1 teacher] scan input: {1 / getattr(policy, 'scan_update_period', 0.02):g} Hz"
    )
    print("[XZ1 teacher] Oracle terrain scan; no depth camera or World Model input")
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
            print(f"[XZ1 teacher] gamepad: {gamepad.name} ({gamepad.path})")
            print(f"[XZ1 teacher] {gamepad.axis_mapping}")
        if not args.headless:
            previous_threads = set(threading.enumerate())
            viewer = mujoco.viewer.launch_passive(
                io.model, io.data, key_callback=on_key
            )
            viewer_threads = set(threading.enumerate()) - previous_threads
        print(
            "[XZ1 teacher] left stick vx/vy; right X yaw; Start / R resets robot and history"
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
            scan = scan_sensor.read(io.data)
            action = policy.step(obs, scan)
            inference_ms.append((time.perf_counter() - before) * 1000)
            io.send_target(io.action_offset + io.action_scale * action)
            io.step()
            builder.advance(CONTROL_DT)
            if not (
                np.isfinite(io.data.qpos).all() and np.isfinite(io.data.qvel).all()
            ):
                raise FloatingPointError("Non-finite simulation state")
            steps += 1
            if args.record is not None:
                assert policy.scan_history is not None
                records.append(
                    {
                        "t": steps * CONTROL_DT,
                        "reset": resets,
                        "command": command.tolist(),
                        "root": io.data.qpos[:7].tolist(),
                        "action": action.tolist(),
                        "teacher_scan": policy.scan_history[-1].tolist(),
                        "teacher_scan_history": policy.scan_history.tolist(),
                    }
                )
            if viewer is not None:
                if not args.no_camera_follow:
                    follow_robot(viewer, io.data.qpos[:3], follow_config)
                viewer.sync()
            if args.duration and steps * CONTROL_DT >= args.duration:
                break
            if viewer is not None:
                deadline += CONTROL_DT
                time.sleep(max(0.0, deadline - time.perf_counter()))
                deadline = max(deadline, time.perf_counter() - CONTROL_DT)
    finally:
        if viewer is not None:
            viewer.close()
            for thread in viewer_threads:
                thread.join(timeout=5.0)
        if gamepad is not None:
            gamepad.close()
    result = {
        "policy": str(args.policy.resolve()),
        "scene": str(args.scene.resolve()),
        "simulated_s": steps * CONTROL_DT,
        "wall_s": time.perf_counter() - started,
        "control_steps": steps,
        "resets": resets,
        "mean_inference_ms": float(np.mean(inference_ms)) if inference_ms else 0,
        "final_base": io.data.qpos[:3].tolist(),
        "records": records,
    }
    if args.record is not None:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        with args.record.open("x") as stream:
            json.dump(result, stream, indent=2)
    print(
        "[XZ1 teacher] "
        + json.dumps({k: v for k, v in result.items() if k != "records"})
    )


if __name__ == "__main__":
    main()
