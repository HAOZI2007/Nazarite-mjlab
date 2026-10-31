from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from sim2sim.go2.baseline import config as baseline_config
from sim2sim.go2.baseline.observation import build_observation
from sim2sim.go2.wtw import config as wtw_config
from sim2sim.go2.wtw.behavior import WTWBehaviorController
from sim2sim.go2.wtw.command import restrict_command as wtw_command
from sim2sim.go2.wtw.observation import WTWObservationBuilder
from sim2sim.shared.camera import FollowCameraConfig, follow_robot
from sim2sim.shared.config import ACTION_SCALE, CONTROL_DT
from sim2sim.shared.gamepad import GamepadController, format_gamepads, list_gamepads
from sim2sim.shared.him import config as him_config
from sim2sim.shared.him.observation import HIMObservationBuilder
from sim2sim.shared.him.policy import HIMPolicy
from sim2sim.shared.him.scene import add_him_scene
from sim2sim.shared.math_utils import action_to_target
from sim2sim.shared.mujoco_io import MuJoCoIO
from sim2sim.shared.policy_runner import PolicyRunner
from sim2sim.shared.scene import add_simple_grid_scene, add_stairs_and_slopes_scene


def default_policy_path(mode: str = "baseline") -> Path:
    if mode == "him":
        policy_path = him_config.POLICY
    elif mode == "wtw":
        policy_path = wtw_config.POLICY
    else:
        policy_path = baseline_config.POLICY
    if not policy_path.is_file():
        raise FileNotFoundError(
            f"Default policy not found: {policy_path}. "
            "Pass --policy /path/to/policy.onnx or /path/to/model.pt."
        )
    return policy_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nazarite Go2 MuJoCo sim2sim")
    parser.add_argument(
        "--mode",
        choices=("baseline", "him", "wtw"),
        default="baseline",
    )
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument(
        "--gamepad", action="store_true", help="control velocity using a Linux gamepad"
    )
    parser.add_argument(
        "--list-gamepads", action="store_true", help="list detected gamepads and exit"
    )
    parser.add_argument(
        "--gamepad-device",
        type=str,
        default=None,
        help="evdev path, for example /dev/input/event21",
    )
    parser.add_argument(
        "--max-vx",
        type=float,
        default=1.0,
        help="gamepad maximum forward velocity (m/s)",
    )
    parser.add_argument(
        "--max-vy",
        type=float,
        default=1.0,
        help="gamepad maximum lateral velocity (m/s)",
    )
    parser.add_argument(
        "--max-yaw", type=float, default=1.0, help="gamepad maximum yaw rate (rad/s)"
    )
    parser.add_argument(
        "--deadzone", type=float, default=0.08, help="stick deadzone in [0, 1)"
    )
    parser.add_argument(
        "--axis-vx", default="ABS_Y", help="forward stick axis (default: ABS_Y)"
    )
    parser.add_argument(
        "--axis-vy", default="ABS_X", help="lateral stick axis (default: ABS_X)"
    )
    parser.add_argument(
        "--axis-yaw", default="ABS_RX", help="yaw stick axis (default: ABS_RX)"
    )
    parser.add_argument(
        "--no-camera-follow",
        action="store_true",
        help="disable the robot-follow camera",
    )
    parser.add_argument(
        "--camera-distance",
        type=float,
        default=2.5,
        help="camera distance from robot (m)",
    )
    parser.add_argument(
        "--camera-azimuth",
        type=float,
        default=135.0,
        help="camera horizontal angle (deg)",
    )
    parser.add_argument(
        "--camera-elevation",
        type=float,
        default=-20.0,
        help="camera elevation angle (deg)",
    )
    parser.add_argument(
        "--camera-height",
        type=float,
        default=0.12,
        help="camera look-at height above robot base (m)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_gamepads:
        gamepads = list_gamepads()
        print(
            format_gamepads(gamepads) if gamepads else "No compatible gamepad detected."
        )
        return

    policy_path = args.policy or default_policy_path(args.mode)
    command = np.array([args.vx, args.vy, args.yaw], dtype=np.float32)
    gamepad: GamepadController | None = None
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

    def scene_builder(spec: mujoco.MjSpec) -> None:
        if args.mode == "wtw":
            add_stairs_and_slopes_scene(spec)
        elif args.mode == "him":
            add_him_scene(spec)
        else:
            add_simple_grid_scene(spec)

    if args.mode == "him":
        io = MuJoCoIO(
            hip_effort=him_config.HIP_EFFORT,
            calf_effort=him_config.CALF_EFFORT,
            scene_builder=scene_builder,
        )
    elif args.mode == "wtw":
        io = MuJoCoIO(
            hip_effort=wtw_config.HIP_EFFORT,
            calf_effort=wtw_config.CALF_EFFORT,
            scene_builder=scene_builder,
        )
    else:
        io = MuJoCoIO(scene_builder=scene_builder)
    standard_policy: PolicyRunner | None = None
    him_policy: HIMPolicy | None = None
    him_builder: HIMObservationBuilder | None = None
    if args.mode == "him":
        him_policy = HIMPolicy(policy_path)
        him_builder = HIMObservationBuilder()
    elif args.mode == "wtw":
        expected_obs_dim = wtw_config.OBS_DIM
        expected_observation_names = wtw_config.OBSERVATION_NAMES
        standard_policy = PolicyRunner(
            policy_path,
            expected_obs_dim=expected_obs_dim,
            expected_observation_names=expected_observation_names,
        )
    else:
        expected_obs_dim = baseline_config.OBS_DIM
        expected_observation_names = baseline_config.OBSERVATION_NAMES
        standard_policy = PolicyRunner(
            policy_path,
            expected_obs_dim=expected_obs_dim,
            expected_observation_names=expected_observation_names,
        )
    wtw_builder: WTWObservationBuilder | None
    if args.mode == "wtw":
        wtw_builder = WTWObservationBuilder()
    else:
        wtw_builder = None
    wtw_behavior_controller: WTWBehaviorController | None = None
    if wtw_builder is not None and gamepad is not None:
        wtw_behavior_controller = WTWBehaviorController(
            wtw_builder,
            gamepad,
            status_callback=print,
        )
    io.reset()
    if standard_policy is not None:
        standard_policy.reset()
    if him_policy is not None:
        him_policy.reset()
    if him_builder is not None:
        him_builder.reset()
    if wtw_builder is not None:
        wtw_builder.reset()
    camera_config = FollowCameraConfig(
        distance=args.camera_distance,
        azimuth=args.camera_azimuth,
        elevation=args.camera_elevation,
        target_height=args.camera_height,
    )

    print(f"[sim2sim] policy: {policy_path}")
    policy_obs_dim = (
        him_policy.obs_dim
        if him_policy is not None
        else standard_policy.obs_dim
        if standard_policy is not None
        else 0
    )
    print(f"[sim2sim] mode={args.mode}, policy_obs_dim={policy_obs_dim}")
    print(
        f"[sim2sim] timestep={io.model.opt.timestep:.4f}s, control_dt={CONTROL_DT:.4f}s"
    )
    if gamepad is None:
        print(f"[sim2sim] fixed command={command.tolist()}")
    else:
        print(f"[sim2sim] gamepad={gamepad.name} ({gamepad.path})")
        print(f"[sim2sim] gamepad axes: {gamepad.axis_mapping}")
        print("[sim2sim] left stick: vx/vy, right stick X: yaw")

    next_tick = time.perf_counter()
    try:
        with mujoco.viewer.launch_passive(io.model, io.data) as viewer:
            if not args.no_camera_follow:
                follow_robot(viewer, io.data.qpos[:3], camera_config)
            while viewer.is_running():
                if gamepad is not None:
                    command = gamepad.poll()
                    if wtw_behavior_controller is not None:
                        wtw_behavior_controller.update()

                if him_builder is not None:
                    if him_policy is None:
                        raise RuntimeError("HIM mode requires a policy")
                    command = np.clip(command, -1.0, 1.0)
                    obs = him_builder.build(io, command, him_policy.last_action)
                    raw_action = him_policy.step(obs)
                elif wtw_builder is not None:
                    command = wtw_command(command)
                    if standard_policy is None:
                        raise RuntimeError("WTW mode requires a policy")
                    obs = wtw_builder.build(io, command, standard_policy.last_action)
                    raw_action = standard_policy.step(obs)
                else:
                    if standard_policy is None:
                        raise RuntimeError("Baseline mode requires a policy")
                    obs = build_observation(io, command, standard_policy.last_action)
                    raw_action = standard_policy.step(obs)
                target_q = action_to_target(raw_action, io.default_q, ACTION_SCALE)
                io.send_target(target_q)
                io.step()
                if him_builder is not None:
                    him_builder.advance(CONTROL_DT)
                if wtw_builder is not None:
                    wtw_builder.advance_phase(command, CONTROL_DT)
                if not args.no_camera_follow:
                    follow_robot(viewer, io.data.qpos[:3], camera_config)
                viewer.sync()

                next_tick += CONTROL_DT
                sleep_time = next_tick - time.perf_counter()
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
                elif sleep_time < -CONTROL_DT:
                    next_tick = time.perf_counter()
    finally:
        if gamepad is not None:
            gamepad.close()


if __name__ == "__main__":
    main()
