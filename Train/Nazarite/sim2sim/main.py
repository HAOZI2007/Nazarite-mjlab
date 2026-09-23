from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from .baseline import config as baseline_config
from .baseline.observation import build_observation
from .camera import FollowCameraConfig, follow_robot
from .config import ACTION_SCALE, CONTROL_DT
from .gamepad import GamepadController, format_gamepads, list_gamepads
from .math_utils import action_to_target
from .mujoco_io import MuJoCoIO
from .policy_runner import PolicyRunner
from .scene import add_stairs_and_slopes_scene
from .wtw import config as wtw_config
from .wtw.behavior import WTWBehaviorController
from .wtw.command import restrict_command as wtw_command
from .wtw.observation import WTWObservationBuilder
from .wtw_delta_residual import config as delta_config
from .wtw_delta_residual.observation import WtwDeltaObservationBuilder
from .wtw_delta_residual.policy import WtwDeltaResidualPolicy
from .wtw_delta_residual.terrain import (
    add_stepping_stones_scene,
    build_delta_map,
)


def default_policy_path(mode: str = "baseline") -> Path:
    if mode == "wtw":
        policy_path = wtw_config.POLICY
    elif mode == "wtw_delta_residual":
        policy_path = delta_config.POLICY
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
        choices=("baseline", "wtw", "wtw_delta_residual"),
        default="baseline",
    )
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--gamepad", action="store_true", help="control velocity using a Linux gamepad")
    parser.add_argument("--list-gamepads", action="store_true", help="list detected gamepads and exit")
    parser.add_argument("--gamepad-device", type=str, default=None, help="evdev path, for example /dev/input/event21")
    parser.add_argument("--max-vx", type=float, default=1.0, help="gamepad maximum forward velocity (m/s)")
    parser.add_argument("--max-vy", type=float, default=1.0, help="gamepad maximum lateral velocity (m/s)")
    parser.add_argument("--max-yaw", type=float, default=1.0, help="gamepad maximum yaw rate (rad/s)")
    parser.add_argument("--deadzone", type=float, default=0.08, help="stick deadzone in [0, 1)")
    parser.add_argument("--axis-vx", default="ABS_Y", help="forward stick axis (default: ABS_Y)")
    parser.add_argument("--axis-vy", default="ABS_X", help="lateral stick axis (default: ABS_X)")
    parser.add_argument("--axis-yaw", default="ABS_RX", help="yaw stick axis (default: ABS_RX)")
    parser.add_argument("--no-camera-follow", action="store_true", help="disable the robot-follow camera")
    parser.add_argument("--camera-distance", type=float, default=2.5, help="camera distance from robot (m)")
    parser.add_argument("--camera-azimuth", type=float, default=135.0, help="camera horizontal angle (deg)")
    parser.add_argument("--camera-elevation", type=float, default=-20.0, help="camera elevation angle (deg)")
    parser.add_argument("--camera-height", type=float, default=0.12, help="camera look-at height above robot base (m)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_gamepads:
        gamepads = list_gamepads()
        print(format_gamepads(gamepads) if gamepads else "No compatible gamepad detected.")
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

    if args.mode == "wtw":
        io = MuJoCoIO(
            hip_effort=wtw_config.HIP_EFFORT,
            calf_effort=wtw_config.CALF_EFFORT,
            scene_builder=add_stairs_and_slopes_scene,
        )
    elif args.mode == "wtw_delta_residual":
        io = MuJoCoIO(
            hip_effort=wtw_config.HIP_EFFORT,
            calf_effort=wtw_config.CALF_EFFORT,
            scene_builder=add_stepping_stones_scene,
        )
    else:
        io = MuJoCoIO()
    standard_policy: PolicyRunner | None = None
    delta_policy: WtwDeltaResidualPolicy | None = None
    if args.mode == "wtw":
        expected_obs_dim = wtw_config.OBS_DIM
        expected_observation_names = wtw_config.OBSERVATION_NAMES
        standard_policy = PolicyRunner(
            policy_path,
            expected_obs_dim=expected_obs_dim,
            expected_observation_names=expected_observation_names,
        )
    elif args.mode == "wtw_delta_residual":
        delta_policy = WtwDeltaResidualPolicy(policy_path)
    else:
        expected_obs_dim = baseline_config.OBS_DIM
        expected_observation_names = baseline_config.OBSERVATION_NAMES
        standard_policy = PolicyRunner(
            policy_path,
            expected_obs_dim=expected_obs_dim,
            expected_observation_names=expected_observation_names,
        )
    wtw_builder: WTWObservationBuilder | None
    if args.mode == "wtw_delta_residual":
        wtw_builder = WtwDeltaObservationBuilder()
    elif args.mode == "wtw":
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
    if delta_policy is not None:
        delta_policy.reset()
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
        delta_config.DELTA_OBS_DIM
        if delta_policy is not None
        else standard_policy.obs_dim if standard_policy is not None else 0
    )
    print(f"[sim2sim] mode={args.mode}, policy_obs_dim={policy_obs_dim}")
    print(f"[sim2sim] timestep={io.model.opt.timestep:.4f}s, control_dt={CONTROL_DT:.4f}s")
    if gamepad is None:
        print(f"[sim2sim] fixed command={command.tolist()}")
    else:
        print(f"[sim2sim] gamepad={gamepad.name} ({gamepad.path})")
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

                if wtw_builder is not None:
                    command = wtw_command(command)
                if args.mode == "wtw_delta_residual":
                    if not isinstance(wtw_builder, WtwDeltaObservationBuilder):
                        raise RuntimeError("DELTA mode requires a DELTA observation builder")
                    if delta_policy is None:
                        raise RuntimeError("DELTA mode requires a DELTA policy")
                    wtw_proprio, delta_proprio = wtw_builder.build_inputs(
                        io, command, delta_policy.last_action
                    )
                    raw_action = delta_policy.step(
                        wtw_proprio,
                        delta_proprio,
                        build_delta_map(io),
                    )
                elif wtw_builder is None:
                    if standard_policy is None:
                        raise RuntimeError("Baseline mode requires a policy")
                    obs = build_observation(
                        io, command, standard_policy.last_action
                    )
                    raw_action = standard_policy.step(obs)
                else:
                    if standard_policy is None:
                        raise RuntimeError("WTW mode requires a policy")
                    obs = wtw_builder.build(
                        io, command, standard_policy.last_action
                    )
                    raw_action = standard_policy.step(obs)
                target_q = action_to_target(
                    raw_action, io.default_q, ACTION_SCALE
                )
                io.send_target(target_q)
                io.step()
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
