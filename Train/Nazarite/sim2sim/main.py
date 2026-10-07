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
from .him import config as him_config
from .him.observation import HIMObservationBuilder
from .him.policy import HIMPolicy
from .him.scene import add_him_scene
from .math_utils import action_to_target
from .mujoco_io import MuJoCoIO
from .policy_runner import PolicyRunner
from .scene import add_simple_grid_scene, add_stairs_and_slopes_scene
from .wtw import config as wtw_config
from .wtw.behavior import WTWBehaviorController
from .wtw.command import restrict_command as wtw_command
from .wtw.observation import WTWObservationBuilder
from .wtw_delta_direct.config import (
    POLICY as DIRECT_POLICY,
)
from .wtw_delta_direct.config import (
    build_configs as build_direct_configs,
)
from .wtw_delta_direct.config import (
    build_policy_map,
)
from .wtw_delta_direct.observation import DirectObservationBuilder
from .wtw_delta_direct.policy import WtwDeltaDirectPolicy
from .wtw_delta_direct.terrain import add_stepping_stones_scene as add_direct_scene
from .wtw_delta_residual import config as delta_config
from .wtw_delta_residual.depth_camera import (
    BevConfig,
    DepthCameraConfig,
    DepthCameraDiagnostics,
)
from .wtw_delta_residual.observation import WtwDeltaObservationBuilder
from .wtw_delta_residual.policy import WtwDeltaResidualPolicy
from .wtw_delta_residual.terrain import (
    add_stepping_stones_scene,
    build_delta_map,
)


def default_policy_path(mode: str = "baseline") -> Path:
    if mode == "him":
        policy_path = him_config.POLICY
    elif mode == "wtw":
        policy_path = wtw_config.POLICY
    elif mode == "wtw_delta_residual":
        policy_path = delta_config.POLICY
    elif mode == "wtw_delta_direct":
        policy_path = DIRECT_POLICY
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
        choices=("baseline", "him", "wtw", "wtw_delta_residual", "wtw_delta_direct"),
        default="baseline",
    )
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--body-height", type=float, default=0.0, help="HIM body-height offset from 0.32 m (m)")
    parser.add_argument("--stance-width", type=float, default=0.235, help="HIM stance-width condition (m)")
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
    parser.add_argument(
        "--depth-debug",
        "--depth-visualize",
        dest="depth_debug",
        action="store_true",
        help="render the D435 depth camera and live depth/BEV plots",
    )
    parser.add_argument(
        "--depth-update-every",
        type=int,
        default=4,
        help="capture/update the diagnostic panel every N control steps",
    )
    parser.add_argument(
        "--depth-external-window",
        action="store_true",
        help="use the legacy external matplotlib window instead of MuJoCo image overlays",
    )
    parser.add_argument("--depth-width", type=int, default=None, help="diagnostic depth width")
    parser.add_argument("--depth-height", type=int, default=None, help="diagnostic depth height")
    parser.add_argument("--depth-pos-x", type=float, default=None, help="camera x position in base frame (m)")
    parser.add_argument("--depth-pos-y", type=float, default=None, help="camera y position in base frame (m)")
    parser.add_argument("--depth-pos-z", type=float, default=None, help="camera z position in base frame (m)")
    parser.add_argument("--depth-pitch", type=float, default=None, help="camera pitch down angle (deg)")
    parser.add_argument("--depth-roll", type=float, default=None, help="camera optical-axis roll (deg)")
    parser.add_argument("--depth-fx", type=float, default=None, help="depth focal length fx (px)")
    parser.add_argument("--depth-fy", type=float, default=None, help="depth focal length fy (px)")
    parser.add_argument("--depth-cx", type=float, default=None, help="depth principal point cx (px)")
    parser.add_argument("--depth-cy", type=float, default=None, help="depth principal point cy (px)")
    parser.add_argument("--depth-max", type=float, default=None, help="maximum displayed/projected depth (m)")
    parser.add_argument("--bev-height", type=int, default=None, help="BEV rows")
    parser.add_argument("--bev-width", type=int, default=None, help="BEV columns")
    parser.add_argument("--bev-x-min", type=float, default=None, help="BEV forward range minimum (m)")
    parser.add_argument("--bev-x-max", type=float, default=None, help="BEV forward range maximum (m)")
    parser.add_argument("--bev-y-min", type=float, default=None, help="BEV lateral range minimum (m)")
    parser.add_argument("--bev-y-max", type=float, default=None, help="BEV lateral range maximum (m)")
    parser.add_argument("--bev-z-scale", type=float, default=None, help="BEV height normalization (m)")
    parser.add_argument("--bev-fill-kernel", type=int, default=None, help="odd BEV hole-fill kernel")
    return parser.parse_args()


def _depth_debug_configs(args: argparse.Namespace) -> tuple[DepthCameraConfig, BevConfig]:
    """Build debug-only camera and BEV configs from defaults plus CLI overrides."""
    defaults = DepthCameraConfig()
    defaults_bev = BevConfig()
    width = defaults.width if args.depth_width is None else args.depth_width
    height = defaults.height if args.depth_height is None else args.depth_height
    # D435 intrinsics are scaled with the image when the user changes only the
    # resolution. Explicit --depth-fx/... values always take precedence.
    scale_x = width / defaults.width
    scale_y = height / defaults.height
    roll = defaults.roll_deg if args.depth_roll is None else args.depth_roll
    portrait = height > width and abs(abs(roll) % 180.0 - 90.0) < 1.0e-6
    if portrait:
        default_fx = defaults.fy_px * width / defaults.height
        default_fy = defaults.fx_px * height / defaults.width
        default_cx = defaults.cy_px * width / defaults.height
        default_cy = defaults.cx_px * height / defaults.width
    else:
        default_fx = defaults.fx_px * scale_x
        default_fy = defaults.fy_px * scale_y
        default_cx = defaults.cx_px * scale_x
        default_cy = defaults.cy_px * scale_y
    camera = DepthCameraConfig(
        width=width,
        height=height,
        pos_b=(
            defaults.pos_b[0] if args.depth_pos_x is None else args.depth_pos_x,
            defaults.pos_b[1] if args.depth_pos_y is None else args.depth_pos_y,
            defaults.pos_b[2] if args.depth_pos_z is None else args.depth_pos_z,
        ),
        pitch_deg=defaults.pitch_deg if args.depth_pitch is None else args.depth_pitch,
        roll_deg=roll,
        fx_px=(default_fx if args.depth_fx is None else args.depth_fx),
        fy_px=(default_fy if args.depth_fy is None else args.depth_fy),
        cx_px=(default_cx if args.depth_cx is None else args.depth_cx),
        cy_px=(default_cy if args.depth_cy is None else args.depth_cy),
        max_depth_m=(defaults.max_depth_m if args.depth_max is None else args.depth_max),
    )
    x_range = (
        defaults_bev.x_range_m[0] if args.bev_x_min is None else args.bev_x_min,
        defaults_bev.x_range_m[1] if args.bev_x_max is None else args.bev_x_max,
    )
    y_range = (
        defaults_bev.y_range_m[0] if args.bev_y_min is None else args.bev_y_min,
        defaults_bev.y_range_m[1] if args.bev_y_max is None else args.bev_y_max,
    )
    bev = BevConfig(
        height=defaults_bev.height if args.bev_height is None else args.bev_height,
        width=defaults_bev.width if args.bev_width is None else args.bev_width,
        x_range_m=x_range,
        y_range_m=y_range,
        z_scale_m=defaults_bev.z_scale_m if args.bev_z_scale is None else args.bev_z_scale,
        fill_kernel_size=(
            defaults_bev.fill_kernel_size
            if args.bev_fill_kernel is None
            else args.bev_fill_kernel
        ),
    )
    return camera, bev


def main() -> None:
    args = parse_args()
    if args.list_gamepads:
        gamepads = list_gamepads()
        print(format_gamepads(gamepads) if gamepads else "No compatible gamepad detected.")
        return

    policy_path = args.policy or default_policy_path(args.mode)
    command = np.array([args.vx, args.vy, args.yaw], dtype=np.float32)
    behavior = np.array([args.body_height, args.stance_width], dtype=np.float32)
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

    depth_camera: DepthCameraConfig | None = None
    depth_bev: BevConfig | None = None
    if args.depth_debug or args.mode == "wtw_delta_direct":
        if args.depth_update_every <= 0:
            raise ValueError("--depth-update-every must be positive")
        if args.mode == "wtw_delta_direct":
            depth_camera, depth_bev = build_direct_configs(args)
        else:
            depth_camera, depth_bev = _depth_debug_configs(args)

    def scene_builder(spec: mujoco.MjSpec) -> None:
        if args.mode == "wtw":
            add_stairs_and_slopes_scene(spec)
        elif args.mode == "him":
            add_him_scene(spec)
        elif args.mode == "wtw_delta_residual":
            add_stepping_stones_scene(spec)
        elif args.mode == "wtw_delta_direct":
            add_direct_scene(spec)
        else:
            # The baseline uses MuJoCoIO's default flat/grid scene.  Keeping
            # the same scene here avoids changing baseline behaviour when the
            # diagnostic flag is enabled.
            add_simple_grid_scene(spec)
        if depth_camera is not None:
            depth_camera.install(spec)

    if args.mode == "him":
        io = MuJoCoIO(
            hip_effort=him_config.HIP_EFFORT,
            calf_effort=him_config.CALF_EFFORT,
            scene_builder=scene_builder,
        )
    elif args.mode in ("wtw", "wtw_delta_residual", "wtw_delta_direct"):
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
    delta_policy: WtwDeltaResidualPolicy | None = None
    direct_policy: WtwDeltaDirectPolicy | None = None
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
    elif args.mode == "wtw_delta_residual":
        delta_policy = WtwDeltaResidualPolicy(policy_path)
    elif args.mode == "wtw_delta_direct":
        direct_policy = WtwDeltaDirectPolicy(policy_path)
    else:
        expected_obs_dim = baseline_config.OBS_DIM
        expected_observation_names = baseline_config.OBSERVATION_NAMES
        standard_policy = PolicyRunner(
            policy_path,
            expected_obs_dim=expected_obs_dim,
            expected_observation_names=expected_observation_names,
        )
    wtw_builder: WTWObservationBuilder | None
    if args.mode in ("wtw_delta_residual", "wtw_delta_direct"):
        if args.mode == "wtw_delta_direct":
            if direct_policy is None:
                raise RuntimeError("Direct observation builder requires Direct policy")
            wtw_builder = DirectObservationBuilder(direct_policy.delta_history_length)
        else:
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
    if him_policy is not None:
        him_policy.reset()
    if delta_policy is not None:
        delta_policy.reset()
    if direct_policy is not None:
        direct_policy.reset()
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
        delta_config.DELTA_OBS_DIM
        if delta_policy is not None
        else 498 + direct_policy.delta_proprio_dim + 16 * 26 * 5
        if direct_policy is not None
        else him_policy.obs_dim if him_policy is not None
        else standard_policy.obs_dim if standard_policy is not None else 0
    )
    print(f"[sim2sim] mode={args.mode}, policy_obs_dim={policy_obs_dim}")
    print(f"[sim2sim] timestep={io.model.opt.timestep:.4f}s, control_dt={CONTROL_DT:.4f}s")
    if depth_camera is not None and depth_bev is not None:
        print(
            "[sim2sim] Direct camera input enabled: "
            if direct_policy is not None
            else "[sim2sim] depth diagnostic enabled (WTW action is unchanged): "
        )
        print(
            f"{depth_camera.width}x{depth_camera.height}, "
            f"pos={depth_camera.pos_b}, pitch={depth_camera.pitch_deg:.1f} deg, "
            f"roll={depth_camera.roll_deg:.1f} deg"
        )
        print(
            "[sim2sim] BEV: "
            f"{depth_bev.height}x{depth_bev.width}, "
            f"x={depth_bev.x_range_m}, y={depth_bev.y_range_m}, "
            f"z_scale={depth_bev.z_scale_m:.2f} m"
        )
    if gamepad is None:
        print(f"[sim2sim] fixed command={command.tolist()}")
    else:
        print(f"[sim2sim] gamepad={gamepad.name} ({gamepad.path})")
        print(f"[sim2sim] gamepad axes: {gamepad.axis_mapping}")
        print("[sim2sim] left stick: vx/vy, right stick X: yaw")

    next_tick = time.perf_counter()
    depth_diagnostics: DepthCameraDiagnostics | None = None
    depth_debug_step = 0

    def viewer_key_callback(key: int) -> None:
        if depth_diagnostics is not None and not args.depth_external_window:
            depth_diagnostics.handle_native_key(key)

    try:
        with mujoco.viewer.launch_passive(
            io.model, io.data, key_callback=viewer_key_callback
        ) as viewer:
            if depth_camera is not None and depth_bev is not None:
                # Renderer creation must happen after the viewer creates an
                # OpenGL context. It shares that context for offscreen depth.
                depth_diagnostics = DepthCameraDiagnostics(
                    io.model,
                    depth_camera,
                    depth_bev,
                    visualize=args.depth_external_window,
                )
                if args.depth_debug and not args.depth_external_window:
                    depth_diagnostics.attach_native_viewer(viewer)
            if not args.no_camera_follow:
                follow_robot(viewer, io.data.qpos[:3], camera_config)
            while viewer.is_running():
                if gamepad is not None:
                    command = gamepad.poll()
                    if wtw_behavior_controller is not None:
                        wtw_behavior_controller.update()

                direct_observation = None
                if args.mode == "wtw_delta_direct":
                    if direct_policy is None or depth_diagnostics is None:
                        raise RuntimeError("Direct mode requires camera and policy")
                    direct_observation = depth_diagnostics.capture(io.data)

                if him_builder is not None:
                    if him_policy is None:
                        raise RuntimeError("HIM mode requires a policy")
                    command = np.clip(command, -1.0, 1.0)
                    obs = him_builder.build(io, command, him_policy.last_action, behavior)
                    raw_action = him_policy.step(obs)
                elif wtw_builder is not None:
                    command = wtw_command(command)
                    if args.mode == "wtw_delta_residual":
                        if not isinstance(wtw_builder, WtwDeltaObservationBuilder):
                            raise RuntimeError("DELTA mode requires a DELTA observation builder")
                        if delta_policy is None:
                            raise RuntimeError("DELTA mode requires a DELTA policy")
                        wtw_proprio, delta_proprio = wtw_builder.build_inputs(
                            io, command, delta_policy.last_action
                        )
                        raw_action = delta_policy.step(wtw_proprio, delta_proprio, build_delta_map(io))
                    elif args.mode == "wtw_delta_direct":
                        if not isinstance(wtw_builder, WtwDeltaObservationBuilder):
                            raise RuntimeError("Direct mode requires a DELTA observation builder")
                        if direct_policy is None or direct_observation is None or depth_bev is None:
                            raise RuntimeError("Direct mode camera observation is unavailable")
                        wtw_proprio, delta_proprio = wtw_builder.build_inputs(
                            io, command, direct_policy.last_action
                        )
                        raw_action = direct_policy.step(
                            wtw_proprio, delta_proprio, build_policy_map(direct_observation, depth_bev)
                        )
                    else:
                        if standard_policy is None:
                            raise RuntimeError("WTW mode requires a policy")
                        obs = wtw_builder.build(io, command, standard_policy.last_action)
                        raw_action = standard_policy.step(obs)
                else:
                    if standard_policy is None:
                        raise RuntimeError("Baseline mode requires a policy")
                    obs = build_observation(io, command, standard_policy.last_action)
                    raw_action = standard_policy.step(obs)
                target_q = action_to_target(
                    raw_action, io.default_q, ACTION_SCALE
                )
                io.send_target(target_q)
                io.step()
                if him_builder is not None:
                    him_builder.advance(CONTROL_DT)
                if wtw_builder is not None:
                    wtw_builder.advance_phase(command, CONTROL_DT)
                if not args.no_camera_follow:
                    follow_robot(viewer, io.data.qpos[:3], camera_config)
                if depth_diagnostics is not None and args.depth_debug:
                    depth_debug_step += 1
                    if depth_debug_step % args.depth_update_every == 0:
                        if args.mode == "wtw_delta_direct" and direct_observation is not None:
                            depth_observation = direct_observation
                        else:
                            depth_observation = depth_diagnostics.capture(io.data)
                        if args.depth_external_window:
                            depth_diagnostics.update_plot(depth_observation)
                        else:
                            depth_diagnostics.update_native(depth_observation)
                viewer.sync()

                next_tick += CONTROL_DT
                sleep_time = next_tick - time.perf_counter()
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
                elif sleep_time < -CONTROL_DT:
                    next_tick = time.perf_counter()
    finally:
        if depth_diagnostics is not None:
            depth_diagnostics.close()
        if gamepad is not None:
            gamepad.close()


if __name__ == "__main__":
    main()
