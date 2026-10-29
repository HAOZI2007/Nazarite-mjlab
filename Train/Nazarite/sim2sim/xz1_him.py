"""XZ1 HIM deployment in the Robocon 2027 arena (CPU MuJoCo)."""

from __future__ import annotations

import argparse
import math
import threading
import time
from collections.abc import Callable
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from .camera import FollowCameraConfig, follow_robot
from .config import CONTROL_DT, DECIMATION, PHYSICS_DT, ROOT
from .gamepad import GamepadController, format_gamepads, list_gamepads
from .him.observation import HIMObservationBuilder
from .him.policy import HIMPolicy
from .mujoco_io import MuJoCoIO

RUN = ROOT / "logs/rsl_rl/xz1_him_complex/2026-10-25_22-04-31"
DEFAULT_POLICY = RUN / "model_31300.pt"
SCENE = Path(__file__).resolve().parent / "assets/Robocon2027.xml"
ROBOT = ROOT / "MJCF-Manager/Robots/XZ1/xmls/xz1.xml"
JOINT_NAMES = [
    f"{leg}_{joint}_joint"
    for leg in ("FL", "FR", "RR", "RL")
    for joint in ("hip", "thigh", "shank")
]
DEFAULT_Q = np.array([0.0, 0.83, -1.7404] * 4, dtype=np.float32)
# Frozen deployment contract from RUN/params/env.yaml: actions.joint_pos.offset.
# DEFAULT_Q remains the reset pose and joint-position observation reference.
# Position targets include the training servo's static load compensation.
ACTION_OFFSET = np.array(
    [
        0.0,
        0.8483,
        -1.5640,  # FL
        0.0,
        0.8481,
        -1.5624,  # FR
        0.0,
        0.8478,
        -1.5427,  # RR
        0.0,
        0.8480,
        -1.5450,  # RL
    ],
    dtype=np.float32,
)


class XZ1IO(MuJoCoIO):
    """Use the trained XZ1 joint order, servos, armature and rest pose."""

    def __init__(
        self,
        scene: Path,
        spawn: tuple[float, ...],
        yaw: float,
        *,
        configure_robot: Callable[[mujoco.MjSpec], None] | None = None,
    ):
        spec = mujoco.MjSpec.from_file(str(scene.expanduser().resolve()))
        robot = mujoco.MjSpec.from_file(str(ROBOT))
        if configure_robot is not None:
            configure_robot(robot)
        for actuator in list(robot.actuators):
            robot.delete(actuator)
        for name in JOINT_NAMES:
            joint = robot.joint(name)
            joint.armature = 0.025392063
            shank = "shank" in name
            actuator = robot.add_actuator(name=f"{name}_position", target=name)
            actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
            actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            actuator.gainprm[0] = 35.0 if shank else 32.0
            actuator.biasprm[1] = -actuator.gainprm[0]
            actuator.biasprm[2] = -1.8 if shank else -1.9
            actuator.forcelimited = True
            actuator.forcerange[:] = [-31.7, 31.7]
        spec.attach(robot, prefix="", frame=spec.worldbody.add_frame())
        # Use the training solver rather than the arena's RK4 integrator.
        spec.option.timestep = PHYSICS_DT
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        spec.option.iterations = 10
        spec.option.ls_iterations = 20
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.default_q = DEFAULT_Q.copy()
        self.action_offset = ACTION_OFFSET.copy()
        self.action_scale = np.full(12, 0.25, dtype=np.float32)
        self.control_dt = CONTROL_DT
        self.decimation = DECIMATION
        ids = np.array([self.model.joint(n).id for n in JOINT_NAMES])
        self.qpos_ids = self.model.jnt_qposadr[ids].copy()
        self.qvel_ids = self.model.jnt_dofadr[ids].copy()
        self.ctrl_ids = np.array(
            [self.model.actuator(f"{n}_position").id for n in JOINT_NAMES]
        )
        self.sensor_ids = {"imu_ang_vel": self.model.sensor("imu_ang_vel").id}
        self.spawn = np.asarray(spawn, dtype=float)
        radians = math.radians(yaw)
        self.quat = [math.cos(radians / 2), 0, 0, math.sin(radians / 2)]
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = self.spawn
        self.data.qpos[3:7] = self.quat
        self.data.qpos[self.qpos_ids] = self.default_q
        self.data.ctrl[self.ctrl_ids] = self.action_offset
        mujoco.mj_forward(self.model, self.data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--scene", type=Path, default=SCENE)
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
        "--duration", type=float, default=0.0, help="sim seconds; 0 runs forever"
    )
    args = parser.parse_args()
    if args.duration < 0 or (args.headless and args.duration <= 0):
        parser.error("--headless requires --duration > 0")
    return args


def main() -> None:
    args = parse_args()
    if args.list_gamepads:
        devices = list_gamepads()
        print(
            format_gamepads(devices) if devices else "No compatible gamepad detected."
        )
        return
    io = XZ1IO(args.scene, args.spawn, args.spawn_yaw)
    policy = HIMPolicy(args.policy, joint_names=JOINT_NAMES)
    builder = HIMObservationBuilder()
    gamepad = None
    viewer = None
    viewer_threads: set[threading.Thread] = set()
    reset_requested = False

    def on_key(key: int) -> None:
        nonlocal reset_requested
        if key == ord("R"):
            reset_requested = True

    print(f"[XZ1 HIM] policy: {args.policy}")
    print(f"[XZ1 HIM] scene: {args.scene}")
    print("[XZ1 HIM] 282 observations, FL/FR/RR/RL, 50 Hz control / 500 Hz physics")
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
            print(f"[XZ1 HIM] gamepad: {gamepad.name} ({gamepad.path})")
            print(f"[XZ1 HIM] {gamepad.axis_mapping}")
            print("[XZ1 HIM] left stick: vx/vy; right X: yaw; Start: reset")
        if not args.headless:
            previous_threads = set(threading.enumerate())
            viewer = mujoco.viewer.launch_passive(
                io.model, io.data, key_callback=on_key
            )
            viewer_threads = set(threading.enumerate()) - previous_threads
            print("[XZ1 HIM] R: reset to spawn")
        camera = FollowCameraConfig()
        deadline = time.perf_counter()
        steps = 0
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
            command = np.clip(command, -1.0, 1.0)
            obs = builder.build(io, command, policy.last_action)
            action = policy.step(obs)
            io.send_target(io.action_offset + io.action_scale * action)
            io.step()
            builder.advance(CONTROL_DT)
            if not np.all(np.isfinite(io.data.qpos)):
                raise FloatingPointError("Non-finite simulation state")
            steps += 1
            if viewer is not None:
                follow_robot(viewer, io.data.qpos[:3], camera)
                viewer.sync()
            if args.duration and steps * CONTROL_DT >= args.duration:
                break
            if viewer is not None:
                deadline += CONTROL_DT
                time.sleep(max(0.0, deadline - time.perf_counter()))
                deadline = max(deadline, time.perf_counter() - CONTROL_DT)
        print(f"[XZ1 HIM] simulated {steps * CONTROL_DT:.2f}s; base={io.data.qpos[:3]}")
    finally:
        if viewer is not None:
            viewer.close()
            # MuJoCo's passive render thread is a daemon. Let it release the
            # X11/GLFW context before Python's atexit glfw.terminate runs.
            for thread in viewer_threads:
                thread.join(timeout=5.0)
        if gamepad is not None:
            gamepad.close()


if __name__ == "__main__":
    main()
