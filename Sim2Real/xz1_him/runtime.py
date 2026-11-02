"""NumPy/ORT-only policy pipeline; all state is in the training body frame."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import onnxruntime as ort

CONFIG = Path(__file__).with_name("config.json")
TERMS = "base_ang_vel,projected_gravity,command,phase,joint_pos,joint_vel,actions"


def vector(value, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite ({size},) vector")
    return result


def load_config(path: Path = CONFIG) -> dict:
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    config["policy"] = str((path.parent / config["policy"]).resolve())
    for name in (
        "default_q",
        "action_offset",
        "action_scale",
        "kp",
        "kd",
        "joint_lower",
        "joint_upper",
    ):
        config[name] = vector(config[name], 12, name)
    if len(config["joint_names"]) != 12 or len(set(config["joint_names"])) != 12:
        raise ValueError("Expected 12 distinct joint names")
    for name in ("control_dt", "phase_period"):
        if not np.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if np.any(config["joint_lower"] >= config["joint_upper"]):
        raise ValueError("Invalid joint limits")
    return config


@dataclass
class RobotState:
    """Radians, rad/s, body gyro xyz, body-to-world quaternion wxyz."""

    joint_pos: np.ndarray
    joint_vel: np.ndarray
    angular_velocity: np.ndarray
    quaternion_wxyz: np.ndarray


@dataclass
class JointCommand:
    position: np.ndarray
    kp: np.ndarray
    kd: np.ndarray


class RobotInterface(Protocol):
    """Adapter must map motor order/sign/zero and IMU axes to this contract.

    Implement fresh-state timeouts, arming, emergency stop, actuator limits,
    and a lower-level command watchdog in the actual hardware adapter.
    """

    def read_state(self) -> RobotState: ...

    def send_command(self, command: JointCommand) -> None: ...

    def stop(self) -> None: ...


class ObservationBuilder:
    def __init__(self, config: dict):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.history = np.zeros((6, 47), dtype=np.float32)
        self.phase = 0.0
        self.initialized = False

    def build(self, state: RobotState, command, last_action) -> np.ndarray:
        q = vector(state.joint_pos, 12, "joint_pos")
        dq = vector(state.joint_vel, 12, "joint_vel")
        gyro = vector(state.angular_velocity, 3, "angular_velocity")
        quat = vector(state.quaternion_wxyz, 4, "quaternion_wxyz")
        norm = np.linalg.norm(quat)
        if not 0.9 <= norm <= 1.1:
            raise ValueError("Quaternion must be approximately unit length")
        w, x, y, z = quat / norm
        gravity = np.array(
            [2 * (w * y - x * z), -2 * (y * z + w * x), 2 * (x * x + y * y) - 1],
            dtype=np.float32,
        )
        command = vector(command, 3, "command")
        if np.any(np.abs(command) > 1):
            raise ValueError("Test command bounds are +/-1 m/s and rad/s")
        phase = np.zeros(2, dtype=np.float32)
        if np.linalg.norm(command) >= 0.1:
            angle = 2 * np.pi * self.phase
            phase[:] = np.sin(angle), np.cos(angle)
        frame = np.concatenate(
            (
                gyro,
                gravity,
                command,
                phase,
                q - self.config["default_q"],
                dq,
                vector(last_action, 12, "last_action"),
            )
        )
        frame = np.clip(frame, -100, 100).astype(np.float32)
        frame[35:47] = np.clip(frame[35:47], -10, 10)
        if not self.initialized:
            self.history[:] = frame
            self.initialized = True
        else:
            self.history[1:] = self.history[:-1].copy()
            self.history[0] = frame
        return self.history.reshape(282).copy()

    def advance(self) -> None:
        self.phase = (
            self.phase + self.config["control_dt"] / self.config["phase_period"]
        ) % 1.0


class ONNXPolicy:
    def __init__(self, path: Path, config: dict):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        for nodes, width in ((inputs, 282), (outputs, 12)):
            if len(nodes) != 1:
                raise ValueError("Expected one ONNX input and one output")
            node = nodes[0]
            if (
                node.type != "tensor(float)"
                or len(node.shape) != 2
                or node.shape[1] != width
                or isinstance(node.shape[0], int)
                and node.shape[0] != 1
            ):
                raise ValueError(f"Unsupported ONNX tensor: {node}")
        metadata = self.session.get_modelmeta().custom_metadata_map
        expected = {
            "him_version": "1",
            "him_frame_size": "47",
            "him_history_size": "6",
            "him_export_history_order": "frame_major_current_first",
            "joint_names": ",".join(config["joint_names"]),
            "observation_names": TERMS,
            "him_last_action": "clipped_policy_action_before_joint_scale_and_offset",
        }
        for name, value in expected.items():
            if metadata.get(name) != value:
                raise ValueError(
                    f"ONNX contract mismatch: {name}={metadata.get(name)!r}"
                )
        for name, key in (
            ("action_scale", "action_scale"),
            ("joint_stiffness", "kp"),
            ("joint_damping", "kd"),
            ("default_joint_pos", "default_q"),
        ):
            values = [float(v) for v in metadata.get(name, "").split(",") if v]
            exported = vector(values, 12, name)
            if not np.allclose(exported, config[key], atol=5.1e-4, rtol=0):
                raise ValueError(f"ONNX metadata disagrees with config: {name}")
        self.input_name, self.output_name = inputs[0].name, outputs[0].name

    def step(self, observation) -> np.ndarray:
        obs = vector(observation, 282, "observation")
        result = np.asarray(
            self.session.run([self.output_name], {self.input_name: obs[None]})[0]
        )
        if result.shape != (1, 12):
            raise ValueError(f"Unexpected action shape {result.shape}")
        return np.clip(vector(result[0], 12, "action"), -10, 10)


class Controller:
    """Compute targets only. Hardware communication belongs to the adapter."""

    def __init__(self, policy: ONNXPolicy, config: dict):
        self.policy, self.config = policy, config
        self.builder = ObservationBuilder(config)
        self.last_action = np.zeros(12, dtype=np.float32)

    def reset(self) -> None:
        self.builder.reset()
        self.last_action.fill(0)

    def step(self, state: RobotState, command) -> tuple[JointCommand, np.ndarray]:
        obs = self.builder.build(state, command, self.last_action)
        action = self.policy.step(obs)
        target = self.config["action_offset"] + self.config["action_scale"] * action
        # Reject rather than silently modify a target and its action feedback.
        if np.any(target < self.config["joint_lower"]) or np.any(
            target > self.config["joint_upper"]
        ):
            raise ValueError("Policy target exceeds XZ1 joint limits; command rejected")
        self.last_action = action.copy()
        self.builder.advance()
        return JointCommand(
            target, self.config["kp"].copy(), self.config["kd"].copy()
        ), obs
