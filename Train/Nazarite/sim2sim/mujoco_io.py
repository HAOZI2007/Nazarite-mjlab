from __future__ import annotations

from collections.abc import Callable

import mujoco
import numpy as np

try:  # Supports both ``python -m sim2sim.main`` and direct execution.
    from .config import (
        ACTION_SCALE,
        CONTROL_DT,
        DAMPING_CALF,
        DAMPING_HIP,
        DECIMATION,
        DEFAULT_Q,
        GO2_XML,
        JOINT_NAMES,
        KD_SCALE,
        KP_SCALE,
        PHYSICS_DT,
        STIFFNESS_CALF,
        STIFFNESS_HIP,
    )
    from .scene import add_simple_grid_scene
except ImportError:  # pragma: no cover - direct script fallback
    from config import (
        ACTION_SCALE,
        CONTROL_DT,
        DAMPING_CALF,
        DAMPING_HIP,
        DECIMATION,
        DEFAULT_Q,
        GO2_XML,
        JOINT_NAMES,
        KD_SCALE,
        KP_SCALE,
        PHYSICS_DT,
        STIFFNESS_CALF,
        STIFFNESS_HIP,
    )
    from scene import add_simple_grid_scene

class MuJoCoIO:
    def __init__(
        self,
        *,
        hip_effort: float = 45.0,
        calf_effort: float = 45.0,
        scene_builder: Callable[[mujoco.MjSpec], None] = add_simple_grid_scene,
    ):
        if not GO2_XML.is_file():
            raise FileNotFoundError(f"Go2 XML not found: {GO2_XML}")
        if hip_effort <= 0.0 or calf_effort <= 0.0:
            raise ValueError("Actuator effort limits must be positive")

        spec = mujoco.MjSpec.from_file(str(GO2_XML))

        # 删除 XML 中原来的 motor actuator
        for actuator in list(spec.actuators):
            spec.delete(actuator)

        # 创建 Nazarite 对应的位置执行器
        for name in JOINT_NAMES:
            actuator = spec.add_actuator(
                name=f"{name}_position",
                target=name,
            )
            actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
            actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
            actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE

            if "calf" in name:
                kp = STIFFNESS_CALF * KP_SCALE
                kd = DAMPING_CALF * KD_SCALE
                effort = calf_effort
            else:
                kp = STIFFNESS_HIP * KP_SCALE
                kd = DAMPING_HIP * KD_SCALE
                effort = hip_effort

            actuator.gainprm[0] = kp
            actuator.biasprm[1] = -kp
            actuator.biasprm[2] = -kd
            actuator.forcelimited = True
            actuator.forcerange[:] = [-effort, effort]

        # The standalone robot XML has no world. Visual grid sites do not
        # affect contacts; a policy may request extra collidable obstacles.
        scene_builder(spec)

        self.model = spec.compile()
        self.model.opt.timestep = PHYSICS_DT
        self.data = mujoco.MjData(self.model)
        self.control_dt = CONTROL_DT
        self.decimation = DECIMATION
        self.default_q = DEFAULT_Q.copy()
        self.action_scale = ACTION_SCALE.copy()

        self.qpos_ids = np.array([
            self.model.jnt_qposadr[
                mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    name,
                )
            ]
            for name in JOINT_NAMES
        ])

        self.qvel_ids = np.array([
            self.model.jnt_dofadr[
                mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    name,
                )
            ]
            for name in JOINT_NAMES
        ])

        self.ctrl_ids = np.array([
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                f"{name}_position",
            )
            for name in JOINT_NAMES
        ])

        if np.any(self.qpos_ids < 0) or np.any(self.qvel_ids < 0):
            raise RuntimeError("Failed to resolve one or more Go2 joint addresses")
        if np.any(self.ctrl_ids < 0):
            raise RuntimeError("Failed to resolve one or more Go2 actuator IDs")

        self.sensor_ids = {}
        for sensor_name in ("imu_ang_vel", "imu_quat"):
            sensor_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name
            )
            if sensor_id >= 0:
                self.sensor_ids[sensor_name] = sensor_id

    def reset(self):
        self.data.qpos[:7] = [0.0, 0.0, 0.32, 1.0, 0.0, 0.0, 0.0]
        self.data.qpos[self.qpos_ids] = DEFAULT_Q
        self.data.qvel[:] = 0.0
        self.data.ctrl[self.ctrl_ids] = self.default_q
        mujoco.mj_forward(self.model, self.data)

    def get_joint_pos(self):
        return self.data.qpos[self.qpos_ids].copy()

    def get_joint_vel(self):
        return self.data.qvel[self.qvel_ids].copy()

    def send_target(self, target_q):
        target_q = np.asarray(target_q, dtype=np.float32)
        if target_q.shape != (len(JOINT_NAMES),):
            raise ValueError(f"target_q must have shape {(len(JOINT_NAMES),)}, got {target_q.shape}")
        self.data.ctrl[self.ctrl_ids] = target_q

    def get_sensor(self, name: str) -> np.ndarray:
        """Return a named MuJoCo sensor without relying on sensor ordering."""
        if name not in self.sensor_ids:
            raise KeyError(f"Sensor {name!r} is not present in {GO2_XML}")
        sensor_id = self.sensor_ids[name]
        start = int(self.model.sensor_adr[sensor_id])
        dim = int(self.model.sensor_dim[sensor_id])
        return self.data.sensordata[start : start + dim].copy()

    def step(self):
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)
