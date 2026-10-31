"""Original XZ1 terrain teacher and its privileged 12x11 height scan."""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import torch
from tensordict import TensorDict

from nazarite.config.train_config.xz1.teacher_env_cfg import SCAN_X, SCAN_Y
from nazarite.rl.xz1.student import load_teacher
from sim2sim.shared.config import CONTROL_DT


class TeacherHeightScan:
    """Yaw-aligned downward rays, with the exact training scan coordinates."""

    def __init__(self, model: mujoco.MjModel, body_name: str = "base_link"):
        self.model = model
        self.body_id = model.body(body_name).id
        x, y = np.meshgrid(SCAN_X, SCAN_Y, indexing="ij")
        self.offsets = np.stack((x.ravel(), y.ravel(), np.ones(132)), axis=-1)
        self.groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        self.direction = np.array([0.0, 0.0, -1.0])
        self.geom_id = np.empty(1, dtype=np.int32)
        self.last_hits = np.full((132, 3), np.nan)

    def read(self, data: mujoco.MjData) -> np.ndarray:
        mujoco.mj_kinematics(self.model, data)
        position = data.xpos[self.body_id]
        w, x, y, z = data.xquat[self.body_id]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        origins = self.offsets @ rotation.T + position
        result = np.ones(132, dtype=np.float32)
        self.last_hits.fill(np.nan)
        for index, origin in enumerate(origins):
            distance = mujoco.mj_ray(
                self.model,
                data,
                origin,
                self.direction,
                self.groups,
                True,
                self.body_id,
                self.geom_id,
            )
            if 0 <= distance <= 5.0:
                hit = origin + distance * self.direction
                self.last_hits[index] = hit
                result[index] = np.clip(position[2] - 0.3 - hit[2], -1, 1)
        return result


class TeacherPolicy:
    def __init__(self, path: Path, device: str = "cpu"):
        self.device = torch.device(device)
        obs = TensorDict(
            {"actor": torch.zeros(1, 6, 47, device=self.device)}, batch_size=[1]
        )
        # Loads the native training actor, including scan encoder and frozen
        # HIM/normalizer. The saved numerical contract is checked by this loader.
        self.actor = load_teacher(obs, path, device)
        self.scan_history_length = self.actor.scan_history_length
        self.scan_update_period = self.actor.scan_update_period
        self.scan_update_steps = round(self.scan_update_period / CONTROL_DT)
        if self.scan_update_steps < 1 or not math.isclose(
            self.scan_update_steps * CONTROL_DT, self.scan_update_period, abs_tol=1e-9
        ):
            raise ValueError("Scan period must be a positive multiple of control dt")
        self.reset()

    def reset(self) -> None:
        self.last_action = np.zeros(12, dtype=np.float32)
        self.scan_history = None
        self.control_step = 0
        self.last_scan_step = None

    @torch.inference_mode()
    def step(self, history: np.ndarray, scan: np.ndarray) -> np.ndarray:
        history = np.asarray(history, dtype=np.float32)
        scan = np.asarray(scan, dtype=np.float32)
        if history.shape != (282,) or not np.isfinite(history).all():
            raise ValueError("Expected finite current-first 6x47 history")
        if scan.shape != (132,) or not np.isfinite(scan).all():
            raise ValueError("Expected 132 finite privileged height values")
        # The sim2sim builder is current-first; the native actor accepts
        # oldest-first observation buffers and reverses them internally.
        frames = history.reshape(6, 47)[::-1].copy()
        if self.last_scan_step is None or (
            self.control_step - self.last_scan_step >= self.scan_update_steps
        ):
            if self.scan_history is None:
                self.scan_history = np.repeat(
                    scan[None], self.scan_history_length, axis=0
                )
            else:
                self.scan_history[:-1] = self.scan_history[1:].copy()
                self.scan_history[-1] = scan
            self.last_scan_step = self.control_step
        assert self.scan_history is not None
        scan_input = (
            self.scan_history[0].copy()
            if self.scan_history_length == 1
            else self.scan_history.copy()
        )
        obs = TensorDict(
            {
                "actor": torch.as_tensor(frames, device=self.device)[None],
                "teacher_scan": torch.as_tensor(scan_input, device=self.device)[None],
            },
            batch_size=[1],
        )
        action = self.actor(obs).clamp(-10, 10).cpu().numpy()[0].copy()
        if not np.isfinite(action).all():
            raise FloatingPointError("Non-finite teacher action")
        self.last_action = action
        self.control_step += 1
        return action
