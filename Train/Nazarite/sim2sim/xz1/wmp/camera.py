"""Native MuJoCo pinhole rays matching the training depth camera.

Like mujoco_warp rendering, depth is optical-axis Z, not Euclidean range.
Native ray queries keep headless inference independent of OpenGL/display setup.
"""

from __future__ import annotations

import mujoco
import numpy as np

CAMERA_NAME = "xz1_wmp_depth"
IMAGE_SIZE = 64


def install_camera(spec: mujoco.MjSpec, camera: dict) -> None:
    body = spec.body("base_link")
    if body is None:
        raise ValueError("XZ1 base_link is missing")
    sensor = body.add_camera(name=CAMERA_NAME)
    sensor.pos[:] = camera["pos"]
    sensor.quat[:] = camera["quat"]
    sensor.fovy = camera["fovy"]
    sensor.resolution[:] = [IMAGE_SIZE, IMAGE_SIZE]


class WMPDepthCamera:
    def __init__(
        self, model: mujoco.MjModel, camera: dict, name: str = CAMERA_NAME
    ) -> None:
        self.model = model
        self.camera_id = model.camera(name).id
        self.far_clip = float(camera["far_clip"])
        rows, cols = np.indices((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float64)
        focal = IMAGE_SIZE / (2 * np.tan(np.deg2rad(camera["fovy"]) / 2))
        local = np.stack(
            ((cols + 0.5 - 32) / focal, (31.5 - rows) / focal, -np.ones_like(rows)),
            axis=-1,
        ).reshape(-1, 3)
        self.rays = local / np.linalg.norm(local, axis=1, keepdims=True)
        self.groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        self.geom_ids = np.empty(IMAGE_SIZE**2, dtype=np.int32)
        self.distances = np.empty(IMAGE_SIZE**2, dtype=np.float64)

    def read(self, data: mujoco.MjData) -> np.ndarray:
        # mj_step leaves derived poses at its last evaluation point. Refresh
        # them at current qpos so camera acquisition precedes this policy step.
        mujoco.mj_kinematics(self.model, data)
        mujoco.mj_camlight(self.model, data)
        rotation = data.cam_xmat[self.camera_id].reshape(3, 3)
        directions = np.ascontiguousarray(self.rays @ rotation.T).reshape(-1)
        mujoco.mj_multiRay(
            self.model,
            data,
            data.cam_xpos[self.camera_id],
            directions,
            self.groups,
            True,
            # Match the training renderer: visibility is filtered by group,
            # not by the camera's parent body. XZ1 uses groups 1/3, terrain 0.
            -1,
            self.geom_ids,
            self.distances,
            None,
            IMAGE_SIZE**2,
            # Z-clipped corner pixels can have Euclidean range > far_clip.
            self.far_clip / float((-self.rays[:, 2]).min()) + 1e-3,
        )
        depth = self.distances * (-self.rays[:, 2])
        depth = np.where(
            (self.distances > 0) & np.isfinite(depth),
            np.minimum(depth, self.far_clip),
            self.far_clip,
        )
        return depth.reshape(IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
