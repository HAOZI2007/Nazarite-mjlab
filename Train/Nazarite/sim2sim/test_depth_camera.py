"""Unit tests for the WTW + DELTA sim2sim camera diagnostic path."""

import mujoco
import numpy as np

from sim2sim.mujoco_io import MuJoCoIO
from sim2sim.scene import add_simple_grid_scene
from sim2sim.wtw_delta_residual.depth_camera import (
    BevConfig,
    DepthCameraConfig,
    depth_to_bev,
)


def test_camera_can_be_attached_to_go2() -> None:
    camera = DepthCameraConfig()

    def scene_builder(spec: mujoco.MjSpec) -> None:
        add_simple_grid_scene(spec)
        camera.install(spec)

    io = MuJoCoIO(scene_builder=scene_builder)
    assert io.model.ncam == 1
    assert mujoco.mj_id2name(io.model, mujoco.mjtObj.mjOBJ_CAMERA, 0) == camera.name
    np.testing.assert_array_equal(io.model.cam_resolution[0], [camera.width, camera.height])


def test_center_depth_hit_lands_at_center_of_bev() -> None:
    camera = DepthCameraConfig(
        width=9,
        height=5,
        fx_px=4.0,
        fy_px=4.0,
        cx_px=4.0,
        cy_px=2.0,
        pos_b=(0.0, 0.0, 0.0),
        pitch_deg=0.0,
        roll_deg=0.0,
    )
    bev = BevConfig(height=4, width=4, x_range_m=(0.0, 2.0), y_range_m=(-1.0, 1.0), fill_kernel_size=1)
    depth = np.zeros((camera.height, camera.width), dtype=np.float32)
    depth[2, 4] = 1.0
    observation = depth_to_bev(depth, camera, bev)
    assert observation.support[2, 2] == 1.0
    assert observation.bev_z[2, 2] == 0.0
    assert float(observation.support.sum()) == 1.0


def test_roll_and_projection_keep_finite_outputs() -> None:
    camera = DepthCameraConfig(roll_deg=90.0)
    depth = np.full((camera.height, camera.width), 1.0, dtype=np.float32)
    observation = depth_to_bev(depth, camera, BevConfig())
    assert np.isfinite(observation.bev_z).all()
    assert np.isfinite(observation.confidence).all()
