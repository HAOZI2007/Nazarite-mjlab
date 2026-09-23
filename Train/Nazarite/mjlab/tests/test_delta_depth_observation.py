from types import SimpleNamespace

import pytest
import torch
from nazarite.delta.d435 import DELTA_D435_INTRINSICS
from nazarite.mdp.delta_observations import delta_depth_image

from mjlab.sensor import CameraSensorCfg


def test_d435_intrinsics_are_scaled_to_training_resolution() -> None:
  intrinsics = DELTA_D435_INTRINSICS
  assert (intrinsics.width, intrinsics.height) == (64, 36)
  assert intrinsics.fx == pytest.approx(31.7385046617)
  assert intrinsics.fy == pytest.approx(31.5401390076)
  assert intrinsics.cx == pytest.approx(32.3089484449)
  assert intrinsics.cy == pytest.approx(17.8522476196)


def test_delta_depth_uses_camera_pixel_intrinsics() -> None:
  cfg = CameraSensorCfg(
    name="delta_depth_camera",
    width=4,
    height=2,
    focal_length_px=(2.0, 2.0),
    principal_point_px=(1.0, 0.5),
    data_types=("depth",),
  )
  sensor = SimpleNamespace(
    cfg=cfg,
    data=SimpleNamespace(depth=torch.ones(1, 2, 4, 1)),
  )
  env = SimpleNamespace(scene={"delta_depth_camera": sensor})
  result = delta_depth_image(
    env,
    map_height=2,
    map_width=4,
    camera_pos_b=(0.0, 0.0, 0.0),
    camera_pitch_deg=0.0,
  ).reshape(1, 2, 4, 3)

  # Pixel (u=1,v=0) lies on the calibrated horizontal principal point.
  assert result[0, 0, 1, 0].item() == pytest.approx(1.0 / 5.0)
  assert result[0, 0, 1, 1].item() == pytest.approx(0.0)
  assert result[0, 0, 1, 2].item() == pytest.approx(0.25 / 0.8)


def test_delta_depth_bev_returns_regular_map_with_validity_channel() -> None:
  cfg = CameraSensorCfg(
    name="delta_depth_camera",
    width=8,
    height=8,
    focal_length_px=(4.0, 4.0),
    principal_point_px=(3.5, 3.5),
    data_types=("depth",),
  )
  sensor = SimpleNamespace(
    cfg=cfg,
    data=SimpleNamespace(depth=torch.full((1, 8, 8, 1), 1.0)),
  )
  env = SimpleNamespace(scene={"delta_depth_camera": sensor})
  result = delta_depth_image(
    env,
    map_height=4,
    map_width=6,
    camera_pos_b=(0.0, 0.0, 0.0),
    camera_pitch_deg=0.0,
    project_to_bev=True,
    map_channels=4,
  ).reshape(1, 4, 6, 4)

  assert result.shape == (1, 4, 6, 4)
  assert torch.isfinite(result).all()
  assert torch.all((result[..., 3] >= 0.0) & (result[..., 3] <= 1.0))
  assert result[..., 3].sum() > 0
