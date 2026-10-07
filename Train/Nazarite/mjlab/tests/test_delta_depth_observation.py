from types import SimpleNamespace

import pytest
import torch
from nazarite.config.train_config.env_cfgs.wtw_delta_direct_env_cfg import (
  Nazarite_Wtw_Delta_Direct_Go2,
)
from nazarite.delta.d435 import (
  DELTA_D435_INTRINSICS,
  DELTA_DIRECT_CAMERA_PITCH_DEG,
  DELTA_DIRECT_CAMERA_POS_B,
  DELTA_DIRECT_CAMERA_QUAT,
  DELTA_DIRECT_CAMERA_ROLL_DEG,
  DELTA_DIRECT_D435_INTRINSICS,
)
from nazarite.mdp.delta_observations import delta_depth_image
from nazarite.mdp.rewards import _support_gap_relief

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


def test_delta_depth_bev_uses_x_columns_and_y_rows() -> None:
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
    map_channels=5,
  ).reshape(1, 4, 6, 5)

  # The first two channels contain normalized grid coordinates. Columns are
  # x (forward), rows are y (left-to-right in the robot frame).
  assert torch.all(result[0, 0, 1:, 0] > result[0, 0, :-1, 0])
  assert torch.all(result[0, 1:, 0, 1] < result[0, :-1, 0, 1])


def test_direct_task_uses_near_field_bev_roi() -> None:
  cfg = Nazarite_Wtw_Delta_Direct_Go2(play=True)
  term = cfg.observations["delta_map"].terms["delta_map"]
  assert term.params["x_range_m"] == (0.0, 2.0)
  assert term.params["y_range_m"] == (-0.6, 0.6)
  assert term.params["fill_kernel_size"] == 3
  assert term.params["camera_pos_b"] == DELTA_DIRECT_CAMERA_POS_B
  assert term.params["camera_pitch_deg"] == DELTA_DIRECT_CAMERA_PITCH_DEG
  assert term.params["camera_roll_deg"] == DELTA_DIRECT_CAMERA_ROLL_DEG
  camera = next(s for s in cfg.scene.sensors if s.name == "delta_depth_camera")
  assert (camera.width, camera.height) == (72, 128)
  assert camera.pos == DELTA_DIRECT_CAMERA_POS_B
  assert camera.quat == DELTA_DIRECT_CAMERA_QUAT
  assert camera.quat == (-0.5, -0.5, 0.5, 0.5)
  assert camera.focal_length_px == pytest.approx(
    (DELTA_DIRECT_D435_INTRINSICS.fx, DELTA_DIRECT_D435_INTRINSICS.fy)
  )
  command = cfg.commands["twist"]
  assert command.forward_only_terrain_names == (
    "gaps",
    "stepping_stones",
    "grid_stones",
  )
  assert "delta_foot_support_scan" in {sensor.name for sensor in cfg.scene.sensors}
  assert (
    cfg.rewards["delta_swing_clearance"].params["terrain_map_cache_attr"]
    == "_delta_privileged_map_cache"
  )


def test_delta_depth_logs_spatial_support_diagnostics() -> None:
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
  env = SimpleNamespace(scene={"delta_depth_camera": sensor}, extras={"log": {}})
  delta_depth_image(
    env,
    map_height=4,
    map_width=6,
    camera_pos_b=(0.0, 0.0, 0.0),
    camera_pitch_deg=0.0,
    project_to_bev=True,
    map_channels=5,
    fill_kernel_size=1,
  )
  for name in (
    "DELTA/support_x_near_ratio",
    "DELTA/support_x_far_ratio",
    "DELTA/support_y_left_ratio",
    "DELTA/support_y_right_ratio",
  ):
    assert name in env.extras["log"]
    assert torch.isfinite(env.extras["log"][name])


def test_delta_depth_bev_fill_kernel_expands_confidence_without_support() -> None:
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
  map_5 = delta_depth_image(
    env,
    map_height=8,
    map_width=10,
    camera_pos_b=(0.0, 0.0, 0.0),
    camera_pitch_deg=0.0,
    project_to_bev=True,
    map_channels=5,
    fill_kernel_size=7,
  ).reshape(1, 8, 10, 5)
  assert torch.all(map_5[..., 3] <= map_5[..., 4])
  assert map_5[..., 4].mean() >= map_5[..., 3].mean()
  assert torch.isfinite(map_5).all()


def test_delta_depth_zero_ablation_preserves_grid_coordinates() -> None:
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
    project_to_bev=True,
    map_channels=5,
    fill_kernel_size=1,
    ablation_mode="zero",
  ).reshape(1, 4, 6, 5)

  assert torch.count_nonzero(result[..., :2]) > 0
  assert torch.count_nonzero(result[..., 2:]) == 0


def test_support_gap_relief_reduces_map_to_environment_scalar() -> None:
  valid = torch.ones(2, 16, 26, dtype=torch.bool)
  support = torch.ones(2, 16, 26)
  support[0, :, 4] = 0.0
  support[1, :, 4] = 0.5
  relief = _support_gap_relief(valid, support)
  assert relief.shape == (2,)
  assert relief[0] == pytest.approx(0.10)
  assert relief[1] == pytest.approx(0.05)
