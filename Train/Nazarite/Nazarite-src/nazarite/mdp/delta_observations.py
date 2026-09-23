"""Observation terms used by the DELTA terrain policy."""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def delta_elevation_map(
  env,
  sensor_name: str = "terrain_scan",
  height: int = 26,
  width: int = 16,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Return a flattened robot-frame DELTA terrain map.

  The raycast sensor stores hit points in world coordinates.  We transform them
  into the current base frame so the policy sees a translation/rotation-local
  map.  Misses are represented by zero and can be randomized during training.

  ObservationManager concatenates all terms along the feature dimension, so this
  term must expose ``[B, H*W*3]`` rather than ``[B, H, W, 3]``.  The DELTA model
  reshapes it back to ``[B, H, W, 3]`` before passing it to the encoder.
  """
  sensor = env.scene[sensor_name]
  asset: Entity = env.scene[asset_cfg.name]
  hit_w = sensor.data.hit_pos_w
  distances = sensor.data.distances
  b = hit_w.shape[0]
  if hit_w.numel() != b * height * width * 3:
    raise ValueError(
      f"{sensor_name} returned {hit_w.shape}; expected [B,{height*width},3]"
    )
  hit_w = hit_w.reshape(b, height * width, 3)
  distances = distances.reshape(b, height * width)
  base_pos = asset.data.root_link_pos_w[:, None, :]
  base_quat = asset.data.root_link_quat_w[:, None, :].expand(-1, height * width, -1)
  hit_b = quat_apply_inverse(base_quat, hit_w - base_pos)
  valid = distances >= 0
  hit_b = torch.where(valid[..., None], hit_b, torch.zeros_like(hit_b))
  # Match the paper's coordinate normalization before the model sees the map.
  hit_b = hit_b.reshape(b, height, width, 3)
  hit_b[..., 0] = hit_b[..., 0] / 1.25
  hit_b[..., 1] = hit_b[..., 1] / 0.75
  hit_b[..., 2] = hit_b[..., 2].clamp(-0.8, 0.8) / 0.6
  hit_b = torch.nan_to_num(hit_b, nan=0.0, posinf=0.0, neginf=0.0)
  return hit_b.reshape(b, height * width * 3)


def delta_depth_image(
  env,
  sensor_name: str = "delta_depth_camera",
  map_height: int = 16,
  map_width: int = 26,
  max_depth: float = 5.0,
  horizontal_fov_deg: float | None = None,
  vertical_fov_deg: float | None = None,
  camera_pos_b: tuple[float, float, float] = (0.30, 0.0, 0.12),
  camera_pitch_deg: float = 20.0,
  fx_px: float | None = None,
  fy_px: float | None = None,
  cx_px: float | None = None,
  cy_px: float | None = None,
  project_to_bev: bool = False,
  x_range_m: tuple[float, float] = (0.0, 3.0),
  y_range_m: tuple[float, float] = (-1.0, 1.0),
  z_scale_m: float = 0.8,
  map_channels: int = 3,
) -> torch.Tensor:
  """Convert one head-mounted depth image to a DELTA terrain map.

  With ``project_to_bev=True`` the perspective image is splatted into a
  regular robot-frame x-y grid.  This is the representation expected by the
  deformable DELTA sampler.  The optional fourth channel is a valid-obstacle
  mask; it is deliberately opt-in so old three-channel DELTA checkpoints keep
  their observation contract.
  """
  sensor: CameraSensor = env.scene[sensor_name]
  depth_data = sensor.data.depth
  if depth_data is None:
    raise RuntimeError(f"Camera '{sensor_name}' has no depth output")
  if depth_data.ndim != 4 or depth_data.shape[-1] != 1:
    raise ValueError(f"Expected depth [B,H,W,1], got {tuple(depth_data.shape)}")

  depth = depth_data[..., 0].float()
  batch, image_height, image_width = depth.shape
  device, dtype = depth.device, depth.dtype
  explicit_intrinsics = (fx_px, fy_px, cx_px, cy_px)
  if any(value is not None for value in explicit_intrinsics):
    if not all(value is not None for value in explicit_intrinsics):
      raise ValueError("fx_px, fy_px, cx_px, and cy_px must be provided together")
    assert fx_px is not None and fy_px is not None
    assert cx_px is not None and cy_px is not None
    if fx_px <= 0.0 or fy_px <= 0.0:
      raise ValueError("Camera focal lengths must be positive")
    fx = depth.new_tensor(fx_px)
    fy = depth.new_tensor(fy_px)
    cx = depth.new_tensor(cx_px)
    cy = depth.new_tensor(cy_px)
  elif sensor.cfg.focal_length_px is not None:
    assert sensor.cfg.principal_point_px is not None
    scale_x = image_width / sensor.cfg.width
    scale_y = image_height / sensor.cfg.height
    fx = depth.new_tensor(sensor.cfg.focal_length_px[0] * scale_x)
    fy = depth.new_tensor(sensor.cfg.focal_length_px[1] * scale_y)
    cx = depth.new_tensor(sensor.cfg.principal_point_px[0] * scale_x)
    cy = depth.new_tensor(sensor.cfg.principal_point_px[1] * scale_y)
  else:
    # Backward-compatible fallback for cameras configured only by FOV.
    horizontal_fov_deg = 87.0 if horizontal_fov_deg is None else horizontal_fov_deg
    vertical_fov_deg = 58.0 if vertical_fov_deg is None else vertical_fov_deg
    if not 0.0 < horizontal_fov_deg < 180.0:
      raise ValueError("horizontal_fov_deg must be between 0 and 180")
    if not 0.0 < vertical_fov_deg < 180.0:
      raise ValueError("vertical_fov_deg must be between 0 and 180")
    half_pi = torch.pi / 360.0
    fx = image_width / (
      2.0 * torch.tan(depth.new_tensor(horizontal_fov_deg * half_pi))
    )
    fy = image_height / (
      2.0 * torch.tan(depth.new_tensor(vertical_fov_deg * half_pi))
    )
    cx = depth.new_tensor((image_width - 1) * 0.5)
    cy = depth.new_tensor((image_height - 1) * 0.5)
  u = torch.arange(image_width, device=device, dtype=dtype)
  v = torch.arange(image_height, device=device, dtype=dtype)
  vv, uu = torch.meshgrid(v, u, indexing="ij")
  valid = torch.isfinite(depth) & (depth > 1.0e-3) & (depth < max_depth)
  z = depth.clamp(min=0.0, max=max_depth)
  x_right = (uu[None] - cx) * z / fx
  y_down = (vv[None] - cy) * z / fy

  # Optical axis points forward and is pitched down by camera_pitch_deg.
  pitch = depth.new_tensor(camera_pitch_deg * torch.pi / 180.0)
  cp, sp = torch.cos(pitch), torch.sin(pitch)
  body_x = cp * z - sp * y_down + camera_pos_b[0]
  body_y = -x_right + camera_pos_b[1]
  body_z = -sp * z - cp * y_down + camera_pos_b[2]
  points = torch.stack((body_x, body_y, body_z), dim=-1)
  points = torch.where(valid[..., None], points, torch.zeros_like(points))
  points = torch.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)

  if project_to_bev:
    if map_channels not in (3, 4):
      raise ValueError("BEV DELTA maps support 3 or 4 channels")
    x0, x1 = (float(x_range_m[0]), float(x_range_m[1]))
    y0, y1 = (float(y_range_m[0]), float(y_range_m[1]))
    if not x1 > x0 or not y1 > y0:
      raise ValueError("x_range_m and y_range_m must be increasing")
    if z_scale_m <= 0.0:
      raise ValueError("z_scale_m must be positive")

    # Rows run from +y to -y so grid_sample's row orientation matches the
    # robot-frame convention used by the attention sampler.
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]
    col = torch.floor((x - x0) / (x1 - x0) * map_width).long()
    row = torch.floor((y1 - y) / (y1 - y0) * map_height).long()
    inside = (
      valid
      & (col >= 0) & (col < map_width)
      & (row >= 0) & (row < map_height)
      & torch.isfinite(z)
    )

    cells = map_height * map_width
    cell = (row.clamp(0, map_height - 1) * map_width + col.clamp(0, map_width - 1))
    batch_offset = torch.arange(batch, device=device).view(batch, 1, 1) * cells
    flat_cell = (cell + batch_offset).reshape(-1)
    flat_valid = inside.reshape(-1)
    flat_z = (z / float(z_scale_m)).clamp(-1.0, 1.0).reshape(-1)

    # Use the highest observed surface in a cell.  This preserves obstacle
    # tops when a pixel also sees the ground behind the obstacle.
    bev_z = torch.full((batch * cells,), -1.0, device=device, dtype=dtype)
    valid_values = torch.where(flat_valid, flat_z, torch.full_like(flat_z, -1.0))
    bev_z.scatter_reduce_(0, flat_cell, valid_values, reduce="amax", include_self=True)
    bev_valid = torch.zeros((batch * cells,), device=device, dtype=dtype)
    bev_valid.scatter_add_(0, flat_cell, flat_valid.to(dtype))
    bev_valid = (bev_valid > 0.0).to(dtype)
    bev_z = bev_z.reshape(batch, map_height, map_width)
    bev_valid = bev_valid.reshape(batch, map_height, map_width)
    raw_bev_valid = bev_valid.clone()

    # A point splat leaves holes when the camera is downsampled to the policy
    # map. Fill nearby holes with a validity-weighted local average so the
    # encoder receives a usable surface instead of a mostly-zero image. The
    # original mask is retained in the fourth channel as a soft confidence.
    valid_f = bev_valid[:, None]
    z_f = bev_z[:, None] * valid_f
    local_kernel = 5
    local_sum = torch.nn.functional.avg_pool2d(
      z_f, local_kernel, stride=1, padding=local_kernel // 2
    ) * (local_kernel ** 2)
    local_count = torch.nn.functional.avg_pool2d(
      valid_f, local_kernel, stride=1, padding=local_kernel // 2
    ) * (local_kernel ** 2)
    local_z = local_sum / local_count.clamp_min(1.0)
    hole = bev_valid <= 0.0
    bev_z = torch.where(hole, local_z[:, 0], bev_z)
    bev_valid = torch.maximum(
      bev_valid,
      (local_count[:, 0] > 0.0).to(dtype) * 0.5,
    )

    x_centers = torch.linspace(
      x0 + 0.5 * (x1 - x0) / map_width,
      x1 - 0.5 * (x1 - x0) / map_width,
      map_width, device=device, dtype=dtype,
    )
    y_centers = torch.linspace(
      y1 - 0.5 * (y1 - y0) / map_height,
      y0 + 0.5 * (y1 - y0) / map_height,
      map_height, device=device, dtype=dtype,
    )
    # The map coordinates are normalized to [-1, 1], matching encoder.limits.
    x_grid = (x_centers - x0) / (x1 - x0) * 2.0 - 1.0
    y_grid = (y_centers - y0) / (y1 - y0) * 2.0 - 1.0
    yy, xx = torch.meshgrid(y_grid, x_grid, indexing="ij")
    xy = torch.stack((xx, yy), dim=-1).expand(batch, -1, -1, -1)
    output = torch.cat((xy, bev_z[..., None]), dim=-1)
    if map_channels == 4:
      output = torch.cat((output, bev_valid[..., None]), dim=-1)
    # Keep the latest map available to DELTA-specific rewards.  Rewards are
    # evaluated before the next observation pass, so this is intentionally a
    # one-step-lagged cache; it is preferable to duplicating the camera
    # projection inside every reward term and remains synchronized with the
    # policy input during normal stepping.
    cached_map = output.detach()
    setattr(env, "_delta_map_cache", cached_map)
    if hasattr(env, "extras") and "log" in env.extras:
      env.extras["log"]["DELTA/raw_map_occupancy_ratio"] = raw_bev_valid.mean()
      env.extras["log"]["DELTA/filled_map_ratio"] = bev_valid.mean()
      env.extras["log"]["DELTA/map_observed_z_span"] = (
        (bev_z.amax(dim=(-1, -2)) - bev_z.amin(dim=(-1, -2))).mean()
      )
    return output.reshape(batch, map_height * map_width * map_channels)

  points = points.permute(0, 3, 1, 2)
  points = torch.nn.functional.interpolate(
    points, size=(map_height, map_width), mode="bilinear", align_corners=True,
  ).permute(0, 2, 3, 1)
  points[..., 0] = points[..., 0].clamp(0.0, max_depth) / max_depth
  points[..., 1] = points[..., 1].clamp(-1.5, 1.5) / 1.5
  points[..., 2] = points[..., 2].clamp(-0.8, 0.8) / 0.8
  return points.reshape(batch, map_height * map_width * 3)
