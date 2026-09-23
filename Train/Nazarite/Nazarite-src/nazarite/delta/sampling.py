"""Differentiable sampling utilities for the DELTA encoder."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bound_location(raw: torch.Tensor, limits: tuple[float, float]) -> torch.Tensor:
  """Bound metric (x,y) locations to the valid map-center rectangle."""
  limit = raw.new_tensor(limits)
  return limit * torch.tanh(raw / limit.clamp_min(1e-6))


def metric_to_grid(
  location: torch.Tensor,
  limits: tuple[float, float],
  center: tuple[float, float] = (0.0, 0.0),
) -> torch.Tensor:
  """Convert robot-frame metres (x forward, y left) to grid_sample coordinates."""
  x_limit, y_limit = limits
  x_center, y_center = center
  # grid_sample uses (column,row); image rows increase downwards, opposite to +y.
  return torch.stack(
    ((location[..., 0] - x_center) / x_limit,
     -(location[..., 1] - y_center) / y_limit), dim=-1
  ).clamp(-1.0, 1.0)


def make_base_reference_grid(
  heads: int, samples: int, limits: tuple[float, float], device: torch.device | None = None
) -> torch.Tensor:
  """Return a deterministic [heads,samples,2] coarse reference grid."""
  x_limit, y_limit = limits
  side = int(samples**0.5)
  if side * side != samples:
    # Fall back to a regular 1-D sweep when K is not a square number.
    x = torch.linspace(-0.7 * x_limit, 0.7 * x_limit, samples, device=device)
    y = torch.zeros_like(x)
    base = torch.stack((x, y), -1)
  else:
    axis_x = torch.linspace(-0.7 * x_limit, 0.7 * x_limit, side, device=device)
    axis_y = torch.linspace(-0.7 * y_limit, 0.7 * y_limit, side, device=device)
    yy, xx = torch.meshgrid(axis_y, axis_x, indexing="ij")
    base = torch.stack((xx.flatten(), yy.flatten()), -1)
  return base.unsqueeze(0).repeat(heads, 1, 1)


def bilinear_patches(
  elevation_map: torch.Tensor,
  centers: torch.Tensor,
  size: int,
  step: float,
  limits: tuple[float, float],
  center: tuple[float, float] = (0.0, 0.0),
) -> torch.Tensor:
  """Sample patches from [B,H,W,3] at [B,Nh,K,2] metric centers.

  Returns [B,Nh,K,3,size,size].
  """
  if elevation_map.ndim != 4 or elevation_map.shape[-1] not in (3, 4):
    raise ValueError("elevation_map must have shape [B,H,W,3] or [B,H,W,4]")
  b, h, w, channels = elevation_map.shape
  _, heads, samples, _ = centers.shape
  radius = (size - 1) / 2.0
  axis = torch.arange(-radius, radius + 1, device=elevation_map.device, dtype=elevation_map.dtype) * step
  dy, dx = torch.meshgrid(axis, axis, indexing="ij")
  offsets = torch.stack((dx, dy), dim=-1)
  metric = centers[:, :, :, None, None, :] + offsets
  grid = metric_to_grid(metric, limits, center).reshape(b * heads * samples, size, size, 2)
  image = elevation_map.permute(0, 3, 1, 2)[:, None].expand(b, heads * samples, channels, h, w)
  image = image.reshape(b * heads * samples, channels, h, w)
  patch = F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=True)
  return patch.reshape(b, heads, samples, channels, size, size)
