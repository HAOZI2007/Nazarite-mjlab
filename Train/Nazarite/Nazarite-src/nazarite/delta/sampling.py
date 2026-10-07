"""Differentiable sampling utilities for the DELTA encoder."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bound_location(
    raw: torch.Tensor,
    limits: tuple[float, float],
    center: tuple[float, float] = (0.0, 0.0),
) -> torch.Tensor:
    """Bound metric locations around the map center.

    ``raw`` is expressed in robot metres while ``limits`` is the half extent of
    the map.  The previous implementation bounded around zero even when the BEV
    map covered only the forward field (for example ``x=[0, 2.5]``), causing a
    large fraction of attention samples to be clamped outside the useful map.
    """
    limit = raw.new_tensor(limits)
    offset = raw - raw.new_tensor(center)
    return raw.new_tensor(center) + limit * torch.tanh(offset / limit.clamp_min(1e-6))


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
        (
            (location[..., 0] - x_center) / x_limit,
            -(location[..., 1] - y_center) / y_limit,
        ),
        dim=-1,
    ).clamp(-1.0, 1.0)


def make_base_reference_grid(
    heads: int,
    samples: int,
    limits: tuple[float, float],
    center: tuple[float, float] = (0.0, 0.0),
    perturbation: float = 0.02,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return a two-dimensional ``[heads, samples, 2]`` reference grid.

    DELTA uses ``K=8`` samples per head.  Treating non-square ``K`` as a 1-D
    sweep puts every initial reference on ``y=0`` and leaves the lateral map
    unexplored.  A compact rows-by-columns lattice preserves two-dimensional
    coverage for arbitrary sample counts.  The small, one-time perturbation is
    the paper's raw-reference initialization jitter; it is a model parameter
    initialization detail, not per-step observation noise.
    """
    if heads < 1 or samples < 1:
        raise ValueError("heads and samples must be positive")
    if perturbation < 0.0:
        raise ValueError("perturbation must be non-negative")
    x_limit, y_limit = limits
    rows = max(1, int(samples**0.5))
    columns = (samples + rows - 1) // rows
    axis_x = torch.linspace(-0.7 * x_limit, 0.7 * x_limit, columns, device=device)
    axis_y = torch.linspace(-0.7 * y_limit, 0.7 * y_limit, rows, device=device)
    yy, xx = torch.meshgrid(axis_y, axis_x, indexing="ij")
    base = torch.stack((xx.flatten(), yy.flatten()), -1)[:samples]
    references = base.unsqueeze(0).repeat(heads, 1, 1)
    if perturbation > 0.0:
        scale = references.new_tensor((x_limit, y_limit)) * float(perturbation)
        references = references + (torch.rand_like(references) * 2.0 - 1.0) * scale
    return references + base.new_tensor(center)


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
    if elevation_map.ndim != 4 or elevation_map.shape[-1] not in (3, 4, 5):
        raise ValueError(
            "elevation_map must have shape [B,H,W,3], [B,H,W,4], or [B,H,W,5]"
        )
    b, h, w, channels = elevation_map.shape
    _, heads, samples, _ = centers.shape
    radius = (size - 1) / 2.0
    axis = (
        torch.arange(
            -radius, radius + 1, device=elevation_map.device, dtype=elevation_map.dtype
        )
        * step
    )
    dy, dx = torch.meshgrid(axis, axis, indexing="ij")
    offsets = torch.stack((dx, dy), dim=-1)
    metric = centers[:, :, :, None, None, :] + offsets
    grid = metric_to_grid(metric, limits, center).reshape(
        b * heads * samples, size, size, 2
    )
    image = elevation_map.permute(0, 3, 1, 2)[:, None].expand(
        b, heads * samples, channels, h, w
    )
    image = image.reshape(b * heads * samples, channels, h, w)
    patch = F.grid_sample(
        image, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    return patch.reshape(b, heads, samples, channels, size, size)
