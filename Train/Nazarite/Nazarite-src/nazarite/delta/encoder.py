"""DELTA: deformable elevation-based local terrain attention encoder."""

from __future__ import annotations

import torch
from torch import nn

from .sampling import bilinear_patches, bound_location, make_base_reference_grid


def _rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
  return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)


def _center_context(patch: torch.Tensor, final: bool) -> tuple[torch.Tensor, torch.Tensor]:
  """Return flattened relative-height center/context, each [B,Nh,K,N]."""
  # z channel is the terrain geometry; x/y are supplied separately as center coords.
  z = patch[:, :, :, 2]
  if not final:
    return z.flatten(-2), z.new_zeros((*z.shape[:3], 0))
  p = z.shape[-1]
  c0 = (p - 3) // 2
  center = z[..., c0:c0 + 3, c0:c0 + 3].flatten(-2)
  mask = torch.ones_like(z, dtype=torch.bool)
  mask[..., c0:c0 + 3, c0:c0 + 3] = False
  context = z.masked_select(mask).reshape(*z.shape[:3], -1)
  return center, context


class _DeltaLayer(nn.Module):
  def __init__(self, dim: int, heads: int, samples: int, final: bool,
               map_channels: int) -> None:
    super().__init__()
    self.dim, self.heads, self.samples, self.final = dim, heads, samples, final
    dh = dim // heads
    if dim % heads:
      raise ValueError("dim must be divisible by heads")
    self.dh = dh
    self.offset = nn.Linear(dim, heads * samples * 2)
    self.map_channels = int(map_channels)
    self.scout = nn.Sequential(
      nn.Linear(9 * self.map_channels, 64), nn.ELU(), nn.Linear(64, dim - 3)
    )
    context_in = 9 * 8 if final else 0
    self.center = nn.Sequential(nn.Linear(9, 64), nn.ELU(), nn.Linear(64, 61))
    self.context = nn.Sequential(nn.Linear(max(context_in, 1), 64), nn.ELU(), nn.Linear(64, 61))
    self.context_proj = nn.Linear(61, 61)
    self.gate = nn.Linear(122, 1)
    self.refine = nn.Sequential(nn.Linear(2 * dim, 64), nn.ELU(), nn.Linear(64, 2))
    self.key, self.value = nn.Linear(dim, dh), nn.Linear(dim, dh)
    self.out = nn.Linear(dim, dim)
    self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
    self.ffn = nn.Sequential(nn.Linear(dim, 256), nn.ELU(), nn.Linear(256, dim))

  def forward(self, q, terrain, raw_ref, limits, step, gamma, map_center):
    d0 = gamma * torch.tanh(self.offset(q)).view(q.shape[0], self.heads, self.samples, 2)
    coarse = bound_location(raw_ref + d0, limits)
    scout_patch = bilinear_patches(terrain, coarse, 3, step, limits, map_center)
    scout_flat = scout_patch.flatten(start_dim=3)
    scout_feat = self.scout(scout_flat)
    xyz = torch.cat((coarse, coarse.new_zeros((*coarse.shape[:3], 1))), -1)
    scout_token = torch.cat((scout_feat, xyz), -1)
    q_expand = q[:, None, None].expand(-1, self.heads, self.samples, -1)
    dr = gamma * torch.tanh(self.refine(torch.cat((q_expand, scout_token), -1))).view(
      q.shape[0], self.heads, self.samples, 2
    )
    refined = bound_location(raw_ref + d0 + dr, limits)

    size = 9 if self.final else 3
    patch = bilinear_patches(terrain, refined, size, step, limits, map_center)
    center, context = _center_context(patch, self.final)
    c = self.center(center)
    if self.final:
      u = self.context(context)
      eta = torch.sigmoid(self.gate(torch.cat((c, u), -1)))
      phi = c + eta * self.context_proj(u)
    else:
      phi = c
    xyz = torch.cat((refined / refined.new_tensor((*limits,)), refined.new_zeros((*refined.shape[:3], 1))), -1)
    token = torch.cat((phi, xyz), -1)
    qh = q.view(q.shape[0], self.heads, self.dh)
    k = self.key(token).view(q.shape[0], self.heads, self.samples, self.dh)
    v = _rms_norm(self.value(token).view(q.shape[0], self.heads, self.samples, self.dh))
    scores = (qh[:, :, None] * k).sum(-1) / (self.dh**0.5)
    valid_score = refined.new_ones(refined.shape[:-1])
    if self.map_channels >= 4:
      # A soft validity bias prevents empty BEV cells from becoming attractive
      # while keeping a finite fallback when an entire local patch is missing.
      valid_score = patch[:, :, :, 3].mean(dim=(-1, -2)).clamp(1.0e-3, 1.0)
      scores = scores + valid_score.log()
    attn = scores.softmax(-1)
    context_out = (attn[..., None] * v).sum(-2).reshape(q.shape[0], self.dim)
    q = self.norm1(q + self.out(context_out))
    q = self.norm2(q + self.ffn(q))
    return q, refined, attn, valid_score


class DeltaEncoder(nn.Module):
  """State-conditioned fixed-budget terrain encoder.

  ``proprio`` may have any dimension, allowing the current project observation
  without adding base linear velocity. The default 45 dimensions correspond to
  the existing proprioception with that term omitted.
  """

  def __init__(self, proprio_dim: int = 45, dim: int = 64, layers: int = 3,
               heads: int = 4, samples: int = 8,
               map_extent: tuple[float, float] = (2.0, 1.2),
               map_channels: int = 3,
               map_center: tuple[float, float] = (0.0, 0.0),
               forward_x_threshold: float = -0.75) -> None:
    super().__init__()
    if map_channels not in (3, 4):
      raise ValueError("DELTA maps support 3 or 4 channels")
    self.dim, self.heads, self.samples = dim, heads, samples
    self.map_channels = int(map_channels)
    self.map_center = tuple(float(value) for value in map_center)
    self.forward_x_threshold = float(forward_x_threshold)
    self.limits = (map_extent[0] / 2.0, map_extent[1] / 2.0)
    self.query = nn.Sequential(nn.Linear(proprio_dim, 64), nn.ELU(), nn.Linear(64, dim), nn.LayerNorm(dim))
    self.layers = nn.ModuleList([
      _DeltaLayer(dim, heads, samples, i == layers - 1, self.map_channels)
      for i in range(layers)
    ])
    self.register_buffer("base_reference", make_base_reference_grid(heads, samples, self.limits))
    # Disabled during training to avoid retaining per-step tensors.  The play
    # viewer enables it explicitly and consumes the detached snapshot below.
    self.attention_cache_enabled = False
    self.last_aux: list[dict[str, torch.Tensor]] = []
    self.last_map: torch.Tensor | None = None
    self.last_stats: dict[str, torch.Tensor] = {}

  def enable_attention_cache(self, enabled: bool = True) -> None:
    """Enable/disable detached sampling-location and attention snapshots."""
    self.attention_cache_enabled = bool(enabled)
    if not enabled:
      self.last_aux = []
      self.last_map = None

  def get_attention(self) -> list[dict[str, torch.Tensor]]:
    """Return the latest detached DELTA attention snapshot."""
    return self.last_aux

  def forward(self, proprio: torch.Tensor, elevation_map: torch.Tensor, return_aux: bool = False):
    if proprio.ndim != 2:
      raise ValueError("proprio must have shape [B,P]")
    if elevation_map.ndim != 4 or elevation_map.shape[-1] != self.map_channels:
      raise ValueError(
        f"elevation_map must have shape [B,H,W,{self.map_channels}], "
        f"got {tuple(elevation_map.shape)}"
      )
    q = self.query(proprio)
    raw_ref = self.base_reference.unsqueeze(0).expand(proprio.shape[0], -1, -1, -1)
    aux = []
    attention_entropies = []
    forward_ratios = []
    attention_valid_ratios = []
    attention_x_means = []
    for layer in self.layers:
      q, raw_ref, attn, valid_score = layer(
        q, elevation_map, raw_ref, self.limits, 0.08, 0.25, self.map_center,
      )
      # Keep only scalar diagnostics during normal training.  The full
      # attention tensors are cached only when the play visualizer requests it.
      attention_entropies.append(
        -(attn.clamp_min(1.0e-8) * attn.clamp_min(1.0e-8).log()).sum(dim=-1).mean()
      )
      forward_ratios.append(
        (attn * (raw_ref[..., 0] > self.forward_x_threshold).to(attn.dtype)).sum(-1).mean()
      )
      attention_x_means.append((attn * raw_ref[..., 0]).sum(-1).mean())
      if self.map_channels >= 4:
        attention_valid_ratios.append((attn * valid_score).sum(-1).mean())
      if return_aux or self.attention_cache_enabled:
        aux.append({"locations": raw_ref, "attention": attn})
    self.last_stats = {
      "DELTA/attention_entropy": torch.stack(attention_entropies).mean().detach(),
      "DELTA/attention_forward_ratio": torch.stack(forward_ratios).mean().detach(),
      "DELTA/attention_x_mean": torch.stack(attention_x_means).mean().detach(),
    }
    if self.map_channels >= 4:
      self.last_stats["DELTA/map_valid_ratio"] = elevation_map[..., 3].mean().detach()
      self.last_stats["DELTA/attention_valid_ratio"] = (
        torch.stack(attention_valid_ratios).mean().detach()
      )
    if self.attention_cache_enabled:
      self.last_aux = [
        {"locations": item["locations"].detach(), "attention": item["attention"].detach()}
        for item in aux
      ]
      self.last_map = elevation_map.detach()
    return (q, aux) if return_aux else q
