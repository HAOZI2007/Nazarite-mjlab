"""Compatibility actor for Direct checkpoints written before the DELTA refactor.

The 2026-09-24 Direct run stores a linear ``action_fusion`` head and uses all
non-coordinate map channels in the center/context token.  Keep this adapter
local to sim2sim so the training implementation remains the current one.
"""

from __future__ import annotations

from typing import cast

import torch
from rsl_rl.modules import MLP
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_class
from tensordict import TensorDict
from torch import nn

from nazarite.delta.sampling import (
    bilinear_patches,
    bound_location,
    make_base_reference_grid,
)


def _rms_norm(value: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps)


def _legacy_center_context(
    patch: torch.Tensor, final: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    # Channels 0/1 are coordinates. The old Direct encoder tokenized z,
    # support and confidence together, which gives 3 channels for a 5-channel map.
    geometry = patch[:, :, :, 2:].permute(0, 1, 2, 4, 5, 3)
    batch, heads, samples, patch_size, _, geometry_channels = geometry.shape
    if not final:
        return geometry.reshape(batch, heads, samples, -1), geometry.new_zeros(
            (batch, heads, samples, 0)
        )
    c0 = (patch_size - 3) // 2
    center = geometry[:, :, :, c0 : c0 + 3, c0 : c0 + 3].reshape(
        batch, heads, samples, 9 * geometry_channels
    )
    mask = torch.ones(
        (1, 1, 1, patch_size, patch_size, 1),
        dtype=torch.bool,
        device=geometry.device,
    )
    mask[:, :, :, c0 : c0 + 3, c0 : c0 + 3, :] = False
    context = geometry.masked_select(mask).reshape(
        batch, heads, samples, (patch_size * patch_size - 9) * geometry_channels
    )
    return center, context


class _LegacyDeltaLayer(nn.Module):
    def __init__(self, dim: int, heads: int, samples: int, final: bool, map_channels: int):
        super().__init__()
        self.dim, self.heads, self.samples, self.final = dim, heads, samples, final
        self.map_channels = int(map_channels)
        dh = dim // heads
        self.dh = dh
        self.offset = nn.Linear(dim, heads * samples * 2)
        self.scout = nn.Sequential(
            nn.Linear(9 * self.map_channels, 64), nn.ELU(), nn.Linear(64, dim - 3)
        )
        geometry_channels = self.map_channels - 2 if self.map_channels >= 5 else 1
        self.center = nn.Sequential(
            nn.Linear(9 * geometry_channels, 64), nn.ELU(), nn.Linear(64, 61)
        )
        context_in = 9 * 8 * geometry_channels if final else 0
        self.context = nn.Sequential(
            nn.Linear(max(context_in, 1), 64), nn.ELU(), nn.Linear(64, 61)
        )
        self.context_proj = nn.Linear(61, 61)
        self.gate = nn.Linear(122, 1)
        self.refine = nn.Sequential(nn.Linear(2 * dim, 64), nn.ELU(), nn.Linear(64, 2))
        self.key = nn.Linear(dim, dh)
        self.value = nn.Linear(dim, dh)
        self.out = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 256), nn.ELU(), nn.Linear(256, dim))

    def forward(self, q, terrain, raw_ref, limits, step, gamma, map_center):
        d0 = gamma * torch.tanh(self.offset(q)).view(
            q.shape[0], self.heads, self.samples, 2
        )
        coarse = bound_location(raw_ref + d0, limits, map_center)
        scout_patch = bilinear_patches(terrain, coarse, 3, step, limits, map_center)
        scout_feat = self.scout(scout_patch.flatten(start_dim=3))
        xyz = torch.cat(
            (
                (coarse - coarse.new_tensor(map_center)) / coarse.new_tensor(limits),
                coarse.new_zeros((*coarse.shape[:3], 1)),
            ),
            dim=-1,
        )
        scout_token = torch.cat((scout_feat, xyz), dim=-1)
        q_expand = q[:, None, None].expand(-1, self.heads, self.samples, -1)
        dr = gamma * torch.tanh(
            self.refine(torch.cat((q_expand, scout_token), dim=-1))
        ).view(q.shape[0], self.heads, self.samples, 2)
        refined = bound_location(raw_ref + d0 + dr, limits, map_center)

        size = 9 if self.final else 3
        patch = bilinear_patches(terrain, refined, size, step, limits, map_center)
        center, context = _legacy_center_context(patch, self.final)
        center_feature = self.center(center)
        if self.final:
            context_feature = self.context(context)
            gate = torch.sigmoid(
                self.gate(torch.cat((center_feature, context_feature), dim=-1))
            )
            phi = center_feature + gate * self.context_proj(context_feature)
        else:
            phi = center_feature
        xyz = torch.cat(
            (
                (refined - refined.new_tensor(map_center)) / refined.new_tensor(limits),
                refined.new_zeros((*refined.shape[:3], 1)),
            ),
            dim=-1,
        )
        token = torch.cat((phi, xyz), dim=-1)
        qh = q.view(q.shape[0], self.heads, self.dh)
        key = self.key(token).view(q.shape[0], self.heads, self.samples, self.dh)
        value = _rms_norm(
            self.value(token).view(q.shape[0], self.heads, self.samples, self.dh)
        )
        scores = (qh[:, :, None] * key).sum(-1) / (self.dh**0.5)
        valid_score = refined.new_ones(refined.shape[:-1])
        if self.map_channels >= 4:
            confidence_channel = 4 if self.map_channels >= 5 else 3
            valid_score = patch[:, :, :, confidence_channel].mean(dim=(-1, -2)).clamp(
                1.0e-3, 1.0
            )
            scores = scores + valid_score.log()
        attention = scores.softmax(-1)
        context_out = (attention[..., None] * value).sum(-2).reshape(q.shape[0], self.dim)
        q = self.norm1(q + self.out(context_out))
        q = self.norm2(q + self.ffn(q))
        return q, refined, attention


class LegacyDeltaEncoder(nn.Module):
    def __init__(self, proprio_dim: int, map_height: int, map_width: int, map_channels: int, map_extent, map_center):
        super().__init__()
        self.dim, self.heads, self.samples = 64, 4, 8
        self.map_channels = int(map_channels)
        self.map_height, self.map_width = int(map_height), int(map_width)
        self.map_center: tuple[float, float] = (
            float(map_center[0]),
            float(map_center[1]),
        )
        self.limits = (float(map_extent[0]) / 2.0, float(map_extent[1]) / 2.0)
        self.query = nn.Sequential(
            nn.Linear(proprio_dim, 64), nn.ELU(), nn.Linear(64, 64), nn.LayerNorm(64)
        )
        self.layers = nn.ModuleList(
            [
                _LegacyDeltaLayer(64, 4, 8, index == 2, self.map_channels)
                for index in range(3)
            ]
        )
        self.register_buffer(
            "base_reference",
            make_base_reference_grid(4, 8, self.limits, self.map_center),
        )

    def forward(self, proprio: torch.Tensor, terrain: torch.Tensor) -> torch.Tensor:
        q = self.query(proprio)
        base_reference = cast(torch.Tensor, self.base_reference)
        raw_ref = base_reference.unsqueeze(0).expand(proprio.shape[0], -1, -1, -1)
        for layer in self.layers:
            q, raw_ref, _ = layer(
                q, terrain, raw_ref, self.limits, 0.08, 0.25, self.map_center
            )
        return q


class LegacyWtwDeltaDirectModel(nn.Module):
    """Exact model shape used by the pre-refactor Direct checkpoint."""

    def __init__(self, proprio_dim: int, map_height: int, map_width: int, map_channels: int):
        super().__init__()
        self.output_dim = 12
        self.delta_proprio_dim = proprio_dim
        self.wtw_policy = MLP(498, 12, (512, 256, 128), "elu")
        self.delta = LegacyDeltaEncoder(
            proprio_dim, map_height, map_width, map_channels, (2.0, 1.2), (1.0, 0.0)
        )
        self.action_fusion = nn.Linear(12 + proprio_dim + 64, 12)
        self.register_buffer("residual_joint_scales", torch.ones(12))
        self.distribution: Distribution = resolve_class(
            {"class_name": "GaussianDistribution"}
        )[0](12, init_std=0.20, std_type="log", std_range=(0.08, 0.25))
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    def forward(self, obs: TensorDict):
        base = obs["wtw_proprio"]
        proprio = obs["delta_proprio"]
        terrain = obs["delta_map"].reshape(
            base.shape[0], self.delta.map_height, self.delta.map_width, self.delta.map_channels
        )
        latent = self.delta(proprio, terrain)
        prior = self.wtw_policy(base)
        mean = self.action_fusion(torch.cat((prior, proprio, latent), dim=-1))
        self.last_diagnostics = {
            "DELTA/direct_action_rms": mean.square().mean().sqrt().detach(),
            "DELTA/direct_action_prior_rms": prior.square().mean().sqrt().detach(),
            "DELTA/encoder_output_rms": latent.square().mean().sqrt().detach(),
        }
        self.distribution.update(mean)
        return self.distribution.deterministic_output(mean)

    def reset(self):
        return None

    def act_inference(self, obs: TensorDict):
        return self(obs)
