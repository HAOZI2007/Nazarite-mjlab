"""DiT-style epsilon-prediction model for Go2 motion windows."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional


class _Timesteps(nn.Module):
    def __init__(self, channels: int = 256):
        super().__init__()
        if channels % 2:
            raise ValueError("timestep channels must be even")
        self.channels = channels

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.channels // 2
        exponent = (
            -math.log(10000.0)
            * torch.arange(half, dtype=torch.float32, device=timesteps.device)
            / half
        )
        embedding = timesteps.float()[:, None] * torch.exp(exponent)[None]
        return torch.cat((torch.cos(embedding), torch.sin(embedding)), dim=-1)


class _TimestepEmbedding(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(input_dim, output_dim)
        self.linear_2 = nn.Linear(output_dim, output_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear_2(functional.silu(self.linear_1(value)))


class _AdaLayerNormSingle(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.time_projection = _Timesteps(256)
        self.time_embedding = _TimestepEmbedding(256, dimension)
        self.linear = nn.Linear(dimension, 6 * dimension)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        embedded = self.time_embedding(self.time_projection(timestep)).unsqueeze(1)
        return self.linear(functional.silu(embedded))


class _PositionalEmbedding(nn.Module):
    embedding: torch.Tensor

    def __init__(self, dimension: int, maximum_length: int):
        super().__init__()
        position = torch.arange(maximum_length).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, dimension, 2) * (-math.log(10000.0) / dimension)
        )
        embedding = torch.zeros(1, maximum_length, dimension)
        embedding[0, :, 0::2] = torch.sin(position * divisor)
        embedding[0, :, 1::2] = torch.cos(position * divisor)
        self.register_buffer("embedding", embedding, persistent=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.embedding[:, : value.shape[1]]


class _FeedForward(nn.Module):
    def __init__(self, dimension: int, dropout: float):
        super().__init__()
        inner = 4 * dimension
        self.input_projection = nn.Linear(dimension, 2 * inner)
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(inner, dimension)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        gate, linear = self.input_projection(value).chunk(2, dim=-1)
        return self.output_projection(self.dropout(functional.silu(gate) * linear))


class _DiTBlock(nn.Module):
    def __init__(self, dimension: int, num_heads: int, dropout: float):
        super().__init__()
        self.dimension = dimension
        self.num_heads = num_heads
        self.head_dim = dimension // num_heads
        self.norm_attention = nn.LayerNorm(dimension, elementwise_affine=False)
        self.query = nn.Linear(dimension, dimension, bias=False)
        self.key = nn.Linear(dimension, dimension, bias=False)
        self.value = nn.Linear(dimension, dimension, bias=False)
        self.attention_output = nn.Linear(dimension, dimension, bias=False)
        self.attention_dropout = nn.Dropout(dropout)
        self.norm_feed_forward = nn.LayerNorm(dimension, elementwise_affine=False)
        self.feed_forward = _FeedForward(dimension, dropout)
        self.scale_shift = nn.Parameter(
            torch.randn(1, 1, 6, dimension) / dimension**0.5
        )

    def _attention(self, value: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = value.shape
        shape = (batch, sequence, self.num_heads, self.head_dim)
        query = self.query(value).reshape(shape).transpose(1, 2)
        key = self.key(value).reshape(shape).transpose(1, 2)
        projected_value = self.value(value).reshape(shape).transpose(1, 2)
        attended = functional.scaled_dot_product_attention(
            query, key, projected_value, is_causal=False
        )
        attended = attended.transpose(1, 2).reshape(batch, sequence, self.dimension)
        return self.attention_dropout(self.attention_output(attended))

    def forward(self, value: torch.Tensor, modulation: torch.Tensor) -> torch.Tensor:
        batch = value.shape[0]
        (
            shift_attention,
            scale_attention,
            gate_attention,
            shift_ff,
            scale_ff,
            gate_ff,
        ) = (self.scale_shift + modulation.reshape(batch, 1, 6, -1)).chunk(6, dim=-2)
        hidden = self.norm_attention(value)
        hidden = hidden * (1.0 + scale_attention.squeeze(-2)) + shift_attention.squeeze(
            -2
        )
        value = value + gate_attention.squeeze(-2) * self._attention(hidden)
        hidden = self.norm_feed_forward(value)
        hidden = hidden * (1.0 + scale_ff.squeeze(-2)) + shift_ff.squeeze(-2)
        return value + gate_ff.squeeze(-2) * self.feed_forward(hidden)


class DiffusionDenoiser(nn.Module):
    """SMP reference architecture parameterized for the 39-D Go2 layout."""

    def __init__(
        self,
        feature_dim: int,
        window_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        if d_model % nhead:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")
        self.feature_dim = feature_dim
        self.window_size = window_size
        self.preprocess = nn.Conv1d(feature_dim, feature_dim, 1, bias=False)
        self.input_projection = nn.Linear(feature_dim, d_model, bias=False)
        self.timestep_modulation = _AdaLayerNormSingle(d_model)
        self.position = _PositionalEmbedding(d_model, max(window_size, 32))
        self.blocks = nn.ModuleList(
            [_DiTBlock(d_model, nhead, dropout) for _ in range(num_layers)]
        )
        self.output_projection = nn.Linear(d_model, feature_dim, bias=False)
        self.postprocess = nn.Conv1d(feature_dim, feature_dim, 1, bias=False)

    def forward(self, noisy: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        if noisy.ndim != 3 or noisy.shape[1:] != (self.window_size, self.feature_dim):
            raise ValueError(
                f"expected [batch,{self.window_size},{self.feature_dim}], got {tuple(noisy.shape)}"
            )
        hidden = noisy.transpose(1, 2)
        hidden = self.preprocess(hidden) + hidden
        hidden = self.input_projection(hidden.transpose(1, 2))
        modulation = self.timestep_modulation(timestep)
        hidden = self.position(hidden)
        for block in self.blocks:
            hidden = block(hidden, modulation)
        output = self.output_projection(hidden).transpose(1, 2)
        output = self.postprocess(output) + output
        return output.transpose(1, 2)
