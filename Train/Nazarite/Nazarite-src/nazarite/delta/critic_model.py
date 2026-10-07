"""Privileged DELTA value model for asymmetric actor-critic training."""

from __future__ import annotations

import torch
from rsl_rl.modules import MLP
from tensordict import TensorDict
from torch import nn

from .encoder import DeltaEncoder


class DeltaPrivilegedCriticModel(nn.Module):
    """Encode a clean terrain map for value estimation only.

    This is asymmetric PPO, not teacher-student training: the privileged map is
    consumed only by the critic while the deployed actor remains camera-only.
    """

    is_recurrent = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=False,
        distribution_cfg=None,
        delta_map_group="delta_privileged_map",
        delta_proprio_group=None,
        map_height=16,
        map_width=26,
        map_channels=5,
        map_extent=(2.0, 1.2),
        map_center=(1.0, 0.0),
        forward_x_threshold=0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        del distribution_cfg, kwargs
        if obs_normalization:
            raise NotImplementedError("Use term-aware normalization for DELTA critic")
        groups = list(obs_groups[obs_set])
        if delta_map_group not in groups:
            raise ValueError(
                "DELTA critic requires a privileged map group"
            )
        self.obs_groups = groups
        self.delta_map_group = delta_map_group
        self.delta_proprio_group = delta_proprio_group or next(
            group for group in groups if group != delta_map_group
        )
        self.base_groups = [group for group in groups if group != delta_map_group]
        self.map_height = int(map_height)
        self.map_width = int(map_width)
        self.map_channels = int(map_channels)
        expected_map_dim = self.map_height * self.map_width * self.map_channels
        if obs[delta_map_group].shape[-1] != expected_map_dim:
            raise ValueError(
                f"Expected {delta_map_group} dimension {expected_map_dim}, "
                f"got {obs[delta_map_group].shape[-1]}"
            )
        proprio_dim = int(obs[self.delta_proprio_group].shape[-1])
        base_dim = sum(int(obs[group].shape[-1]) for group in self.base_groups)
        self.delta = DeltaEncoder(
            proprio_dim=proprio_dim,
            dim=64,
            layers=3,
            heads=4,
            samples=8,
            map_channels=self.map_channels,
            map_extent=(float(map_extent[0]), float(map_extent[1])),
            map_center=(float(map_center[0]), float(map_center[1])),
            forward_x_threshold=float(forward_x_threshold),
        )
        self.value_head = MLP(base_dim + 64, output_dim, hidden_dims, activation)
        self.obs_normalizer = nn.Identity()

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        del masks, hidden_state, stochastic_output
        proprio = obs[self.delta_proprio_group]
        terrain = obs[self.delta_map_group].reshape(
            proprio.shape[0], self.map_height, self.map_width, self.map_channels
        )
        terrain_latent = self.delta(proprio, terrain)
        base = torch.cat([obs[group] for group in self.base_groups], dim=-1)
        return self.value_head(torch.cat((base, terrain_latent), dim=-1))

    def reset(self, dones=None, hidden_state=None) -> None:
        del dones, hidden_state

    def get_hidden_state(self):
        return None

    def detach_hidden_state(self, dones=None) -> None:
        del dones

    def update_normalization(self, obs) -> None:
        del obs
