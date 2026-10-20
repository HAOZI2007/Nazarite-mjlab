"""HIM actor augmented with a compact latent from the visual world model."""

from __future__ import annotations

import torch
from rsl_rl.models import HIMActorModel
from rsl_rl.modules import MLP
from tensordict import TensorDict
from torch import nn


class WMPHIMActorModel(HIMActorModel):
    """Keep HIM history estimation and add a learned visual-latent projection."""

    def __init__(
        self,
        *args,
        world_model_feature_group: str = "world_model_feature",
        world_model_feature_dim: int = 512,
        visual_latent_dim: int = 32,
        visual_hidden_dim: int = 128,
        **kwargs,
    ) -> None:
        hidden_dims = kwargs.get("hidden_dims", (512, 256, 128))
        activation = kwargs.get("activation", "elu")
        super().__init__(*args, **kwargs)
        self.world_model_feature_group = world_model_feature_group
        self.world_model_feature_dim = int(world_model_feature_dim)
        self.visual_latent_dim = int(visual_latent_dim)
        self.visual_encoder = nn.Sequential(
            nn.Linear(self.world_model_feature_dim, int(visual_hidden_dim)),
            nn.ELU(),
            nn.Linear(int(visual_hidden_dim), self.visual_latent_dim),
            nn.Tanh(),
        )
        self.mlp = MLP(
            self.num_one_step_obs
            + 3
            + self.estimator.num_latent
            + self.visual_latent_dim,
            self.distribution.input_dim,
            hidden_dims,
            activation,
        )
        self.distribution.init_mlp_weights(self.mlp)

    def get_latent(
        self, obs: TensorDict, masks=None, hidden_state=None
    ) -> torch.Tensor:
        him_latent = super().get_latent(obs, masks, hidden_state)
        if self.world_model_feature_group not in obs:
            raise KeyError(
                f"Missing World Model observation group {self.world_model_feature_group!r}"
            )
        feature = obs[self.world_model_feature_group]
        if feature.shape[-1] != self.world_model_feature_dim:
            raise ValueError(
                f"Expected {self.world_model_feature_dim}-D World Model state, got "
                f"{feature.shape[-1]}"
            )
        return torch.cat((him_latent, self.visual_encoder(feature)), dim=-1)
