"""Compact Dreamer-style visual World Model adapted from the WMP reference.

This module preserves the reference's 64x64 depth CNN, proprioceptive encoder,
discrete RSSM, image decoder, and dynamics/representation KL objectives. AMP
and expert-motion components are intentionally absent.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class WMPWorldModelCfg:
    image_height: int = 64
    image_width: int = 64
    image_channels: int = 1
    prop_dim: int = 33
    action_dim: int = 12
    action_chunk: int = 1
    """Policy steps folded into one world-model step.

    The reference implementation advances the world model at the depth-camera
    rate and feeds it the actions taken since the last frame, so a 5-frame
    interval stacks 5x12 actions into one 60-dim vector.  Folding a single
    action while holding the image for five steps makes the dynamics model
    predict five identical transitions, which is what the 2026-10-17 runs did.
    """

    cnn_depth: int = 32
    prop_hidden_dim: int = 1024
    prop_layers: int = 5
    deter_dim: int = 512
    stoch_groups: int = 32
    stoch_classes: int = 32
    rssm_hidden_dim: int = 512
    kl_free: float = 1.0
    dyn_scale: float = 0.5
    rep_scale: float = 0.1
    learning_rate: float = 1.0e-4
    grad_clip: float = 1000.0
    reward_loss_scale: float = 0.0

    @property
    def flat_action_dim(self) -> int:
        """Action width the RSSM consumes: one policy action per folded step."""
        return self.action_dim * self.action_chunk


class DepthEncoder(nn.Module):
    def __init__(self, cfg: WMPWorldModelCfg) -> None:
        super().__init__()
        channels = (
            cfg.cnn_depth,
            cfg.cnn_depth * 2,
            cfg.cnn_depth * 4,
            cfg.cnn_depth * 8,
        )
        layers: list[nn.Module] = []
        in_channels = cfg.image_channels
        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.GroupNorm(1, out_channels),
                    nn.SiLU(),
                ]
            )
            in_channels = out_channels
        self.layers = nn.Sequential(*layers)
        self.output_dim = (
            channels[-1] * (cfg.image_height // 16) * (cfg.image_width // 16)
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        leading = image.shape[:-3]
        image = image.reshape(-1, *image.shape[-3:]).permute(0, 3, 1, 2)
        # Input is normalized to [-0.5, 0.5]; the reference CNN subtracts 0.5.
        image = self.layers(image - 0.5)
        return image.flatten(1).reshape(*leading, -1)


class PropEncoder(nn.Module):
    def __init__(self, cfg: WMPWorldModelCfg) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = cfg.prop_dim
        for _ in range(cfg.prop_layers):
            layers.extend(
                [
                    nn.Linear(in_dim, cfg.prop_hidden_dim),
                    nn.LayerNorm(cfg.prop_hidden_dim),
                    nn.SiLU(),
                ]
            )
            in_dim = cfg.prop_hidden_dim
        self.layers = nn.Sequential(*layers)

    def forward(self, prop: torch.Tensor) -> torch.Tensor:
        prop = torch.sign(prop) * torch.log1p(prop.abs())
        return self.layers(prop)


class DepthDecoder(nn.Module):
    def __init__(self, cfg: WMPWorldModelCfg, feature_dim: int) -> None:
        super().__init__()
        self.height = cfg.image_height
        self.width = cfg.image_width
        self.seed_channels = cfg.cnn_depth * 8
        self.seed_height = cfg.image_height // 16
        self.seed_width = cfg.image_width // 16
        self.project = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, self.seed_channels * self.seed_height * self.seed_width),
            nn.SiLU(),
        )
        channel_sizes = (
            cfg.cnn_depth * 4,
            cfg.cnn_depth * 2,
            cfg.cnn_depth,
            cfg.image_channels,
        )
        layers: list[nn.Module] = []
        in_channels = self.seed_channels
        for index, out_channels in enumerate(channel_sizes):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size=4, stride=2, padding=1
                )
            )
            if index < len(channel_sizes) - 1:
                layers.extend([nn.GroupNorm(1, out_channels), nn.SiLU()])
            in_channels = out_channels
        self.layers = nn.Sequential(*layers)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        leading = feature.shape[:-1]
        flat = feature.reshape(-1, feature.shape[-1])
        seed = self.project(flat).reshape(
            -1, self.seed_channels, self.seed_height, self.seed_width
        )
        image = self.layers(seed).permute(0, 2, 3, 1)
        return image.reshape(*leading, self.height, self.width, 1)


class DiscreteRSSM(nn.Module):
    def __init__(self, cfg: WMPWorldModelCfg, embed_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        flat_stoch = cfg.stoch_groups * cfg.stoch_classes
        self.input_layer = nn.Sequential(
            nn.Linear(flat_stoch + cfg.flat_action_dim, cfg.rssm_hidden_dim),
            nn.LayerNorm(cfg.rssm_hidden_dim),
            nn.SiLU(),
        )
        self.gru = nn.Sequential(
            nn.Linear(
                cfg.rssm_hidden_dim + cfg.deter_dim,
                3 * cfg.deter_dim,
                bias=False,
            ),
            nn.LayerNorm(3 * cfg.deter_dim, eps=1.0e-3),
        )
        self.prior = nn.Sequential(
            nn.Linear(cfg.deter_dim, cfg.rssm_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.rssm_hidden_dim, cfg.stoch_groups * cfg.stoch_classes),
        )
        self.posterior = nn.Sequential(
            nn.Linear(cfg.deter_dim + embed_dim, cfg.rssm_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.rssm_hidden_dim, cfg.stoch_groups * cfg.stoch_classes),
        )

    def initial(self, batch: int, device: torch.device) -> dict[str, torch.Tensor]:
        shape = (batch, self.cfg.stoch_groups, self.cfg.stoch_classes)
        logits = torch.zeros(shape, device=device)
        return {
            "deter": torch.zeros(batch, self.cfg.deter_dim, device=device),
            "stoch": torch.softmax(logits, dim=-1),
            "logits": logits,
        }

    def _sample(self, logits: torch.Tensor, deterministic: bool) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        if deterministic:
            indices = probs.argmax(dim=-1)
        else:
            indices = torch.distributions.Categorical(probs=probs).sample()
        hard = F.one_hot(indices, self.cfg.stoch_classes).to(probs.dtype)
        return probs + (hard - probs).detach()

    def img_step(
        self, state: dict[str, torch.Tensor], action: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        stoch = state["stoch"].flatten(start_dim=-2)
        hidden = self.input_layer(torch.cat((stoch, action), dim=-1))
        reset, candidate, update = self.gru(
            torch.cat((hidden, state["deter"]), dim=-1)
        ).chunk(3, dim=-1)
        reset = torch.sigmoid(reset)
        candidate = torch.tanh(reset * candidate)
        update = torch.sigmoid(update - 1.0)
        deter = update * candidate + (1.0 - update) * state["deter"]
        logits = self.prior(deter).reshape(
            -1, self.cfg.stoch_groups, self.cfg.stoch_classes
        )
        return {
            "deter": deter,
            "stoch": self._sample(logits, deterministic=not self.training),
            "logits": logits,
        }

    def obs_step(
        self,
        state: dict[str, torch.Tensor] | None,
        action: torch.Tensor,
        embed: torch.Tensor,
        is_first: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if state is None:
            state = self.initial(embed.shape[0], embed.device)
        init = self.initial(embed.shape[0], embed.device)
        reset = is_first.reshape(-1, 1).to(embed.dtype)
        state = {
            key: value * (1.0 - reset.reshape(reset.shape + (1,) * (value.ndim - 2)))
            + init[key] * reset.reshape(reset.shape + (1,) * (value.ndim - 2))
            for key, value in state.items()
        }
        prior = self.img_step(state, action * (1.0 - reset))
        logits = self.posterior(torch.cat((prior["deter"], embed), dim=-1)).reshape(
            -1, self.cfg.stoch_groups, self.cfg.stoch_classes
        )
        post = {
            "deter": prior["deter"],
            "stoch": self._sample(logits, deterministic=not self.training),
            "logits": logits,
        }
        return post, prior

    def observe(
        self, embed: torch.Tensor, action: torch.Tensor, is_first: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        state = None
        posts: dict[str, list[torch.Tensor]] = {
            key: [] for key in ("deter", "stoch", "logits")
        }
        priors: dict[str, list[torch.Tensor]] = {
            key: [] for key in ("deter", "stoch", "logits")
        }
        for step in range(embed.shape[1]):
            post, prior = self.obs_step(
                state, action[:, step], embed[:, step], is_first[:, step]
            )
            state = post
            for key, post_values in posts.items():
                post_values.append(post[key])
                priors[key].append(prior[key])
        return (
            {key: torch.stack(values, dim=1) for key, values in posts.items()},
            {key: torch.stack(values, dim=1) for key, values in priors.items()},
        )

    @staticmethod
    def feature(state: dict[str, torch.Tensor]) -> torch.Tensor:
        return state["deter"]

    def kl_loss(
        self, post: dict[str, torch.Tensor], prior: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        post_log = torch.log_softmax(post["logits"], dim=-1)
        prior_log = torch.log_softmax(prior["logits"], dim=-1)
        post_prob = torch.softmax(post["logits"], dim=-1)
        dyn = (post_prob.detach() * (post_log.detach() - prior_log)).sum(dim=(-1, -2))
        rep = (post_prob * (post_log - prior_log.detach())).sum(dim=(-1, -2))
        value = (post_prob.detach() * (post_log.detach() - prior_log.detach())).sum(
            dim=(-1, -2)
        )
        dyn = dyn.clamp_min(self.cfg.kl_free)
        rep = rep.clamp_min(self.cfg.kl_free)
        return self.cfg.dyn_scale * dyn + self.cfg.rep_scale * rep, value, dyn + rep


class WMPWorldModel(nn.Module):
    """Depth-plus-proprioception latent dynamics and image reconstruction model."""

    def __init__(self, cfg: WMPWorldModelCfg | None = None) -> None:
        super().__init__()
        self.cfg = cfg or WMPWorldModelCfg()
        self.image_encoder = DepthEncoder(self.cfg)
        self.prop_encoder = PropEncoder(self.cfg)
        self.embed_dim = self.image_encoder.output_dim + self.cfg.prop_hidden_dim
        self.dynamics = DiscreteRSSM(self.cfg, self.embed_dim)
        self.feature_dim = self.cfg.deter_dim
        feature_dim = (
            self.cfg.deter_dim + self.cfg.stoch_groups * self.cfg.stoch_classes
        )
        self.decoder = DepthDecoder(self.cfg, feature_dim)
        self.reward_head = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.SiLU(), nn.Linear(256, 1)
        )
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.cfg.learning_rate, eps=1e-8
        )

    @staticmethod
    def normalize_depth(depth: torch.Tensor, far_clip: float = 2.0) -> torch.Tensor:
        depth = torch.nan_to_num(depth, nan=far_clip, posinf=far_clip, neginf=0.0)
        return depth.clamp(0.0, far_clip) / far_clip - 0.5

    def encode(self, image: torch.Tensor, prop: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.image_encoder(image), self.prop_encoder(prop)), dim=-1)

    def observe_step(
        self,
        image: torch.Tensor,
        prop: torch.Tensor,
        action: torch.Tensor,
        is_first: torch.Tensor,
        state: dict[str, torch.Tensor] | None,
    ) -> dict[str, torch.Tensor]:
        embed = self.encode(image, prop)
        posterior, _ = self.dynamics.obs_step(state, action, embed, is_first)
        return posterior

    def train_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        image = batch["image"].float() / 255.0 - 0.5
        prop = batch["prop"].float()
        action = batch["action"].float()
        reward = batch["reward"].float()
        first = batch["is_first"].bool()
        embed = self.encode(image, prop)
        post, prior = self.dynamics.observe(embed, action, first)
        feature = torch.cat(
            (post["stoch"].flatten(start_dim=-2), post["deter"]), dim=-1
        )
        reconstruction = self.decoder(feature)
        image_loss = F.mse_loss(reconstruction, image)
        kl, kl_value, kl_parts = self.dynamics.kl_loss(post, prior)
        reward_prediction = self.reward_head(feature).squeeze(-1)
        reward_loss = F.mse_loss(reward_prediction, reward)
        loss = image_loss + kl.mean() + self.cfg.reward_loss_scale * reward_loss
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(), self.cfg.grad_clip
        )
        self.optimizer.step()
        return {
            "loss": float(loss.detach()),
            "image_loss": float(image_loss.detach()),
            "reward_loss": float(reward_loss.detach()),
            "kl": float(kl_value.mean().detach()),
            "kl_dyn_rep": float(kl_parts.mean().detach()),
            "grad_norm": float(torch.as_tensor(grad_norm).detach()),
        }
