"""XZ1/Go2 I/O adapter around the World Model shipped by wmp_go1-main.

The network, distributions, recurrent state, training objective and optimizer
live in reference_dreamer. This module translates our replay format and exposes
the reference's deterministic 512-D state to the existing HIM runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import torch
from torch import nn

from .reference_dreamer.models import WorldModel

REFERENCE_IMPLEMENTATION = "wmp_go1_dreamer_v1"


@dataclass(frozen=True)
class WMPWorldModelCfg:
    image_height: int = 64
    image_width: int = 64
    image_channels: int = 1
    prop_dim: int = 33
    action_dim: int = 12
    action_chunk: int = 1
    """Policy commands per image; XZ1 uses 5 x 12 = 60 inputs at 10 Hz."""

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
    # Defaults from reference_dreamer/configs.yaml, including the reward head
    # even though its loss is disabled by the reference's default configuration.
    unimix_ratio: float = 0.01
    initial: str = "learned"
    head_units: int = 512
    reward_layers: int = 2
    opt_eps: float = 1.0e-8
    weight_decay: float = 0.0

    @property
    def flat_action_dim(self) -> int:
        return self.action_dim * self.action_chunk

    def reference_config(self, device: str | torch.device) -> SimpleNamespace:
        """Map local dimension names to the reference YAML without new knobs."""
        common = {
            "mlp_keys": ".*",
            "cnn_keys": "image",
            "act": "SiLU",
            "norm": True,
            "cnn_depth": self.cnn_depth,
            "kernel_size": 4,
            "minres": 4,
            "mlp_layers": self.prop_layers,
            "mlp_units": self.prop_hidden_dim,
        }
        return SimpleNamespace(
            device=str(device),
            precision=32,
            dyn_stoch=self.stoch_groups,
            dyn_deter=self.deter_dim,
            dyn_hidden=self.rssm_hidden_dim,
            dyn_rec_depth=1,
            dyn_discrete=self.stoch_classes,
            dyn_mean_act="none",
            dyn_std_act="sigmoid2",
            dyn_min_std=0.1,
            unimix_ratio=self.unimix_ratio,
            initial=self.initial,
            num_actions=self.flat_action_dim,
            act="SiLU",
            norm=True,
            encoder={**common, "symlog_inputs": True},
            decoder={
                **common,
                "cnn_sigmoid": False,
                "image_dist": "mse",
                "vector_dist": "symlog_mse",
                "outscale": 1.0,
            },
            reward_head={
                "layers": self.reward_layers,
                "dist": "symlog_disc",
                "loss_scale": self.reward_loss_scale,
                "outscale": 0.0,
            },
            grad_heads=["decoder", "reward"],
            units=self.head_units,
            kl_free=self.kl_free,
            dyn_scale=self.dyn_scale,
            rep_scale=self.rep_scale,
            model_lr=self.learning_rate,
            opt_eps=self.opt_eps,
            grad_clip=self.grad_clip,
            weight_decay=self.weight_decay,
            opt="adam",
        )


class WMPWorldModel(nn.Module):
    """Reference WorldModel with replay conversion and read-only validation."""

    implementation = REFERENCE_IMPLEMENTATION

    def __init__(
        self,
        cfg: WMPWorldModelCfg | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.cfg = cfg or WMPWorldModelCfg()
        if min(self.cfg.flat_action_dim, self.cfg.prop_dim) < 1:
            raise ValueError("World Model input dimensions must be positive")
        self.core = WorldModel(
            self.cfg.reference_config(device),
            {
                "prop": (self.cfg.prop_dim,),
                "image": (
                    self.cfg.image_height,
                    self.cfg.image_width,
                    self.cfg.image_channels,
                ),
            },
            use_camera=True,
        ).to(device)
        self.feature_dim = self.cfg.deter_dim
        self.embed_dim = self.core.embed_size

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse)
        # Upstream keeps device strings outside parameters/buffers. Keep them
        # aligned when callers use model.to(...), including after loading on CPU.
        device = str(next(self.parameters()).device)
        self.core.device = device
        self.core._config.device = device
        self.core.dynamics._device = device
        cast(Any, self.core.heads["reward"])._device = device
        return result

    @property
    def optimizer(self):
        return self.core._model_opt._opt

    @property
    def dynamics(self):
        return self.core.dynamics

    @staticmethod
    def normalize_depth(depth: torch.Tensor, far_clip: float = 2.0) -> torch.Tensor:
        depth = torch.nan_to_num(depth, nan=far_clip, posinf=far_clip, neginf=0.0)
        return depth.clamp(0.0, far_clip) / far_clip - 0.5

    def encode(self, image: torch.Tensor, prop: torch.Tensor) -> torch.Tensor:
        return self.core.encoder({"image": image, "prop": prop})

    def observe_step(self, image, prop, action, is_first, state):
        if action.shape != (image.shape[0], self.cfg.flat_action_dim):
            raise RuntimeError("World Model needs one folded camera-interval action")
        # RSSM.obs_step mutates its inputs during partial resets. Keep replay and
        # the caller's preceding posterior intact, while retaining native math.
        previous = None if state is None else dict(state)
        post, _ = self.dynamics.obs_step(
            previous, action.clone(), self.encode(image, prop), is_first.float()
        )
        return post

    def decode_image(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        """Read the reference decoder's image mode, including one-step states."""
        feature = self.dynamics.get_feat(state)
        single = feature.ndim == 2
        if single:
            feature = feature.unsqueeze(1)
        image = self.core.heads["decoder"](feature)["image"].mode()
        return image[:, 0] if single else image

    def _reference_batch(self, batch):
        device = next(self.parameters()).device
        return {
            "image": batch["image"].to(device).float() / 255.0 - 0.5,
            "prop": batch["prop"].to(device).float(),
            "action": batch["action"].to(device).float().clone(),
            "reward": batch["reward"].to(device).float(),
            # Reference runner uses floating first flags, not bool subtraction.
            "is_first": batch["is_first"].to(device).float(),
        }

    def compute_loss(self, batch: dict[str, torch.Tensor]):
        """Read-only equivalent of upstream _train; parity-tested against it.

        Training itself calls the unmodified upstream _train below. Validation
        repeats its forward objective because upstream has no loss-only entry.
        """
        data = self.core.preprocess(self._reference_batch(batch))
        embed = self.core.encoder(data)
        post, prior = self.dynamics.observe(embed, data["action"], data["is_first"])
        kl_loss, kl, dyn_loss, rep_loss = self.dynamics.kl_loss(
            post, prior, self.cfg.kl_free, self.cfg.dyn_scale, self.cfg.rep_scale
        )
        predictions = {}
        for name, head in self.core.heads.items():
            feature = self.dynamics.get_feat(post)
            if name not in self.core._config.grad_heads:
                feature = feature.detach()
            prediction = head(feature)
            predictions.update(
                prediction if isinstance(prediction, dict) else {name: prediction}
            )
        losses = {
            name: -prediction.log_prob(data[name])
            for name, prediction in predictions.items()
        }
        scaled = {
            name: loss * self.core._scales.get(name, 1.0)
            for name, loss in losses.items()
        }
        loss = (sum(scaled.values()) + kl_loss).mean()
        return loss, {
            "loss": loss,
            **{f"{name}_loss": value.mean() for name, value in losses.items()},
            "kl": kl.mean(),
            "dyn_loss": dyn_loss.mean(),
            "rep_loss": rep_loss.mean(),
            "kl_dyn_rep": (dyn_loss + rep_loss).mean(),
            "prior_ent": self.dynamics.get_dist(prior).entropy().mean(),
            "post_ent": self.dynamics.get_dist(post).entropy().mean(),
        }

    def train_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        data = self._reference_batch(batch)
        if not all(torch.isfinite(value).all() for value in data.values()):
            raise RuntimeError("Non-finite World Model input; update rejected")
        _, _, raw = self.core._train(data)
        metrics = {
            key: float(torch.as_tensor(value).float().mean())
            for key, value in raw.items()
        }
        metrics["loss"] = metrics.pop("model_loss")
        metrics["grad_norm"] = metrics.pop("model_grad_norm")
        metrics["kl_dyn_rep"] = metrics["dyn_loss"] + metrics["rep_loss"]
        return metrics
