# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Heterogeneous information mixing (HIM) state estimator."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rsl_rl.utils import resolve_nn_activation


class HIMEstimator(nn.Module):
    """Estimate base velocity and a contrastive latent from observation history.

    The estimator follows the HIMLoco objective: an encoder predicts velocity and a
    normalized latent, while a target network and a bank of prototypes provide the
    swapped-assignment loss.  ``forward`` is deliberately detached because the
    estimator is optimized by its own optimizer in :class:`HIMPPO`.
    """

    def __init__(
        self,
        temporal_steps: int,
        num_one_step_obs: int,
        enc_hidden_dims: tuple[int, ...] | list[int] = (128, 64, 16),
        tar_hidden_dims: tuple[int, ...] | list[int] = (128, 64),
        activation: str = "elu",
        learning_rate: float = 1e-3,
        max_grad_norm: float = 10.0,
        num_prototype: int = 32,
        temperature: float = 3.0,
        sinkhorn_eps: float = 0.05,
        sinkhorn_iters: int = 3,
        device: str | None = None,
        **kwargs,
    ) -> None:
        """Initialize the estimator and its independent optimizer."""
        super().__init__()
        del kwargs
        if temporal_steps < 1 or num_one_step_obs < 1:
            raise ValueError("temporal_steps and num_one_step_obs must be positive")
        if len(enc_hidden_dims) < 1 or len(tar_hidden_dims) < 1:
            raise ValueError("enc_hidden_dims and tar_hidden_dims must not be empty")
        if num_prototype < 1 or temperature <= 0 or sinkhorn_eps <= 0 or sinkhorn_iters < 1:
            raise ValueError("num_prototype must be positive and Sinkhorn/temperature parameters must be valid")

        self.temporal_steps = int(temporal_steps)
        self.num_one_step_obs = int(num_one_step_obs)
        self.num_latent = int(enc_hidden_dims[-1])
        self.max_grad_norm = float(max_grad_norm)
        self.temperature = float(temperature)
        self.sinkhorn_eps = float(sinkhorn_eps)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.learning_rate = float(learning_rate)

        # Encoder: history -> velocity (3) + latent.
        enc_input_dim = self.temporal_steps * self.num_one_step_obs
        enc_layers: list[nn.Module] = []
        for hidden_dim in enc_hidden_dims[:-1]:
            enc_layers.extend([nn.Linear(enc_input_dim, hidden_dim), resolve_nn_activation(activation)])
            enc_input_dim = hidden_dim
        enc_layers.append(nn.Linear(enc_input_dim, self.num_latent + 3))
        self.encoder = nn.Sequential(*enc_layers)

        # Target: one-step observation -> latent.
        tar_input_dim = self.num_one_step_obs
        tar_layers: list[nn.Module] = []
        for hidden_dim in tar_hidden_dims:
            tar_layers.extend([nn.Linear(tar_input_dim, hidden_dim), resolve_nn_activation(activation)])
            tar_input_dim = hidden_dim
        tar_layers.append(nn.Linear(tar_input_dim, self.num_latent))
        self.target = nn.Sequential(*tar_layers)

        self.proto = nn.Embedding(num_prototype, self.num_latent)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        if device is not None:
            self.to(device)

    def encode(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode history and return velocity plus a normalized latent."""
        self._validate_history(obs_history)
        parts = self.encoder(obs_history.detach())
        velocity, latent = parts[..., :3], parts[..., 3:]
        return velocity, F.normalize(latent, dim=-1, p=2)

    def get_latent(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return detached estimator outputs for policy inference."""
        velocity, latent = self.encode(obs_history)
        return velocity.detach(), latent.detach()

    def forward(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return detached velocity and latent predictions."""
        return self.get_latent(obs_history)

    def update(
        self,
        obs_history: torch.Tensor,
        velocity_target: torch.Tensor | None = None,
        target_obs: torch.Tensor | None = None,
        *,
        next_critic_obs: torch.Tensor | None = None,
        velocity_slice: tuple[int, int] | None = None,
        target_slice: tuple[int, int] | None = None,
        target_slices: tuple[tuple[int, int], ...] | list[tuple[int, int]] | None = None,
        learning_rate: float | None = None,
        gradient_reducer: Callable[[Iterable[nn.Parameter]], None] | None = None,
        numerics_guard=None,
        lr: float | None = None,
    ) -> tuple[float, float]:
        """Perform one estimator update and return estimation and swap losses."""
        self._validate_history(obs_history)
        if learning_rate is None:
            learning_rate = lr
        if learning_rate is not None:
            self.learning_rate = float(learning_rate)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

        # Preserve the pre-explicit-target API: update(history, critic_obs, ...).
        if next_critic_obs is None and target_obs is None and velocity_target is not None:
            if velocity_target.ndim == 2 and velocity_target.shape[-1] != 3:
                next_critic_obs = velocity_target
                velocity_target = None

        if velocity_target is None or target_obs is None:
            if next_critic_obs is None:
                raise ValueError("Either explicit velocity_target/target_obs or next_critic_obs must be provided")
            if next_critic_obs.ndim != 2:
                raise ValueError(f"next_critic_obs must be a 2D tensor, got {next_critic_obs.shape}")
            velocity_slice = velocity_slice or (self.num_one_step_obs, self.num_one_step_obs + 3)
            target_slices = _normalize_target_slices(target_slice, target_slices, self.num_one_step_obs)
            velocity_target = _slice_range(next_critic_obs, velocity_slice).detach()
            target_obs = torch.cat([_slice_range(next_critic_obs, slc) for slc in target_slices], dim=-1).detach()
        else:
            velocity_target = velocity_target.detach()
            target_obs = target_obs.detach()

        if velocity_target.ndim != 2 or target_obs.ndim != 2:
            raise ValueError(
                f"HIM estimator targets must be 2D tensors, got {velocity_target.shape} and {target_obs.shape}"
            )
        if velocity_target.shape[-1] != 3 or target_obs.shape[-1] != self.num_one_step_obs:
            raise ValueError(
                "Invalid HIM estimator slices: expected velocity width 3 and target width "
                f"{self.num_one_step_obs}, got {velocity_target.shape[-1]} and {target_obs.shape[-1]}"
            )

        if numerics_guard is not None:
            numerics_guard.check("estimator_input", {
                "history": obs_history, "velocity_target": velocity_target, "target_obs": target_obs,
            })

        def optimizer_step(loss):
            if numerics_guard is not None:
                numerics_guard.check("estimator_loss", loss, limit=1e8)
            self.optimizer.zero_grad()
            loss.backward()
            if numerics_guard is not None:
                numerics_guard.gradients("estimator_gradients", self.parameters())
            if gradient_reducer is not None:
                gradient_reducer(self.parameters())
            norm = nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
            if numerics_guard is not None:
                numerics_guard.check("estimator_gradient_norm", norm, limit=1e8)
            self.optimizer.step()
            if numerics_guard is not None:
                numerics_guard.check("estimator_parameters", {
                    "parameters": list(self.parameters()), "optimizer": self.optimizer.state,
                })

        with torch.no_grad():
            normalized_proto = F.normalize(self.proto.weight.data, dim=-1, p=2)
            self.proto.weight.copy_(normalized_proto)

        if obs_history.shape[0] == 0:
            self.optimizer.zero_grad()
            if gradient_reducer is not None:
                zero_loss = torch.zeros((), dtype=obs_history.dtype, device=obs_history.device)
                for parameter in self.parameters():
                    zero_loss = zero_loss + parameter.sum() * 0.0
                optimizer_step(zero_loss)
            return 0.0, 0.0

        encoded = self.encoder(obs_history)
        pred_velocity, source_latent = encoded[..., :3], encoded[..., 3:]
        source_latent = F.normalize(source_latent, dim=-1, p=2)
        target_latent = F.normalize(self.target(target_obs), dim=-1, p=2)

        source_scores = source_latent @ self.proto.weight.T
        target_scores = target_latent @ self.proto.weight.T
        with torch.no_grad():
            source_assignments = sinkhorn(source_scores, self.sinkhorn_eps, self.sinkhorn_iters)
            target_assignments = sinkhorn(target_scores, self.sinkhorn_eps, self.sinkhorn_iters)

        source_log_probs = F.log_softmax(source_scores / self.temperature, dim=-1)
        target_log_probs = F.log_softmax(target_scores / self.temperature, dim=-1)
        swap_loss = -0.5 * (
            source_assignments * target_log_probs
            + target_assignments * source_log_probs
        ).mean()
        estimation_loss = F.mse_loss(pred_velocity, velocity_target)
        loss = estimation_loss + swap_loss

        optimizer_step(loss)
        return estimation_loss.item(), swap_loss.item()

    def _validate_history(self, obs_history: torch.Tensor) -> None:
        """Validate the flattened history shape before passing it to a linear layer."""
        expected_dim = self.temporal_steps * self.num_one_step_obs
        if obs_history.ndim != 2 or obs_history.shape[-1] != expected_dim:
            raise ValueError(f"Expected observation history shape [batch, {expected_dim}], got {obs_history.shape}")


@torch.no_grad()
def sinkhorn(out: torch.Tensor, eps: float = 0.05, iters: int = 3) -> torch.Tensor:
    """Compute balanced prototype assignments for a batch of logits."""
    if out.ndim != 2:
        raise ValueError(f"Sinkhorn input must be 2D, got {out.shape}")
    if eps <= 0 or iters < 1:
        raise ValueError("Sinkhorn eps must be positive and iters must be at least one")
    logits = out / eps
    logits = logits - logits.max()
    assignments = torch.exp(logits).T
    num_prototypes, batch_size = assignments.shape
    assignments /= assignments.sum().clamp_min(torch.finfo(assignments.dtype).eps)
    for _ in range(iters):
        assignments /= assignments.sum(dim=1, keepdim=True).clamp_min(torch.finfo(assignments.dtype).eps)
        assignments /= num_prototypes
        assignments /= assignments.sum(dim=0, keepdim=True).clamp_min(torch.finfo(assignments.dtype).eps)
        assignments /= batch_size
    return (assignments * batch_size).T


def _slice_range(tensor: torch.Tensor, slc: tuple[int, int]) -> torch.Tensor:
    """Return a validated feature slice from a 2D tensor."""
    if len(slc) != 2:
        raise ValueError(f"Slice must be a pair, got {slc}")
    start, stop = int(slc[0]), int(slc[1])
    if start < 0 or stop <= start or stop > tensor.shape[-1]:
        raise ValueError(f"Invalid slice {(start, stop)} for tensor width {tensor.shape[-1]}")
    return tensor[:, start:stop]


def _normalize_target_slices(
    target_slice: tuple[int, int] | None,
    target_slices: tuple[tuple[int, int], ...] | list[tuple[int, int]] | None,
    num_one_step_obs: int,
) -> tuple[tuple[int, int], ...]:
    """Normalize singular/plural target-slice configuration."""
    if target_slices is not None:
        return tuple((int(start), int(stop)) for start, stop in target_slices)
    if target_slice is not None:
        return ((int(target_slice[0]), int(target_slice[1])),)
    return ((3, num_one_step_obs + 3),)
