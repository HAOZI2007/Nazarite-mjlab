"""Load, score, and sample a frozen Go2 SMP diffusion prior."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .features import GO2_SMP_FEATURE_DIM
from .model import DiffusionDenoiser
from .scheduler import DDPMScheduler


def load_prior(
    checkpoint_path: str,
    device: torch.device | str,
) -> tuple[DiffusionDenoiser, DDPMScheduler, torch.Tensor, torch.Tensor, int, int]:
    """Load and freeze a Nazarite SMP checkpoint."""
    target_device = torch.device(device)
    checkpoint: dict[str, Any] = torch.load(
        checkpoint_path, map_location=target_device, weights_only=False
    )
    config = checkpoint["cfg"]
    feature_dim = int(config["feature_dim"])
    window_size = int(config["window_size"])
    if feature_dim != GO2_SMP_FEATURE_DIM:
        raise ValueError(
            f"checkpoint feature_dim={feature_dim}, expected Go2 {GO2_SMP_FEATURE_DIM}"
        )
    model = DiffusionDenoiser(
        feature_dim=feature_dim,
        window_size=window_size,
        d_model=int(config.get("d_model", 256)),
        nhead=int(config.get("nhead", 4)),
        num_layers=int(config.get("num_layers", 2)),
        dropout=float(config.get("dropout", 0.0)),
    ).to(target_device)
    state = checkpoint.get("model_ema", checkpoint["model"])
    model.load_state_dict(state)
    model.eval().requires_grad_(False)
    scheduler = DDPMScheduler(int(config.get("num_timesteps", 50))).to(target_device)
    q_low = torch.from_numpy(np.asarray(checkpoint["q_low"], dtype=np.float32)).to(
        target_device
    )
    q_high = torch.from_numpy(np.asarray(checkpoint["q_high"], dtype=np.float32)).to(
        target_device
    )
    return model, scheduler, q_low, q_high, feature_dim, window_size


@torch.no_grad()
def sample_prior(
    model: DiffusionDenoiser,
    scheduler: DDPMScheduler,
    q_low: torch.Tensor,
    q_high: torch.Tensor,
    count: int,
) -> torch.Tensor:
    """Generate denormalized ``[count,W,39]`` windows by ancestral DDPM."""
    if count <= 0:
        raise ValueError("sample count must be positive")
    noisy = torch.randn(
        count,
        model.window_size,
        model.feature_dim,
        device=q_low.device,
    )
    for timestep in reversed(range(scheduler.num_timesteps)):
        time_batch = torch.full(
            (count,), timestep, dtype=torch.long, device=q_low.device
        )
        predicted_noise = model(noisy, time_batch)
        noisy = scheduler.step(predicted_noise, noisy, timestep)
    return (noisy + 1.0) * 0.5 * (q_high - q_low) + q_low


class DiffusionErrorNormalizer:
    """Count-weighted per-timestep reference for online denoising MSE."""

    def __init__(self, num_timesteps: int, device: torch.device | str):
        self.mean = torch.ones(num_timesteps, device=device)
        self.count = torch.zeros(num_timesteps, dtype=torch.long, device=device)

    def update_and_normalize(self, timestep: int, errors: torch.Tensor) -> torch.Tensor:
        batch_count = errors.numel()
        old_count = int(self.count[timestep].item())
        new_count = old_count + batch_count
        batch_mean = errors.mean()
        if old_count == 0:
            self.mean[timestep] = batch_mean
        else:
            self.mean[timestep] = (
                old_count * self.mean[timestep] + batch_count * batch_mean
            ) / new_count
        self.count[timestep] = new_count
        return errors / self.mean[timestep].clamp_min(1.0e-4)
