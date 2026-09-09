"""DDPM scheduler used by Go2 SMP pretraining and online guidance."""

from __future__ import annotations

import math

import torch
from torch import nn


def cosine_betas(num_timesteps: int, max_beta: float = 0.999) -> torch.Tensor:
    """Nichol-Dhariwal cosine schedule, matching the SMP reference project."""
    if num_timesteps <= 0:
        raise ValueError("num_timesteps must be positive")

    def alpha_bar(time: float) -> float:
        return math.cos((time + 0.008) / 1.008 * math.pi / 2.0) ** 2

    betas = []
    for index in range(num_timesteps):
        time_0 = index / num_timesteps
        time_1 = (index + 1) / num_timesteps
        betas.append(min(1.0 - alpha_bar(time_1) / alpha_bar(time_0), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


class DDPMScheduler(nn.Module):
    """Minimal cosine DDPM scheduler with module buffers for device movement."""

    betas: torch.Tensor
    alphas_cumprod: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    sqrt_recip_alphas_cumprod: torch.Tensor
    sqrt_recipm1_alphas_cumprod: torch.Tensor
    posterior_variance: torch.Tensor
    posterior_mean_coef1: torch.Tensor
    posterior_mean_coef2: torch.Tensor

    def __init__(self, num_timesteps: int = 50):
        super().__init__()
        self.num_timesteps = num_timesteps
        betas = cosine_betas(num_timesteps)
        alphas = 1.0 - betas
        cumulative = torch.cumprod(alphas, dim=0)
        cumulative_previous = torch.cat((torch.ones(1), cumulative[:-1]))
        posterior_variance = betas * (1.0 - cumulative_previous) / (1.0 - cumulative)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", cumulative)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(cumulative))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - cumulative)
        )
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / cumulative))
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / cumulative - 1.0)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(cumulative_previous) / (1.0 - cumulative),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - cumulative_previous) * torch.sqrt(alphas) / (1.0 - cumulative),
        )

    def add_noise(
        self, clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        if clean.shape != noise.shape:
            raise ValueError(
                f"clean/noise shape mismatch: {clean.shape} != {noise.shape}"
            )
        shape = (-1, *([1] * (clean.ndim - 1)))
        return (
            self.sqrt_alphas_cumprod[timesteps].view(shape) * clean
            + self.sqrt_one_minus_alphas_cumprod[timesteps].view(shape) * noise
        )

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)

    def step(
        self, predicted_noise: torch.Tensor, noisy: torch.Tensor, timestep: int
    ) -> torch.Tensor:
        """Perform one ancestral denoising step from ``t`` to ``t-1``."""
        clean_estimate = (
            self.sqrt_recip_alphas_cumprod[timestep] * noisy
            - self.sqrt_recipm1_alphas_cumprod[timestep] * predicted_noise
        )
        mean = (
            self.posterior_mean_coef1[timestep] * clean_estimate
            + self.posterior_mean_coef2[timestep] * noisy
        )
        if timestep == 0:
            return mean
        return mean + torch.sqrt(self.posterior_variance[timestep]) * torch.randn_like(
            noisy
        )
