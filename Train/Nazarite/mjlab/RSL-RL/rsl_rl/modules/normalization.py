# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2020 Preferred Networks, Inc.


from __future__ import annotations

import torch
from torch import nn

from rsl_rl.utils.numerics import globally_bad


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, shape: int | tuple[int, ...] | list[int], eps: float = 1e-2, until: int | None = None) -> None:
        """Initialize EmpiricalNormalization module.

        .. note:: The normalization parameters are computed over the whole batch, not for each environment separately.

        Args:
            shape: Shape of input values except batch axis.
            eps: Small value for stability.
            until: If this arg is specified, the module learns input values until the sum of batch sizes exceeds it.
        """
        super().__init__()
        self.eps = eps
        self.until = until
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    @property
    def mean(self) -> torch.Tensor:
        """Return the current running mean."""
        return self._mean.squeeze(0).clone()  # type: ignore

    @property
    def std(self) -> torch.Tensor:
        """Return the current running standard deviation."""
        return self._std.squeeze(0).clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize mean and variance of values based on empirical values."""
        return (x - self._mean) / (self._std + self.eps)

    @torch.jit.unused
    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Learn input values without computing the output values of them."""
        if not self.training:
            return
        if self.until is not None and self.count >= self.until:
            return

        if globally_bad((~torch.isfinite(x)).any()):
            raise FloatingPointError("Normalizer input is non-finite on at least one rank")
        count_x = torch.tensor(x.shape[0], dtype=self.count.dtype, device=self.count.device)
        if x.shape[0]:
            var_x, local_mean = torch.var_mean(x.double(), dim=0, unbiased=False, keepdim=True)
        else:
            local_mean = torch.zeros_like(self._mean, dtype=torch.float64)
            var_x = torch.zeros_like(local_mean)
        mean_sum = local_mean * x.shape[0]
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(count_x)
            torch.distributed.all_reduce(mean_sum)
        if count_x.item() == 0:
            return
        mean_x = mean_sum / count_x
        m2_x = x.shape[0] * (var_x + (local_mean - mean_x).square())
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(m2_x)
        next_count = self.count + count_x
        rate = count_x.double() / next_count
        delta = mean_x - self._mean.double()
        mean = self._mean.double() + rate * delta
        var = (1 - rate) * self._var.double() + m2_x / next_count + rate * (1 - rate) * delta.square()
        mean = mean.to(self._mean.dtype)
        var = var.to(self._var.dtype)
        std = var.sqrt()
        bad = (~torch.isfinite(mean) | ~torch.isfinite(var) | ~torch.isfinite(std) | (var < 0)).any()
        if globally_bad(bad):
            raise FloatingPointError("Normalizer candidate is invalid; running statistics were not changed")
        self.count.copy_(next_count)
        self._mean.copy_(mean)
        self._var.copy_(var)
        self._std.copy_(std)

    @torch.jit.unused
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """De-normalize values based on empirical values."""
        return y * (self._std + self.eps) + self._mean


class EmpiricalDiscountedVariationNormalization(nn.Module):
    """Reward normalization from Pathak's large scale study on PPO.

    Reward normalization. Since the reward function is non-stationary, it is useful to normalize the scale of the
    rewards so that the value function can learn quickly. We did this by dividing the rewards by a running estimate of
    the standard deviation of the sum of discounted rewards.
    """

    def __init__(
        self,
        shape: int | tuple[int, ...] | list[int],
        eps: float = 1e-2,
        gamma: float = 0.99,
        until: int | None = None,
    ) -> None:
        """Initialize discounted-reward normalization with running moments."""
        super().__init__()

        self.emp_norm = EmpiricalNormalization(shape, eps, until)
        self.disc_avg = _DiscountedAverage(gamma)

    def forward(self, rew: torch.Tensor) -> torch.Tensor:
        """Normalize rewards using the running std of discounted returns."""
        if self.training:
            # Update discounted rewards
            avg = self.disc_avg.update(rew)
            # Update moments from discounted rewards
            self.emp_norm.update(avg)

        # Normalize rewards with the empirical std
        if self.emp_norm._std > 0:  # type: ignore
            return rew / self.emp_norm._std  # type: ignore
        else:
            return rew


class _DiscountedAverage:
    r"""Discounted average of rewards.

    The discounted average is defined as:

    .. math::

        \bar{R}_t = \gamma \bar{R}_{t-1} + r_t
    """

    def __init__(self, gamma: float) -> None:
        """Initialize discounted accumulation with a fixed discount factor."""
        self.avg = None
        self.gamma = gamma

    def update(self, rew: torch.Tensor) -> torch.Tensor:
        """Update and return the discounted running average."""
        if self.avg is None:
            self.avg = rew
        else:
            self.avg = self.avg * self.gamma + rew
        return self.avg
