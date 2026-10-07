# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Building blocks for neural models."""

from .cnn import CNN
from .distribution import BetaDistribution, Distribution, GaussianDistribution, HeteroscedasticGaussianDistribution
from .mlp import MLP
from .him_estimator import HIMEstimator
from .normalization import EmpiricalDiscountedVariationNormalization, EmpiricalNormalization
from .rnn import RNN, HiddenState

__all__ = [
    "CNN",
    "MLP",
    "HIMEstimator",
    "RNN",
    "BetaDistribution",
    "Distribution",
    "EmpiricalDiscountedVariationNormalization",
    "EmpiricalNormalization",
    "GaussianDistribution",
    "HeteroscedasticGaussianDistribution",
    "HiddenState",
    "HIMActorCritic",
]


def __getattr__(name: str):
    """Lazily expose the legacy HIM actor name without an import cycle."""
    if name == "HIMActorCritic":
        from rsl_rl.models.him_actor_model import HIMActorCritic

        return HIMActorCritic
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
