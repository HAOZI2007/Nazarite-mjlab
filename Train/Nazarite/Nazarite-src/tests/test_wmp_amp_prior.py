"""The motion prior: discriminator shape, style reward bounds and its update."""

from __future__ import annotations

import copy

import pytest
import torch

from nazarite.rl.wmp.amp import AdversarialMotionPrior, AMPDiscriminator
from nazarite.rl.wmp.motion import AMP_OBSERVATION_DIM, load_dataset

BATCH = 8


def _prior(**kwargs) -> AdversarialMotionPrior:
    options = {
        "hidden_dims": (32, 16),
        "batch_size": 8,
        "learning_rate": 1.0e-3,
        "seed": 3,
    }
    options.update(kwargs)
    return AdversarialMotionPrior(device="cpu", **options)


def _batch(size: int = BATCH) -> tuple[torch.Tensor, torch.Tensor]:
    states = torch.zeros(size, AMP_OBSERVATION_DIM)
    states[:, 0:12] = 0.1
    following = states.clone()
    following[:, 0:12] = 0.2
    return states, following


def test_discriminator_maps_a_transition_to_one_score() -> None:
    discriminator = AMPDiscriminator(AMP_OBSERVATION_DIM, (32, 16))
    score = discriminator(torch.zeros(BATCH, 30), torch.ones(BATCH, 30))
    assert score.shape == (BATCH,)
    assert discriminator.observation_dim == AMP_OBSERVATION_DIM
    assert discriminator.hidden_dims == (32, 16)


def test_style_reward_stays_inside_its_clamp() -> None:
    prior = _prior()
    states, following = _batch()
    reward = prior.style_reward(states, following)
    assert reward.shape == (BATCH,)
    # 0.15 * clamp(1 - (d-1)^2 / 4, 0): never negative, never above the ceiling.
    assert float(reward.min()) >= 0.0
    assert float(reward.max()) <= prior.style_coefficient


def test_update_trains_the_discriminator_and_reports_finite_metrics() -> None:
    prior = _prior()
    before = copy.deepcopy(prior.discriminator.state_dict())
    states, following = _batch()
    # Four steps so the discriminator has to separate the two populations.
    for _ in range(4):
        metrics = prior.update(states, following)
    assert set(metrics) == {"loss", "grad_pen", "d_expert", "d_policy"}
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    after = prior.discriminator.state_dict()
    assert any(not torch.equal(before[key], after[key]) for key in before), (
        "the discriminator never moved"
    )
    # A constant policy transition cannot look like the (varied) reference set.
    assert metrics["d_expert"] > metrics["d_policy"]


def test_checkpoint_round_trip_restores_the_same_reward() -> None:
    prior = _prior()
    states, following = _batch()
    for _ in range(3):
        prior.update(states, following)
    expected = prior.style_reward(states, following)

    other = _prior()
    other.load_state_dict(prior.state_dict())
    assert torch.allclose(other.style_reward(states, following), expected, atol=1e-6)
    assert other.bad_values == prior.bad_values


def test_non_finite_transitions_cannot_produce_a_non_finite_reward() -> None:
    prior = _prior()
    states, following = _batch()
    states[0, 0] = float("inf")
    following[1, 5] = float("nan")
    reward = prior.style_reward(states, following)
    assert torch.isfinite(reward).all()


def test_the_bundled_dataset_is_the_go2_trot_set() -> None:
    dataset = load_dataset()
    assert len(dataset.clips) >= 10
    assert dataset.num_frames > 1000
    assert dataset.all_states().shape[1] == AMP_OBSERVATION_DIM


@pytest.mark.parametrize("batch_size", [0, 1])
def test_update_needs_at_least_two_policy_transitions(batch_size: int) -> None:
    prior = _prior()
    states, following = _batch(batch_size)
    with pytest.raises(ValueError):
        prior.update(states, following)
