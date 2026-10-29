"""Evaluation budgets include slow episodes and exact training geometry rows."""

from types import SimpleNamespace

import pytest

from nazarite.scripts.evaluate_xz1_wmp import (
    EpisodeAllocation,
    evaluation_difficulties,
)


@pytest.mark.parametrize("episodes,num_envs", [(1, 4), (16, 16), (19, 16), (20, 4)])
def test_fast_failures_cannot_consume_slow_environment_episode_quotas(
    episodes, num_envs
):
    allocation = EpisodeAllocation(episodes, num_envs)
    assert sum(allocation.quotas) == episodes
    records = []
    # Environment 0 fails immediately and repeatedly; the other environments
    # finish much later. None of their assigned episodes can be censored.
    for index in [0] * (episodes + 5) + list(range(1, num_envs)) * episodes:
        episode_index = allocation.finish(index)
        if episode_index is not None:
            records.append((index, episode_index))
    assert len(records) == episodes
    assert allocation.completed == allocation.quotas
    assert len(set(records)) == episodes


def test_all_ten_levels_and_legacy_difficulties():
    args = SimpleNamespace(all_levels=True, levels=None, difficulties=None)
    points = evaluation_difficulties(args)
    assert [level for level, _ in points] == list(range(10))
    assert [difficulty for _, difficulty in points] == pytest.approx(
        [level / 9 for level in range(10)]
    )
    args.all_levels, args.levels = False, [1, 9]
    assert evaluation_difficulties(args) == [(1, 1 / 9), (9, 1)]
    args.levels = [10]
    with pytest.raises(ValueError, match="levels"):
        evaluation_difficulties(args)
    args.levels, args.difficulties = None, [0, 0.5, 1]
    assert evaluation_difficulties(args) == [(None, 0), (None, 0.5), (None, 1)]
