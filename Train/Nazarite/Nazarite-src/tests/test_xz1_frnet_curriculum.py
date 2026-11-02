"""Promotion must reflect terminal recovery in every practiced fall family."""

from types import SimpleNamespace
from typing import cast

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.tasks.registry import load_env_cfg
from nazarite.mdp.xz1 import frnet
from nazarite.mdp.xz1.frnet_curriculum import RECOVERY_STAGES, RecoveryCurriculum


@pytest.fixture
def curriculum_env():
    cfg = load_env_cfg("Nazarite-FRNet-Paper-Recovery-XZ1")
    return cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=6,
            device="cpu",
            common_step_counter=0,
            reset_buf=torch.ones(6, dtype=torch.bool),
            reward_manager=SimpleNamespace(get_term_cfg=cfg.rewards.__getitem__),
            extras={
                "frnet_reset_kind": torch.arange(6),
                "frnet_reset_level": torch.zeros(6, dtype=torch.long),
                "frnet_episode_steps": torch.full((6,), 350),
                "frnet_episode_success": torch.ones(6, dtype=torch.bool),
                "frnet_final_stable": torch.ones(6, dtype=torch.bool),
            },
        ),
    )


def test_promotion_requires_both_sides_and_final_stability(curriculum_env):
    env = curriculum_env
    curriculum = RecoveryCurriculum(CurriculumTermCfg(func=RecoveryCurriculum), env)
    # Initial construction cannot count a zero-step pose as a completed trial.
    env.extras["frnet_episode_steps"].zero_()
    curriculum(env, None, min_episodes_per_case=1, min_stage_steps=100)
    assert curriculum.trials.eq(0).all()
    env.extras["frnet_episode_steps"].fill_(350)
    env.common_step_counter = 100
    # Having once stood is insufficient if a side ends the episode fallen.
    env.extras["frnet_final_stable"][0] = False
    curriculum(env, None, min_episodes_per_case=1, min_stage_steps=100)
    assert curriculum.level == 0
    assert curriculum.final_rates[0] == 0
    env.extras["frnet_final_stable"].fill_(True)
    env.common_step_counter = 101
    curriculum(env, None, min_episodes_per_case=1, min_stage_steps=100)
    assert curriculum.level == 1
    assert curriculum.trials.eq(0).all()
    # In-flight old-level and rehearsal episodes cannot promote the new level.
    env.common_step_counter = 1000
    curriculum(env, None, min_episodes_per_case=1, min_stage_steps=100)
    assert curriculum.level == 1 and curriculum.trials.eq(0).all()
    # Standing / semi-fallen successes never establish roll mastery.
    env.extras["frnet_reset_level"].fill_(1)
    curriculum(env, torch.tensor([4, 5]), min_episodes_per_case=1, min_stage_steps=0)
    assert curriculum.level == 1 and curriculum.trials.eq(0).all()


def test_back_failure_blocks_promotion_and_weights_restore(curriculum_env):
    env = curriculum_env
    curriculum = RecoveryCurriculum(CurriculumTermCfg(func=RecoveryCurriculum), env)
    full_weight = -0.01
    assert env.reward_manager.get_term_cfg("action_rate").weight == full_weight * 0.1
    curriculum.apply_stage()
    assert env.reward_manager.get_term_cfg("action_rate").weight == full_weight * 0.1
    curriculum.level = 2
    env.extras["frnet_reset_level"].fill_(2)
    env.extras["frnet_episode_success"][2] = False
    env.extras["frnet_final_stable"][2] = False
    env.common_step_counter = 100
    curriculum(env, None, min_episodes_per_case=1, min_stage_steps=0)
    assert curriculum.level == 2
    assert curriculum.success_rates[2] == 0
    curriculum.level = len(RECOVERY_STAGES) - 1
    curriculum.apply_stage()
    assert env.reward_manager.get_term_cfg("action_rate").weight == full_weight
    assert env.reward_manager.get_term_cfg("joint_limits").weight == -1
    assert env.reward_manager.get_term_cfg("safety_force").weight == -0.01
    # State restores the stage, progress and exact effective reward weights.
    curriculum.trials[2] = 19
    state = curriculum.state_dict()
    curriculum.level = 0
    curriculum.trials.zero_()
    curriculum.load_state_dict(state)
    assert curriculum.level == 5 and curriculum.trials[2] == 19
    assert env.reward_manager.get_term_cfg("action_rate").weight == full_weight


def test_terminal_progress_survives_other_metric_calls_and_partial_reset(
    curriculum_env, monkeypatch
):
    env = curriculum_env
    env.step_dt = 0.02
    stable = torch.tensor([True, False, True, False, True, True])
    monkeypatch.setattr(frnet, "stable_recovery_mask", lambda *args, **kw: stable)
    tracked = frnet.StableRecoveryMetric(
        MetricsTermCfg(
            func=frnet.StableRecoveryMetric,
            params={"hold_duration_s": 0.04, "track_progress": True},
        ),
        env,
    )
    other = frnet.StableRecoveryMetric(
        MetricsTermCfg(func=frnet.StableRecoveryMetric), env
    )
    for _ in range(2):
        tracked(env)
        other(env)
    assert env.extras["frnet_episode_success"].tolist() == stable.tolist()
    stable[0] = False
    tracked(env)
    assert env.extras["frnet_episode_success"][0]
    assert not env.extras["frnet_final_stable"][0]
    tracked.reset(torch.tensor([0, 2]))
    assert env.extras["frnet_episode_steps"].tolist() == [0, 3, 0, 3, 3, 3]
    assert env.extras["frnet_episode_success"].tolist() == [
        False,
        False,
        False,
        False,
        True,
        True,
    ]
