"""AMP wiring end to end: the style reward reaches PPO and the prior checkpoints.

The unit tests pin the discriminator in isolation; this one runs the real runner
over a real environment for two iterations, which is the only place the style
reward, the transition bookkeeping and the checkpoint path meet.
"""

from __future__ import annotations

import copy
from dataclasses import asdict

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from nazarite.config.train_config.env_cfgs.wmp_him_go2_env_cfg import (
    Nazarite_WMP_HIM_Complex_Terrain_Go2,
)
from nazarite.config.train_config.wmp_him_go2_cfg import go2_wmp_him_runner_cfg
from nazarite.rl.wmp.runner import WMPHIMOnPolicyRunner

NUM_ENVS = 2
STEPS_PER_ENV = 8


def _agent_cfg() -> dict:
    agent = asdict(go2_wmp_him_runner_cfg())
    agent["num_steps_per_env"] = STEPS_PER_ENV
    agent["logger"] = "tensorboard"
    agent["save_interval"] = 10_000
    # A small discriminator and batch keep the test quick; the reward shape is
    # what matters here, not the capacity.
    agent["amp"].update({"hidden_dims": (32, 16), "batch_size": 8})
    # Skip world model training so the iteration only exercises the AMP path.
    agent["world_model"].update({"train_start_frames": 10**9, "pretrain_updates": 0})
    return agent


def _make_env() -> ManagerBasedRlEnv:
    cfg = Nazarite_WMP_HIM_Complex_Terrain_Go2(flat=True)
    cfg.scene.num_envs = NUM_ENVS
    return ManagerBasedRlEnv(cfg=cfg, device="cpu")


@pytest.mark.slow
def test_the_style_reward_trains_the_discriminator_and_checkpoints(tmp_path) -> None:
    env = _make_env()
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=10.0)
        runner = WMPHIMOnPolicyRunner(
            wrapped, _agent_cfg(), str(tmp_path), device="cpu"
        )
        assert runner.amp is not None
        before = copy.deepcopy(runner.amp.discriminator.state_dict())

        runner.learn(2)

        after = runner.amp.discriminator.state_dict()
        assert any(not torch.equal(before[key], after[key]) for key in before), (
            "the discriminator never trained"
        )
        assert runner.amp.bad_values == 0
        # Every transition of the rollout is a policy transition for the prior;
        # the counters must not grow across iterations.
        assert runner._amp_pairs == []

        checkpoint = tmp_path / "model_1.pt"
        runner.save(str(checkpoint))
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert "amp_state_dict" in saved

        expected = runner.amp.style_reward(torch.zeros(4, 30), torch.ones(4, 30))
    finally:
        env.close()

    # A fresh environment: replaying a rollout inside ``torch.inference_mode``
    # leaves mjlab's delay buffer holding inference tensors, which cannot be
    # written to afterwards. Training and playback are separate processes, so the
    # runner is the thing under test here, not buffer reuse.
    fresh = _make_env()
    try:
        restored = WMPHIMOnPolicyRunner(
            RslRlVecEnvWrapper(fresh, clip_actions=10.0),
            _agent_cfg(),
            str(tmp_path / "restore"),
            device="cpu",
        )
        restored.load(str(checkpoint), map_location="cpu")
        assert torch.allclose(
            restored.amp.style_reward(torch.zeros(4, 30), torch.ones(4, 30)),
            expected,
            atol=1e-6,
        )
    finally:
        fresh.close()
