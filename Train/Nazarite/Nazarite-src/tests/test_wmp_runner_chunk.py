"""Chunking behaviour of the WMP runner's observation encoding.

The runner is the seam where the policy rate (50 Hz) meets the camera rate
(10 Hz). These tests drive ``_encode_current_observation`` directly with a stub
environment so the fold/hold contract is pinned without needing a rollout.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from nazarite.rl.wmp.runner import _WM_MAX_BAD_FEATURES, WMPHIMOnPolicyRunner
from nazarite.rl.wmp.world_model import WMPWorldModel, WMPWorldModelCfg

NUM_ENVS = 2
NUM_ACTIONS = 12
PROP_DIM = 33
CHUNK = 5


class _RecordingReplay:
    def __init__(self) -> None:
        self.frames: list[tuple[torch.Tensor, ...]] = []

    def add(self, image, prop, action, reward, is_first) -> None:
        self.frames.append(
            (
                image.clone(),
                prop.clone(),
                action.clone(),
                reward.clone(),
                is_first.clone(),
            )
        )


def _runner() -> WMPHIMOnPolicyRunner:
    cfg = WMPWorldModelCfg(
        cnn_depth=4,
        prop_hidden_dim=16,
        prop_layers=1,
        deter_dim=32,
        stoch_groups=4,
        stoch_classes=8,
        rssm_hidden_dim=32,
        action_dim=NUM_ACTIONS,
        action_chunk=CHUNK,
    )
    runner = WMPHIMOnPolicyRunner.__new__(WMPHIMOnPolicyRunner)
    runner.env = SimpleNamespace(num_envs=NUM_ENVS, num_actions=NUM_ACTIONS)
    runner.device = torch.device("cpu")
    runner.world_model = WMPWorldModel(cfg)
    runner.replay = _RecordingReplay()
    runner._wm_runtime_cfg = {
        "action_chunk": CHUNK,
        "capacity_per_env": 256,
        "train_start_frames": 10000,
        "updates_per_iteration": 10,
        "batch_size": 16,
        "batch_length": 64,
        "pretrain_updates": 0,
    }
    runner._wm_state = None
    runner._wm_prev_reward = torch.zeros(NUM_ENVS)
    runner._wm_is_first = torch.ones(NUM_ENVS, dtype=torch.bool)
    runner._wm_action_hist = torch.zeros(NUM_ENVS, CHUNK, NUM_ACTIONS)
    runner._wm_feature = torch.zeros(NUM_ENVS, cfg.deter_dim)
    runner._wm_step_index = 0
    runner._wm_pretrained = False
    runner._camera_depth = lambda: torch.zeros(NUM_ENVS, 64, 64, 1)
    runner._prop_from_estimator = lambda obs: torch.zeros(NUM_ENVS, PROP_DIM)
    return runner


def _obs() -> TensorDict:
    return TensorDict({"estimator": torch.zeros(NUM_ENVS, 50)}, batch_size=[NUM_ENVS])


def _step(runner: WMPHIMOnPolicyRunner, value: float) -> torch.Tensor:
    runner._wm_prev_action = torch.full((NUM_ENVS, NUM_ACTIONS), value)
    return runner._encode_current_observation(_obs())["world_model_feature"]


def test_world_model_is_stepped_once_per_chunk() -> None:
    runner = _runner()
    # The first observation flushes immediately: there is no history to fold and
    # leaving the policy blind for a whole interval would be pointless.
    _step(runner, 1.0)
    assert len(runner.replay.frames) == 1
    for value in (2.0, 3.0, 4.0):
        _step(runner, value)
    assert len(runner.replay.frames) == 1  # still inside the chunk
    _step(runner, 5.0)
    assert len(runner.replay.frames) == 2


def test_folded_action_holds_every_action_of_the_interval() -> None:
    runner = _runner()
    for value in (0.1, 0.2, 0.3, 0.4, 0.5):
        _step(runner, value)
    _, _, action, _, is_first = runner.replay.frames[-1]
    assert action.shape == (NUM_ENVS, CHUNK * NUM_ACTIONS)
    folded = action[0].reshape(CHUNK, NUM_ACTIONS)
    for slot, value in enumerate((0.1, 0.2, 0.3, 0.4, 0.5)):
        assert torch.allclose(folded[slot], torch.full((NUM_ACTIONS,), value))
    # The first chunk carries the episode-start flag; the next one does not.
    assert runner.replay.frames[0][4].all()
    assert not is_first.any()


def test_feature_is_held_between_camera_frames() -> None:
    runner = _runner()
    first = _step(runner, 0.1)
    second = _step(runner, 0.2)
    third = _step(runner, 0.3)
    assert torch.equal(first, second)
    assert torch.equal(second, third)
    # A held feature must still be a tensor of the width the actor expects.
    assert third.shape == (NUM_ENVS, runner.world_model.feature_dim)


def test_inference_folds_actions_like_training() -> None:
    """The play loop must go through the same fold.

    ``play`` used to call ``world_model.observe_step`` with a single 12-dim action
    and crashed with ``mat1 and mat2 shapes cannot be multiplied (1x1036 and
    1084x512)`` as soon as the trained network expected a folded 5x12 interval.
    """
    image = torch.zeros(NUM_ENVS, 64, 64, 1)
    prop = torch.zeros(NUM_ENVS, PROP_DIM)
    # Both spellings, because the training loop passes ``self.device`` as a string
    # while inference passes a ``torch.device``. Comparing the two resets the fold
    # buffers every call, so the chunk never fills and the world model only ever
    # sees one real action plus zero padding.
    for device in (torch.device("cpu"), "cpu"):
        runner = _runner()
        results = [
            runner._fold_wm_action(
                image,
                prop,
                torch.full((NUM_ENVS, NUM_ACTIONS), value),
                runner._wm_is_first,
                device,  # type: ignore[arg-type]
            )
            for value in (0.1, 0.2, 0.3, 0.4, 0.5)
        ]
        assert results[0][1] is not None, device
        assert all(stacked is None for _, stacked in results[1:4]), device
        _, stacked = results[4]
        assert stacked is not None, device
        assert stacked.shape == (NUM_ENVS, CHUNK * NUM_ACTIONS), device
        # The fifth action must land in the last slot, not in slot 0 of a buffer that
        # was silently recreated.
        assert torch.equal(
            stacked[:, -NUM_ACTIONS:], torch.full((NUM_ENVS, NUM_ACTIONS), 0.5)
        )


def test_nonfinite_proprioception_cannot_reach_the_encoder() -> None:
    """One bad physics step must not become a NaN categorical sample in the RSSM."""
    runner = _runner()
    seen: list[torch.Tensor] = []
    observe_step = runner.world_model.observe_step

    def spy(image, prop, action, is_first, state):
        seen.append(prop.clone())
        return observe_step(image, prop, action, is_first, state)

    runner.world_model.observe_step = spy  # type: ignore[method-assign]
    runner._prop_from_estimator = lambda obs: torch.tensor(
        [[float("inf")] + [0.1] * (PROP_DIM - 1), [float("nan")] * PROP_DIM],
        dtype=torch.float32,
    )
    _step(runner, 0.1)
    assert torch.isfinite(seen[-1]).all()
    assert seen[-1].abs().max() <= 100.0


def test_nonfinite_feature_is_zeroed_and_a_persistent_failure_raises() -> None:
    """A transient NaN recovers; a world model whose weights are NaN must stop."""
    runner = _runner()
    nan_feature = torch.full((NUM_ENVS, runner.world_model.feature_dim), float("nan"))

    # Never call through a real dynamics model: this is the failure being tested.
    runner.world_model.dynamics.get_deter_feat = lambda state: nan_feature

    out = _step(runner, 0.1)
    assert torch.equal(out, torch.zeros_like(out))
    assert runner._wm_state is None, "history must be dropped, not carried as NaN"
    assert runner._wm_bad_features == 1

    for _ in range(_WM_MAX_BAD_FEATURES - 1):
        _step(runner, 0.1)
    assert runner._wm_bad_features == _WM_MAX_BAD_FEATURES
    with pytest.raises(RuntimeError, match="non-finite features"):
        _step(runner, 0.1)


def test_reset_between_camera_frames_clears_memory_and_stays_pending():
    runner = _runner()
    _step(runner, 0.1)
    unaffected = runner._wm_feature[1].clone()
    _step(runner, 0.2)
    runner._mark_wm_resets(torch.tensor([True, False]))
    assert runner._wm_feature[0].count_nonzero() == 0
    assert runner._wm_action_hist[0].count_nonzero() == 0
    initial = runner.world_model.dynamics.initial(NUM_ENVS)
    for key in initial:
        torch.testing.assert_close(runner._wm_state[key][0], initial[key][0])
    torch.testing.assert_close(runner._wm_feature[1], unaffected)
    _step(runner, 0.3)
    assert runner._wm_feature[0].count_nonzero() == 0
    assert len(runner.replay.frames) == 1
    # A later control step with no done must not erase the previous reset.
    runner._wm_is_first.zero_()
    _step(runner, 0.4)
    _step(runner, 0.5)
    assert runner.replay.frames[-1][4].tolist() == [True, False]
    assert not runner._wm_pending_reset.any()
