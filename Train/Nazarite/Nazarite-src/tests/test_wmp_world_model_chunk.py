"""The world model must fold one camera interval of actions into one transition.

The 2026-10-17 runs stepped the RSSM every policy step while the depth image only
refreshed every fifth step, so the dynamics model was asked to predict five
identical transitions from the same frame. It responded by parking `kl_dyn_rep`
on the free-bits ceiling. These tests pin the fix: the RSSM consumes a stacked
action vector whose width is ``action_dim * action_chunk``.
"""

from __future__ import annotations

import pytest
import torch

from nazarite.rl.wmp.world_model import WMPWorldModel, WMPWorldModelCfg


def _small_cfg(**overrides: object) -> WMPWorldModelCfg:
    values: dict[str, object] = {
        "cnn_depth": 4,
        "prop_hidden_dim": 16,
        "prop_layers": 1,
        "deter_dim": 32,
        "stoch_groups": 4,
        "stoch_classes": 8,
        "rssm_hidden_dim": 32,
    }
    values.update(overrides)
    return WMPWorldModelCfg(**values)  # type: ignore[arg-type]


def test_flat_action_dim_multiplies_the_chunk() -> None:
    assert WMPWorldModelCfg(action_dim=12, action_chunk=1).flat_action_dim == 12
    assert WMPWorldModelCfg(action_dim=12, action_chunk=5).flat_action_dim == 60


def test_rssm_action_input_width_follows_the_chunk() -> None:
    cfg = _small_cfg(action_dim=12, action_chunk=5)
    model = WMPWorldModel(cfg)
    linear = model.dynamics.input_layer[0]
    assert isinstance(linear, torch.nn.Linear)
    expected = cfg.stoch_groups * cfg.stoch_classes + cfg.flat_action_dim
    assert linear.in_features == expected

    single = WMPWorldModel(_small_cfg(action_dim=12, action_chunk=1))
    assert isinstance(single.dynamics.input_layer[0], torch.nn.Linear)
    assert single.dynamics.input_layer[0].in_features == (
        12 + cfg.stoch_groups * cfg.stoch_classes
    )


def test_observe_step_accepts_a_folded_action_sequence() -> None:
    cfg = _small_cfg(image_height=16, image_width=16, action_dim=12, action_chunk=5)
    model = WMPWorldModel(cfg)
    batch = 3
    image = torch.zeros(batch, cfg.image_height, cfg.image_width, cfg.image_channels)
    prop = torch.zeros(batch, cfg.prop_dim)
    folded = torch.zeros(batch, cfg.flat_action_dim)
    is_first = torch.ones(batch, dtype=torch.bool)

    state = model.observe_step(image, prop, folded, is_first, None)
    feature = model.dynamics.feature(state)
    assert feature.shape == (batch, cfg.deter_dim)

    with pytest.raises(RuntimeError):
        # A single policy action is not a camera interval: the shape check is the
        # guard that would have caught a half-migrated runner.
        model.observe_step(image, prop, torch.zeros(batch, 12), is_first, state)
