"""Reference objective/update parity, recurrent resets, and checkpoint boundary."""

from dataclasses import asdict

import pytest
import torch

from nazarite.rl.wmp.reference_dreamer.models import WorldModel
from nazarite.rl.wmp.world_model import WMPWorldModel, WMPWorldModelCfg


def small_cfg():
    return WMPWorldModelCfg(
        action_chunk=5,
        cnn_depth=2,
        prop_hidden_dim=16,
        prop_layers=1,
        deter_dim=16,
        stoch_groups=2,
        stoch_classes=4,
        rssm_hidden_dim=16,
        head_units=16,
    )


def batch():
    return {
        "image": torch.randint(0, 256, (2, 4, 64, 64, 1), dtype=torch.uint8),
        "prop": torch.randn(2, 4, 33),
        "action": torch.randn(2, 4, 60),
        "reward": torch.randn(2, 4),
        "is_first": torch.tensor(
            [[True, False, False, False], [True, False, True, False]]
        ),
    }


def test_validation_objective_and_adam_update_match_native_reference():
    torch.manual_seed(17)
    cfg = small_cfg()
    model = WMPWorldModel(cfg)
    native = WorldModel(
        cfg.reference_config("cpu"), {"prop": (33,), "image": (64, 64, 1)}, True
    )
    native.load_state_dict(model.core.state_dict())
    data = batch()
    original_action = data["action"].clone()
    reference_data = model._reference_batch(data)
    torch.manual_seed(82)
    with torch.no_grad():
        _, validation = model.compute_loss(data)
    torch.manual_seed(82)
    _, _, expected = native._train(reference_data)
    torch.manual_seed(82)
    actual = model.train_batch(data)
    assert actual["loss"] == pytest.approx(float(expected["model_loss"]), rel=1e-6)
    for key in (
        "image_loss",
        "prop_loss",
        "reward_loss",
        "kl",
        "dyn_loss",
        "rep_loss",
        "prior_ent",
        "post_ent",
    ):
        mean = float(torch.as_tensor(expected[key]).mean())
        assert actual[key] == pytest.approx(mean, rel=1e-6)
        assert float(validation[key]) == pytest.approx(mean, rel=1e-6)
    assert float(validation["loss"]) == pytest.approx(actual["loss"], rel=1e-6)
    # Exact finite update equality includes reference initialization, optimizer
    # epsilon, losses, reductions and gradient clipping, not just forward shapes.
    for key, parameter in model.core.state_dict().items():
        torch.testing.assert_close(parameter, native.state_dict()[key], rtol=0, atol=0)
    for index, state in model.optimizer.state_dict()["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(
                value,
                native._model_opt._opt.state_dict()["state"][index][key],
                rtol=0,
                atol=0,
            )
    torch.testing.assert_close(data["action"], original_action)


def test_reference_image_sum_prop_loss_and_learned_reset_are_retained():
    model = WMPWorldModel(small_cfg())
    data = batch()
    torch.manual_seed(61)
    loss, metrics = model.compute_loss(data)
    torch.manual_seed(61)
    inputs = model._reference_batch(data)
    post, _ = model.dynamics.observe(
        model.encode(inputs["image"], inputs["prop"]),
        inputs["action"],
        inputs["is_first"],
    )
    expected_image = (
        (model.decode_image(post) - inputs["image"])
        .square()
        .sum(dim=(-1, -2, -3))
        .mean()
    )
    torch.testing.assert_close(metrics["image_loss"], expected_image)
    torch.testing.assert_close(
        loss,
        metrics["image_loss"]
        + metrics["prop_loss"]
        + 0.5 * metrics["dyn_loss"]
        + 0.1 * metrics["rep_loss"],
    )
    loss.backward()
    assert (
        model.core.heads["decoder"]._mlp.mean_layer["prop"].weight.grad.abs().sum() > 0
    )
    assert model.dynamics.W.grad is not None
    with torch.no_grad():
        model.dynamics.W.fill_(0.3)
    initial = model.dynamics.initial(2)
    torch.testing.assert_close(initial["deter"], torch.full((2, 16), 0.3).tanh())


def test_partial_reset_matches_independent_fresh_rollout_without_mutating_inputs():
    model = WMPWorldModel(small_cfg())
    model.eval()
    first = torch.ones(2)
    image, prop, action = (
        torch.randn(2, 64, 64, 1),
        torch.randn(2, 33),
        torch.randn(2, 60),
    )
    state = model.observe_step(image, prop, action, first, None)
    saved = {key: value.clone() for key, value in state.items()}
    saved_action = action.clone()
    torch.manual_seed(9)
    reset = model.observe_step(image, prop, action, torch.tensor([True, False]), state)
    torch.manual_seed(9)
    fresh = model.observe_step(image, prop, action, first, None)
    for key in state:
        torch.testing.assert_close(state[key], saved[key])
        torch.testing.assert_close(reset[key][0], fresh[key][0])
    torch.testing.assert_close(action, saved_action)


def test_nonfinite_input_rejects_update_and_checkpoint_preserves_native_weights():
    cfg = small_cfg()
    model = WMPWorldModel(cfg)
    data = batch()
    data["prop"][0, 1, 0] = torch.nan
    before = {key: value.clone() for key, value in model.state_dict().items()}
    with pytest.raises(RuntimeError, match="Non-finite"):
        model.train_batch(data)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])
    restored = WMPWorldModel(WMPWorldModelCfg(**asdict(cfg)))
    restored.load_state_dict(model.state_dict(), strict=True)
    assert restored.implementation == "wmp_go1_dreamer_v1"


def test_native_optimizer_rejects_nonfinite_loss_and_gradients():
    model = WMPWorldModel(small_cfg())
    parameter = next(model.parameters())
    saved = parameter.detach().clone()
    with pytest.raises(RuntimeError, match="Non-finite"):
        model.core._model_opt(
            parameter.sum() * torch.tensor(float("inf")), model.parameters()
        )
    # Finite scalar value with an infinite derivative at zero.
    with pytest.raises(RuntimeError, match="non-finite"):
        model.core._model_opt((parameter.sum() * 0).sqrt(), model.parameters())
    torch.testing.assert_close(parameter, saved)
