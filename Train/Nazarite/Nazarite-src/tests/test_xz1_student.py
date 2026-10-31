"""Visual student transfer, loss, frozen modules and asynchronous camera timing."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from nazarite.config.train_config.xz1.collect_env_cfg import XZ1DepthCfg
from nazarite.rl.wmp.xz1_dataset import CONTRACT
from nazarite.rl.xz1.student import (
    STUDENT_FORMAT,
    XZ1VisualPolicy,
    action_distillation_step,
    actor_kwargs,
    build_student,
    load_frozen_world_model,
    student_eval_cfg,
    warm_start_student,
)
from nazarite.rl.xz1.teacher import XZ1TerrainTeacherActor
from nazarite.scripts.xz1.distill_xz1_wmp import teacher_probability


def models():
    torch.manual_seed(4)
    obs = TensorDict(
        {
            "actor": torch.randn(3, 6, 47),
            "teacher_scan": torch.randn(3, 132),
            "world_model_feature": torch.randn(3, 512),
        },
        batch_size=[3],
    )
    teacher = (
        XZ1TerrainTeacherActor(obs, {"actor": ["actor"]}, "actor", 12, **actor_kwargs())
        .eval()
        .requires_grad_(False)
    )
    teacher.update_normalization(obs)
    student = build_student(obs, "cpu")
    warm_start_student(student, teacher)
    return obs, teacher, student


def test_common_teacher_weights_and_98d_actor_are_preserved():
    obs, teacher, student = models()
    for key, value in teacher.state_dict().items():
        if not key.startswith("scan_encoder."):
            torch.testing.assert_close(value, student.state_dict()[key])
    assert student.get_latent(obs).shape == (3, 98)
    # Same actor/HIM with an oracle visual embedding reproduces teacher actions.
    latent = student.get_latent(obs).clone()
    latent[:, -32:] = teacher.scan_encoder(obs["teacher_scan"])
    actions = student.distribution.deterministic_output(student.mlp(latent))
    torch.testing.assert_close(actions.clamp(-10, 10), teacher(obs))
    no_oracle = obs.select("actor", "world_model_feature")
    torch.testing.assert_close(student(no_oracle), student(obs))
    changed = no_oracle.clone()
    changed["world_model_feature"] *= -1
    assert not torch.allclose(student(changed), student(no_oracle))
    with pytest.raises(NotImplementedError, match="RSSM"):
        student.as_onnx()


def test_action_l2_updates_only_projection_and_actor_not_him_or_normalization():
    obs, teacher, student = models()
    before = {k: v.clone() for k, v in student.state_dict().items()}
    labels = teacher(obs)
    expected = (labels - student(obs)).norm(p=2, dim=-1).mean().item()
    optimizer = torch.optim.Adam(
        [p for p in student.parameters() if p.requires_grad], lr=1e-4
    )
    loss, norm = action_distillation_step(student, optimizer, obs, labels)
    assert loss == pytest.approx(expected)
    assert norm > 0
    for key, value in student.state_dict().items():
        if not key.startswith(("mlp.", "visual_encoder.")):
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)
    assert not torch.equal(before["mlp.0.weight"], student.mlp[0].weight)
    assert not torch.equal(
        before["visual_encoder.0.weight"], student.visual_encoder[0].weight
    )
    assert all(p.grad is None for p in teacher.parameters())
    assert all(p.grad is None for p in student.estimator.parameters())


class FakeWM:
    def __init__(self):
        self.dynamics = SimpleNamespace(
            initial=lambda n: {"deter": torch.zeros(n, 512)}
        )
        self.calls = []

    @staticmethod
    def normalize_depth(depth, far):
        return depth / far - 0.5

    def observe_step(self, image, prop, action, is_first, state):
        self.calls.append(
            (image.clone(), prop.clone(), action.clone(), is_first.clone())
        )
        value = torch.where(is_first[:, None], 1, state["deter"] + 1)
        return {"deter": value.expand(-1, 512).clone()}


def runtime():
    model = FakeWM()
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        step_dt=0.02,
        sim=SimpleNamespace(sense=lambda **kw: None),
        scene={
            "xz1_depth_camera": SimpleNamespace(
                data=SimpleNamespace(depth=torch.zeros(2, 64, 64, 1))
            )
        },
    )
    policy = XZ1VisualPolicy(env, None, model, XZ1DepthCfg())
    obs = TensorDict(
        {"actor": torch.randn(2, 6, 47), "estimator": torch.randn(2, 50)},
        batch_size=[2],
    )
    return model, policy, obs


def test_preceding_executed_chunk_and_first_frame_far_depth():
    model, policy, obs = runtime()
    first = policy.observations(obs)
    image, prop, action, is_first = model.calls[-1]
    assert image.eq(0.5).all()  # No ray hit -> far, never near.
    assert action.count_nonzero() == 0 and is_first.all()
    assert prop.shape == (2, 33)
    changed = obs.clone()
    changed["estimator"][:, 47:] = 999  # True velocity is excluded.
    policy.observations(changed)
    assert len(model.calls) == 1  # No duplicate posterior at same timestamp.
    for i in range(5):
        policy.advance(torch.full((2, 12), float(i + 1)))
        current = policy.observations(obs)
        if i < 4:
            torch.testing.assert_close(
                current["world_model_feature"], first["world_model_feature"]
            )
    assert len(model.calls) == 2
    torch.testing.assert_close(
        model.calls[-1][2].reshape(2, 5, 12)[0, :, 0], torch.arange(1, 6).float()
    )
    assert not model.calls[-1][3].any()
    # Buffered training features must not alias the mutable runtime state.
    assert first["world_model_feature"].eq(1).all()


def test_partial_reset_isolates_state_action_and_camera_phase():
    model, policy, obs = runtime()
    policy.observations(obs)
    for _ in range(3):
        policy.advance(torch.ones(2, 12))
        policy.observations(obs)
    policy.reset(torch.tensor([True, False]))
    policy.observations(obs)
    assert model.calls[-1][2].shape == (1, 60)
    assert model.calls[-1][2].count_nonzero() == 0
    assert model.calls[-1][3].all()
    for _ in range(2):
        policy.advance(torch.full((2, 12), 2.0))
        policy.observations(obs)
    # Env1 reaches t=5 while env0 is t=2 in its new episode.
    assert policy.ticks.tolist() == [2, 5]
    assert not model.calls[-1][3].any()
    chunk = model.calls[-1][2].reshape(1, 5, 12)[0, :, 0]
    torch.testing.assert_close(chunk, torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0]))
    assert policy.feature[0].eq(1).all() and policy.feature[1].eq(2).all()


def test_teacher_execution_schedule_reference_default_and_optional_mix():
    assert teacher_probability(0, 0, 0) == 0
    assert [teacher_probability(i, 2, 2) for i in range(5)] == [1, 1, 0.5, 0, 0]


def test_world_model_prop_excludes_true_velocity_and_previous_action():
    model, policy, obs = runtime()
    policy.observations(obs)
    prop = model.calls[0][1]
    changed = obs.clone()
    changed["estimator"][:, 35:] = 1000
    policy.reset(torch.ones(2, dtype=torch.bool))
    policy.observations(changed)
    torch.testing.assert_close(model.calls[-1][1], prop)


def test_student_eval_has_camera_but_generates_no_oracle_scan(tmp_path):
    checkpoint = tmp_path / "student.pt"
    torch.save(
        {
            "format": STUDENT_FORMAT,
            "contract": CONTRACT,
            "camera": asdict(XZ1DepthCfg()),
        },
        checkpoint,
    )
    cfg = student_eval_cfg(checkpoint, terrain="flat")
    assert "teacher_scan" not in cfg.observations
    names = {sensor.name for sensor in cfg.scene.sensors}
    assert "teacher_terrain_scan" not in names
    assert "xz1_depth_camera" in names


def test_rejects_legacy_world_model_before_building_model(tmp_path):
    checkpoint = tmp_path / "wm.pt"
    torch.save({"world_model_implementation": "compact"}, checkpoint)
    with pytest.raises(ValueError, match="original WMP"):
        load_frozen_world_model(checkpoint, "cpu")


def test_nonfinite_action_loss_does_not_update_weights():
    obs, _, student = models()
    optimizer = torch.optim.Adam([p for p in student.parameters() if p.requires_grad])
    with pytest.raises(RuntimeError, match="Non-finite"):
        action_distillation_step(
            student, optimizer, obs, torch.full((3, 12), float("nan"))
        )
    assert not optimizer.state


def test_play_wrapper_notifies_policy_on_step_and_ui_reset(monkeypatch):
    from mjlab.rl import RslRlVecEnvWrapper
    from nazarite.scripts.xz1.play_xz1_wmp_student import StudentPlayWrapper

    calls = []
    wrapper = object.__new__(StudentPlayWrapper)
    wrapper.num_envs, wrapper.device = 2, torch.device("cpu")
    wrapper.policy = SimpleNamespace(
        advance=lambda actions: calls.append("advance"),
        reset=lambda dones: calls.append(dones.tolist()),
    )
    result = (None, None, torch.tensor([0, 1]), {})
    monkeypatch.setattr(RslRlVecEnvWrapper, "step", lambda self, a: result)
    wrapper.step(torch.zeros(2, 12))
    assert calls == ["advance", [False, True]]
    monkeypatch.setattr(RslRlVecEnvWrapper, "reset", lambda self: (None, {}))
    wrapper.reset()
    assert calls[-1] == [True, True]
