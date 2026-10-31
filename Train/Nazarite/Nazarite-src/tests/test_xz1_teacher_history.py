"""Temporal terrain input, policy-preserving warm starts and actual PPO IO."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from nazarite.config.train_config.XZ1_env_cfgs.him_rl_cfg import xz1_him_runner_cfg
from nazarite.rl.xz1_teacher import (
    XZ1TerrainTeacherActor,
    XZ1TerrainTeacherRunner,
    expand_teacher_state,
)


def models():
    obs = TensorDict(
        {
            "actor": torch.randn(3, 6, 47),
            "teacher_scan": torch.randn(3, 3, 132).clamp(-1, 1),
        },
        batch_size=[3],
    )
    kwargs = asdict(xz1_him_runner_cfg().actor)
    kwargs.pop("class_name")
    old = XZ1TerrainTeacherActor(obs, {"actor": ["actor"]}, "actor", 12, **kwargs)
    new = XZ1TerrainTeacherActor(
        obs,
        {"actor": ["actor"]},
        "actor",
        12,
        scan_history_length=3,
        scan_update_period=0.1,
        **kwargs,
    )
    return obs, old.eval(), new.eval()


def test_expansion_preserves_old_teacher_and_export_history_order():
    obs, old, new = models()
    old_obs = obs.clone()
    old_obs["teacher_scan"] = obs["teacher_scan"][:, -1]
    new.load_state_dict(expand_teacher_state(new, old.state_dict()))
    torch.testing.assert_close(new(obs), old(old_obs))
    assert new.get_latent(obs).shape == (3, 162)
    assert new.mlp[0].weight[:, 98:].count_nonzero() == 0
    expected = new.scan_encoder(obs["teacher_scan"]).flip(1).flatten(1)
    torch.testing.assert_close(new.get_latent(obs)[:, 66:], expected)
    export = torch.jit.script(new.as_jit().eval())
    torch.testing.assert_close(
        export(new.get_observation_tensor(obs), obs["teacher_scan"]), new(obs)
    )


def test_history_has_gradient_and_minibatch_evaluation_has_no_mutable_state():
    obs, _, model = models()
    model(obs).square().sum().backward()
    assert model.scan_encoder[0].weight.grad.abs().sum() > 0
    assert model.mlp[0].weight.grad[:, 98:].abs().sum() > 0
    expected = model(obs).detach()
    shuffled = torch.tensor([2, 0, 1])
    torch.testing.assert_close(model(obs[shuffled]), expected[shuffled])
    torch.testing.assert_close(model(obs), expected)


def test_standalone_teacher_preserves_environment_and_rl_defaults():
    from nazarite.config.train_config.XZ1_env_cfgs.him_env_cfg import (
        XZ1_WMP_COMPLEX_TERRAINS_CFG,
        _build_xz1_env,
    )
    from nazarite.config.train_config.XZ1_env_cfgs.teacher_env_cfg import (
        Nazarite_WMP_HIM_Teacher_XZ1,
    )
    from nazarite.config.train_config.XZ1_env_cfgs.teacher_rl_cfg import (
        xz1_teacher_runner_cfg,
    )

    legacy = _build_xz1_env(False, XZ1_WMP_COMPLEX_TERRAINS_CFG, obstacle_courses=True)
    teacher = Nazarite_WMP_HIM_Teacher_XZ1(scan_history_length=6)
    for name in ("commands", "events", "rewards", "terminations", "curriculum"):
        assert {
            key: asdict(value) for key, value in getattr(teacher, name).items()
        } == {key: asdict(value) for key, value in getattr(legacy, name).items()}
    assert asdict(teacher.scene.terrain.terrain_generator) == asdict(
        legacy.scene.terrain.terrain_generator
    )
    assert (
        teacher.scene.terrain.max_init_terrain_level
        == legacy.scene.terrain.max_init_terrain_level
    )
    assert asdict(teacher.sim) == asdict(legacy.sim)
    assert teacher.decimation == legacy.decimation
    for name in ("actor", "critic", "estimator"):
        assert asdict(teacher.observations[name]) == asdict(legacy.observations[name])
    previous = asdict(xz1_him_runner_cfg(experiment_name="xz1_wmp_him_teacher"))
    current = asdict(xz1_teacher_runner_cfg())
    current.pop("initial_checkpoint")
    current["actor"].pop("scan_group")
    current["actor"].pop("scan_history_length")
    current["actor"].pop("scan_update_period")
    current["class_name"] = previous["class_name"]
    current["actor"]["class_name"] = previous["actor"]["class_name"]
    assert current == previous


def test_teacher_configuration_does_not_read_him_task_settings(monkeypatch):
    from nazarite.config.train_config.XZ1_env_cfgs import him_env_cfg, him_rl_cfg
    from nazarite.config.train_config.XZ1_env_cfgs.teacher_env_cfg import (
        XZ1_TEACHER_NUM_ENVS,
        Nazarite_WMP_HIM_Teacher_XZ1,
    )
    from nazarite.config.train_config.XZ1_env_cfgs.teacher_rl_cfg import (
        xz1_height_history_training_cfg,
    )

    def unrelated_task_factory(*args, **kwargs):
        raise AssertionError("Teacher must not call the base HIM task factory")

    monkeypatch.setattr(him_env_cfg, "_build_xz1_env", unrelated_task_factory)
    monkeypatch.setattr(him_env_cfg, "XZ1_HIM_NUM_ENVS", 13)
    monkeypatch.setattr(him_env_cfg, "XZ1_HIM_POSE_WEIGHT", -123.0)
    monkeypatch.setattr(him_rl_cfg, "xz1_him_runner_cfg", unrelated_task_factory)
    env = Nazarite_WMP_HIM_Teacher_XZ1(scan_history_length=6)
    rl = xz1_height_history_training_cfg()
    assert env.scene.num_envs == XZ1_TEACHER_NUM_ENVS
    assert env.rewards["pose"].weight == 0.2
    assert rl.algorithm.entropy_coef == 0.01
    assert rl.actor.scan_history_length == 1
    assert rl.actor.scan_update_period == 0.1
    # Different task configurations must never share mutable training settings.
    env.rewards["pose"].params["std_standing"][".*_hip_joint"] = 99.0
    rl.actor.estimator_cfg["temperature"] = 99.0
    other_env = Nazarite_WMP_HIM_Teacher_XZ1()
    other_rl = xz1_height_history_training_cfg()
    assert other_env.rewards["pose"].params["std_standing"][".*_hip_joint"] == 0.1
    assert other_rl.actor.estimator_cfg["temperature"] == 3.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA training")
def test_registered_task_warm_start_ppo_reset_resume_export_and_sim2sim(tmp_path):
    import onnxruntime as ort
    from sim2sim.teacher import TeacherPolicy

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    cfg = load_env_cfg("Nazarite-WMP-HIM-Teacher-XZ1")
    rl_cfg = load_rl_cfg("Nazarite-WMP-HIM-Teacher-XZ1")
    assert cfg.observations["teacher_scan"].history_length is None
    assert cfg.observations["teacher_scan"].terms["height_scan"].params == {
        "history_length": 1,
        "update_period": 0.1,
    }
    assert rl_cfg.actor.scan_history_length == 1
    assert rl_cfg.actor.scan_update_period == 0.1
    assert Path(rl_cfg.initial_checkpoint).is_file()
    assert Path(rl_cfg.initial_checkpoint).parts[-3:] == (
        "xz1_him_complex",
        "2026-10-25_22-04-31",
        "model_31300.pt",
    )
    cfg.scene.num_envs = 4
    cfg.scene.terrain.terrain_generator.num_rows = 2
    cfg.scene.terrain.terrain_generator.num_cols = 9
    cfg.auto_reset = False
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=10)
        raw, _ = env.reset()
        assert raw["teacher_scan"].shape == (4, 132)
        for _ in range(3):
            obs, _, _, _, _ = env.step(torch.randn(4, 12, device="cuda:0") * 0.1)
            torch.testing.assert_close(obs["teacher_scan"], raw["teacher_scan"])
        untouched = obs["teacher_scan"][1:].clone()
        obs, _ = env.reset(env_ids=torch.tensor([0], device="cuda:0"))
        torch.testing.assert_close(obs["teacher_scan"][1:], untouched)
        rl_cfg.num_steps_per_env = 6
        rl_cfg.algorithm.num_learning_epochs = 1
        rl_cfg.algorithm.num_mini_batches = 1
        rl_cfg.logger = "tensorboard"
        runner = XZ1TerrainTeacherRunner(
            wrapped, asdict(rl_cfg), str(tmp_path), "cuda:0"
        )
        assert not runner.alg.optimizer.state
        assert runner.alg.actor.mlp[0].weight.shape[1] == 98
        assert not runner.alg.actor.mlp[0].weight[:, 66:].any()
        from rsl_rl.models.him_actor_model import HIMActorModel

        initial_obs = wrapped.get_observations()
        kwargs = asdict(xz1_him_runner_cfg().actor)
        kwargs.pop("class_name")
        legacy = (
            HIMActorModel(initial_obs, {"actor": ["actor"]}, "actor", 12, **kwargs)
            .to("cuda:0")
            .eval()
        )
        source = torch.load(
            rl_cfg.initial_checkpoint, map_location="cuda:0", weights_only=False
        )
        legacy.load_state_dict(source["actor_state_dict"], strict=True)
        runner.alg.eval_mode()
        with torch.inference_mode():
            torch.testing.assert_close(
                runner.alg.actor(initial_obs), legacy(initial_obs), atol=2e-5, rtol=2e-5
            )
        runner.learn(num_learning_iterations=2)
        assert runner.alg.actor.mlp[0].weight[:, 66:].abs().sum() > 0
        saved = tmp_path / "single_10hz_from_him.pt"
        runner.save(str(saved))
        runner.alg.eval_mode()
        obs = wrapped.get_observations()
        with torch.inference_mode():
            expected = runner.alg.actor(obs).cpu().numpy()
            history = runner.alg.actor.get_observation_tensor(obs).cpu().numpy()
        ort_session = ort.InferenceSession(
            str(tmp_path / "policy.onnx"), providers=["CPUExecutionProvider"]
        )
        actual = ort_session.run(
            None,
            {
                "obs_history": history,
                "teacher_scan": obs["teacher_scan"].cpu().numpy(),
            },
        )[0]
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
        deploy = TeacherPolicy(saved)
        scan = obs["teacher_scan"][0].cpu().numpy()
        action = deploy.step(history[0], scan)
        np.testing.assert_allclose(action, expected[0], atol=2e-5, rtol=2e-5)
        assert deploy.scan_history_length == 1
        assert deploy.scan_update_steps == 5
        for _ in range(4):
            action = deploy.step(history[0], -scan)
            np.testing.assert_allclose(action, expected[0], atol=2e-5, rtol=2e-5)
        deploy.step(history[0], -scan)
        np.testing.assert_array_equal(deploy.scan_history[0], -scan)
        deploy.reset()
        assert deploy.scan_history is None
        rl_cfg.resume = True
        resumed = XZ1TerrainTeacherRunner(wrapped, asdict(rl_cfg), device="cuda:0")
        resumed.load(str(saved))
        assert resumed.alg.optimizer.state
        assert resumed.warm_start_source == runner.warm_start_source
    finally:
        env.close()


def test_scan_acquisition_holds_history_and_partial_reset_has_independent_clock(
    monkeypatch,
):
    from nazarite.config.train_config.XZ1_env_cfgs import teacher_env_cfg as module

    env = SimpleNamespace(num_envs=2, device="cpu", step_dt=0.02, common_step_counter=0)
    cfg = SimpleNamespace(params={"history_length": 3, "update_period": 0.1})
    scan = torch.zeros(2, 132)
    calls = []

    def read_scan(env):
        calls.append(env.common_step_counter)
        return scan.clone()

    monkeypatch.setattr(module, "parkour_height_scan", read_scan)
    term = module.SampledParkourHeightScan(cfg, env)
    first = term(env, **cfg.params).clone()
    for step in range(1, 5):
        env.common_step_counter = step
        scan.fill_(step)
        torch.testing.assert_close(term(env, **cfg.params), first)
    assert calls == [0]
    env.common_step_counter = 5
    scan.fill_(5)
    result = term(env, **cfg.params).clone()
    torch.testing.assert_close(
        result[:, :, 0], torch.tensor([[0.0, 0.0, 5.0]]).expand(2, -1)
    )
    # Recomputing at the same control step must not shift history twice.
    torch.testing.assert_close(term(env, **cfg.params), result)
    assert calls == [0, 5]
    env.common_step_counter = 7
    term.reset(torch.tensor([0]))
    scan[0].fill_(7)
    reset = term(env, **cfg.params).clone()
    torch.testing.assert_close(reset[0], torch.full((3, 132), 7.0))
    torch.testing.assert_close(reset[1], result[1])
    env.common_step_counter = 10
    scan.fill_(10)
    result = term(env, **cfg.params).clone()
    torch.testing.assert_close(result[0], reset[0])
    torch.testing.assert_close(result[1, :, 0], torch.tensor([0.0, 5.0, 10.0]))
    env.common_step_counter = 12
    scan.fill_(12)
    result = term(env, **cfg.params)
    torch.testing.assert_close(result[0, :, 0], torch.tensor([7.0, 7.0, 12.0]))
    torch.testing.assert_close(result[1, :, 0], torch.tensor([0.0, 5.0, 10.0]))


def test_sim2sim_acquires_three_distinct_scans_at_ten_hz_and_reset(monkeypatch):
    from sim2sim import teacher as deploy_module

    from nazarite.rl.xz1_teacher import teacher_scan_contract

    _, _, actor = models()
    monkeypatch.setattr(deploy_module, "load_teacher", lambda *args: actor)
    policy = deploy_module.TeacherPolicy(Path("unused.pt"))
    history = np.zeros(282, dtype=np.float32)
    for step in range(11):
        policy.step(history, np.full(132, step / 20, dtype=np.float32))
        expected = (
            [0.0, 0.0, 0.0]
            if step < 5
            else [0.0, 0.0, 0.25]
            if step < 10
            else [0.0, 0.25, 0.5]
        )
        np.testing.assert_array_equal(policy.scan_history[:, 0], expected)
    policy.reset()
    policy.step(history, np.full(132, 0.75, dtype=np.float32))
    np.testing.assert_array_equal(policy.scan_history, 0.75)
    assert teacher_scan_contract(3, 0.1)["history_dt"] == 0.1
    assert "update_period" not in teacher_scan_contract(6)
