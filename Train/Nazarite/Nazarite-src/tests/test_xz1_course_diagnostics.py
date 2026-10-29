"""Original-type/row outcomes and recovery evidence survive automatic resets."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from rsl_rl.runners.him_on_policy_runner import HIMOnPolicyRunner
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from nazarite.config.train_config.XZ1_env_cfgs.him_env_cfg import (
    Nazarite_HIM_Complex_Terrain_XZ1,
)
from nazarite.config.train_config.XZ1_env_cfgs.teacher_env_cfg import (
    Nazarite_WMP_HIM_Teacher_XZ1,
)
from nazarite.config.train_config.XZ1_env_cfgs.teacher_rl_cfg import (
    xz1_teacher_runner_cfg,
)
from nazarite.mdp import xz1_diagnostics as diagnostics
from nazarite.rl.xz1_teacher import XZ1TerrainTeacherRunner


class Scene(dict):
    pass


def fake_env():
    cfg = Nazarite_WMP_HIM_Teacher_XZ1()
    generator = cfg.scene.terrain.terrain_generator
    names = list(generator.sub_terrains)
    scene = Scene(
        robot=SimpleNamespace(
            data=SimpleNamespace(
                root_link_pos_w=torch.zeros(3, 3),
                projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]] * 3),
            )
        )
    )
    scene.terrain = SimpleNamespace(
        cfg=SimpleNamespace(terrain_generator=generator),
        terrain_types=torch.tensor(
            [names.index("gap"), names.index("narrow_stairs"), names.index("gap")]
        ),
        terrain_levels=torch.tensor([2, 5, 3]),
    )
    scene.env_origins = torch.zeros(3, 3)
    env = SimpleNamespace(
        cfg=cfg,
        scene=scene,
        num_envs=3,
        device="cpu",
        common_step_counter=100,
        episode_length_buf=torch.ones(3, dtype=torch.long),
        reset_terminated=torch.tensor([True, False, False]),
        termination_manager=SimpleNamespace(
            get_term=lambda _: torch.tensor([False, False, True])
        ),
        sim=SimpleNamespace(
            data=SimpleNamespace(
                qpos=torch.zeros(3, 19),
                qvel=torch.zeros(3, 18),
                qacc=torch.zeros(3, 18),
            )
        ),
    )
    return env, diagnostics.XZ1CourseDiagnostics(None, env)


def test_wmp_fixed_type_curriculum_and_him_independence():
    cfg = Nazarite_WMP_HIM_Teacher_XZ1()
    baseline = Nazarite_HIM_Complex_Terrain_XZ1()
    assert not cfg.curriculum["terrain_levels"].params["adaptive_type_sampling"]
    assert cfg.curriculum["terrain_levels"].params["randomize_completed_levels"]
    assert cfg.scene.terrain.max_init_terrain_level == 0
    assert baseline.curriculum["terrain_levels"].params["adaptive_type_sampling"]
    assert not baseline.curriculum["terrain_levels"].params[
        "randomize_completed_levels"
    ]
    assert baseline.scene.terrain.max_init_terrain_level == 1


def test_outcomes_use_ended_episode_type_level_and_real_timeout(monkeypatch):
    env, recorder = fake_env()
    monkeypatch.setattr(
        diagnostics,
        "narrow_course_complete",
        lambda _: torch.tensor([False, True, False]),
    )
    recorder.record_pre_reset(torch.arange(3))
    # Simulate reset/curriculum changing the row AFTER the pre-reset hook.
    env.scene.terrain.terrain_levels[:] = 9
    scalars = recorder.scalars()
    assert scalars["Terrain/gap/level_2/episodes"] == 1
    assert scalars["Terrain/gap/level_2/success_rate"] == 0
    assert scalars["Terrain/gap/level_3/timeout_rate"] == 1
    assert scalars["Terrain/narrow_stairs/level_5/success_rate"] == 1
    assert scalars["Terrain/narrow_stairs/level_5/timeout_rate"] == 0
    assert "Terrain/gap/level_9/success_rate" not in scalars
    assert scalars["Terrain/gap/all/timeout_rate"] == 0.5
    recorder.record_pre_reset(torch.arange(3))
    assert recorder.counts[..., 0].sum() == 3
    env.common_step_counter += 1
    env.episode_length_buf[:] = 0
    recorder.record_pre_reset(torch.arange(3))
    assert recorder.counts[..., 0].sum() == 3


def test_cached_terminal_scene_is_saved_before_recovery_reset(tmp_path, monkeypatch):
    env, recorder = fake_env()
    env.num_envs = 3
    recorder.pending_scenes[0] = {
        "terrain_type": "gap",
        "terrain_level": 2,
        "root_pose_w": torch.tensor([float("nan")]),
        "joint_pos": torch.tensor([4.0]),
        "contacts": {"geom_ids": [[17, 23]], "geom_names": [["foot", "gap_floor"]]},
    }
    # The live environment has already been auto-reset to a different row.
    env.scene.terrain.terrain_levels[0] = 9
    recorder.last_recorded_step[0] = env.common_step_counter
    runner = object.__new__(XZ1TerrainTeacherRunner)
    runner.env = SimpleNamespace(unwrapped=env)
    runner.logger = SimpleNamespace(log_dir=str(tmp_path))
    runner._nan_recoveries = 0
    runner.current_learning_iteration = 8
    monkeypatch.setattr(
        "nazarite.rl.xz1_teacher.capture_failure_scenes",
        Mock(side_effect=AssertionError("must use pre-reset scene")),
    )
    runner._record_nonfinite_envs(
        torch.tensor([0]), TensorDict({"actor": torch.zeros(3, 2)}, batch_size=[3])
    )
    saved = torch.load(next((tmp_path / "numerics").glob("*.pt")), weights_only=False)
    assert saved["scenes"][0]["terrain_level"] == 2
    assert saved["scenes"][0]["contacts"]["geom_ids"] == [[17, 23]]
    assert torch.isnan(saved["scenes"][0]["root_pose_w"]).all()
    assert not recorder.pending_scenes


def test_scene_capture_failure_does_not_disable_numerical_recovery():
    def reset(*, env_ids):
        return {"actor": torch.ones(2, 2)}, {}

    env = SimpleNamespace(num_envs=2, device="cpu", reset=reset)
    runner = SimpleNamespace(
        env=SimpleNamespace(unwrapped=env, num_envs=2),
        _nan_recoveries=0,
        _record_nonfinite_envs=Mock(side_effect=OSError("disk full")),
    )
    obs = TensorDict(
        {"actor": torch.tensor([[float("nan"), 0.0], [0.0, 0.0]])}, batch_size=[2]
    )
    rewards, dones = torch.ones(2), torch.zeros(2, dtype=torch.bool)
    with pytest.warns(UserWarning, match="scene capture failed"):
        HIMOnPolicyRunner._recover_nonfinite_envs(runner, obs, rewards, dones)
    assert runner._nan_recoveries == 1
    assert dones.tolist() == [True, False]
    assert rewards.tolist() == [0.0, 1.0]
    assert torch.isfinite(obs["actor"]).all()


@pytest.mark.slow
def test_real_teacher_warm_start_contact_capture_and_ppo_on_gpu(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    checkpoint = Path(
        "logs/rsl_rl/xz1_wmp_him_teacher/"
        "2026-10-27_22-45-01_stage3_recycle_curriculum/model_1499.pt"
    )
    if not checkpoint.is_file():
        pytest.skip("User-local checkpoint is not present")
    cfg = Nazarite_WMP_HIM_Teacher_XZ1()
    cfg.scene.num_envs = 18
    cfg.seed = 42
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=10.0)
        runner_cfg = xz1_teacher_runner_cfg()
        runner_cfg.initial_checkpoint = str(checkpoint)
        runner_cfg.num_steps_per_env = 4
        runner_cfg.algorithm.num_learning_epochs = 1
        runner_cfg.algorithm.num_mini_batches = 1
        runner = XZ1TerrainTeacherRunner(
            wrapped, asdict(runner_cfg), str(tmp_path), "cuda:0"
        )
        assert env.common_step_counter == runner.current_learning_iteration == 0
        assert not runner.alg.optimizer.state
        assert not runner.alg.estimator.optimizer.state
        assert env.scene.terrain.terrain_levels.eq(0).all()
        assert env._delta_success_streak.eq(0).all()
        source = torch.load(checkpoint, map_location="cuda:0", weights_only=False)
        for name, value in runner.alg.actor.state_dict().items():
            torch.testing.assert_close(
                value, source["actor_state_dict"][name], atol=0, rtol=0
            )
        with pytest.raises(ValueError, match="curriculum contract changed"):
            runner.load(str(checkpoint))
        original_types = env.scene.terrain.terrain_types.clone()
        # Let the initially unloaded feet settle before asserting contacts.
        for _ in range(20):
            wrapped.step(torch.zeros(18, 12, device="cuda:0"))
        ids = torch.tensor([0], device="cuda:0")
        scenes = diagnostics.capture_failure_scenes(env, ids, phase="test")
        assert scenes[0]["joint_pos"].shape == (12,)
        assert scenes[0]["root_pose_w"].shape == (7,)
        assert scenes[0]["contacts"]["geom_ids"]
        assert len(scenes[0]["contacts"]["geom_names"]) == len(
            scenes[0]["contacts"]["geom_ids"]
        )
        # Inject the detection flag, not a solver NaN, to exercise production
        # recovery/evidence paths without damaging the physics state.
        env._him_numerics_bad[0] = True
        env._xz1_course_diagnostics.record_pre_reset(ids)
        runner._nan_recoveries = 0
        obs = wrapped.get_observations()
        runner._recover_nonfinite_envs(
            obs,
            torch.ones(18, device="cuda:0"),
            torch.zeros(18, device="cuda:0", dtype=torch.long),
            {},
        )
        assert runner._nan_recoveries == 1
        saved = torch.load(
            next((tmp_path / "numerics").glob("*.pt")), weights_only=False
        )
        assert saved["scenes"][0]["phase"] == "before_auto_reset"
        assert saved["scenes"][0]["contacts"]["geom_ids"]
        runner.learn(num_learning_iterations=1)
        torch.testing.assert_close(env.scene.terrain.terrain_types, original_types)
        assert runner.alg.optimizer.state
        assert runner.alg.estimator.optimizer.state
        assert (tmp_path / "terrain_outcomes.json").is_file()
        assert (tmp_path / "model_0.pt").is_file()
    finally:
        env.close()
