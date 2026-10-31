"""Real CPU environment, PPO, checkpoint and two-input export coverage."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from nazarite.config.train_config.xz1.teacher_env_cfg import Nazarite_WMP_HIM_Eval_XZ1, Nazarite_WMP_HIM_Teacher_XZ1
from nazarite.config.train_config.xz1.teacher_rl_cfg import xz1_teacher_runner_cfg
from nazarite.rl.xz1.teacher import XZ1TerrainTeacherRunner
from nazarite.scripts.xz1.evaluate_xz1_wmp import evaluate_case

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper


@pytest.mark.slow
def test_training_managers_and_reference_scan_on_real_terrain():
  cfg = Nazarite_WMP_HIM_Teacher_XZ1()
  cfg.scene.num_envs = 4
  cfg.scene.terrain.terrain_generator.num_rows = 2
  cfg.seed = 42
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    obs, _ = env.reset()
    assert obs["actor"].shape == (4, 6, 47)
    assert obs["critic"].shape == (4, 244)
    assert obs["estimator"].shape == (4, 50)
    assert obs["teacher_scan"].shape == (4, 132)
    for _ in range(3):
      obs, reward, _, _, _ = env.step(torch.zeros(4, 12))
      assert torch.isfinite(reward).all()
      assert torch.isfinite(obs["teacher_scan"]).all()
      assert obs["teacher_scan"].abs().max() <= 1
    sensor = env.scene["teacher_terrain_scan"]
    data = sensor.data
    expected = (data.frame_pos_w[:, 0, 2:3] - 0.3 - data.hit_pos_w[..., 2]).clamp(-1, 1)
    expected = torch.where(data.distances < 0, 1.0, expected)
    torch.testing.assert_close(obs["teacher_scan"], expected)
    for manager in (env.reward_manager, env.termination_manager, env.metrics_manager):
      for term in manager._term_cfgs:
        result = term.func(env, **term.params)
        assert result.shape == (4,) and torch.isfinite(result.float()).all()
  finally:
    env.close()


@pytest.mark.slow
def test_warm_start_ppo_resume_export_and_evaluator(tmp_path):
  cfg = Nazarite_WMP_HIM_Eval_XZ1(terrain="flat")
  cfg.scene.num_envs = 4
  env = ManagerBasedRlEnv(cfg, device="cpu")
  teacher_checkpoint = tmp_path / "teacher.pt"
  try:
    wrapped = RslRlVecEnvWrapper(env, clip_actions=10.0)
    runner_cfg = xz1_teacher_runner_cfg()
    runner_cfg.num_steps_per_env = 4
    runner_cfg.algorithm.num_learning_epochs = 1
    runner_cfg.algorithm.num_mini_batches = 1
    checkpoint = Path("logs/rsl_rl/xz1_him_complex/2026-10-25_22-04-31/model_31300.pt")
    if checkpoint.is_file():
      runner_cfg.initial_checkpoint = str(checkpoint)
    runner = XZ1TerrainTeacherRunner(wrapped, asdict(runner_cfg), device="cpu")
    assert runner.current_learning_iteration == env.common_step_counter == 0
    assert not runner.alg.optimizer.state
    assert not runner.alg.estimator.optimizer.state
    if checkpoint.is_file():
      saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
      for key, value in saved["estimator_state_dict"].items():
        torch.testing.assert_close(value, runner.alg.estimator.state_dict()[key])
    first = runner.alg.actor.mlp[0].weight.detach().clone()
    estimator = runner.alg.estimator.encoder[0].weight.detach().clone()
    runner.learn(num_learning_iterations=2)
    assert runner.alg.optimizer.state
    assert runner.alg.estimator.optimizer.state
    assert not torch.equal(first, runner.alg.actor.mlp[0].weight)
    assert runner.alg.actor.mlp[0].weight[:, 66:].abs().sum() > 0
    assert not torch.equal(estimator, runner.alg.estimator.encoder[0].weight)
    assert runner._nan_recoveries == 0
    with pytest.raises(ValueError, match="fresh runner"):
      runner.warm_start(str(checkpoint))
    saved_counter = env.common_step_counter
    runner.save(str(teacher_checkpoint))
    onnx_path = tmp_path / "policy.onnx"
    model = onnx.load(onnx_path)
    assert [item.name for item in model.graph.input] == ["obs_history", "teacher_scan"]
    runner.alg.eval_mode()
    obs = wrapped.get_observations()
    with torch.no_grad():
      expected = runner.alg.actor(obs).numpy()
      history = runner.alg.actor.get_observation_tensor(obs).numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(
      None,
      {
        "obs_history": history,
        "teacher_scan": obs["teacher_scan"].numpy(),
      },
    )[0]
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
    runner_cfg.initial_checkpoint = None
    resume = XZ1TerrainTeacherRunner(wrapped, asdict(runner_cfg), device="cpu")
    resume.load(str(teacher_checkpoint))
    assert resume.current_learning_iteration == runner.current_learning_iteration
    assert resume.alg.optimizer.state
    assert resume.alg.estimator.optimizer.state
    assert env.common_step_counter == saved_counter
    assert resume.warm_start_source == runner.warm_start_source
    for key, value in runner.alg.actor.state_dict().items():
      torch.testing.assert_close(value, resume.alg.actor.state_dict()[key])
  finally:
    env.close()
  args = SimpleNamespace(
    speed=0.4,
    seed=42,
    num_envs=2,
    episode_seconds=0.04,
    device="cpu",
    policy="teacher",
    checkpoint=teacher_checkpoint,
    episodes=3,
  )
  result = evaluate_case(args, "flat", 0.0)
  assert len(result["episodes"]) == 3
  assert result["success_rate"] == 0
  assert result["failure_causes"] == {"time_out": 3}
  assert evaluate_case(args, "flat", 0.0) == result


@pytest.mark.slow
def test_teacher_scan_sees_wall_top_above_trunk():
  cfg = Nazarite_WMP_HIM_Eval_XZ1(terrain="high_wall", difficulty=1.0)
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    env.reset()
    robot = env.scene["robot"]
    pose = robot.data.root_link_pose_w.clone()
    # The 0.30 m wall is 2 m ahead of this origin. Place a scan x=0.6
    # exactly on the top, with the trunk below the wall's top surface.
    pose[:, 0] = env.scene.env_origins[:, 0] + 1.4
    pose[:, 2] = env.scene.env_origins[:, 2] + 0.20
    robot.data.write_root_pose(pose)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.sim.sense()
    scan = env.observation_manager.compute_group("teacher_scan")
    # x=0.6 is index 7, y=0 is index 5. Formula: .20-.30-.30=-.40.
    assert scan[0, 7 * 11 + 5].item() == pytest.approx(-0.40, abs=1e-4)
  finally:
    env.close()
