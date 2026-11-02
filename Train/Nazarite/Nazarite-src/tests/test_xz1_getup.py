"""XZ1 Playground port: physical resets, action semantics, smoothing and PPO."""

import json
import math
from dataclasses import asdict
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import mujoco
import numpy as np
import onnx
import pytest
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv, mdp
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.sensor import ContactSensor
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from nazarite.config.robot_config.xz1_cfg import get_spec
from nazarite.mdp.xz1 import getup
from nazarite.rl.xz1.getup_runner import XZ1GetupRunner
from nazarite.scripts.xz1.evaluate_frnet_recovery import EvaluateConfig, evaluate

TASK = "Nazarite-FRNet-Recovery-XZ1"


def _support_env(num_envs: int) -> ManagerBasedRlEnv:
    sensor = Mock(spec=ContactSensor)
    sensor.data = SimpleNamespace(force=torch.zeros(num_envs, 4, 3))
    return cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=num_envs,
            device="cpu",
            step_dt=0.02,
            extras={
                "getup_action_age": torch.full((num_envs,), 26),
                "settle_mask": torch.ones(num_envs, dtype=torch.bool),
                "frnet_recovery_mask": torch.ones(num_envs, dtype=torch.bool),
            },
            scene={
                "feet_ground_contact": sensor,
                "robot": SimpleNamespace(
                    data=SimpleNamespace(
                        projected_gravity_b=torch.tensor([[0.0, 0, -1]]).repeat(
                            num_envs, 1
                        ),
                        root_link_ang_vel_b=torch.zeros(num_envs, 3),
                        actuator_force=torch.full((num_envs, 12), 10.0),
                    )
                ),
                "base_height_scan": SimpleNamespace(
                    data=SimpleNamespace(heights=torch.full((num_envs, 1), 0.3))
                ),
            },
        ),
    )


def test_feet_support_requires_height_upright_and_finished_settle():
    env = _support_env(7)
    force = env.scene["feet_ground_contact"].data.force
    force[:, :, 2] = 10
    force[1, 3] = 0
    force[2, 2:] = 0
    env.scene["robot"].data.projected_gravity_b[3, 2] = 1
    env.scene["base_height_scan"].data.heights[4] = 0.1
    env.scene["robot"].data.root_link_ang_vel_b[5, 0] = 2
    env.extras["getup_action_age"][6] = 25
    torch.testing.assert_close(
        getup.feet_support_reward(env), torch.tensor([1, 0.5625, 0.25, 0, 0, 0, 0])
    )
    # Stance resets need not wait for a nonexistent drop/settle phase.
    env.extras["settle_mask"][6] = False
    assert getup.feet_support_reward(env)[6] == 1
    torch.testing.assert_close(mdp.joint_torques_l2(env), torch.full((7,), 1200.0))
    torch.testing.assert_close(getup.joint_torque_rms(env), torch.full((7,), 10.0))


def test_landing_window_one_off_bonus_and_partial_reset():
    env = _support_env(3)
    term = getup.SynchronizedLandingReward(
        RewardTermCfg(func=getup.SynchronizedLandingReward, weight=0.1), env
    )
    force = env.scene["feet_ground_contact"].data.force
    force[:, :, 2] = 10
    assert not term(env).any()  # Initial contact is not a touchdown.
    force.zero_()
    term(env)
    term(env)  # All feet have now been airborne for 0.04 s.
    force[:, :2, 2] = 10
    force[0, 2:, 2] = 10
    torch.testing.assert_close(term(env) * env.step_dt, torch.tensor([1.0, 0, 0]))
    term(env)
    force[1, 2:, 2] = 10
    torch.testing.assert_close(term(env) * env.step_dt, torch.tensor([0, 1.0, 0]))
    for _ in range(6):
        assert not term(env).any()
    force[2, 2:, 2] = 10
    assert not term(env).any()  # First pair landed more than 0.12 s ago.
    force.zero_()
    term(env)
    term(env)
    force[:, :, 2] = 10
    torch.testing.assert_close(term(env) * env.step_dt, torch.tensor([0, 0, 1.0]))
    assert not term(env).any()  # No repeated bonus for standing or hopping.
    term.reset(torch.tensor([0]))
    assert not term(env).any()  # Reset cannot turn existing contact into landing.
    force.zero_()
    term(env)
    term(env)
    force[:, :, 2] = 10
    torch.testing.assert_close(term(env) * env.step_dt, torch.tensor([1.0, 0, 0]))


def test_landing_excludes_settle_stance_and_belly_contacts():
    env = _support_env(3)
    term = getup.SynchronizedLandingReward(
        RewardTermCfg(func=getup.SynchronizedLandingReward, weight=0.1), env
    )
    env.extras["getup_action_age"][0] = 25
    env.extras["frnet_recovery_mask"][1] = False
    env.scene["base_height_scan"].data.heights[2] = 0.1
    term(env)
    term(env)
    force = env.scene["feet_ground_contact"].data.force
    force[:, :, 2] = 10
    assert not term(env).any()
    env.extras["getup_action_age"][0] = 26
    assert not term(env).any()  # Ignore touchdown during the passive settle.


def test_playground_rewards_distinguish_supine_and_height():
    gravity = torch.tensor([[0.0, 0, -1], [1.0, 0, 0], [0.0, 0, 1]])
    env = cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            scene={
                "robot": SimpleNamespace(
                    data=SimpleNamespace(projected_gravity_b=gravity)
                ),
                "base_height_scan": SimpleNamespace(
                    data=SimpleNamespace(heights=torch.tensor([[0.3], [0.15], [0.6]]))
                ),
            }
        ),
    )
    torch.testing.assert_close(
        getup.orientation_reward(env), torch.tensor([1.0, math.exp(-4), math.exp(-8)])
    )
    torch.testing.assert_close(
        getup.height_reward(env, 0.3),
        torch.tensor([1.0, math.expm1(0.15) / math.expm1(0.3), 1.0]),
    )


def test_random_drop_bank_covers_rotations_and_soft_joints_without_floor_overlap():
    spec = get_spec()
    spec.worldbody.add_geom(
        name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0, 0, 0.1)
    )
    model = spec.compile()
    names = tuple(
        model.joint(i).name
        for i in range(model.njnt)
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
    )
    bank = getup.build_drop_bank(model, names, seed=7, samples=128, prefix="")
    data = mujoco.MjData(model)
    addresses = [model.joint(name).qposadr[0] for name in names]
    collision_ids = [
        i
        for i in range(model.ngeom)
        if model.geom_bodyid[i] and (model.geom_contype[i] or model.geom_conaffinity[i])
    ]
    upright = []
    for pose in bank:
        data.qpos[:3] = (0, 0, max(0.5, pose[0]))
        data.qpos[3:7] = pose[1:5]
        data.qpos[addresses] = pose[5:]
        mujoco.mj_kinematics(model, data)
        distances = [
            mujoco.mj_geomDistance(model, data, model.geom("ground").id, i, 2.0, None)
            for i in collision_ids
        ]
        assert min(distances) >= 0.008 - 1e-6
        upright.append(data.xmat[model.body("base_link").id].reshape(3, 3)[2, 2])
    # Uniform random rotations include actual back, side and upright starts.
    assert min(upright) < -0.95 and max(upright) > 0.95
    assert np.abs(upright).min() < 0.05
    assert bank[:, 5:].std(0).min() > 0.2


def test_new_baseline_configuration_is_plain_ppo_and_legacy_is_available():
    cfg = load_env_cfg(TASK)
    play = load_env_cfg(TASK, play=True)
    rl = load_rl_cfg(TASK)
    assert isinstance(rl, RslRlOnPolicyRunnerCfg)
    assert set(cfg.observations) == {"actor", "critic"}
    assert cfg.episode_length_s == 6
    action_cfg = cfg.actions["joint_pos"]
    assert isinstance(action_cfg, getup.SettleRelativeJointPositionActionCfg)
    assert action_cfg.scale == 0.6
    assert cfg.events["reset_base"].params["fall_probability"] == 0.6
    assert play.events["reset_base"].params["fall_probability"] == 1
    assert rl.algorithm.class_name == "PPO" and rl.actor.class_name == "MLPModel"
    assert rl.algorithm.learning_rate == 0.001
    assert rl.clip_actions is None
    assert rl.experiment_name == "xz1_getup_flat"
    assert "recovery" not in cfg.curriculum
    assert not play.curriculum
    assert play.rewards["action_rate_l2"].weight == -0.1
    assert play.rewards["joint_torques_l2"].weight == -1e-5
    legacy = load_env_cfg("Nazarite-FRNet-Paper-Recovery-XZ1")
    assert "frnet_aux" in legacy.observations
    assert "recovery" in legacy.curriculum


def test_cpu_getup_control_schedules_training_and_eval(tmp_path):
    cfg = load_env_cfg(TASK)
    cfg.scene.num_envs = 4
    cfg.seed = 42
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        raw_obs, _ = env.reset()
        obs = cast(TensorDict, raw_obs)
        assert obs["actor"].shape == (4, 42)
        assert obs["critic"].shape == (4, 45)
        action = env.action_manager.get_term("joint_pos")
        assert isinstance(action, getup.SettleRelativeJointPositionAction)
        # Real compiled model: foot rolling/spin contact, no visual collisions.
        model = env.sim.mj_model
        for i in range(model.ngeom):
            name = model.geom(i).name
            if "foot_link_primitive" in name:
                assert model.geom_condim[i] == 6
            if name.startswith("robot/") and model.geom_group[i] == 1:
                assert not model.geom_contype[i] and not model.geom_conaffinity[i]
        # The clock must ignore an artificially randomized episode length.
        env.extras["settle_mask"][:] = torch.tensor([True, False, True, False])
        env.episode_length_buf.fill_(200)
        data = env.scene["robot"].data
        action.process_actions(torch.ones(4, 12))
        action.apply_actions()
        expected_delta = torch.tensor([0.0, 0.6, 0.0, 0.6])[:, None]
        torch.testing.assert_close(
            data.joint_pos_target, data.joint_pos + expected_delta - data.encoder_bias
        )
        # Unlike the old nominal-offset action, each apply reads current q.
        target_before = data.joint_pos_target.clone()
        initial_joints = data.joint_pos.clone()
        robot = env.scene["robot"]
        robot.write_joint_state_to_sim(initial_joints + 0.01, data.joint_vel)
        action.apply_actions()
        torch.testing.assert_close(data.joint_pos_target, target_before + 0.01)
        robot.write_joint_state_to_sim(initial_joints, data.joint_vel)
        action._age.fill_(25)
        action.process_actions(torch.ones(4, 12))
        action.apply_actions()
        torch.testing.assert_close(
            data.joint_pos_target, data.joint_pos + 0.6 - data.encoder_bias
        )
        ages = action._age.clone()
        untouched = data.joint_pos[[1, 3]].clone()
        env.reset(env_ids=torch.tensor([0, 2]))
        assert action._age[[0, 2]].eq(0).all()
        torch.testing.assert_close(action._age[[1, 3]], ages[[1, 3]])
        torch.testing.assert_close(data.joint_pos[[1, 3]], untouched)
        for _ in range(30):
            raw_obs, reward, _, _, _ = env.step(torch.zeros(4, 12))
            obs = cast(TensorDict, raw_obs)
            assert torch.isfinite(reward).all()
            assert all(torch.isfinite(obs[key]).all() for key in ("actor", "critic"))
        assert env.scene["self_collision"].data.force_history is not None
        assert env.scene["base_height_scan"].data.distances.ge(0).all()
        for iteration, rate, velocity, torque, power in (
            (0, -0.01, 0.0, -2e-6, float("inf")),
            (500, -0.05, -0.005, -5e-6, 1000.0),
            (900, -0.08, -0.008, -8e-6, 700.0),
            (1200, -0.1, -0.01, -1e-5, 400.0),
        ):
            env.common_step_counter = iteration * 24
            env.reset()
            assert env.reward_manager.get_term_cfg("action_rate_l2").weight == rate
            assert env.reward_manager.get_term_cfg("joint_vel_l2").weight == velocity
            assert env.reward_manager.get_term_cfg("joint_torques_l2").weight == torque
            assert (
                env.termination_manager.get_term_cfg("energy").params["threshold"]
                == power
            )
        # Settle power protection uses the same independent age as the action.
        env.extras["getup_action_age"] = action._age
        action._age.zero_()
        assert not getup.energy_termination(env, threshold=-1.0).any()
        action._age.fill_(26)
        assert getup.energy_termination(env, threshold=-1.0).all()
        env.reset()
        runner_cfg = load_rl_cfg(TASK)
        assert isinstance(runner_cfg, RslRlOnPolicyRunnerCfg)
        runner_cfg.num_steps_per_env = 4
        runner_cfg.algorithm.num_learning_epochs = 1
        runner_cfg.algorithm.num_mini_batches = 1
        wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
        runner_cls = load_runner_cls(TASK)
        assert runner_cls is XZ1GetupRunner
        runner = runner_cls(wrapped, asdict(runner_cfg), device="cpu")
        actor = cast(MLPModel, runner.alg.actor)
        linear = cast(torch.nn.Linear, actor.mlp[0])
        before = linear.weight.detach().clone()
        runner.learn(num_learning_iterations=1)
        assert not torch.equal(before, linear.weight)
        checkpoint = tmp_path / "model.pt"
        runner.save(str(checkpoint))
        exported = checkpoint.with_name(tmp_path.name + ".onnx")
        assert exported.is_file()
        metadata = {item.key: item.value for item in onnx.load(exported).metadata_props}
        assert metadata["policy_family"] == "playground_getup_ppo_v1"
        assert metadata["target_update"] == "every_physics_substep"
        env.common_step_counter = 0
        with torch.inference_mode():
            env.reset()
        assert env.reward_manager.get_term_cfg("action_rate_l2").weight == -0.01
        runner.load(str(checkpoint), map_location="cpu")
        assert env.common_step_counter >= 1200 * 24
        assert env.reward_manager.get_term_cfg("action_rate_l2").weight == -0.1
        assert env.reward_manager.get_term_cfg("joint_torques_l2").weight == -1e-5
        obs = cast(TensorDict, wrapped.get_observations())
        with torch.no_grad():
            torch.testing.assert_close(
                runner.alg.actor(obs), runner.alg.actor.as_jit()(obs["actor"])
            )
        incompatible = tmp_path / "old_frnet.pt"
        torch.save(
            {"actor_state_dict": {"mcp_encoder.0.weight": torch.zeros(1)}}, incompatible
        )
        with pytest.raises(ValueError, match="Nazarite-FRNet-Paper-Recovery-XZ1"):
            runner.load(str(incompatible), map_location="cpu")
        # Exercise automatic episode reset inside the same inference context
        # used by PPO and play (contact-history buffers use inference tensors).
        with torch.inference_mode():
            env.episode_length_buf.fill_(env.max_episode_length - 1)
            _, reward, terminated, truncated, _ = env.step(torch.zeros(4, 12))
            assert (terminated | truncated).all() and torch.isfinite(reward).all()
            assert action._age.eq(0).all()
    finally:
        env.close()
    output = tmp_path / "eval.json"
    metrics = evaluate(
        EvaluateConfig(
            checkpoint_file=checkpoint,
            num_envs=2,
            horizon_s=0.06,
            stable_duration_s=0.04,
            device="cpu",
            output_file=output,
        )
    )
    assert set(metrics["falls"]) == {
        "left_side",
        "right_side",
        "back",
        "belly",
        "random_drop",
    }
    assert all(row["invalid_state_rate"] == 0 for row in metrics["falls"].values())
    assert all(
        "standing_joint_speed_rms_rad_s" in row for row in metrics["falls"].values()
    )
    assert metrics == json.loads(output.read_text())
