"""XZ1 FR-Net geometry, label semantics and real CPU training coverage."""

import json
from dataclasses import asdict
from typing import cast

import mujoco
import numpy as np
import pytest
import torch
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.curriculum_manager import CurriculumManager
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.string import resolve_matching_names_values
from nazarite.config.robot_config.xz1_cfg import XZ1_INIT_STATE, get_spec, get_xz1_cfg
from nazarite.mdp.fr_net.frnet_observations import frnet_mass_target
from nazarite.mdp.xz1.frnet import (
    FALL_CASES,
    RecoveryJointPositionAction,
    build_flat_reset_bank,
)
from nazarite.mdp.xz1.frnet_curriculum import RECOVERY_STAGES, RecoveryCurriculum
from nazarite.rl.fr_net.frnet_actor import FRNetActor
from nazarite.rl.fr_net.frnet_ppo import FRNetPPO
from nazarite.scripts.xz1.evaluate_frnet_recovery import EvaluateConfig, evaluate

TASK_ID = "Nazarite-FRNet-Paper-Recovery-XZ1"


@pytest.fixture(scope="module")
def native_robot():
    spec = get_spec()
    spec.worldbody.add_geom(
        name="test_ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0, 0, 0.1)
    )
    model = spec.compile()
    names = tuple(
        model.joint(i).name
        for i in range(model.njnt)
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
    )
    assert XZ1_INIT_STATE.joint_pos is not None
    _, ordered, values = resolve_matching_names_values(XZ1_INIT_STATE.joint_pos, names)
    assert tuple(ordered) == names
    return model, names, np.asarray(values)


def test_pose_bank_has_floor_clearance_real_supine_and_low_belly(native_robot):
    model, names, defaults = native_robot
    bank = build_flat_reset_bank(model, names, defaults, prefix="", samples_per_case=8)
    data = mujoco.MjData(model)
    addresses = np.array([model.joint(name).qposadr[0] for name in names])
    ground_id = model.geom("test_ground").id
    collision_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if model.geom_bodyid[geom_id] != 0
        and (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id])
    ]
    for case_id, case in enumerate(FALL_CASES):
        for pose in bank[case_id]:
            data.qpos[:3] = (0, 0, pose[0])
            data.qpos[3:7] = pose[1:5]
            data.qpos[addresses] = pose[5:]
            mujoco.mj_kinematics(model, data)
            # Check against MuJoCo's geometry distance API, independently of
            # the analytic support functions used to construct the pose bank.
            distances = [
                mujoco.mj_geomDistance(model, data, ground_id, geom_id, 2.0, None)
                for geom_id in collision_ids
            ]
            assert min(distances) == pytest.approx(0.008, abs=1e-6)
            rotation = data.xmat[model.body("base_link").id].reshape(3, 3)
            if case == "back":
                assert rotation[2, 2] < -0.95  # Feet point upwards.
            if case == "belly":
                assert pose[0] < 0.25  # Not an already-standing reset.
            if case == "standing":
                np.testing.assert_allclose(pose[5:], defaults)


def test_registration_is_flat_and_does_not_mutate_robot_or_go2():
    train = load_env_cfg(TASK_ID)
    play = load_env_cfg(TASK_ID, play=True)
    assert train.scene.terrain is not None
    assert train.scene.terrain.terrain_type == "plane"
    assert train.commands == {} and not any(
        "terrain" in key for key in train.curriculum
    )
    assert "recovery_success" not in train.terminations
    assert "recovery_success" not in play.terminations
    assert play.events["reset_base"].params["standing_probability"] == 0
    assert not play.events["reset_base"].params["use_curriculum"]
    assert train.events["reset_base"].params["use_curriculum"]
    baseline_articulation = get_xz1_cfg().articulation
    articulation = train.scene.entities["robot"].articulation
    assert baseline_articulation is not None and articulation is not None
    assert baseline_articulation.actuators[0].delay_max_lag == 9
    assert articulation.actuators[0].delay_max_lag == 0
    go2 = load_env_cfg("Nazarite-FRNet-Recovery-Go2")
    assert "frnet_terrain_height_map" in {sensor.name for sensor in go2.scene.sensors}
    assert go2.observations["critic"].terms["frnet_mass_target"].params["body_groups"][
        0
    ] == ("FL_hip", "FL_thigh", "FL_calf")


@pytest.mark.parametrize("level", [0, 2, 5])
def test_curriculum_poses_have_geometry_clearance_and_progressive_roll(
    native_robot, level
):
    model, names, defaults = native_robot
    stage = RECOVERY_STAGES[level]
    bank = build_flat_reset_bank(
        model,
        names,
        defaults,
        prefix="",
        samples_per_case=4,
        side_roll_range=tuple(np.radians(stage.side_roll)),
        back_roll_range=tuple(np.radians(stage.back_roll)),
        joint_noise=stage.joint_noise,
        folded_probability=0.5,
    )
    data = mujoco.MjData(model)
    addresses = [model.joint(name).qposadr[0] for name in names]
    collision_ids = [
        i
        for i in range(model.ngeom)
        if model.geom_bodyid[i] and (model.geom_contype[i] or model.geom_conaffinity[i])
    ]
    for case_id, poses in enumerate(bank):
        for pose in poses:
            data.qpos[:3] = (0, 0, pose[0])
            data.qpos[3:7] = pose[1:5]
            data.qpos[addresses] = pose[5:]
            mujoco.mj_kinematics(model, data)
            distances = [
                mujoco.mj_geomDistance(
                    model, data, model.geom("test_ground").id, i, 2.0, None
                )
                for i in collision_ids
            ]
            assert min(distances) == pytest.approx(0.008, abs=1e-6)
            if case_id in (0, 1):
                tilt = np.degrees(
                    np.arccos(data.xmat[model.body("base_link").id].reshape(3, 3)[2, 2])
                )
                assert stage.side_roll[0] - 1 <= tilt <= stage.side_roll[1] + 1


def test_vae_next_frame_alignment_reset_mask_gradients_and_policy_replay():
    """Next targets survive shuffling/end of rollout without crossing resets."""

    def observations(time):
        return TensorDict(
            {
                "actor": torch.full((2, 210), float(time)),
                "critic": torch.zeros(2, 3),
                "frnet_aux": torch.cat((torch.ones(2, 4), torch.zeros(2, 13)), -1),
                "frnet_proprio": torch.full((2, 42), float(time + 10)),
            },
            batch_size=[2],
        )

    obs = observations(0)
    groups = {"actor": ["actor"], "critic": ["critic"]}
    actor = FRNetActor(
        obs,
        groups,
        "actor",
        12,
        hidden_dims=(32,),
        mcp_hidden_dims=(32,),
        decoder_hidden_dims=(32,),
        use_vae=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.8},
    )
    critic = MLPModel(obs, groups, "critic", 1, hidden_dims=(32,))
    storage = RolloutStorage("rl", 2, 2, obs, [12])
    alg = FRNetPPO(actor, critic, storage, num_learning_epochs=1, num_mini_batches=1)
    with torch.no_grad():
        for time in range(2):
            alg.act(observations(time))
            alg.process_env_step(
                observations(time + 1),
                torch.zeros(2),
                torch.tensor([False, time == 0]),
                {},
            )
    assert "frnet_next_proprio" not in obs
    torch.testing.assert_close(
        storage.observations["frnet_next_proprio"][0, 0], torch.full((42,), 11.0)
    )
    assert not storage.observations["frnet_next_valid"][0, 1].item()
    assert storage.observations["frnet_next_proprio"][0, 1].eq(0).all()
    # The final transition uses obs at t+1, even beyond the rollout boundary.
    assert storage.observations["frnet_next_proprio"][1].eq(12).all()
    batch = next(storage.mini_batch_generator(1, 1))
    sample = batch.observations
    assert sample is not None
    valid = sample["frnet_next_valid"].squeeze(-1)
    torch.testing.assert_close(
        sample["frnet_next_proprio"][valid, 0], sample["actor"][valid, 0] + 11
    )
    actor(sample, stochastic_output=True)
    assert batch.actions is not None
    old_log_prob = actor.get_output_log_prob(batch.actions).detach().clone()
    rec, kl = alg._vae_losses(sample)
    assert torch.isfinite(rec + kl) and kl > 0
    (rec + 0.01 * kl).backward()
    assert actor.mcp_decoder is not None and actor.latent_logvar_head is not None
    for module in (actor.mcp_decoder, actor.latent_logvar_head):
        gradients = [p.grad for p in module.parameters()]
        assert all(grad is not None for grad in gradients)
        assert sum(grad.abs().sum() for grad in gradients if grad is not None) > 0
    # Resampling the decoder must never change PPO's action likelihood.
    actor(sample, stochastic_output=True)
    torch.testing.assert_close(actor.get_output_log_prob(batch.actions), old_log_prob)
    sample["frnet_next_valid"] = torch.zeros_like(sample["frnet_next_valid"])
    rec, kl = alg._vae_losses(sample)
    assert rec.item() == 0 and torch.isfinite(kl)


@pytest.mark.slow
def test_cpu_observations_partial_reset_ppo_and_checkpoint(tmp_path):
    cfg = load_env_cfg(TASK_ID)
    cfg.scene.num_envs = 4
    cfg.seed = 42
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        raw_obs, _ = env.reset()
        obs = cast(TensorDict, raw_obs)
        assert obs["actor"].shape == (4, 210)
        assert obs["critic"].shape == (4, 270)
        assert obs["frnet_aux"].shape == (4, 17)
        assert obs["frnet_proprio"].shape == (4, 42)
        # Check actual ray hits, not just finite fallback observations.
        for sensor_name in (
            "base_height_scan",
            "foot_height_scan",
            "frnet_terrain_height_map",
        ):
            data = env.scene[sensor_name].data
            assert data.distances.ge(0).all(), sensor_name
            torch.testing.assert_close(
                data.hit_pos_w[..., 2],
                torch.zeros_like(data.hit_pos_w[..., 2]),
                atol=1e-5,
                rtol=0,
            )
        heights = env.scene["frnet_terrain_height_map"].data.heights
        assert heights.shape == (4, 1, 187)
        torch.testing.assert_close(heights.flatten(1), obs["critic"][:, -187:])
        torch.testing.assert_close(
            env.scene["base_height_scan"].data.heights[:, 0], heights[:, 0, 0]
        )
        for sensor_name, bodies in (
            (
                "hip_ground_touch",
                ("FL_hip_link", "FR_hip_link", "RR_hip_link", "RL_hip_link"),
            ),
            (
                "shank_ground_touch",
                ("FL_shank_link", "FR_shank_link", "RR_shank_link", "RL_shank_link"),
            ),
        ):
            assert (
                tuple(
                    name.split("/")[-1] for name in env.scene[sensor_name].primary_names
                )
                == bodies
            )
        # Label 0 is base mass; labels 1..3 aggregate hip/thigh/shank mass.
        groups = cfg.observations["frnet_aux"].terms["targets"].params["body_groups"]
        targets = frnet_mass_target(env, groups)
        torch.testing.assert_close(obs["frnet_aux"][:, :4], targets)
        model = env.sim.mj_model
        current_mass = env.sim.model.body_mass
        for group_id, group in enumerate(groups):
            ids = [model.body("robot/" + name).id for name in group]
            expected = current_mass[:, ids].sum(-1) / torch.as_tensor(
                model.body_mass[ids].sum(), device=env.device
            )
            torch.testing.assert_close(targets[:, group_id], expected.float())

        action = env.action_manager.get_term("joint_pos")
        assert isinstance(action, RecoveryJointPositionAction)
        for _ in range(8):
            raw_obs, reward, _, _, _ = env.step(torch.randn(4, 12))
            obs = cast(TensorDict, raw_obs)
            assert torch.isfinite(reward).all()
            for key in ("actor", "critic", "frnet_aux", "frnet_proprio"):
                assert torch.isfinite(obs[key]).all()
        previous_target = action._target.clone()
        previous_reset_xy = env.extras["frnet_reset_xy"].clone()
        previous_level = env.extras["frnet_reset_level"].clone()
        env.reset(env_ids=torch.tensor([0, 2]))
        torch.testing.assert_close(
            env.extras["frnet_reset_xy"][[1, 3]], previous_reset_xy[[1, 3]]
        )
        torch.testing.assert_close(
            env.extras["frnet_reset_xy"][[0, 2]],
            env.scene["robot"].data.root_link_pos_w[[0, 2], :2],
        )
        torch.testing.assert_close(action._target[[1, 3]], previous_target[[1, 3]])
        torch.testing.assert_close(
            env.extras["frnet_reset_level"][[1, 3]], previous_level[[1, 3]]
        )
        assert action._age[[0, 2]].eq(0).all()
        torch.testing.assert_close(
            action._target[[0, 2]], env.scene["robot"].data.joint_pos[[0, 2]]
        )
        # Control-step target slew must not repeat on each physics substep.
        action._age[:] = 20
        before_target = action._target.clone()
        action.process_actions(torch.full((4, 12), 5.0))
        latched = action._target.clone()
        assert (latched - before_target).abs().max() <= 0.150001
        for _ in range(cfg.decimation):
            action.apply_actions()
        torch.testing.assert_close(action._target, latched)
        assert isinstance(env.curriculum_manager, CurriculumManager)
        curriculum = env.curriculum_manager.get_term_cfg("recovery").func
        assert isinstance(curriculum, RecoveryCurriculum)
        # A real timeout records terminal outcomes before reset changes family
        # labels and clears the per-episode progress buffers.
        kinds_before_timeout = env.extras["frnet_reset_kind"].clone()
        trials_before_timeout = curriculum.trials.clone()
        env.episode_length_buf.fill_(env.max_episode_length - 1)
        env.step(torch.zeros(4, 12))
        assert env.extras["frnet_episode_steps"].eq(0).all()
        for case in range(4):
            assert curriculum.trials[case] == (
                trials_before_timeout[case] + (kinds_before_timeout == case).sum()
            )
        runner_cfg = load_rl_cfg(TASK_ID)
        assert isinstance(runner_cfg, RslRlOnPolicyRunnerCfg)
        runner_cfg.num_steps_per_env = 4
        runner_cfg.algorithm.num_learning_epochs = 1
        runner_cfg.algorithm.num_mini_batches = 1
        wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
        runner_cls = load_runner_cls(TASK_ID)
        assert runner_cls is not None
        runner = runner_cls(wrapped, asdict(runner_cfg), device="cpu")
        before = runner.alg.actor.mass_head.weight.detach().clone()
        before_decoder = runner.alg.actor.mcp_decoder[0].weight.detach().clone()
        before_logvar = runner.alg.actor.latent_logvar_head.weight.detach().clone()
        runner.learn(num_learning_iterations=1)
        assert not torch.equal(before, runner.alg.actor.mass_head.weight)
        assert not torch.equal(before_decoder, runner.alg.actor.mcp_decoder[0].weight)
        assert not torch.equal(
            before_logvar, runner.alg.actor.latent_logvar_head.weight
        )
        assert runner.alg.actor.obs_normalizer.count > 0
        assert runner.alg.optimizer.state
        checkpoint = tmp_path / "model.pt"
        curriculum.level = 3
        curriculum.trials[2] = 17
        curriculum.apply_stage()
        runner.save(str(checkpoint))
        curriculum.level = 0
        curriculum.trials.zero_()
        curriculum.apply_stage()
        restored = runner_cls(wrapped, asdict(runner_cfg), device="cpu")
        restored.load(str(checkpoint), map_location="cpu")
        assert curriculum.level == 3 and curriculum.trials[2] == 17
        assert env.reward_manager.get_term_cfg("action_rate").weight == -0.01 * 0.55
        for key, value in runner.alg.actor.state_dict().items():
            torch.testing.assert_close(value, restored.alg.actor.state_dict()[key])
        observation = wrapped.get_observations()
        with torch.no_grad():
            torch.testing.assert_close(
                runner.alg.actor(observation),
                runner.alg.actor.as_jit()(observation["actor"]),
            )
            torch.testing.assert_close(
                runner.alg.actor(observation),
                torch.jit.script(runner.alg.actor.as_jit())(observation["actor"]),
            )
        old_state = torch.load(checkpoint, weights_only=False)
        del old_state["infos"]["xz1_recovery_curriculum"]
        old_state["infos"]["env_state"]["common_step_counter"] = 99999
        old_checkpoint = tmp_path / "legacy_curriculum.pt"
        torch.save(old_state, old_checkpoint)
        restored.load(str(old_checkpoint), map_location="cpu")
        assert curriculum.level == 0 and curriculum.trials.eq(0).all()
        assert curriculum.stage_started_at == 99999
        assert env.reward_manager.get_term_cfg("action_rate").weight == -0.001
    finally:
        env.close()
    metrics_file = tmp_path / "evaluation.json"
    metrics = evaluate(
        EvaluateConfig(
            checkpoint_file=checkpoint,
            task=TASK_ID,
            num_envs=2,
            horizon_s=0.06,
            stable_duration_s=0.04,
            device="cpu",
            output_file=metrics_file,
        )
    )
    assert set(metrics["falls"]) == {"left_side", "right_side", "back", "belly"}
    assert metrics["falls"]["back"]["success_rate"] == 0.0
    assert all(
        result["invalid_state_rate"] == 0 for result in metrics["falls"].values()
    )
    assert json.loads(metrics_file.read_text())["overall"] == metrics["overall"]
