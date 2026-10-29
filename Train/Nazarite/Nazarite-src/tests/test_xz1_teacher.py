"""Reference scan, HIM transfer, export and evaluation regressions."""

from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from rsl_rl.models.him_actor_model import HIMActorModel
from tensordict import TensorDict

from nazarite.config.train_config.XZ1_env_cfgs.him_env_cfg import (
    Nazarite_HIM_Complex_Terrain_XZ1,
)
from nazarite.config.train_config.XZ1_env_cfgs.him_rl_cfg import xz1_him_runner_cfg
from nazarite.config.train_config.XZ1_env_cfgs.teacher_env_cfg import (
    SCAN_X,
    SCAN_Y,
    Nazarite_WMP_HIM_Eval_XZ1,
    Nazarite_WMP_HIM_Teacher_XZ1,
    ParkourScanPatternCfg,
)
from nazarite.config.train_config.XZ1_env_cfgs.teacher_rl_cfg import (
    xz1_teacher_runner_cfg,
)
from nazarite.rl.xz1_teacher import XZ1TerrainTeacherActor, expand_him_state
from nazarite.scripts.evaluate_xz1_wmp import course_goal, landing_mask


def _models():
    torch.manual_seed(17)
    obs = TensorDict(
        {
            "actor": torch.randn(5, 6, 47),
            "teacher_scan": torch.rand(5, 132) * 2 - 1,
        },
        batch_size=[5],
    )
    args = asdict(xz1_him_runner_cfg().actor)
    args.pop("class_name")
    baseline = HIMActorModel(obs, {"actor": ["actor"]}, "actor", 12, **args)
    teacher = XZ1TerrainTeacherActor(obs, {"actor": ["actor"]}, "actor", 12, **args)
    return obs, baseline, teacher


def test_reference_grid_exact_coordinates_and_x_major_order():
    points, directions = ParkourScanPatternCfg().generate_rays(None, "cpu")
    assert points.shape == directions.shape == (132, 3)
    torch.testing.assert_close(points[:, 0], torch.tensor(SCAN_X).repeat_interleave(11))
    torch.testing.assert_close(points[:, 1], torch.tensor(SCAN_Y).repeat(12))
    assert (points[:, 2] == 1).all()
    assert (directions[:, 2] == -1).all()


def test_weight_only_transfer_preserves_actions_and_history_normalization():
    obs, baseline, teacher = _models()
    baseline.update_normalization(obs)
    teacher.load_state_dict(expand_him_state(teacher, baseline.state_dict()))
    baseline.eval()
    teacher.eval()
    torch.testing.assert_close(baseline(obs), teacher(obs), rtol=1e-5, atol=2e-6)
    assert teacher.get_latent(obs).shape == (5, 98)
    assert teacher.mlp[0].weight[:, 66:].count_nonzero() == 0
    altered = obs.clone()
    altered["teacher_scan"] *= -1
    torch.testing.assert_close(teacher(obs), teacher(altered))
    for key, value in baseline.estimator.state_dict().items():
        torch.testing.assert_close(value, teacher.estimator.state_dict()[key])
    bad = dict(baseline.state_dict())
    bad["mlp.0.weight"] = torch.zeros(512, 64)
    with pytest.raises(ValueError, match="66-D"):
        expand_him_state(teacher, bad)


def test_teacher_scan_receives_gradient_and_two_input_export_matches():
    obs, _, teacher = _models()
    teacher.eval()
    teacher(obs).square().sum().backward()
    assert teacher.scan_encoder[0].weight.grad.abs().sum() > 0
    assert teacher.estimator.encoder[0].weight.grad is None
    export = teacher.as_onnx().eval()
    history = teacher.get_observation_tensor(obs)
    torch.testing.assert_close(export(history, obs["teacher_scan"]), teacher(obs))
    scripted = torch.jit.script(teacher.as_jit().eval())
    torch.testing.assert_close(scripted(history, obs["teacher_scan"]), teacher(obs))


def test_teacher_and_eval_configs_do_not_mutate_base_xz1():
    base = Nazarite_HIM_Complex_Terrain_XZ1()
    teacher = Nazarite_WMP_HIM_Teacher_XZ1()
    fixed = Nazarite_WMP_HIM_Eval_XZ1(terrain="high_wall", difficulty=0.5)
    assert list(base.rewards) == list(teacher.rewards)
    assert set(teacher.terminations) == set(base.terminations) | {
        "obstacle_ground_contact"
    }
    assert base.observations["actor"].history_length == 6
    assert "teacher_scan" not in base.observations
    assert len(base.scene.sensors) + 1 == len(teacher.scene.sensors)
    assert tuple(base.scene.terrain.terrain_generator.sub_terrains) == (
        "flat",
        "pyramid_stairs",
        "random_grid",
        "trapezoid_ramp",
        "high_wall",
        "narrow_stairs",
        "narrow_ramp",
    )
    assert len(teacher.scene.terrain.terrain_generator.sub_terrains) == 9
    assert len(fixed.scene.terrain.terrain_generator.sub_terrains) == 1
    assert fixed.scene.terrain.terrain_generator.difficulty_range == (0.5, 0.5)
    assert not fixed.auto_reset and not fixed.curriculum
    assert set(fixed.events) == {"reset_base", "reset_robot_joints"}
    assert fixed.commands["twist"].heading_command is False
    assert xz1_teacher_runner_cfg().obs_groups["actor"] == ("actor",)


@pytest.mark.parametrize("play", [False, True])
def test_him_terrain_rollback_keeps_visual_course_configuration_isolated(play):
    from nazarite.config.train_config.XZ1_env_cfgs.collect_env_cfg import (
        Nazarite_WMP_HIM_Collect_XZ1,
    )
    from nazarite.config.train_config.XZ1_env_cfgs.him_obstacles import (
        XZ1_TERRAIN_NAMES,
    )

    base = Nazarite_HIM_Complex_Terrain_XZ1(play=play)
    teacher = Nazarite_WMP_HIM_Teacher_XZ1(play=play)
    collect = Nazarite_WMP_HIM_Collect_XZ1(play=play)
    base_gen = base.scene.terrain.terrain_generator
    assert len(base_gen.sub_terrains) == 7
    assert [t.proportion for t in base_gen.sub_terrains.values()] == [
        0.10,
        0.15,
        0.25,
        0.10,
        0.15,
        0.15,
        0.10,
    ]
    assert "obstacle_ground_contact" not in base.terminations
    for cfg in (teacher, collect):
        generator = cfg.scene.terrain.terrain_generator
        assert tuple(generator.sub_terrains) == XZ1_TERRAIN_NAMES
        if play:
            assert generator.num_cols == 9
        assert "obstacle_ground_contact" in cfg.terminations
    if not play:
        assert base.curriculum["terrain_levels"].params["sparse_terrain_names"] == (
            "trapezoid_ramp",
            "high_wall",
            "narrow_stairs",
            "narrow_ramp",
        )
        assert (
            "plum_blossom"
            in teacher.curriculum["terrain_levels"].params["sparse_terrain_names"]
        )
    teacher.scene.terrain.terrain_generator.sub_terrains["flat"].proportion = 0.99
    base_gen.sub_terrains["narrow_stairs"].width_range = (0.2, 0.2)
    assert (
        collect.scene.terrain.terrain_generator.sub_terrains["flat"].proportion == 0.10
    )
    assert teacher.scene.terrain.terrain_generator.sub_terrains[
        "narrow_stairs"
    ].width_range == (0.8, 0.8)
    assert (
        Nazarite_HIM_Complex_Terrain_XZ1()
        .scene.terrain.terrain_generator.sub_terrains["narrow_stairs"]
        .width_range
        == (0.8, 0.8)
    )


@pytest.mark.parametrize("terrain", ["high_wall", "narrow_stairs", "narrow_ramp"])
def test_evaluation_goals_and_landing_reject_partial_crossing(terrain):
    cfg = Nazarite_WMP_HIM_Eval_XZ1(terrain=terrain, difficulty=1.0)
    goal, bound = course_goal(cfg.scene.terrain.terrain_generator, terrain, 1.0)
    assert 0 < goal < 6
    roots = torch.tensor([[goal + 0.5, 0, 0.3]] * 4)
    feet = torch.zeros(4, 4, 3)
    feet[..., 0] = goal + 0.1
    gravity = torch.tensor([[0, 0, -1.0]] * 4)
    support = torch.ones(4, 4, dtype=torch.bool)
    feet[1, 3, 0] = goal - 0.1
    roots[2, 1] = bound + 0.2
    support[3] = False
    assert landing_mask(roots, feet, gravity, support, goal, bound).tolist() == [
        True,
        False,
        False,
        False,
    ]


def test_user_checkpoint_can_warm_start_without_shape_substitution():
    checkpoint = Path("logs/rsl_rl/xz1_him_complex/2026-10-25_22-04-31/model_31300.pt")
    if not checkpoint.is_file():
        pytest.skip("User-local checkpoint is not present")
    obs, baseline, teacher = _models()
    source = torch.load(checkpoint, map_location="cpu", weights_only=False)
    baseline.load_state_dict(source["actor_state_dict"], strict=True)
    teacher.load_state_dict(expand_him_state(teacher, source["actor_state_dict"]))
    baseline.eval()
    teacher.eval()
    torch.testing.assert_close(baseline(obs), teacher(obs), atol=3e-5, rtol=1e-5)
