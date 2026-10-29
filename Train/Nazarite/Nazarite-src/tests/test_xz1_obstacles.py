"""Physical holes, safe origins and shared runtime/evaluation course bounds."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
import warp as wp

from nazarite.config.robot_config.xz1_cfg import XZ1_FOOT_GEOMS
from nazarite.config.train_config.XZ1_env_cfgs import him_course as course
from nazarite.config.train_config.XZ1_env_cfgs.him_obstacles import (
    GAP_BOTTOM_PREFIX,
    OBSTACLE_TERRAIN_NAMES,
    XZ1_TERRAIN_NAMES,
)
from nazarite.config.train_config.XZ1_env_cfgs.teacher_env_cfg import (
    Nazarite_WMP_HIM_Teacher_XZ1,
)
from nazarite.scripts.evaluate_xz1_wmp import TERRAINS, course_goal, landing_mask


def patch(name, difficulty):
    generator = Nazarite_WMP_HIM_Teacher_XZ1().scene.terrain.terrain_generator
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    cfg = generator.sub_terrains[name]
    output = cfg.function(difficulty, spec, np.random.default_rng(42))
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return cfg, output, model, data


def height_at(model, data, x, y):
    origin = np.array([x, y, 2.0])
    distance = mujoco.mj_ray(
        model,
        data,
        origin,
        np.array([0.0, 0.0, -1.0]),
        None,
        True,
        -1,
        np.zeros(1, dtype=np.int32),
    )
    assert distance >= 0
    return origin[2] - distance


@pytest.mark.parametrize("name", XZ1_TERRAIN_NAMES)
@pytest.mark.parametrize("difficulty", [0.0, 0.5, 1.0])
def test_all_nine_compile_and_spawn_on_flat_support(name, difficulty):
    _, out, model, data = patch(name, difficulty)
    x, y, z = out.origin
    # Standing XZ1 plus the course reset jitter fits on the spawn deck.
    for dx in (-0.35, 0.35):
        for dy in (-0.25, 0.25):
            assert height_at(model, data, x + dx, y + dy) == pytest.approx(z)
    assert (model.geom_group == 0).all()


def test_stones_and_platforms_share_height_and_bridge_is_continuous():
    cfg, _, model, data = patch("plum_blossom", 1.0)
    for x, y in cfg.stone_centers():
        assert height_at(model, data, x, 4 + y) == pytest.approx(0.2)
    assert height_at(model, data, 1.0, 4.0) == pytest.approx(0.2)
    assert height_at(model, data, 3.7, 4.0) == pytest.approx(0.2)
    assert height_at(model, data, 2.05, 4.0) == pytest.approx(0.0)
    cfg, _, model, data = patch("single_beam", 1.0)
    for x in np.linspace(cfg.start_x, cfg.start_x + cfg.length, 20):
        assert height_at(model, data, x, 4.0) == pytest.approx(0.1)
    assert height_at(model, data, 3.0, 4.3) == pytest.approx(0.0)


def test_tire_center_is_a_collision_hole_and_gap_is_not_filled():
    cfg, _, model, data = patch("tire_array", 1.0)
    x, y = cfg.start_x + cfg.outer_diameter / 2, 4 + cfg.pitch / 2
    assert height_at(model, data, x, y) == pytest.approx(0.0)
    radius = (cfg.outer_diameter + cfg.inner_diameter) / 4
    assert height_at(model, data, x + radius, y) == pytest.approx(0.165)
    cfg, _, model, data = patch("gap", 1.0)
    for i in range(cfg.num_gaps):
        edge = cfg.start_x + i * (0.36 + cfg.landing_length)
        assert height_at(model, data, edge + 0.18, 4.0) == pytest.approx(-0.36)
        assert height_at(model, data, edge + 0.36 + 0.1, 4.0) == pytest.approx(0.0)


def test_frame_internal_clearance_matches_hardest_row():
    cfg, _, model, data = patch("low_frame", 1.0)
    # Downward height scans report roof height, not free space below it.
    assert height_at(model, data, cfg.start_x, 4.0) == pytest.approx(0.33)
    geom_id = np.zeros(1, dtype=np.int32)
    distance = mujoco.mj_ray(
        model,
        data,
        np.array([cfg.start_x, 4.0, 0.1]),
        np.array([0.0, 0.0, 1.0]),
        None,
        True,
        -1,
        geom_id,
    )
    assert distance == pytest.approx(0.19)  # internal underside is .29 m


class Scene(dict):
    pass


def runtime(name, difficulty):
    cfg = Nazarite_WMP_HIM_Teacher_XZ1()
    generator = cfg.scene.terrain.terrain_generator
    generator.difficulty_range = (difficulty, difficulty)
    names = list(generator.sub_terrains)
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.zeros(1, 3), site_pos_w=torch.zeros(1, 4, 3)
        ),
        find_sites=lambda *args, **kwargs: (list(range(4)), []),
    )
    terrain = SimpleNamespace(
        cfg=SimpleNamespace(terrain_generator=generator),
        terrain_levels=torch.zeros(1, dtype=torch.long),
        terrain_types=torch.tensor([names.index(name)]),
    )
    force = torch.zeros(1, 4, 3)
    force[..., 2] = 10
    scene = Scene(
        robot=robot,
        feet_ground_contact=SimpleNamespace(data=SimpleNamespace(force=force)),
    )
    scene.terrain = terrain
    scene.env_origins = torch.zeros(1, 3)
    geom_names = [f"robot/{name}" for name in XZ1_FOOT_GEOMS]
    geom_names.append(f"{GAP_BOTTOM_PREFIX}0")
    model = SimpleNamespace(
        ngeom=len(geom_names),
        geom=lambda index: SimpleNamespace(name=geom_names[index]),
    )
    data = SimpleNamespace(
        nacon=wp.array([0], dtype=int, device="cpu"),
        contact=SimpleNamespace(
            worldid=wp.array([0], dtype=int, device="cpu"),
            geom=wp.array([[0, 4]], dtype=wp.vec2i, device="cpu"),
            dist=wp.array([0.0], dtype=float, device="cpu"),
        ),
    )
    return generator, SimpleNamespace(
        scene=scene,
        num_envs=1,
        device="cpu",
        sim=SimpleNamespace(mj_model=model, wp_data=data),
    )


@pytest.mark.parametrize("name", OBSTACLE_TERRAIN_NAMES)
@pytest.mark.parametrize("difficulty", [0.0, 0.5, 1.0])
def test_course_goals_reject_bypass_partial_crossing_and_ground_landing(
    name, difficulty
):
    generator, env = runtime(name, difficulty)
    goal, limit = course_goal(generator, name, difficulty)
    selected, _, half_width, start, end = course.course_bounds(env)
    assert selected.all()
    assert end.item() == pytest.approx(goal)
    assert half_width.item() - 0.12 == pytest.approx(limit)
    root = env.scene["robot"].data.root_link_pos_w
    feet = env.scene["robot"].data.site_pos_w
    root[:, 0] = goal + 0.36
    feet[..., 0] = goal + 0.1
    feet[..., 2] = 0.025
    assert course.narrow_course_complete(env).all()
    feet[:, 0, 0] = goal - 0.01
    assert not course.narrow_course_complete(env).any()
    feet[..., 0] = goal + 0.1
    feet[..., 2] = -0.1
    assert not course.narrow_course_complete(env).any()
    assert not landing_mask(
        root,
        feet,
        torch.tensor([[0.0, 0.0, -1.0]]),
        torch.ones(1, 4, dtype=torch.bool),
        goal,
        limit,
        minimum_foot_height=-0.015,
    ).any()
    root[:, 0] = (start + end) / 2
    root[:, 1] = limit + 0.01
    assert course.left_narrow_course(env).all()


@pytest.mark.parametrize("name", ["plum_blossom", "single_beam", "gap", "tire_array"])
def test_ground_contact_cannot_cheat_raised_course_but_tire_holes_are_valid(name):
    _, env = runtime(name, 1.0)
    check = course.ObstacleGroundContact(None, env)
    *_, start, end = course.course_bounds(env)
    feet = env.scene["robot"].data.site_pos_w
    feet[..., 0] = (start + end)[:, None] / 2
    feet[..., 2] = 0.025
    assert not check(env).any()
    feet[:, :2, 2] = -0.075
    assert bool(check(env)) == (name in ("plum_blossom", "single_beam"))
    if name == "gap":
        # Below the deck with vertical force still isn't a pit-floor contact.
        env.sim.wp_data.nacon.assign([1])
        assert check(env).all()


def test_task_and_evaluator_share_exactly_nine_terrain_types():
    generator = Nazarite_WMP_HIM_Teacher_XZ1().scene.terrain.terrain_generator
    assert tuple(generator.sub_terrains) == TERRAINS == XZ1_TERRAIN_NAMES
    assert sum(t.proportion for t in generator.sub_terrains.values()) == pytest.approx(
        1.0
    )


def test_student_rejects_seven_type_curriculum_before_restoring():
    from nazarite.scripts.distill_xz1_wmp import restore_curriculum

    _, env = runtime("plum_blossom", 0.0)
    for state in (
        {"terrain_names": ["flat"], "buffers": {}},
        {"buffers": {"_delta_type_episode_count": torch.zeros(7)}},
    ):
        with pytest.raises(ValueError, match="terrain layout differs"):
            restore_curriculum(env, state)


@pytest.mark.slow
def test_nine_courses_teacher_ppo_and_real_depth_student_on_gpu():
    if not torch.cuda.is_available():
        pytest.skip("Depth integration requires CUDA")
    teacher_path = Path(
        "logs/rsl_rl/xz1_wmp_him_teacher/"
        "2026-10-26_19-23-47_stage1_scan_teacher/model_999.pt"
    )
    student_path = Path("logs/rsl_rl/xz1_wmp_student/student_v1/student_1000.pt")
    if not teacher_path.is_file() or not student_path.is_file():
        pytest.skip("Local XZ1 checkpoints unavailable")
    from tensordict import TensorDict

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from nazarite.config.train_config.XZ1_env_cfgs.collect_env_cfg import (
        attach_xz1_depth,
    )
    from nazarite.config.train_config.XZ1_env_cfgs.teacher_env_cfg import (
        Nazarite_WMP_HIM_Teacher_XZ1,
    )
    from nazarite.config.train_config.XZ1_env_cfgs.teacher_rl_cfg import (
        xz1_teacher_runner_cfg,
    )
    from nazarite.rl.xz1_student import load_student_policy
    from nazarite.rl.xz1_teacher import XZ1TerrainTeacherRunner

    cfg = attach_xz1_depth(Nazarite_WMP_HIM_Teacher_XZ1())
    cfg.scene.num_envs = 18
    cfg.scene.terrain.terrain_generator.num_rows = 2
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.curriculum = {}
    cfg.auto_reset = False
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        terrain = env.scene.terrain
        terrain.terrain_types[:] = torch.arange(18, device=env.device) % 9
        terrain.terrain_levels[:] = torch.arange(18, device=env.device) // 9
        terrain.env_origins[:] = terrain.terrain_origins[
            terrain.terrain_levels, terrain.terrain_types
        ]
        raw, _ = env.reset()
        obs = TensorDict(raw, batch_size=[18])
        assert obs["actor"].shape == (18, 6, 47)
        assert obs["critic"].shape == (18, 244)
        assert obs["teacher_scan"].shape == (18, 132)
        wrapper = RslRlVecEnvWrapper(env, clip_actions=10.0)
        runner_cfg = xz1_teacher_runner_cfg()
        runner_cfg.initial_checkpoint = str(teacher_path)
        runner_cfg.num_steps_per_env = 4
        runner_cfg.algorithm.num_learning_epochs = 1
        runner_cfg.algorithm.num_mini_batches = 1
        runner = XZ1TerrainTeacherRunner(wrapper, asdict(runner_cfg), device=env.device)
        source = torch.load(teacher_path, map_location=env.device, weights_only=False)
        for key, value in runner.alg.actor.state_dict().items():
            torch.testing.assert_close(value, source["actor_state_dict"][key])
        assert (
            not runner.alg.optimizer.state and not runner.alg.estimator.optimizer.state
        )
        assert runner.current_learning_iteration == env.common_step_counter == 0
        initial = runner.alg.actor.mlp[0].weight.detach().clone()
        runner.learn(num_learning_iterations=1)
        assert not torch.equal(initial, runner.alg.actor.mlp[0].weight)
        assert runner._nan_recoveries == 0
        assert not env.termination_manager.terminated.any()
        # Compatibility smoke only: old student ability on new terrain is not
        # measured by running a few control steps near the spawn platform.
        obs = wrapper.get_observations()
        visual = load_student_policy(env, obs, student_path, env.device)
        obs = obs.exclude("teacher_scan")
        with torch.inference_mode():
            for _ in range(6):
                actions = visual(obs)
                assert actions.shape == (18, 12) and torch.isfinite(actions).all()
                visual.advance(actions)
                obs, reward, dones, _ = wrapper.step(actions)
                assert torch.isfinite(reward).all()
                assert not dones.any()
                obs = obs.exclude("teacher_scan")
        depth = env.scene["xz1_depth_camera"].data.depth
        assert depth.shape == (18, 64, 64, 1) and torch.isfinite(depth).all()
        assert (depth > 0).flatten(1).sum(-1).min() > 100
    finally:
        env.close()
