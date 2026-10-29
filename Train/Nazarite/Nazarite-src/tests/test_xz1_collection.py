"""Complete episode quotas, explicit coverage and synchronized depth recording."""

import json
from types import SimpleNamespace

import pytest
import torch

from nazarite.config.train_config.XZ1_env_cfgs.collect_env_cfg import XZ1DepthCfg
from nazarite.config.train_config.XZ1_env_cfgs.teacher_env_cfg import (
    Nazarite_WMP_HIM_Teacher_XZ1,
)
from nazarite.rl.wmp.xz1_dataset import EpisodeWriter, validate_episode
from nazarite.scripts import collect_xz1_wmp as collect
from nazarite.scripts.evaluate_xz1_wmp import (
    course_goal,
    crossing_route_limit,
    landing_half_width,
    landing_mask,
)


def arguments(tmp_path, *extra):
    checkpoint = tmp_path / "teacher.pt"
    checkpoint.touch()
    return collect.parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(tmp_path / "data"),
            *extra,
        ]
    )


def test_collection_defaults_cover_all_rows_and_do_not_duplicate_flat(tmp_path):
    args = arguments(tmp_path, "--seeds", "42", "44", "43", "--episodes-per-case", "16")
    plan = collect.collection_plan(args)
    assert len(plan) == 243
    assert sum(case["episodes"] for case in plan) == 3888
    assert {
        (case["training_level"], case["difficulty"])
        for case in plan
        if case["terrain"] == "gap"
    } == {(i, i / 9) for i in range(10)}
    assert len([case for case in plan if case["terrain"] == "flat"]) == 3
    assert all(case["speed_m_s"] is None for case in plan)
    assert [case["case_id"] for case in plan] == list(range(len(plan)))


def test_nominal_speed_grid_and_legacy_difficulties(tmp_path):
    args = arguments(
        tmp_path,
        "--nominal",
        "--speeds",
        "0.3",
        "0.5",
        "--terrains",
        "gap",
        "--levels",
        "1",
        "2",
        "9",
    )
    plan = collect.collection_plan(args)
    assert len(plan) == 12
    assert {case["speed_m_s"] for case in plan} == {0.3, 0.5}
    args.levels, args.difficulties = None, [0, 0.5, 1]
    assert {case["difficulty"] for case in collect.collection_plan(args)} == {0, 0.5, 1}
    for extra in (
        ("--speed", "0.4"),
        ("--nominal", "--speed", "0"),
        ("--levels", "10"),
        ("--levels", "1", "1"),
        ("--levels", "1", "--all-levels"),
    ):
        with pytest.raises(SystemExit):
            arguments(tmp_path, *extra)


def test_bridge_landing_uses_platform_only_after_full_crossing():
    generator = Nazarite_WMP_HIM_Teacher_XZ1().scene.terrain.terrain_generator
    goal, route = course_goal(generator, "single_beam", 1.0)
    platform = landing_half_width(generator, "single_beam", 1.0)
    assert route == pytest.approx(0.08)
    assert platform == pytest.approx(0.4)
    root = torch.tensor([[goal + 0.4, 0.0, 0.3]])
    feet = torch.tensor([[[goal + 0.1, y, 0.0] for y in (-0.22, 0.22, -0.22, 0.22)]])
    gravity = torch.tensor([[0.0, 0.0, -1.0]])
    support = torch.ones(1, 4, dtype=torch.bool)
    assert not landing_mask(root, feet, gravity, support, goal, route).any()
    assert landing_mask(
        root, feet, gravity, support, goal, route, -0.015, platform
    ).all()
    assert crossing_route_limit(
        root, feet, goal, route, platform
    ).item() == pytest.approx(0.28)
    feet[:, 0, 0] = goal - 0.1
    assert crossing_route_limit(
        root, feet, goal, route, platform
    ).item() == pytest.approx(route)
    assert not landing_mask(
        root, feet, gravity, support, goal, route, -0.015, platform
    ).any()
    feet[..., 0] = goal + 0.1
    feet[:, 0, 1] = 0.41
    assert not landing_mask(
        root, feet, gravity, support, goal, route, -0.015, platform
    ).any()


def test_partial_collection_summary_does_not_claim_missing_slow_trials(tmp_path):
    args = arguments(
        tmp_path,
        "--terrains",
        "gap",
        "--levels",
        "1",
        "--episodes-per-case",
        "3",
        "--num-envs",
        "2",
    )
    writer = SimpleNamespace(
        directory=tmp_path,
        manifest={
            "status": "interrupted",
            "metadata": {"collection_plan": collect.collection_plan(args)},
            "episodes": [
                {
                    "case_id": 0,
                    "env_id": 0,
                    "success": False,
                    "timeout_failure": False,
                    "termination": ["obstacle_ground_contact"],
                    "frames": 3,
                    "split": "train",
                }
            ]
            * 2,
        },
    )
    collect.write_collection_summary(writer)
    result = json.loads((tmp_path / "collection_summary.json").read_text())
    case = result["cases"][0]
    assert not case["complete"]
    assert case["episodes_by_env"] == [2, 0]
    assert case["episode_quotas"] == [2, 1]
    assert case["success_rate"] == 0
    assert case["eligible_32_frames"] == 0
    assert result["status"] == "interrupted"


def test_stable_crossing_never_overrides_physical_failure():
    generator = Nazarite_WMP_HIM_Teacher_XZ1().scene.terrain.terrain_generator
    goal, _ = course_goal(generator, "single_beam", 1.0)

    class Scene(dict):
        pass

    scene = Scene(
        robot=SimpleNamespace(
            find_sites=lambda *args, **kwargs: (list(range(4)), []),
            data=SimpleNamespace(
                root_link_pos_w=torch.tensor([[goal + 0.4, 0.0, 0.3]] * 2),
                site_pos_w=torch.tensor(
                    [[[goal + 0.1, y, 0.0] for y in (-0.22, 0.22, -0.22, 0.22)]] * 2
                ),
                projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]] * 2),
            ),
        ),
        feet_ground_contact=SimpleNamespace(
            data=SimpleNamespace(force=torch.tensor([[[0.0, 0.0, 10.0]] * 4] * 2))
        ),
    )
    scene.terrain = SimpleNamespace(cfg=SimpleNamespace(terrain_generator=generator))
    scene.env_origins = torch.zeros(2, 3)
    env = SimpleNamespace(
        scene=scene,
        num_envs=2,
        device="cpu",
        step_dt=0.02,
        termination_manager=SimpleNamespace(terminated=torch.tensor([False, True])),
    )
    tracker = collect.CollectionTraversal(env, "single_beam", 1.0)
    for _ in range(9):
        assert not tracker.update()[0].any()
    success, off_route = tracker.update()
    assert success.tolist() == [True, False]
    assert not off_route.any()
    tracker.reset(torch.tensor([0]))
    assert tracker.hold.tolist() == [0, 10]
    assert tracker.crossed.tolist() == [False, True]


def test_real_collector_loop_keeps_slow_quota_and_reset_action_chunks(
    tmp_path, monkeypatch
):
    """A fast environment retires before a slow environment's first episode."""

    class FakeEnv:
        def __init__(self, cfg, **kwargs):
            self.num_envs, self.device, self.step_dt = 2, "cpu", 0.02
            self.ticks = torch.zeros(2, dtype=torch.long)
            self.pending = torch.zeros(2, dtype=torch.bool)
            self.done = torch.zeros(2, dtype=torch.bool)
            self.scene = {
                "xz1_depth_camera": SimpleNamespace(
                    data=SimpleNamespace(depth=torch.ones(2, 64, 64, 1))
                )
            }
            self.sim = SimpleNamespace(sense=lambda **kwargs: None)
            self.termination_manager = SimpleNamespace(
                active_terms=["time_out"],
                get_term=lambda name: self.done,
                terminated=torch.zeros(2, dtype=torch.bool),
            )

        def observations(self):
            return {
                "actor": torch.zeros(2, 6, 47),
                "estimator": torch.zeros(2, 50),
                "teacher_scan": torch.zeros(2, 132),
            }

        def reset(self, env_ids=None, **kwargs):
            ids = torch.arange(2) if env_ids is None else env_ids
            self.ticks[ids] = 0
            self.pending[ids] = False
            return self.observations(), {}

        def step(self, actions):
            assert not self.pending.any(), "Retired done environments must still reset"
            self.ticks += 1
            self.done = self.ticks >= torch.tensor([2, 10])
            self.pending |= self.done
            return self.observations(), torch.ones(2), self.done.clone(), {}

        def close(self):
            pass

    class FakeActor:
        def scan_encoder(self, scan):
            return torch.zeros(2, 32)

    class FakePolicy:
        def __call__(self, obs):
            return torch.ones(2, 12)

        def reset(self, dones):
            pass

    class FakeRunner:
        def __init__(self, *args, **kwargs):
            self.alg = SimpleNamespace(actor=FakeActor())

        def load(self, *args, **kwargs):
            pass

        def get_inference_policy(self, **kwargs):
            return FakePolicy()

    class FakeTraversal:
        def __init__(self, *args):
            self.crossed = torch.zeros(2, dtype=torch.bool)
            self.longest = torch.zeros(2, dtype=torch.long)

        def update(self):
            return torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool)

        def reset(self, ids):
            pass

    monkeypatch.setattr(collect, "ManagerBasedRlEnv", FakeEnv)
    monkeypatch.setattr(collect, "RslRlVecEnvWrapper", lambda env, **kwargs: env)
    monkeypatch.setattr(collect, "XZ1TerrainTeacherRunner", FakeRunner)
    monkeypatch.setattr(collect, "XZ1TerrainTeacherActor", FakeActor)
    monkeypatch.setattr(collect, "CollectionTraversal", FakeTraversal)
    monkeypatch.setattr(collect, "xz1_teacher_runner_cfg", XZ1DepthCfg)
    monkeypatch.setattr(collect, "collection_cfg", lambda *args: (None, XZ1DepthCfg()))
    writer = EpisodeWriter(tmp_path / "recorded", {"camera": {"far_clip": 2.0}})
    args = SimpleNamespace(
        device="cpu",
        checkpoint="fake",
        episodes_per_case=3,
        validation_seeds=[43],
        nominal=False,
    )
    collect.collect_case(args, writer, "gap", 1 / 9, 42, training_level=1)
    rows = writer.manifest["episodes"]
    assert [(row["env_id"], row["episode"]) for row in rows] == [(0, 0), (0, 1), (1, 0)]
    assert [row["frames"] for row in rows] == [1, 1, 3]
    assert [row["terminal_time_s"] for row in rows] == pytest.approx([0.04, 0.04, 0.2])
    for row in rows:
        data = torch.load(writer.directory / row["file"], weights_only=True)["tensors"]
        validate_episode(data, 2.0)
        assert data["action"][0].count_nonzero() == 0
        assert row["timeout_failure"] and not row["success"]
