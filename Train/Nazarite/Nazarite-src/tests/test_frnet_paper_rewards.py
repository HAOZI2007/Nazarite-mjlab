"""Paper reward semantics: known poses, physical units and reset boundaries."""

import math
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import load_env_cfg
from nazarite.mdp.fr_net import paper_rewards as rewards
from nazarite.mdp.xz1 import frnet


@pytest.fixture
def env():
    data = SimpleNamespace(
        projected_gravity_b=torch.tensor(
            [[0, 0, -1], [1, 0, 0], [0, 0, 1], [math.sqrt(0.19), 0, -0.9]]
        ),
        joint_pos=torch.tensor(
            [[0.0, 0, 0], [1, 0, 0], [0.25, -0.25, 0], [0.3, -0.4, 0]]
        ),
        default_joint_pos=torch.zeros(4, 3),
        joint_pos_limits=torch.tensor([-0.25, 0.25]).expand(4, 3, 2),
        joint_vel=torch.tensor(
            [[0.0, 0, 0], [0.8, -1.8, 2.8], [0.8, -0.8, 0], [-0.9, 0, 0]]
        ),
        root_link_pos_w=torch.tensor(
            [[13.0, 24, 0], [30, -40, 0], [2.3, 2.4, 0], [0, 0, 0]]
        ),
    )
    return cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=4,
            device="cpu",
            step_dt=0.02,
            scene={"robot": SimpleNamespace(data=data)},
            extras={
                "frnet_reset_xy": torch.tensor([[10.0, 20], [30, -40], [2, 2], [0, 0]]),
                "frnet_recovery_mask": torch.tensor([True, True, True, False]),
            },
        ),
    )


def test_orientation_and_posture_reward_known_poses(env):
    torch.testing.assert_close(
        rewards.base_orientation(env), torch.tensor([0.0, 1, 0, 0.19])
    )
    upright = rewards.upright_orientation(env)
    assert upright[0] == 1
    assert upright[1] < 1e-9 and upright[2] < 1e-30
    assert upright[3].item() == pytest.approx(math.exp(-0.01 / 0.045))
    # Only near-upright examples receive posture reward; squared error is
    # summed over joints, without an additional width or per-joint averaging.
    torch.testing.assert_close(
        rewards.target_posture(env, asset_cfg=SceneEntityCfg("robot")),
        torch.tensor([1.0, 0, 0, math.exp(-0.25)]),
    )


def test_contacts_count_feet_once_and_safety_force_has_physical_units(env, monkeypatch):
    class ContactStub:
        def __init__(self, found, force=None, slots=1):
            self.cfg = SimpleNamespace(num_slots=slots, reduce="netforce")
            self.data = SimpleNamespace(found=torch.tensor(found), force=force)

    monkeypatch.setattr(rewards, "ContactSensor", ContactStub)
    # Two slots per foot: multiple contact points must not inflate foot count.
    env.scene["feet_ground_contact"] = ContactStub(
        [
            [1, 1, 1, 0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0] * 8,
            [2, 2, 0, 0, 3, 1, 0, 0],
        ],
        slots=2,
    )
    torch.testing.assert_close(rewards.feet_contact(env), torch.tensor([4.0, 1, 0, 2]))
    force = torch.zeros(4, 4, 3)
    force[0, 0] = torch.tensor([3.0, 4, 1000])
    force[0, 1] = torch.tensor([0.0, 12, 0])
    force[1, :, 2] = 1000  # A vertical support force is not a horizontal cost.
    env.scene["shank_ground_touch"] = ContactStub(
        [[2, 1, 0, 0], [1, 1, 1, 1], [0] * 4, [0] * 4],
        force,
    )
    torch.testing.assert_close(
        rewards.body_contact(env, ("shank_ground_touch",)), torch.tensor([1.0, 1, 0, 0])
    )
    torch.testing.assert_close(rewards.safety_force(env), torch.tensor([17.0, 0, 0, 0]))
    force[..., :2] *= 2
    assert rewards.safety_force(env)[0] == 34  # No squaring or tanh saturation.


def test_drift_clipping_actual_limits_and_joint_velocity_threshold(env):
    torch.testing.assert_close(rewards.body_bias(env), torch.tensor([4.0, 0, 0.5, 0]))
    torch.testing.assert_close(
        rewards.position_limits(env), torch.tensor([0.0, 1, 0, 2])
    )
    torch.testing.assert_close(
        rewards.angular_velocity_limit(env), torch.tensor([0.0, 3, 0, 0.1])
    )


def test_success_metric_holds_and_resets_without_terminations(env, monkeypatch):
    stable = torch.ones(4, dtype=torch.bool)
    monkeypatch.setattr(frnet, "stable_recovery_mask", lambda *args, **kwargs: stable)
    cfg = MetricsTermCfg(
        func=frnet.StableRecoveryMetric, params={"hold_duration_s": 0.04}
    )
    metric = frnet.StableRecoveryMetric(cfg, env)
    assert metric(env).eq(0).all()
    assert metric(env).eq(1).all()
    metric.reset(torch.tensor([0, 2]))
    torch.testing.assert_close(metric(env), torch.tensor([0.0, 1, 0, 1]))
    torch.testing.assert_close(
        metric(env, fallen_only=True), torch.tensor([1.0, 1, 1, 0])
    )
    stable[1] = False
    assert metric(env)[1] == 0


def test_xz1_table_i_final_weights_and_success_only_logs():
    expected = {
        "base_orientation": -0.5,
        "upright": 6.0,
        "stand_pose": 4.0,
        "foot_contacts": 0.3,
        "body_contact": -0.2,
        "safety_force": -0.01,
        "body_bias": -0.1,
        "joint_limits": -1.0,
        "joint_velocity_limit": -0.1,
        "joint_acc": -2.5e-6,
        "joint_vel": -0.01,
        "action_rate": -0.01,
        "joint_torques": -5e-4,
    }
    for play in (False, True):
        cfg = load_env_cfg("Nazarite-FRNet-Paper-Recovery-XZ1", play=play)
        assert {name: term.weight for name, term in cfg.rewards.items()} == expected
        assert bool(cfg.curriculum) is not play
        assert cfg.reward_composition is None
        assert cfg.episode_length_s == pytest.approx(7.0)
        assert (
            math.ceil(cfg.episode_length_s / (cfg.sim.mujoco.timestep * cfg.decimation))
            == 350
        )
        assert "recovery_success" not in cfg.terminations
        assert {"recovery_success", "recovery_final_stable"} <= cfg.metrics.keys()
        assert cfg.rewards["upright"].params["epsilon"] == 0.15
        assert cfg.rewards["stand_pose"].params["epsilon"] == 0.15
