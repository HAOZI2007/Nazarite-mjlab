"""Unit tests for the WMP reward terms that do not need a simulator.

The rewritten terms (``feet_air_time``, ``foot_clearance_terrain``, ``stuck``,
``course_checkpoint_bonus``) are plain tensor code, so they can be exercised
against lightweight stubs. This catches sign, masking and state bugs without a
GPU.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from nazarite.mdp import wmp_rewards

N_ENVS = 4
TERRAIN_NAMES = (
    "stairs",
    "low_frame",
    "high_wall",
    "plum_blossom",
    "gap",
    "tire_array",
    "single_beam",
    "ramp",
)
FORWARD_CMD = torch.tensor([[0.3, 0.0, 0.0]] * N_ENVS, dtype=torch.float32)


class ContactSensorStub:
    """Minimal stand-in for a ``ContactSensor``."""

    def __init__(self) -> None:
        self.data = SimpleNamespace(
            last_air_time=torch.zeros(N_ENVS, 4),
            heights=torch.zeros(N_ENVS, 4),
        )
        self.first_contact = torch.zeros(N_ENVS, 4, dtype=torch.bool)

    def compute_first_contact(self, dt: float) -> torch.Tensor:
        return self.first_contact


class SceneStub:
    def __init__(self, env_origins: torch.Tensor, items: dict) -> None:
        self.env_origins = env_origins
        self._items = items

    def __getitem__(self, key: str):
        return self._items[key]


def make_env(
    command: torch.Tensor | None = None,
    terrain_types: torch.Tensor | None = None,
):
    """Build an env stub plus its asset and foot sensor."""
    command = torch.zeros(N_ENVS, 3) if command is None else command
    asset = SimpleNamespace(
        data=SimpleNamespace(
            root_link_lin_vel_b=torch.zeros(N_ENVS, 3),
            root_link_pos_w=torch.zeros(N_ENVS, 3),
            site_lin_vel_w=torch.zeros(N_ENVS, 4, 3),
            # Upright: gravity in the base frame points down the -z axis.
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]] * N_ENVS),
        )
    )
    sensor = ContactSensorStub()
    terrain = SimpleNamespace(
        cfg=SimpleNamespace(
            terrain_generator=SimpleNamespace(sub_terrains=dict.fromkeys(TERRAIN_NAMES))
        ),
        # Default: gap, stairs, ramp, high_wall.
        terrain_types=(
            torch.tensor([4, 0, 7, 2]) if terrain_types is None else terrain_types
        ),
    )
    env = SimpleNamespace(
        num_envs=N_ENVS,
        device="cpu",
        step_dt=0.02,
        command_manager=SimpleNamespace(get_command=lambda name: command),
        scene=SceneStub(
            torch.tensor([[1.0, 4.0, 0.0]] * N_ENVS),
            {
                "robot": asset,
                "feet_ground_contact": sensor,
                "feet_terrain_height": sensor,
                "terrain": terrain,
            },
        ),
    )
    return env, asset, sensor


def test_feet_air_time_charges_short_swings_by_default() -> None:
    """Two-sided, as in the reference: a short swing is a cost, not a zero."""
    env, _, sensor = make_env(command=FORWARD_CMD.clone())
    sensor.data.last_air_time = torch.tensor([[0.10, 0.30, 0.60, 0.25]] * N_ENVS)
    sensor.first_contact = torch.ones(N_ENVS, 4, dtype=torch.bool)

    reward = wmp_rewards.feet_air_time(env, threshold=0.25)

    # (0.10-0.25) + (0.30-0.25) + (0.60-0.25) + (0.25-0.25) = 0.40 - 0.15 = 0.25
    assert torch.allclose(reward, torch.full((N_ENVS,), 0.25))


def test_feet_air_time_one_sided_mode_pays_only_excess() -> None:
    """``signed=False`` keeps the leap-bonus form: short swings pay nothing."""
    env, _, sensor = make_env(command=FORWARD_CMD.clone())
    sensor.data.last_air_time = torch.tensor([[0.10, 0.30, 0.60, 0.25]] * N_ENVS)
    sensor.first_contact = torch.ones(N_ENVS, 4, dtype=torch.bool)

    reward = wmp_rewards.feet_air_time(env, threshold=0.25, signed=False)

    assert torch.allclose(reward, torch.full((N_ENVS,), 0.40))


def test_feet_air_time_scales_on_obstacle_terrains() -> None:
    """Obstacle tiles multiply the term, other tiles keep the flat value."""
    env, _, sensor = make_env(command=FORWARD_CMD.clone())
    sensor.data.last_air_time = torch.tensor([[0.10, 0.30, 0.60, 0.25]] * N_ENVS)
    sensor.first_contact = torch.ones(N_ENVS, 4, dtype=torch.bool)

    reward = wmp_rewards.feet_air_time(
        env,
        threshold=0.25,
        signed=False,
        obstacle_terrain_names=("gap", "high_wall"),
        obstacle_scale=2.0,
    )

    # terrain_types = (gap, stairs, ramp, high_wall) -> scales (2, 1, 1, 2),
    # applied to the one-sided 0.40 of the swing excess.
    assert torch.allclose(reward, torch.tensor([0.8, 0.4, 0.4, 0.8]))


def test_feet_air_time_requires_a_command() -> None:
    env, _, sensor = make_env(command=torch.zeros(N_ENVS, 3))
    sensor.data.last_air_time = torch.full((N_ENVS, 4), 0.6)
    sensor.first_contact = torch.ones(N_ENVS, 4, dtype=torch.bool)

    assert (wmp_rewards.feet_air_time(env, threshold=0.25) == 0).all()


def test_foot_clearance_penalizes_shortfall_only() -> None:
    env, asset, sensor = make_env(command=FORWARD_CMD.clone())
    sensor.data.heights = torch.tensor([[0.02, 0.30, 0.10, 0.18]] * N_ENVS)
    asset.data.site_lin_vel_w[:, :, 0] = 1.0

    flat = wmp_rewards.foot_clearance_terrain(
        env, target_height=0.5, flat_target_height=0.1
    )
    # Only the 0.02 m foot misses the 0.10 m flat target, by 0.08.
    assert torch.allclose(flat, torch.full((N_ENVS,), 0.08))

    obstacle = wmp_rewards.foot_clearance_terrain(
        env,
        target_height=0.5,
        flat_target_height=0.1,
        obstacle_terrain_names=TERRAIN_NAMES,
    )
    # Every foot is 0.48/0.20/0.40/0.32 below the 0.5 m target.
    assert torch.allclose(obstacle, torch.full((N_ENVS,), 1.40))


def test_foot_clearance_does_not_punish_lifting_high() -> None:
    env, asset, sensor = make_env(command=FORWARD_CMD.clone())
    asset.data.site_lin_vel_w.zero_()

    sensor.data.heights = torch.tensor([[0.02, 0.90, 0.02, 0.90]] * N_ENVS)
    high = wmp_rewards.foot_clearance_terrain(
        env, target_height=0.18, flat_target_height=0.1
    )
    sensor.data.heights = torch.tensor([[0.02, 0.18, 0.02, 0.18]] * N_ENVS)
    on_target = wmp_rewards.foot_clearance_terrain(
        env, target_height=0.18, flat_target_height=0.1
    )

    assert (high == 0).all()
    assert (on_target == 0).all()


def test_stuck_respects_the_grace_period() -> None:
    """A single slow step is free; only a sustained stall is penalized."""
    env, asset, _sensor = make_env(command=FORWARD_CMD.clone())
    fired = []
    for _ in range(30):
        asset.data.root_link_lin_vel_b[0, 0] = 0.0  # env 0 stalls.
        asset.data.root_link_lin_vel_b[1:, 0] = 0.5  # Everyone else walks.
        fired.append(wmp_rewards.stuck(env, speed_threshold=0.12, grace_steps=25))

    first_fire = next(i for i, value in enumerate(fired) if value[0] > 0)
    assert first_fire == 25
    assert (fired[-1][1:] == 0).all()


def test_course_checkpoint_pays_once_per_spacing() -> None:
    env, asset, _ = make_env(command=FORWARD_CMD.clone())
    asset.data.root_link_pos_w[:, 0] = 1.0

    assert (wmp_rewards.course_checkpoint_bonus(env, spacing=1.0) == 0).all()
    # Nothing moved: the term is paid only on the step the index advances.
    assert (wmp_rewards.course_checkpoint_bonus(env, spacing=1.0) == 0).all()

    asset.data.root_link_pos_w[:, 0] = 1.0 + 2.4
    assert torch.allclose(
        wmp_rewards.course_checkpoint_bonus(env, spacing=1.0),
        torch.full((N_ENVS,), 2.0),
    )
    assert (wmp_rewards.course_checkpoint_bonus(env, spacing=1.0) == 0).all()


def test_course_checkpoint_handles_moving_backwards_and_resets() -> None:
    env, asset, _ = make_env(command=FORWARD_CMD.clone())
    asset.data.root_link_pos_w[:, 0] = 1.5
    wmp_rewards.course_checkpoint_bonus(env, spacing=1.0)  # Prime the index at 0.

    asset.data.root_link_pos_w[0, 0] = 1.2  # Env 0 backs up, no penalty.
    assert (wmp_rewards.course_checkpoint_bonus(env, spacing=1.0) == 0).all()

    asset.data.root_link_pos_w[:, 0] = 1.0  # Episode reset.
    assert (wmp_rewards.course_checkpoint_bonus(env, spacing=1.0) == 0).all()

    asset.data.root_link_pos_w[:, 0] = 1.0 + 1.0  # Crosses one checkpoint.
    assert torch.allclose(
        wmp_rewards.course_checkpoint_bonus(env, spacing=1.0),
        torch.ones(N_ENVS),
    )


def test_course_checkpoint_respects_terrain_and_command_masks() -> None:
    env, asset, _ = make_env(command=FORWARD_CMD.clone())
    asset.data.root_link_pos_w[:, 0] = 1.0 + 3.0

    reward = wmp_rewards.course_checkpoint_bonus(
        env, spacing=1.0, terrain_names=("gap", "high_wall")
    )
    # terrain_types = (gap, stairs, ramp, high_wall): ramps are excluded.
    assert torch.allclose(reward, torch.tensor([3.0, 0.0, 0.0, 3.0]))

    env.command_manager = SimpleNamespace(
        get_command=lambda name: torch.zeros(N_ENVS, 3)
    )
    env.scene["robot"].data.root_link_pos_w[:, 0] = 1.0 + 6.0
    assert (wmp_rewards.course_checkpoint_bonus(env, spacing=1.0) == 0).all()


def test_course_checkpoint_stops_paying_when_the_base_leans() -> None:
    env, asset, _ = make_env(command=FORWARD_CMD.clone())
    # Paying a nose-down lunge is what turned the 2026-10-17_22-32-26 run into
    # dragging hind legs: gate on the tilt instead.
    asset.data.projected_gravity_b[:, 2] = -1.0
    asset.data.root_link_pos_w[:, 0] = 1.0 + 3.0
    assert torch.allclose(
        wmp_rewards.course_checkpoint_bonus(env, spacing=1.0),
        torch.tensor([3.0, 3.0, 3.0, 3.0]),
    )

    env, asset, _ = make_env(command=FORWARD_CMD.clone())
    # 45 deg of pitch: -cos(45) = -0.707, below the 0.85 upright floor.
    asset.data.projected_gravity_b[:, 2] = -0.707
    asset.data.root_link_pos_w[:, 0] = 1.0 + 3.0
    assert (wmp_rewards.course_checkpoint_bonus(env, spacing=1.0) == 0).all()

    # 20 deg is still within the gate (cos = 0.94). The tilt only suppresses the
    # payment; the index still tracks the robot so it cannot bank progress while
    # it is falling over and cash it in once it recovers.
    asset.data.projected_gravity_b[:, 2] = -0.94
    asset.data.root_link_pos_w[:, 0] = 1.0 + 6.0
    assert torch.allclose(
        wmp_rewards.course_checkpoint_bonus(env, spacing=1.0),
        torch.tensor([3.0, 3.0, 3.0, 3.0]),
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
