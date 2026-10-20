"""Unit tests for the gait-shaping terms added after the 2026-10-18 run.

The run walked nose-down with its hind feet skidding, and every gait term in the
task was worth less than one percent of the velocity term, so the diagnostics and
the new penalties are the part that has to be demonstrably right. Everything here
is plain tensor code driven by stubs, so it runs without a GPU.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from nazarite.mdp import wmp_rewards, wmp_terminations

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


class _Scene:
    def __init__(self, items: dict) -> None:
        self._items = items
        self.env_origins = torch.zeros(N_ENVS, 3)
        self.terrain = items["terrain"]

    def __getitem__(self, key: str):
        return self._items[key]


def _make_env(
    command: torch.Tensor | None = None,
    terrain_types: torch.Tensor | None = None,
    heights: torch.Tensor | None = None,
) -> SimpleNamespace:
    command = torch.zeros(N_ENVS, 3) if command is None else command
    sensor = SimpleNamespace(
        data=SimpleNamespace(
            last_air_time=torch.zeros(N_ENVS, 4),
            current_contact_time=torch.zeros(N_ENVS, 4),
            found=torch.zeros(N_ENVS, 4, dtype=torch.bool),
            heights=torch.zeros(N_ENVS, 4),
        ),
        compute_first_contact=lambda dt: torch.ones(N_ENVS, 4, dtype=torch.bool),
    )
    asset = SimpleNamespace(
        data=SimpleNamespace(
            root_link_lin_vel_b=torch.zeros(N_ENVS, 3),
            root_link_pos_w=torch.zeros(N_ENVS, 3),
            site_lin_vel_w=torch.zeros(N_ENVS, 4, 3),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]] * N_ENVS),
        )
    )
    clearance = SimpleNamespace(
        data=SimpleNamespace(
            heights=torch.zeros(N_ENVS, 1) if heights is None else heights
        )
    )
    terrain = SimpleNamespace(
        cfg=SimpleNamespace(
            terrain_generator=SimpleNamespace(sub_terrains=dict.fromkeys(TERRAIN_NAMES))
        ),
        terrain_types=(
            torch.tensor([4, 0, 7, 2]) if terrain_types is None else terrain_types
        ),
        terrain_levels=torch.full((N_ENVS,), 5, dtype=torch.long),
    )
    env = SimpleNamespace(
        num_envs=N_ENVS,
        device="cpu",
        step_dt=0.02,
        episode_length_buf=torch.zeros(N_ENVS, dtype=torch.long),
        command_manager=SimpleNamespace(get_command=lambda name: command),
        scene=_Scene(
            {
                "robot": asset,
                "feet_ground_contact": sensor,
                "feet_terrain_height": sensor,
                "wmp_base_clearance": clearance,
                "terrain": terrain,
            }
        ),
    )
    return env


def test_feet_air_time_band_charges_swings_that_are_too_short() -> None:
    """The one-sided term paid zero for a micro-hop; the band makes it negative."""
    env = _make_env(command=FORWARD_CMD.clone())
    sensor = env.scene["feet_ground_contact"]
    sensor.data.last_air_time = torch.tensor([[0.10, 0.30, 0.60, 0.40]] * N_ENVS)

    reward = wmp_rewards.feet_air_time_band(env, threshold_min=0.28, threshold_max=0.55)

    # (0.10-0.28) + (0.30-0.28) + (0.55-0.28) + (0.40-0.28) = 0.23
    assert torch.allclose(reward, torch.full((N_ENVS,), 0.23))


def test_feet_air_time_band_saturates_above_the_window() -> None:
    env = _make_env(command=FORWARD_CMD.clone())
    sensor = env.scene["feet_ground_contact"]
    sensor.data.last_air_time = torch.tensor([[0.55, 0.90, 0.55, 0.55]] * N_ENVS)

    # 0.90 is capped at 0.55 so that flight time beyond the window is not paid.
    assert torch.allclose(
        wmp_rewards.feet_air_time_band(env, threshold_min=0.28, threshold_max=0.55),
        torch.full((N_ENVS,), 0.27 * 4),
    )


def test_foot_clearance_swing_only_ignores_a_stance_foot() -> None:
    env = _make_env(command=FORWARD_CMD.clone())
    sensor = env.scene["feet_ground_contact"]
    sensor.data.found = torch.tensor([[True, False, True, False]] * N_ENVS)
    sensor.data.heights = torch.full((N_ENVS, 4), 0.02)
    env.scene["robot"].data.site_lin_vel_w[:, :, 0] = 1.0

    stance_and_swing = wmp_rewards.foot_clearance_terrain(
        env, target_height=0.5, flat_target_height=0.1, swing_only=True
    )
    everything = wmp_rewards.foot_clearance_terrain(
        env, target_height=0.5, flat_target_height=0.1
    )

    # Four feet x 0.08 shortfall -> 0.32; only the two swing feet -> 0.16.
    assert torch.allclose(everything, torch.full((N_ENVS,), 0.32))
    assert torch.allclose(stance_and_swing, torch.full((N_ENVS,), 0.16))


def test_base_clearance_is_terrain_relative_and_one_sided() -> None:
    """Low is penalized, level is full marks, flying above a trench is free."""
    heights = torch.tensor([[0.30], [0.10], [0.60], [0.28]])
    env = _make_env(command=FORWARD_CMD.clone(), heights=heights)

    reward = wmp_rewards.base_clearance_reward(
        env, target_height=0.30, std=0.06, excluded_terrain_names=()
    )

    assert reward[0] == pytest.approx(1.0)
    assert reward[1] < 1.0e-3  # 0.20 m below target: effectively zero.
    assert reward[2] == pytest.approx(1.0)  # Above target is not an error.
    assert reward[3] == pytest.approx(torch.exp(torch.tensor(-((0.02 / 0.06) ** 2))))


def test_base_clearance_is_waived_on_the_low_ceiling_tile() -> None:
    heights = torch.tensor([[0.10], [0.10], [0.10], [0.10]])
    env = _make_env(command=FORWARD_CMD.clone(), heights=heights)

    # terrain_types = (gap, stairs, ramp, high_wall): the first env is the gap.
    reward = wmp_rewards.base_clearance_reward(
        env, target_height=0.30, std=0.06, excluded_terrain_names=("gap",)
    )

    assert (reward[0] == 0).all()
    assert (reward[1:] < 1.0e-3).all()


def test_feet_gait_terrain_softens_only_the_listed_tiles() -> None:
    env = _make_env(command=FORWARD_CMD.clone())
    env.episode_length_buf = torch.full((N_ENVS,), 7, dtype=torch.long)
    env.scene["feet_ground_contact"].data.current_contact_time = torch.tensor(
        [[0.0, 0.1, 0.0, 0.1]] * N_ENVS
    )

    full = wmp_rewards.feet_gait_terrain(env, period=0.6, offset=(0.0, 0.5, 0.5, 0.0))
    softened = wmp_rewards.feet_gait_terrain(
        env,
        period=0.6,
        offset=(0.0, 0.5, 0.5, 0.0),
        soft_terrain_names=("gap", "high_wall"),
        soft_scale=0.3,
    )

    assert softened[0] == pytest.approx(float(full[0]) * 0.3)
    assert softened[3] == pytest.approx(float(full[3]) * 0.3)
    assert softened[1] == pytest.approx(float(full[1]))
    assert softened[2] == pytest.approx(float(full[2]))


def test_stall_timeout_waits_for_the_time_limit_and_clears_on_reset() -> None:
    env = _make_env(command=torch.tensor([[0.5, 0.0, 0.0]] * N_ENVS))
    asset = env.scene["robot"]
    asset.data.root_link_lin_vel_b[:, 0] = 0.1  # 20 % of the command: stalled.

    fired = []
    for step in range(120):
        env.episode_length_buf += 1
        fired.append(wmp_terminations.stall_timeout(env, time_limit=2.0))
        assert step == 0 or fired[-1][0] == (step >= 99)

    # 2.0 s at a 0.02 s step is 100 stalled steps.
    assert not any(bool(value[0]) for value in fired[:99])
    assert bool(fired[99][0])

    env.episode_length_buf.zero_()  # Terminal reset.
    assert not bool(wmp_terminations.stall_timeout(env, time_limit=2.0)[0])


def test_stall_timeout_ignores_easy_rows_and_standing_commands() -> None:
    env = _make_env(command=torch.tensor([[0.5, 0.0, 0.0]] * N_ENVS))
    env.scene["robot"].data.root_link_lin_vel_b[:, 0] = 0.0
    env.scene["terrain"].terrain_levels = torch.zeros(N_ENVS, dtype=torch.long)

    fired = []
    for _ in range(120):
        env.episode_length_buf += 1
        fired.append(wmp_terminations.stall_timeout(env, time_limit=2.0))

    # The level gate keeps the termination off the easy rows.
    assert not any(bool(value.any()) for value in fired)

    env.scene["terrain"].terrain_levels = torch.full((N_ENVS,), 5, dtype=torch.long)
    env.command_manager = SimpleNamespace(
        get_command=lambda name: torch.zeros(N_ENVS, 3)
    )
    fired = []
    for _ in range(120):
        env.episode_length_buf += 1
        fired.append(wmp_terminations.stall_timeout(env, time_limit=2.0))

    # A standing command is not a stall.
    assert not any(bool(value.any()) for value in fired)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
