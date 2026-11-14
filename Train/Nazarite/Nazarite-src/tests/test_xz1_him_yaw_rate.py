"""Near-zero steering cost and its base-HIM-only configuration."""

from types import SimpleNamespace

import pytest
import torch

from nazarite.config.train_config.xz1 import him_course
from nazarite.config.train_config.xz1.him_env_cfg import (
    Nazarite_HIM_Complex_Terrain_XZ1,
)
from nazarite.config.train_config.xz1.teacher_env_cfg import (
    Nazarite_WMP_HIM_Teacher_XZ1,
)


def test_near_zero_yaw_cost_tracks_small_commands_and_releases_turns():
    # Include stationary, reverse, lateral, threshold boundaries and turns.
    command = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.05],
            [0.0, 1.0, -0.05],
            [1.0, 0.0, 0.1],
            [1.0, 0.0, -0.1],
            [1.0, 0.0, 0.1001],
            [1.0, 0.0, -0.1001],
        ]
    )
    angular_velocity = torch.tensor(
        [
            [2.0, -3.0, 0.0],
            [0.0, 0.0, 0.2],
            [0.0, 0.0, 0.05],
            [0.0, 0.0, 0.05],
            [0.0, 0.0, 0.2],
            [0.0, 0.0, -0.2],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_command=lambda _: command),
        scene={
            "robot": SimpleNamespace(
                data=SimpleNamespace(
                    root_link_ang_vel_b=angular_velocity,
                )
            )
        },
    )
    cost = him_course.xz1_near_zero_yaw_rate_cost(env, "twist")
    torch.testing.assert_close(
        cost, torch.tensor([0.0, 0.04, 0.0, 0.01, 0.01, 0.01, 0.0, 0.0])
    )


@pytest.mark.parametrize("play", [False, True])
def test_near_zero_yaw_cost_is_local_to_base_him(play):
    cfg = Nazarite_HIM_Complex_Terrain_XZ1(play=play)
    term = cfg.rewards["near_zero_yaw_rate"]
    assert term.func is him_course.xz1_near_zero_yaw_rate_cost
    assert term.weight == -1.0
    assert term.params["command_threshold"] == 0.1
    assert "near_zero_yaw_rate" not in Nazarite_WMP_HIM_Teacher_XZ1().rewards
