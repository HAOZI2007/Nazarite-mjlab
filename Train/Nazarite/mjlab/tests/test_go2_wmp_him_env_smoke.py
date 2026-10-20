"""End-to-end smoke test for both Go2 WMP-HIM stages.

The unit tests cover the reward maths against stubs, but the two failures that
actually stopped training runs were integration bugs: a curriculum that received
terrain names its generator did not have, and a ``found`` tensor that turned out
to be an integer count rather than a boolean. Both only appeared once a real
environment stepped. This test builds each stage on CPU, steps it, and evaluates
every reward, termination and metric term through the real managers.
"""

from __future__ import annotations

import pytest
import torch
from nazarite.config.train_config.env_cfgs.wmp_him_go2_env_cfg import (
  Nazarite_WMP_HIM_Complex_Terrain_Go2,
)
from nazarite.config.train_config.wmp_him_go2_cfg import go2_wmp_him_runner_cfg

from mjlab.envs import ManagerBasedRlEnv

NUM_ENVS = 4
ACTION_DIM = 12


def _build(flat: bool) -> ManagerBasedRlEnv:
  cfg = Nazarite_WMP_HIM_Complex_Terrain_Go2(flat=flat)
  cfg.scene.num_envs = NUM_ENVS
  return ManagerBasedRlEnv(cfg=cfg, device="cpu")


@pytest.mark.slow
@pytest.mark.parametrize("flat", [True, False])
def test_stage_steps_and_every_term_evaluates(flat: bool) -> None:
  env = _build(flat)
  try:
    env.reset()
    for _ in range(3):
      _, reward, terminated, truncated, _ = env.step(torch.zeros(NUM_ENVS, ACTION_DIM))

    assert torch.isfinite(reward).all()
    assert not terminated.any()
    assert not truncated.any()

    # Every configured term must return a finite, per-environment value through
    # the real signature (params included), not just through a stub.
    for manager in (
      env.reward_manager,
      env.termination_manager,
      env.metrics_manager,
    ):
      for name, term in zip(manager._term_names, manager._term_cfgs, strict=False):
        value = term.func(env, **term.params)
        assert value.shape == (NUM_ENVS,), name
        assert torch.isfinite(value.float()).all(), name

    # The curriculum gate reads these, so they have to be wired up here too.
    assert "gait_diagonal_match" in env.metrics_manager._term_names

    # One-step widths. The actor lost its gait clock (47 -> 45) so that the
    # discriminator owns swing timing, and the estimator group is a copy of the
    # critic terms, so it drops to 48 (45 plus base linear velocity). The HIM
    # target slices in the runner config are tied to these two numbers.
    dims = env.observation_manager.group_obs_dim
    # Actor is (history, one-step): six frames of 45 numbers.
    assert dims["actor"] == (6, 45)
    assert dims["estimator"] == (48,)
    assert "phase" not in env.observation_manager.cfg["actor"].terms

    # Tie the runner's HIM constants to the widths the environment actually
    # reports. A stale 47 in any of them is the failure mode this guards: it
    # only surfaces on the first PPO update, long after the environment builds.
    runner_cfg = go2_wmp_him_runner_cfg()
    one_step = dims["actor"][1]
    estimator_width = dims["estimator"][0]
    assert runner_cfg.actor.num_one_step_obs == one_step
    assert sum(runner_cfg.actor.history_term_dims) == one_step
    assert runner_cfg.actor.action_observation_slice[1] == one_step
    start, stop = runner_cfg.algorithm.estimator_velocity_slice
    assert stop == estimator_width and stop - start == 3
    assert (
      sum(b - a for a, b in runner_cfg.algorithm.estimator_target_slices) == one_step
    )
  finally:
    env.close()


@pytest.mark.slow
def test_stateful_terms_survive_a_mode_switch() -> None:
  """Playback after training used to fail on "Inplace update to inference tensor".

  The training loop steps inside ``torch.inference_mode()``, so every lazily
  allocated counter (stall, stuck, checkpoint index, gait metrics) is created as
  an inference tensor; the same terms then raised on the first playback step
  because they updated those tensors in place from outside that mode. Only the
  terms are exercised here: mjlab's own actuator delay buffer has the same latent
  issue, but training and playback are separate processes so it never sees it.
  """
  env = _build(flat=False)
  try:
    env.reset()
    managers = (
      env.reward_manager,
      env.termination_manager,
      env.metrics_manager,
    )

    def evaluate_all() -> None:
      for manager in managers:
        for _name, term in zip(manager._term_names, manager._term_cfgs, strict=False):
          term.func(env, **term.params)

    with torch.inference_mode():
      evaluate_all()
    # The counters exist now as inference tensors. Evaluating from the default
    # mode is what used to raise.
    evaluate_all()
  finally:
    env.close()


@pytest.mark.slow
def test_base_clearance_reads_the_deck_under_the_trunk() -> None:
  """Guards the sensor wiring: a bad frame or group reads as a permanent crouch."""
  env = _build(flat=True)
  try:
    env.reset()
    env.step(torch.zeros(NUM_ENVS, ACTION_DIM))
    heights = env.scene["wmp_base_clearance"].data.heights
    # Nominal standing height is 0.32 m; anything far below means the rays are
    # hitting the robot itself or missing the deck entirely.
    assert heights.min() > 0.28
    assert heights.max() < 0.40
  finally:
    env.close()


@pytest.mark.slow
def test_obstacle_columns_steer_back_towards_plus_x() -> None:
  """Heading mode must target +x on the course instead of clamping yaw to zero.

  A zero yaw command lets drift accumulate; a zero heading target actively turns
  the robot back, which is what the reference implementation does.
  """
  env = _build(flat=False)
  try:
    env.reset()
    term = env.command_manager.get_term("twist")
    assert term.cfg.heading_command is True
    assert term.is_heading_env.all()
    assert torch.all(term.heading_target == 0.0)
    env.step(torch.zeros(NUM_ENVS, ACTION_DIM))
    yaw = term.vel_command_b[:, 2]
    low, high = term.cfg.ranges.ang_vel_z
    assert torch.all(yaw >= low) and torch.all(yaw <= high)
  finally:
    env.close()
