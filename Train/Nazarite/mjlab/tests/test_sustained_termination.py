from types import SimpleNamespace

import torch

from mjlab.tasks.velocity.mdp.terminations import sustained_illegal_contact


def test_sustained_illegal_contact_ignores_single_impulse():
  force_history = torch.zeros(2, 1, 4, 3)
  force_history[0, 0, 1, 0] = 20.0
  force_history[1, 0, 1:, 0] = 20.0
  env = SimpleNamespace(
    scene={
      "trunk_ground_touch": SimpleNamespace(
        data=SimpleNamespace(force_history=force_history, force=None, found=None)
      )
    },
    extras={"log": {}},
  )

  result = sustained_illegal_contact(
    env,
    "trunk_ground_touch",
    force_threshold=10.0,
    min_consecutive_steps=2,
  )

  assert result.tolist() == [False, True]
  assert "Metrics/trunk_ground_touch_sustained" in env.extras["log"]
