"""Exercise Go2 control loops without a GUI or a trained checkpoint."""

import importlib
from pathlib import Path

import numpy as np
import pytest

from sim2sim.go2 import main as launcher


@pytest.mark.parametrize(
    "mode,dimension", [("baseline", 45), ("wtw", 498), ("him", 282)]
)
def test_go2_control_loop(mode, dimension, monkeypatch):
    monkeypatch.setattr("sys.argv", ["sim2sim", "--mode", mode, "--no-camera-follow"])
    monkeypatch.setattr(launcher, "default_policy_path", lambda mode: Path("unused.pt"))
    observations = []

    class Policy:
        def __init__(self, path, **kwargs):
            self.obs_dim = kwargs.get("expected_obs_dim", dimension)
            self.last_action = np.zeros(12, dtype=np.float32)

        def reset(self):
            self.last_action.fill(0)

        def step(self, obs):
            assert obs.shape == (dimension,)
            assert np.isfinite(obs).all()
            observations.append(obs.copy())
            return self.last_action.copy()

    class Viewer:
        steps = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def is_running(self):
            self.steps += 1
            return self.steps <= 3

        def sync(self):
            pass

    monkeypatch.setattr(launcher, "PolicyRunner", Policy)
    monkeypatch.setattr(launcher, "HIMPolicy", Policy)
    monkeypatch.setattr(
        launcher.mujoco.viewer, "launch_passive", lambda *a, **kw: Viewer()
    )
    launcher.main()
    assert len(observations) == 3


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("sim2sim.main", "sim2sim.go2.main"),
        ("sim2sim.xz1_him", "sim2sim.xz1.him.main"),
        ("sim2sim.xz1_teacher", "sim2sim.xz1.teacher.main"),
        ("sim2sim.xz1_wmp", "sim2sim.xz1.wmp.main"),
        ("sim2sim.him.policy", "sim2sim.shared.him.policy"),
        ("sim2sim.wmp.policy", "sim2sim.xz1.wmp.policy"),
    ],
)
def test_old_launchers_and_policy_imports_forward(old, new):
    assert importlib.import_module(old) is importlib.import_module(new)
