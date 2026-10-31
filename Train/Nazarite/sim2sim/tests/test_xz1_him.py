"""Verify the XZ1 deployment contract against its trained artifacts."""

import mujoco
import numpy as np
import pytest
import yaml

from sim2sim.shared.him.observation import HIMObservationBuilder
from sim2sim.shared.him.policy import HIMPolicy
from sim2sim.xz1.him.main import (
    ACTION_OFFSET,
    DEFAULT_POLICY,
    DEFAULT_Q,
    JOINT_NAMES,
    RUN,
    SCENE,
    XZ1IO,
)


def test_xz1_servos_and_scene_reset():
    io = XZ1IO(SCENE, (-6.2, -2.0, 0.66), 90.0)
    assert io.model.nu == 12
    assert io.model.geom("arena").id >= 0
    assert io.model.geom("start_zone_2_fill").id >= 0
    assert [io.model.joint(i).name for i in range(1, 13)] == JOINT_NAMES
    np.testing.assert_allclose(io.get_joint_pos(), DEFAULT_Q)
    np.testing.assert_allclose(io.data.ctrl[io.ctrl_ids], ACTION_OFFSET)
    obs = HIMObservationBuilder().build(io, np.zeros(3), np.zeros(12))
    np.testing.assert_array_equal(obs[11:23], 0.0)
    np.testing.assert_allclose(io.model.dof_armature[io.qvel_ids], 0.025392063)
    np.testing.assert_allclose(io.model.actuator_gainprm[:, 0], [32, 32, 35] * 4)
    np.testing.assert_allclose(io.model.actuator_biasprm[:, 2], [-1.9, -1.9, -1.8] * 4)
    np.testing.assert_allclose(io.model.actuator_forcerange, [[-31.7, 31.7]] * 12)
    assert io.model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    io.send_target(DEFAULT_Q + 0.1)
    io.step()
    io.reset()
    np.testing.assert_allclose(io.data.qpos[:3], [-6.2, -2, 0.66])
    np.testing.assert_allclose(io.data.qpos[3:7], [2**-0.5, 0, 0, 2**-0.5])
    np.testing.assert_array_equal(io.data.qvel, 0)
    np.testing.assert_allclose(io.data.ctrl[io.ctrl_ids], ACTION_OFFSET)


def test_action_offset_matches_saved_training_contract():
    config_path = RUN / "params/env.yaml"
    if not config_path.exists():
        pytest.skip("Local training config is not available")
    # BaseLoader treats Python-specific YAML tags as data, without executing them.
    config = yaml.load(config_path.read_text(), Loader=yaml.BaseLoader)
    offsets = config["actions"]["joint_pos"]["offset"]
    expected = np.array([float(offsets[name]) for name in JOINT_NAMES])
    np.testing.assert_allclose(ACTION_OFFSET, expected, atol=1e-7, rtol=0)


def test_checkpoint_stands_at_training_height_in_deployment_loop(monkeypatch):
    from sim2sim.xz1.him import main as xz1_him

    if not DEFAULT_POLICY.exists():
        pytest.skip("Local training checkpoint is not available")
    heights = []

    class RecordedIO(XZ1IO):
        def step(self):
            super().step()
            heights.append(self.data.qpos[2])

    monkeypatch.setattr(xz1_him, "XZ1IO", RecordedIO)
    monkeypatch.setattr("sys.argv", ["xz1_him", "--headless", "--duration", "15"])
    xz1_him.main()
    # Final five seconds: catch the missing-offset regression (~0.239 m).
    settled = np.asarray(heights[-250:])
    assert len(heights) == 750
    assert np.all(np.isfinite(settled))
    assert 0.29 < settled.mean() < 0.32
    assert np.ptp(settled) < 0.01


def test_checkpoint_matches_export():
    checkpoint = DEFAULT_POLICY
    export = RUN / "policy.onnx"
    if not checkpoint.exists() or not export.exists():
        pytest.skip("Local training artifacts are not available")
    torch_policy = HIMPolicy(checkpoint, joint_names=JOINT_NAMES)
    onnx_policy = HIMPolicy(export, joint_names=JOINT_NAMES)
    obs = np.random.default_rng(7).normal(size=282).astype(np.float32)
    np.testing.assert_allclose(
        torch_policy.step(obs), onnx_policy.step(obs), atol=2e-5, rtol=2e-5
    )


def test_gamepad_commands_and_reset(monkeypatch):
    from sim2sim.xz1.him import main as xz1_him

    if not DEFAULT_POLICY.exists():
        pytest.skip("Local training checkpoint is not available")
    instances = []

    class FakePad:
        name = "test pad"
        path = "test device"
        axis_mapping = "test axes"

        def __init__(self, *args, **kwargs):
            self.polls = 0
            self.closed = False
            instances.append(self)

        def poll(self):
            self.polls += 1
            return np.array([0.2, -0.1, 0.1], dtype=np.float32)

        def consume_button(self, button):
            assert button == "BTN_START"
            return self.polls == 3

        def close(self):
            self.closed = True

    observations = []
    original = HIMObservationBuilder.build

    def capture(self, io, command, last_action):
        obs = original(self, io, command, last_action)
        observations.append(obs.copy())
        return obs

    monkeypatch.setattr(xz1_him, "GamepadController", FakePad)
    monkeypatch.setattr(HIMObservationBuilder, "build", capture)
    monkeypatch.setattr(
        "sys.argv", ["xz1_him", "--headless", "--duration", "0.1", "--gamepad"]
    )
    xz1_him.main()
    assert instances[0].polls == 5
    assert instances[0].closed
    np.testing.assert_allclose(observations[0][6:9], [0.2, -0.1, 0.1])
    # Start resets phase, action and history to the original spawn frame.
    np.testing.assert_allclose(observations[2], observations[0])
