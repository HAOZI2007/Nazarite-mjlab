"""Check privileged scan geometry and native teacher inference/reset parity."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from sim2sim.teacher import TeacherHeightScan, TeacherPolicy
from sim2sim.xz1_teacher import DEFAULT_POLICY


def test_height_scan_order_yaw_visibility_and_missing_surface():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom type="plane" size="5 5 .1"/>
      <geom type="box" pos=".6 .3 .1" size=".02 .02 .1"/>
      <geom type="box" pos="-.3 .6 .1" size=".02 .02 .1"/>
      <geom type="box" pos=".3 .3 .4" size=".02 .02 .1" group="3"/>
      <body name="base_link" pos="0 0 .3"><freejoint/>
        <geom type="sphere" size=".05" group="3"/>
      </body>
    </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    sensor = TeacherHeightScan(model)
    original = sensor.read(data).reshape(12, 11)
    assert original[7, 7] == pytest.approx(-0.2)  # x=.6, y=.3
    assert original[5, 7] == pytest.approx(0)  # Ignore group3 box.
    data.qpos[3:7] = [2**-0.5, 0, 0, 2**-0.5]
    rotated = sensor.read(data).reshape(12, 11)
    assert rotated[7, 7] == pytest.approx(-0.2)  # World x=-.3, y=.6.
    np.testing.assert_allclose(sensor.last_hits[7 * 11 + 7], [-0.3, 0.6, 0.2])
    data.qpos[2] = 10.0
    np.testing.assert_array_equal(sensor.read(data), 1)


@pytest.fixture(scope="module")
def policy():
    if not DEFAULT_POLICY.exists():
        pytest.skip("Local teacher checkpoint unavailable")
    torch.set_num_threads(2)
    return TeacherPolicy(DEFAULT_POLICY)


def test_teacher_matches_current_first_export_and_resets(policy):
    rng = np.random.default_rng(4)
    history = rng.normal(size=282).astype(np.float32)
    history.reshape(6, 47)[:, 35:] *= 20
    scan = rng.uniform(-1, 1, size=132).astype(np.float32)
    policy.reset()
    actual = policy.step(history, scan)
    export = policy.actor.as_onnx()
    with torch.inference_mode():
        scan_input = scan if policy.scan_history_length == 1 else policy.scan_history
        expected = export(
            torch.from_numpy(history)[None], torch.from_numpy(scan_input.copy())[None]
        )
    np.testing.assert_allclose(actual, expected.numpy()[0], atol=2e-6, rtol=2e-6)
    if policy.scan_history_length > 1 or policy.scan_update_steps > 1:
        initial_action = actual.copy()
        expected_history = np.repeat(scan[None], policy.scan_history_length, axis=0)
        count = policy.scan_update_steps * (policy.scan_history_length + 1)
        for index in range(1, count + 1):
            scan_next = scan * index / (count + 2)
            actual = policy.step(history, scan_next)
            if index % policy.scan_update_steps == 0:
                expected_history[:-1] = expected_history[1:].copy()
                expected_history[-1] = scan_next
            np.testing.assert_array_equal(policy.scan_history, expected_history)
            with torch.inference_mode():
                expected = export(
                    torch.from_numpy(history)[None],
                    torch.from_numpy(
                        (
                            policy.scan_history[0]
                            if policy.scan_history_length == 1
                            else policy.scan_history
                        ).copy()
                    )[None],
                )
            np.testing.assert_allclose(
                actual, expected.numpy()[0], atol=2e-6, rtol=2e-6
            )
        policy.reset()
        assert policy.scan_history is None
        np.testing.assert_allclose(policy.step(history, scan), initial_action)
    policy.reset()
    np.testing.assert_array_equal(policy.last_action, 0)
    with pytest.raises(ValueError, match="132"):
        policy.step(history, scan[:100])


def test_teacher_gamepad_reset_restarts_history_and_scan(monkeypatch, policy):
    from sim2sim import xz1_teacher

    observations = []
    scans = []
    pads = []

    class FakePad:
        name = path = axis_mapping = "test pad"

        def __init__(self, *args, **kwargs):
            self.polls = 0
            self.closed = False
            pads.append(self)

        def poll(self):
            self.polls += 1
            return np.array([0.2, -0.1, 0.1], dtype=np.float32)

        def consume_button(self, button):
            return self.polls == 3

        def close(self):
            self.closed = True

    class RecordedPolicy:
        scan_history_length = 1

        def __init__(self, *args):
            policy.reset()

        @property
        def last_action(self):
            return policy.last_action

        def reset(self):
            policy.reset()

        def step(self, history, scan):
            observations.append(history.copy())
            scans.append(scan.copy())
            return policy.step(history, scan)

    monkeypatch.setattr(xz1_teacher, "TeacherPolicy", RecordedPolicy)
    monkeypatch.setattr(xz1_teacher, "GamepadController", FakePad)
    monkeypatch.setattr(
        "sys.argv", ["teacher", "--gamepad", "--headless", "--duration", "0.1"]
    )
    xz1_teacher.main()
    assert pads[0].polls == 5 and pads[0].closed
    np.testing.assert_array_equal(observations[2], observations[0])
    np.testing.assert_array_equal(scans[2], scans[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires Warp raycast")
def test_height_scan_matches_training_sensor():
    from mjlab.entity import EntityCfg
    from mjlab.scene import Scene, SceneCfg
    from mjlab.sensor import ObjRef, RayCastSensorCfg
    from mjlab.sim.sim import Simulation, SimulationCfg
    from nazarite.config.train_config.XZ1_env_cfgs.teacher_env_cfg import (
        ParkourScanPatternCfg,
        parkour_height_scan,
    )

    xml = """<mujoco><worldbody>
      <geom type="plane" size="5 5 .1"/>
      <geom type="box" pos=".6 .3 .1" size=".06 .06 .1"/>
      <geom type="box" pos=".3 0 .4" size=".1 .1 .1" group="3"/>
      <body name="base_link"><geom type="sphere" size=".05" group="3"/></body>
    </worldbody></mujoco>"""
    scene = Scene(
        SceneCfg(
            num_envs=1,
            entities={
                "robot": EntityCfg(
                    spec_fn=lambda: mujoco.MjSpec.from_string(xml),
                    init_state=EntityCfg.InitialStateCfg(
                        pos=(0, 0, 0.3), rot=(0.951251, 0.044943, 0.167731, 0.254887)
                    ),
                )
            },
            sensors=(
                RayCastSensorCfg(
                    name="teacher_terrain_scan",
                    frame=ObjRef(type="body", name="base_link", entity="robot"),
                    pattern=ParkourScanPatternCfg(),
                    ray_alignment="yaw",
                    max_distance=5,
                    include_geom_groups=(0,),
                    exclude_parent_body=True,
                ),
            ),
        ),
        "cuda:0",
    )
    sim = Simulation(
        num_envs=1, cfg=SimulationCfg(), model=scene.compile(), device="cuda:0"
    )
    scene.initialize(sim.mj_model, sim.model, sim.data)
    sim.set_sensor_context(scene.sensor_context)
    sim.forward()
    sim.sense()
    expected = parkour_height_scan(SimpleNamespace(scene=scene)).cpu().numpy()[0]
    data = mujoco.MjData(sim.mj_model)
    mujoco.mj_forward(sim.mj_model, data)
    actual = TeacherHeightScan(sim.mj_model, "robot/base_link").read(data)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
