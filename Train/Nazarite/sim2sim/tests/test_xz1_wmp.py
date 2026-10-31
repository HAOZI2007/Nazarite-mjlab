"""Verify native depth geometry and parity with the training visual runtime."""

from dataclasses import asdict
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
from tensordict import TensorDict

from nazarite.config.train_config.xz1.collect_env_cfg import XZ1DepthCfg
from nazarite.rl.xz1.student import XZ1VisualPolicy
from sim2sim.xz1.him.main import SCENE, XZ1IO
from sim2sim.xz1.wmp.camera import CAMERA_NAME, WMPDepthCamera, install_camera
from sim2sim.xz1.wmp.main import DEFAULT_POLICY
from sim2sim.xz1.wmp.policy import WMPStudentPolicy, load_bundle


def test_depth_is_planar_and_hides_robot_groups_without_hiding_visual_meshes():
    model = mujoco.MjModel.from_xml_string(f"""
    <mujoco><worldbody>
      <camera name="{CAMERA_NAME}" pos="0 0 1" fovy="58"/>
      <geom type="plane" size="5 5 .1"/>
      <geom type="plane" pos="0 0 .7" size="5 5 .1" group="3"/>
      <geom name="visual" type="box" pos="0 0 .5" size=".1 .1 .05"
            contype="0" conaffinity="0"/>
    </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = WMPDepthCamera(model, asdict(XZ1DepthCfg()))
    image = camera.read(data)
    assert image.shape == (64, 64)
    # Optical-axis Z is constant on a frontoparallel plane, unlike ray range.
    np.testing.assert_allclose(image[0], 1.0, atol=1e-6)
    np.testing.assert_allclose(image[-1], 1.0, atol=1e-6)
    assert image[32, 32] == pytest.approx(0.45, abs=1e-6)


def test_camera_misses_are_far_and_install_matches_mount():
    spec = mujoco.MjSpec.from_string(
        '<mujoco><worldbody><body name="base_link"/></worldbody></mujoco>'
    )
    cfg = asdict(XZ1DepthCfg())
    install_camera(spec, cfg)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(model.cam_pos[0], cfg["pos"])
    np.testing.assert_allclose(model.cam_quat[0], cfg["quat"], atol=1e-6)
    np.testing.assert_array_equal(WMPDepthCamera(model, cfg).read(data), 2.0)


@pytest.fixture(scope="module")
def saved_bundle():
    if not DEFAULT_POLICY.is_file():
        pytest.skip("Local student checkpoint is unavailable")
    torch.set_num_threads(2)
    return load_bundle(DEFAULT_POLICY)


def test_visual_runtime_matches_training_history_preprocessing_rssm_and_actions(
    saved_bundle,
):
    runtime = WMPStudentPolicy(saved_bundle)
    current_depth = torch.zeros(1, 64, 64, 1)
    sense_calls = []
    env = SimpleNamespace(
        step_dt=0.02,
        num_envs=1,
        device="cpu",
        sim=SimpleNamespace(sense=lambda **kw: sense_calls.append(kw)),
        scene={
            "xz1_depth_camera": SimpleNamespace(
                data=SimpleNamespace(depth=current_depth)
            )
        },
    )
    camera = XZ1DepthCfg(**saved_bundle["camera"])
    reference = XZ1VisualPolicy(env, runtime.actor, runtime.model, camera)
    rng = np.random.default_rng(17)
    reads = []
    for tick in range(13):
        history = rng.normal(size=(6, 47)).astype(np.float32)
        history[:, 35:] *= 15  # Exercise the tighter previous-action clip.
        frame = torch.from_numpy(history[0])
        estimator = torch.cat((frame, torch.zeros(3)))[None]
        obs = TensorDict(
            {
                "actor": torch.from_numpy(history[::-1].copy())[None],
                "estimator": estimator,
            },
            batch_size=[1],
        )
        image = rng.uniform(0.05, 3, size=(64, 64)).astype(np.float32)
        image.flat[:4] = [0, np.nan, np.inf, -1]
        current_depth.copy_(torch.from_numpy(image)[None, ..., None])

        def read(frame=image, step=tick):
            reads.append(step)
            return frame

        torch.manual_seed(100 + tick)
        actual = runtime.step(history.flatten(), read)
        torch.manual_seed(100 + tick)
        with torch.no_grad():
            expected = reference(obs)
        np.testing.assert_allclose(actual, expected.numpy()[0], atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(runtime.feature, reference.feature, atol=0, rtol=0)
        for key in runtime.state:
            torch.testing.assert_close(
                runtime.state[key], reference.state[key], atol=0, rtol=0
            )
        # Model history must contain actual executed actions, including clipping.
        executed = np.clip(actual + 0.01, -10, 10)
        runtime.advance(executed)
        reference.advance(torch.from_numpy(executed)[None])
        torch.testing.assert_close(runtime.actions, reference.actions)
    assert reads == [0, 5, 10]
    assert len(sense_calls) == 3
    runtime.reset()
    assert runtime.tick == runtime.image_updates == 0
    assert not runtime.actions.any() and not runtime.feature.any()
    np.testing.assert_array_equal(runtime.last_action, 0)
    runtime.step(history.flatten(), read)
    assert runtime.image_updates == 1


def test_bundle_reloads_without_reading_source_checkpoints(
    saved_bundle, tmp_path, monkeypatch
):
    path = tmp_path / "deploy.pt"
    torch.save(saved_bundle, path)

    def no_external_files(*args):
        raise AssertionError("Bundle tried to open a source checkpoint")

    monkeypatch.setattr("sim2sim.wmp.policy.file_sha256", no_external_files)
    loaded = load_bundle(path)
    assert loaded["sources"] == saved_bundle["sources"]
    wrong = dict(saved_bundle, control={})
    torch.save(wrong, path)
    with pytest.raises(ValueError, match="servo"):
        load_bundle(path)


def test_robocon_camera_does_not_change_servo_contract(saved_bundle):
    io = XZ1IO(
        SCENE,
        (0, 0, 0.30),
        0,
        configure_robot=lambda spec: install_camera(spec, saved_bundle["camera"]),
    )
    assert io.model.geom("arena").id >= 0
    assert io.model.camera(CAMERA_NAME).id >= 0
    assert io.model.nu == 12
    assert np.isfinite(
        WMPDepthCamera(io.model, saved_bundle["camera"]).read(io.data)
    ).all()


def test_gamepad_start_resets_observations_actions_and_world_model(
    saved_bundle, monkeypatch
):
    from sim2sim.xz1.wmp import main as xz1_wmp

    pads = []
    observations = []
    reset_states = []

    class FakePad:
        name, path, axis_mapping = "test pad", "test device", "test axes"

        def __init__(self, *args, **kwargs):
            self.polls = 0
            self.closed = False
            pads.append(self)

        def poll(self):
            self.polls += 1
            return np.array([0.2, -0.1, 0.1], dtype=np.float32)

        def consume_button(self, button):
            assert button == "BTN_START"
            return self.polls == 3

        def close(self):
            self.closed = True

    class RecordedPolicy(WMPStudentPolicy):
        def step(self, history, depth_reader):
            observations.append(history.copy())
            if self.tick == 0:
                assert not self.actions.any() and not self.feature.any()
                reset_states.append({k: v.clone() for k, v in self.state.items()})
            return super().step(history, depth_reader)

    monkeypatch.setattr(xz1_wmp, "GamepadController", FakePad)
    monkeypatch.setattr(xz1_wmp, "WMPStudentPolicy", RecordedPolicy)
    monkeypatch.setattr(xz1_wmp, "load_bundle", lambda *args: saved_bundle)
    monkeypatch.setattr(
        "sys.argv", ["xz1_wmp", "--headless", "--duration", "0.1", "--gamepad"]
    )
    xz1_wmp.main()
    assert pads[0].polls == 5 and pads[0].closed
    np.testing.assert_allclose(observations[0][6:9], [0.2, -0.1, 0.1])
    np.testing.assert_array_equal(observations[2], observations[0])
    assert len(reset_states) == 2
    for key in reset_states[0]:
        torch.testing.assert_close(reset_states[0][key], reset_states[1][key])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires Warp CUDA renderer")
def test_native_depth_matches_training_warp_renderer():
    from mjlab.entity import EntityCfg
    from mjlab.scene import Scene, SceneCfg
    from mjlab.sensor import CameraSensorCfg
    from mjlab.sim.sim import Simulation, SimulationCfg

    cfg = asdict(XZ1DepthCfg())
    xml = f"""<mujoco><worldbody>
      <geom type="plane" size="5 5 .1"/>
      <geom type="box" pos="1 .1 .2" size=".1 .3 .2"/>
      <geom type="box" pos=".7 -.3 .1" size=".1 .1 .1"
            contype="0" conaffinity="0"/>
      <geom type="box" pos=".3 0 .4" size=".1 .1 .1" group="3"/>
      <camera name="depth" pos="0 0 .45"
              quat="{" ".join(map(str, cfg["quat"]))}" fovy="{cfg["fovy"]}"/>
    </worldbody></mujoco>"""
    scene = Scene(
        SceneCfg(
            num_envs=1,
            entities={
                "world": EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(xml))
            },
            sensors=(
                CameraSensorCfg(
                    name="depth",
                    camera_name="world/depth",
                    width=64,
                    height=64,
                    data_types=("depth",),
                    enabled_geom_groups=(0,),
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
    expected = scene["depth"].data.depth[0, ..., 0].cpu().numpy()
    expected = np.where(
        np.isfinite(expected) & (expected > 0), np.minimum(expected, 2), 2
    )
    data = mujoco.MjData(sim.mj_model)
    mujoco.mj_forward(sim.mj_model, data)
    actual = WMPDepthCamera(sim.mj_model, cfg, "world/depth").read(data)
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
