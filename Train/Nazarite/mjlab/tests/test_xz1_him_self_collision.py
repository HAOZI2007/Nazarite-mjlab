"""Check real XZ1 internal contacts and bounded control-step penalties."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
from nazarite.config.robot_config.xz1_cfg import get_spec
from nazarite.config.train_config.xz1.him_course import xz1_self_collision_cost
from nazarite.config.train_config.xz1.him_env_cfg import (
  XZ1_HIM_JOINT_NAMES,
  Nazarite_HIM_Complex_Terrain_XZ1,
)
from nazarite.config.train_config.xz1.teacher_env_cfg import (
  Nazarite_WMP_HIM_Teacher_XZ1,
)

from mjlab.scene import Scene
from mjlab.sensor import ContactSensorCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.lab_api.string import resolve_matching_names_values


def test_control_interval_cost_catches_brief_contacts_without_count_amplification():
  history = torch.zeros(4, 2, 10, 3)
  history[0, 0, 0, 0] = 2  # Latest substep.
  history[1, 0, -1, 0] = 2  # Brief impact at the beginning of this interval.
  history[2, 0, :, 0] = 2
  history[2, 1, :, 0] = -2  # Opposite contact forces must not cancel.
  history[3, :, :, 0] = 0.5  # Below threshold.
  env = SimpleNamespace(
    scene={
      "self_collision": SimpleNamespace(data=SimpleNamespace(force_history=history))
    }
  )
  torch.testing.assert_close(
    xz1_self_collision_cost(env, "self_collision"),
    torch.tensor([1.0, 1.0, 1.0, 0.0]),
  )
  # After ten collision-free substeps, the next interval must incur no cost.
  history.zero_()
  assert torch.equal(xz1_self_collision_cost(env, "self_collision"), torch.zeros(4))


def test_self_collision_is_local_to_base_him_and_enabled_in_play():
  for play in (False, True):
    cfg = Nazarite_HIM_Complex_Terrain_XZ1(play=play)
    term = cfg.rewards["self_collision"]
    sensor = next(s for s in cfg.scene.sensors if s.name == term.params["sensor_name"])
    assert isinstance(sensor, ContactSensorCfg)
    assert sensor.history_length == cfg.decimation
    assert sensor.reduce == "maxforce"
    assert term.weight == -1.0
    assert term.params["force_threshold"] == 1.0
    assert "self_collision" not in cfg.terminations
  teacher = Nazarite_WMP_HIM_Teacher_XZ1()
  assert "self_collision" not in teacher.rewards
  assert "self_collision" not in {s.name for s in teacher.scene.sensors}


def _physics_only_spec():
  spec = get_spec()
  for geom in list(spec.geoms):
    if geom.contype == 0 and geom.conaffinity == 0:
      spec.delete(geom)
  for mesh in list(spec.meshes):
    spec.delete(mesh)
  return spec


@pytest.fixture(scope="module")
def model():
  cfg = Nazarite_HIM_Complex_Terrain_XZ1()
  cfg.scene.num_envs = 1
  cfg.scene.terrain = TerrainEntityCfg(terrain_type="plane")
  cfg.scene.entities["robot"].spec_fn = _physics_only_spec
  cfg.scene.sensors = tuple(s for s in cfg.scene.sensors if s.name == "self_collision")
  return Scene(cfg.scene, "cpu").compile()


def test_ground_contacts_do_not_count_as_self_collision(model):
  cfg = Nazarite_HIM_Complex_Terrain_XZ1()
  init = cfg.scene.entities["robot"].init_state.joint_pos
  assert init is not None
  _, _, values = resolve_matching_names_values(init, XZ1_HIM_JOINT_NAMES)
  data = mujoco.MjData(model)
  data.qpos[:3] = [0, 0, 0.28]  # Compress the feet into the plane.
  for name, value in zip(XZ1_HIM_JOINT_NAMES, values, strict=True):
    data.qpos[model.joint(f"robot/{name}").qposadr] = value
  mujoco.mj_forward(model, data)
  assert data.ncon >= 4
  assert data.sensor("self_collision_base_link_found").data[0] == 0
  assert np.linalg.norm(data.sensor("self_collision_base_link_force").data) == 0


@pytest.mark.parametrize(
  "pose,expected_pair",
  [
    (
      [
        0.54846,
        -1.03419,
        -2.09456,
        -1.03563,
        1.45078,
        -1.28308,
        0.06047,
        1.69951,
        -1.64250,
        0.36978,
        -0.98522,
        -2.24357,
      ],
      {"robot/base_link_primitive_1", "robot/FR_thigh_link_primitive_1"},
    ),
    (
      [
        0.39390,
        0.92444,
        -2.29608,
        -0.39405,
        -1.41101,
        -1.58128,
        -0.52905,
        -0.96541,
        -2.26299,
        0.39947,
        -0.85622,
        -0.82882,
      ],
      {"robot/RL_shank_link_primitive_2", "robot/FL_thigh_link_primitive_1"},
    ),
  ],
  ids=["leg_body", "leg_leg"],
)
def test_actual_internal_contact_pairs_trigger_sensor(model, pose, expected_pair):
  data = mujoco.MjData(model)
  data.qpos[:3] = [0, 0, 1]  # Keep the robot clear of external geometry.
  for name, value in zip(XZ1_HIM_JOINT_NAMES, pose, strict=True):
    data.qpos[model.joint(f"robot/{name}").qposadr] = value
  mujoco.mj_forward(model, data)
  pairs = [
    {model.geom(int(geom_id)).name for geom_id in contact.geom}
    for contact in data.contact
  ]
  assert expected_pair in pairs
  assert data.sensor("self_collision_base_link_found").data[0] > 0
  assert np.linalg.norm(data.sensor("self_collision_base_link_force").data) > 1
