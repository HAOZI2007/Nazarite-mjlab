from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from nazarite.config.train_config.XZ1_env_cfgs.him_terrain import (
  BoxTireArrayTerrainCfg,
)
from nazarite.mdp import events as custom_events

from mjlab.managers.scene_entity_config import SceneEntityCfg


def test_tire_array_keeps_half_tile_clear_and_fills_the_other_half():
  cfg = BoxTireArrayTerrainCfg(
    size=(8.0, 8.0),
    tire_outer_diameter=0.55,
    tire_inner_diameter=0.33,
    tire_width=0.165,
    tire_pitch=0.56,
    startup_zone_length=4.0,
  )
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  output = cfg.function(0.0, spec, np.random.default_rng(0))
  tires = [geom.geom for geom in output.geometries if geom.geom.type == mujoco.mjtGeom.mjGEOM_MESH]

  assert len(tires) == 7 * 14
  assert np.isclose(output.origin[0], 2.0)
  tire_x = np.array([geom.pos[0] for geom in tires])
  tire_y = np.array([geom.pos[1] for geom in tires])
  assert np.min(tire_x) - cfg.tire_outer_diameter / 2 >= cfg.startup_zone_length
  assert np.max(tire_x) + cfg.tire_outer_diameter / 2 <= cfg.size[0]
  assert np.min(tire_y) - cfg.tire_outer_diameter / 2 >= 0.0
  assert np.max(tire_y) + cfg.tire_outer_diameter / 2 <= cfg.size[1]
  spec.compile()


def test_high_wall_reset_uses_fixed_yaw_only_for_wall_envs(monkeypatch):
  calls = []

  def record_reset(_env, env_ids, pose_range, velocity_range, asset_cfg):
    calls.append((env_ids.clone(), dict(pose_range), velocity_range, asset_cfg))

  monkeypatch.setattr(custom_events, "reset_root_state_uniform", record_reset)
  generator = SimpleNamespace(sub_terrains={"stairs_up": object(), "high_wall": object()})
  terrain = SimpleNamespace(
    cfg=SimpleNamespace(terrain_generator=generator),
    terrain_types=torch.tensor([0, 1, 1]),
  )
  env = SimpleNamespace(
    num_envs=3,
    device=torch.device("cpu"),
    scene=SimpleNamespace(terrain=terrain),
  )
  pose_range = {"x": (-0.5, 0.5), "yaw": (-3.14, 3.14)}
  asset_cfg = SceneEntityCfg("robot")

  custom_events.reset_root_state_uniform_facing_high_wall(
    env, torch.tensor([0, 1, 2]), pose_range, {}, asset_cfg
  )

  assert len(calls) == 2
  assert torch.equal(calls[0][0], torch.tensor([0]))
  assert calls[0][1]["yaw"] == (-3.14, 3.14)
  assert torch.equal(calls[1][0], torch.tensor([1, 2]))
  assert calls[1][1]["yaw"] == (0.0, 0.0)
  assert pose_range["yaw"] == (-3.14, 3.14)
