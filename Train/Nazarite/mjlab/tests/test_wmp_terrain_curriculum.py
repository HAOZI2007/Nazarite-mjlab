from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
from nazarite.config.train_config.env_cfgs.him_complex_env_cfg import (
  HIM_COMPLEX_TERRAINS_CFG,
)
from nazarite.config.train_config.env_cfgs.wmp_him_go2_env_cfg import (
  WMP_FEET_EDGE_TERRAINS,
  Nazarite_WMP_HIM_Complex_Terrain_Go2,
)
from nazarite.config.train_config.env_cfgs.wmp_terrain import (
  WMP_FLAT_TERRAINS_CFG,
  WMP_FORWARD_TERRAIN_NAMES,
  WMP_NUM_ROWS,
  WMP_OMNI_TERRAIN_NAMES,
  WMP_TERRAIN_NAMES,
  WMP_TERRAINS_CFG,
  WMP_TRACK_VOID,
  _difficulty,
)
from nazarite.mdp.commands import TerrainConditionedVelocityCommand
from nazarite.mdp.wmp_curriculums import (
  forward_velocity_wmp_reference,
  terrain_levels_wmp_reference,
)

from mjlab.terrains.terrain_generator import TerrainGenerator

_NUM_COLUMNS = len(WMP_TERRAIN_NAMES) + len(WMP_OMNI_TERRAIN_NAMES)


def test_wmp_task_uses_its_own_ten_level_competition_curriculum():
  cfg = Nazarite_WMP_HIM_Complex_Terrain_Go2()
  terrain = cfg.scene.terrain
  assert terrain is not None and terrain.terrain_generator is not None
  generator = terrain.terrain_generator

  assert generator.curriculum
  assert generator.num_rows == WMP_NUM_ROWS
  assert generator.size == (8.0, 8.0)
  assert generator.border_width == 25.0
  assert generator.difficulty_range == (0.0, 0.9)
  assert (
    tuple(generator.sub_terrains)
    == (*WMP_TERRAIN_NAMES, *WMP_OMNI_TERRAIN_NAMES)
    == (
      "stairs",
      "low_frame",
      "high_wall",
      "plum_blossom",
      "gap",
      "tire_array",
      "single_beam",
      "ramp",
      "omni_flat",
    )
  )
  assert terrain.max_init_terrain_level == 0
  assert cfg.curriculum["terrain_levels"].func is terrain_levels_wmp_reference
  assert cfg.curriculum["forward_velocity"].func is forward_velocity_wmp_reference
  assert cfg.commands["twist"].forward_only_terrain_names == WMP_FORWARD_TERRAIN_NAMES
  assert cfg.commands["twist"].rel_forward_envs == 0.0
  # The pad is the only omni-directional slice of the obstacle course, so it may
  # never appear in the forward-only mask.
  assert "omni_flat" not in cfg.commands["twist"].forward_only_terrain_names
  assert (
    cfg.curriculum["terrain_levels"].params["omni_terrain_names"]
    == WMP_OMNI_TERRAIN_NAMES
  )
  assert cfg.commands["twist"].forward_only_x_min == 0.15
  assert cfg.commands["twist"].forward_only_x_max == 0.5
  assert cfg.commands["twist"].forward_only_zero_below == 0.0
  assert (
    cfg.events["reset_base"].func.__name__ == "reset_root_state_uniform_facing_course"
  )
  assert cfg.events["reset_base"].params["velocity_range"] == {"x": (0.1, 0.35)}
  assert cfg.rewards["feet_edge"].params["terrain_names"] == WMP_FEET_EDGE_TERRAINS

  # Building WMP must not mutate the shared HIM terrain template.
  assert tuple(HIM_COMPLEX_TERRAINS_CFG.sub_terrains) == (
    "stairs_up",
    "stairs_down",
    "tire_array",
    "high_wall",
  )


def test_wmp_play_uses_reference_ordered_grid_without_runtime_curriculum():
  cfg = Nazarite_WMP_HIM_Complex_Terrain_Go2(play=True)
  terrain = cfg.scene.terrain
  assert terrain is not None and terrain.terrain_generator is not None
  generator = terrain.terrain_generator
  assert generator.curriculum
  assert generator.size == (8.0, 8.0)
  assert generator.num_rows == WMP_NUM_ROWS
  assert generator.difficulty_range == (0.0, 0.9)
  assert len(generator.sub_terrains) == len(WMP_TERRAIN_NAMES) + len(
    WMP_OMNI_TERRAIN_NAMES
  )
  assert "terrain_levels" not in cfg.curriculum
  assert "forward_velocity" not in cfg.curriculum


def test_wmp_geometry_difficulty_normalizes_reference_row_values():
  values = [_difficulty(row / 10) for row in range(10)]
  np.testing.assert_allclose(values, np.linspace(0.0, 1.0, 10))


def test_reset_jitter_stays_inside_flat_start_platforms():
  platform_ends = {
    "stairs": 1.55,
    "gap": 2.60,
    "plum_blossom": 1.65,
    "single_beam": 2.0,
    "ramp": 1.65,
  }
  for terrain_name, platform_end in platform_ends.items():
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    output = WMP_TERRAINS_CFG.sub_terrains[terrain_name].function(
      0.0, spec, np.random.default_rng(0)
    )
    reset_x_min = output.origin[0] - 0.5
    reset_x_max = output.origin[0] + 0.5
    assert reset_x_min >= 0.0
    assert reset_x_max <= platform_end


@pytest.mark.parametrize("terrain_name", WMP_TERRAIN_NAMES)
@pytest.mark.parametrize("difficulty", (0.0, 1.0))
def test_each_wmp_course_terrain_compiles_at_curriculum_endpoints(
  terrain_name: str, difficulty: float
):
  generator_cfg = deepcopy(WMP_TERRAINS_CFG)
  subterrain = generator_cfg.sub_terrains[terrain_name]
  subterrain.size = generator_cfg.size
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")

  output = subterrain.function(difficulty, spec, np.random.default_rng(0))
  assert np.allclose(output.origin, (1.0, generator_cfg.size[1] / 2, 0.0))
  if terrain_name in ("high_wall", "low_frame"):
    # Repeated cross-tile obstacles prevent lateral bypass without adding
    # continuous side walls or converting the terrain into a corridor.
    assert len(output.geometries) >= 4
  if terrain_name in ("gap", "plum_blossom"):
    assert len(output.geometries) >= 8
  if terrain_name == "single_beam":
    bridge_geometries = output.geometries[3:]
    assert len(bridge_geometries) == 1
    bridge = bridge_geometries[0].geom
    assert bridge is not None
    assert bridge.type == mujoco.mjtGeom.mjGEOM_BOX
    assert bridge.pos[1] == pytest.approx(generator_cfg.size[1] / 2)
    assert bridge.size[0] == pytest.approx((6.2 - 2.0) / 2)
  if terrain_name == "tire_array":
    tire = WMP_TERRAINS_CFG.sub_terrains["tire_array"]
    assert (
      1 + 12 * tire.ring_segments
      <= len(output.geometries)
      <= 1 + 20 * tire.ring_segments
    )
    assert all(
      geometry.geom is not None and geometry.geom.type == mujoco.mjtGeom.mjGEOM_BOX
      for geometry in output.geometries[1:]
    )
  if terrain_name == "ramp":
    assert len(output.geometries) == 2
    ramp = output.geometries[1].geom
    assert ramp is not None and ramp.type == mujoco.mjtGeom.mjGEOM_MESH
  model = spec.compile()
  assert model.ngeom >= 2


@pytest.mark.parametrize("difficulty", (0.0, 1.0))
def test_plum_blossom_stones_are_flush_with_start_platform(difficulty: float):
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  output = WMP_TERRAINS_CFG.sub_terrains["plum_blossom"].function(
    difficulty, spec, np.random.default_rng(0)
  )

  start_platform = output.geometries[0].geom
  stones = [geometry.geom for geometry in output.geometries[3:]]
  assert start_platform is not None
  assert stones
  platform_top = start_platform.pos[2] + start_platform.size[2]
  assert platform_top == pytest.approx(0.0)
  for stone in stones:
    assert stone is not None
    assert stone.pos[2] + stone.size[2] == pytest.approx(platform_top)


def _geom_y_span(
  model: mujoco.MjModel, geom_index: int, spec_vertices
) -> tuple[float, float]:
  """The y extent a single compiled geom occupies inside its tile."""
  geom_type = model.geom_type[geom_index]
  position = model.geom_pos[geom_index]
  size = model.geom_size[geom_index]
  if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
    return float(position[1] - size[1]), float(position[1] + size[1])
  if geom_type in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_SPHERE):
    return float(position[1] - size[0]), float(position[1] + size[0])
  assert geom_type == mujoco.mjtGeom.mjGEOM_MESH
  # Mesh vertices are stored around the geom origin, so the extent has to be
  # rebuilt from both the vertices and ``geom_pos``.
  return (
    float(spec_vertices[:, 1].min() + position[1]),
    float(spec_vertices[:, 1].max() + position[1]),
  )


@pytest.mark.parametrize("terrain_name", WMP_TERRAIN_NAMES)
@pytest.mark.parametrize("difficulty", (0.0, 1.0))
def test_no_terrain_geometry_reaches_into_the_track_void(
  terrain_name: str, difficulty: float
):
  """Neighbouring tracks are 0.80 m apart, so the strip has to stay empty.

  A deck that extends into the strip gives the policy a bypass around every
  obstacle; a beam or a ramp that does is the same bypass at knee height.
  """
  generator_cfg = deepcopy(WMP_TERRAINS_CFG)
  subterrain = generator_cfg.sub_terrains[terrain_name]
  subterrain.size = generator_cfg.size
  low = WMP_TRACK_VOID
  high = generator_cfg.size[1] - WMP_TRACK_VOID
  assert high - low == pytest.approx(7.20)

  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  output = subterrain.function(difficulty, spec, np.random.default_rng(0))
  model = spec.compile()
  assert model.ngeom == len(output.geometries)

  for index in range(len(output.geometries)):
    vertices = np.zeros((0, 3))
    if model.geom_type[index] == mujoco.mjtGeom.mjGEOM_MESH:
      mesh_id = int(model.geom_dataid[index])
      start = int(model.mesh_vertadr[mesh_id])
      count = int(model.mesh_vertnum[mesh_id])
      vertices = model.mesh_vert[start : start + count]
    span = _geom_y_span(model, index, vertices)
    assert span[0] >= low - 1e-6, (terrain_name, difficulty, index, span)
    assert span[1] <= high + 1e-6, (terrain_name, difficulty, index, span)


def test_the_practice_pad_keeps_its_full_tile_width():
  """The pad has no obstacle to bypass, and omni commands drive the robot sideways."""
  generator_cfg = deepcopy(WMP_FLAT_TERRAINS_CFG)
  subterrain = generator_cfg.sub_terrains["flat"]
  subterrain.size = generator_cfg.size
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  subterrain.function(0.0, spec, np.random.default_rng(0))

  model = spec.compile()
  assert model.ngeom == 1
  deck = model.geom_pos[0]
  size = model.geom_size[0]
  assert deck[1] - size[1] == pytest.approx(0.0)
  assert deck[1] + size[1] == pytest.approx(generator_cfg.size[1])


def test_the_void_keeps_nine_terrain_columns():
  generator_cfg = deepcopy(WMP_TERRAINS_CFG)
  assert len(generator_cfg.sub_terrains) == 9
  assert generator_cfg.size == (8.0, 8.0)
  assert len(WMP_TERRAINS_CFG.sub_terrains) == _NUM_COLUMNS


def test_full_curriculum_generator_compiles_all_patches():
  cfg = deepcopy(WMP_TERRAINS_CFG)
  generator = TerrainGenerator(cfg, device="cpu")
  spec = mujoco.MjSpec()
  generator.compile(spec)
  model = spec.compile()
  assert model.ngeom > 900
  assert generator.terrain_origins.shape == (
    WMP_NUM_ROWS,
    len(WMP_TERRAIN_NAMES) + len(WMP_OMNI_TERRAIN_NAMES),
    3,
  )


def test_reference_course_curriculum_promotes_and_regresses_by_distance():
  terrain_generator = deepcopy(WMP_TERRAINS_CFG)
  terrain = SimpleNamespace(
    cfg=SimpleNamespace(terrain_generator=terrain_generator),
    terrain_levels=torch.tensor([1, 4, 2, 0, 2]),
    terrain_types=torch.tensor([0, 1, 2, 3, 4]),
    terrain_origins=torch.zeros(10, _NUM_COLUMNS, 3),
    updates=None,
  )

  def update_origins(env_ids, move_up, move_down):
    terrain.updates = (env_ids.clone(), move_up.clone(), move_down.clone())

  terrain.update_env_origins = update_origins
  positions = torch.tensor(
    [
      [6.3, 0.0, 0.0],
      [0.5, 0.0, 0.0],
      [3.0, 0.0, 0.0],
      [6.1, 0.0, 0.0],
      [3.8, 0.0, 0.0],
    ]
  )

  class Scene(SimpleNamespace):
    def __getitem__(self, name):
      return getattr(self, name)

  env = SimpleNamespace(
    scene=Scene(
      terrain=terrain,
      env_origins=torch.zeros(5, 3),
      robot=SimpleNamespace(data=SimpleNamespace(root_link_pos_w=positions)),
    ),
    command_manager=SimpleNamespace(
      get_command=lambda _name: torch.tensor(
        [
          [1.0, 0.0, 0.0],
          [1.0, 0.0, 0.0],
          [0.1, 0.0, 0.0],
          [0.8, 0.0, 0.0],
          [1.0, 0.0, 0.0],
        ]
      )
    ),
    max_episode_length_s=20.0,
    common_step_counter=1,
    num_envs=5,
    device="cpu",
  )
  ids = torch.tensor([0, 1, 2, 3, 4])

  metrics = terrain_levels_wmp_reference(env, ids, "twist")

  updated_ids, move_up, move_down = terrain.updates
  assert torch.equal(updated_ids, ids)
  assert torch.equal(move_up, torch.tensor([True, False, False, True, False]))
  # Progress is measured along +x through the entire course. The 3.8 m case is
  # still below the finish marker, although the reference distance rule would
  # otherwise count it as a successful tile traversal.
  assert torch.equal(move_down, torch.tensor([False, True, False, False, True]))
  assert set(metrics) >= {"mean", "max", *WMP_TERRAIN_NAMES[:4]}


def test_curriculum_demotes_a_stalled_high_command_episode():
  # The 2026-10-17_22-32-26 run parked its population at 1.5-4.0 m: the demotion
  # threshold was capped at 2.4 m, so an episode ordered to run at 1.0 m/s that
  # died after 3.5 m neither promoted nor demoted, and the level stopped tracking
  # the policy's actual competence. Demotion is now floored, not capped: a slow
  # command still has to reach 3.0 m, a fast one has to reach its full budget.
  terrain = SimpleNamespace(
    cfg=SimpleNamespace(terrain_generator=deepcopy(WMP_TERRAINS_CFG)),
    terrain_levels=torch.tensor([3, 3]),
    terrain_types=torch.tensor([4, 4]),
    terrain_origins=torch.zeros(10, _NUM_COLUMNS, 3),
  )
  terrain.update_env_origins = lambda _ids, move_up, move_down: setattr(
    terrain, "updates", (move_up.clone(), move_down.clone())
  )

  class Scene(SimpleNamespace):
    def __getitem__(self, name):
      return getattr(self, name)

  env = SimpleNamespace(
    scene=Scene(
      terrain=terrain,
      env_origins=torch.zeros(2, 3),
      robot=SimpleNamespace(
        data=SimpleNamespace(
          root_link_pos_w=torch.tensor([[3.5, 0.0, 0.0], [3.5, 0.0, 0.0]])
        )
      ),
    ),
    command_manager=SimpleNamespace(
      get_command=lambda _name: torch.tensor([[1.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
    ),
    max_episode_length_s=20.0,
    common_step_counter=1,
    num_envs=2,
    device="cpu",
  )

  terrain_levels_wmp_reference(env, torch.tensor([0, 1]), "twist")

  move_up, move_down = terrain.updates
  assert torch.equal(move_up, torch.tensor([False, False]))
  assert torch.equal(move_down, torch.tensor([True, False]))


def test_reference_course_curriculum_does_not_promote_for_lateral_bypass():
  terrain = SimpleNamespace(
    cfg=SimpleNamespace(terrain_generator=deepcopy(WMP_TERRAINS_CFG)),
    terrain_levels=torch.tensor([2]),
    terrain_types=torch.tensor([1]),
    terrain_origins=torch.zeros(10, _NUM_COLUMNS, 3),
  )
  terrain.update_env_origins = lambda _ids, move_up, move_down: setattr(
    terrain, "updates", (move_up.clone(), move_down.clone())
  )

  class Scene(SimpleNamespace):
    def __getitem__(self, name):
      return getattr(self, name)

  env = SimpleNamespace(
    scene=Scene(
      terrain=terrain,
      env_origins=torch.zeros(1, 3),
      robot=SimpleNamespace(
        data=SimpleNamespace(root_link_pos_w=torch.tensor([[0.2, 6.5, 0.0]]))
      ),
    ),
    command_manager=SimpleNamespace(
      get_command=lambda _name: torch.tensor([[0.0, 1.0, 0.0]])
    ),
    max_episode_length_s=20.0,
    common_step_counter=1,
    num_envs=1,
    device="cpu",
  )

  terrain_levels_wmp_reference(env, torch.tensor([0]), "twist")

  move_up, _move_down = terrain.updates
  assert torch.equal(move_up, torch.tensor([False]))


def test_reference_velocity_curriculum_accumulates_a_group_then_caps():
  terrain = SimpleNamespace(
    cfg=SimpleNamespace(terrain_generator=deepcopy(WMP_TERRAINS_CFG)),
    terrain_types=torch.tensor([1, 2, 3, 4]),
  )
  command_term = object.__new__(TerrainConditionedVelocityCommand)
  command_term.cfg = SimpleNamespace(forward_only_x_max=None)
  reward_sums = {"track_linear_velocity": torch.full((4,), 19.0)}

  class Scene(SimpleNamespace):
    pass

  env = SimpleNamespace(
    scene=Scene(terrain=terrain),
    device="cpu",
    episode_length_buf=torch.tensor([100, 100, 0, 0]),
    max_episode_length_s=20.0,
    max_episode_length=1000,
    cfg=SimpleNamespace(scale_rewards_by_dt=True),
    command_manager=SimpleNamespace(get_term=lambda _name: command_term),
    reward_manager=SimpleNamespace(
      _episode_sums=reward_sums,
      get_term_cfg=lambda _name: SimpleNamespace(weight=1.0),
    ),
  )
  ids = torch.arange(4)

  first = forward_velocity_wmp_reference(
    env, ids, "twist", "track_linear_velocity", WMP_FORWARD_TERRAIN_NAMES
  )
  assert first == {"x_max": 0.5, "score": 0.0, "episodes": 2.0}

  env.episode_length_buf[:] = 100
  second = forward_velocity_wmp_reference(
    env,
    torch.tensor([2, 3]),
    "twist",
    "track_linear_velocity",
    WMP_FORWARD_TERRAIN_NAMES,
  )
  assert second["score"] == pytest.approx(0.95)
  assert second["x_max"] == pytest.approx(0.7)

  for _ in range(3):
    forward_velocity_wmp_reference(
      env, ids, "twist", "track_linear_velocity", WMP_FORWARD_TERRAIN_NAMES
    )
  assert command_term.cfg.forward_only_x_max == pytest.approx(1.0)


def _gate_env(positions, terrain_types, accumulators):
  terrain_generator = deepcopy(WMP_TERRAINS_CFG)
  terrain = SimpleNamespace(
    cfg=SimpleNamespace(terrain_generator=terrain_generator),
    terrain_levels=torch.ones(len(positions), dtype=torch.long),
    terrain_types=terrain_types,
    terrain_origins=torch.zeros(WMP_NUM_ROWS, _NUM_COLUMNS, 3),
    updates=None,
  )

  def update_origins(env_ids, move_up, move_down):
    terrain.updates = (env_ids.clone(), move_up.clone(), move_down.clone())

  terrain.update_env_origins = update_origins

  class Scene(SimpleNamespace):
    def __getitem__(self, name):
      return getattr(self, name)

  env = SimpleNamespace(
    scene=Scene(
      terrain=terrain,
      env_origins=torch.zeros(len(positions), 3),
      robot=SimpleNamespace(
        data=SimpleNamespace(root_link_pos_w=torch.tensor(positions))
      ),
    ),
    command_manager=SimpleNamespace(
      get_command=lambda _name: torch.tensor([[1.0, 0.0, 0.0]] * len(positions))
    ),
    max_episode_length_s=20.0,
    common_step_counter=1,
    init_done=True,
    num_envs=len(positions),
    device="cpu",
  )
  for key, (value, count) in accumulators.items():
    env.__dict__[key] = {
      "sum": torch.full((len(positions),), float(value)) * float(count),
      "count": torch.full((len(positions),), float(count)),
    }
  return env, terrain


def test_gait_gate_blocks_promotion_for_a_crawling_episode():
  """Distance alone let the 2026-10-18 run crawl up to level 4.9."""
  positions = [[6.3, 0.0, 0.0]] * 4
  ids = torch.arange(4)
  # terrain_types: gap, stairs, ramp, high_wall.
  terrain_types = torch.tensor([4, 0, 7, 2])

  env, terrain = _gate_env(
    positions,
    terrain_types,
    {
      # A dragging hind leg, as measured on the 2026-10-18 crawl.
      "_wmp_hind_swing": (0.13, 50),
      "_wmp_gait_tilt": (0.62, 50),
      "_wmp_gait_clearance": (0.30, 50),
    },
  )
  terrain_levels_wmp_reference(env, ids, "twist")
  assert not terrain.updates[1].any()
  # The gate consumed the accumulators so the next episode starts clean.
  assert (env._wmp_hind_swing["count"] == 0).all()

  env, terrain = _gate_env(
    positions,
    terrain_types,
    {
      # Half a cycle in swing, level trunk, normal clearance.
      "_wmp_hind_swing": (0.38, 50),
      "_wmp_gait_tilt": (0.20, 50),
      "_wmp_gait_clearance": (0.30, 50),
    },
  )
  terrain_levels_wmp_reference(env, ids, "twist")
  assert terrain.updates[1].all()


def test_gait_gate_opens_when_a_dragging_leg_is_the_only_problem():
  """Hind swing alone rejects a drag; the tilt and clearance must not double up."""
  positions = [[6.3, 0.0, 0.0]] * 2
  env, terrain = _gate_env(
    positions,
    torch.tensor([4, 0]),
    {
      "_wmp_hind_swing": (0.10, 50),
      "_wmp_gait_tilt": (0.10, 50),
      "_wmp_gait_clearance": (0.32, 50),
    },
  )
  terrain_levels_wmp_reference(env, torch.arange(2), "twist")
  assert not terrain.updates[1].any()


def test_the_diagonal_match_floor_is_off_in_the_shipped_config():
  """The 0.62 floor froze the 2026-10-20 run: back it off unless a phase term is.

  The match is measured against a 0.6 s phase clock, and the reference reward
  set leaves the style to the discriminator without pinning a period. The run
  measured 0.44 for 4500 iterations against the old floor and never promoted,
  while its episodes were already walking seven metres.
  """
  cfg = Nazarite_WMP_HIM_Complex_Terrain_Go2(flat=False)
  term = cfg.curriculum["terrain_levels"]
  assert term.params["min_gait_match"] == 0.0
  assert term.params["min_hind_swing"] == 0.25


def test_gait_gate_waives_base_clearance_on_the_low_ceiling_tile():
  positions = [[6.3, 0.0, 0.0]] * 4
  ids = torch.arange(4)
  # terrain_types: low_frame, stairs, ramp, high_wall.
  terrain_types = torch.tensor([1, 0, 7, 2])

  env, terrain = _gate_env(
    positions,
    terrain_types,
    {
      "_wmp_gait_match": (0.80, 50),
      "_wmp_gait_tilt": (0.20, 50),
      # Crouched to 0.15 m: only legal under the low frame.
      "_wmp_gait_clearance": (0.15, 50),
    },
  )
  terrain_levels_wmp_reference(env, ids, "twist")
  assert terrain.updates[1].tolist() == [True, False, False, False]


def test_gait_gate_leaves_the_curriculum_open_without_metrics():
  """A missing accumulator must not freeze the curriculum silently."""
  env, terrain = _gate_env([[6.3, 0.0, 0.0]] * 2, torch.tensor([4, 0]), {})
  terrain_levels_wmp_reference(env, torch.arange(2), "twist")
  assert terrain.updates[1].all()


def test_flat_stage_is_a_single_deck_without_a_terrain_curriculum():
  cfg = Nazarite_WMP_HIM_Complex_Terrain_Go2(flat=True)
  terrain = cfg.scene.terrain
  assert terrain is not None and terrain.terrain_generator is not None
  generator = terrain.terrain_generator
  assert tuple(generator.sub_terrains) == ("flat",)
  assert not generator.curriculum
  assert (generator.num_rows, generator.num_cols) == (1, 1)
  assert "terrain_levels" not in cfg.curriculum
  assert "forward_velocity" in cfg.curriculum
  # The forward-only command mask has to name the deck it is sampling on.
  assert cfg.commands["twist"].forward_only_terrain_names == ()
  assert cfg.commands["twist"].rel_standing_envs == 0.15
  # The curriculum validates its own names and raised on 2026-10-18 17:03 when
  # the obstacle list was forwarded to a deck that only has "flat".
  assert cfg.curriculum["forward_velocity"].params["terrain_names"] == ("flat",)
  # No terrain rows to gate on, so the stall floor applies from the start.
  assert cfg.terminations["stall_timeout"].params["min_terrain_level"] == 0
  # Terrain-masked terms degrade to "no mask" on a deck with no obstacles.
  assert cfg.rewards["feet_edge"].params["terrain_names"] == WMP_FEET_EDGE_TERRAINS


def test_flat_deck_geometry_builds():
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  sub = WMP_FLAT_TERRAINS_CFG.sub_terrains["flat"]
  sub.size = WMP_FLAT_TERRAINS_CFG.size
  output = sub.function(0.0, spec, np.random.default_rng(0))
  assert np.allclose(np.asarray(output.origin), (1.0, 4.0, 0.0))
  assert len(output.geometries) == 1


def test_flat_stage_forward_velocity_curriculum_runs_on_the_deck():
  cfg = Nazarite_WMP_HIM_Complex_Terrain_Go2(flat=True)
  params = cfg.curriculum["forward_velocity"].params
  terrain = SimpleNamespace(
    cfg=SimpleNamespace(terrain_generator=cfg.scene.terrain.terrain_generator),
    terrain_types=torch.zeros(4, dtype=torch.long),
  )
  command_term = TerrainConditionedVelocityCommand.__new__(
    TerrainConditionedVelocityCommand
  )
  command_term.cfg = SimpleNamespace(forward_only_x_max=0.5, forward_only_x_min=0.15)
  env = SimpleNamespace(
    scene=SimpleNamespace(terrain=terrain),
    command_manager=SimpleNamespace(get_term=lambda _name: command_term),
    device="cpu",
    episode_length_buf=torch.zeros(4, dtype=torch.long),
    max_episode_length_s=20.0,
    max_episode_length=1000,
    cfg=SimpleNamespace(scale_rewards_by_dt=True),
    reward_manager=SimpleNamespace(_episode_sums={}, get_term_cfg=None),
  )
  metrics = forward_velocity_wmp_reference(env, torch.arange(4), **params)
  assert metrics == {"x_max": 0.5, "score": 0.0, "episodes": 0.0}


def test_omni_pad_is_practice_ground_outside_the_curriculum():
  # The flat pad exists so the obstacle stage keeps some omni-directional and
  # stand-still experience. It must never advance, never demote, and never drag
  # the reported mean level down: it is the same deck at every row.
  pad_column = WMP_TERRAIN_NAMES.index("plum_blossom")
  pad_type = len(WMP_TERRAIN_NAMES)  # omni_flat is the last column
  terrain = SimpleNamespace(
    cfg=SimpleNamespace(terrain_generator=deepcopy(WMP_TERRAINS_CFG)),
    terrain_levels=torch.tensor([0, 4]),
    terrain_types=torch.tensor([pad_type, pad_column]),
    terrain_origins=torch.zeros(10, _NUM_COLUMNS, 3),
  )
  terrain.update_env_origins = lambda _ids, move_up, move_down: setattr(
    terrain, "updates", (move_up.clone(), move_down.clone())
  )

  class Scene(SimpleNamespace):
    def __getitem__(self, name):
      return getattr(self, name)

  # Both robots crossed the whole tile; only the course robot may level up.
  positions = torch.tensor([[6.5, 0.0, 0.0], [6.5, 0.0, 0.0]])
  env = SimpleNamespace(
    scene=Scene(
      terrain=terrain,
      env_origins=torch.zeros(2, 3),
      robot=SimpleNamespace(data=SimpleNamespace(root_link_pos_w=positions)),
    ),
    command_manager=SimpleNamespace(
      get_command=lambda _name: torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    ),
    max_episode_length_s=20.0,
    common_step_counter=1,
    num_envs=2,
    device="cpu",
  )

  metrics = terrain_levels_wmp_reference(
    env,
    torch.tensor([0, 1]),
    "twist",
    omni_terrain_names=WMP_OMNI_TERRAIN_NAMES,
    gait_gate=False,
  )

  move_up, move_down = terrain.updates
  assert torch.equal(move_up, torch.tensor([False, True]))
  assert torch.equal(move_down, torch.tensor([False, False]))
  # mean is taken over the course column only, so the pad's level 0 is invisible.
  assert metrics["mean"].item() == pytest.approx(4.0)
