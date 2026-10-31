"""Actual collision pairs, not edge friction, determine missed gaps."""

from copy import deepcopy
from types import SimpleNamespace

import mujoco
import pytest
import torch
import warp as wp

from mjlab.terrains.terrain_generator import TerrainGenerator
from nazarite.config.robot_config.xz1_cfg import XZ1_FOOT_GEOMS
from nazarite.config.train_config.xz1.him_env_cfg import XZ1_WMP_COMPLEX_TERRAINS_CFG
from nazarite.config.train_config.xz1.him_obstacles import GAP_BOTTOM_PREFIX
from nazarite.mdp.xz1.contacts import GapBottomContact


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_bottom_pairs_ignore_walls_other_bodies_separation_and_stale_slots(device):
    if device != "cpu" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cfg = deepcopy(XZ1_WMP_COMPLEX_TERRAINS_CFG)
    spec = mujoco.MjSpec()
    TerrainGenerator(cfg).compile(spec)
    for name in XZ1_FOOT_GEOMS:
        spec.worldbody.add_geom(
            name=f"robot/{name}", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.024, 0, 0]
        )
    model = spec.compile()
    bottoms = [
        i
        for i in range(model.ngeom)
        if model.geom(i).name.startswith(GAP_BOTTOM_PREFIX)
    ]
    assert len(bottoms) == 10 and len(set(bottoms)) == 10
    foot = model.geom(f"robot/{XZ1_FOOT_GEOMS[0]}").id
    wall = next(
        i
        for i in range(model.ngeom)
        if i not in bottoms and not model.geom(i).name.startswith("robot/")
    )
    # 0: wall friction, 1 and 2: real bottoms in different curriculum rows,
    # 3: trunk, 4: separated speculative contact, then invalid/stale contacts.
    pairs = [
        [foot, wall],
        [foot, bottoms[1]],
        [bottoms[9], foot],
        [wall, bottoms[1]],
        [foot, bottoms[1]],
        [foot, bottoms[1]],
        [-1, bottoms[1]],
        [foot, bottoms[1]],
        [foot, bottoms[1]],
    ]
    data = SimpleNamespace(
        nacon=wp.array([8], dtype=int, device=device),
        contact=SimpleNamespace(
            worldid=wp.array([0, 1, 2, 3, 4, -1, 5, 99, 0], dtype=int, device=device),
            geom=wp.array(pairs, dtype=wp.vec2i, device=device),
            dist=wp.array(
                [0, 0, -0.001, 0, 0.002, 0, 0, 0, 0], dtype=float, device=device
            ),
        ),
    )
    check = GapBottomContact(model, 6, device)
    assert check(data).tolist() == [False, True, True, False, False, False]
    data.nacon.assign([0])
    assert not check(data).any()  # No result from the previous step persists.


def test_real_sphere_collision_with_gap_sidewall_and_bottom():
    cfg = deepcopy(XZ1_WMP_COMPLEX_TERRAINS_CFG.sub_terrains["gap"])
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    cfg.function(1 / 9, spec, None)
    foot_body = spec.worldbody.add_body(name="foot")
    foot_body.add_freejoint()
    foot_body.add_geom(
        name=f"robot/{XZ1_FOOT_GEOMS[0]}",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.024, 0, 0],
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    check = GapBottomContact(model, 1, "cpu")
    gap = cfg.gap_width_range[0] + (cfg.gap_width_range[1] - cfg.gap_width_range[0]) / 9
    depth = (
        cfg.pit_depth_range[0] + (cfg.pit_depth_range[1] - cfg.pit_depth_range[0]) / 9
    )
    for x, z, expected in [
        (cfg.start_x + 0.012, -0.04, False),
        (cfg.start_x + gap / 2, -depth + 0.023, True),
    ]:
        data.qpos[:3] = [x, cfg.size[1] / 2, z]
        mujoco.mj_forward(model, data)
        assert data.ncon > 0
        contacts = SimpleNamespace(
            nacon=wp.array([data.ncon], dtype=int, device="cpu"),
            contact=SimpleNamespace(
                worldid=wp.zeros(data.ncon, dtype=int, device="cpu"),
                geom=wp.array(data.contact.geom.copy(), dtype=wp.vec2i, device="cpu"),
                dist=wp.array(data.contact.dist.copy(), dtype=float, device="cpu"),
            ),
        )
        assert bool(check(contacts)[0]) == expected
