"""Episode outcomes and reset-before-overwrite evidence for XZ1 WMP courses."""

from __future__ import annotations

import math
import warnings

import torch
import warp as wp

from mjlab.managers.recorder_manager import RecorderTerm
from nazarite.config.train_config.XZ1_env_cfgs.him_course import (
    narrow_course_complete,
)
from nazarite.config.train_config.XZ1_env_cfgs.him_terrain import BoxHighWallTerrainCfg


def bad_state_mask(env, env_ids):
    """Check raw state too: terminal observations may not yet have been built."""
    bad = torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)
    flagged = getattr(env, "_him_numerics_bad", None)
    if flagged is not None:
        bad |= flagged[env_ids]
    for name in ("qpos", "qvel", "qacc"):
        value = getattr(env.sim.data, name)[env_ids]
        bad |= ~torch.isfinite(value).flatten(1).all(1)
    return bad


def capture_failure_scenes(env, env_ids, *, phase):
    """Copy raw state and current Warp contact pairs without running physics."""
    terrain = env.scene.terrain
    names = list(terrain.cfg.terrain_generator.sub_terrains)
    robot = env.scene["robot"]
    data = robot.data
    contact = env.sim.wp_data.contact
    # Only nacon entries are valid; allocated slots beyond it can be stale.
    count = min(int(wp.to_torch(env.sim.wp_data.nacon)[0]), contact.worldid.shape[0])
    worlds = wp.to_torch(contact.worldid)[:count]
    pairs = wp.to_torch(contact.geom)
    scenes = {}
    for index in env_ids.tolist():
        selected = (worlds == index).nonzero().flatten()
        pair_ids = pairs[selected].cpu().tolist()
        pair_names = [
            [env.sim.mj_model.geom(g).name if g >= 0 else "flex" for g in pair]
            for pair in pair_ids
        ]
        scene = {
            "env_id": index,
            "common_step_counter": env.common_step_counter,
            "phase": phase,
            "terrain_type": names[int(terrain.terrain_types[index])],
            "terrain_level": int(terrain.terrain_levels[index]),
            "origin": env.scene.env_origins[index].detach().cpu().clone(),
            "episode_length": int(env.episode_length_buf[index]),
            "root_pose_w": data.root_link_pose_w[index].detach().cpu().clone(),
            "root_lin_vel_w": data.root_link_lin_vel_w[index].detach().cpu().clone(),
            "root_ang_vel_w": data.root_link_ang_vel_w[index].detach().cpu().clone(),
            "joint_names": list(robot.joint_names),
            "joint_pos": data.joint_pos[index].detach().cpu().clone(),
            "joint_vel": data.joint_vel[index].detach().cpu().clone(),
            "action": env.action_manager.action[index].detach().cpu().clone(),
            "contacts": {
                "geom_ids": pair_ids,
                "geom_names": pair_names,
                "position_w": wp.to_torch(contact.pos)[selected].cpu().clone(),
                "distance": wp.to_torch(contact.dist)[selected].cpu().clone(),
            },
            "raw_state": {
                name: getattr(env.sim.data, name)[index].detach().cpu().clone()
                for name in ("qpos", "qvel", "qacc")
            },
            "pre_clip": {
                name: value[index].detach().cpu().clone()
                for name, value in getattr(env, "_him_numerics_evidence", {}).items()
            },
        }
        scenes[index] = scene
    return scenes


class XZ1CourseDiagnostics(RecorderTerm):
    """Count completed episodes by their ORIGINAL terrain type and level.

    Columns: ended episodes, successful traversals, time-limit failures,
    numerical failures. Counts are cumulative, never a success EMA. Explicit
    administrative resets do not count as completed episodes.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        terrain = env.scene.terrain
        assert terrain is not None
        generator = terrain.cfg.terrain_generator
        assert generator is not None
        self.names = list(generator.sub_terrains)
        self.counts = torch.zeros(
            len(self.names),
            generator.num_rows,
            4,
            dtype=torch.long,
            device=env.device,
        )
        self.last_recorded_step = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
        self.pending_scenes = {}
        # Optional application-owned extension; upstream env has no XZ1 fields.
        env._xz1_course_diagnostics = self  # pyright: ignore[reportAttributeAccessIssue]

    def record_pre_reset(self, env_ids):
        env = self._env
        ids = env_ids[env.episode_length_buf[env_ids] > 0]
        ids = ids[self.last_recorded_step[ids] != env.common_step_counter]
        if not ids.numel():
            return
        bad = bad_state_mask(env, ids)
        if bad.any():
            try:
                self.pending_scenes.update(
                    capture_failure_scenes(env, ids[bad], phase="before_auto_reset")
                )
            except (
                OSError,
                RuntimeError,
                ValueError,
                IndexError,
                AttributeError,
            ) as exc:
                warnings.warn(f"Pre-reset scene capture failed: {exc}", stacklevel=2)
        success = narrow_course_complete(env)[ids]
        terrain = env.scene.terrain
        assert terrain is not None
        types = terrain.terrain_types[ids]
        generator = terrain.cfg.terrain_generator
        assert generator is not None
        root = env.scene["robot"].data.root_link_pos_w - env.scene.env_origins
        if "high_wall" in self.names:
            wall = generator.sub_terrains["high_wall"]
            assert isinstance(wall, BoxHighWallTerrainCfg)
            far = generator.size[0] * (0.5 - wall.spawn_x_fraction)
            far += wall.wall_thickness / 2 + 0.4
            success |= (types == self.names.index("high_wall")) & (root[ids, 0] >= far)
        if "flat" in self.names:
            success |= (types == self.names.index("flat")) & (
                torch.linalg.vector_norm(root[ids, :2], dim=1) >= 2.4
            )
        gravity = env.scene["robot"].data.projected_gravity_b[ids]
        success &= ~env.reset_terminated[ids] & ~bad
        success &= -gravity[:, 2] >= math.cos(math.radians(45))
        # reset_time_outs also includes successful course completion. Only the
        # real episode time limit counts as a timeout failure here.
        timeout = env.termination_manager.get_term("time_out")[ids] & ~success & ~bad
        self._add(ids, success, timeout, bad)

    def _add(self, ids, success, timeout, bad):
        env = self._env
        terrain = env.scene.terrain
        assert terrain is not None
        levels = terrain.terrain_levels[ids]
        slots = terrain.terrain_types[ids] * self.counts.shape[1] + levels
        values = torch.stack((torch.ones_like(bad), success, timeout, bad), dim=1)
        self.counts.view(-1, 4).index_add_(0, slots, values.long())
        self.last_recorded_step[ids] = env.common_step_counter

    def record_recovery(self, ids):
        """Count abandoned episodes even when no normal termination fired."""
        fresh = ids[self.last_recorded_step[ids] != self._env.common_step_counter]
        if fresh.numel():
            zeros = torch.zeros_like(fresh, dtype=torch.bool)
            self._add(fresh, zeros, zeros, ~zeros)

    def record_post_step(self):
        # The manual-reset collector has no automatic record_pre_reset hook.
        if not self._env.cfg.auto_reset:
            self.record_pre_reset(self._env.reset_buf.nonzero().flatten())

    def scalars(self):
        """Transfer one small count table, not hundreds of CUDA scalar reads."""
        result = {}
        counts = self.counts.cpu()
        for index, name in enumerate(self.names):
            for label, row in [
                ("all", counts[index].sum(0)),
                *[
                    (f"level_{level}", values)
                    for level, values in enumerate(counts[index])
                ],
            ]:
                ended, success, timeout, numerical = row.tolist()
                prefix = f"Terrain/{name}/{label}"
                result[f"{prefix}/episodes"] = ended
                # No sample means unknown, not zero percent. episodes=0 is
                # logged, and rates begin only after an episode ends.
                if ended:
                    result[f"{prefix}/success_rate"] = success / ended
                    result[f"{prefix}/timeout_rate"] = timeout / ended
                    result[f"{prefix}/numerical_failure_rate"] = numerical / ended
        return result
