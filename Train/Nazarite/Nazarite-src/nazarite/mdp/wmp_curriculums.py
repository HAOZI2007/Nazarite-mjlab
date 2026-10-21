"""Terrain progression used by the Go2 WMP competition-skill course."""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from nazarite.mdp.commands import TerrainConditionedVelocityCommand
from nazarite.mdp.wmp_metrics import episode_average

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def terrain_levels_wmp_reference(
    env,
    env_ids: torch.Tensor,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    finish_fraction: float = 0.5,
    min_progress_fraction: float = 0.75,
    command_progress_fraction: float = 0.25,
    gait_gate: bool = True,
    min_gait_match: float = 0.0,
    min_hind_swing: float = 0.25,
    max_gait_tilt: float = 0.35,
    min_base_clearance: float = 0.25,
    crouch_terrain_names: tuple[str, ...] = ("low_frame",),
    omni_terrain_names: tuple[str, ...] = (),
) -> dict[str, torch.Tensor]:
    """Advance difficulty only after reaching the obstacle course finish deck.

    Args:
      finish_fraction: Distance that counts as "crossed the course", as a
        fraction of the terrain tile length. The reference uses half a tile;
        the port required 0.75 of a tile measured from an origin that already
        sits 1 m into the tile, i.e. 87.5 % of the patch, which no policy could
        reach on the pit tiles.
      min_progress_fraction: Lower edge of the hold band, as a fraction of the
        promotion distance. Episodes that stop between this line and the finish
        marker keep their level; below it they lose one. This is a floor on the
        demotion threshold, so a slow command cannot hide a failure: an episode
        that stalls in front of the first obstacle is demoted no matter how low
        the commanded speed was.
      command_progress_fraction: Requested progress is ``|command| * episode
        length`` scaled by this factor. It never drops the threshold below the
        hold band, so the demotion radius is at least
        ``finish_progress * min_progress_fraction``. The 2026-10-17 run capped
        the same expression from above instead, which left a 1.5-4.0 m window in
        which an episode neither promoted nor demoted: the population parked
        there, on terrain it could not cross, and the policy received no
        learning signal from the only terrain it was standing on. Note the
        reference used ``|command| * T * 0.5`` with no bound at all, which at
        the curriculum's ceiling demand of 0.9 m/s asks for 9 m of travel on an
        8 m tile and therefore demoted every episode and froze the levels at
        ~1.9.
      gait_gate: Require a walkable gait, not only distance, before promoting.
      min_gait_match: Lower bound on the diagonal contact match, where 0.5 is
        what random contacts produce. Off by default: the match is measured
        against a 0.6 s phase clock, so it only means anything while the reward
        set carries a phase term of the same period. With the reference reward
        set the style is learned by the discriminator and nothing pins the
        period; the 2026-10-20 complex run then measured 0.44 for 4500
        iterations against the old 0.62 floor and never promoted once.
      min_hind_swing: Lower bound on the fraction of time the hind feet spend in
        swing, which rejects a leg that never leaves the ground without assuming
        any particular gait. A dragging leg sits near 0.13, the 2026-10-20 stage
        one run held 0.38, and half of a cycle is 0.44.
      max_gait_tilt: Upper bound on the total base tilt in radians. 0.35 rad is
        20 deg, above the 15-20 deg a staircase or a 20 deg ramp needs and well
        below the 0.6 rad of the nose-down crawl it is meant to reject.
      min_base_clearance: Lower bound on the base height above the local ground.
      crouch_terrain_names: Terrain families where crouching is mandatory, so
        ``min_base_clearance`` is waived for them.
    """
    terrain = env.scene.terrain
    assert terrain is not None and terrain.cfg.terrain_generator is not None
    generator = terrain.cfg.terrain_generator
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    displacement = (
        asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]
    )
    distance = torch.linalg.vector_norm(displacement, dim=1)
    # Every WMP course has a start deck and a finish deck. Promote only after
    # crossing the whole obstacle field along +x; Euclidean distance allowed a
    # robot to walk sideways around a single central obstacle and level up.
    finish_progress = float(generator.size[0]) * finish_fraction
    move_up = displacement[:, 0] > finish_progress
    expected_distance = (
        torch.linalg.vector_norm(command[env_ids, :2], dim=1)
        * env.max_episode_length_s
        * command_progress_fraction
    )
    demote_distance = torch.clamp(
        expected_distance, min=finish_progress * min_progress_fraction
    )
    move_down = (distance < demote_distance) & ~move_up

    # The flat omni pad is practice ground, not a course: it neither advances nor
    # demotes, and it must not drag the reported mean level down either.
    pad = _terrain_type_mask(terrain, generator, env_ids, omni_terrain_names)
    move_up = move_up & ~pad
    move_down = move_down & ~pad

    # Initial reset positions are not traversal outcomes.
    if not getattr(env, "init_done", env.common_step_counter > 0):
        move_up = torch.zeros_like(move_up)
        move_down = torch.zeros_like(move_down)

    if gait_gate and bool(move_up.any()):
        move_up = move_up & _gait_gate(
            env,
            env_ids,
            terrain,
            generator,
            min_gait_match=min_gait_match,
            min_hind_swing=min_hind_swing,
            max_gait_tilt=max_gait_tilt,
            min_base_clearance=min_base_clearance,
            crouch_terrain_names=crouch_terrain_names,
        )

    terrain.update_env_origins(env_ids, move_up, move_down)
    levels = terrain.terrain_levels.float()
    course_envs = ~torch.isin(
        terrain.terrain_types,
        torch.tensor(
            [
                list(generator.sub_terrains).index(name)
                for name in omni_terrain_names
                if name in generator.sub_terrains
            ],
            dtype=terrain.terrain_types.dtype,
            device=terrain.terrain_types.device,
        ),
    )
    course_levels = levels[course_envs] if course_envs.any() else levels
    metrics = {"mean": course_levels.mean(), "max": course_levels.max()}
    terrain_names = list(generator.sub_terrains)
    origins = terrain.terrain_origins
    if origins is not None and origins.shape[1] == len(terrain_names):
        for terrain_index, name in enumerate(terrain_names):
            type_mask = terrain.terrain_types == terrain_index
            if type_mask.any():
                metrics[name] = levels[type_mask].mean()
        # How far the episodes actually got, per family. Terrain level alone
        # cannot tell "walked up to the obstacle and timed out" from "fell into
        # the pit", and those two failures need opposite fixes: the first is a
        # policy that will not commit, the second is one that commits badly.
        for terrain_index, name in enumerate(terrain_names):
            family = terrain.terrain_types[env_ids] == terrain_index
            if family.any():
                metrics[f"progress/{name}"] = displacement[family, 0].mean()
    return metrics


def _gait_gate(
    env,
    env_ids: torch.Tensor,
    terrain,
    generator,
    min_gait_match: float = 0.0,
    min_hind_swing: float = 0.25,
    max_gait_tilt: float = 0.35,
    min_base_clearance: float = 0.25,
    crouch_terrain_names: tuple[str, ...] = ("low_frame",),
) -> torch.Tensor:
    """Require a walkable gait before the terrain level may increase.

    Without this the curriculum measures distance only, so the 2026-10-18 run
    promoted a nose-down crawl to level 4.9 in 3592 iterations: crossing the
    course was enough, and the crawl crossed seven of the nine terrain families.
    Difficulty then outran ability and the two families that genuinely need a
    dynamic gait stayed at 0.0.

    The three numbers come from ``nazarite.mdp.wmp_metrics``, which accumulates
    per episode. A missing accumulator (metrics not configured) leaves the gate
    open rather than silently freezing the curriculum.
    """
    passed = torch.ones(len(env_ids), dtype=torch.bool, device=env.device)
    if float(min_gait_match) > 0.0:
        match = episode_average(env, "_wmp_gait_match", env_ids)
        if match is not None:
            passed = passed & (match >= float(min_gait_match))
    if float(min_hind_swing) > 0.0:
        hind_swing = episode_average(env, "_wmp_hind_swing", env_ids)
        if hind_swing is not None:
            passed = passed & (hind_swing >= float(min_hind_swing))
    tilt = episode_average(env, "_wmp_gait_tilt", env_ids)
    if tilt is not None:
        passed = passed & (tilt <= float(max_gait_tilt))
    clearance = episode_average(env, "_wmp_gait_clearance", env_ids)
    if clearance is not None:
        # The low-clearance frame tile can only be crossed crouched, so its
        # clearance requirement is waived instead of freezing that column.
        passed = passed & (
            (clearance >= float(min_base_clearance))
            | _terrain_type_mask(terrain, generator, env_ids, crouch_terrain_names)
        )
    return passed


def _terrain_type_mask(
    terrain,
    generator,
    env_ids: torch.Tensor,
    terrain_names: tuple[str, ...],
) -> torch.Tensor:
    """Return a boolean mask selecting the environments on ``terrain_names``."""
    all_names = list(generator.sub_terrains)
    selected = [all_names.index(name) for name in terrain_names if name in all_names]
    device = terrain.terrain_types.device
    if not selected:
        return torch.zeros(len(env_ids), dtype=torch.bool, device=device)
    type_ids = torch.tensor(selected, dtype=terrain.terrain_types.dtype, device=device)
    return torch.isin(terrain.terrain_types[env_ids], type_ids)


def forward_velocity_wmp_reference(
    env,
    env_ids: torch.Tensor,
    command_name: str,
    reward_name: str,
    terrain_names: tuple[str, ...],
    initial_max: float = 0.5,
    increment: float = 0.2,
    maximum: float = 1.0,
    score_threshold: float = 0.8,
) -> dict[str, float]:
    """Increase the forward command limit using the source WMP score rule."""
    command_term = env.command_manager.get_term(command_name)
    if not isinstance(command_term, TerrainConditionedVelocityCommand):
        raise TypeError(
            f"Command {command_name!r} must use TerrainConditionedVelocityCommand"
        )

    terrain = env.scene.terrain
    assert terrain is not None and terrain.cfg.terrain_generator is not None
    all_names = list(terrain.cfg.terrain_generator.sub_terrains)
    missing = set(terrain_names) - set(all_names)
    if missing:
        raise ValueError("Unknown WMP terrain names: " + ", ".join(sorted(missing)))
    type_ids = torch.tensor(
        [all_names.index(name) for name in terrain_names],
        dtype=terrain.terrain_types.dtype,
        device=env.device,
    )

    state = getattr(env, "_wmp_forward_velocity_curriculum", None)
    if state is None:
        group_size = int(torch.isin(terrain.terrain_types, type_ids).sum().item())
        state = {
            "reward_sum": torch.zeros((), device=env.device),
            "episode_count": 0,
            "group_size": max(group_size, 1),
            "x_max": float(initial_max),
            "last_score": 0.0,
        }
        env.__dict__["_wmp_forward_velocity_curriculum"] = state

    command_term.cfg.forward_only_x_max = float(state["x_max"])

    # The environment performs one manager reset during construction. It has no
    # completed trajectory and must not contribute a perfect/empty episode.
    completed = env_ids[env.episode_length_buf[env_ids] > 0]
    completed = completed[torch.isin(terrain.terrain_types[completed], type_ids)]
    if len(completed) > 0:
        episode_sums = env.reward_manager._episode_sums
        if reward_name not in episode_sums:
            raise KeyError(f"Reward term {reward_name!r} is not active")
        # Rebound, not updated in place, for the same inference-mode reason as
        # the reward counters: resets run inside the training loop's mode block.
        state["reward_sum"] = (
            state["reward_sum"] + episode_sums[reward_name][completed].sum()
        )
        state["episode_count"] = state["episode_count"] + len(completed)

    if state["episode_count"] >= state["group_size"]:
        reward_cfg = env.reward_manager.get_term_cfg(reward_name)
        reward_weight = float(reward_cfg.weight)
        if reward_weight <= 0.0:
            raise ValueError(f"Reward term {reward_name!r} must have positive weight")
        episode_horizon = (
            float(env.max_episode_length_s)
            if env.cfg.scale_rewards_by_dt
            else float(env.max_episode_length)
        )
        score = state["reward_sum"] / (
            state["episode_count"] * episode_horizon * reward_weight
        )
        state["last_score"] = float(score.item())
        if state["last_score"] > score_threshold:
            state["x_max"] = min(float(state["x_max"]) + increment, maximum)
            command_term.cfg.forward_only_x_max = float(state["x_max"])
        state["reward_sum"].zero_()
        state["episode_count"] = 0

    return {
        "x_max": float(state["x_max"]),
        "score": float(state["last_score"]),
        "episodes": float(state["episode_count"]),
    }


__all__ = ["forward_velocity_wmp_reference", "terrain_levels_wmp_reference"]
