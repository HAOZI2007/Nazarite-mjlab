"""Nazarite curriculum hooks."""

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


def terrain_levels_vel_strict(
    env,
    env_ids: torch.Tensor,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
):
    """Advance terrain levels without impossible distance requirements.

    The native curriculum compares walked distance against
    ``|command| * episode_length * 0.5``.  On Nazarite's 8 m tiles that value
    can exceed the physically available half-tile distance, so a successful
    traversal is incorrectly downgraded.  The target is capped to a safe
    fraction of the tile length while preserving command-dependent early-fall
    detection.
    """
    asset: Entity = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    assert terrain is not None and terrain.cfg.terrain_generator is not None
    generator = terrain.cfg.terrain_generator
    command = env.command_manager.get_command(command_name)
    assert command is not None

    distance = torch.linalg.vector_norm(
        asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
        dim=1,
    )
    tile_half = float(generator.size[0]) * 0.5
    move_up = distance > tile_half
    command_distance = torch.linalg.vector_norm(command[env_ids, :2], dim=1)
    target_distance = command_distance * env.max_episode_length_s * 0.5
    # A failed traversal should only downgrade after covering less than 60% of
    # the available half-tile, never require more distance than the tile offers.
    target_distance = torch.minimum(
        target_distance,
        torch.full_like(target_distance, tile_half * 0.6),
    )
    move_down = (distance < target_distance) & ~move_up
    if env.common_step_counter == 0:
        move_up = torch.zeros_like(move_up)
        move_down = torch.zeros_like(move_down)

    terrain.update_env_origins(env_ids, move_up, move_down)
    levels = terrain.terrain_levels.float()
    result = {"mean": torch.mean(levels), "max": torch.max(levels)}
    names = list(generator.sub_terrains.keys())
    origins = terrain.terrain_origins
    if origins is not None and origins.shape[1] == len(names):
        for i, name in enumerate(names):
            mask = terrain.terrain_types == i
            if mask.any():
                result[name] = torch.mean(levels[mask])
    return result


def terrain_levels_delta(
    env,
    env_ids: torch.Tensor,
    command_name: str,
    success_distance_scale: float = 0.8,
    min_success_distance: float = 0.8,
    max_success_distance: float = 3.2,
    success_streak_length: int = 2,
    failure_streak_length: int = 3,
    sparse_terrain_names: tuple[str, ...] = (),
    sparse_success_distance_scale: float = 0.5,
    sparse_min_success_distance: float = 0.5,
    sparse_max_success_distance: float = 1.2,
    sparse_success_streak_length: int = 3,
    sparse_failure_streak_length: int = 4,
    sparse_max_level: int | None = None,
    adaptive_type_sampling: bool = False,
    adaptive_sampling_floor: float = 0.25,
    adaptive_sampling_temperature: float = 1.0,
    adaptive_warmup_episodes: int = 20,
    type_success_ema_alpha: float = 0.05,
):
    """DELTA curriculum with a gentle, type-aware sparse-terrain branch.

    This hook runs immediately before an automatic reset.  The counters are kept
    per environment, while the terrain row is changed directly. Continuous
    terrains use the normal two-success/three-failure rule. Sparse terrains use
    a shorter early success distance and a conservative level cap until the
    residual policy demonstrates repeatable traversals.
    """
    if success_streak_length < 1 or failure_streak_length < 1:
        raise ValueError("Curriculum streak lengths must be positive.")
    if sparse_success_streak_length < 1 or sparse_failure_streak_length < 1:
        raise ValueError("Sparse curriculum streak lengths must be positive.")
    terrain = env.scene.terrain
    assert terrain is not None and terrain.terrain_origins is not None
    generator = terrain.cfg.terrain_generator
    assert generator is not None
    command = env.command_manager.get_command(command_name)
    assert command is not None
    terrain_names = list(generator.sub_terrains.keys())
    num_terrain_types = len(terrain_names)

    success_streak = getattr(env, "_delta_success_streak", None)
    failure_streak = getattr(env, "_delta_failure_streak", None)
    if success_streak is None or success_streak.shape[0] != env.num_envs:
        success_streak = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        failure_streak = torch.zeros_like(success_streak)
        env._delta_success_streak = success_streak
        env._delta_failure_streak = failure_streak
    assert isinstance(success_streak, torch.Tensor)
    assert isinstance(failure_streak, torch.Tensor)

    if env.common_step_counter == 0:
        success_streak[env_ids] = 0
        failure_streak[env_ids] = 0
        levels = terrain.terrain_levels.float()
        return {"mean": levels.mean(), "max": levels.max()}

    asset: Entity = env.scene["robot"]
    distance = torch.linalg.vector_norm(
        asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
        dim=1,
    )
    command_speed = torch.linalg.vector_norm(command[env_ids, :2], dim=1)
    sparse_indices = {
        terrain_names.index(name)
        for name in sparse_terrain_names
        if name in terrain_names
    }
    sparse = torch.zeros_like(terrain.terrain_types[env_ids], dtype=torch.bool)
    for terrain_index in sparse_indices:
        sparse |= terrain.terrain_types[env_ids] == terrain_index
    normal_scale = command_speed.new_full(
        command_speed.shape, float(success_distance_scale)
    )
    normal_min = command_speed.new_full(
        command_speed.shape, float(min_success_distance)
    )
    normal_max = command_speed.new_full(
        command_speed.shape, float(max_success_distance)
    )
    sparse_scale = command_speed.new_full(
        command_speed.shape, float(sparse_success_distance_scale)
    )
    sparse_min = command_speed.new_full(
        command_speed.shape, float(sparse_min_success_distance)
    )
    sparse_max = command_speed.new_full(
        command_speed.shape, float(sparse_max_success_distance)
    )
    scale = torch.where(sparse, sparse_scale, normal_scale)
    target_min = torch.where(sparse, sparse_min, normal_min)
    target_max = torch.where(sparse, sparse_max, normal_max)
    target = (command_speed * env.max_episode_length_s * scale).clamp(
        min=target_min, max=target_max
    )
    active = command_speed > 0.1
    terminated = env.reset_terminated[env_ids]
    success = active & ~terminated & (distance >= target)
    failure = active & ~success
    old_types = terrain.terrain_types[env_ids].clone()

    type_success_ema = getattr(env, "_delta_type_success_ema", None)
    type_episode_count = getattr(env, "_delta_type_episode_count", None)
    if type_success_ema is None or type_success_ema.numel() != num_terrain_types:
        type_success_ema = torch.full(
            (num_terrain_types,), 0.5, dtype=distance.dtype, device=env.device
        )
        type_episode_count = torch.zeros(
            num_terrain_types, dtype=torch.long, device=env.device
        )
        env._delta_type_success_ema = type_success_ema
        env._delta_type_episode_count = type_episode_count
    assert isinstance(type_episode_count, torch.Tensor)
    ema_alpha = min(max(float(type_success_ema_alpha), 0.0), 1.0)
    for terrain_index in range(num_terrain_types):
        type_mask = (old_types == terrain_index) & active
        if type_mask.any():
            batch_success = success[type_mask].float().mean()
            type_success_ema[terrain_index] = (1.0 - ema_alpha) * type_success_ema[
                terrain_index
            ] + ema_alpha * batch_success
            type_episode_count[terrain_index] += type_mask.sum()

    success_streak[env_ids] = torch.where(
        success, success_streak[env_ids] + 1, torch.zeros_like(success_streak[env_ids])
    )
    failure_streak[env_ids] = torch.where(
        failure, failure_streak[env_ids] + 1, torch.zeros_like(failure_streak[env_ids])
    )
    success_required = torch.where(
        sparse,
        torch.full_like(success_streak[env_ids], int(sparse_success_streak_length)),
        torch.full_like(success_streak[env_ids], int(success_streak_length)),
    )
    failure_required = torch.where(
        sparse,
        torch.full_like(failure_streak[env_ids], int(sparse_failure_streak_length)),
        torch.full_like(failure_streak[env_ids], int(failure_streak_length)),
    )
    move_up = success_streak[env_ids] >= success_required
    move_down = failure_streak[env_ids] >= failure_required
    old_levels = terrain.terrain_levels[env_ids]
    new_levels = old_levels.clone()
    if sparse_max_level is None:
        sparse_cap = terrain.max_terrain_level - 1
    else:
        sparse_cap = max(0, min(int(sparse_max_level), terrain.max_terrain_level - 1))
    max_allowed = torch.where(
        sparse,
        torch.full_like(old_levels, sparse_cap),
        torch.full_like(old_levels, terrain.max_terrain_level - 1),
    )

    if move_up.any():
        up_ids = move_up.nonzero(as_tuple=False).flatten()
        highest = old_levels[up_ids] >= max_allowed[up_ids]
        up_sparse = sparse[up_ids]
        regular = up_ids[~highest]
        new_levels[regular] += 1
        # Sparse terrain stays at its cap once it is mastered.  Continuous terrain
        # retains the paper-style restart from a low/high row after reaching the
        # final row.
        sparse_high = up_ids[highest & up_sparse]
        normal_high = up_ids[highest & ~up_sparse]
        if normal_high.numel() > 0:
            restart_low = torch.rand(len(normal_high), device=env.device) < 0.2
            restart_high = torch.randint(
                5, terrain.max_terrain_level, (len(normal_high),), device=env.device
            )
            new_levels[normal_high] = torch.where(
                restart_low, torch.zeros_like(restart_high), restart_high
            )
        if sparse_high.numel() > 0:
            new_levels[sparse_high] = max_allowed[sparse_high]
    new_levels = torch.where(move_down, torch.clamp(old_levels - 1, min=0), new_levels)
    new_levels = torch.minimum(new_levels, max_allowed)
    terrain.terrain_levels[env_ids] = new_levels

    base_proportions = torch.tensor(
        [generator.sub_terrains[name].proportion for name in terrain_names],
        dtype=distance.dtype,
        device=env.device,
    )
    base_proportions /= base_proportions.sum().clamp_min(1.0e-6)
    sampling_probabilities = base_proportions
    warmup_total = max(int(adaptive_warmup_episodes), 0) * num_terrain_types
    if adaptive_type_sampling and int(type_episode_count.sum()) >= warmup_total:
        weakness = max(float(adaptive_sampling_floor), 0.0) + 1.0 - type_success_ema
        temperature = max(float(adaptive_sampling_temperature), 1.0e-6)
        sampling_probabilities = base_proportions * weakness.clamp_min(1.0e-6).pow(
            temperature
        )
        sampling_probabilities /= sampling_probabilities.sum().clamp_min(1.0e-6)
        terrain.terrain_types[env_ids] = torch.multinomial(
            sampling_probabilities, len(env_ids), replacement=True
        ).to(terrain.terrain_types.dtype)

        new_sparse = torch.zeros_like(terrain.terrain_types[env_ids], dtype=torch.bool)
        for terrain_index in sparse_indices:
            new_sparse |= terrain.terrain_types[env_ids] == terrain_index
        if sparse_max_level is not None:
            terrain.terrain_levels[env_ids] = torch.where(
                new_sparse,
                terrain.terrain_levels[env_ids].clamp(max=sparse_cap),
                terrain.terrain_levels[env_ids],
            )
    terrain.env_origins[env_ids] = terrain.terrain_origins[
        terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]
    ]
    success_streak[env_ids] = torch.where(
        move_up, torch.zeros_like(success_streak[env_ids]), success_streak[env_ids]
    )
    failure_streak[env_ids] = torch.where(
        move_down, torch.zeros_like(failure_streak[env_ids]), failure_streak[env_ids]
    )

    levels = terrain.terrain_levels.float()
    result = {"mean": levels.mean(), "max": levels.max()}
    if terrain.terrain_origins.shape[1] == len(terrain_names):
        for i, name in enumerate(terrain_names):
            mask = terrain.terrain_types == i
            if mask.any():
                result[name] = levels[mask].mean()
            result[f"{name}_success_ema"] = type_success_ema[i]
            result[f"{name}_sampling_probability"] = sampling_probabilities[i]
    return result


__all__ = ["terrain_levels_delta", "terrain_levels_vel_strict"]
