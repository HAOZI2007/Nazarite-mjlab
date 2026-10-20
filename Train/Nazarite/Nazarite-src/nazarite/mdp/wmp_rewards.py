"""Reference WMP obstacle rewards adapted to Nazarite's Go2 sensors."""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply
from nazarite.mdp.rewards import feet_gait

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def tracking_linear_velocity(
    env,
    command_name: str = "twist",
    sigma: float = 0.25,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    terrain_names: tuple[str, ...] = ("tire_array", "high_wall"),
    terrain_velocity_clip: float = 0.1,
) -> torch.Tensor:
    """Match planar tracking and the source tolerance on obstacle terrain."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    velocity = asset.data.root_link_lin_vel_b[:, :2].clone()
    sparse = _terrain_mask(env, terrain_names)
    if sparse is not None:
        upper = torch.where(
            command[:, :2] < 0.0,
            torch.full_like(command[:, :2], 1.0e5),
            command[:, :2] + terrain_velocity_clip,
        )
        lower = torch.where(
            command[:, :2] > 0.0,
            torch.full_like(command[:, :2], -1.0e5),
            command[:, :2] - terrain_velocity_clip,
        )
        clipped = torch.maximum(torch.minimum(velocity, upper), lower)
        velocity = torch.where(sparse[:, None], clipped, velocity)
    error = (command[:, :2] - velocity).square().sum(dim=1)
    return torch.exp(-error / max(float(sigma), 1.0e-6))


def tracking_angular_velocity(
    env,
    command_name: str = "twist",
    sigma: float = 0.25,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Match the reference yaw-rate tracking reward."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = (command[:, 2] - asset.data.root_link_ang_vel_b[:, 2]).square()
    return torch.exp(-error / max(float(sigma), 1.0e-6))


def joint_position_limits(
    env,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    soft_limit_factor: float = 0.9,
) -> torch.Tensor:
    """Penalize excursions beyond the source task's 90% joint limits."""
    asset: Entity = env.scene[asset_cfg.name]
    limits = asset.data.joint_pos_limits[:, asset_cfg.joint_ids]
    center = limits.mean(dim=-1)
    half_range = (limits[..., 1] - limits[..., 0]) * (0.5 * soft_limit_factor)
    lower = center - half_range
    upper = center + half_range
    position = asset.data.joint_pos[:, asset_cfg.joint_ids]
    below = (lower - position).clamp_min(0.0)
    above = (position - upper).clamp_min(0.0)
    return (below + above).sum(dim=1)


def joint_velocity_limits(
    env,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    velocity_limit: float = 27.0,
) -> torch.Tensor:
    """Penalize velocity beyond 90% of the Go2 30 rad/s source limit."""
    asset: Entity = env.scene[asset_cfg.name]
    excess = (
        asset.data.joint_vel[:, asset_cfg.joint_ids].abs() - velocity_limit
    ).clamp(0.0, 1.0)
    return excess.sum(dim=1)


def dof_error(
    env,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Squared joint deviation from the default stand.

    The reference recipe uses this instead of a tolerance-banded posture reward:
    it is a single smooth basin around the default pose, so a leg held straight
    back (the crawl's rear "kickstand") is charged continuously rather than only
    once it leaves a band.
    """
    asset: Entity = env.scene[asset_cfg.name]
    default = asset.data.default_joint_pos
    assert default is not None
    joint_ids = asset_cfg.joint_ids
    deviation = asset.data.joint_pos[:, joint_ids] - default[:, joint_ids]
    return torch.sum(torch.square(deviation), dim=1)


def feet_air_time(
    env,
    sensor_name: str = "feet_ground_contact",
    threshold: float = 0.25,
    command_name: str = "twist",
    command_threshold: float = 0.1,
    obstacle_terrain_names: tuple[str, ...] = (),
    obstacle_scale: float = 1.0,
    signed: bool = True,
) -> torch.Tensor:
    """Pay completed swing time on foot landing events.

    ``signed=True`` reproduces the reference term: a swing shorter than
    ``threshold`` is charged its shortfall, a longer one pays its excess. The
    reference's 0.5 s threshold against a 0.6 s gait period is not reachable by a
    trot, so this term is what pulls the policy towards long, bounding swings
    instead of the front-leg skitter of the 2026-10-17 runs.

    ``signed=False`` keeps the one-sided form: nothing is paid until a swing
    exceeds ``threshold``, and short swings are free. That variant is for callers
    who want a leap bonus on top of a gait shaped elsewhere; against a 0.5 s
    threshold it is a no-op for every ordinary step, which is exactly what
    happened to the 2026-10-20 S1 run.

    ``obstacle_terrain_names``/``obstacle_scale`` let the same term pay more on
    the tiles where a long swing is the goal.
    """
    try:
        sensor = env.scene[sensor_name]
    except KeyError:
        return torch.zeros(env.num_envs, device=env.device)
    last_air_time = sensor.data.last_air_time
    if last_air_time is None:
        return torch.zeros(env.num_envs, device=env.device)
    first_contact = sensor.compute_first_contact(dt=env.step_dt)
    excess = last_air_time - threshold
    if not signed:
        excess = excess.clamp_min(0.0)
    reward = (excess * first_contact.float()).sum(dim=1)
    command = env.command_manager.get_command(command_name)
    active = torch.linalg.vector_norm(command[:, :2], dim=1) > command_threshold
    scale = torch.ones_like(reward)
    if obstacle_terrain_names:
        sparse = _terrain_mask(env, obstacle_terrain_names)
        if sparse is not None:
            scale = torch.where(sparse, torch.full_like(reward, obstacle_scale), scale)
    return reward * active.float() * scale


def feet_stumble(
    env, sensor_name: str = "feet_ground_contact", horizontal_ratio: float = 4.0
) -> torch.Tensor:
    """Apply one penalty when any foot hits a vertical face."""
    try:
        force = env.scene[sensor_name].data.force
    except KeyError:
        return torch.zeros(env.num_envs, device=env.device)
    if force is None:
        return torch.zeros(env.num_envs, device=env.device)
    horizontal = torch.linalg.vector_norm(force[..., :2], dim=-1)
    vertical = force[..., 2].abs()
    return (horizontal > horizontal_ratio * vertical).any(dim=1).float()


def _terrain_mask(env, terrain_names: tuple[str, ...]) -> torch.Tensor | None:
    try:
        terrain = env.scene["terrain"]
    except (AttributeError, KeyError):
        return None
    generator = terrain.cfg.terrain_generator
    names = list(generator.sub_terrains) if generator is not None else []
    selected = [names.index(name) for name in terrain_names if name in names]
    if not selected:
        return None
    return torch.isin(terrain.terrain_types, torch.tensor(selected, device=env.device))


def feet_edge_curriculum_scale(
    iteration: int,
    start_iteration: int = 4000,
    end_iteration: int = 10000,
    start_scale: float = 0.1,
    end_scale: float = 1.0,
) -> float:
    """Linearly ramp the feet-edge coefficient using the source schedule."""
    if end_iteration <= start_iteration:
        raise ValueError("end_iteration must be greater than start_iteration")
    progress = min(
        max((iteration - start_iteration) / (end_iteration - start_iteration), 0.0), 1.0
    )
    return start_scale + progress * (end_scale - start_scale)


def feet_edge(
    env,
    support_sensor_name: str = "wmp_feet_support",
    contact_sensor_name: str = "feet_ground_contact",
    command_name: str = "twist",
    command_threshold: float = 0.1,
    terrain_names: tuple[str, ...] = ("tire_array", "high_wall"),
    curriculum_start_iteration: int | None = None,
    curriculum_end_iteration: int = 10000,
    curriculum_start_scale: float = 0.1,
    curriculum_end_scale: float = 1.0,
) -> torch.Tensor:
    """Penalize contact near unsupported foot edges on sparse Go2 terrain."""
    try:
        support_sensor = env.scene[support_sensor_name]
        support = support_sensor.data.distances
        contact = env.scene[contact_sensor_name].data.found
    except (AttributeError, KeyError):
        return torch.zeros(env.num_envs, device=env.device)
    if support is None or contact is None:
        return torch.zeros(env.num_envs, device=env.device)

    # RayCastSensor stores distances flattened as [B, F * N], whereas contact
    # data is [B, F]. Restore the sensor's frame/ray dimensions before reducing
    # rays so the support fraction remains per-foot rather than per-environment.
    num_frames = int(getattr(support_sensor, "num_frames", 0))
    rays_per_frame = int(getattr(support_sensor, "num_rays_per_frame", 0))
    if num_frames < 1 or rays_per_frame < 1:
        return torch.zeros(env.num_envs, device=env.device)
    if contact.ndim != 2 or contact.shape[1] != num_frames:
        raise ValueError(
            f"Support sensor {support_sensor_name!r} has {num_frames} frames, "
            f"but contact data has shape {tuple(contact.shape)}."
        )
    if support.ndim < 2 or support.shape[0] != contact.shape[0]:
        raise ValueError(
            f"Support sensor {support_sensor_name!r} distances have shape "
            f"{tuple(support.shape)}, incompatible with contact data shape "
            f"{tuple(contact.shape)}."
        )
    expected_rays = num_frames * rays_per_frame
    if support.numel() != support.shape[0] * expected_rays:
        raise ValueError(
            f"Support sensor {support_sensor_name!r} distances have shape "
            f"{tuple(support.shape)}, expected [B, {expected_rays}] for "
            f"{num_frames} frames with {rays_per_frame} rays each."
        )
    support = support.reshape(contact.shape[0], num_frames, rays_per_frame)
    sparse = _terrain_mask(env, terrain_names)
    if sparse is None:
        return torch.zeros(env.num_envs, device=env.device)
    support_fraction = (support >= 0.0).float().mean(dim=-1)
    command = env.command_manager.get_command(command_name)
    active = (
        torch.linalg.vector_norm(command[:, :2], dim=1) + command[:, 2].abs()
    ) > command_threshold
    cost = ((1.0 - support_fraction).clamp(0.0, 1.0) * contact.float()).sum(dim=1)
    scale = 1.0
    if curriculum_start_iteration is not None:
        iteration = int(getattr(env, "wmp_learning_iteration", 0))
        scale = feet_edge_curriculum_scale(
            iteration,
            curriculum_start_iteration,
            curriculum_end_iteration,
            curriculum_start_scale,
            curriculum_end_scale,
        )
    return cost * sparse.float() * active.float() * scale


def cheat(
    env,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    terrain_names: tuple[str, ...] = ("tire_array", "high_wall"),
) -> torch.Tensor:
    """Discourage turning away from a forward command on obstacle tiles."""
    sparse = _terrain_mask(env, terrain_names)
    if sparse is None:
        return torch.zeros(env.num_envs, device=env.device)
    asset: Entity = env.scene[asset_cfg.name]
    forward = quat_apply(asset.data.root_link_quat_w, asset.data.forward_vec_b)
    heading = torch.atan2(forward[:, 1], forward[:, 0])
    command = env.command_manager.get_command(command_name)
    forward_command = command[:, 0] > 0.1
    return ((heading.abs() > 1.0) & forward_command & sparse).float()


def stuck(
    env,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    speed_threshold: float = 0.12,
    grace_steps: int = 25,
) -> torch.Tensor:
    """Penalize being stalled for longer than a grace period.

    The instantaneous form fired on any single slow step, so it mostly taxed the
    deliberate slow-down before a trench instead of the actual failure mode. The
    counter form only fires once the robot has been slow for ``grace_steps``
    (0.5 s at 50 Hz) while a forward command is active, i.e. when it is really
    stuck in front of an obstacle.
    """
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    count = _counter_state(env, "_wmp_stuck_counter")
    slow = asset.data.root_link_lin_vel_b[:, 0].abs() < speed_threshold
    active = command[:, 0].abs() > 0.1
    stalled = active & slow
    # Replaced, not updated in place: the counter is allocated inside the training
    # loop's inference-mode block, so an in place write during playback raises.
    count = torch.where(stalled, count + 1, torch.zeros_like(count))
    env.__dict__["_wmp_stuck_counter"] = count
    return (count > grace_steps).float()


def _counter_state(env, key: str) -> torch.Tensor:
    """Return a lazily allocated per-environment step counter."""
    counter = getattr(env, key, None)
    if counter is None or counter.shape[0] != env.num_envs:
        counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env.__dict__[key] = counter
    return counter


def foot_clearance_terrain(
    env,
    target_height: float = 0.18,
    flat_target_height: float = 0.10,
    height_sensor_name: str = "feet_terrain_height",
    command_name: str = "twist",
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    obstacle_terrain_names: tuple[str, ...] = (),
    swing_only: bool = False,
    contact_sensor_name: str = "feet_ground_contact",
) -> torch.Tensor:
    """Penalize foot clearance *shortfalls* only, weighted by foot velocity.

    ``velocity_mdp.feet_clearance`` penalizes ``|height - target|``, so it pushes
    back exactly as hard when the foot goes higher than the target as when it
    stays too low. On a course whose hard obstacles are 0.35-0.55 m trenches and
    0.18 m steps, going high is the solution, not the error. This term keeps the
    velocity weighting of the reference but truncates the low side only.
    """
    asset: Entity = env.scene[asset_cfg.name]
    height_sensor = env.scene[height_sensor_name]
    foot_height = height_sensor.data.heights
    if foot_height is None:
        return torch.zeros(env.num_envs, device=env.device)
    foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
    vel_norm = torch.linalg.vector_norm(foot_vel_xy, dim=-1)
    target = torch.full_like(foot_height, float(flat_target_height))
    if obstacle_terrain_names:
        sparse = _terrain_mask(env, obstacle_terrain_names)
        if sparse is not None:
            target = torch.where(
                sparse[:, None], torch.full_like(target, float(target_height)), target
            )
    shortfall = (target - foot_height).clamp_min(0.0)
    weighted = shortfall * vel_norm
    if swing_only:
        weighted = weighted * _swing_mask(env, contact_sensor_name, weighted)
    cost = weighted.sum(dim=1)
    command = env.command_manager.get_command(command_name)
    active = (
        torch.linalg.vector_norm(command[:, :2], dim=1) + command[:, 2].abs()
    ) > command_threshold
    return cost * active.float()


def _swing_mask(env, contact_sensor_name: str, like: torch.Tensor) -> torch.Tensor:
    """Return a 1.0/0.0 mask that is one for feet currently off the ground.

    A dragging foot barely moves relative to the ground, so the velocity weight
    above already makes its shortfall cheap; the same weighting also makes a
    foot that is genuinely lifted but slow look free. Restricting the term to
    the swing phase keeps the incentive on the lift itself, which is the
    behaviour the rear legs of the 2026-10-18 run never learned.
    """
    try:
        contact = env.scene[contact_sensor_name].data.found
    except (AttributeError, KeyError):
        return torch.ones_like(like)
    if contact is None or contact.shape != like.shape:
        return torch.ones_like(like)
    # ``found`` counts matched contacts per primary, so it is an integer tensor
    # rather than a boolean one; 0 is the only value that means "in the air".
    if contact.dtype != torch.bool:
        contact = contact > 0
    return (~contact).float()


def feet_air_time_band(
    env,
    sensor_name: str = "feet_ground_contact",
    threshold_min: float = 0.22,
    threshold_max: float = 0.55,
    command_name: str = "twist",
    command_threshold: float = 0.1,
    obstacle_terrain_names: tuple[str, ...] = (),
    obstacle_scale: float = 1.0,
) -> torch.Tensor:
    """Score completed swing time against a target window on landing events.

    ``feet_air_time`` is one-sided: a swing shorter than its threshold pays
    zero, so the front-leg skittering of the 2026-10-18 run cost nothing. Here
    a shortfall below ``threshold_min`` is *negative*, so micro-hops are
    charged, while ``threshold_max`` saturates the reward instead of paying for
    ever longer flight.
    """
    try:
        sensor = env.scene[sensor_name]
    except KeyError:
        return torch.zeros(env.num_envs, device=env.device)
    last_air_time = sensor.data.last_air_time
    if last_air_time is None:
        return torch.zeros(env.num_envs, device=env.device)
    first_contact = sensor.compute_first_contact(dt=env.step_dt)
    low = float(min(threshold_min, threshold_max))
    high = float(max(threshold_min, threshold_max))
    signed = last_air_time.clamp(max=high) - low
    reward = (signed * first_contact.float()).sum(dim=1)
    command = env.command_manager.get_command(command_name)
    active = torch.linalg.vector_norm(command[:, :2], dim=1) > command_threshold
    scale = torch.ones_like(reward)
    if obstacle_terrain_names:
        sparse = _terrain_mask(env, obstacle_terrain_names)
        if sparse is not None:
            scale = torch.where(sparse, torch.full_like(reward, obstacle_scale), scale)
    return reward * active.float() * scale


def feet_gait_terrain(
    env,
    period: float = 0.6,
    offset: list[float] | tuple[float, ...] = (0.0, 0.5, 0.5, 0.0),
    threshold: float = 0.56,
    command_threshold: float = 0.1,
    command_name: str = "twist",
    sensor_name: str = "feet_ground_contact",
    soft_terrain_names: tuple[str, ...] = (),
    soft_scale: float = 0.3,
) -> torch.Tensor:
    """Diagonal gait matching, softened on the tiles that need a different gait.

    A gap crossing and a 0.55 m wide beam both need non-diagonal leg patterns,
    so a single hard trot constraint would fight the obstacles. The match is
    kept at full weight on the terrain families where a periodic gait is the
    answer and scaled down where it is not.
    """
    reward = feet_gait(
        env,
        period=period,
        offset=list(offset),
        threshold=threshold,
        command_threshold=command_threshold,
        command_name=command_name,
        sensor_name=sensor_name,
    )
    if soft_terrain_names:
        mask = _terrain_mask(env, soft_terrain_names)
        if mask is not None:
            reward = reward * torch.where(
                mask, torch.full_like(reward, soft_scale), torch.ones_like(reward)
            )
    return reward


def base_clearance_reward(
    env,
    sensor_name: str = "wmp_base_clearance",
    target_height: float = 0.30,
    std: float = 0.06,
    excluded_terrain_names: tuple[str, ...] = ("low_frame",),
) -> torch.Tensor:
    """Reward keeping the base at working height above the *local* ground.

    Two properties matter here.

    First, the measurement is terrain relative: the sensor reports
    ``frame_z - hit_z``, so on a ramp, on stairs and on top of a wall the term
    sees the same clearance for the same posture, which an absolute base height
    cannot do.

    Second, the shape is a one-sided gate rather than a symmetric target. Over a
    trench the ray reaches the pit floor and the clearance jumps to 0.6-0.9 m;
    a symmetric ``exp(-(h - target)^2 / std^2)`` would collapse to zero exactly
    during the jump, i.e. it would punish flying. Only being *too low* is an
    error, so only the low side is penalized.

    ``excluded_terrain_names`` defaults to the low-clearance frame tile, whose
    beam sits at 0.42-0.60 m while the standing robot is about 0.44 m tall:
    there the only way through is to crouch, and a height reward would fight it.
    """
    try:
        sensor = env.scene[sensor_name]
    except KeyError:
        return torch.zeros(env.num_envs, device=env.device)
    heights = sensor.data.heights
    if heights is None:
        return torch.zeros(env.num_envs, device=env.device)
    if heights.ndim > 1:
        heights = heights.reshape(heights.shape[0], -1).min(dim=1).values
    clearance = heights.clamp(0.0, 2.0)
    shortfall = (float(target_height) - clearance).clamp_min(0.0)
    reward = torch.exp(-shortfall.square() / max(float(std), 1.0e-6) ** 2)
    if excluded_terrain_names:
        excluded = _terrain_mask(env, excluded_terrain_names)
        if excluded is not None:
            reward = torch.where(excluded, torch.zeros_like(reward), reward)
    return reward


def course_checkpoint_bonus(
    env,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    terrain_names: tuple[str, ...] = (),
    spacing: float = 0.5,
    max_checkpoints: int = 16,
    min_upright: float = 0.85,
) -> torch.Tensor:
    """Pay once per ``spacing`` metres of forward progress along the course.

    ``tracking_linear_velocity`` is an exponential of the velocity error, so on a
    tile the robot cannot yet cross it saturates near zero while every penalty
    term keeps accruing: standing in front of a trench is locally optimal. This
    term pays for distance actually covered, so a 0.3 m improvement in clearing a
    trench is worth something even when the commanded speed is still missed. The
    per-environment checkpoint index is monotone within an episode and follows the
    robot back down on reset.

    The bonus is only paid while the base is upright, where ``min_upright`` is a
    floor on ``-projected_gravity_b[:, 2]`` (the cosine of the tilt, so 0.85 is
    about 32 deg). Without that gate the cheapest way to collect it is a
    controlled fall: tip the trunk forward, paddle the front legs in short hops and
    let the hind feet drag. That gait reaches the checkpoints at a fraction of the
    effort, and in the 2026-10-17_22-32-26 run it became the policy's whole
    locomotion -- nose down, front feet skittering, rear feet never leaving the
    ground (contact slip velocity 0.37 m/s against 0.11 in the previous run,
    landing force 94 N against 61).
    """
    asset: Entity = env.scene[asset_cfg.name]
    index = _counter_state(env, "_wmp_course_checkpoint")
    origin_x = env.scene.env_origins[:, 0]
    progress = asset.data.root_link_pos_w[:, 0] - origin_x
    current = (progress / spacing).floor().clamp(0.0, float(max_checkpoints))
    current = current.to(torch.long)
    reward = (current - index).clamp_min(0).float()
    # Replaced, not updated in place: the index is allocated inside the training
    # loop's inference-mode block, so an in place write during playback raises.
    env.__dict__["_wmp_course_checkpoint"] = current
    if terrain_names:
        sparse = _terrain_mask(env, terrain_names)
        if sparse is not None:
            reward = reward * sparse.float()
    command = env.command_manager.get_command(command_name)
    upright = -asset.data.projected_gravity_b[:, 2] >= min_upright
    active = (command[:, 0] > 0.1) & upright
    return reward * active.float()


def lin_vel_z_l2(env, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Penalize squared vertical base velocity."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_b[:, 2].square()
