"""XZ1-only narrow-course commands, rewards and traversal checks.

Course geometry is privileged training state. The actor still receives the
same 47-D frame and deployment still consumes ordinary body velocity commands.
"""

from dataclasses import dataclass

import torch

from mjlab.envs.mdp.events import reset_root_state_uniform
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import wrap_to_pi
from nazarite.config.robot_config.xz1_cfg import XZ1_FOOT_SITES
from nazarite.mdp.commands import (
    TerrainConditionedVelocityCommand,
    TerrainConditionedVelocityCommandCfg,
)
from nazarite.mdp.events import reset_root_state_uniform_facing_high_wall
from nazarite.mdp.rewards import (
    hip_joint_deviation_penalty,
    terrain_variable_posture,
)
from nazarite.mdp.xz1_contacts import GapBottomContact

from .him_obstacles import OBSTACLE_TERRAIN_NAMES, lerp

NARROW_TERRAIN_NAMES = ("narrow_stairs", "narrow_ramp")
COURSE_TERRAIN_NAMES = (*NARROW_TERRAIN_NAMES, *OBSTACLE_TERRAIN_NAMES)
HIP_NAMES = tuple(f"{leg}_hip_joint" for leg in ("FL", "FR", "RR", "RL"))


class XZ1TerrainPosture(terrain_variable_posture):
    """Track the default joint pose with terrain relief, except on high walls."""

    def __call__(
        self,
        env,
        std_standing,
        std_walking,
        std_running,
        terrain_tolerance,
        asset_cfg,
        command_name,
        walking_threshold=0.1,
        running_threshold=1.5,
        scan_sensor_name="terrain_scan",
        terrain_radius=0.35,
        terrain_height_std=(0.01, 0.08),
    ):
        reward = super().__call__(
            env,
            std_standing,
            std_walking,
            std_running,
            terrain_tolerance,
            asset_cfg,
            command_name,
            walking_threshold,
            running_threshold,
            scan_sensor_name,
            terrain_radius,
            terrain_height_std,
        )
        terrain = env.scene.terrain
        generator = terrain.cfg.terrain_generator if terrain is not None else None
        names = list(generator.sub_terrains) if generator is not None else []
        for name in ("high_wall", "low_frame"):
            if name in names:
                assert terrain is not None
                # Permit jumping and crouching without a default-pose bonus.
                selected = terrain.terrain_types == names.index(name)
                reward = torch.where(selected, 0.0, reward)
        return reward


def course_bounds(env):
    """Return selected mask, root xy, half width and obstacle x bounds at spawn."""
    terrain = env.scene.terrain
    position = env.scene["robot"].data.root_link_pos_w - env.scene.env_origins
    selected = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    half_width = torch.zeros_like(position[:, 0])
    start = torch.zeros_like(half_width)
    end = torch.zeros_like(half_width)
    generator = terrain.cfg.terrain_generator if terrain is not None else None
    if terrain is None or generator is None:
        return selected, position, half_width, start, end
    names = list(generator.sub_terrains)
    lower, upper = generator.difficulty_range
    difficulty = lower + (upper - lower) * terrain.terrain_levels / max(
        generator.num_rows - 1, 1
    )
    difficulty = difficulty.clamp(0.0, 1.0)
    for name in COURSE_TERRAIN_NAMES:
        if name not in names:
            continue
        mask = terrain.terrain_types == names.index(name)
        cfg = generator.sub_terrains[name]
        if name in OBSTACLE_TERRAIN_NAMES:
            x0, x1, width = cfg.course_profile(difficulty)
            length = x1 - x0
        elif name == "narrow_stairs":
            x0 = cfg.start_x
            width = lerp(difficulty, cfg.width_range)
            length = 2 * (cfg.num_steps - 1) * cfg.step_depth + cfg.deck_length
        else:
            x0 = cfg.start_x
            width = lerp(difficulty, cfg.width_range)
            rise = cfg.height_range[0] + difficulty * (
                cfg.height_range[1] - cfg.height_range[0]
            )
            angle = cfg.slope_angle_range[0] + difficulty * (
                cfg.slope_angle_range[1] - cfg.slope_angle_range[0]
            )
            length = 2 * rise / torch.tan(torch.deg2rad(angle)) + cfg.deck_length
        start_x = x0 - generator.size[0] * cfg.spawn_x_fraction
        selected |= mask
        half_width = torch.where(mask, width / 2, half_width)
        start = torch.where(mask, start_x, start)
        end = torch.where(mask, start_x + length, end)
    return selected, position, half_width, start, end


def narrow_course_complete(env, clearance: float = 0.35):
    """Finish after the trunk clears the obstacle, with its rear feet in mind."""
    selected, position, half_width, _, end = course_bounds(env)
    complete = (
        selected
        & (position[:, 0] >= end + clearance)
        & (position[:, 1].abs() <= half_width - 0.12)
    )
    terrain = env.scene.terrain
    names = list(terrain.cfg.terrain_generator.sub_terrains)
    # The added courses need all four feet on the far landing. A leading trunk
    # or a front-foot touch alone cannot end the episode or promote difficulty.
    new_course = torch.zeros_like(selected)
    for name in OBSTACLE_TERRAIN_NAMES:
        if name in names:
            new_course |= terrain.terrain_types == names.index(name)
    if new_course.any():
        feet_ids, _ = env.scene["robot"].find_sites(XZ1_FOOT_SITES, preserve_order=True)
        feet = env.scene["robot"].data.site_pos_w[:, feet_ids]
        feet = feet - env.scene.env_origins[:, None, :]
        loaded = (
            env.scene["feet_ground_contact"]
            .data.force[..., 2]
            .reshape(env.num_envs, 4)
            .abs()
            > 5.0
        )
        landed = (
            (feet[..., 0].amin(-1) >= end + 0.05)
            & (feet[..., 1].abs().amax(-1) <= half_width + 0.10)
            & (feet[..., 2].amin(-1) > -0.015)
            & (loaded.sum(-1) >= 2)
        )
        complete &= ~new_course | landed
    return complete


class ObstacleGroundContact:
    """Reject support on the floor below stones/bridge, and in gap bottoms.

    Tire holes are valid contacts: only lateral bypass is rejected there.
    This is a training traversal check, not the competition's full referee.
    """

    def __init__(self, cfg, env):
        del cfg
        self.feet_ids, _ = env.scene["robot"].find_sites(
            XZ1_FOOT_SITES, preserve_order=True
        )
        names = env.scene.terrain.cfg.terrain_generator.sub_terrains
        self.gap_bottom = (
            GapBottomContact(env.sim.mj_model, env.num_envs, env.device)
            if "gap" in names
            else None
        )

    def __call__(self, env):
        _, _, _, start, end = course_bounds(env)
        terrain = env.scene.terrain
        names = list(terrain.cfg.terrain_generator.sub_terrains)
        feet = env.scene["robot"].data.site_pos_w[:, self.feet_ids]
        feet = feet - env.scene.env_origins[:, None, :]
        force = env.scene["feet_ground_contact"].data.force
        loaded = force[..., 2].reshape(env.num_envs, 4).abs() > 5.0
        alongside = (feet[..., 0] >= start[:, None]) & (feet[..., 0] <= end[:, None])
        below = feet[..., 2] < -0.015
        fallen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for name in ("plum_blossom", "single_beam"):
            if name not in names:
                continue
            on_course = terrain.terrain_types == names.index(name)
            ground_feet = (below & loaded & alongside).sum(-1)
            # On stones/bridge two feet on the surrounding ground fail.
            fallen |= on_course & (ground_feet >= 2)
        if self.gap_bottom is not None:
            # A foot below deck height can still be pushing on a vertical
            # platform edge. Only actual foot/floor contact is a missed gap.
            fallen |= (terrain.terrain_types == names.index("gap")) & self.gap_bottom(
                env.sim.wp_data
            )
        return fallen


def left_narrow_course(env):
    """Reject lateral bypasses and body drift off the raised support corridor."""
    selected, position, half_width, start, end = course_bounds(env)
    alongside = (position[:, 0] >= start - 0.25) & (position[:, 0] <= end + 0.35)
    return selected & alongside & (position[:, 1].abs() > half_width - 0.12)


def narrow_success_filter(env, env_ids, success):
    selected, *_ = course_bounds(env)
    return torch.where(
        selected[env_ids], success & narrow_course_complete(env)[env_ids], success
    )


def narrow_support_edge_cost(
    env,
    sensor_name: str,
    asset_cfg: SceneEntityCfg,
    edge_margin: float = 0.03,
    foot_radius: float = 0.024,
    force_threshold: float = 5.0,
    distance_scale: float = 0.10,
):
    """Keep loaded foot spheres inside the lateral edges of narrow obstacles.

    Check each foot's longitudinal position, not the trunk's: front feet can
    reach the ascent before the trunk, and rear feet leave the descent later.
    Swing feet and contacts before/after the obstacle remain unconstrained.
    """
    if min(edge_margin, foot_radius, force_threshold) < 0 or distance_scale <= 0:
        raise ValueError("Edge dimensions/force must be nonnegative; scale positive.")
    selected, _, half_width, start, end = course_bounds(env)
    feet = env.scene[asset_cfg.name].data.site_pos_w[:, asset_cfg.site_ids]
    feet = feet - env.scene.env_origins[:, None, :]
    contact = env.scene[sensor_name].data
    found = contact.found.reshape(env.num_envs, -1) > 0
    vertical_force = contact.force[..., 2].reshape(env.num_envs, -1).abs()
    loaded = found & (vertical_force > force_threshold)
    alongside = (feet[..., 0] >= start[:, None]) & (feet[..., 0] <= end[:, None])
    excess = (
        feet[..., 1].abs() + foot_radius + edge_margin - half_width[:, None]
    ).clamp_min(0.0)
    # A 5 cm intrusion costs 0.25 per foot before the reward weight. Bound
    # outliers so a failed landing cannot dominate an entire PPO minibatch.
    cost = torch.nan_to_num((excess / distance_scale).clamp_max(2.0).square())
    return (cost * loaded * alongside * selected[:, None]).sum(-1)


def reset_xz1_course(env, env_ids, pose_range, velocity_range=None):
    """Center narrow-course resets; retain the existing wall/wide reset rules."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    terrain = env.scene.terrain
    names = list(terrain.cfg.terrain_generator.sub_terrains)
    mask = torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)
    for name in COURSE_TERRAIN_NAMES:
        if name in names:
            mask |= terrain.terrain_types[env_ids] == names.index(name)
    regular = env_ids[~mask]
    if regular.numel():
        reset_root_state_uniform_facing_high_wall(
            env, regular, pose_range, velocity_range
        )
    narrow = env_ids[mask]
    if narrow.numel():
        centered = dict(pose_range)
        centered.update(x=(-0.1, 0.1), y=(-0.05, 0.05), yaw=(-0.05, 0.05))
        reset_root_state_uniform(env, narrow, centered, velocity_range)


@dataclass(kw_only=True)
class XZ1CourseVelocityCommandCfg(TerrainConditionedVelocityCommandCfg):
    """Ordinary velocity sampling, with centerline following on narrow courses."""

    narrow_speed_range: tuple[float, float] = (0.3, 0.7)
    lateral_gain: float = 0.8
    lateral_limit: float = 0.15
    yaw_gain: float = 0.8
    yaw_limit: float = 0.3

    def __post_init__(self):
        super().__post_init__()
        if not 0 < self.narrow_speed_range[0] <= self.narrow_speed_range[1]:
            raise ValueError("Narrow-course speeds must be positive and increasing.")
        if (
            min(self.lateral_gain, self.lateral_limit, self.yaw_gain, self.yaw_limit)
            <= 0
        ):
            raise ValueError("Course correction gains and limits must be positive.")

    def build(self, env):
        return XZ1CourseVelocityCommand(self, env)


class XZ1CourseVelocityCommand(TerrainConditionedVelocityCommand):
    def __init__(self, cfg: XZ1CourseVelocityCommandCfg, env):
        super().__init__(cfg, env)
        self.course_cfg: XZ1CourseVelocityCommandCfg = cfg
        self._narrow_speed = torch.zeros(self.num_envs, device=self.device)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        selected, *_ = course_bounds(self._env)
        ids = env_ids[selected[env_ids]]
        if not len(ids):
            return
        self._narrow_speed[ids] = torch.empty(len(ids), device=self.device).uniform_(
            *self.course_cfg.narrow_speed_range
        )
        self.is_standing_env[ids] = False
        self.is_forward_env[ids] = False
        self.is_world_env[ids] = False
        self.is_heading_env[ids] = False

    def _update_command(self, env_ids=None):
        super()._update_command(env_ids)
        selected, position, *_ = course_bounds(self._env)
        ids = selected.nonzero(as_tuple=False).flatten()
        yaw = self.robot.data.heading_w[ids]
        world_y = (-self.course_cfg.lateral_gain * position[ids, 1]).clamp(
            -self.course_cfg.lateral_limit, self.course_cfg.lateral_limit
        )
        world_x = self._narrow_speed[ids]
        self.vel_command_b[ids, 0] = torch.cos(yaw) * world_x + torch.sin(yaw) * world_y
        self.vel_command_b[ids, 1] = (
            -torch.sin(yaw) * world_x + torch.cos(yaw) * world_y
        )
        self.vel_command_b[ids, 2] = (
            -self.course_cfg.yaw_gain * wrap_to_pi(yaw)
        ).clamp(-self.course_cfg.yaw_limit, self.course_cfg.yaw_limit)


def xz1_hip_position_cost(
    env,
    command_name: str,
    outward_allowance: float = 0.15,
    inward_allowance: float = 0.15,
    soft_gain: float = 6.0,
    base_gain: float = 0.25,
    lateral_full: float = 0.2,
    lateral_free: float = 0.6,
    yaw_full: float = 0.2,
    yaw_free: float = 0.6,
):
    """Allow normal splay, taper with maneuver demand; wall cost stays exact."""
    if not (0 <= lateral_full < lateral_free and 0 <= yaw_full < yaw_free):
        raise ValueError("Hip command taper thresholds must be increasing.")
    if min(outward_allowance, inward_allowance, soft_gain, base_gain) < 0:
        raise ValueError("Hip allowances and cost gains must be nonnegative.")
    asset = env.scene["robot"]
    joint_ids, _ = asset.find_joints(HIP_NAMES, preserve_order=True)
    deviation = (
        asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
    )
    outward = deviation * deviation.new_tensor([1.0, -1.0, -1.0, 1.0])
    cost = (outward - outward_allowance).clamp_min(0).square().sum(-1)
    cost += (-outward - inward_allowance).clamp_min(0).square().sum(-1)
    cost = soft_gain * cost + base_gain * deviation.square().sum(-1)
    command = env.command_manager.get_command(command_name)
    gate = ((lateral_free - command[:, 1].abs()) / (lateral_free - lateral_full)).clamp(
        0, 1
    )
    gate *= ((yaw_free - command[:, 2].abs()) / (yaw_free - yaw_full)).clamp(0, 1)
    terrain = env.scene.terrain
    generator = terrain.cfg.terrain_generator if terrain is not None else None
    names = list(generator.sub_terrains) if generator is not None else []
    protected = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if terrain is not None and "high_wall" in names:
        protected = terrain.terrain_types == names.index("high_wall")
    legacy = hip_joint_deviation_penalty(env, command_name)
    return torch.where(protected, legacy, cost * gate)


def hip_outward_peak(env):
    asset = env.scene["robot"]
    joint_ids, _ = asset.find_joints(HIP_NAMES, preserve_order=True)
    angles = (
        asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
    )
    return (angles * angles.new_tensor([1.0, -1.0, -1.0, 1.0])).clamp_min(0).amax(-1)
