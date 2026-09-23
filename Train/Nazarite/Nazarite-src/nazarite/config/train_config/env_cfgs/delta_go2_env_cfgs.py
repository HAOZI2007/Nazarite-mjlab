"""Standalone DELTA Go2 environment with one head-mounted depth camera."""

import math
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.terrains import TerrainEntityCfg, TerrainGeneratorCfg
from mjlab.terrains.config import (
    box_random_grid,
    hf_pyramid_slope,
    nested_rings,
    open_stairs,
    pyramid_stairs,
    random_rough,
    wave_terrain,
)
from nazarite.config.robot_config.go2_cfg import (
    GO2_BASE_BODY,
    GO2_CALF_BODIES,
    GO2_FOOT_BODIES,
    GO2_THIGH_BODIES,
)
from nazarite.config.train_config.env_cfgs.go2_env_cfgs import (
    Nazarite_Velocity_Flat_Go2,
)
from nazarite.delta.d435 import (
    DELTA_D435_INTRINSICS,
    DELTA_DEPTH_HEIGHT,
    DELTA_DEPTH_WIDTH,
)
from nazarite.delta.terrains import BoxApproachSteppingStonesTerrainCfg
from nazarite.mdp import commands as custom_commands
from nazarite.mdp import curriculums as custom_curriculums
from nazarite.mdp import rewards as custom_rewards
from nazarite.mdp.delta_observations import delta_depth_image

DELTA_TERRAINS_CFG = TerrainGeneratorCfg(
    # 4 m tiles give the 10 s episode enough opportunities to cross several
    # terrain types while keeping the terrain inside the D435/DELTA local map.
    size=(4.0, 4.0),
    border_width=20.0,
    num_rows=10,
    num_cols=8,
    curriculum=True,
    # Rows 0--9 are the ten DELTA difficulty stages.  Stage 0 is flat/easy;
    # each primitive scales its height, gap, or slope with this value.
    difficulty_range=(0.0, 1.0),
    sub_terrains={
        "hills": wave_terrain(
            proportion=0.125, amplitude_range=(0.0, 0.16), num_waves=3
        ),
        "slope": hf_pyramid_slope(proportion=0.125, slope_range=(0.0, 0.30)),
        "stairs": pyramid_stairs(proportion=0.125, step_height_range=(0.015, 0.16)),
        "steps": open_stairs(
            proportion=0.125,
            step_height_range=(0.02, 0.16),
            step_width_range=(0.45, 0.85),
        ),
        "rough_ground": random_rough(
            proportion=0.125,
            noise_range=(0.01, 0.12),
            noise_step=0.02,
            scale_with_difficulty=True,
        ),
        "gaps": nested_rings(
            proportion=0.125,
            num_rings=6,
            ring_width_range=(0.45, 0.75),
            gap_range=(0.08, 0.30),
            height_range=(0.05, 0.25),
        ),
        "stepping_stones": BoxApproachSteppingStonesTerrainCfg(
            proportion=0.125,
            stone_size_range=(0.55, 0.95),
            stone_distance_range=(0.08, 0.30),
            stone_height=0.12,
            stone_height_variation=0.05,
            stone_size_variation=0.10,
            displacement_range=0.06,
            # Spawn on a short rear platform, then approach the sparse stones
            # in +x.  This avoids resetting in the pit at the tile centre.
            approach_platform_length=1.20,
            approach_platform_width=1.40,
            approach_gap=0.25,
            platform_width=0.0,
        ),
        "grid_stones": box_random_grid(
            proportion=0.125,
            grid_width=0.45,
            grid_height_range=(0.0, 0.18),
            # Keep one connected random-height surface; cell gaps are filled
            # by a shallow floor and there is no separate center patch.
            platform_width=0.0,
            fill_gaps=True,
            holes=False,
        ),
    },
    add_lights=True,
)


DELTA_DEPTH_CAMERA = CameraSensorCfg(
    name="delta_depth_camera",
    parent_body="robot/base_link",
    pos=(0.30, 0.0, 0.12),
    # D435 optical axis forward, pitched 20 degrees down in the robot frame.
    quat=(0.579228, 0.405580, -0.405580, -0.579228),
    width=DELTA_DEPTH_WIDTH,
    height=DELTA_DEPTH_HEIGHT,
    focal_length_px=(DELTA_D435_INTRINSICS.fx, DELTA_D435_INTRINSICS.fy),
    principal_point_px=(DELTA_D435_INTRINSICS.cx, DELTA_D435_INTRINSICS.cy),
    data_types=("depth",),
    enabled_geom_groups=(0,),
    use_shadows=False,
    use_textures=False,
)


def Nazarite_Delta_Go2(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build an isolated DELTA environment without rough-task config sharing."""
    cfg = Nazarite_Velocity_Flat_Go2(play=play, enable_wtw=False)
    if not play:
      cfg.episode_length_s = 10.0

    # Make the DELTA robot definition genuinely local.  The shared Go2
    # definition models a long (0--9 physics-step) command delay and starts
    # the rear thighs 0.2 rad deeper than the front thighs.  That combination
    # is particularly visible as rear-leg lag and a crouched reset posture on
    # rough terrain.  Do not mutate the baseline task: DELTA gets its own
    # copies of the entity, initial state, and actuator configs.
    robot_cfg = deepcopy(cfg.scene.entities["robot"])
    robot_cfg.init_state = deepcopy(robot_cfg.init_state)
    robot_cfg.init_state.pos = (0.0, 0.0, 0.38)
    robot_cfg.init_state.joint_pos = {
        r".*_thigh_joint": 0.8,
        r".*_calf_joint": -1.5,
    }
    if robot_cfg.articulation is not None:
      robot_cfg.articulation = deepcopy(robot_cfg.articulation)
      for actuator_cfg in robot_cfg.articulation.actuators:
        # The project-wide Go2 gains were reduced for another task.  DELTA's
        # zero-command standing test showed that the robot then settles into
        # calf/shank contact instead of supporting its body.  Keep this gain
        # change local to DELTA so other tasks retain their dynamics.
        if hasattr(actuator_cfg, "stiffness"):
          actuator_cfg.stiffness = max(float(actuator_cfg.stiffness), 40.0)
        if hasattr(actuator_cfg, "damping"):
          actuator_cfg.damping = max(float(actuator_cfg.damping), 2.0)
        actuator_cfg.delay_min_lag = 0
        actuator_cfg.delay_max_lag = 2
        actuator_cfg.delay_update_period = 10
    cfg.scene.entities["robot"] = robot_cfg
    cfg.scene.num_envs = 256
    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=deepcopy(DELTA_TERRAINS_CFG),
        max_init_terrain_level=0,
    )
    cfg.commands["twist"] = custom_commands.TerrainConditionedVelocityCommandCfg(
        entity_name="robot",
        resampling_time_range=(4.0, 6.0),
        rel_standing_envs=0.10,
        rel_heading_envs=0.0,
        rel_forward_envs=0.0,
        heading_command=False,
        ranges=custom_commands.TerrainConditionedVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.00, 1.00),
            lin_vel_y=(-0.50, 0.50),
            ang_vel_z=(-0.50, 0.50),
            heading=None,
        ),
        # 24 rollout steps/env/iteration in delta_rl_cfg.py.
        # Boundaries: iterations 0, 3000, 8000, 15000.
        stages=(
            custom_commands.ScheduledVelocityStageCfg(
                step=0,
                lin_vel_x=(-0.30, 0.30),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-0.15, 0.15),
            ),
            custom_commands.ScheduledVelocityStageCfg(
                step=72_000,
                lin_vel_x=(-0.50, 0.50),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-0.25, 0.25),
            ),
            custom_commands.ScheduledVelocityStageCfg(
                step=192_000,
                lin_vel_x=(-0.80, 0.80),
                lin_vel_y=(-0.35, 0.35),
                ang_vel_z=(-0.35, 0.35),
            ),
            custom_commands.ScheduledVelocityStageCfg(
                step=360_000,
                lin_vel_x=(-1.00, 1.00),
                lin_vel_y=(-0.50, 0.50),
                ang_vel_z=(-0.50, 0.50),
            ),
        ),
        forward_only_terrain_names=("stepping_stones",),
        forward_only_x_min=0.10,
    )
    cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
        func=custom_curriculums.terrain_levels_delta,
        params={
            "command_name": "twist",
            "success_distance_scale": 0.8,
            "min_success_distance": 0.8,
            "max_success_distance": 1.6,
            "success_streak_length": 2,
            "failure_streak_length": 3,
            "sparse_terrain_names": ("gaps", "stepping_stones", "grid_stones"),
            "sparse_success_distance_scale": 0.5,
            "sparse_min_success_distance": 0.5,
            "sparse_max_success_distance": 1.2,
            "sparse_success_streak_length": 3,
            "sparse_failure_streak_length": 4,
            "sparse_max_level": 4,
        },
    )
    cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + (deepcopy(DELTA_DEPTH_CAMERA),)

    cfg.events.pop("base_mass", None)
    cfg.events.pop("link_mass", None)
    cfg.events["effort_limits"] = EventTermCfg(
        func=dr.effort_limits,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", actuator_names=".*"),
            "operation": "scale",
            "effort_limit_range": (0.9, 1.1),
        },
    )
    cfg.events["pseudo_inertia"] = EventTermCfg(
        func=dr.pseudo_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=(GO2_BASE_BODY,)
                + GO2_THIGH_BODIES
                + GO2_CALF_BODIES
                + GO2_FOOT_BODIES,
            ),
            "alpha_range": (0.5 * math.log(0.8), 0.5 * math.log(1.2)),
            "distribution": "uniform",
        },
    )

    actor_terms = deepcopy(cfg.observations["actor"].terms)
    actor_terms.pop("height_scan", None)
    actor_terms["delta_map"] = ObservationTermCfg(
        func=delta_depth_image,
        params={"sensor_name": "delta_depth_camera", "map_height": 16, "map_width": 26},
    )
    cfg.observations["actor"] = ObservationGroupCfg(
        terms=actor_terms,
        concatenate_terms=True,
        enable_corruption=not play,
    )
    critic_terms = deepcopy(cfg.observations["critic"].terms)
    critic_terms.pop("height_scan", None)
    critic_terms["delta_map"] = ObservationTermCfg(
        func=delta_depth_image,
        params={"sensor_name": "delta_depth_camera", "map_height": 16, "map_width": 26},
    )
    cfg.observations["critic"] = ObservationGroupCfg(
        terms=critic_terms,
        concatenate_terms=True,
        enable_corruption=False,
    )

    # DELTA uses a sparse-terrain reward set.  Fixed-pose, fixed-gait and
    # prolonged-stance terms from the flat task are intentionally absent.
    base_cfg = SceneEntityCfg("robot", body_names=(GO2_BASE_BODY,))
    leg_cfg = SceneEntityCfg("robot", joint_names=(".*",))
    foot_cfg = SceneEntityCfg("robot", site_names=("FL", "FR", "RL", "RR"))
    cfg.rewards = {
        "track_linear_velocity": RewardTermCfg(
            func=custom_rewards.track_linear_velocity_l1,
            weight=4.5,
            params={"command_name": "twist", "std": 0.5, "asset_cfg": base_cfg},
        ),
        "track_angular_velocity": RewardTermCfg(
            func=custom_rewards.track_angular_velocity,
            weight=2.0,
            params={"command_name": "twist", "std": 0.5, "asset_cfg": base_cfg},
        ),
        "lin_vel_z": RewardTermCfg(
            func=custom_rewards.lin_vel_z_l2,
            weight=-0.5,
            params={"asset_cfg": base_cfg},
        ),
        "ang_vel_xy": RewardTermCfg(
            func=custom_rewards.body_angular_velocity_penalty,
            weight=-0.3,
            params={"asset_cfg": base_cfg},
        ),
        "roll_penalty": RewardTermCfg(
            func=custom_rewards.roll_penalty,
            weight=-1.0,
            params={"asset_cfg": base_cfg},
        ),
        "pitch_penalty": RewardTermCfg(
            func=custom_rewards.pitch_penalty,
            weight=-1.5,
            params={"max_pitch_rad": 0.50, "asset_cfg": base_cfg},
        ),
        # One-sided standing-height guard.  It does not penalize a higher base
        # on an uphill patch, but prevents the crouched/low-trunk gait seen in
        # the 2026-09-17 run.
        "low_base_height": RewardTermCfg(
            func=custom_rewards.low_base_height_penalty,
            weight=-10.0,
            params={"minimum_height": 0.27, "asset_cfg": base_cfg},
        ),
        "joint_torques": RewardTermCfg(
            func=custom_rewards.joint_torques_l2,
            weight=-1.0e-4,
            params={"asset_cfg": SceneEntityCfg("robot", actuator_names=".*")},
        ),
        "joint_acc": RewardTermCfg(
            func=custom_rewards.joint_acc_l2,
            weight=-2.5e-7,
            params={"asset_cfg": leg_cfg},
        ),
        "action_rate": RewardTermCfg(func=custom_rewards.action_rate_l2, weight=-0.005),
        "dof_pos_limits": RewardTermCfg(
            func=custom_rewards.joint_pos_limits,
            weight=-0.2,
            params={"asset_cfg": leg_cfg},
        ),
        "foot_slip": RewardTermCfg(
            func=custom_rewards.feet_slip,
            weight=-0.05,
            params={
                "sensor_name": "feet_ground_contact",
                "command_name": "twist",
                "command_threshold": 0.05,
                "asset_cfg": foot_cfg,
            },
        ),
        "soft_landing": RewardTermCfg(
            func=custom_rewards.soft_landing,
            weight=-1.0e-5,
            params={"sensor_name": "feet_ground_contact", "command_name": "twist"},
        ),
        "stand_still": RewardTermCfg(
            func=custom_rewards.zero_command_pose_penalty,
            weight=-2.0,
            params={
                "command_name": "twist",
                "command_threshold": 0.1,
                "asset_cfg": leg_cfg,
            },
        ),
        "zero_command_stillness": RewardTermCfg(
            func=custom_rewards.zero_command_stillness,
            weight=-0.5,
            params={
                "command_name": "twist",
                "command_threshold": 0.1,
                "asset_cfg": leg_cfg,
                "linear_velocity_weight": 1.0,
                "angular_velocity_weight": 0.5,
                "joint_velocity_weight": 0.02,
            },
        ),
        "feet_contact_without_cmd": RewardTermCfg(
            func=custom_rewards.feet_contact_without_cmd,
            weight=0.1,
            params={"command_name": "twist", "sensor_name": "feet_ground_contact"},
        ),
        "thigh_collision": RewardTermCfg(
            func=velocity_mdp.self_collision_cost,
            weight=-1.0,
            params={"sensor_name": "thigh_ground_touch"},
        ),
        "hip_collision": RewardTermCfg(
            func=velocity_mdp.self_collision_cost,
            weight=-0.5,
            params={"sensor_name": "hip_ground_touch"},
        ),
        "shank_collision": RewardTermCfg(
            func=custom_rewards.terrain_collision_cost,
            weight=-0.25,
            params={"sensor_name": "shank_ground_touch", "force_threshold": 25.0},
        ),
        "base_collision": RewardTermCfg(
            func=velocity_mdp.self_collision_cost,
            weight=-5.0,
            params={"sensor_name": "trunk_ground_touch"},
        ),
        "delta_swing_clearance": RewardTermCfg(
            func=custom_rewards.delta_swing_clearance_cost,
            weight=-4.0,
            params={
                "height_sensor_name": "foot_height_scan",
                "contact_sensor_name": "feet_ground_contact",
                "command_name": "twist",
                "command_threshold": 0.05,
                "minimum_clearance": 0.055,
                "obstacle_gain": 0.45,
                "max_obstacle_extra": 0.12,
                "forward_x_min": -0.25,
                "clearance_std": 0.04,
            },
        ),
        "is_terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-50.0),
    }

    cfg.terminations["terrain_edge_reached"] = TerminationTermCfg(
        func=velocity_mdp.terrain_edge_reached,
        time_out=True,
    )
    cfg.sim.nconmax = 160
    cfg.sim.njmax = 2_000
    cfg.sim.contact_sensor_maxmatch = 160
    if play:
        cfg.scene.num_envs = 1
        cfg.terminations.pop("terrain_edge_reached", None)
        cfg.curriculum = {}
    return cfg
