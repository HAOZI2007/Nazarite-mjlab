"""WTW-history + DELTA geometric-token attention task."""

from __future__ import annotations

from copy import deepcopy
import os

from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import GridPatternCfg, ObjRef, RingPatternCfg, TerrainHeightSensorCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import box_random_grid, pyramid_stairs, pyramid_stairs_inv
from nazarite.config.robot_config.go2_cfg import GO2_FOOT_SITES
from nazarite.mdp import rewards as custom_rewards
from nazarite.mdp.delta_observations import delta_privileged_terrain_map
from nazarite.delta.terrains import PillarApproachTerrainCfg

from .wtw_delta_env_cfg import _make_wtw_delta_shared_cfg


BEV_VARIANTS: dict[str, tuple[int, int]] = {
    "16x26": (16, 26),
    "25x40": (25, 40),
}
# Dense local elevation map in front of the base. The ray grid is wider than
# this ROI because GridPatternCfg is centered on the body frame.
BEV_X_RANGE = (0.0, 1.5)
BEV_Y_RANGE = (-0.5, 0.5)
BEV_MAP_EXTENT = (BEV_X_RANGE[1] - BEV_X_RANGE[0], BEV_Y_RANGE[1] - BEV_Y_RANGE[0])
BEV_MAP_CENTER = (
    0.5 * (BEV_X_RANGE[0] + BEV_X_RANGE[1]),
    0.5 * (BEV_Y_RANGE[0] + BEV_Y_RANGE[1]),
)


def Nazarite_Wtw_Delta_Go2(
    play: bool = False,
    bev_variant: str | None = None,
    camera_orientation: str | None = None,
    camera_pitch_deg: float | None = None,
):
    """Build WTW+DELTA using a local privileged elevation map.

    The camera arguments remain accepted for CLI compatibility but are ignored;
    this task is the sensor-geometry oracle experiment.
    """
    bev_variant = bev_variant or os.getenv("NAZARITE_WTW_DELTA_BEV", "25x40")
    if bev_variant not in BEV_VARIANTS:
        raise ValueError(f"Unknown BEV variant {bev_variant}; choose {tuple(BEV_VARIANTS)}")
    del camera_orientation, camera_pitch_deg

    cfg = deepcopy(_make_wtw_delta_shared_cfg(play=play))
    if not play:
        cfg.scene.num_envs = 64
        cfg.episode_length_s = 10.0

    # Keep this task focused on the four terrain families used for the
    # elevation-map experiment.  Up/down stairs are separate curriculum types
    # so the policy sees both ascent and descent geometry.
    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=deepcopy(
            cfg.scene.terrain.terrain_generator  # type: ignore[union-attr]
        ),
        max_init_terrain_level=0,
    )
    terrain_generator = cfg.scene.terrain.terrain_generator
    assert terrain_generator is not None
    terrain_generator.sub_terrains = {
        "plum_stones": PillarApproachTerrainCfg(
            proportion=0.30,
            pillar_radius_range=(0.09, 0.14),
            pillar_spacing_x=0.42,
            pillar_spacing_y=0.34,
            pillar_height=0.08,
            pillar_height_variation=0.015,
            floor_depth=1.0,
            approach_platform_length=1.0,
            approach_platform_width=1.2,
            approach_gap=0.18,
            border_width=0.25,
        ),
        "grid_rough": box_random_grid(
            proportion=0.25,
            grid_width=0.35,
            grid_height_range=(0.0, 0.14),
            platform_width=0.0,
            fill_gaps=True,
            holes=False,
        ),
        "stairs_up": pyramid_stairs(
            proportion=0.225,
            step_height_range=(0.02, 0.14),
            step_width=0.35,
            platform_width=0.8,
        ),
        "stairs_down": pyramid_stairs_inv(
            proportion=0.225,
            step_height_range=(0.02, 0.14),
            step_width=0.35,
            platform_width=0.8,
        ),
    }

    # DELTA-Go2 follows the paper's velocity-command formulation.  There is no
    # explicit target point or target-point reward in this task.
    cfg.commands.pop("terrain_goal", None)

    map_height, map_width = BEV_VARIANTS[bev_variant]
    sensors = [
        sensor for sensor in (cfg.scene.sensors or ())
        if getattr(sensor, "name", "") not in {"delta_depth_camera", "delta_privileged_scan"}
    ]
    sensors.append(
        TerrainHeightSensorCfg(
            name="delta_privileged_scan",
            frame=ObjRef(type="body", name="base_link", entity="robot"),
            pattern=GridPatternCfg(
                size=(3.0, 1.0), resolution=0.04, direction=(0.0, 0.0, -1.0)
            ),
            ray_alignment="yaw",
            max_distance=1.5,
            exclude_parent_body=True,
            include_geom_groups=(0,),
            reduction="none",
        )
    )
    sensors.append(
        TerrainHeightSensorCfg(
            name="delta_foot_support_scan",
            frame=tuple(
                ObjRef(type="site", name=site_name, entity="robot")
                for site_name in GO2_FOOT_SITES
            ),
            ray_alignment="yaw",
            max_distance=0.8,
            exclude_parent_body=True,
            include_geom_groups=(0,),
            pattern=RingPatternCfg.single_ring(radius=0.04, num_samples=4),
            reduction="none",
        )
    )
    cfg.scene.sensors = tuple(sensors)

    map_params = cfg.observations["delta_map"].terms["delta_map"].params
    map_params.update(
        {
            "map_height": map_height,
            "map_width": map_width,
            "map_channels": 3,
            "project_to_bev": False,
            "x_range_m": BEV_X_RANGE,
            "y_range_m": BEV_Y_RANGE,
            "coordinate_normalization": "paper",
            "x_scale_m": 1.25,
            "y_scale_m": 0.75,
            "z_normalization_m": 0.6,
        }
    )
    privileged_params = cfg.observations["delta_privileged_map"].terms[
        "delta_privileged_map"
    ].params
    privileged_params.update(
        {
            "map_height": map_height,
            "map_width": map_width,
            "x_range_m": BEV_X_RANGE,
            "y_range_m": BEV_Y_RANGE,
            "map_channels": 3,
            "coordinate_normalization": "paper",
        }
    )
    # Both public map groups use the same dense x/y/z elevation-map function;
    # there is deliberately no map history stacking.
    cfg.observations["delta_map"].terms["delta_map"].func = delta_privileged_terrain_map
    cfg.observations["delta_map"].terms["delta_map"].params = deepcopy(privileged_params)

    # The requested public observation contract intentionally exposes no
    # duplicate delta_proprio group. The attention model slices its query
    # context from the newest frame of wtw_proprio.
    cfg.observations = {
        "wtw_proprio": ObservationGroupCfg(
            terms=cfg.observations["wtw_proprio"].terms,
            concatenate_terms=True,
            enable_corruption=not play,
            history_length=None,
            flatten_history_dim=True,
        ),
        "critic_privileged": ObservationGroupCfg(
            terms=cfg.observations["critic_privileged"].terms,
            concatenate_terms=True,
            enable_corruption=False,
            history_length=None,
            flatten_history_dim=True,
        ),
        "delta_map": ObservationGroupCfg(
            terms=cfg.observations["delta_map"].terms,
            concatenate_terms=True,
            enable_corruption=False,
            history_length=None,
            flatten_history_dim=True,
        ),
        "delta_privileged_map": ObservationGroupCfg(
            terms=cfg.observations["delta_privileged_map"].terms,
            concatenate_terms=True,
            enable_corruption=False,
            history_length=None,
            flatten_history_dim=True,
        ),
    }

    # Use the ordinary velocity command terms again.  The WTW checkpoint
    # supplies the baseline gait; these terms only preserve command tracking.
    cfg.rewards.pop("tracking_goal_vel", None)
    cfg.rewards.pop("tracking_yaw", None)
    cfg.rewards["track_velocity_x"].weight = 1.0
    cfg.rewards["track_velocity_y"].weight = 1.0
    cfg.rewards["track_yaw_velocity"].weight = 2.0

    # Dense x/y/z elevation data now drives terrain-aware swing clearance.
    cfg.rewards["delta_swing_clearance"].weight = -0.35
    cfg.rewards["delta_swing_clearance"].params.update(
        {
            "terrain_map_cache_attr": "_delta_privileged_map_cache",
            "map_z_scale_m": 0.6,
            "sparse_only": True,
            "sparse_terrain_names": (
                "plum_stones",
                "grid_rough",
                "stairs_up",
                "stairs_down",
            ),
        }
    )

    # Keep WTW as a locomotion prior, not as a rigid flat-ground gait teacher.
    # Sparse terrain must be allowed to change swing timing and foothold
    # placement.
    wtw_scales = {
        "wtw_swing_phase_force": 0.08,
        "wtw_stance_phase_velocity": 0.08,
        "wtw_contact_schedule": 0.08,
        "wtw_group_contact_consistency": 0.05,
        "wtw_body_height": 0.15,
        "wtw_body_pitch": 0.15,
        "wtw_foot_clearance": 0.10,
        "wtw_raibert_foot_position": 0.08,
    }
    for name, scale in wtw_scales.items():
        if name in cfg.rewards:
            cfg.rewards[name].weight *= scale

    cfg.rewards["feet_edge"] = RewardTermCfg(
        func=custom_rewards.feet_edge,
        weight=-0.5,
        params={
            "support_sensor_name": "delta_foot_support_scan",
            "contact_sensor_name": "feet_ground_contact",
            "command_name": "twist",
        },
    )
    cfg.rewards["feet_stumble"] = RewardTermCfg(
        func=custom_rewards.feet_stumble,
        weight=-0.25,
        params={"sensor_name": "feet_ground_contact", "horizontal_ratio": 1.5},
    )
    cfg.rewards["collision"] = RewardTermCfg(
        func=custom_rewards.collision,
        weight=-1.0,
        params={
            "sensor_names": (
                "hip_ground_touch",
                "thigh_ground_touch",
                "shank_ground_touch",
                "trunk_ground_touch",
            ),
            "force_threshold": 10.0,
        },
    )
    cfg.rewards["orientation"] = RewardTermCfg(
        func=custom_rewards.orientation,
        weight=-0.5,
        params={},
    )
    cfg.rewards["lin_vel_z"] = RewardTermCfg(
        func=custom_rewards.lin_vel_z_l2,
        weight=-0.20,
        params={},
    )
    # ``collision`` already aggregates shank/hip/thigh/trunk contacts; keeping
    # the older standalone shank term would count the same impact twice.
    if "wtw_shank_contact" in cfg.rewards:
        cfg.rewards["wtw_shank_contact"].weight = 0.0
    # ``track_velocity_x/y`` and ``track_yaw_velocity`` are the canonical WTW
    # command-tracking terms.  Keep them active: unlike the old goal-point
    # implementation there are no duplicate tracking terms in this task.
    if "upright" in cfg.rewards:
        cfg.rewards["upright"].weight = 0.0
    if "body_ang_vel" in cfg.rewards:
        cfg.rewards["body_ang_vel"].weight = 0.0

    # DELTA paper reward composition:
    #   r = r_plus * exp(0.1 * r_minus)
    # where the negative terms attenuate, rather than linearly overwhelm, the
    # positive locomotion objective.  Failure receives the paper's -50 terminal
    # penalty; timeouts are not failures.
    cfg.reward_composition = {
        "positive_terms": (
            "track_velocity_x",
            "track_velocity_y",
            "track_yaw_velocity",
            "wtw_body_pitch",
        ),
        "negative_terms": (
            "wtw_swing_phase_force",
            "wtw_stance_phase_velocity",
            "wtw_contact_schedule",
            "wtw_group_contact_consistency",
            "wtw_body_height",
            "wtw_foot_clearance",
            "wtw_raibert_foot_position",
            "delta_swing_clearance",
            "dof_pos_limits",
            "joint_acc_l2",
            "joint_torques_l2",
            "action_rate_l2",
            "action_acc_l2",
            "foot_slip",
            "soft_landing",
            "stand_pose",
            "feet_edge",
            "feet_stumble",
            "collision",
            "orientation",
            "lin_vel_z",
        ),
        "negative_exponent": 0.1,
        "terminal_penalty": -50.0,
    }
    terrain_level = cfg.curriculum.get("terrain_levels") if cfg.curriculum else None
    if terrain_level is not None:
        terrain_level.params["sparse_terrain_names"] = (
            "plum_stones",
            "grid_rough",
            "stairs_up",
            "stairs_down",
        )
        terrain_level.params["sparse_max_level"] = 4
    twist_cfg = cfg.commands.get("twist")
    if twist_cfg is not None and hasattr(twist_cfg, "forward_only_terrain_names"):
        twist_cfg.forward_only_terrain_names = ("plum_stones",)
    if play:
        cfg.scene.num_envs = 1
        cfg.curriculum = {}
    return cfg
