"""Isolated direct-action WTW+DELTA environment."""

from copy import deepcopy

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import (
    CameraSensorCfg,
    GridPatternCfg,
    ObjRef,
    RayCastSensorCfg,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from nazarite.config.robot_config.go2_cfg import GO2_FOOT_SITES
from nazarite.delta.d435 import (
    DELTA_DIRECT_CAMERA_PITCH_DEG,
    DELTA_DIRECT_CAMERA_POS_B,
    DELTA_DIRECT_CAMERA_QUAT,
    DELTA_DIRECT_CAMERA_ROLL_DEG,
    DELTA_DIRECT_D435_INTRINSICS,
    DELTA_DIRECT_DEPTH_HEIGHT,
    DELTA_DIRECT_DEPTH_WIDTH,
)
from nazarite.mdp import rewards as custom_rewards
from nazarite.mdp.commands import TerrainConditionedVelocityCommandCfg

from .wtw_delta_env_cfg import _make_wtw_delta_shared_cfg


def Nazarite_Wtw_Delta_Direct_Go2(play: bool = False):
    """Build a deployment-matched WTW+DELTA task without distillation."""
    cfg = deepcopy(_make_wtw_delta_shared_cfg(play=play))

    if not play:
        # Direct training omits student-map history and distillation state. The
        # critic retains one clean terrain scan, but the rollout still fits a
        # substantially larger batch than the former teacher-student task.
        # Reduce this target at launch if the GPU cannot hold the camera buffers.
        cfg.scene.num_envs = 512

    # A single front camera cannot observe terrain behind the robot. Keep all
    # sparse-support courses positive-forward while continuous terrain retains
    # the original omnidirectional command schedule.
    twist_cfg = cfg.commands["twist"]
    assert isinstance(twist_cfg, TerrainConditionedVelocityCommandCfg)
    twist_cfg.forward_only_terrain_names = (
        "gaps",
        "stepping_stones",
        "grid_stones",
    )

    terrain_cfg = cfg.scene.terrain
    assert terrain_cfg is not None
    terrain_generator = terrain_cfg.terrain_generator
    assert terrain_generator is not None
    terrain_generator.sub_terrains["gaps"].proportion = 0.10
    terrain_generator.sub_terrains["stepping_stones"].proportion = 0.15
    terrain_generator.sub_terrains["grid_stones"].proportion = 0.10
    for terrain_name in ("hills", "slope", "stairs", "steps", "rough_ground"):
        terrain_generator.sub_terrains[terrain_name].proportion = 0.13

    # Use the higher-resolution D435 stream only for this task.  The camera
    # FOV and physical intrinsics remain unchanged; the pinhole parameters are
    # rotated/scaled from the supplied 848x480 calibration to 72x128 portrait.
    sensors = list(cfg.scene.sensors or ())
    delta_camera = next(
        sensor
        for sensor in sensors
        if getattr(sensor, "name", "") == "delta_depth_camera"
    )
    assert isinstance(delta_camera, CameraSensorCfg)
    delta_camera.width = DELTA_DIRECT_DEPTH_WIDTH
    delta_camera.height = DELTA_DIRECT_DEPTH_HEIGHT
    # MuJoCo camera quaternions are [w, x, y, z]. The calibrated pose maps
    # -Z to robot-forward (+X), +X to robot-right (-Y), and +Y upward (+Z).
    # Thus the long portrait image axis is vertical in the robot frame.
    delta_camera.pos = DELTA_DIRECT_CAMERA_POS_B
    delta_camera.quat = DELTA_DIRECT_CAMERA_QUAT
    delta_camera.focal_length_px = (
        DELTA_DIRECT_D435_INTRINSICS.fx,
        DELTA_DIRECT_D435_INTRINSICS.fy,
    )
    delta_camera.principal_point_px = (
        DELTA_DIRECT_D435_INTRINSICS.cx,
        DELTA_DIRECT_D435_INTRINSICS.cy,
    )
    cfg.scene.sensors = tuple(sensors)

    # The deployed actor receives the current camera map. During training only,
    # the asymmetric critic receives a clean raycast map and uncorrupted
    # proprioception; privileged terrain never enters the actor action path.
    cfg.observations = {
        "wtw_proprio": cfg.observations["wtw_proprio"],
        "critic_privileged": cfg.observations["critic_privileged"],
        "delta_proprio": cfg.observations["delta_proprio"],
        "delta_proprio_critic": cfg.observations["delta_proprio_critic"],
        "delta_map": cfg.observations["delta_map"],
        "delta_privileged_map": cfg.observations["delta_privileged_map"],
    }

    # DELTA does not observe privileged base linear velocity.  Five deployment-
    # compatible proprioceptive frames provide angular/joint/action/contact-phase
    # motion context so the terrain query can distinguish slow and fast approach.
    # WTW keeps its own original observation history unchanged.
    for group_name in ("delta_proprio", "delta_proprio_critic"):
        for term in cfg.observations[group_name].terms.values():
            term.history_length = 5
            term.flatten_history_dim = True

    # Direct-action training uses the raw observed channel to preserve gaps and
    # stepping-stone voids. A small 3x3 fill only bridges isolated renderer
    # holes; the previous 7x7 fill could erase a narrow physical gap. The
    # near-field ROI improves pixel density before we expand the range again.
    delta_map_term = cfg.observations["delta_map"].terms["delta_map"]
    delta_map_term.params.update(
        {
            "x_range_m": (0.0, 2.0),
            "y_range_m": (-0.6, 0.6),
            "fill_kernel_size": 3,
            "camera_pos_b": DELTA_DIRECT_CAMERA_POS_B,
            "camera_pitch_deg": DELTA_DIRECT_CAMERA_PITCH_DEG,
            "camera_roll_deg": DELTA_DIRECT_CAMERA_ROLL_DEG,
        }
    )

    # Align the clean critic map with the actor map: 26 forward samples at
    # 8 cm spacing cover x=[0, 2.0], while 16 lateral samples cover y=[-0.6, 0.6].
    privileged_sensor = next(
        sensor
        for sensor in sensors
        if getattr(sensor, "name", "") == "delta_privileged_scan"
    )
    assert isinstance(privileged_sensor, RayCastSensorCfg)
    privileged_sensor.pattern = GridPatternCfg(
        size=(4.0, 1.2), resolution=0.08, direction=(0.0, 0.0, -1.0)
    )
    privileged_term = cfg.observations["delta_privileged_map"].terms[
        "delta_privileged_map"
    ]
    privileged_term.params.update(
        {
            "x_range_m": (0.0, 2.0),
            "y_range_m": (-0.6, 0.6),
            "scan_width": 51,
        }
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

    cfg.rewards["delta_swing_clearance"].params["terrain_map_cache_attr"] = (
        "_delta_privileged_map_cache"
    )
    cfg.rewards["delta_foot_support"] = RewardTermCfg(
        func=custom_rewards.delta_foot_support_cost,
        weight=-2.0,
        params={
            "support_sensor_name": "delta_foot_support_scan",
            "contact_sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "command_threshold": 0.05,
        },
    )
    if not play:
        cfg.curriculum["terrain_levels"].params.update(
            {
                "adaptive_type_sampling": True,
                "adaptive_sampling_floor": 0.25,
                "adaptive_sampling_temperature": 1.0,
                "adaptive_warmup_episodes": 20,
                "type_success_ema_alpha": 0.05,
            }
        )

    # No camera-map history or distillation model is constructed. The actor uses
    # the current D435 map, while the privileged raycast remains critic/reward-only.
    if play:
        cfg.scene.num_envs = 1
        cfg.curriculum = {}
    return cfg
