"""WTW velocity environment for the XZ1 quadruped."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactSensorCfg, ObjRef, TerrainHeightSensorCfg
from nazarite.config.robot_config.xz1_cfg import (
    XZ1_ACTION_SCALE,
    XZ1_BASE_BODY,
    XZ1_CALF_BODIES,
    XZ1_FOOT_BODIES,
    XZ1_FOOT_GEOMS,
    XZ1_FOOT_SITES,
    XZ1_HIP_BODIES,
    XZ1_THIGH_BODIES,
    get_xz1_cfg,
)
from nazarite.config.train_config.env_cfgs.go2_env_cfgs import (
    Nazarite_Velocity_Flat_Go2_WTW,
)


def Nazarite_Velocity_Flat_XZ1_WTW(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create the WTW task with the XZ1 asset and its entity names.

    The WTW observation, command, reward, curriculum, and termination logic is
    shared with the Go2 task. Only robot-specific selectors and the nominal body
    height are overridden here.
    """
    cfg = Nazarite_Velocity_Flat_Go2_WTW(play=play)
    cfg.scene.num_envs = 2048
    if play:
        cfg.scene.num_envs = 1
    cfg.scene.entities = {"robot": get_xz1_cfg()}
    # XZ1's dense collision meshes produce more simultaneous contacts than the
    # Go2 WTW default (35), so reserve a larger per-world contact buffer.
    cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 128)

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = XZ1_ACTION_SCALE

    # Replace the Go2-specific sensor selectors while preserving the WTW sensor
    # contracts and history settings.
    for sensor in cfg.scene.sensors or ():
        if isinstance(sensor, ContactSensorCfg):
            sensor.preserve_order = True
            if sensor.name == "feet_ground_contact":
                sensor.primary.pattern = XZ1_FOOT_GEOMS
            elif sensor.name == "hip_ground_touch":
                sensor.primary.pattern = XZ1_HIP_BODIES
            elif sensor.name == "thigh_ground_touch":
                sensor.primary.pattern = XZ1_THIGH_BODIES
            elif sensor.name == "shank_ground_touch":
                sensor.primary.pattern = XZ1_CALF_BODIES
            elif sensor.name == "trunk_ground_touch":
                sensor.primary.pattern = (XZ1_BASE_BODY,)
        elif (
            isinstance(sensor, TerrainHeightSensorCfg)
            and sensor.name == "foot_height_scan"
        ):
            sensor.frame = tuple(
                ObjRef(type="site", name=site_name, entity="robot")
                for site_name in XZ1_FOOT_SITES
            )

    # Domain randomization selectors.
    cfg.events["base_com"].params["asset_cfg"].body_names = (XZ1_BASE_BODY,)
    cfg.events["base_mass"].params["asset_cfg"].body_names = (XZ1_BASE_BODY,)
    cfg.events["link_mass"].params["asset_cfg"].body_names = (
        XZ1_HIP_BODIES + XZ1_THIGH_BODIES + XZ1_CALF_BODIES + XZ1_FOOT_BODIES
    )

    # WTW body-height targets are absolute heights. XZ1's nominal standing
    # height is 0.30 m rather than Go2's 0.32 m.
    cfg.rewards["wtw_body_height"].params["base_height_target"] = 0.30
    cfg.rewards["upright"].params["asset_cfg"].body_names = (XZ1_BASE_BODY,)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (XZ1_BASE_BODY,)
    cfg.rewards["wtw_raibert_foot_position"].params[
        "asset_cfg"
    ].site_names = XZ1_FOOT_SITES
    cfg.rewards["foot_slip"].params["asset_cfg"].site_names = XZ1_FOOT_SITES

    cfg.viewer.body_name = XZ1_BASE_BODY
    return cfg
