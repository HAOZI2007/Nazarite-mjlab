"""Teacher-driven XZ1 depth recording; no visual input to the teacher actor."""

from dataclasses import dataclass

from mjlab.sensor import CameraSensorCfg
from nazarite.config.robot_config.xz1_cfg import XZ1_BASE_BODY
from nazarite.config.train_config.xz1.teacher_env_cfg import (
    Nazarite_WMP_HIM_Teacher_XZ1,
)


@dataclass(frozen=True)
class XZ1DepthCfg:
    # Provisional simulation mount. Calibrate against the physical XZ1 camera
    # before deployment; this is not a measured hardware extrinsic.
    pos: tuple[float, float, float] = (0.20, 0.0, 0.08)
    quat: tuple[float, float, float, float] = (0.579228, 0.405580, -0.405580, -0.579228)
    fovy: float = 58.0
    far_clip: float = 2.0
    interval: int = 5


def attach_xz1_depth(cfg, camera: XZ1DepthCfg | None = None):
    camera = camera or XZ1DepthCfg()
    if camera.interval < 1 or camera.far_clip <= 0:
        raise ValueError("Camera interval and far clip must be positive")
    cfg.scene.sensors = (
        *cfg.scene.sensors,
        CameraSensorCfg(
            name="xz1_depth_camera",
            parent_body=f"robot/{XZ1_BASE_BODY}",
            pos=camera.pos,
            quat=camera.quat,
            fovy=camera.fovy,
            width=64,
            height=64,
            data_types=("depth",),
            enabled_geom_groups=(0,),
            use_shadows=False,
            use_textures=False,
            update_interval=camera.interval,
        ),
    )
    return cfg


def Nazarite_WMP_HIM_Collect_XZ1(play: bool = False):
    cfg = attach_xz1_depth(Nazarite_WMP_HIM_Teacher_XZ1(play=play))
    cfg.scene.num_envs = 8
    # The recorder reads the terminal state before explicitly resetting.
    cfg.auto_reset = play
    return cfg
