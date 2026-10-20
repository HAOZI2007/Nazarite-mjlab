"""Reset helpers for the Go2 WMP forward obstacle course."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.events import reset_root_state_uniform
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def reset_root_state_uniform_facing_course(
    env,
    env_ids: torch.Tensor | None,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]] | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Randomize the reset pose while pointing every course run along +x."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    course_pose = dict(pose_range)
    course_pose["yaw"] = (0.0, 0.0)
    reset_root_state_uniform(env, env_ids, course_pose, velocity_range, asset_cfg)


def push_root_velocity_xy(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    max_velocity: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Overwrite horizontal root velocity, matching the source push callback."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    asset = env.scene[asset_cfg.name]
    velocity = asset.data.root_link_vel_w[env_ids].clone()
    velocity[:, :2] = sample_uniform(
        -max_velocity, max_velocity, (len(env_ids), 2), device=env.device
    )
    asset.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)


@requires_model_fields("cam_pos", "cam_quat")
def randomize_depth_camera_mount(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    camera_name: str,
    position_noise: float = 0.01,
    roll_noise_degrees: float = 2.0,
    pitch_noise_degrees: float = 5.0,
    yaw_noise_degrees: float = 2.0,
) -> None:
    """Randomize the WMP camera extrinsics per environment at startup.

    The reference samples camera position within +/-1 cm, pitch within 15-25
    degrees, and roll/yaw within +/-2 degrees. Since this camera's nominal
    pitch is already 20 degrees down, the equivalent pitch perturbation here
    is +/-5 degrees around its configured quaternion.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.int)

    camera_id = env.sim.mj_model.camera(camera_name).id
    default_pos = env.sim.get_default_field("cam_pos")[camera_id].to(env.device)
    default_quat = env.sim.get_default_field("cam_quat")[camera_id].to(env.device)

    position_delta = sample_uniform(
        -position_noise,
        position_noise,
        (len(env_ids), 3),
        device=env.device,
    )
    degrees_to_radians = math.pi / 180.0
    roll = sample_uniform(
        -roll_noise_degrees * degrees_to_radians,
        roll_noise_degrees * degrees_to_radians,
        len(env_ids),
        device=env.device,
    )
    pitch = sample_uniform(
        -pitch_noise_degrees * degrees_to_radians,
        pitch_noise_degrees * degrees_to_radians,
        len(env_ids),
        device=env.device,
    )
    yaw = sample_uniform(
        -yaw_noise_degrees * degrees_to_radians,
        yaw_noise_degrees * degrees_to_radians,
        len(env_ids),
        device=env.device,
    )
    perturbation = quat_from_euler_xyz(roll, pitch, yaw)
    nominal_quat = default_quat.expand(len(env_ids), 4)

    env.sim.model.cam_pos[env_ids, camera_id] = default_pos + position_delta
    env.sim.model.cam_quat[env_ids, camera_id] = quat_mul(
        perturbation, nominal_quat
    )


__all__ = [
    "push_root_velocity_xy",
    "randomize_depth_camera_mount",
    "reset_root_state_uniform_facing_course",
]
