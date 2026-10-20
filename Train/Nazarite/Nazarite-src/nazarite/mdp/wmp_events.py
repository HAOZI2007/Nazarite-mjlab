"""Reset helpers for the Go2 WMP forward obstacle course."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.events import reset_root_state_uniform
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform
from nazarite.rl.wmp.motion import MotionDataset, load_dataset

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
    env.sim.model.cam_quat[env_ids, camera_id] = quat_mul(perturbation, nominal_quat)


_MOTION_DATASETS: dict[tuple[str, str], MotionDataset] = {}


def _motion_dataset(motion_dir: str | None, device: torch.device) -> MotionDataset:
    """Cache one dataset per directory and device, keyed so CUDA runs stay separate."""
    key = (str(motion_dir), str(device))
    dataset = _MOTION_DATASETS.get(key)
    if dataset is None:
        dataset = MotionDataset(load_dataset(motion_dir).clips, device=device)
        _MOTION_DATASETS[key] = dataset
    return dataset


def reference_state_initialization(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    probability: float = 0.0,
    motion_dir: str | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Start a share of episodes mid-stride, from a reference frame.

    The reference task initialises 85% of its episodes from a clip frame so the
    policy never has to discover a gait from a standstill; discovering it is what
    took 2000 iterations and produced the crouched crawl earlier in this project.

    Registered after the course reset, this only overwrites the fields a clip
    describes: joints, joint velocity, base velocity and heading. The x/y landing
    spot stays wherever the reset put it, and z follows the terrain under it. Root
    velocities are rotated out of the clip frame, because a clip recorded heading
    -20 degrees would otherwise drive the robot sideways.
    """
    if probability <= 0.0:
        return
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if len(env_ids) == 0:
        return
    mask = torch.rand(len(env_ids), device=env.device) < probability
    if not bool(mask.any()):
        return
    ids = env_ids[mask]

    asset = env.scene[asset_cfg.name]
    states, yaw = _motion_dataset(motion_dir, env.device).sample_states(len(ids))
    joint_pos = states[:, 0:12]
    joint_vel = states[:, 18:30]
    cos, sin = torch.cos(yaw), torch.sin(yaw)
    linear = states[:, 12:15]
    angular = states[:, 15:18]
    linear_world = torch.stack(
        (
            cos * linear[:, 0] - sin * linear[:, 1],
            sin * linear[:, 0] + cos * linear[:, 1],
            linear[:, 2],
        ),
        dim=-1,
    )
    angular_world = torch.stack(
        (
            cos * angular[:, 0] - sin * angular[:, 1],
            sin * angular[:, 0] + cos * angular[:, 1],
            angular[:, 2],
        ),
        dim=-1,
    )

    root_pos = asset.data.root_link_pos_w[ids].clone()
    root_pos[:, 2] = (
        asset.data.default_root_state[ids, 2] + env.scene.env_origins[ids, 2]
    )
    root_quat = torch.zeros(len(ids), 4, device=env.device)
    root_quat[:, 0] = torch.cos(0.5 * yaw)
    root_quat[:, 3] = torch.sin(0.5 * yaw)

    # Joints first: the pose write below moves the base, which is what the clip's
    # joint angles are expressed relative to.
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=ids)
    asset.write_root_link_pose_to_sim(torch.cat((root_pos, root_quat), dim=-1), ids)
    asset.write_root_link_velocity_to_sim(
        torch.cat((linear_world, angular_world), dim=-1), ids
    )


__all__ = [
    "push_root_velocity_xy",
    "randomize_depth_camera_mount",
    "reference_state_initialization",
    "reset_root_state_uniform_facing_course",
]
