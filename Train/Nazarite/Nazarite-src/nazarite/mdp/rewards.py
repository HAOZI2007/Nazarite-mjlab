from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.tasks.velocity.mdp.terrain_utils import terrain_normal_from_sensors
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_apply_yaw

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

_SAFE_STATE_LIMIT = 100.0
_SAFE_REWARD_LIMIT = 1.0e6


def _safe_tensor(value: torch.Tensor, limit: float) -> torch.Tensor:
    """安全清洗输入 tensor: 将 NaN/Inf 替换为有限值, 并限制异常大的数值."""
    return torch.nan_to_num(
        value,
        nan=0.0,
        posinf=limit,
        neginf=-limit,
    ).clamp(min=-limit, max=limit)


def _zero_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    """当数据缺失或传感器不可用时, 返回与环境 batch 匹配的安全零 reward."""
    return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)


def _safe_std(std: float) -> float:
    """保证 Gaussian reward 的 std 有限且不为 0, 避免除零和数值爆炸."""
    if not math.isfinite(std):
        return 1.0
    return max(abs(std), 1.0e-6)


def _safe_command(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor | None:
    """读取 velocity command, 并清除其中的 NaN/Inf; 找不到 command 时返回 None."""
    command = env.command_manager.get_command(command_name)
    if command is None:
        return None
    return _safe_tensor(command, _SAFE_STATE_LIMIT)


def _command_is_active(
    command: torch.Tensor,
    command_threshold: float,
) -> torch.Tensor:
    """计算每个 environment 是否存在有效的非零 velocity command."""
    threshold = max(float(command_threshold), 0.0)
    magnitude = torch.linalg.vector_norm(command[:, :2], dim=1) + torch.abs(
        command[:, 2]
    )
    return (magnitude > threshold).to(dtype=command.dtype)


def _get_contact_sensor(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> ContactSensor | None:
    """安全读取可选的 ContactSensor, 避免传感器缺失导致 reward 计算崩溃."""
    try:
        sensor = env.scene[sensor_name]
    except KeyError:
        return None
    return sensor if isinstance(sensor, ContactSensor) else None


def _support_gap_relief(
    valid: torch.Tensor,
    support: torch.Tensor,
    scale: float = 0.10,
) -> torch.Tensor:
    """Reduce a support-aware map to one gap-relief value per environment.

    Support is averaged laterally for each forward column, then the largest
    deficit across forward columns is used.  Keeping the final ``amax`` here is
    important: the clearance reward consumes one scalar per environment, while
    the intermediate support fraction naturally has shape ``[B, W]``.
    """
    support_fraction = torch.where(valid, support, torch.zeros_like(support)).mean(
        dim=-2
    )
    return (1.0 - support_fraction).clamp_min(0.0).amax(dim=-1) * float(scale)


def _expand_batch_vector(value: torch.Tensor, batch_size: int) -> torch.Tensor:
    """将共享的 3D vector 扩展成每个 environment 一份的 batch tensor."""
    if value.ndim == 1:
        return value.unsqueeze(0).expand(batch_size, -1)
    return value


def safe_height_scan(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """安全读取 terrain height scan, 并将异常值限制在合理范围内."""
    from mjlab.envs.mdp.observations import height_scan

    try:
        result = height_scan(env, sensor_name)
    except (AssertionError, KeyError, RuntimeError, ValueError):
        return torch.zeros((env.num_envs, 0), dtype=torch.float32, device=env.device)
    return _safe_tensor(result, limit=5.0)


def safe_base_lin_vel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """安全读取机器人 base 的线速度, 异常时返回有限 tensor."""
    from mjlab.envs.mdp.observations import base_lin_vel

    try:
        result = base_lin_vel(env, asset_cfg=asset_cfg)
    except (AssertionError, KeyError, RuntimeError, ValueError):
        return torch.zeros((env.num_envs, 3), dtype=torch.float32, device=env.device)
    return _safe_tensor(result, limit=100.0)


def safe_base_ang_vel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """安全读取机器人 base 的角速度, 异常时返回有限 tensor."""
    from mjlab.envs.mdp.observations import base_ang_vel

    try:
        result = base_ang_vel(env, asset_cfg=asset_cfg)
    except (AssertionError, KeyError, RuntimeError, ValueError):
        return torch.zeros((env.num_envs, 3), dtype=torch.float32, device=env.device)
    return _safe_tensor(result, limit=100.0)


def zero_command_stillness(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    command_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    linear_velocity_weight: float = 1.0,
    angular_velocity_weight: float = 1.0,
    joint_velocity_weight: float = 0.05,
) -> torch.Tensor:
    """在零速度指令下惩罚机体和关节的非必要运动.

    该函数返回正的 cost, 配置时应使用负的 reward weight。只有当
    ``command_name`` 对应的 [vx, vy, wz] 都接近零时才启用惩罚，因此不会
    把正常行走所需的机体摆动和关节运动误判为错误。

    ``asset_cfg`` 同时用于选择 base 和参与计算的关节。Go2 配置中传入
    ``joint_names=(".*",)``，即可覆盖全部关节。
    """
    command = _safe_command(env, command_name)
    if command is None:
        return _zero_reward(env)

    asset: Entity = env.scene[asset_cfg.name]
    standing_mask = 1.0 - _command_is_active(command, command_threshold)

    # 使用机体坐标系速度，和 actor/critic 观测中的本体感知定义保持一致。
    base_lin_vel = safe_base_lin_vel(env, asset_cfg)
    base_ang_vel = safe_base_ang_vel(env, asset_cfg)
    joint_vel = _safe_tensor(
        asset.data.joint_vel[:, asset_cfg.joint_ids],
        limit=_SAFE_STATE_LIMIT,
    )

    lin_cost = torch.sum(torch.square(base_lin_vel), dim=1)
    ang_cost = torch.sum(torch.square(base_ang_vel), dim=1)
    joint_cost = torch.sum(torch.square(joint_vel), dim=1)
    cost = standing_mask * (
        max(float(linear_velocity_weight), 0.0) * lin_cost
        + max(float(angular_velocity_weight), 0.0) * ang_cost
        + max(float(joint_velocity_weight), 0.0) * joint_cost
    )
    cost = _safe_tensor(cost, limit=_SAFE_REWARD_LIMIT)

    # 只统计零速度环境，避免行走环境的高速运动污染静止诊断曲线。
    standing_count = torch.clamp(torch.sum(standing_mask), min=1.0)
    if hasattr(env, "extras") and "log" in env.extras:
        env.extras["log"]["WTW/stand_base_lin_vel"] = _safe_tensor(
            torch.sum(lin_cost * standing_mask) / standing_count,
            limit=_SAFE_REWARD_LIMIT,
        )
        env.extras["log"]["WTW/stand_base_ang_vel"] = _safe_tensor(
            torch.sum(ang_cost * standing_mask) / standing_count,
            limit=_SAFE_REWARD_LIMIT,
        )
        env.extras["log"]["WTW/stand_joint_vel"] = _safe_tensor(
            torch.sum(joint_cost * standing_mask) / standing_count,
            limit=_SAFE_REWARD_LIMIT,
        )
        env.extras["log"]["WTW/stand_still_cost"] = _safe_tensor(
            torch.sum(cost) / standing_count,
            limit=_SAFE_REWARD_LIMIT,
        )

    return cost


def safe_base_height(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """安全读取 base 高度, 异常时返回与 environment batch 匹配的零 tensor."""
    try:
        asset: Entity = env.scene[asset_cfg.name]
        result = asset.data.root_link_pos_w[:, 2:3]
    except (AttributeError, AssertionError, KeyError, RuntimeError, ValueError):
        return torch.zeros((env.num_envs, 1), dtype=torch.float32, device=env.device)
    return _safe_tensor(result, limit=10.0)


def base_height_reward(
    env: ManagerBasedRlEnv,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """奖励 base 保持在目标高度附近, 防止 Go2 高速前进时过度下蹲."""
    height = safe_base_height(env, asset_cfg).squeeze(-1)
    target = target_height if math.isfinite(target_height) else 0.32
    height_error = torch.square(height - target)
    reward = torch.exp(-height_error / _safe_std(std) ** 2)
    return _safe_tensor(reward, limit=1.0)


def low_base_height_penalty(
    env: ManagerBasedRlEnv,
    minimum_height: float = 0.27,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize only a crouched base, without constraining uphill motion.

    A symmetric target-height reward is undesirable on slopes because the
    world-frame base height naturally changes with terrain elevation.  This
    one-sided term therefore activates only below ``minimum_height`` and is
    suitable for DELTA's rough-terrain curriculum.
    """
    height = safe_base_height(env, asset_cfg).squeeze(-1)
    deficit = torch.relu(float(minimum_height) - height)
    return _safe_tensor(deficit.square(), limit=_SAFE_REWARD_LIMIT)


def safe_foot_height(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """安全读取足端 foot height, 清理 NaN/Inf 并限制最大高度."""
    from mjlab.tasks.velocity.mdp.observations import foot_height

    try:
        result = foot_height(env, sensor_name)
    except (AssertionError, KeyError, RuntimeError, ValueError):
        return torch.zeros((env.num_envs, 0), dtype=torch.float32, device=env.device)
    return _safe_tensor(result, limit=1.0).clamp_min(0.0)


def safe_foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """安全读取足端 foot air time, 将负值和 NaN/Inf 清理掉."""
    from mjlab.tasks.velocity.mdp.observations import foot_air_time

    try:
        result = foot_air_time(env, sensor_name)
    except (AssertionError, KeyError, RuntimeError, ValueError):
        return torch.zeros((env.num_envs, 0), dtype=torch.float32, device=env.device)
    return _safe_tensor(result, limit=10.0).clamp_min(0.0)


def safe_foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """安全读取足端 contact flags, 并限制到 0/1 范围."""
    from mjlab.tasks.velocity.mdp.observations import foot_contact

    try:
        result = foot_contact(env, sensor_name)
    except (AssertionError, KeyError, RuntimeError, ValueError):
        return torch.zeros((env.num_envs, 0), dtype=torch.float32, device=env.device)
    return _safe_tensor(result, limit=1.0).clamp(0.0, 1.0)


def safe_foot_contact_forces(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """安全读取经过变换的 foot contact forces, 避免异常力值污染训练."""
    from mjlab.tasks.velocity.mdp.observations import foot_contact_forces

    try:
        result = foot_contact_forces(env, sensor_name)
    except (AssertionError, KeyError, RuntimeError, ValueError):
        return torch.zeros((env.num_envs, 0), dtype=torch.float32, device=env.device)
    return _safe_tensor(result, limit=100.0)


# 速度追踪奖励计算函数
def track_linear_velocity(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """计算平面线速度 tracking reward, 同时惩罚机器人上下跳动的 vertical velocity."""
    command = _safe_command(env, command_name)
    if command is None:
        return _zero_reward(env)
    actual = safe_base_lin_vel(env, asset_cfg)
    xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
    z_error = torch.square(actual[:, 2])
    reward = torch.exp(-(xy_error + z_error) / _safe_std(std) ** 2)
    return _safe_tensor(reward, limit=1.0)


def track_velocity_x(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Track the commanded forward/backward velocity independently."""
    command = _safe_command(env, command_name)
    if command is None:
        return _zero_reward(env)
    actual = safe_base_lin_vel(env, asset_cfg)
    error = torch.square(command[:, 0] - actual[:, 0])
    return _safe_tensor(torch.exp(-error / _safe_std(std) ** 2), limit=1.0)


def track_velocity_y(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Track the commanded lateral velocity independently."""
    command = _safe_command(env, command_name)
    if command is None:
        return _zero_reward(env)
    actual = safe_base_lin_vel(env, asset_cfg)
    error = torch.square(command[:, 1] - actual[:, 1])
    return _safe_tensor(torch.exp(-error / _safe_std(std) ** 2), limit=1.0)


def track_yaw_velocity(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Track commanded yaw rate independently from roll/pitch stabilization."""
    command = _safe_command(env, command_name)
    if command is None:
        return _zero_reward(env)
    actual = safe_base_ang_vel(env, asset_cfg)
    error = torch.square(command[:, 2] - actual[:, 2])
    return _safe_tensor(torch.exp(-error / _safe_std(std) ** 2), limit=1.0)


# 角度 (自转) 追踪奖励计算函数
def track_angular_velocity(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """计算 yaw rate tracking reward, 同时抑制 roll/pitch 方向的 angular motion."""
    command = _safe_command(env, command_name)
    if command is None:
        return _zero_reward(env)
    actual = safe_base_ang_vel(env, asset_cfg)
    z_error = torch.square(command[:, 2] - actual[:, 2])
    xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
    reward = torch.exp(-(z_error + xy_error) / _safe_std(std) ** 2)
    return _safe_tensor(reward, limit=1.0)


def him_track_linear_velocity(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reference Go2 HIM linear-velocity tracking reward.

    The reference weights vertical velocity twice as strongly as horizontal
    tracking error.  Keep this separate from the general Nazarite term so
    existing non-HIM tasks retain their established behavior.
    """
    command = _safe_command(env, command_name)
    if command is None:
        return _zero_reward(env)
    actual = safe_base_lin_vel(env, asset_cfg)
    xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
    z_error = torch.square(actual[:, 2])
    reward = torch.exp(-(xy_error + 2.0 * z_error) / _safe_std(std) ** 2)
    return _safe_tensor(reward, limit=1.0)


def him_track_angular_velocity(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reference Go2 HIM angular-velocity tracking reward.

    Roll/pitch angular velocity is softly regularized with the reference
    coefficient 0.05 while yaw tracking remains the primary objective.
    """
    command = _safe_command(env, command_name)
    if command is None:
        return _zero_reward(env)
    actual = safe_base_ang_vel(env, asset_cfg)
    z_error = torch.square(command[:, 2] - actual[:, 2])
    xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
    reward = torch.exp(-(z_error + 0.05 * xy_error) / _safe_std(std) ** 2)
    return _safe_tensor(reward, limit=1.0)


def current_self_collision_cost(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Return a binary self-collision cost for the current physics sample.

    Contact histories overlap between policy steps. Summing the complete history
    therefore penalizes one sustained collision repeatedly. Only the newest force
    sample is inspected here, and all matching contact pairs are reduced to one
    binary value per environment.
    """
    sensor = _get_contact_sensor(env, sensor_name)
    if sensor is None:
        return _zero_reward(env)
    data = sensor.data
    threshold = max(float(force_threshold), 0.0)
    if data.force_history is not None:
        hit = (
            torch.linalg.vector_norm(data.force_history[..., -1, :], dim=-1) > threshold
        )
    elif data.force is not None:
        hit = torch.linalg.vector_norm(data.force, dim=-1) > threshold
    elif data.found is not None:
        hit = data.found > 0
    else:
        return _zero_reward(env)
    return hit.reshape(env.num_envs, -1).any(dim=1).to(torch.float32)


def him_body_height(
    env: ManagerBasedRlEnv,
    command_name: str = "behavior",
    terrain_sensor_name: str | None = None,
    base_height_target: float = 0.32,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Track only the HIM body-height condition; no WTW gait state is used."""
    command = env.command_manager.get_command(command_name)
    if command is None or command.shape[-1] < 1:
        return _zero_reward(env)
    try:
        asset: Entity = env.scene[asset_cfg.name]
        actual = asset.data.root_link_pos_w[:, 2]
    except (AttributeError, KeyError, RuntimeError, ValueError):
        return _zero_reward(env)
    reference = torch.zeros_like(actual)
    if terrain_sensor_name is not None:
        try:
            sensor = env.scene[terrain_sensor_name]
            hit_pos = sensor.data.hit_pos_w
            hit_z = hit_pos[..., 2]
            valid = (sensor.data.distances >= 0.0) & torch.isfinite(hit_z)
            # Use the valid hit closest to the robot's horizontal position.
            # The scan covers the terrain ahead; taking its global minimum
            # would incorrectly reference a lower floor beyond a stair/bridge.
            root_xy = asset.data.root_link_pos_w[:, None, :2]
            horizontal_distance = torch.linalg.vector_norm(
                hit_pos[..., :2] - root_xy, dim=-1
            )
            horizontal_distance = torch.where(
                valid,
                horizontal_distance,
                torch.full_like(horizontal_distance, float("inf")),
            )
            center_index = horizontal_distance.argmin(dim=-1)
            reference = hit_z.gather(-1, center_index.unsqueeze(-1)).squeeze(-1)
            reference = torch.where(
                torch.isfinite(reference), reference, torch.zeros_like(reference)
            )
        except (AttributeError, KeyError, RuntimeError, ValueError):
            pass
    return _safe_tensor(
        -torch.square(actual - reference - float(base_height_target) - command[:, 0]),
        limit=_SAFE_REWARD_LIMIT,
    )


def him_stance_width(
    env: ManagerBasedRlEnv,
    command_name: str = "behavior",
    std: float = 0.035,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Track only the commanded lateral foot spacing."""
    command = env.command_manager.get_command(command_name)
    if command is None or command.shape[-1] < 2:
        return _zero_reward(env)
    try:
        asset: Entity = env.scene[asset_cfg.name]
        positions_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]
        root_pos_w = asset.data.root_link_pos_w
        root_quat_w = asset.data.root_link_quat_w
        foot_count = positions_w.shape[1]
        root_quat = root_quat_w.unsqueeze(1).expand(-1, foot_count, -1)
        positions_b = quat_apply_inverse(
            root_quat, positions_w - root_pos_w.unsqueeze(1)
        )
    except (AttributeError, KeyError, RuntimeError, ValueError):
        return _zero_reward(env)
    actual = positions_b[:, :, 1].abs()
    target = command[:, 1].unsqueeze(-1) * 0.5
    reward = torch.exp(
        -torch.square(actual - target) / max(abs(std), 1.0e-6) ** 2
    ).mean(dim=1)
    return _safe_tensor(reward, limit=1.0)


# 机身水平保持奖励计算函数
class upright:
    """计算保持机器人 base upright 的 reward.

    不提供 ``terrain_sensor_names`` 时, 相对于 world up 判断姿态, 适合 flat ground.

    提供 ``terrain_sensor_names`` 时, 相对于 terrain surface normal 判断姿态.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        """保存 terrain sensor, debug visualization 和 robot 配置."""
        self._terrain_sensor_names: tuple[str, ...] | None = cfg.params.get(
            "terrain_sensor_names"
        )
        self._debug_vis_enabled = True
        self._env = env
        self._asset_cfg: SceneEntityCfg = cfg.params.get(
            "asset_cfg", _DEFAULT_ASSET_CFG
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        std: float,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
        terrain_sensor_names: tuple[str, ...] | None = None,
    ) -> torch.Tensor:
        """根据 base 相对 world up 或 terrain normal 的倾斜程度计算 upright reward."""
        asset: Entity = env.scene[asset_cfg.name]

        if isinstance(asset_cfg.body_ids, list) and len(asset_cfg.body_ids) == 1:
            body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids[0], :]
        else:
            body_quat_w = asset.data.root_link_quat_w
        body_quat_w = _safe_tensor(body_quat_w, limit=1.0)

        terrain_sensor_names = terrain_sensor_names or self._terrain_sensor_names
        terrain_normal: torch.Tensor | None = None
        if terrain_sensor_names is not None:
            try:
                terrain_normal = terrain_normal_from_sensors(env, terrain_sensor_names)
            except (AssertionError, KeyError, RuntimeError, ValueError):
                terrain_normal = None

        if terrain_normal is not None:
            terrain_normal = _safe_tensor(terrain_normal, limit=1.0)
            terrain_normal = _expand_batch_vector(terrain_normal, env.num_envs)
            target_b = quat_apply_inverse(body_quat_w, terrain_normal)  # [B, 3]
            xy_squared = torch.sum(torch.square(target_b[:, :2]), dim=1)
        else:
            gravity_w = _safe_tensor(asset.data.gravity_vec_w, limit=1.0)
            gravity_w = _expand_batch_vector(gravity_w, env.num_envs)
            projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
            xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)

        reward = torch.exp(-xy_squared / _safe_std(std) ** 2)
        return _safe_tensor(reward, limit=1.0)

    def reset(self, env_ids: torch.Tensor) -> None:
        """重置接口; 该 reward 没有跨 step 的内部状态, 因此无需处理 env_ids."""
        del env_ids

    def debug_vis(self, visualizer: DebugVisualizer) -> None:
        """在 viewer 中绘制 terrain normal 和 robot up direction, 辅助检查姿态."""
        if not self._debug_vis_enabled or self._terrain_sensor_names is None:
            return

        env = self._env
        asset: Entity = env.scene[self._asset_cfg.name]

        env_indices = list(visualizer.get_env_indices(env.num_envs))
        if not env_indices:
            return

        terrain_normal = terrain_normal_from_sensors(env, self._terrain_sensor_names)
        if self._asset_cfg.body_ids:
            body_quat_w = asset.data.body_link_quat_w[
                :, self._asset_cfg.body_ids, :
            ].squeeze(1)
        else:
            body_quat_w = asset.data.root_link_quat_w
        up_local = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand_as(
            body_quat_w[:, :3]
        )
        body_up_w = quat_apply(body_quat_w, up_local)

        positions = asset.data.root_link_pos_w.cpu().numpy()
        offset = np.array([0.0, 0.3, 0.0])
        terrain_normal_np = terrain_normal.cpu().numpy()
        body_up_np = body_up_w.cpu().numpy()
        scale = 0.25

        for i in env_indices:
            origin = positions[i] + offset
            # Terrain normal (magenta).
            visualizer.add_arrow(
                start=origin,
                end=origin + terrain_normal_np[i] * scale,
                color=(0.8, 0.2, 0.8, 0.8),
                width=0.01,
            )
            # Body up (orange).
            visualizer.add_arrow(
                start=origin,
                end=origin + body_up_np[i] * scale,
                color=(1.0, 0.5, 0.0, 0.8),
                width=0.01,
            )


# 平地姿态保持函数
def flat_orientation_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """计算 roll/pitch orientation cost; 该值为正, 配置时应使用负 reward weight."""
    asset: Entity = env.scene[asset_cfg.name]
    projected_gravity = _safe_tensor(asset.data.projected_gravity_b, limit=1.0)
    cost = torch.sum(torch.square(projected_gravity[:, :2]), dim=1)
    return _safe_tensor(cost, limit=_SAFE_REWARD_LIMIT)


# 抑制 roll/pitch 机身晃动
def body_angular_velocity_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """计算 roll/pitch angular velocity cost; 该值为正, 配置时应使用负 reward weight."""
    asset: Entity = env.scene[asset_cfg.name]
    ang_vel = _safe_tensor(
        asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :],
        limit=_SAFE_STATE_LIMIT,
    )
    cost_per_body = torch.sum(torch.square(ang_vel[..., :2]), dim=-1)
    cost = torch.mean(cost_per_body, dim=1)
    return _safe_tensor(cost, limit=_SAFE_REWARD_LIMIT)


# 腿部软着陆奖励计算函数
def soft_landing(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str = "twist",
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """计算运动状态下首次落脚的 impact cost, 用于抑制硬着陆."""
    contact_sensor = _get_contact_sensor(env, sensor_name)
    if contact_sensor is None or contact_sensor.data.force is None:
        return _zero_reward(env)
    command = _safe_command(env, command_name)
    if command is None:
        return _zero_reward(env)

    forces = _safe_tensor(contact_sensor.data.force, limit=_SAFE_STATE_LIMIT)
    first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)
    landing_impact = torch.linalg.vector_norm(forces, dim=-1) * first_contact.float()
    cost = torch.sum(landing_impact, dim=1)
    cost *= _command_is_active(command, command_threshold)
    return _safe_tensor(cost, limit=_SAFE_REWARD_LIMIT)


# 惩罚支撑脚打滑
def feet_slip(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """计算足端接触地面时的 xy sliding cost, 用于抑制支撑脚打滑."""
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor = _get_contact_sensor(env, sensor_name)
    if contact_sensor is None or contact_sensor.data.found is None:
        return _zero_reward(env)
    command = _safe_command(env, command_name)
    if command is None:
        return _zero_reward(env)

    in_contact = _safe_tensor(contact_sensor.data.found.float(), limit=1.0).clamp(
        0.0, 1.0
    )
    foot_vel_xy = _safe_tensor(
        asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2],
        limit=_SAFE_STATE_LIMIT,
    )
    slip_speed = torch.linalg.vector_norm(foot_vel_xy, dim=-1)
    contact_mask = (in_contact > 0).float()
    cost = torch.sum(torch.square(slip_speed) * contact_mask, dim=1)
    cost *= _command_is_active(command, command_threshold)
    if hasattr(env, "extras") and "log" in env.extras:
        env.extras["log"]["Metrics/slip_velocity_mean"] = _safe_tensor(
            torch.sum(slip_speed * contact_mask)
            / torch.clamp(torch.sum(contact_mask), min=1),
            limit=_SAFE_STATE_LIMIT,
        )
    return _safe_tensor(cost, limit=_SAFE_REWARD_LIMIT)


# 脚部腾空时间奖励
def feet_air_time(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    threshold: float = 0.1,
    command_name: str | None = None,
    command_threshold: float = 0.5,
) -> torch.Tensor:
    """在足端 landing 时奖励合理的 swing phase.

    只有发生 ``first_contact`` 的 step 才会根据 ``air_time - threshold`` 产生
    reward, 避免 policy 通过长时间悬空获得虚假的 reward.
    """
    sensor = _get_contact_sensor(env, sensor_name)
    if sensor is None:
        return _zero_reward(env)
    sensor_data = sensor.data
    current_air_time = sensor_data.current_air_time
    last_air_time = sensor_data.last_air_time
    if current_air_time is None or last_air_time is None:
        return _zero_reward(env)
    current_air_time = _safe_tensor(current_air_time, limit=10.0).clamp_min(0.0)
    last_air_time = _safe_tensor(last_air_time, limit=10.0).clamp_min(0.0)
    # ``current_air_time`` is already reset to 0 on the landing step, so the
    # completed swing duration lives in ``last_air_time``.
    first_contact = sensor.compute_first_contact(dt=env.step_dt)  # [B, F]
    reward = torch.sum(
        torch.clamp(last_air_time - threshold, min=0.0) * first_contact.float(),
        dim=1,
    )
    in_air = current_air_time > 0
    num_in_air = torch.sum(in_air.float())
    mean_air_time = torch.sum(current_air_time * in_air.float()) / torch.clamp(
        num_in_air, min=1
    )
    if hasattr(env, "extras") and "log" in env.extras:
        env.extras["log"]["Metrics/air_time_mean"] = _safe_tensor(
            mean_air_time, limit=10.0
        )
    if command_name is not None:
        command = _safe_command(env, command_name)
        if command is not None:
            reward = reward * _command_is_active(command, command_threshold)
    return _safe_tensor(reward, limit=_SAFE_REWARD_LIMIT)


def delta_swing_clearance_cost(
    env: ManagerBasedRlEnv,
    height_sensor_name: str = "foot_height_scan",
    contact_sensor_name: str = "feet_ground_contact",
    command_name: str = "twist",
    command_threshold: float = 0.05,
    minimum_clearance: float = 0.055,
    obstacle_gain: float = 0.45,
    max_obstacle_extra: float = 0.12,
    forward_x_min: float = -0.25,
    min_map_valid_ratio: float = 0.35,
    min_obstacle_relief: float = 0.025,
    clearance_std: float = 0.04,
    terrain_map_cache_attr: str = "_delta_map_cache",
    map_z_scale_m: float = 0.8,
    sparse_only: bool = False,
    sparse_terrain_names: tuple[str, ...] = (
        "gaps",
        "stepping_stones",
        "grid_stones",
    ),
) -> torch.Tensor:
    """Penalize terrain-aware low feet during DELTA swing phases.

    WTW's phase-conditioned clearance term remains the main gait objective. The
    DELTA term adds a target derived from the latest local elevation map:
    ``target = minimum_clearance + obstacle_gain * local_obstacle_relief``.
    The term is one-sided and active only while a foot is airborne, so it does
    not reward hopping on flat ground.
    """
    command = _safe_command(env, command_name)
    if command is None:
        return _zero_reward(env)
    try:
        height_sensor = env.scene[height_sensor_name]
    except (AttributeError, KeyError):
        return _zero_reward(env)
    contact_sensor = _get_contact_sensor(env, contact_sensor_name)
    if not isinstance(height_sensor, TerrainHeightSensor) or contact_sensor is None:
        return _zero_reward(env)
    heights = _safe_tensor(height_sensor.data.heights, limit=1.0).clamp_min(0.0)
    if contact_sensor.data.found is None:
        return _zero_reward(env)
    swing = 1.0 - _safe_tensor(contact_sensor.data.found.float(), limit=1.0).clamp(
        0.0, 1.0
    )
    terrain_map = getattr(env, terrain_map_cache_attr, None)
    obstacle_relief = torch.zeros(
        env.num_envs, dtype=heights.dtype, device=heights.device
    )
    map_valid_ratio = torch.zeros_like(obstacle_relief)
    if (
        isinstance(terrain_map, torch.Tensor)
        and terrain_map.ndim == 4
        and terrain_map.shape[0] == env.num_envs
        and terrain_map.shape[-1] >= 3
    ):
        map_x = terrain_map[..., 0]
        map_z = terrain_map[..., 2]
        if terrain_map.shape[-1] >= 4:
            map_support = terrain_map[..., 3].clamp(0.0, 1.0)
            map_confidence = terrain_map[..., -1].clamp(0.0, 1.0)
        else:
            # Paper-compatible x/y/z maps have no support/confidence planes.
            # The dense elevation sensor provides one valid sample per cell.
            map_support = torch.ones_like(map_z)
            map_confidence = torch.ones_like(map_z)
        valid = (
            (map_confidence > 0.25)
            & (map_x >= float(forward_x_min))
            & torch.isfinite(map_z)
        )
        # Compare neighboring cells in both map axes.  The elevation sensor is
        # already in a local frame, so unlike a perspective image we can use
        # forward as well as lateral height changes.  This is important for a
        # stair riser whose height is nearly constant across the whole width.
        safe_z_min = torch.where(valid, map_z, torch.full_like(map_z, 1.0e6))
        safe_z_max = torch.where(valid, map_z, torch.full_like(map_z, -1.0e6))
        column_min = safe_z_min.amin(dim=-2)
        column_max = safe_z_max.amax(dim=-2)
        column_has_data = (column_min < 1.0e5) & (column_max > -1.0e5)
        column_relief = torch.where(
            column_has_data,
            (column_max - column_min).clamp_min(0.0),
            torch.zeros_like(column_min),
        )
        pair_x = valid[..., 1:] & valid[..., :-1]
        pair_y = valid[:, 1:, :] & valid[:, :-1, :]
        x_step = torch.where(
            pair_x,
            (map_z[..., 1:] - map_z[..., :-1]).abs(),
            torch.zeros_like(map_z[..., 1:]),
        ).amax(dim=(-1, -2))
        y_step = torch.where(
            pair_y,
            (map_z[:, 1:, :] - map_z[:, :-1, :]).abs(),
            torch.zeros_like(map_z[:, 1:, :]),
        ).amax(dim=(-1, -2))
        observed = torch.maximum(
            column_relief.amax(dim=-1), torch.maximum(x_step, y_step)
        )
        obstacle_relief = (
            observed.clamp_min(0.0) * max(float(map_z_scale_m), 1.0e-6)
        ).clamp(
            max=max(float(max_obstacle_extra), 0.0),
        )
        # A sparse terrain can have equal-height stones and still be dangerous:
        # the missing support cells are the obstacle.  Treat the per-column
        # support deficit as a bounded virtual step so the clearance term also
        # supplies a learning signal on gaps and stepping stones.
        gap_relief = _support_gap_relief(valid, map_support)
        obstacle_relief = torch.maximum(obstacle_relief, gap_relief).clamp(
            max=max(float(max_obstacle_extra), 0.0),
        )
        map_valid_ratio = map_confidence.mean(dim=(-1, -2))

    target = float(minimum_clearance) + max(float(obstacle_gain), 0.0) * obstacle_relief
    target = target.unsqueeze(-1)
    deficit = torch.relu(target - heights)
    normalized_deficit = deficit / max(_safe_std(clearance_std), 1.0e-3)
    cost = torch.mean(normalized_deficit * swing, dim=1)
    obstacle_active = (map_valid_ratio >= max(float(min_map_valid_ratio), 0.0)) & (
        obstacle_relief >= max(float(min_obstacle_relief), 0.0)
    )
    if sparse_only:
        terrain = getattr(env.scene, "terrain", None)
        generator = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
        terrain_types = getattr(terrain, "terrain_types", None)
        terrain_names = list(getattr(generator, "sub_terrains", {}).keys())
        if isinstance(terrain_types, torch.Tensor) and terrain_names:
            sparse_ids = [
                terrain_names.index(name)
                for name in sparse_terrain_names
                if name in terrain_names
            ]
            sparse_mask = torch.zeros(
                env.num_envs, dtype=torch.bool, device=terrain_types.device
            )
            if sparse_ids:
                sparse_mask = torch.isin(
                    terrain_types,
                    torch.as_tensor(sparse_ids, device=terrain_types.device),
                ).to(device=obstacle_active.device)
            else:
                sparse_mask = sparse_mask.to(device=obstacle_active.device)
            obstacle_active &= sparse_mask
    cost *= _command_is_active(command, command_threshold)
    cost *= obstacle_active.to(cost.dtype)
    if hasattr(env, "extras") and "log" in env.extras:
        env.extras["log"]["DELTA/swing_clearance_cost"] = _safe_tensor(
            cost.mean(),
            limit=_SAFE_REWARD_LIMIT,
        )
        swing_count = (swing > 0.0).to(target.dtype).sum().clamp_min(1.0)
        env.extras["log"]["DELTA/target_clearance"] = _safe_tensor(
            (target * (swing > 0.0)).sum() / swing_count,
            limit=_SAFE_REWARD_LIMIT,
        )
        env.extras["log"]["DELTA/obstacle_relief"] = _safe_tensor(
            obstacle_relief.mean(),
            limit=_SAFE_REWARD_LIMIT,
        )
        env.extras["log"]["DELTA/reward_map_valid_ratio"] = _safe_tensor(
            map_valid_ratio.mean(),
            limit=1.0,
        )
        env.extras["log"]["DELTA/obstacle_active_ratio"] = _safe_tensor(
            obstacle_active.to(cost.dtype).mean(),
            limit=1.0,
        )
        if sparse_only:
            env.extras["log"]["DELTA/sparse_terrain_active_ratio"] = _safe_tensor(
                obstacle_active.to(cost.dtype).mean(),
                limit=1.0,
            )
    return _safe_tensor(cost, limit=_SAFE_REWARD_LIMIT)


def delta_foot_support_cost(
    env: ManagerBasedRlEnv,
    support_sensor_name: str = "delta_foot_support_scan",
    contact_sensor_name: str = "feet_ground_contact",
    command_name: str = "twist",
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize contacting near an edge using simulator-only support rays.

    Four downward rays surround each foot.  A fully supported landing has four
    hits; a landing close to a gap or stepping-stone edge has fewer.  This signal
    is ground truth used only by the reward and is not part of actor observations.
    """
    command = _safe_command(env, command_name)
    contact_sensor = _get_contact_sensor(env, contact_sensor_name)
    try:
        support_sensor = env.scene[support_sensor_name]
    except (AttributeError, KeyError):
        return _zero_reward(env)
    if command is None or contact_sensor is None or contact_sensor.data.found is None:
        return _zero_reward(env)
    distances = support_sensor.data.distances
    num_frames = int(getattr(support_sensor, "num_frames", 0))
    rays_per_frame = int(getattr(support_sensor, "num_rays_per_frame", 0))
    if num_frames < 1 or rays_per_frame < 1:
        return _zero_reward(env)
    distances = distances.reshape(env.num_envs, num_frames, rays_per_frame)
    support_fraction = (distances >= 0.0).to(distances.dtype).mean(dim=-1)
    contact = _safe_tensor(contact_sensor.data.found.float(), limit=1.0).clamp(0.0, 1.0)
    cost = ((1.0 - support_fraction) * contact).mean(dim=-1)
    cost *= _command_is_active(command, command_threshold)
    if hasattr(env, "extras") and "log" in env.extras:
        env.extras["log"]["DELTA/contact_support_ratio"] = (
            support_fraction * contact
        ).sum() / contact.sum().clamp_min(1.0)
        env.extras["log"]["DELTA/foot_support_cost"] = cost.mean()
    return _safe_tensor(cost, limit=1.0)


def feet_edge(
    env: ManagerBasedRlEnv,
    support_sensor_name: str = "delta_foot_support_scan",
    contact_sensor_name: str = "feet_ground_contact",
    command_name: str = "twist",
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Named sparse-terrain alias for the simulator support-edge cost."""
    return delta_foot_support_cost(
        env,
        support_sensor_name=support_sensor_name,
        contact_sensor_name=contact_sensor_name,
        command_name=command_name,
        command_threshold=command_threshold,
    )


def _resolve_gait_offsets(
    offset: list[float] | tuple[float, ...],
    num_feet: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Resolve compact diagonal-pair offsets to the configured foot order."""
    offsets = torch.as_tensor(offset, device=device, dtype=dtype).flatten()
    if offsets.numel() == 1:
        return offsets.repeat(num_feet)
    if offsets.numel() == 2 and num_feet == 4:
        return torch.stack((offsets[0], offsets[1], offsets[1], offsets[0]))
    if offsets.numel() != num_feet:
        raise ValueError(
            f"Gait offset has {offsets.numel()} entries, but the contact sensor "
            f"has {num_feet} feet."
        )
    return offsets


def _gait_phase(env: ManagerBasedRlEnv, period: float) -> torch.Tensor:
    """Return the scalar gait phase used by observations and reward terms."""
    if period <= 0.0:
        raise ValueError("Gait period must be positive.")
    return torch.remainder(env.episode_length_buf * env.step_dt, period) / period


def feet_gait(
    env: ManagerBasedRlEnv,
    period: float,
    offset: list[float],
    threshold: float,
    command_threshold: float,
    command_name: str,
    sensor_name: str,
) -> torch.Tensor:
    """Reward phase-consistent foot contacts (HIMLoco-style).

    ``offset`` specifies the phase of each leg in cycles.  A foot is expected
    to be in contact during the first ``threshold`` fraction of its phase.
    The term is disabled for near-zero velocity commands.
    """
    sensor = _get_contact_sensor(env, sensor_name)
    if sensor is None or sensor.data.current_contact_time is None:
        return _zero_reward(env)
    phase = _gait_phase(env, period).unsqueeze(1)
    num_feet = sensor.data.current_contact_time.shape[1]
    offsets = _resolve_gait_offsets(
        offset, num_feet, device=env.device, dtype=phase.dtype
    )
    offsets = offsets.view(1, -1)
    desired_contact = ((phase + offsets) % 1.0) < threshold
    contact = sensor.data.current_contact_time > 0
    reward = (desired_contact == contact).float().mean(dim=1)
    command = _safe_command(env, command_name)
    if command is not None:
        reward *= _command_is_active(command, command_threshold)
    return _safe_tensor(reward, limit=_SAFE_REWARD_LIMIT)


def him_feet_air_time(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    gait_name: str = "foot_gait",
    command_name: str | None = None,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Reward RI-4438-style air/support timing synchronized to a gait term.

    A reward is produced only when exactly half the feet support the robot and
    the actual support pair matches the contact phase configured by
    ``gait_name``. Reading the gait parameters from the live reward config keeps
    period, offsets, and stance fraction synchronized without duplicating them.
    """
    gait = env.cfg.rewards[gait_name].params
    period = float(gait["period"])
    stance_fraction = float(gait["threshold"])
    if not 0.0 < stance_fraction < 1.0:
        raise ValueError("Gait stance threshold must be in (0, 1).")

    sensor = _get_contact_sensor(env, sensor_name)
    if sensor is None:
        return _zero_reward(env)
    air_time = sensor.data.current_air_time
    contact_time = sensor.data.current_contact_time
    if air_time is None or contact_time is None:
        return _zero_reward(env)

    phase = _gait_phase(env, period).unsqueeze(1)
    offsets = _resolve_gait_offsets(
        gait["offset"],
        contact_time.shape[1],
        device=contact_time.device,
        dtype=phase.dtype,
    ).view(1, -1)
    desired_contact = ((phase + offsets) % 1.0) < stance_fraction
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    target_time = period * (1.0 - stance_fraction)
    mode_time = in_mode_time.min(dim=1).values
    reward = torch.clamp(target_time - torch.abs(mode_time - target_time), min=0.0)
    correct_support = (in_contact == desired_contact).all(dim=1)
    half_stance = in_contact.sum(dim=1) * 2 == in_contact.shape[1]
    reward = torch.where(correct_support & half_stance, reward, 0.0)

    command_key = command_name or gait.get("command_name")
    if command_key is not None:
        command = _safe_command(env, command_key)
        if command is not None:
            reward *= _command_is_active(command, command_threshold)
    return _safe_tensor(reward, limit=_SAFE_REWARD_LIMIT)


def terrain_relative_feet_clearance(
    env: ManagerBasedRlEnv,
    target_height: float,
    gait_name: str,
    scan_sensor_name: str,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    max_height: float = 0.25,
    lookahead_distance: float = 0.25,
    path_half_width: float = 0.10,
    foot_radius: float = 0.022,
) -> torch.Tensor:
    """Track a terrain-relative swing arc with obstacle-aware lookahead.

    The desired sole height follows a sinusoidal swing profile and rises above
    upcoming terrain. Unlike velocity-weighted clearance, a blocked or dragging
    swing remains penalized even when its horizontal foot velocity is zero.
    """
    if not 0.0 <= target_height <= max_height:
        raise ValueError("Require 0 <= target_height <= max_height.")
    if lookahead_distance <= 0.0 or path_half_width <= 0.0 or foot_radius < 0.0:
        raise ValueError("Clearance scan dimensions must be positive.")

    asset: Entity = env.scene[asset_cfg.name]
    feet = asset.data.site_pos_w[:, asset_cfg.site_ids]
    gait = env.cfg.rewards[gait_name].params
    period = float(gait["period"])
    stance_fraction = float(gait["threshold"])
    if not 0.0 < stance_fraction < 1.0:
        raise ValueError("Gait stance threshold must be in (0, 1).")
    offsets = _resolve_gait_offsets(
        gait["offset"],
        feet.shape[1],
        device=feet.device,
        dtype=feet.dtype,
    ).view(1, -1)
    phase = (_gait_phase(env, period).unsqueeze(1) + offsets) % 1.0
    swing = phase >= stance_fraction
    swing_phase = ((phase - stance_fraction) / (1.0 - stance_fraction)).clamp(0.0, 1.0)
    profile = torch.sin(torch.pi * swing_phase).clamp_min(0.0)

    command = _safe_command(env, command_name)
    if command is None:
        return _zero_reward(env)
    translation = torch.cat((command[:, :2], torch.zeros_like(command[:, :1])), dim=-1)
    translation_w = quat_apply_yaw(asset.data.root_link_quat_w, translation)[:, :2]
    relative_xy = feet[..., :2] - asset.data.root_link_pos_w[:, None, :2]
    tangent = torch.stack((-relative_xy[..., 1], relative_xy[..., 0]), dim=-1)
    velocity = translation_w[:, None] + command[:, None, 2:3] * tangent
    direction = velocity / torch.linalg.vector_norm(
        velocity, dim=-1, keepdim=True
    ).clamp_min(1.0e-6)

    scan = env.scene[scan_sensor_name].data
    valid_ray = (
        (scan.distances >= 0.0)
        & torch.isfinite(scan.distances)
        & torch.isfinite(scan.hit_pos_w).all(dim=-1)
        & (scan.normals_w[..., 2] > 0.0)
    )
    hits = torch.where(valid_ray[..., None], scan.hit_pos_w, 0.0)
    delta = hits[:, None, :, :2] - feet[:, :, None, :2]
    distance_sq = (
        delta.square().sum(dim=-1).masked_fill(~valid_ray[:, None], float("inf"))
    )
    nearest_distance, nearest = distance_sq.min(dim=-1)
    valid_ground = nearest_distance <= path_half_width**2
    ground_z = hits[..., 2].gather(1, nearest)

    along = (delta * direction[:, :, None]).sum(dim=-1)
    across = delta - along[..., None] * direction[:, :, None]
    ahead = (
        valid_ray[:, None]
        & (along >= 0.0)
        & (along <= lookahead_distance)
        & (across.square().sum(dim=-1) <= path_half_width**2)
    )
    obstacle_z = (
        hits[:, None, :, 2]
        .expand_as(along)
        .masked_fill(~ahead, -float("inf"))
        .amax(dim=-1)
    )
    obstacle_z = torch.where(ahead.any(dim=-1), obstacle_z, ground_z)
    rise = (obstacle_z - ground_z).clamp_min(0.0)
    peak_height = (target_height + rise).clamp_max(max_height)
    desired_z = ground_z + foot_radius + peak_height * profile
    cost = torch.where(swing & valid_ground, torch.abs(feet[..., 2] - desired_z), 0.0)
    cost = cost.sum(dim=1) * _command_is_active(command, command_threshold)
    return _safe_tensor(cost, limit=_SAFE_REWARD_LIMIT)


# 脚部腾空超时惩罚
def prolonged_air_time(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    max_air_time: float = 0.3,
) -> torch.Tensor:
    """惩罚足端 airborne 时间超过 ``max_air_time`` 的情况.

    该 penalty 与 ``feet_air_time`` 配合, 防止 policy 让某只脚长期悬空不落地.
    """
    sensor = _get_contact_sensor(env, sensor_name)
    if sensor is None:
        return _zero_reward(env)
    sensor_data = sensor.data
    current_air_time = sensor_data.current_air_time
    if current_air_time is None:
        return _zero_reward(env)
    current_air_time = _safe_tensor(current_air_time, limit=10.0).clamp_min(0.0)
    cost = torch.sum(
        torch.clamp(current_air_time - max(max_air_time, 0.0), min=0.0), dim=1
    )
    return _safe_tensor(cost, limit=_SAFE_REWARD_LIMIT)


# 脚部落地检测
def feet_stance_contact(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    command_threshold: float = 0.05,
    force_threshold: float = 5.0,
) -> torch.Tensor:
    """在 velocity command 接近 0 时, 惩罚没有有效接触地面的足端.

    返回值是缺失有效接触的足端比例, 因此配置时应使用负 reward weight;
    运动状态下该项自动关闭, 不干扰正常 swing phase.
    """
    sensor = _get_contact_sensor(env, sensor_name)
    command = _safe_command(env, command_name)
    if sensor is None or command is None:
        return _zero_reward(env)
    force = sensor.data.force
    if force is None:
        return _zero_reward(env)

    force = _safe_tensor(force, limit=_SAFE_STATE_LIMIT)
    standing = 1.0 - _command_is_active(command, command_threshold)
    contact_force = torch.linalg.vector_norm(force, dim=-1)
    in_contact = contact_force > force_threshold
    missing_fraction = (~in_contact).float().mean(dim=1)

    if hasattr(env, "extras") and "log" in env.extras:
        env.extras["log"]["Metrics/stance_contact_fraction"] = _safe_tensor(
            in_contact.float().mean(), limit=1.0
        )
    return _safe_tensor(missing_fraction * standing, limit=_SAFE_REWARD_LIMIT)


# 关节角加速度限制
def joint_acc_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """计算关节 acceleration 的 L2 squared cost, 抑制关节加速度过大."""
    asset: Entity = env.scene[asset_cfg.name]
    joint_acc = _safe_tensor(
        asset.data.joint_acc[:, asset_cfg.joint_ids],
        limit=_SAFE_STATE_LIMIT,
    )
    return _safe_tensor(
        torch.sum(torch.square(joint_acc), dim=1),
        limit=_SAFE_REWARD_LIMIT,
    )


# 关节扭矩限制
def joint_torques_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """计算 actuator torque 的 L2 squared cost, 抑制电机输出过大."""
    asset: Entity = env.scene[asset_cfg.name]
    actuator_force = _safe_tensor(
        asset.data.actuator_force[:, asset_cfg.actuator_ids],
        limit=_SAFE_STATE_LIMIT,
    )
    return _safe_tensor(
        torch.sum(torch.square(actuator_force), dim=1),
        limit=_SAFE_REWARD_LIMIT,
    )


# 关节位置限制
def joint_pos_limits(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """计算关节超出 soft joint limits 的 cost, 防止姿态进入危险范围."""
    asset: Entity = env.scene[asset_cfg.name]
    soft_joint_pos_limits = asset.data.soft_joint_pos_limits
    if soft_joint_pos_limits is None:
        return _zero_reward(env)
    joint_pos = _safe_tensor(
        asset.data.joint_pos[:, asset_cfg.joint_ids],
        limit=_SAFE_STATE_LIMIT,
    )
    selected_limits = _safe_tensor(
        soft_joint_pos_limits[:, asset_cfg.joint_ids, :],
        limit=_SAFE_STATE_LIMIT,
    )
    lower_violation = torch.clamp(selected_limits[..., 0] - joint_pos, min=0.0)
    upper_violation = torch.clamp(joint_pos - selected_limits[..., 1], min=0.0)
    cost = torch.sum(lower_violation + upper_violation, dim=1)
    return _safe_tensor(cost, limit=_SAFE_REWARD_LIMIT)


# 动作学习率平滑
def action_rate_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
    """计算相邻 step 的 action 变化量, 用于抑制 policy 输出抖动."""
    action = _safe_tensor(env.action_manager.action, limit=_SAFE_STATE_LIMIT)
    prev_action = _safe_tensor(
        env.action_manager.prev_action,
        limit=_SAFE_STATE_LIMIT,
    )
    return _safe_tensor(
        torch.sum(torch.square(action - prev_action), dim=1),
        limit=_SAFE_REWARD_LIMIT,
    )


def action_acc_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Penalize the discrete second derivative of policy actions safely."""
    action = _safe_tensor(env.action_manager.action, limit=_SAFE_STATE_LIMIT)
    prev_action = _safe_tensor(
        env.action_manager.prev_action,
        limit=_SAFE_STATE_LIMIT,
    )
    prev_prev_action = _safe_tensor(
        env.action_manager.prev_prev_action,
        limit=_SAFE_STATE_LIMIT,
    )
    action_acc = _safe_tensor(
        action - 2.0 * prev_action + prev_prev_action,
        limit=_SAFE_STATE_LIMIT,
    )
    return _safe_tensor(
        torch.sum(torch.square(action_acc), dim=1),
        limit=_SAFE_REWARD_LIMIT,
    )


def zero_command_pose_penalty(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    command_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """在 velocity command 接近 0 时, 惩罚关节偏离初始 joint pose.

    返回值是正的 pose cost, 因此配置时应使用负 reward weight.
    参考姿态 ``default_joint_pos`` 来自 robot 的 initial state.
    """
    command = _safe_command(env, command_name)
    if command is None:
        return _zero_reward(env)

    asset: Entity = env.scene[asset_cfg.name]
    default_joint_pos = asset.data.default_joint_pos
    if default_joint_pos is None:
        return _zero_reward(env)

    joint_pos = _safe_tensor(
        asset.data.joint_pos[:, asset_cfg.joint_ids],
        limit=_SAFE_STATE_LIMIT,
    )
    default_pos = _safe_tensor(
        default_joint_pos[:, asset_cfg.joint_ids],
        limit=_SAFE_STATE_LIMIT,
    )
    pose_error = torch.mean(torch.square(joint_pos - default_pos), dim=1)
    pose_error *= 1.0 - _command_is_active(command, command_threshold)
    return _safe_tensor(pose_error, limit=_SAFE_REWARD_LIMIT)


def terrain_collision_cost(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float = 25.0,
) -> torch.Tensor:
    """Return the fraction of links with a *current* terrain collision.

    The old ``self_collision_cost`` summed all four history slots.  Because the
    history overlaps successive 20 ms policy steps, one sustained calf contact
    was counted up to four times per step, producing the abnormal
    ``Episode_Reward/shank_collision`` curve.  We intentionally inspect only
    the newest history sample and normalize by the number of links.
    """
    sensor = _get_contact_sensor(env, sensor_name)
    if sensor is None:
        return _zero_reward(env)
    data = sensor.data
    if data.force_history is not None:
        force = data.force_history[..., -1, :]
        hit = torch.linalg.vector_norm(force, dim=-1) > max(float(force_threshold), 0.0)
    elif data.force is not None:
        hit = torch.linalg.vector_norm(data.force, dim=-1) > max(
            float(force_threshold), 0.0
        )
    elif data.found is not None:
        hit = data.found > 0.0
    else:
        return _zero_reward(env)
    return _safe_tensor(hit.to(torch.float32).mean(dim=1), limit=1.0)


def body_orientation_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reference HIM cost for roll/pitch tilt (use a negative weight)."""
    return flat_orientation_l2(env, asset_cfg)


def stand_still(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize deviation from the default pose while the command is zero."""
    asset: Entity = env.scene[asset_cfg.name]
    default = asset.data.default_joint_pos
    if default is None:
        return _zero_reward(env)
    cost = torch.sum(
        torch.square(
            asset.data.joint_pos[:, asset_cfg.joint_ids]
            - default[:, asset_cfg.joint_ids]
        ),
        dim=1,
    )
    command = _safe_command(env, command_name)
    if command is not None:
        active = (
            torch.linalg.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
            <= command_threshold
        ).to(cost.dtype)
        cost = cost * active
    return _safe_tensor(cost, limit=_SAFE_REWARD_LIMIT)


def hip_joint_deviation_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reference HIM hip-centering penalty for standing/lateral commands."""
    asset: Entity = env.scene[asset_cfg.name]
    command = _safe_command(env, command_name)
    if command is None:
        return _zero_reward(env)
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice) and asset_cfg.joint_names is None:
        joint_ids, _ = asset.find_joints(r".*_hip_joint")
    default = asset.data.default_joint_pos
    if default is None:
        return _zero_reward(env)
    cost = torch.sum(
        torch.square(asset.data.joint_pos[:, joint_ids] - default[:, joint_ids]), dim=1
    )
    active = (
        (torch.abs(command[:, 1]) <= command_threshold)
        & (torch.abs(command[:, 2]) <= command_threshold)
    ).to(cost.dtype)
    return _safe_tensor(cost * active, limit=_SAFE_REWARD_LIMIT)


def track_linear_velocity_l1(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """RAIBO2-style planar velocity tracking using an L1 norm in the kernel."""
    command = _safe_command(env, command_name)
    if command is None:
        return _zero_reward(env)
    actual = safe_base_lin_vel(env, asset_cfg)
    error = torch.linalg.vector_norm(command[:, :2] - actual[:, :2], dim=1)
    reward = torch.exp(-error / max(float(std), 1.0e-6))
    return _safe_tensor(reward, limit=1.0)


def tracking_goal_vel(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    goal_command_name: str = "terrain_goal",
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    goal_distance_m: float | None = None,
) -> torch.Tensor:
    """Track velocity toward the explicit terrain goal command.

    ``goal_command_name`` points to a :class:`TerrainGoalCommand` whose world
    position is also rendered by play. ``goal_distance_m`` remains an optional
    compatibility fallback for older configurations that do not define that
    command.
    """
    command = _safe_command(env, command_name)
    if command is None:
        return _zero_reward(env)
    asset: Entity = env.scene[asset_cfg.name]
    try:
        goal_term = env.command_manager.get_term(goal_command_name)
    except KeyError:
        goal_term = None
    goal_pos_w = getattr(goal_term, "goal_pos_w", None)
    if goal_pos_w is None:
        if goal_distance_m is None:
            return _zero_reward(env)
        goal_pos_w = env.scene.env_origins.clone()
        goal_pos_w[:, 0] += max(float(goal_distance_m), 1.0e-3)
    goal_delta_w = goal_pos_w[:, :2] - asset.data.root_link_pos_w[:, :2]
    direction = torch.nn.functional.normalize(goal_delta_w, dim=-1, eps=1.0e-5)
    speed = torch.linalg.vector_norm(command[:, :2], dim=-1)
    desired_world = direction * speed.unsqueeze(-1)
    desired_body = quat_apply_inverse(
        asset.data.root_link_quat_w,
        torch.cat((desired_world, torch.zeros_like(speed.unsqueeze(-1))), dim=-1),
    )[:, :2]
    # Root-link velocity is body-frame in mjlab's entity data.
    actual = safe_base_lin_vel(env, asset_cfg)[:, :2]
    error = torch.linalg.vector_norm(actual - desired_body, dim=-1)
    reward = torch.exp(-error / _safe_std(std))
    reward *= _command_is_active(command, 0.05)
    return _safe_tensor(reward, limit=1.0)


def tracking_yaw(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    goal_command_name: str = "terrain_goal",
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    goal_distance_m: float | None = None,
) -> torch.Tensor:
    """Align body yaw with the explicit terrain goal direction."""
    command = _safe_command(env, command_name)
    if command is None:
        return _zero_reward(env)
    asset: Entity = env.scene[asset_cfg.name]
    try:
        goal_term = env.command_manager.get_term(goal_command_name)
    except KeyError:
        goal_term = None
    goal_pos_w = getattr(goal_term, "goal_pos_w", None)
    if goal_pos_w is None:
        if goal_distance_m is None:
            return _zero_reward(env)
        goal_pos_w = env.scene.env_origins.clone()
        goal_pos_w[:, 0] += max(float(goal_distance_m), 1.0e-3)
    delta = goal_pos_w[:, :2] - asset.data.root_link_pos_w[:, :2]
    goal_angle = torch.atan2(delta[:, 1], delta[:, 0])
    forward_w = quat_apply(
        asset.data.root_link_quat_w,
        torch.tensor(
            [1.0, 0.0, 0.0], device=goal_pos_w.device, dtype=goal_pos_w.dtype
        ).expand(goal_pos_w.shape[0], -1),
    )
    body_angle = torch.atan2(forward_w[:, 1], forward_w[:, 0])
    angle_error = torch.atan2(
        torch.sin(body_angle - goal_angle), torch.cos(body_angle - goal_angle)
    )
    reward = torch.exp(-torch.square(angle_error) / _safe_std(std) ** 2)
    reward *= _command_is_active(command, 0.05)
    return _safe_tensor(reward, limit=1.0)


def feet_stumble(
    env: ManagerBasedRlEnv,
    sensor_name: str = "feet_ground_contact",
    horizontal_ratio: float = 1.5,
) -> torch.Tensor:
    """Penalize foot impacts dominated by horizontal force."""
    sensor = _get_contact_sensor(env, sensor_name)
    if sensor is None:
        return _zero_reward(env)
    force = sensor.data.force
    if force is None and sensor.data.force_history is not None:
        force = sensor.data.force_history[..., -1, :]
    if force is None:
        return _zero_reward(env)
    force = _safe_tensor(force, _SAFE_STATE_LIMIT)
    horizontal = torch.linalg.vector_norm(force[..., :2], dim=-1)
    vertical = torch.abs(force[..., 2])
    return _safe_tensor(
        (horizontal > max(float(horizontal_ratio), 0.0) * vertical.clamp_min(1.0))
        .to(torch.float32)
        .mean(dim=1),
        limit=1.0,
    )


def collision(
    env: ManagerBasedRlEnv,
    sensor_names: tuple[str, ...] = (
        "hip_ground_touch",
        "thigh_ground_touch",
        "shank_ground_touch",
        "trunk_ground_touch",
    ),
    force_threshold: float = 10.0,
) -> torch.Tensor:
    """Return normalized body-terrain collision cost across selected sensors."""
    costs = []
    for sensor_name in sensor_names:
        costs.append(terrain_collision_cost(env, sensor_name, force_threshold))
    if not costs:
        return _zero_reward(env)
    return _safe_tensor(torch.stack(costs, dim=-1).mean(dim=-1), limit=1.0)


def orientation(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Roll/pitch orientation cost for sparse terrain."""
    return body_orientation_l2(env, asset_cfg)


def lin_vel_z_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Penalize vertical base velocity while allowing terrain-following pitch."""
    asset: Entity = env.scene[asset_cfg.name]
    return _safe_tensor(
        torch.square(
            _safe_tensor(asset.data.root_link_lin_vel_b[:, 2], limit=_SAFE_STATE_LIMIT)
        ),
        limit=_SAFE_REWARD_LIMIT,
    )


def roll_penalty(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Penalize roll while leaving pitch available for climbing."""
    asset: Entity = env.scene[asset_cfg.name]
    return _safe_tensor(
        torch.square(_safe_tensor(asset.data.projected_gravity_b[:, 1], limit=1.0)),
        limit=_SAFE_REWARD_LIMIT,
    )


def pitch_penalty(
    env: ManagerBasedRlEnv,
    max_pitch_rad: float = 0.50,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Dead-zone pitch penalty: normal climbing pitch is not discouraged."""
    asset: Entity = env.scene[asset_cfg.name]
    gravity_x = torch.abs(_safe_tensor(asset.data.projected_gravity_b[:, 0], limit=1.0))
    threshold = float(torch.sin(torch.tensor(max_pitch_rad)))
    return _safe_tensor(
        torch.square(torch.clamp(gravity_x - threshold, min=0.0)), limit=1.0
    )


def feet_contact_without_cmd(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
) -> torch.Tensor:
    """Reward all-foot support only for a commanded standstill."""
    contact = safe_foot_contact(env, sensor_name)
    command = _safe_command(env, command_name)
    if command is None or contact.numel() == 0:
        return _zero_reward(env)
    standing = 1.0 - _command_is_active(command, 0.1)
    return torch.sum(contact > 0.0, dim=1).float() * standing
