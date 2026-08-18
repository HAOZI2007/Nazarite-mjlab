from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.velocity.mdp.terrain_utils import terrain_normal_from_sensors
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

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
  magnitude = torch.linalg.vector_norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
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

#速度追踪奖励计算函数
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

#机身水平保持奖励计算函数
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
    self._asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)

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
      body_quat_w = asset.data.body_link_quat_w[:, self._asset_cfg.body_ids, :].squeeze(
        1
      )
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

#平地姿态保持函数
def flat_orientation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """计算 roll/pitch orientation cost; 该值为正, 配置时应使用负 reward weight."""
  asset: Entity = env.scene[asset_cfg.name]
  projected_gravity = _safe_tensor(asset.data.projected_gravity_b, limit=1.0)
  cost = torch.sum(torch.square(projected_gravity[:, :2]), dim=1)
  return _safe_tensor(cost, limit=_SAFE_REWARD_LIMIT)

#抑制 roll/pitch 机身晃动
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

#腿部软着陆奖励计算函数
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

#惩罚支撑脚打滑
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

  in_contact = _safe_tensor(contact_sensor.data.found.float(), limit=1.0).clamp(0.0, 1.0)
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

#脚部腾空时间奖励
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

#脚部腾空超时惩罚
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
  cost = torch.sum(torch.clamp(current_air_time - max(max_air_time, 0.0), min=0.0), dim=1)
  return _safe_tensor(cost, limit=_SAFE_REWARD_LIMIT)

#脚部落地检测
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

#关节角加速度限制
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

#关节扭矩限制
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

#关节位置限制
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

#动作学习率平滑
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
