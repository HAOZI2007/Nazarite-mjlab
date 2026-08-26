from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from nazarite.mdp.wtw import BEHAVIOR_INDEX, WTWBehaviorCommand

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _zero_reward(env: ManagerBasedRlEnv) -> Tensor:
    """传感器缺失时返回安全的零 reward。"""
    return torch.zeros(env.num_envs, device=env.device)


def _safe(value: Tensor, limit: float = 100.0) -> Tensor:
    """清理 NaN/Inf，避免传感器异常污染训练。"""
    return torch.nan_to_num(
        value,
        nan=0.0,
        posinf=limit,
        neginf=-limit,
    ).clamp(-limit, limit)


def _get_behavior_term(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> WTWBehaviorCommand | None:
    """安全读取 WTW 行为命令。"""
    try:
        term = env.command_manager.get_term(command_name)
    except (AttributeError, KeyError, TypeError):
        return None
    return term if isinstance(term, WTWBehaviorCommand) else None


def _active_mask(
    env: ManagerBasedRlEnv,
    command_name: str,
    threshold: float,
) -> Tensor:
    """根据速度命令关闭站立环境中的行为奖励。"""
    try:
        command = env.command_manager.get_command(command_name)
    except (AttributeError, KeyError, TypeError):
        return torch.zeros(env.num_envs, device=env.device)
    if command is None:
        return torch.zeros(env.num_envs, device=env.device)
    magnitude = torch.linalg.vector_norm(command[:, :2], dim=1) + command[:, 2].abs()
    return (magnitude > max(float(threshold), 0.0)).float()


def _behavior_value(
    term: WTWBehaviorCommand,
    name: str,
) -> Tensor:
    return term.behavior[:, BEHAVIOR_INDEX[name]]


def _smooth_contact_target(
    phase: Tensor,
    smoothing: float,
) -> Tensor:
    """按照 WTW 附录生成平滑的周期性支撑目标。

    论文不是直接对 ``sin(2πt)`` 做 sigmoid，而是把相位 ``t`` 看成
    [0, 1] 内的时间变量，再用两个高斯 CDF 的乘积表示支撑区间：

      C(t) = Φ(t) [1 - Φ(t - 0.5)]
           + Φ(t - 1) [1 - Φ(t - 1.5)]

    第二项负责连接周期边界，使相位从 1 回到 0 时接触目标连续。
    返回值范围为 [0, 1]，1 表示期望支撑，0 表示期望摆动。
    """
    sigma = max(float(smoothing), 1.0e-3)
    root_two = math.sqrt(2.0)

    def normal_cdf(value: Tensor) -> Tensor:
        return 0.5 * (1.0 + torch.erf(value / (sigma * root_two)))

    return (
        normal_cdf(phase) * (1.0 - normal_cdf(phase - 0.5))
        + normal_cdf(phase - 1.0) * (1.0 - normal_cdf(phase - 1.5))
    ).clamp(0.0, 1.0)



def _contact_phase_costs(
    env: ManagerBasedRlEnv,
    sensor: ContactSensor,
    term: WTWBehaviorCommand,
    command_name: str,
    command_threshold: float,
    smoothing: float,
    force_std: float,
    velocity_std: float,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """计算论文式的摆动相接触力和支撑相足端速度代价。

    论文没有只比较 ``found`` 二值接触状态，而是把两个动力学目标分开：

    - 摆动相：足端接触力应尽量小；
    - 支撑相：足端水平速度应尽量小。

    返回值依次为摆动相代价、支撑相代价、接触时序诊断误差、有效运动掩码
    和四条腿分别的摆动相接触力代价。
    """
    desired_contact = _smooth_contact_target(term.phase, smoothing)
    swing_mask = 1.0 - desired_contact
    active = _active_mask(env, command_name, command_threshold)

    force = sensor.data.force
    if force is None:
        force_cost = torch.zeros_like(desired_contact)
    else:
        force_magnitude = torch.linalg.vector_norm(_safe(force, limit=100.0), dim=-1)
        # 与官方 CoRLRewards 一致，采用饱和型代价，而不是未饱和的
        # 平方代价。这样大碰撞力不会把辅助奖励瞬间压到几乎为零。
        # force_std 在这里对应官方的 gait_force_sigma。
        force_cost = 1.0 - torch.exp(
            -torch.square(force_magnitude) / max(abs(force_std), 1.0e-6)
        )

    asset = env.scene["robot"]
    foot_ids, _ = asset.find_sites(("FL", "FR", "RL", "RR"), preserve_order=True)
    foot_velocity_w = _safe(asset.data.site_lin_vel_w[:, foot_ids, :], limit=100.0)
    foot_velocity_xy = torch.linalg.vector_norm(foot_velocity_w[:, :, :2], dim=-1)
    # velocity_std 对应官方的 gait_vel_sigma，而不是高斯标准差。
    velocity_cost = 1.0 - torch.exp(
        -torch.square(foot_velocity_xy) / max(abs(velocity_std), 1.0e-6)
    )

    # 保留逐腿代价，供 TensorBoard 诊断前后腿或左右腿的不对称问题。
    # 当前腿顺序固定为 [FL, FR, RL, RR]。
    swing_force_cost_per_foot = force_cost * swing_mask
    swing_cost = swing_force_cost_per_foot.mean(dim=1) * active
    stance_cost = (velocity_cost * desired_contact).mean(dim=1) * active

    if sensor.data.found is None:
        schedule_error = torch.zeros(env.num_envs, device=env.device)
    else:
        actual_contact = (sensor.data.found > 0).float()
        schedule_error = torch.abs(actual_contact - desired_contact).mean(dim=1)

    return (
        _safe(swing_cost, limit=100.0),
        _safe(stance_cost, limit=100.0),
        _safe(schedule_error, limit=1.0),
        active,
        _safe(swing_force_cost_per_foot, limit=100.0),
    )


def wtw_swing_phase_contact_cost(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    behavior_command_name: str = "behavior",
    command_name: str = "twist",
    command_threshold: float = 0.05,
    smoothing: float = 0.15,
    force_std: float = 10.0,
) -> Tensor:
    """返回摆动相足端接触力代价，作为独立 RewardManager 项使用。"""
    term = _get_behavior_term(env, behavior_command_name)
    try:
        sensor = env.scene[sensor_name]
    except (AttributeError, KeyError):
        return _zero_reward(env)
    if term is None or not isinstance(sensor, ContactSensor):
        return _zero_reward(env)
    (
        swing_cost,
        _stance_cost,
        _schedule_error,
        active,
        swing_force_cost_per_foot,
    ) = _contact_phase_costs(
        env,
        sensor,
        term,
        command_name,
        command_threshold,
        smoothing,
        force_std,
        velocity_std=1.0,
    )

    if hasattr(env, "extras") and "log" in env.extras:
        active_count = active.sum()
        if active_count > 0.0:
            swing_cost_mean = (swing_cost * active).sum() / active_count
            swing_force_cost_per_foot_mean = (
                swing_force_cost_per_foot * active.unsqueeze(-1)
            ).sum(dim=0) / active_count
        else:
            swing_cost_mean = torch.zeros((), device=env.device)
            swing_force_cost_per_foot_mean = torch.zeros(
                4, device=env.device
            )

        # 独立奖励同时写入诊断指标，便于在 TensorBoard 中观察每条腿的表现。
        env.extras["log"]["WTW/swing_phase_force_cost"] = swing_cost_mean
        env.extras["log"]["WTW/swing_phase_force_score"] = 1.0 - swing_cost_mean
        env.extras["log"]["WTW/foot_FL_swing_force"] = (
            swing_force_cost_per_foot_mean[0]
        )
        env.extras["log"]["WTW/foot_FR_swing_force"] = (
            swing_force_cost_per_foot_mean[1]
        )
        env.extras["log"]["WTW/foot_RL_swing_force"] = (
            swing_force_cost_per_foot_mean[2]
        )
        env.extras["log"]["WTW/foot_RR_swing_force"] = (
            swing_force_cost_per_foot_mean[3]
        )

    return swing_cost


def wtw_stance_phase_velocity_cost(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    behavior_command_name: str = "behavior",
    command_name: str = "twist",
    command_threshold: float = 0.05,
    smoothing: float = 0.15,
    velocity_std: float = 0.5,
) -> Tensor:
    """返回支撑相足端水平速度代价，作为独立 RewardManager 项使用。"""
    term = _get_behavior_term(env, behavior_command_name)
    try:
        sensor = env.scene[sensor_name]
    except (AttributeError, KeyError):
        return _zero_reward(env)
    if term is None or not isinstance(sensor, ContactSensor):
        return _zero_reward(env)
    (
        _swing_cost,
        stance_cost,
        _schedule_error,
        active,
        _swing_force_cost_per_foot,
    ) = _contact_phase_costs(
        env,
        sensor,
        term,
        command_name,
        command_threshold,
        smoothing,
        force_std=10.0,
        velocity_std=velocity_std,
    )

    if hasattr(env, "extras") and "log" in env.extras:
        active_count = active.sum()
        if active_count > 0.0:
            stance_cost_mean = (stance_cost * active).sum() / active_count
        else:
            stance_cost_mean = torch.zeros((), device=env.device)
        env.extras["log"][
            "WTW/stance_phase_velocity_cost"
        ] = stance_cost_mean
        env.extras["log"][
            "WTW/stance_phase_velocity_score"
        ] = 1.0 - stance_cost_mean

    return stance_cost


def wtw_body_height(
    env: ManagerBasedRlEnv,
    behavior_command_name: str = "behavior",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.035,
    command_name: str = "twist",
    command_threshold: float = 0.05,
) -> Tensor:
    """奖励机身高度跟随行为参数 hz。"""
    term = _get_behavior_term(env, behavior_command_name)
    if term is None:
        return _zero_reward(env)
    try:
        asset: Entity = env.scene[asset_cfg.name]
        actual = asset.data.root_link_pos_w[:, 2]
    except (AttributeError, KeyError, RuntimeError, ValueError):
        return _zero_reward(env)

    target = _behavior_value(term, "body_height")
    reward = torch.exp(-torch.square(actual - target) / max(abs(std), 1.0e-6) ** 2)

    if hasattr(env, "extras") and "log" in env.extras:
        active = _active_mask(env, command_name, command_threshold)
        active_count = active.sum()
        if active_count > 0.0:
            env.extras["log"]["WTW/body_height_actual"] = (
                (actual * active).sum() / active_count
            )
            env.extras["log"]["WTW/body_height_error"] = (
                (torch.abs(actual - target) * active).sum() / active_count
            )
            env.extras["log"]["WTW/body_height_min"] = torch.amin(
                torch.where(
                    active > 0.0,
                    actual,
                    torch.full_like(actual, float("inf")),
                )
            )
        else:
            env.extras["log"]["WTW/body_height_actual"] = torch.zeros(
                (), device=env.device
            )
            env.extras["log"]["WTW/body_height_error"] = torch.zeros(
                (), device=env.device
            )
            env.extras["log"]["WTW/body_height_min"] = torch.zeros(
                (), device=env.device
            )

    return _safe(reward, limit=1.0)


def wtw_body_pitch(
    env: ManagerBasedRlEnv,
    behavior_command_name: str = "behavior",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.08,
    command_name: str = "twist",
    command_threshold: float = 0.05,
) -> Tensor:
    """奖励机身俯仰角跟随行为参数 phi。"""
    term = _get_behavior_term(env, behavior_command_name)
    if term is None:
        return _zero_reward(env)
    try:
        asset: Entity = env.scene[asset_cfg.name]
        gravity_b = asset.data.projected_gravity_b
    except (AttributeError, KeyError, RuntimeError, ValueError):
        return _zero_reward(env)

    # 对小角度姿态，重力在 body x 方向的分量可近似表示 pitch。
    # gravity_b 的竖直分量在直立时约为 -1，因此分母要取其相反数。
    pitch = torch.atan2(
        gravity_b[:, 0],
        (-gravity_b[:, 2]).clamp_min(1.0e-6),
    )
    target = _behavior_value(term, "body_pitch")
    reward = torch.exp(-torch.square(pitch - target) / max(abs(std), 1.0e-6) ** 2)
    reward *= _active_mask(env, command_name, command_threshold)
    return _safe(reward, limit=1.0)


class wtw_foot_swing_height:
    """在每次落脚时奖励摆动阶段的峰值足端高度。"""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        height_sensor = env.scene[cfg.params["height_sensor_name"]]
        if not isinstance(height_sensor, TerrainHeightSensor):
            raise TypeError("wtw_foot_swing_height requires a TerrainHeightSensor.")
        self.peak_heights = torch.zeros(
            env.num_envs,
            height_sensor.num_frames,
            device=env.device,
        )
        self.step_dt = env.step_dt

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        height_sensor_name: str,
        behavior_command_name: str = "behavior",
        command_name: str = "twist",
        command_threshold: float = 0.05,
        std: float = 0.04,
        smoothing: float = 0.15,
    ) -> Tensor:
        term = _get_behavior_term(env, behavior_command_name)
        if term is None:
            return _zero_reward(env)
        try:
            contact_sensor = env.scene[sensor_name]
            height_sensor = env.scene[height_sensor_name]
        except (AttributeError, KeyError):
            return _zero_reward(env)
        if not isinstance(contact_sensor, ContactSensor):
            return _zero_reward(env)
        if not isinstance(height_sensor, TerrainHeightSensor):
            return _zero_reward(env)
        if contact_sensor.data.found is None:
            return _zero_reward(env)

        heights = _safe(height_sensor.data.heights, limit=1.0).clamp_min(0.0)
        desired_swing = 1.0 - _smooth_contact_target(term.phase, smoothing)
        self.peak_heights = torch.where(
            desired_swing > 0.5,
            torch.maximum(self.peak_heights, heights),
            self.peak_heights,
        )

        first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
        target = _behavior_value(term, "foot_swing_height").unsqueeze(-1)
        error = torch.square((self.peak_heights - target) / max(abs(std), 1.0e-6))
        reward = torch.exp(-error) * first_contact.float()
        reward = reward.mean(dim=1) * _active_mask(env, command_name, command_threshold)
        # 先记录落脚瞬间的峰值误差，再清零缓存；否则日志永远接近零。
        if hasattr(env, "extras") and "log" in env.extras:
            env.extras["log"]["WTW/swing_height_error"] = torch.mean(
                torch.abs(self.peak_heights - target) * first_contact.float()
            )
        self.peak_heights = torch.where(
            first_contact,
            torch.zeros_like(self.peak_heights),
            self.peak_heights,
        )
        return _safe(reward, limit=1.0)

    def reset(self, env_ids: Tensor) -> None:
        self.peak_heights[env_ids] = 0.0


def wtw_stance_width(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    behavior_command_name: str = "behavior",
    command_name: str = "twist",
    command_threshold: float = 0.05,
    std: float = 0.035,
) -> Tensor:
    """奖励足端横向站姿宽度跟随 sy，使用较小权重避免限制转弯。"""
    term = _get_behavior_term(env, behavior_command_name)
    if term is None:
        return _zero_reward(env)
    try:
        asset: Entity = env.scene[asset_cfg.name]
        positions_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]
        root_pos_w = asset.data.root_link_pos_w
        root_quat_w = asset.data.root_link_quat_w
    except (AttributeError, KeyError, RuntimeError, ValueError):
        return _zero_reward(env)

    from mjlab.utils.lab_api.math import quat_apply_inverse

    # quat_apply_inverse 会把输入展平为 [B*F, ...]，因此必须把每个
    # environment 的 root quaternion 显式扩展到每一只脚，不能只 unsqueeze。
    foot_count = positions_w.shape[1]
    root_quat_per_foot = root_quat_w.unsqueeze(1).expand(-1, foot_count, -1)
    positions_b = quat_apply_inverse(
        root_quat_per_foot,
        positions_w - root_pos_w.unsqueeze(1),
    )
    actual_width = positions_b[:, :, 1].abs()
    target_width = _behavior_value(term, "stance_width").unsqueeze(-1) * 0.5
    reward = torch.exp(
        -torch.square(actual_width - target_width) / max(abs(std), 1.0e-6) ** 2
    ).mean(dim=1)
    reward *= _active_mask(env, command_name, command_threshold)
    return _safe(reward, limit=1.0)


class wtw_raibert_foot_position:
    """用简化 Raibert 启发式约束摆动期的足端水平落点。

    仅奖励固定的左右距离会在快速转弯时限制足端的横向调整。这里以
    reset 后记录的足端名义位置为基准，再加入三项补偿：

    1. ``stance_width`` 决定左右方向的基础落点；
    2. 速度误差和步频决定前后/侧向落点修正；
    3. 偏航角速度通过刚体旋转项修正左右足的落点。

    这是适配当前 Nazarite 的工程化简化版本，不会替代完整的全身动力学
    控制器。只在期望摆动相激活，避免在支撑相用足端位置奖励限制身体运动。
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
        asset_cfg = cfg.params["asset_cfg"]
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError("wtw_raibert_foot_position requires an asset_cfg.")
        num_sites = len(asset_cfg.site_names or ())
        if num_sites == 0:
            raise ValueError("wtw_raibert_foot_position requires site_names.")

        self._asset_cfg = asset_cfg
        self._nominal_pos_b = torch.zeros(
            env.num_envs,
            num_sites,
            2,
            device=env.device,
        )
        self._initialized = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg,
        behavior_command_name: str = "behavior",
        command_name: str = "twist",
        command_threshold: float = 0.05,
        std: float = 0.08,
        smoothing: float = 0.15,
    ) -> Tensor:
        term = _get_behavior_term(env, behavior_command_name)
        if term is None:
            return _zero_reward(env)

        try:
            asset: Entity = env.scene[asset_cfg.name]
            positions_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]
            root_pos_w = asset.data.root_link_pos_w
            root_quat_w = asset.data.root_link_quat_w
            command = env.command_manager.get_command(command_name)
        except (AttributeError, KeyError, RuntimeError, ValueError):
            return _zero_reward(env)
        if command is None:
            return _zero_reward(env)

        from mjlab.utils.lab_api.math import quat_apply_inverse

        foot_count = positions_w.shape[1]
        root_quat_per_foot = root_quat_w.unsqueeze(1).expand(-1, foot_count, -1)
        positions_b = quat_apply_inverse(
            root_quat_per_foot,
            positions_w - root_pos_w.unsqueeze(1),
        )[:, :, :2]

        # 第一次调用通常发生在 reset 后；记录当前形态的足端名义位置。
        # detach 保证这个缓存只作为目标参考，不参与任何 autograd 图。
        new_envs = ~self._initialized
        if new_envs.any():
            self._nominal_pos_b[new_envs] = positions_b[new_envs].detach()
            self._initialized[new_envs] = True

        nominal = self._nominal_pos_b
        stance_width = _behavior_value(term, "stance_width").unsqueeze(-1)
        nominal_y_sign = torch.where(nominal[:, :, 1] >= 0.0, 1.0, -1.0)
        target = nominal.clone()
        target[:, :, 1] = nominal_y_sign * stance_width * 0.5

        # 半个步态周期是典型的支撑/摆动时间尺度。速度误差越大，
        # 足端目标越向前方或侧方修正；这样转弯时不会被固定站姿锁死。
        frequency = _behavior_value(term, "frequency").clamp_min(0.1)
        half_cycle = (0.5 / frequency).unsqueeze(-1)
        velocity_error = command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]
        target += velocity_error.unsqueeze(1) * half_cycle.unsqueeze(-1)

        # 偏航运动的平面速度为 ω × p = [-ω*y, ω*x]。
        yaw_rate = command[:, 2].unsqueeze(-1).unsqueeze(-1)
        yaw_velocity = torch.stack(
            (-nominal[:, :, 1], nominal[:, :, 0]),
            dim=-1,
        )
        target += yaw_rate * yaw_velocity * half_cycle.unsqueeze(-1)

        desired_contact = _smooth_contact_target(term.phase, smoothing)
        swing_mask = 1.0 - desired_contact
        error = torch.sum(torch.square(positions_b - target), dim=-1)
        reward = torch.exp(-error / max(abs(std), 1.0e-6) ** 2)
        reward = (reward * swing_mask).mean(dim=1)
        reward *= _active_mask(env, command_name, command_threshold)
        return _safe(reward, limit=1.0)

    def reset(self, env_ids: Tensor | slice | None) -> None:
        """让部分 reset 环境在下一次调用时重新记录名义足端位置。"""
        if env_ids is None:
            self._initialized[:] = False
        else:
            self._initialized[env_ids] = False
