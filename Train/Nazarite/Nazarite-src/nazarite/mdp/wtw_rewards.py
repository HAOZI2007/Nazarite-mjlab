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
    duty_factor: float = 0.5,
) -> Tensor:
    """按照 WTW 附录生成平滑的周期性支撑目标。

    论文不是直接对 ``sin(2πt)`` 做 sigmoid，而是把相位 ``t`` 看成
    [0, 1] 内的时间变量，再用两个高斯 CDF 的乘积表示支撑区间：

      C(t) = Φ(t) [1 - Φ(t - d)]
           + Φ(t - 1) [1 - Φ(t - d - 1)]

    其中 ``d`` 是 duty factor。官方训练把 gait duration 固定为 0.5，
    因而默认行为仍严格等价于其实现。

    第二项负责连接周期边界，使相位从 1 回到 0 时接触目标连续。
    返回值范围为 [0, 1]，1 表示期望支撑，0 表示期望摆动。
    """
    sigma = max(float(smoothing), 1.0e-3)
    duty = min(max(float(duty_factor), 1.0e-3), 1.0 - 1.0e-3)
    root_two = math.sqrt(2.0)

    def normal_cdf(value: Tensor) -> Tensor:
        return 0.5 * (1.0 + torch.erf(value / (sigma * root_two)))

    return (
        normal_cdf(phase) * (1.0 - normal_cdf(phase - duty))
        + normal_cdf(phase - 1.0) * (1.0 - normal_cdf(phase - duty - 1.0))
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
    desired_contact = _smooth_contact_target(
        term.phase,
        smoothing,
        term.duty_factor,
    )
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


def wtw_contact_schedule_cost(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    behavior_command_name: str = "behavior",
    command_name: str = "twist",
    command_threshold: float = 0.05,
    smoothing: float = 0.07,
) -> Tensor:
    """约束四足实际接触向量匹配当前 gait 的期望接触时序。

    官方 WTW 的摆动相接触力项和支撑相足端速度项均按单腿计算；它们可
    以让每条腿"大致正确"，但未直接要求接触开关发生在正确的 phase。
    本项使用 ContactSensor 的二值接触状态，比较四条腿组成的完整向量：

      cost = mean_i | actual_contact_i - desired_contact_i |

    这不是只给 Pronking 使用的四脚同步项：

    - Pronking 的四个 desired_contact 相同，因此会同时约束四脚支撑、
      四脚离地与四脚落地；
    - Trot、Bound、Pace 的 desired_contact 由各自 phase 生成，因此会
      约束正确的交替接触关系，而不会错误要求四脚同步。

    ``desired_contact`` 在支撑/摆动切换处采用与官方一致的平滑目标，
    因此该项是连续软代价，推荐从较小负权重开始训练。
    """
    term = _get_behavior_term(env, behavior_command_name)
    try:
        sensor = env.scene[sensor_name]
    except (AttributeError, KeyError):
        return _zero_reward(env)
    if term is None or not isinstance(sensor, ContactSensor):
        return _zero_reward(env)
    if sensor.data.found is None:
        return _zero_reward(env)

    desired_contact = _smooth_contact_target(
        term.phase,
        smoothing,
        term.duty_factor,
    )
    actual_contact = (sensor.data.found > 0).to(dtype=desired_contact.dtype)
    active = _active_mask(env, command_name, command_threshold)
    per_foot_cost = torch.abs(actual_contact - desired_contact)
    cost = per_foot_cost.mean(dim=1) * active

    if hasattr(env, "extras") and "log" in env.extras:
        active_count = active.sum().clamp_min(1.0)
        active_per_foot = active.unsqueeze(-1)
        per_foot_cost_mean = (
            per_foot_cost * active_per_foot
        ).sum(dim=0) / active_count
        schedule_cost_mean = (cost * active).sum() / active_count

        # 通用时序指标：可直接对比不同 gait 的接触跟踪质量。
        env.extras["log"]["WTW/contact_schedule_error"] = schedule_cost_mean
        env.extras["log"]["WTW/contact_schedule_score"] = (
            1.0 - schedule_cost_mean
        )
        env.extras["log"]["WTW/foot_FL_schedule_error"] = per_foot_cost_mean[0]
        env.extras["log"]["WTW/foot_FR_schedule_error"] = per_foot_cost_mean[1]
        env.extras["log"]["WTW/foot_RL_schedule_error"] = per_foot_cost_mean[2]
        env.extras["log"]["WTW/foot_RR_schedule_error"] = per_foot_cost_mean[3]
        env.extras["log"]["WTW/all_feet_air_ratio"] = (
            ((1.0 - actual_contact).prod(dim=1) * active).sum() / active_count
        )
        env.extras["log"]["WTW/all_feet_contact_ratio"] = (
            (actual_contact.prod(dim=1) * active).sum() / active_count
        )

        # 只有 theta 全为零时才是 Pronking；Trot/Bound/Pace 本来就不应
        # 四脚同步，不能将它们混入这一专用诊断指标。
        is_pronking = (term.behavior[:, :3].abs().sum(dim=1) < 1.0e-6).float()
        pronking_active = active * is_pronking
        pronking_count = pronking_active.sum().clamp_min(1.0)
        disagreement = torch.abs(
            actual_contact[:, 1:] - actual_contact[:, :1]
        ).mean(dim=1)
        env.extras["log"]["WTW/pronking_sync_error"] = (
            (disagreement * pronking_active).sum() / pronking_count
        )

    return _safe(cost, limit=1.0)


def wtw_group_contact_consistency_cost(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    behavior_command_name: str = "behavior",
    command_name: str = "twist",
    command_threshold: float = 0.05,
    smoothing: float = 0.07,
    confidence_threshold: float = 0.9,
) -> Tensor:
    """惩罚同步 gait 高置信相位内的混合接触状态。

    逐脚时序误差允许短暂错峰：例如四脚中仅一条腿早一点离地，平均 L1
    仍可能很小。对 Pronking，这种状态恰恰违反“四脚同时支撑、同时腾空”
    的核心定义。本项只在四脚期望接触相同且远离相位边界时生效：

    ``mixed = 1 - all_contact - all_air``。

    因此 Trot、Bound、Pace 不会被错误要求四脚同步，接触切换边界也不会
    因传感器的一个时间步延迟受到额外惩罚。
    """
    term = _get_behavior_term(env, behavior_command_name)
    try:
        sensor = env.scene[sensor_name]
    except (AttributeError, KeyError):
        return _zero_reward(env)
    if term is None or not isinstance(sensor, ContactSensor):
        return _zero_reward(env)
    if sensor.data.found is None:
        return _zero_reward(env)
    if not 0.5 < confidence_threshold < 1.0:
        raise ValueError("confidence_threshold must be in (0.5, 1.0).")

    desired_contact = _smooth_contact_target(
        term.phase,
        smoothing,
        term.duty_factor,
    )
    actual_contact = (sensor.data.found > 0).to(dtype=desired_contact.dtype)
    active = _active_mask(env, command_name, command_threshold)
    synchronous = (
        desired_contact.amax(dim=1) - desired_contact.amin(dim=1) < 1.0e-3
    )
    desired_mean = desired_contact.mean(dim=1)
    high_confidence = (desired_mean > confidence_threshold) | (
        desired_mean < 1.0 - confidence_threshold
    )
    applicable = active * synchronous * high_confidence
    mixed_contact = (
        1.0 - actual_contact.prod(dim=1) - (1.0 - actual_contact).prod(dim=1)
    )
    cost = mixed_contact * applicable

    if hasattr(env, "extras") and "log" in env.extras:
        applicable_count = applicable.sum().clamp_min(1.0)
        env.extras["log"]["WTW/pronking_mixed_contact_ratio"] = (
            (mixed_contact * applicable).sum() / applicable_count
        )
        confident_error = torch.abs(actual_contact - desired_contact).mean(dim=1)
        env.extras["log"]["WTW/pronking_confident_schedule_error"] = (
            (confident_error * applicable).sum() / applicable_count
        )
        env.extras["log"]["WTW/pronking_group_contact_applicable_ratio"] = (
            applicable.mean()
        )

    return _safe(cost, limit=1.0)


def wtw_stance_contact(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    behavior_command_name: str = "behavior",
    command_name: str = "twist",
    command_threshold: float = 0.05,
    smoothing: float = 0.15,
) -> Tensor:
    """奖励期望支撑相中的足端实际接触地面。

    现有的 ``wtw_stance_phase_velocity`` 只约束支撑脚的足端速度，
    无法区分“足端静止地悬在空中”和“足端静止支撑在地面”。这个奖励
    用 ContactSensor 的 ``found`` 字段补上这一约束：

    - 期望支撑相且检测到接触：cost 为 0；
    - 期望支撑相但足端悬空：产生正的缺失接触 cost；
    - 期望摆动相：不参与该 cost，避免阻止正常抬脚。

    返回值是每个环境在当前时刻的“支撑相未接触比例” cost，范围为
    [0, 1]，因此配置时使用负的 reward weight。这样期望支撑相未触地
    会直接产生负奖励，而不是仅仅失去一部分正奖励。
    """
    term = _get_behavior_term(env, behavior_command_name)
    try:
        sensor = env.scene[sensor_name]
    except (AttributeError, KeyError):
        return _zero_reward(env)
    if term is None or not isinstance(sensor, ContactSensor):
        return _zero_reward(env)

    found = sensor.data.found
    if found is None:
        return _zero_reward(env)

    desired_contact = _smooth_contact_target(
        term.phase,
        smoothing,
        term.duty_factor,
    )
    actual_contact = (found > 0).to(dtype=desired_contact.dtype)
    active = _active_mask(env, command_name, command_threshold)

    # 用期望支撑权重归一化，避免平滑相位边界改变奖励量级。
    desired_stance_total = desired_contact.sum(dim=1).clamp_min(1.0e-6)
    contact_score = (
        (desired_contact * actual_contact).sum(dim=1)
        / desired_stance_total
    ).clamp(0.0, 1.0)
    # 支撑相未接触地面的比例作为 cost。运动指令下 active=1，
    # 零速度站立下 active=0，不干扰冻结 phase 后的站立行为。
    missing_contact_cost = (1.0 - contact_score) * active

    if hasattr(env, "extras") and "log" in env.extras:
        active_weight = desired_contact * active.unsqueeze(-1)
        denominator = active_weight.sum(dim=0).clamp_min(1.0e-6)
        per_foot_score = (
            (active_weight * actual_contact).sum(dim=0) / denominator
        ).clamp(0.0, 1.0)

        # 这些日志表示“该腿处于期望支撑相时，有多大比例确实接触地面”。
        # 采用按期望支撑权重归一化后的比例，便于直接比较四条腿。
        env.extras["log"]["WTW/stance_contact_score"] = (
            (contact_score * active).sum() / active.sum().clamp_min(1.0)
        )
        env.extras["log"]["WTW/stance_contact_FL"] = per_foot_score[0]
        env.extras["log"]["WTW/stance_contact_FR"] = per_foot_score[1]
        env.extras["log"]["WTW/stance_contact_RL"] = per_foot_score[2]
        env.extras["log"]["WTW/stance_contact_RR"] = per_foot_score[3]

    return _safe(missing_contact_cost, limit=1.0)


def wtw_shank_contact_cost(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str = "twist",
    command_threshold: float = 0.05,
    force_threshold: float = 5.0,
    force_scale: float = 20.0,
) -> Tensor:
    """对小腿触地施加连续软惩罚，而不是直接终止 episode。

    小腿触地通常说明腿部已经折叠，但 Pronking 在训练早期需要保留从
    异常姿态恢复的机会。因此这里不使用 ``illegal_contact`` 的硬终止，
    而是只惩罚超过死区阈值的当前接触力：

      cost = clamp((||F_shank|| - force_threshold) / force_scale, 0, 1)

    - 小于阈值的接触力不产生惩罚，减少接触数值噪声的影响；
    - 接触力越大，惩罚越大；
    - cost 按四条腿取平均，最终由 RewardManager 使用负权重累加；
    - 零速度站立时关闭该项，避免与冻结 phase 和站立姿态奖励竞争。
    """
    try:
        sensor = env.scene[sensor_name]
    except (AttributeError, KeyError):
        return _zero_reward(env)
    if not isinstance(sensor, ContactSensor) or sensor.data.force is None:
        return _zero_reward(env)

    force_threshold = max(float(force_threshold), 0.0)
    force_scale = max(float(force_scale), 1.0e-6)
    force = _safe(sensor.data.force, limit=100.0)
    force_magnitude = torch.linalg.vector_norm(force, dim=-1)
    excess_force = torch.clamp(force_magnitude - force_threshold, min=0.0)
    per_shank_cost = torch.clamp(excess_force / force_scale, min=0.0, max=1.0)
    cost = per_shank_cost.mean(dim=1) * _active_mask(
        env,
        command_name,
        command_threshold,
    )

    if hasattr(env, "extras") and "log" in env.extras:
        active = _active_mask(env, command_name, command_threshold)
        active_count = active.sum().clamp_min(1.0)
        per_shank_cost_mean = (
            per_shank_cost * active.unsqueeze(-1)
        ).sum(dim=0) / active_count
        env.extras["log"]["WTW/shank_contact_cost"] = (
            (cost * active).sum() / active_count
        )
        env.extras["log"]["WTW/shank_FL_contact"] = per_shank_cost_mean[0]
        env.extras["log"]["WTW/shank_FR_contact"] = per_shank_cost_mean[1]
        env.extras["log"]["WTW/shank_RL_contact"] = per_shank_cost_mean[2]
        env.extras["log"]["WTW/shank_RR_contact"] = per_shank_cost_mean[3]

    return _safe(cost, limit=1.0)


def wtw_body_height(
    env: ManagerBasedRlEnv,
    behavior_command_name: str = "behavior",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    base_height_target: float = 0.30,
    command_name: str = "twist",
    command_threshold: float = 0.05,
) -> Tensor:
    """按照官方 WTW 语义奖励机体高度。

    behavior 中的 body_height 是相对于基础高度的偏移量，而不是绝对
    高度。目标高度为 ``base_height_target + body_height_offset``，奖励
    使用官方 jump reward 的负平方误差形式。

    当前任务是平地版本，因此 reference height 为 0，实际高度使用
    root_link_pos_w[:, 2]；这与官方平地配置中的 reference_heights=0
    一致。复杂地形训练时应再引入地形参考高度。
    """
    term = _get_behavior_term(env, behavior_command_name)
    if term is None:
        return _zero_reward(env)
    try:
        asset: Entity = env.scene[asset_cfg.name]
        actual = asset.data.root_link_pos_w[:, 2]
    except (AttributeError, KeyError, RuntimeError, ValueError):
        return _zero_reward(env)

    base_target = float(base_height_target)
    if not math.isfinite(base_target):
        base_target = 0.30
    body_height_offset = _behavior_value(term, "body_height")
    target = body_height_offset + base_target
    # 官方 CoRLRewards._reward_jump：- (body_height - target_height)^2。
    # 奖励项的权重在配置中设置为正值 10.0。
    reward = -torch.square(actual - target)

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
            # 正值表示平均高度高于目标，负值表示整体下沉。它和绝对误差
            # 配合使用，可以区分“垂向振幅变大”和“机体整体上漂”。
            env.extras["log"]["WTW/body_height_signed_error"] = (
                ((actual - target) * active).sum() / active_count
            )
            env.extras["log"]["WTW/body_height_target"] = (
                (target * active).sum() / active_count
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
            env.extras["log"]["WTW/body_height_signed_error"] = torch.zeros(
                (), device=env.device
            )
            env.extras["log"]["WTW/body_height_target"] = torch.zeros(
                (), device=env.device
            )
            env.extras["log"]["WTW/body_height_min"] = torch.zeros(
                (), device=env.device
            )

    return _safe(reward, limit=100.0)


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
        desired_swing = 1.0 - _smooth_contact_target(
            term.phase,
            smoothing,
            term.duty_factor,
        )
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


def wtw_foot_clearance_cmd_linear_cost(
    env: ManagerBasedRlEnv,
    height_sensor_name: str,
    behavior_command_name: str = "behavior",
    command_name: str = "twist",
    command_threshold: float = 0.05,
    smoothing: float = 0.07,
    foot_radius: float = 0.02,
) -> Tensor:
    """复刻官方 ``feet_clearance_cmd_linear`` 的连续摆动轨迹代价。

    官方不是只在落脚瞬间检查摆腿峰值，而是在整个摆动相用三角轨迹约束
    足端高度：离地和落脚时接近地面，摆动中点达到行为命令给定的高度。
    对 Pronking 而言，四只脚使用相同 phase，因此该项会同时鼓励四脚
    腾空并在半个周期后同时回到地面。

    返回的是未加权的平方误差和，应使用负 reward weight；官方权重为
    ``-30``，当前配置沿用该量级。
    """
    term = _get_behavior_term(env, behavior_command_name)
    if term is None:
        return _zero_reward(env)
    try:
        height_sensor = env.scene[height_sensor_name]
    except (AttributeError, KeyError):
        return _zero_reward(env)
    if not isinstance(height_sensor, TerrainHeightSensor):
        return _zero_reward(env)

    heights = _safe(height_sensor.data.heights, limit=1.0).clamp_min(0.0)
    duty = term.duty_factor
    # 将摆动相 [duty, 1) 映射到 [0, 1]，再构造 0 -> 1 -> 0 的高度轨迹。
    swing_phase = ((term.phase - duty) / (1.0 - duty)).clamp(0.0, 1.0)
    swing_profile = 1.0 - torch.abs(1.0 - 2.0 * swing_phase)
    desired_contact = _smooth_contact_target(term.phase, smoothing, duty)
    target_height = (
        _behavior_value(term, "foot_swing_height").unsqueeze(-1)
        * swing_profile
        + max(float(foot_radius), 0.0)
    )
    cost = torch.square(target_height - heights) * (1.0 - desired_contact)
    active = _active_mask(env, command_name, command_threshold)
    cost = cost.sum(dim=1) * active

    if hasattr(env, "extras") and "log" in env.extras:
        active_count = active.sum().clamp_min(1.0)
        env.extras["log"]["WTW/foot_clearance_cost"] = (
            (cost * active).sum() / active_count
        )
        env.extras["log"]["WTW/foot_clearance_target"] = (
            (target_height * (1.0 - desired_contact) * active.unsqueeze(-1)).sum()
            / ((1.0 - desired_contact) * active.unsqueeze(-1)).sum().clamp_min(1.0)
        )

    return _safe(cost, limit=100.0)


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
    """官方 WTW 风格的全身 Raibert 足端落点代价。

    官方实现以期望速度（而不是速度跟踪误差）和当前 phase 生成四条腿的
    目标落点，并对全部四足的机体坐标系位置施加平方误差。这样在 Pronking
    中四脚能获得一致的前后落点修正，而不是各自寻找局部稳定位置。

    Nazarite 保留了对 ``lin_vel_y`` 的对称扩展；当横向速度为零时，该公式
    与官方的前向速度和 yaw 补偿完全同形。
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
        stance_length: float = 0.45,
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

        # 第一次调用通常发生在 reset 后；只记录前后、左右的腿位符号，
        # 兼容 Nazarite 的 [FL, FR, RL, RR] site 顺序。
        new_envs = ~self._initialized
        if new_envs.any():
            self._nominal_pos_b[new_envs] = positions_b[new_envs].detach()
            self._initialized[new_envs] = True

        nominal = self._nominal_pos_b
        stance_width = _behavior_value(term, "stance_width").unsqueeze(-1)
        stance_length = max(float(stance_length), 1.0e-3)
        lateral_sign = torch.where(nominal[:, :, 1] >= 0.0, 1.0, -1.0)
        longitudinal_sign = torch.where(nominal[:, :, 0] >= 0.0, 1.0, -1.0)
        target = torch.empty_like(nominal)
        target[:, :, 0] = longitudinal_sign * stance_length * 0.5
        target[:, :, 1] = lateral_sign * stance_width * 0.5

        # 与官方 _reward_raibert_heuristic 相同：phase_offset 在每个 gait
        # 周期内从 +0.5 平滑变为 -0.5，再回到 +0.5；速度项的时间尺度为
        # 半个周期。这里使用期望速度，而非 (期望 - 实际) 误差，避免速度
        # 跟踪尚未稳定时把落点目标来回拉动。
        frequency = _behavior_value(term, "frequency").clamp_min(0.1)
        half_cycle = (0.5 / frequency).unsqueeze(-1)
        phase_offset = torch.abs(1.0 - 2.0 * term.phase) - 0.5
        target[:, :, 0] += phase_offset * command[:, 0:1] * half_cycle
        # 官方代码未单列 vy；这里保留对称项以支持 Grid Adaptive 的横向命令。
        target[:, :, 1] += phase_offset * command[:, 1:2] * half_cycle
        target[:, :, 1] += (
            phase_offset
            * command[:, 2:3]
            * (stance_length * 0.5)
            * longitudinal_sign
            * half_cycle
        )

        cost = torch.square(positions_b - target).sum(dim=(1, 2))
        active = _active_mask(env, command_name, command_threshold)
        cost *= active
        if hasattr(env, "extras") and "log" in env.extras:
            env.extras["log"]["WTW/raibert_foot_position_cost"] = (
                (cost * active).sum() / active.sum().clamp_min(1.0)
            )
        return _safe(cost, limit=100.0)

    def reset(self, env_ids: Tensor | slice | None) -> None:
        """让部分 reset 环境在下一次调用时重新记录名义足端位置。"""
        if env_ids is None:
            self._initialized[:] = False
        else:
            self._initialized[env_ids] = False
