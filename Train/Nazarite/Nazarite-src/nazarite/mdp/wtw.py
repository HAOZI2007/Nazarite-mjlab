from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

BEHAVIOR_DIM = 8

BEHAVIOR_INDEX = {
    "theta1": 0,
    "theta2": 1,
    "theta3": 2,
    "frequency": 3,
    "body_height": 4,
    "body_pitch": 5,
    "stance_width": 6,
    "foot_swing_height": 7,
}

# 论文使用的四种对称接触模式。
# theta 的顺序为 [theta1, theta2, theta3]，单位是 gait 周期比例。
GAIT_THETA = {
    "pronking": (0.0, 0.0, 0.0),
    "trot": (0.5, 0.0, 0.0),
    "bounding": (0.0, 0.5, 0.0),
    "pacing": (0.0, 0.0, 0.5),
}


@dataclass(kw_only=True)
class WTWBehaviorCommandCfg(CommandTermCfg):
    """Walk These Ways 行为参数和步态相位配置。"""

    entity_name: str
    frequency_range: tuple[float, float] = (1.5, 3.0)
    body_height_range: tuple[float, float] = (0.30, 0.34)
    body_pitch_range: tuple[float, float] = (-0.05, 0.05)
    stance_width_range: tuple[float, float] = (0.18, 0.24)
    foot_swing_height_range: tuple[float, float] = (0.05, 0.10)
    gait_names: tuple[str, ...] = (
      "trot",
      "pronking",
      "bounding",
      "pacing",
    )
    # 只有存在有效速度指令时才推进 phase；零速度时保持站立 phase 不变。
    phase_command_name: str = "twist"
    phase_command_threshold: float = 0.05

    def build(self, env: ManagerBasedRlEnv) -> WTWBehaviorCommand:
        return WTWBehaviorCommand(self, env)

    def __post_init__(self) -> None:
        if not self.gait_names:
            raise ValueError("WTW gait_names must not be empty.")
        unknown_gaits = set(self.gait_names) - set(GAIT_THETA)
        if unknown_gaits:
            raise ValueError(f"Unknown WTW gaits: {sorted(unknown_gaits)}")
        for name, value_range in (
            ("frequency_range", self.frequency_range),
            ("body_height_range", self.body_height_range),
            ("body_pitch_range", self.body_pitch_range),
            ("stance_width_range", self.stance_width_range),
            ("foot_swing_height_range", self.foot_swing_height_range),
        ):
            if value_range[1] < value_range[0]:
                raise ValueError(f"{name} must be increasing, got {value_range}.")
        if self.frequency_range[0] <= 0.0:
            raise ValueError("WTW frequency must be positive.")
        if self.stance_width_range[0] <= 0.0:
            raise ValueError("WTW stance width must be positive.")
        if self.foot_swing_height_range[0] <= 0.0:
            raise ValueError("WTW foot swing height must be positive.")
        if self.phase_command_threshold < 0.0:
            raise ValueError("WTW phase command threshold must be non-negative.")


class WTWBehaviorCommand(CommandTerm):
    """生成 WTW 行为参数，并维护 Go2 四条腿的 gait phase。

    四条腿内部顺序固定为 [FL, FR, RL, RR]，与 Nazarite 的
    GO2_FOOT_SITES 和足端接触传感器顺序保持一致。
    """

    def __init__(
        self,
        cfg: WTWBehaviorCommandCfg,
        env: ManagerBasedRlEnv,
    ) -> None:
        super().__init__(cfg, env)
        self._wtw_cfg = cfg

        # 每个并行环境独立保存一组 8 维行为参数。
        self.behavior = torch.zeros(
            self.num_envs,
            BEHAVIOR_DIM,
            device=self.device,
        )
        # gait 的公共时间变量，范围为 [0, 1)。
        self._base_phase = torch.zeros(self.num_envs, device=self.device)
        # 当前四条腿的 phase，顺序为 [FL, FR, RL, RR]。
        self.phase = torch.zeros(self.num_envs, 4, device=self.device)
        self.gait_ids = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )

        # 这些 metrics 只用于检查行为参数是否真的被采样。
        self.metrics["wtw_frequency"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["wtw_body_height"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["wtw_swing_height"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["wtw_gait_id"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> Tensor:
        """返回 [N, 8] 行为参数，供 generated_commands 使用。"""
        return self.behavior

    def _update_metrics(self) -> None:
        """把当前行为参数复制到命令 metrics。"""
        self.metrics["wtw_frequency"] = self.behavior[:, BEHAVIOR_INDEX["frequency"]]
        self.metrics["wtw_body_height"] = self.behavior[
            :, BEHAVIOR_INDEX["body_height"]
        ]
        self.metrics["wtw_swing_height"] = self.behavior[
            :, BEHAVIOR_INDEX["foot_swing_height"]
        ]
        self.metrics["wtw_gait_id"] = self.gait_ids.float()

    def _update_command(self, env_ids: Tensor | None) -> None:
        """行为参数本身不依赖机器人状态，不需要额外更新。"""
        del env_ids

    def _sample_uniform(
        self,
        value_range: tuple[float, float],
        count: int,
    ) -> Tensor:
        """为一批环境独立均匀采样一个标量行为参数。"""
        return torch.empty(count, device=self.device).uniform_(*value_range)

    def _theta_to_phase_offsets(self, theta: Tensor) -> Tensor:
        """使用论文附录公式，将 theta 转成当前 Go2 腿顺序的 phase。

        论文公式的腿顺序是 [FR, FL, RR, RL]：

          [t + theta2 + theta3,
           t + theta1 + theta3,
           t + theta1,
           t + theta2]

        当前项目顺序是 [FL, FR, RL, RR]，因此重排为：

          [theta1 + theta3,
           theta2 + theta3,
           theta2,
           theta1]
        """
        theta1, theta2, theta3 = theta.unbind(dim=-1)
        return (
            torch.stack(
                (
                    theta1 + theta3,
                    theta2 + theta3,
                    theta2,
                    theta1,
                ),
                dim=-1,
            )
            % 1.0
        )

    def _resample_command(self, env_ids: Tensor) -> None:
        """在 reset 或行为计时结束时采样新的行为。"""
        count = len(env_ids)
        gait_ids = torch.randint(
            len(self._wtw_cfg.gait_names),
            (count,),
            device=self.device,
        )
        self.gait_ids[env_ids] = gait_ids

        theta = torch.empty(count, 3, device=self.device)
        for gait_id, gait_name in enumerate(self._wtw_cfg.gait_names):
            selected = gait_ids == gait_id
            if selected.any():
                theta[selected] = torch.tensor(
                    GAIT_THETA[gait_name],
                    device=self.device,
                )

        self.behavior[env_ids, 0:3] = theta
        self.behavior[env_ids, 3] = self._sample_uniform(
            self._wtw_cfg.frequency_range, count
        )
        self.behavior[env_ids, 4] = self._sample_uniform(
            self._wtw_cfg.body_height_range, count
        )
        self.behavior[env_ids, 5] = self._sample_uniform(
            self._wtw_cfg.body_pitch_range, count
        )
        self.behavior[env_ids, 6] = self._sample_uniform(
            self._wtw_cfg.stance_width_range, count
        )
        self.behavior[env_ids, 7] = self._sample_uniform(
            self._wtw_cfg.foot_swing_height_range, count
        )

        # 随机化每个环境的起始时刻，避免所有环境同步进入同一接触阶段。
        self._base_phase[env_ids] = torch.rand(count, device=self.device)
        self._update_phase(env_ids)

    def _update_phase(self, env_ids: Tensor | None) -> None:
        """根据 theta 和公共时间变量计算四条腿的 phase。"""
        theta = self.behavior[:, 0:3]
        phase_offsets = self._theta_to_phase_offsets(theta)
        phase = (self._base_phase.unsqueeze(-1) + phase_offsets) % 1.0
        if env_ids is None:
            self.phase = phase
        else:
            self.phase[env_ids] = phase[env_ids]

    def compute(
        self,
        dt: float | Tensor,
        env_ids: Tensor | None = None,
    ) -> None:
        """推进 gait phase，并保留 CommandTerm 的计时/resample 生命周期。"""
        super().compute(dt, env_ids)

        # reset 路径已经在 _resample_command 中初始化 phase，不再推进。
        if env_ids is not None:
            return

        if isinstance(dt, Tensor):
            dt_batch = dt.to(device=self.device)
        else:
            dt_batch = torch.full(
                (self.num_envs,),
                float(dt),
                device=self.device,
            )
        frequency = self.behavior[:, BEHAVIOR_INDEX["frequency"]]
        next_phase = (self._base_phase + frequency * dt_batch) % 1.0

        # 速度指令为零时冻结 phase，使 WTW 任务能够自然进入静止站立。
        # 这里使用与奖励函数相同的阈值定义：平面速度和 yaw 速度都接近
        # 零时视为 standing；一旦出现有效速度指令，phase 立即恢复推进。
        command = self._env.command_manager.get_command(
            self._wtw_cfg.phase_command_name
        )
        command_magnitude = (
            torch.linalg.vector_norm(command[:, :2], dim=1)
            + torch.abs(command[:, 2])
        )
        phase_active = command_magnitude > self._wtw_cfg.phase_command_threshold
        self._base_phase = torch.where(
            phase_active,
            next_phase,
            self._base_phase,
        )
        self._update_phase(env_ids=None)


def wtw_phase_reference(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> Tensor:
    """返回论文使用的四足正弦 timing reference，形状为 [N, 4]。"""
    term = env.command_manager.get_term(command_name)
    if not isinstance(term, WTWBehaviorCommand):
        raise TypeError(f"{command_name} is not a WTWBehaviorCommand.")
    return torch.sin(2.0 * math.pi * term.phase)
