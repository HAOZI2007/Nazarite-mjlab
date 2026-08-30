from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
    import viser

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
    # 与官方 WTW 一致：body_height 表示相对于基础高度的偏移量，而不是
    # 绝对机体高度。绝对目标由 base_height_target + body_height_offset 得到。
    body_height_range: tuple[float, float] = (-0.25, 0.15)
    body_pitch_range: tuple[float, float] = (-0.05, 0.05)
    stance_width_range: tuple[float, float] = (0.18, 0.24)
    foot_swing_height_range: tuple[float, float] = (0.05, 0.10)
    # 与官方 ``gait_duration_cmd`` 对应。0.5 表示每个周期前半段支撑、
    # 后半段摆动；它不是额外输入 actor 的行为维度，而是当前任务固定的
    # 接触时序形状。需要设计非对称步态时再显式修改此参数。
    duty_factor: float = 0.5
    gait_names: tuple[str, ...] = (
      "trot",
      "pronking",
      "bounding",
      "pacing",
    )
    # 只有存在有效速度指令时才推进 phase；零速度时保持站立 phase 不变。
    phase_command_name: str = "twist"
    phase_command_threshold: float = 0.05
    # 默认随机初始 phase 以提升多环境采样效率。严格 Pronking 单元阶段可
    # 关闭它，使 reset 从 phase=0 的四脚支撑相开始，与物理初始站姿一致。
    randomize_initial_phase: bool = True

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
        if not 0.0 < self.duty_factor < 1.0:
            raise ValueError(
                "WTW duty_factor must be strictly between 0 and 1, got "
                f"{self.duty_factor}."
            )
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
        # 奖励项读取该值来构造官方式 desired_contact_states。把它保存在
        # command term 而不是各 reward 的 params 中，可避免接触奖励、
        # 足端高度奖励和 Raibert 项出现不一致的支撑/摆动边界。
        self.duty_factor = float(cfg.duty_factor)

        # 这些 metrics 只用于检查行为参数是否真的被采样。
        self.metrics["wtw_frequency"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["wtw_body_height"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["wtw_swing_height"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["wtw_gait_id"] = torch.zeros(self.num_envs, device=self.device)

        # Viser GUI 的状态只在 play 模式创建。行为参数仍以张量为唯一
        # 真值来源；GUI 回调不直接写入仿真张量，避免浏览器线程和仿真线程
        # 同时修改同一组命令。每个仿真步由 _apply_gui_override() 统一写入。
        self._gui_override_enabled: viser.GuiCheckboxHandle | None = None
        self._gui_gait: viser.GuiDropdownHandle | None = None
        self._gui_behavior_sliders: dict[str, viser.GuiSliderHandle] = {}
        self._gui_get_env_idx: Callable[[], int] | None = None
        self._gui_reset_phase_requested = False
        self._gui_last_gait: str | None = None

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

        # 通用 WTW 随机化起始时刻；严格同步步态的单元训练可从支撑相开始，
        # 避免零速冻结或 reset 后的物理接触与 phase 参考天然不一致。
        if self._wtw_cfg.randomize_initial_phase:
            self._base_phase[env_ids] = torch.rand(count, device=self.device)
        else:
            self._base_phase[env_ids] = 0.0
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

    # GUI.

    def create_gui(
        self,
        name: str,
        server: viser.ViserServer,
        get_env_idx: Callable[[], int],
        on_change: Callable[[], None] | None = None,
        request_action: Callable[[str, Any], None] | None = None,
    ) -> None:
        """在网页 play 中创建 WTW 行为参数控制面板。

        面板只覆盖当前 viewer 选中的环境。打开 ``Enable override`` 后，
        每一步都会重新写入该环境的行为命令，因此不会被 command 的定时
        resample 覆盖。下拉框只显示当前训练配置包含的 gait，防止在一个只
        训练 Trot 的 checkpoint 上误把未训练 gait 当成可用行为。
        """
        from viser import Icon

        del request_action
        default_gait = self._wtw_cfg.gait_names[0]
        nominal_values = {
            "frequency": sum(self._wtw_cfg.frequency_range) / 2.0,
            "body_height": sum(self._wtw_cfg.body_height_range) / 2.0,
            "body_pitch": sum(self._wtw_cfg.body_pitch_range) / 2.0,
            "stance_width": sum(self._wtw_cfg.stance_width_range) / 2.0,
            "foot_swing_height": sum(self._wtw_cfg.foot_swing_height_range)
            / 2.0,
        }

        with server.gui.add_folder(name.capitalize()):
            enabled = server.gui.add_checkbox("Enable override", initial_value=False)
            gait = server.gui.add_dropdown(
                "Gait",
                options=list(self._wtw_cfg.gait_names),
                initial_value=default_gait,
            )
            frequency = server.gui.add_slider(
                "Frequency (Hz)",
                min=0.5,
                max=5.0,
                step=0.05,
                initial_value=nominal_values["frequency"],
            )
            height = server.gui.add_slider(
                "Body height offset (m, target = 0.32 + offset)",
                min=-0.10,
                max=0.10,
                step=0.005,
                initial_value=nominal_values["body_height"],
            )
            pitch = server.gui.add_slider(
                "Body pitch (rad)",
                min=-0.30,
                max=0.30,
                step=0.01,
                initial_value=nominal_values["body_pitch"],
            )
            width = server.gui.add_slider(
                "Stance width (m)",
                min=0.12,
                max=0.35,
                step=0.005,
                initial_value=nominal_values["stance_width"],
            )
            swing_height = server.gui.add_slider(
                "Foot swing height (m)",
                min=0.02,
                max=0.16,
                step=0.005,
                initial_value=nominal_values["foot_swing_height"],
            )
            reset_phase = server.gui.add_button("Reset phase", icon=Icon.REFRESH)
            reset_behavior = server.gui.add_button(
                "Use training defaults",
                icon=Icon.ROTATE_CLOCKWISE,
            )

        # GUI 事件只刷新 viewer 或修改 GUI 本身；实际 torch 写入统一在
        # compute() 的仿真线程完成。
        if on_change is not None:
            enabled.on_update(lambda _: on_change())
            gait.on_update(lambda _: on_change())
            for slider in (frequency, height, pitch, width, swing_height):
                slider.on_update(lambda _: on_change())

        @reset_phase.on_click
        def _(_) -> None:
            self._gui_reset_phase_requested = True
            if on_change is not None:
                on_change()

        @reset_behavior.on_click
        def _(_) -> None:
            # 恢复的是训练配置区间的中点，而不是某个并行环境刚好采样到的
            # 随机值；这样在网页里有一个确定、可复现的起点。
            enabled.value = True
            gait.value = default_gait
            frequency.value = nominal_values["frequency"]
            height.value = nominal_values["body_height"]
            pitch.value = nominal_values["body_pitch"]
            width.value = nominal_values["stance_width"]
            swing_height.value = nominal_values["foot_swing_height"]
            self._gui_reset_phase_requested = True
            if on_change is not None:
                on_change()

        self._gui_override_enabled = enabled
        self._gui_gait = gait
        self._gui_behavior_sliders = {
            "frequency": frequency,
            "body_height": height,
            "body_pitch": pitch,
            "stance_width": width,
            "foot_swing_height": swing_height,
        }
        self._gui_get_env_idx = get_env_idx

    def _apply_gui_override(self) -> None:
        """将网页行为参数写入当前选中环境，并保持其余环境不受影响。"""
        if self._gui_override_enabled is None or not self._gui_override_enabled.value:
            return
        if self._gui_gait is None or self._gui_get_env_idx is None:
            return

        env_idx = self._gui_get_env_idx()
        if not 0 <= env_idx < self.num_envs:
            return

        gait_name = self._gui_gait.value
        # 下拉选项来自 gait_names，但保留保护分支，避免外部 GUI 状态异常。
        if gait_name not in GAIT_THETA:
            return

        if self._gui_last_gait != gait_name:
            # 切换 gait 时从公共支撑相开始，避免旧 gait 的相位偏移直接带到
            # 新 gait，造成一帧内不可解释的接触参考跳变。
            self._base_phase[env_idx] = 0.0
            self._gui_last_gait = gait_name
        if self._gui_reset_phase_requested:
            self._base_phase[env_idx] = 0.0
            self._gui_reset_phase_requested = False

        self.behavior[env_idx, 0:3] = torch.tensor(
            GAIT_THETA[gait_name],
            device=self.device,
            dtype=self.behavior.dtype,
        )
        self.gait_ids[env_idx] = self._wtw_cfg.gait_names.index(gait_name)
        for behavior_name, slider in self._gui_behavior_sliders.items():
            self.behavior[env_idx, BEHAVIOR_INDEX[behavior_name]] = slider.value

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

        # 放在 phase 推进前，使本个仿真步已经使用网页设置的 frequency 和
        # gait；定时重采样即使发生，也会立刻被这个覆盖重新写回。
        self._apply_gui_override()

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
        if command is None:
            raise RuntimeError(
                f"Command '{self._wtw_cfg.phase_command_name}' is not available."
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
    """返回四条腿的 sin/cos timing reference，形状为 [N, 8]。

    sin 和 cos 共同表示一个完整周期，避免仅使用 sin 时不同 phase
    位置可能得到相同观测值的问题。输出顺序为：

      [sin_FL, sin_FR, sin_RL, sin_RR,
       cos_FL, cos_FR, cos_RL, cos_RR]
    """
    term = env.command_manager.get_term(command_name)
    if not isinstance(term, WTWBehaviorCommand):
        raise TypeError(f"{command_name} is not a WTWBehaviorCommand.")
    phase_angle = 2.0 * math.pi * term.phase
    return torch.cat(
        (
            torch.sin(phase_angle),
            torch.cos(phase_angle),
        ),
        dim=-1,
    )
