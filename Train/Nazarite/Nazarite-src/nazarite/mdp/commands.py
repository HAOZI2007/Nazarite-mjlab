from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)


@dataclass(kw_only=True)
class GridAdaptiveVelocityCommandCfg(UniformVelocityCommandCfg):
  """基于速度网格的自适应命令配置。

  网格的两个维度分别对应前向速度 ``lin_vel_x`` 和偏航角速度
  ``ang_vel_z``。横向速度仍在 ``ranges.lin_vel_y`` 中独立均匀采样。
  """

  grid_num_x: int = 9
  """前向速度方向的网格数量。"""

  grid_num_yaw: int = 7
  """偏航角速度方向的网格数量。"""

  initial_cell: tuple[int, int] | None = None
  """初始激活网格，格式为 ``(x_index, yaw_index)``。"""

  min_cell_visits: int = 20
  """扩展邻居前，一个网格至少需要完成的命令段数量。"""

  success_window_size: int = 100
  """用于判断课程扩展的近期成功率窗口大小。"""

  max_new_cells_per_update: int = 4
  """单个仿真步最多新激活的网格数量。"""

  require_all_active_cells_ready: bool = False
  """是否要求所有已激活 cell 达标后，才允许继续扩展新的邻居。"""

  success_rate_threshold: float = 0.8
  """扩展邻居所需的最低成功率。"""

  velocity_error_threshold: float = 0.4
  """线速度平均误差成功阈值，单位为 m/s。"""

  yaw_error_threshold: float = 0.35
  """偏航角速度平均误差成功阈值，单位为 rad/s。"""

  gait_quality_behavior_command_name: str | None = None
  """可选的 WTW 行为命令名；设置后，课程扩展会额外检查接触质量。"""

  gait_quality_sensor_name: str | None = None
  """用于读取四足接触状态的传感器名。"""

  gait_schedule_error_threshold: float | None = None
  """命令段平均接触时序误差上限；``None`` 表示不作为课程门槛。"""

  gait_sync_error_threshold: float | None = None
  """同步步态的四足接触分歧上限；``None`` 表示不作为课程门槛。"""

  gait_mixed_contact_threshold: float | None = None
  """同步步态高置信相位内的混合接触比例上限；``None`` 表示不检查。"""

  gait_contact_smoothing: float = 0.07
  """构造接触时序目标时使用的高斯 CDF 平滑宽度。"""

  def build(self, env: ManagerBasedRlEnv) -> GridAdaptiveVelocityCommand:
    return GridAdaptiveVelocityCommand(self, env)

  def __post_init__(self) -> None:
    super().__post_init__()
    if self.grid_num_x < 1 or self.grid_num_yaw < 1:
      raise ValueError("Grid dimensions must be positive.")
    if self.min_cell_visits < 1:
      raise ValueError("min_cell_visits must be at least 1.")
    if self.success_window_size < 1:
      raise ValueError("success_window_size must be at least 1.")
    if self.max_new_cells_per_update < 1:
      raise ValueError("max_new_cells_per_update must be at least 1.")
    if not 0.0 < self.success_rate_threshold <= 1.0:
      raise ValueError("success_rate_threshold must be in (0, 1].")
    if self.velocity_error_threshold <= 0.0 or self.yaw_error_threshold <= 0.0:
      raise ValueError("Velocity error thresholds must be positive.")
    gait_thresholds = (
      self.gait_schedule_error_threshold,
      self.gait_sync_error_threshold,
      self.gait_mixed_contact_threshold,
    )
    if any(threshold is not None for threshold in gait_thresholds):
      if (
        self.gait_quality_behavior_command_name is None
        or self.gait_quality_sensor_name is None
      ):
        raise ValueError(
          "Gait quality thresholds require both "
          "gait_quality_behavior_command_name and gait_quality_sensor_name."
        )
      if any(
        threshold is not None and not 0.0 <= threshold <= 1.0
        for threshold in gait_thresholds
      ):
        raise ValueError("Gait quality thresholds must be in [0, 1].")
    if self.gait_contact_smoothing <= 0.0:
      raise ValueError("gait_contact_smoothing must be positive.")
    if self.heading_command or self.rel_heading_envs != 0.0:
      raise ValueError(
        "GridAdaptiveVelocityCommand requires heading_command=False and "
        "rel_heading_envs=0.0."
      )


class GridAdaptiveVelocityCommand(UniformVelocityCommand):
  """使用激活网格均匀采样，并根据成功率扩展四连通邻居。"""

  def __init__(
    self, cfg: GridAdaptiveVelocityCommandCfg, env: ManagerBasedRlEnv
  ) -> None:
    super().__init__(cfg, env)
    # 父类的 cfg 属性保持 UniformVelocityCommandCfg 类型，避免违反
    # Pylance 对可变属性类型覆盖的检查；自定义字段通过 _grid_cfg 访问。
    self._grid_cfg = cfg

    # 每个环境可以处于不同 cell，但所有并行环境共享一张课程地图。
    self.current_cell = torch.zeros(
      self.num_envs, 2, dtype=torch.long, device=self.device
    )
    self.active_cells = torch.zeros(
      self._grid_cfg.grid_num_x,
      self._grid_cfg.grid_num_yaw,
      dtype=torch.bool,
      device=self.device,
    )
    initial_cell = self._initial_cell()
    self.active_cells[initial_cell[0], initial_cell[1]] = True

    # 全局网格统计量：访问次数和成功次数。
    self.cell_visits = torch.zeros(
      self._grid_cfg.grid_num_x,
      self._grid_cfg.grid_num_yaw,
      dtype=torch.long,
      device=self.device,
    )
    self.cell_successes = torch.zeros_like(self.cell_visits)

    # 每个 cell 保存最近若干个命令段的成败结果。
    # 第一维展平为 cell，便于在不同 cell 上独立维护 ring buffer。
    num_cells = self._grid_cfg.grid_num_x * self._grid_cfg.grid_num_yaw
    self.cell_success_history = torch.zeros(
      num_cells,
      self._grid_cfg.success_window_size,
      dtype=torch.bool,
      device=self.device,
    )
    self.cell_recent_visits = torch.zeros(
      num_cells, dtype=torch.long, device=self.device
    )
    self.cell_recent_successes = torch.zeros_like(self.cell_recent_visits)
    self.cell_history_ptr = torch.zeros_like(self.cell_recent_visits)
    # 防止同一个 common_step_counter 内的多个 reset 批次重复扩展课程。
    self._last_expansion_step = -1

    # 当前命令段的逐环境统计量。
    self._segment_error_xy = torch.zeros(self.num_envs, device=self.device)
    self._segment_error_yaw = torch.zeros(self.num_envs, device=self.device)
    # 接触质量按命令段累积，和速度误差一样仅在命令切换/reset 时结算。
    self._segment_gait_schedule_error = torch.zeros(
      self.num_envs, device=self.device
    )
    self._segment_gait_sync_error = torch.zeros(self.num_envs, device=self.device)
    self._segment_gait_mixed_contact = torch.zeros(
      self.num_envs, device=self.device
    )
    self._segment_steps = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self.metrics["grid_error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["grid_error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["grid_success"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["grid_gait_schedule_error"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["grid_gait_sync_error"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["grid_gait_mixed_contact"] = torch.zeros(
      self.num_envs, device=self.device
    )

  def _initial_cell(self) -> tuple[int, int]:
    """获取并检查初始 cell，默认使用网格中心。"""
    cell = self._grid_cfg.initial_cell
    if cell is None:
      cell = (
        self._grid_cfg.grid_num_x // 2,
        self._grid_cfg.grid_num_yaw // 2,
      )
    if not (
      0 <= cell[0] < self._grid_cfg.grid_num_x
      and 0 <= cell[1] < self._grid_cfg.grid_num_yaw
    ):
      raise ValueError(
        f"initial_cell={cell} is outside the grid "
        f"({self._grid_cfg.grid_num_x}, {self._grid_cfg.grid_num_yaw})."
      )
    return cell

  def _grid_edges(
    self, value_range: tuple[float, float], num_bins: int
  ) -> torch.Tensor:
    """根据配置范围生成网格边界。"""
    if value_range[1] < value_range[0]:
      raise ValueError(f"Velocity range must be increasing, got {value_range}.")
    # ``low == high`` 表示该维度被刻意锁死，例如 Pronking 单元阶段固定
    # yaw=0。linspace 会返回重复边界，随后区间采样自然得到唯一常量。
    return torch.linspace(
      value_range[0], value_range[1], num_bins + 1, device=self.device
    )

  def _sample_cells(self, env_ids: torch.Tensor) -> None:
    """从激活 cell 中均匀采样指定环境的 cell。"""
    active_ids = torch.nonzero(self.active_cells.flatten(), as_tuple=False).flatten()
    if len(active_ids) == 0:
      raise RuntimeError("Grid Adaptive Curriculum has no active cells.")

    # torch.randint 对 active_ids 等概率取样：P(cell)=1/N_active。
    selected = active_ids[
      torch.randint(len(active_ids), (len(env_ids),), device=self.device)
    ]
    self.current_cell[env_ids, 0] = selected // self._grid_cfg.grid_num_yaw
    self.current_cell[env_ids, 1] = selected % self._grid_cfg.grid_num_yaw

  def _sample_uniform(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """为每个环境在各自的区间内均匀采样。"""
    return low + (high - low) * torch.rand(len(low), device=self.device)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # 命令计时器到期和环境 reset 都会走这里。先结算旧命令段，再抽取新 cell。
    self._settle_segments(env_ids)
    self._sample_cells(env_ids)

    x_edges = self._grid_edges(
      self._grid_cfg.ranges.lin_vel_x, self._grid_cfg.grid_num_x
    )
    yaw_edges = self._grid_edges(
      self._grid_cfg.ranges.ang_vel_z, self._grid_cfg.grid_num_yaw
    )
    x_index = self.current_cell[env_ids, 0]
    yaw_index = self.current_cell[env_ids, 1]

    x_low, x_high = x_edges[x_index], x_edges[x_index + 1]
    yaw_low, yaw_high = yaw_edges[yaw_index], yaw_edges[yaw_index + 1]
    self.vel_command_b[env_ids, 0] = self._sample_uniform(x_low, x_high)
    self.vel_command_b[env_ids, 1] = self._sample_uniform(
      torch.full_like(x_low, self._grid_cfg.ranges.lin_vel_y[0]),
      torch.full_like(x_low, self._grid_cfg.ranges.lin_vel_y[1]),
    )
    self.vel_command_b[env_ids, 2] = self._sample_uniform(yaw_low, yaw_high)

    # 保留普通速度命令的 standing/world/forward 逻辑；当前仅启用 standing。
    r = torch.rand(len(env_ids), device=self.device)
    self.is_standing_env[env_ids] = r <= self.cfg.rel_standing_envs
    self.is_world_env[env_ids] = r <= self.cfg.rel_world_envs
    self.vel_command_w[env_ids] = self.vel_command_b[env_ids]
    self.is_forward_env[env_ids] = r <= self.cfg.rel_forward_envs

    # standing 命令是独立的零速度任务，不应被当作当前速度 cell 的样本。
    # _update_command 会在每一步将这些环境的实际命令置零。
    self.metrics["grid_success"][env_ids] = 0.0

    forward_ids = env_ids[self.is_forward_env[env_ids]]
    if len(forward_ids) > 0:
      self.vel_command_b[forward_ids, 0] = (
        self.vel_command_b[forward_ids, 0].abs().clamp(min=0.3)
      )
      self.vel_command_b[forward_ids, 1:] = 0.0

    self._segment_error_xy[env_ids] = 0.0
    self._segment_error_yaw[env_ids] = 0.0
    self._segment_gait_schedule_error[env_ids] = 0.0
    self._segment_gait_sync_error[env_ids] = 0.0
    self._segment_gait_mixed_contact[env_ids] = 0.0
    self._segment_steps[env_ids] = 0

  def _gait_quality_errors(
    self,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """计算 WTW 接触质量，避免课程与 reward 使用两套成功定义。

    此处只通过命令/传感器的公开属性访问 WTW，不导入 WTW 类，因而 Grid
    Adaptive 仍可独立用于没有 WTW 的 baseline。返回的三个量依次为：逐脚
    时序 L1 误差、四脚接触分歧、在高置信支撑/摆动区间的混合接触指示。
    """
    if self._grid_cfg.gait_quality_behavior_command_name is None:
      return None
    try:
      behavior = self._env.command_manager.get_term(
        self._grid_cfg.gait_quality_behavior_command_name
      )
      sensor = self._env.scene[self._grid_cfg.gait_quality_sensor_name]
      phase = behavior.phase
      duty_factor = float(getattr(behavior, "duty_factor", 0.5))
      found = sensor.data.found
    except (AttributeError, KeyError, TypeError):
      return None
    if phase is None or found is None:
      return None

    # 与 wtw_rewards._smooth_contact_target 使用相同的周期高斯 CDF 公式。
    sigma = max(float(self._grid_cfg.gait_contact_smoothing), 1.0e-3)
    duty = min(max(duty_factor, 1.0e-3), 1.0 - 1.0e-3)
    root_two = math.sqrt(2.0)

    def normal_cdf(value: torch.Tensor) -> torch.Tensor:
      return 0.5 * (1.0 + torch.erf(value / (sigma * root_two)))

    desired_contact = (
      normal_cdf(phase) * (1.0 - normal_cdf(phase - duty))
      + normal_cdf(phase - 1.0)
      * (1.0 - normal_cdf(phase - duty - 1.0))
    ).clamp(0.0, 1.0)
    actual_contact = (found > 0).to(dtype=desired_contact.dtype)
    schedule_error = torch.abs(actual_contact - desired_contact).mean(dim=1)

    # 非同步 gait 本来就不应让四脚接触相同，因此只在四腿期望接触几乎
    # 相同的 Pronking 类时计算同步门槛；其他 gait 的这两个量为零。
    synchronous = (
      desired_contact.amax(dim=1) - desired_contact.amin(dim=1) < 1.0e-3
    )
    sync_error = torch.abs(
      actual_contact[:, 1:] - actual_contact[:, :1]
    ).mean(dim=1) * synchronous
    high_confidence = (
      (desired_contact.mean(dim=1) > 0.9)
      | (desired_contact.mean(dim=1) < 0.1)
    )
    mixed_contact = (
      1.0 - actual_contact.prod(dim=1) - (1.0 - actual_contact).prod(dim=1)
    ) * (synchronous & high_confidence)
    return schedule_error, sync_error, mixed_contact

  def _update_metrics(self) -> None:
    # 保留 mjlab 原有的 error_vel_xy/error_vel_yaw metrics。
    super()._update_metrics()
    error_xy = torch.norm(
      self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2],
      dim=-1,
    )
    error_yaw = torch.abs(
      self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2]
    )
    self._segment_error_xy += error_xy
    self._segment_error_yaw += error_yaw
    gait_errors = self._gait_quality_errors()
    if gait_errors is not None:
      schedule_error, sync_error, mixed_contact = gait_errors
      self._segment_gait_schedule_error += schedule_error
      self._segment_gait_sync_error += sync_error
      self._segment_gait_mixed_contact += mixed_contact
    self._segment_steps += 1
    steps = self._segment_steps.clamp_min(1).float()
    self.metrics["grid_error_vel_xy"] = self._segment_error_xy / steps
    self.metrics["grid_error_vel_yaw"] = self._segment_error_yaw / steps
    self.metrics["grid_gait_schedule_error"] = (
      self._segment_gait_schedule_error / steps
    )
    self.metrics["grid_gait_sync_error"] = self._segment_gait_sync_error / steps
    self.metrics["grid_gait_mixed_contact"] = (
      self._segment_gait_mixed_contact / steps
    )

  def _settle_segments(self, env_ids: torch.Tensor) -> None:
    """结算命令段并根据成功率更新课程地图。"""
    if len(env_ids) == 0:
      return
    # standing 任务不参与速度网格难度评估，否则大量零速度成功会虚高
    # 非零速度 cell 的成功率，造成 curriculum 过早扩展。
    valid = (self._segment_steps[env_ids] > 0) & (
      ~self.is_standing_env[env_ids]
    )
    # 没有可用于网格统计的 segment 时，清理当前段状态后直接返回。
    if not valid.any():
      self._segment_error_xy[env_ids] = 0.0
      self._segment_error_yaw[env_ids] = 0.0
      self._segment_gait_schedule_error[env_ids] = 0.0
      self._segment_gait_sync_error[env_ids] = 0.0
      self._segment_gait_mixed_contact[env_ids] = 0.0
      self._segment_steps[env_ids] = 0
      return

    settled_ids = env_ids[valid]
    steps = self._segment_steps[settled_ids].float()
    mean_xy = self._segment_error_xy[settled_ids] / steps
    mean_yaw = self._segment_error_yaw[settled_ids] / steps
    mean_schedule_error = self._segment_gait_schedule_error[settled_ids] / steps
    mean_sync_error = self._segment_gait_sync_error[settled_ids] / steps
    mean_mixed_contact = self._segment_gait_mixed_contact[settled_ids] / steps

    # 提前摔倒算失败；正常 time_out 不直接算失败，由跟踪误差判定。
    terminated = getattr(self._env, "reset_terminated", None)
    if terminated is None:
      terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    success = (
      ~terminated[settled_ids]
      & (mean_xy <= self._grid_cfg.velocity_error_threshold)
      & (mean_yaw <= self._grid_cfg.yaw_error_threshold)
    )
    # 可选的 gait-aware gate：只要配置阈值，课程就不会把“速度能跟上、
    # 但仍是错峰 trot”的命令段判成 Pronking 成功。
    if self._grid_cfg.gait_schedule_error_threshold is not None:
      success &= mean_schedule_error <= self._grid_cfg.gait_schedule_error_threshold
    if self._grid_cfg.gait_sync_error_threshold is not None:
      success &= mean_sync_error <= self._grid_cfg.gait_sync_error_threshold
    if self._grid_cfg.gait_mixed_contact_threshold is not None:
      success &= mean_mixed_contact <= self._grid_cfg.gait_mixed_contact_threshold
    self.metrics["grid_success"][settled_ids] = success.float()

    cell_x = self.current_cell[settled_ids, 0]
    cell_yaw = self.current_cell[settled_ids, 1]
    ones = torch.ones_like(cell_x, dtype=torch.long)
    self.cell_visits.index_put_((cell_x, cell_yaw), ones, accumulate=True)
    self.cell_successes.index_put_((cell_x, cell_yaw), success.long(), accumulate=True)
    self._record_recent_results(cell_x, cell_yaw, success)
    self._expand_ready_cells()

    # 防止同一个 segment 在 reset 流程中被重复统计。
    self._segment_error_xy[env_ids] = 0.0
    self._segment_error_yaw[env_ids] = 0.0
    self._segment_gait_schedule_error[env_ids] = 0.0
    self._segment_gait_sync_error[env_ids] = 0.0
    self._segment_gait_mixed_contact[env_ids] = 0.0
    self._segment_steps[env_ids] = 0

  def _expand_ready_cells(self) -> None:
    """根据近期表现，有限数量地激活四连通邻居。"""
    current_step = int(self._env.common_step_counter)
    if self._last_expansion_step == current_step:
      return
    self._last_expansion_step = current_step

    recent_visits = self.cell_recent_visits.reshape_as(self.cell_visits)
    recent_successes = self.cell_recent_successes.reshape_as(self.cell_visits)
    recent_success_rate = recent_successes.float() / recent_visits.clamp_min(1.0)
    ready = (
      self.active_cells
      & (self.cell_visits >= self._grid_cfg.min_cell_visits)
      & (recent_visits >= self._grid_cfg.min_cell_visits)
      & (recent_success_rate >= self._grid_cfg.success_rate_threshold)
    )

    # 严格 frontier 模式用于逐 cell 课程：新激活的邻居在完成自己的
    # 访问量和成功率验证前，会阻止已经成熟的旧 cell 继续向外扩张。
    # 否则在大规模并行环境中，中心 cell 往往会在相邻两次 reset 中迅速
    # 打开多个方向，导致“新速度格尚未验证，课程已全部展开”。
    if self._grid_cfg.require_all_active_cells_ready and (
      self.active_cells & ~ready
    ).any():
      return

    # 使用确定性顺序选取候选 cell，保证相同 checkpoint 能复现实验轨迹。
    candidate_ids: set[tuple[int, int]] = set()
    ready_ids = torch.nonzero(ready, as_tuple=False).tolist()
    for x_index, yaw_index in ready_ids:
      for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        next_x = x_index + dx
        next_yaw = yaw_index + dy
        if (
          0 <= next_x < self._grid_cfg.grid_num_x
          and 0 <= next_yaw < self._grid_cfg.grid_num_yaw
          and not self.active_cells[next_x, next_yaw]
        ):
          candidate_ids.add((next_x, next_yaw))

    selected_ids = sorted(candidate_ids)[: self._grid_cfg.max_new_cells_per_update]
    for x_index, yaw_index in selected_ids:
      self.active_cells[x_index, yaw_index] = True

  def _record_recent_results(
    self,
    cell_x: torch.Tensor,
    cell_yaw: torch.Tensor,
    success: torch.Tensor,
  ) -> None:
    """将本批次结果写入各 cell 独立的近期成功率 ring buffer。"""
    flat_cell_ids = cell_x * self._grid_cfg.grid_num_yaw + cell_yaw
    window_size = self._grid_cfg.success_window_size

    # 同一批次内可能有多个环境属于同一个 cell。先按 cell 分组，再一次性
    # 写入该组结果，避免直接 scatter 时多个环境争用同一个 ring 指针。
    for cell_id in torch.unique(flat_cell_ids).tolist():
      group_mask = flat_cell_ids == cell_id
      group_success = success[group_mask].bool()
      num_results = len(group_success)
      ptr = int(self.cell_history_ptr[cell_id].item())
      previous_visits = int(self.cell_recent_visits[cell_id].item())

      if num_results >= window_size:
        # 本批次已经覆盖整个窗口，最终状态只由最后 window_size 个结果决定。
        final_results = group_success[-window_size:]
        start = (ptr + num_results - window_size) % window_size
        positions = (
          torch.arange(window_size, device=self.device) + start
        ) % window_size
        self.cell_success_history[cell_id, positions] = final_results
        self.cell_recent_successes[cell_id] = final_results.long().sum()
        self.cell_recent_visits[cell_id] = window_size
      else:
        positions = (
          torch.arange(num_results, device=self.device) + ptr
        ) % window_size
        overwritten = max(0, previous_visits + num_results - window_size)
        removed_successes = 0
        if overwritten > 0:
          removed_successes = int(
            self.cell_success_history[cell_id, positions[-overwritten:]]
            .long()
            .sum()
            .item()
          )
        self.cell_success_history[cell_id, positions] = group_success
        self.cell_recent_successes[cell_id] = (
          self.cell_recent_successes[cell_id]
          + group_success.long().sum()
          - removed_successes
        )
        self.cell_recent_visits[cell_id] = min(
          window_size, previous_visits + num_results
        )

      self.cell_history_ptr[cell_id] = (ptr + num_results) % window_size

  def curriculum_state_dict(self) -> dict[str, Any]:
    """返回可写入 checkpoint 的课程状态。"""
    state: dict[str, Any] = {
      "version": 3,
      "active_cells": self.active_cells.detach().cpu().clone(),
      "cell_visits": self.cell_visits.detach().cpu().clone(),
      "cell_successes": self.cell_successes.detach().cpu().clone(),
      "cell_success_history": self.cell_success_history.detach().cpu().clone(),
      "cell_recent_visits": self.cell_recent_visits.detach().cpu().clone(),
      "cell_recent_successes": self.cell_recent_successes.detach().cpu().clone(),
      "cell_history_ptr": self.cell_history_ptr.detach().cpu().clone(),
      "current_cell": self.current_cell.detach().cpu().clone(),
      "vel_command_b": self.vel_command_b.detach().cpu().clone(),
      "vel_command_w": self.vel_command_w.detach().cpu().clone(),
      "time_left": self.time_left.detach().cpu().clone(),
      "command_counter": self.command_counter.detach().cpu().clone(),
      "is_standing_env": self.is_standing_env.detach().cpu().clone(),
      "is_world_env": self.is_world_env.detach().cpu().clone(),
      "is_forward_env": self.is_forward_env.detach().cpu().clone(),
      "segment_error_xy": self._segment_error_xy.detach().cpu().clone(),
      "segment_error_yaw": self._segment_error_yaw.detach().cpu().clone(),
      "segment_gait_schedule_error": self._segment_gait_schedule_error.detach().cpu().clone(),
      "segment_gait_sync_error": self._segment_gait_sync_error.detach().cpu().clone(),
      "segment_gait_mixed_contact": self._segment_gait_mixed_contact.detach().cpu().clone(),
      "segment_steps": self._segment_steps.detach().cpu().clone(),
      "last_expansion_step": self._last_expansion_step,
    }
    return state

  def load_curriculum_state_dict(self, state: dict[str, Any]) -> None:
    """从 checkpoint 恢复课程状态；兼容没有近期窗口的旧状态。"""
    # checkpoint 中的字段名保持简洁，但命令段累计量在类中使用下划线表示
    # 内部状态，因此这里显式建立字段名到成员变量的映射。
    # 网格统计是课程的核心状态，形状不一致通常意味着配置发生了变化。
    grid_targets = {
      "active_cells": self.active_cells,
      "cell_visits": self.cell_visits,
      "cell_successes": self.cell_successes,
      "cell_success_history": self.cell_success_history,
      "cell_recent_visits": self.cell_recent_visits,
      "cell_recent_successes": self.cell_recent_successes,
      "cell_history_ptr": self.cell_history_ptr,
    }
    # 逐环境状态在训练和 play 之间可能有不同形状（例如 4096 -> 1），
    # 这种情况下跳过恢复，下一次 reset 会重新生成当前环境的运行状态。
    runtime_targets = {
      "current_cell": self.current_cell,
      "vel_command_b": self.vel_command_b,
      "vel_command_w": self.vel_command_w,
      "time_left": self.time_left,
      "command_counter": self.command_counter,
      "is_standing_env": self.is_standing_env,
      "is_world_env": self.is_world_env,
      "is_forward_env": self.is_forward_env,
      "segment_error_xy": self._segment_error_xy,
      "segment_error_yaw": self._segment_error_yaw,
      "segment_gait_schedule_error": self._segment_gait_schedule_error,
      "segment_gait_sync_error": self._segment_gait_sync_error,
      "segment_gait_mixed_contact": self._segment_gait_mixed_contact,
      "segment_steps": self._segment_steps,
    }

    for name, target in grid_targets.items():
      value = state.get(name)
      if value is None:
        continue
      if not isinstance(value, torch.Tensor) or value.shape != target.shape:
        raise ValueError(
          f"Invalid Grid Adaptive state for '{name}': "
          f"expected tensor shape {tuple(target.shape)}."
        )
      target.copy_(value.to(device=self.device, dtype=target.dtype))

    for name, target in runtime_targets.items():
      value = state.get(name)
      if value is None:
        continue
      if not isinstance(value, torch.Tensor) or value.shape != target.shape:
        continue
      target.copy_(value.to(device=self.device, dtype=target.dtype))

    last_expansion_step = state.get("last_expansion_step")
    if last_expansion_step is not None:
      self._last_expansion_step = int(last_expansion_step)

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    # 先结算旧段，再调用父类清理 metrics 并采样新命令。
    assert isinstance(env_ids, torch.Tensor)
    self._settle_segments(env_ids)
    extras = super().reset(env_ids)
    self._segment_error_xy[env_ids] = 0.0
    self._segment_error_yaw[env_ids] = 0.0
    self._segment_gait_schedule_error[env_ids] = 0.0
    self._segment_gait_sync_error[env_ids] = 0.0
    self._segment_gait_mixed_contact[env_ids] = 0.0
    self._segment_steps[env_ids] = 0
    extras["grid_active_cells"] = float(self.active_cells.sum().item())
    extras["grid_total_visits"] = float(self.cell_visits.sum().item())
    return extras
