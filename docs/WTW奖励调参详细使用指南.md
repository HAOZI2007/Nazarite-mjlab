# WTW 奖励调参详细使用指南

> 对应当前 `Nazarite-Velocity-Flat-Go2-WTW`。当前默认目标为单一 Trot、x 速度三格 Grid、频率 `2.0–2.4 Hz`。WTW 使用独立 reward term，不存在也不应再调旧的 `wtw_combined`。

## 1. 调参位置与原则

奖励定义集中在：

```text
Train/Nazarite/Nazarite-src/nazarite/config/train_config/base_env_cfg.py
```

- `_make_wtw_rewards()`：所有 WTW 奖励、权重与函数参数；
- `_configure_wtw_rewards()`：移除会冲突的 baseline gait 奖励并注册 WTW 项；
- `env_cfgs/go2_env_cfgs.py`：将 Go2 足端 site 绑定给 Raibert 奖励，并设置通用安全奖励、传感器和终止条件；
- `mdp/wtw_rewards.py`：奖励函数和 TensorBoard 日志；
- `mdp/commands.py`：Grid 的 gait-aware 成功门槛，不能只看总 reward。

调参的基本单位是“一次只改一个行为假设”。一次同时扩大频率、修改高度、调接触权重和改变 Grid，会使结果无法归因。

## 2. 当前奖励表

| term | 权重 | 物理含义 | Trot 下的主要日志 |
|---|---:|---|---|
| `track_linear_velocity` | +2.0 | 跟踪 x/y 速度 | `Metrics/twist/error_vel_xy` |
| `track_angular_velocity` | +2.0 | 跟踪 yaw 速度 | `Metrics/twist/error_vel_yaw` |
| `wtw_swing_phase_force` | -4.0 | 摆动腿不应明显受力 | `WTW/swing_phase_force_cost`、四脚 swing force |
| `wtw_stance_phase_velocity` | -4.0 | 支撑腿足端水平速度应小 | `WTW/stance_phase_velocity_cost` |
| `wtw_contact_schedule` | -1.5 | 实际四脚接触向量匹配 phase | `WTW/contact_schedule_error`、四脚 schedule error |
| `wtw_group_contact_consistency` | -0.5 | 同步 gait 的高置信 phase 不应混合接触 | 仅 Pronking 类 gait 有意义 |
| `wtw_body_height` | +40.0 | 跟踪 `0.32 + height_offset` | `WTW/body_height_*` |
| `wtw_body_pitch` | +0.10 | 跟踪 pitch 行为命令 | reward 曲线与 play 姿态 |
| `wtw_foot_clearance` | -30.0 | 摆动相连续足端高度轨迹 | `WTW/foot_clearance_cost` |
| `wtw_raibert_foot_position` | -10.0 | 速度/phase 感知的落点误差 | `WTW/raibert_foot_position_cost` |
| `wtw_shank_contact` | -0.1 | 小腿触地软惩罚 | `WTW/shank_contact_cost` |

权重的绝对值不能跨项直接比较。`wtw_foot_clearance` 返回的是四足平方误差和，因此 `-30` 未必比返回 `[0,1]` 代价的 `-4` 更强。判断一项是否主导，必须同时查看未加权日志、episode reward 和 play 行为。

以下项在 WTW 中明确不启用：

```text
air_time
prolonged_air_time
stance_contact
wtw_stance_contact
wtw_stance_width
wtw_foot_swing_height
wtw_combined
```

前三项不读取 phase，容易与指定摆动/支撑结构冲突。`wtw_stance_width` 与旧的落脚峰值高度项分别被 Raibert 落点和连续 clearance 项取代。

## 3. active mask、phase 冻结与站立

phase 相关 WTW 项共享：

```text
active = norm([vx, vy]) + abs(wz) > 0.05
```

非 active 时，phase 冻结，摆动相力、支撑相速度、schedule、group consistency、clearance、Raibert、pitch 和小腿接触项都不再优化 gait。零速度的姿态由 `stand_pose` 维持；`zero_command_stillness` 当前权重为 `0.0`，只保留实现和日志，不参与优化。

这意味着低速 cell 并不等于真正站立：只要命令幅值大于 0.05，phase 仍在推进。若低速 cell 难学，先检查它是否频繁落在阈值附近，再决定改阈值或设计专门站立课程。

## 4. 三层接触约束应该如何协作

项目用 `duty_factor=0.5` 和 `smoothing=0.07` 生成平滑的 `desired_contact ∈ [0,1]`：前半周期趋向支撑，后半周期趋向摆动。三个主要接触项层级不同，不能互相替代。

### 4.1 摆动相接触力：`wtw_swing_phase_force`

近似为：

```text
mean((1 - desired_contact) × (1 - exp(-force² / force_std)))
```

当前 `force_std=100`。它关心“摆动腿是否拖地/撞地”，不保证支撑腿接触正确。

- 某一脚长期最高：先检查 foot 顺序、传感器映射和相位映射；
- 四脚都偏高且机器人刻意跳高：增加 `force_std` 或略减小负权重；
- 摆动脚频繁拖地但姿态正常：先小幅减小 `force_std`，再考虑把权重从 `-4` 调到 `-4.5`。

### 4.2 支撑相足端速度：`wtw_stance_phase_velocity`

近似为：

```text
mean(desired_contact × (1 - exp(-foot_velocity_xy² / velocity_std)))
```

当前 `velocity_std=10`。它抑制足端滑动，但不直接要求该脚必须接触地面。

- 支撑脚打滑：优先减小 `velocity_std`，例如 `10 → 8`；
- 步态僵硬、拒绝前进：先将权重绝对值略减小，而不是继续加大；
- 高速时打滑：同时看 Raibert 项、摩擦随机化和速度命令范围。

### 4.3 四脚 schedule：`wtw_contact_schedule`

```text
cost = mean_i |actual_contact_i - desired_contact_i|
```

它才是 Trot 的结构验收项：FR/RL 与 FL/RR 应按相反 phase 交替。当前后期较好的 Trot 结果中该误差约为 `0.12`，且 Grid 成功门槛为 `0.16`。

- `>0.16`：课程不会把该命令段认定为成功，应先查 gait 结构或过快频率；
- 四脚日志长期不对称：首先查腿顺序和 site/sensor，不要直接改权重；
- 数值下降但 play 仍像非 Trot：检查 phase 是否冻结、速度命令是否真的有效；
- 时序正确但机器人不稳定：再检查高度、Raibert、推力和终止比例。

`wtw_group_contact_consistency` 面向四脚期望同步的 gait。当前 Trot 下四足期望接触不相同，因此其相关的 `pronking_*` 日志为零是正确现象，不代表奖励失效。

## 5. 行为风格项的调法

### 高度：`wtw_body_height`

函数为官方 jump 高度语义的负平方误差：

```text
target = 0.32 + behavior.body_height_offset
reward = -(actual_height - target)²
```

当前权重为 `+40`。不要把它理解成“越高越好”。应同时查看：

```text
WTW/body_height_actual
WTW/body_height_target
WTW/body_height_signed_error
WTW/body_height_error
WTW/body_height_min
```

`2026-08-30_19-18-04` 的宽频 `2–3 Hz` Trot 平均高度约 `0.343 m`，目标是 `0.32 m`，出现约 `+0.023 m` 稳定上漂。固定 `2 Hz` 的此前运行约为 `0.319 m`。因此面对高度上漂，优先顺序是：

1. 收窄频率范围（当前默认已收为 `2.0–2.4 Hz`）；
2. 用网页 Behavior 面板固定频率分别验证高度；
3. 若同一频率仍稳定上漂，再小幅提高高度项权重或调整基础目标；
4. 不要在同一轮同时改接触 reward，否则难以判断原因。

### Clearance：`wtw_foot_clearance`

该项在整个摆动相监督 `0 → foot_swing_height + foot_radius → 0`，当前 `foot_swing_height=0.06 m`、`foot_radius=0.02 m`。它比“落脚瞬间只看峰值”稳定。

- cost 很低而脚仍拖地：检查 contact force 和实际足端接触时机；
- cost 高且机器人过度抬腿：减小摆腿高度命令，或降低 `-30` 的绝对值；
- 不要重新叠加旧 `wtw_foot_swing_height`，它会提供第二套不同的高度监督。

### Raibert 落点：`wtw_raibert_foot_position`

该项使用速度命令、实际速度误差、phase、频率、yaw 补偿和名义落点构造四脚目标。它包含名义站宽含义，因此当前不额外启用 `wtw_stance_width`。

前后腿向相反方向撇，排查顺序应是：腿顺序 → theta/phase 映射 → 是否训练过横向/yaw → Raibert 项。只有前三项确定正确后再微调权重或 `stance_length`。

## 6. 课程参数不是 reward，但必须一起看

当前 WTW Grid：

```python
grid_num_x = 3
grid_num_yaw = 1
lin_vel_x = (-1.0, 1.0)
initial_cell = (1, 0)
min_cell_visits = success_window_size = 8192
success_rate_threshold = 0.8
require_all_active_cells_ready = True
velocity_error_threshold = 0.25
gait_schedule_error_threshold = 0.16
```

`grid_success` 是一个命令段是否同时满足速度、终止和 gait 时序门槛的结果，不是 reward。`grid_active_cells` 应从 `1 → 2 → 3` 串行增长；短暂显示小数是不同环境 reset 时的日志平均，不表示存在“半个 cell”。

分析 checkpoint 时应看每格近期窗口成功率，而非仅看累计成功率。中间低速 cell 的累计指标会被训练早期样本拉低，近期 `8192` 窗口才描述当前策略水平。

## 7. 推荐实验顺序

| 阶段 | 只改变的变量 | 验收 |
|---|---|---|
| A | 固定 Trot、固定 2 Hz | 速度、时序、站立都稳定。 |
| B | 频率 `2.0–2.4 Hz` | 全部 3 cell 激活，schedule `<0.16`，高度不系统上漂。 |
| C | 仅开放一个风格参数 | 对应误差跟随、速度和时序不退化。 |
| D | 新 gait 的独立任务 | 使用该 gait 自己的时序指标。 |
| E | 多 gait 条件策略 | 所有 gait 组合采样且分别评估。 |

每一轮训练结束至少检查：

```text
Train/mean_episode_length
Metrics/twist/grid_active_cells
Metrics/twist/grid_success
Metrics/twist/error_vel_xy
Metrics/twist/grid_gait_schedule_error
WTW/contact_schedule_error
WTW/body_height_actual / target / signed_error
WTW/swing_phase_force_cost
WTW/stance_phase_velocity_cost
Episode_Termination/*
```

总 reward 上升不是单独的验收标准；它可能来自降低动作惩罚，却掩盖接触时序或高度偏置的恶化。
