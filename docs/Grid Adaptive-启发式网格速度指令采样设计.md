# Grid Adaptive-启发式网格速度指令采样设计

本文档介绍 Nazarite-mjlab 项目中的 Grid Adaptive 速度指令采样器，包括设计动机、网格建模、代码执行流程、成功率统计、课程扩展、checkpoint 保存恢复、参数含义、调参方法，以及当前实现的限制。

对应实现文件：

- Train/Nazarite/Nazarite-src/nazarite/mdp/commands.py
- Train/Nazarite/Nazarite-src/nazarite/config/train_config/base_env_cfg.py
- Train/Nazarite/Nazarite-src/nazarite/config/train_config/env_cfgs.py
- Train/Nazarite/mjlab/src/mjlab/rl/runner.py

---

## 1. 设计目标

普通的均匀速度指令采样，通常直接在完整速度范围内随机采样：

~~~text
lin_vel_x ∈ [-2.0, 3.0]
lin_vel_y ∈ [-1.0, 1.0]
ang_vel_z ∈ [-0.7, 0.7]
~~~

这种方式实现简单，但训练早期可能直接采到高难度速度，机器人容易摔倒；同时，训练是否已经掌握某一速度区域也无法反馈给采样器。

Grid Adaptive 的核心思想是：

> 将速度命令空间离散成二维网格，只激活一个或少量简单 cell；机器人在当前 cell 上达到足够访问量和成功率后，再激活它的邻居。

因此，课程难度由机器人的实际表现驱动，而不是完全由固定训练步数驱动。

Grid Adaptive 主要解决以下问题：

1. 避免训练初期直接面对完整速度范围。
2. 将速度能力分解为可观察、可统计的局部区域。
3. 让课程扩展与机器人成功率关联。
4. 防止并行环境一次性把课程扩展得过快。
5. 将课程地图保存到 checkpoint，支持断点续训。

---

## 2. 总体结构

当前实现继承自 mjlab 的 UniformVelocityCommand，复用了父类的命令计时器、命令管理器接口、命令观测和基础命令更新流程。

~~~mermaid
flowchart TD
    A[环境初始化] --> B[创建 GridAdaptiveVelocityCommand]
    B --> C[创建 active_cells 和统计量]
    C --> D[激活 initial_cell]
    D --> E[为环境采样 active cell]
    E --> F[在 cell 内采样连续速度]
    F --> G[机器人执行命令段]
    G --> H{命令到期或环境 reset?}
    H -- 否 --> G
    H -- 是 --> I[结算当前命令段]
    I --> J[更新累计统计]
    J --> K[更新近期成功率窗口]
    K --> L{是否满足扩展条件?}
    L -- 否 --> E
    L -- 是 --> M[激活有限数量四连通邻居]
    M --> E
~~~

所有并行环境共享一张课程地图，但每个环境可以拥有不同的当前 cell 和不同的连续速度命令。

---

## 3. 网格如何定义速度空间

### 3.1 网格维度

当前网格离散两个速度维度：

~~~text
网格 x 轴     → lin_vel_x，前向/后向线速度
网格 yaw 轴   → ang_vel_z，偏航角速度
~~~

横向速度 lin_vel_y 不参与网格扩展，而是在选定 cell 后独立均匀采样。

当前 Go2 配置为：

~~~python
grid_num_x=9
grid_num_yaw=7
ranges=GridAdaptiveVelocityCommandCfg.Ranges(
  lin_vel_x=(-2.0, 3.0),
  lin_vel_y=(-1.0, 1.0),
  ang_vel_z=(-0.7, 0.7),
  heading=None,
)
~~~

总 cell 数为：

~~~text
9 × 7 = 63 个 cell
~~~

### 3.2 网格边界

_grid_edges() 使用 torch.linspace() 将完整范围均匀切分：

~~~python
return torch.linspace(
  value_range[0], value_range[1], num_bins + 1, device=self.device
)
~~~

当前配置中：

~~~text
lin_vel_x 单个 cell 宽度 = (3.0 - (-2.0)) / 9 ≈ 0.556 m/s
ang_vel_z 单个 cell 宽度   = (0.7 - (-0.7)) / 7 = 0.2 rad/s
~~~

cell (x_index, yaw_index) 的采样区间为：

~~~text
lin_vel_x ∈ [x_edges[x_index], x_edges[x_index + 1]]
ang_vel_z ∈ [yaw_edges[yaw_index], yaw_edges[yaw_index + 1]]
~~~

例如初始 cell (3, 3) 大致覆盖：

~~~text
lin_vel_x ≈ [-0.333, 0.222] m/s
ang_vel_z = [-0.1, 0.1] rad/s
~~~

注意：(3, 3) 不是严格的零速度 cell，只是接近零速度的区域。

### 3.3 横向速度的独立采样

当前实现中：

~~~python
self.vel_command_b[env_ids, 1] = self._sample_uniform(
  torch.full_like(x_low, self._grid_cfg.ranges.lin_vel_y[0]),
  torch.full_like(x_low, self._grid_cfg.ranges.lin_vel_y[1]),
)
~~~

这意味着即使 lin_vel_x≈0，lin_vel_y 仍可能接近 -1.0 或 1.0。

因此：

~~~text
lin_vel_x≈0 ≠ 完整速度指令为 0
~~~

只有 standing 任务会将三个速度分量全部置为零。如果项目主要训练前后运动和转向，可以先把横向范围收窄：

~~~python
lin_vel_y=(-0.3, 0.3)
~~~

---

## 4. Cell 的采样概率

### 4.1 激活 cell 之间均匀采样

_sample_cells() 首先获取所有激活 cell：

~~~python
active_ids = torch.nonzero(
  self.active_cells.flatten(),
  as_tuple=False,
).flatten()
~~~

然后使用 torch.randint() 选择其中一个：

~~~python
selected = active_ids[
  torch.randint(len(active_ids), (len(env_ids),), device=self.device)
]
~~~

如果当前有 N_active 个激活 cell，则单个环境采到任意激活 cell 的概率为：

~~~text
P(cell_i) = 1 / N_active
~~~

例如：

~~~text
激活 1 个 cell：该 cell 概率 100%
激活 4 个 cell：每个 cell 概率 25%
激活 20 个 cell：每个 cell 概率 5%
激活 63 个 cell：每个 cell 概率约 1.587%
~~~

当前实现不是按成功率加权采样，而是对当前 active map 中的 cell 均匀采样。

### 4.2 Cell 内部连续采样

选中 cell 后，lin_vel_x 和 ang_vel_z 分别在各自 cell 区间内均匀采样：

~~~python
self.vel_command_b[env_ids, 0] = self._sample_uniform(x_low, x_high)
self.vel_command_b[env_ids, 2] = self._sample_uniform(yaw_low, yaw_high)
~~~

所以一个命令的采样过程是：

~~~text
先以 1/N_active 的概率选 cell
再在该 cell 的矩形速度区域内连续均匀采样
~~~

---

## 5. 命令段生命周期

### 5.1 命令持续时间

父类 CommandTerm._resample() 负责设置命令计时器：

~~~python
self.time_left[env_ids] = self.time_left[env_ids].uniform_(
  *self.cfg.resampling_time_range
)
~~~

当前 Go2 配置：

~~~python
resampling_time_range=(8.0, 12.0)
~~~

表示每个环境的命令段持续时间从 8～12 秒之间独立均匀采样。

在每个环境步中，计时器减去环境步长：

~~~python
self.time_left -= dt
~~~

当：

~~~python
self.time_left <= 0.0
~~~

该环境就会重新采样命令。当前环境步长为 0.02 s，所以实际切换时间会量化到约 0.02 s 的步长。

### 5.2 命令重新采样流程

每次命令重新采样时，Grid Adaptive 依次执行：

1. 结算旧命令段。
2. 根据 active map 选择新 cell。
3. 根据 cell 边界采样 lin_vel_x 和 ang_vel_z。
4. 独立采样 lin_vel_y。
5. 采样 standing/world/forward 标志。
6. 清理新命令段的误差统计。

核心入口是：

~~~python
def _resample_command(self, env_ids: torch.Tensor) -> None:
  self._settle_segments(env_ids)
  self._sample_cells(env_ids)
  ...
~~~

### 5.3 Standing 任务

当前配置：

~~~python
rel_standing_envs=0.1
~~~

每次命令采样时，约 10% 的环境被标记为 standing：

~~~python
self.is_standing_env[env_ids] = r <= self.cfg.rel_standing_envs
~~~

父类 _update_command() 会把 standing 环境的命令置零：

~~~python
self.vel_command_b[standing_env_ids, :] = 0.0
self.vel_command_w[standing_env_ids, :] = 0.0
~~~

standing 任务训练保持姿态，但不参与速度 cell 的访问/成功统计：

~~~python
valid = (self._segment_steps[env_ids] > 0) & (
  ~self.is_standing_env[env_ids]
)
~~~

这样可以防止大量简单零速度成功样本抬高速度课程成功率。

---

## 6. 命令段成功判定

### 6.1 误差统计

每个环境维护当前命令段的：

~~~python
_segment_error_xy
_segment_error_yaw
_segment_steps
~~~

每个仿真步计算：

~~~python
error_xy = torch.norm(
  self.vel_command_b[:, :2]
  - self.robot.data.root_link_lin_vel_b[:, :2],
  dim=-1,
)

error_yaw = torch.abs(
  self.vel_command_b[:, 2]
  - self.robot.data.root_link_ang_vel_b[:, 2],
)
~~~

命令段结束时计算平均误差：

~~~python
mean_xy = _segment_error_xy / _segment_steps
mean_yaw = _segment_error_yaw / _segment_steps
~~~

### 6.2 成功条件

当前配置：

~~~python
velocity_error_threshold=0.35
yaw_error_threshold=0.35
~~~

成功条件为：

~~~python
success = (
  ~terminated[settled_ids]
  & (mean_xy <= velocity_error_threshold)
  & (mean_yaw <= yaw_error_threshold)
)
~~~

一个命令段只有同时满足以下条件才算成功：

1. 机器人没有在命令段中途摔倒终止。
2. 平均平面速度误差不超过 0.35 m/s。
3. 平均偏航角速度误差不超过 0.35 rad/s。

当前实现对超时结束不直接判失败，主要由跟踪误差决定；如果是 reset_terminated=True 的摔倒终止，则直接失败。

### 6.3 全生命周期统计

每个 cell 保存：

~~~python
cell_visits
cell_successes
~~~

结算一个命令段时，根据当前 cell 累加：

~~~python
self.cell_visits.index_put_((cell_x, cell_yaw), ones, accumulate=True)
self.cell_successes.index_put_(
  (cell_x, cell_yaw),
  success.long(),
  accumulate=True,
)
~~~

全生命周期成功率为：

~~~text
success_rate = cell_successes / max(cell_visits, 1)
~~~

它稳定，但早期失败会永久保留，后续成功也无法完全消除早期影响。

---

## 7. 近期成功率 Ring Buffer

为解决全生命周期统计过于僵化的问题，当前实现还为每个 cell 保存近期命令段结果。

### 7.1 数据结构

当前配置：

~~~python
success_window_size=100
~~~

内部状态为：

~~~python
cell_success_history  # [num_cells, window_size]，bool
cell_recent_visits    # [num_cells]
cell_recent_successes # [num_cells]
cell_history_ptr      # [num_cells]
~~~

每个 cell 有独立 ring buffer，不同 cell 的近期成功率互不污染。

### 7.2 写入流程

_record_recent_results() 会执行：

1. 将二维 cell 坐标展平为 cell id。
2. 将同一批次中属于同一个 cell 的环境分组。
3. 使用该 cell 自己的写指针写入结果。
4. 如果窗口已满，扣除被覆盖的旧结果。
5. 更新近期访问数和近期成功数。
6. 更新写指针。

近期成功率为：

~~~text
recent_success_rate
  = cell_recent_successes / max(cell_recent_visits, 1)
~~~

### 7.3 最低近期样本数

不能只因为某个 cell 最近 1 次成功就立即扩展，否则小样本偶然性会使课程过早扩散。

因此扩展条件同时要求：

~~~python
cell_visits >= min_cell_visits
recent_visits >= min_cell_visits
recent_success_rate >= success_rate_threshold
~~~

当前 Go2 配置：

~~~python
min_cell_visits=50
success_window_size=100
~~~

表示至少需要 50 次累计访问，并且近期窗口中至少有 50 次有效结果，才可能扩展。


## 8. Cell 扩展算法

### 8.1 四连通邻居

每个 ready cell 只向上下左右四个方向扩展：

~~~python
(-1, 0)  # lin_vel_x 减小
( 1, 0)  # lin_vel_x 增大
( 0,-1)  # ang_vel_z 减小
( 0, 1)  # ang_vel_z 增大
~~~

不使用对角线扩展，是因为对角线同时增加前向速度和转向难度，跨度更大，不利于稳定课程。

### 8.2 Ready 条件

~~~python
ready = (
  self.active_cells
  & (self.cell_visits >= self._grid_cfg.min_cell_visits)
  & (recent_visits >= self._grid_cfg.min_cell_visits)
  & (recent_success_rate >= self._grid_cfg.success_rate_threshold)
)
~~~

只有已经激活且表现达标的 cell 才能扩展邻居。

### 8.3 扩展数量限制

当前配置：

~~~python
max_new_cells_per_update=4
~~~

候选 cell 会先去重，再排序，最后只选前 4 个：

~~~python
selected_ids = sorted(candidate_ids)[
  : self._grid_cfg.max_new_cells_per_update
]
~~~

这样可以避免一次结算大量并行环境时，课程从：

~~~text
1 → 5 → 20 → 63
~~~

瞬间跳到很大的范围。

### 8.4 防止同一个仿真步重复扩展

由于多个并行环境可能在同一个仿真步同时 reset，_settle_segments() 可能被多次调用。

因此使用：

~~~python
self._last_expansion_step
~~~

并检查：

~~~python
current_step = int(self._env.common_step_counter)
if self._last_expansion_step == current_step:
  return
~~~

这个保护保证每个仿真步最多执行一次课程扩展。

---

## 9. 当前 Go2 配置解释

当前核心配置如下：

~~~python
"twist": GridAdaptiveVelocityCommandCfg(
  entity_name="robot",
  resampling_time_range=(8.0, 12.0),
  rel_standing_envs=0.1,
  rel_heading_envs=0.0,
  rel_forward_envs=0.0,
  heading_command=False,
  grid_num_x=9,
  grid_num_yaw=7,
  initial_cell=(3, 3),
  min_cell_visits=50,
  success_window_size=100,
  max_new_cells_per_update=4,
  success_rate_threshold=0.8,
  velocity_error_threshold=0.35,
  yaw_error_threshold=0.35,
  ranges=GridAdaptiveVelocityCommandCfg.Ranges(
    lin_vel_x=(-2.0, 3.0),
    lin_vel_y=(-1.0, 1.0),
    ang_vel_z=(-0.7, 0.7),
    heading=None,
  ),
)
~~~

### 9.1 参数表

| 参数 | 当前值 | 作用 |
|---|---:|---|
| grid_num_x | 9 | lin_vel_x 方向的 cell 数量 |
| grid_num_yaw | 7 | ang_vel_z 方向的 cell 数量 |
| initial_cell | (3, 3) | 初始激活 cell，默认接近零速度 |
| ranges.lin_vel_x | (-2.0, 3.0) | 最终前后速度覆盖范围 |
| ranges.lin_vel_y | (-1.0, 1.0) | 横向速度独立采样范围 |
| ranges.ang_vel_z | (-0.7, 0.7) | 最终偏航速度覆盖范围 |
| resampling_time_range | (8.0, 12.0) | 每个命令段持续时间，单位秒 |
| rel_standing_envs | 0.1 | standing 任务比例 |
| rel_heading_envs | 0.0 | heading 控制任务比例，Grid Adaptive 要求为 0 |
| rel_forward_envs | 0.0 | forward-only 任务比例 |
| heading_command | False | Grid Adaptive 当前不使用 heading target |
| min_cell_visits | 50 | cell 达到扩展条件所需的累计/近期最低访问数 |
| success_window_size | 100 | 近期成功率窗口大小 |
| success_rate_threshold | 0.8 | 近期成功率扩展阈值 |
| max_new_cells_per_update | 4 | 每个仿真步最多新增 cell 数 |
| velocity_error_threshold | 0.35 | 平面速度平均误差阈值，单位 m/s |
| yaw_error_threshold | 0.35 | 偏航速度平均误差阈值，单位 rad/s |

### 9.2 参数约束

配置初始化时会检查：

~~~text
grid_num_x >= 1
grid_num_yaw >= 1
min_cell_visits >= 1
success_window_size >= 1
max_new_cells_per_update >= 1
0 < success_rate_threshold <= 1
velocity_error_threshold > 0
yaw_error_threshold > 0
~~~

以下组合不允许：

~~~python
heading_command=True
rel_heading_envs != 0.0
~~~

因为当前 Grid Adaptive 的网格定义是 lin_vel_x × ang_vel_z，而不是 heading target 课程。

---

## 10. 调参方法

调参时建议按照“先保证可学习，再提高难度”的顺序进行，不要同时修改所有参数。

### 10.1 第一步：确认初始 cell 能稳定完成

先观察：

~~~text
grid_success
grid_active_cells
grid_total_visits
~~~

理想状态：

~~~text
grid_success 稳定高于 0.8
grid_active_cells 从 1 开始缓慢增长
grid_total_visits 持续增加
fell_over 逐渐降低
~~~

如果初始 cell 都无法成功，先检查 domain randomization、摔倒终止和奖励配置，不要直接扩大网格。

### 10.2 success_rate_threshold

该参数控制扩展是否保守：

~~~text
0.90～0.95：非常保守，课程扩展慢，但稳定
0.80～0.85：推荐起点
0.65～0.75：扩展较快，但可能过早引入困难 cell
~~~

如果 active cell 长时间停留在 1：

1. 确认 grid_success 是否真的低。
2. 确认 min_cell_visits 是否已经达到。
3. 再考虑将阈值从 0.8 降到 0.75。

### 10.3 min_cell_visits

该参数决定统计可靠性：

~~~text
10～20：扩展快，统计噪声大
30～50：常用范围
50～100：更稳定，但训练时间更长
~~~

当前使用：

~~~python
min_cell_visits=50
~~~

如果命令段很长、访问增长慢，可以降低到 30；如果 domain randomization 较强，建议保持 50 或提高到 80。

### 10.4 success_window_size

窗口越小，课程对最近表现越敏感；窗口越大，课程越稳定：

~~~text
50：响应快，容易受偶然成功/失败影响
100：当前推荐值
200：稳定，但对能力变化反应慢
~~~

建议满足：

~~~text
success_window_size >= min_cell_visits
~~~

否则近期成功率样本不足，扩展会有较大随机性。

### 10.5 max_new_cells_per_update

该参数控制课程地图扩展速度：

~~~text
1～2：非常平滑，适合高难度任务
4：当前 Go2 推荐值
8 以上：扩展快，可能引入大量未掌握速度
~~~

如果训练表现为 active cell 快速增长、随后 grid_success 快速下降，可以改为：

~~~python
max_new_cells_per_update=1
~~~

### 10.6 网格分辨率

grid_num_x 和 grid_num_yaw 越大，控制越细，但每个 cell 获得的样本越少：

~~~text
粗网格：3×3、5×5，课程扩展快，边界粗
中等网格：9×7，当前 Go2 配置
细网格：15×11，控制细，但训练时间显著增加
~~~

如果训练资源有限，不建议一开始使用非常细的网格。

### 10.7 命令段持续时间

当前建议：

~~~python
resampling_time_range=(8.0, 12.0)
~~~

它有两个作用：

1. 给机器人足够时间建立稳定步态。
2. 让训练中出现运动状态下的命令切换。

如果命令切换过于频繁：

~~~python
resampling_time_range=(12.0, 16.0)
~~~

如果机器人只会跟踪固定命令、不会处理切换：

~~~python
resampling_time_range=(6.0, 10.0)
~~~

时间过短会导致命令段尚未稳定就被结算，不适合训练初期。

---

## 11. 高速运动切换到零速度时摔倒

这是当前实现需要重点注意的行为问题。

### 11.1 问题原因

如果使用：

~~~python
resampling_time_range=(30.0, 30.0)
episode_length_s=30.0
~~~

训练中速度命令切换通常和 episode reset 同时发生，机器人学到的是：

~~~text
高速运动 → 环境 reset → 新命令
~~~

而不是：

~~~text
高速运动 → 指令变为零 → 主动刹车 → 站稳
~~~

在 play 模式中，episode_length_s 被设置成很大的值，命令会在机器人仍然运行时切换，因此会暴露这个问题。

### 11.2 当前配置的改进

将命令段设置为：

~~~python
resampling_time_range=(8.0, 12.0)
~~~

可以让训练过程中出现多次命令切换，训练数据中包含更多刹车场景。

### 11.3 仍然存在的瞬时切换问题

当前 _resample_command() 会直接写入新命令：

~~~python
self.vel_command_b[env_ids, 0] = new_x_command
self.vel_command_b[env_ids, 1] = new_y_command
self.vel_command_b[env_ids, 2] = new_yaw_command
~~~

如果旧命令为 2.0 m/s，新命令接近 0 m/s，策略会在一个环境步内看到明显目标变化。

更稳妥的后续改进是加入 command ramp：

~~~text
旧命令 → 经过 0.5～1.0 秒平滑过渡 → 新命令
~~~

实现时需要同时考虑：

1. policy 观察到的是平滑后的命令。
2. reward 使用的也是当前平滑命令。
3. transition 阶段是否计入 cell 成功率。
4. checkpoint 是否保存 ramp 的中间状态。

推荐的后续配置参数：

~~~python
command_ramp_time=0.5
~~~

或者：

~~~python
command_ramp_time=1.0
~~~

如果加入 ramp，建议对 ramp 阶段使用独立的 transition 统计，避免把必然存在的刹车过程全部算成 cell 失败。

---

## 12. Domain Randomization 与 Grid Adaptive

Grid Adaptive 的成功率不只由速度决定，还会受到 domain randomization 影响。

当前 Go2 训练配置包括：

~~~text
脚底摩擦系数随机化：0.3～1.2
编码器偏置随机化：(-0.015, 0.015)
质心偏移随机化：约 ±0.025～0.03 m
训练时随机推力：每 1～3 秒触发一次
~~~

这些随机化会影响：

1. 某个速度 cell 的平均跟踪误差。
2. 摔倒概率。
3. 近期成功率收敛速度。
4. cell 扩展时间。

### 12.1 推荐训练阶段

#### 阶段一：先学习基础速度能力

使用较小的摩擦随机化、质心偏移和推力，让初始 cell 稳定达到 0.8 以上成功率。

#### 阶段二：逐步增加随机化

在初始 cell 和附近 cell 已经稳定后，再增大：

~~~text
摩擦范围
质心偏移
推力强度
编码器偏置
~~~

#### 阶段三：检查最终速度边界

确认高速度和大角速度 cell 在当前随机化范围下仍有足够成功率。

不要在课程尚未稳定时同时扩大速度范围、增加网格分辨率、提高随机化强度和降低成功率阈值。否则很难判断失败来自速度难度、随机化还是课程逻辑。

## 13. Checkpoint 保存与恢复

Grid Adaptive 不仅需要保存神经网络，还需要保存课程地图。否则恢复训练时，策略已经学会了较大速度范围，但课程会从初始 cell 重新开始。

### 13.1 保存内容

curriculum_state_dict() 保存：

~~~text
active_cells
cell_visits
cell_successes
cell_success_history
cell_recent_visits
cell_recent_successes
cell_history_ptr
current_cell
vel_command_b
vel_command_w
time_left
command_counter
standing/world/forward 标志
当前命令段误差
last_expansion_step
~~~

其中：

- active_cells 和 cell 统计是课程核心状态。
- ring buffer 决定近期成功率。
- 当前命令和命令段状态用于避免恢复后把旧统计错误归因到新命令。

### 13.2 runner 的保存逻辑

MjlabOnPolicyRunner.save() 会遍历 command manager 中的命令项：

~~~python
for term_name in command_manager.active_terms:
  command_term = command_manager.get_term(term_name)
  if hasattr(command_term, "curriculum_state_dict"):
    env_state["grid_curriculum"] = (
      command_term.curriculum_state_dict()
    )
    env_state["grid_curriculum_term"] = term_name
    break
~~~

这种写法避免 runner 强耦合到固定的命令名称 twist。

### 13.3 训练和 play 环境数量不一致

训练通常使用 4096 个环境，play 通常使用 1 个环境。

因此当前恢复逻辑分成两类：

1. 网格级状态必须形状一致，否则报错。
2. 逐环境运行状态如果形状不同则跳过恢复，由当前 play 环境重新初始化。

这样可以恢复课程地图，同时兼容训练 checkpoint 在单环境 play 中使用。

---

## 14. 当前日志

当前 Grid Adaptive 关注的全局日志为：

~~~text
Metrics/twist/grid_success
Metrics/twist/grid_active_cells
Metrics/twist/grid_total_visits
~~~

### 14.1 grid_success

表示本次 reset 批次中命令段成功率的平均值。它是局部时间窗口内的表现指标，不等同于整个课程地图的平均成功率。

观察重点：

~~~text
是否长期高于 0.8
是否在 active cell 扩张后明显下降
是否在增加 domain randomization 后下降
~~~

### 14.2 grid_active_cells

表示当前 active map 中激活的 cell 数量：

~~~text
初始通常为 1
最大值为 grid_num_x × grid_num_yaw
~~~

观察重点：

~~~text
是否长期停在 1
是否扩展过快
是否最终达到预期覆盖范围
~~~

### 14.3 grid_total_visits

表示所有 cell 的累计有效访问数量总和。

它可以帮助判断：

1. 课程是否真的获得了有效命令段样本。
2. active cell 增加后，每个 cell 的样本是否被过度稀释。
3. active cell 不变时，是否只是因为命令段还没有结算。

TensorBoard 启动命令：

~~~bash
cd /home/haozi/桌面/Nazarite-mjlab/Train/Nazarite

./mjlab/.venv/bin/tensorboard \
  --logdir logs/rsl_rl/go2_flat \
  --port 6006
~~~

浏览器访问：

~~~text
http://localhost:6006
~~~

> 说明：commands.py 中还维护 grid_error_vel_xy 和 grid_error_vel_yaw 作为诊断 metric。如果要严格让 TensorBoard 只显示上面三个 Grid 指标，需要同时删除这两个诊断 metric 的初始化、更新和日志注册代码。

---

## 15. 常见现象与排查

### 15.1 grid_active_cells 一直为 1

按以下顺序检查：

1. grid_success 是否达到阈值。
2. min_cell_visits 是否达到。
3. success_window_size 是否过大。
4. 是否只有 standing 任务被结算。
5. velocity_error_threshold 和 yaw_error_threshold 是否过严。
6. domain randomization 是否过强。

### 15.2 grid_active_cells 增长过快

可以：

~~~python
max_new_cells_per_update=1
success_rate_threshold=0.9
min_cell_visits=80
~~~

### 15.3 grid_success 在扩展后突然下降

这通常表示新激活的边界速度超过了当前策略能力。建议：

1. 降低 max_new_cells_per_update。
2. 增大 min_cell_visits。
3. 提高 success_rate_threshold。
4. 检查四连通扩展是否需要改成单轴扩展。
5. 暂时缩小最终速度范围。

### 15.4 高速运行后切换到零速度摔倒

重点检查：

1. 训练时是否真的发生过 mid-episode 命令切换。
2. resampling_time_range 是否与 episode 长度完全相同。
3. 新命令是否瞬时写入。
4. lin_vel_y 是否仍然可能接近 ±1.0 m/s。
5. 是否需要增加 command ramp。

### 15.5 看到 lin_vel_x≈0，但机器人没有停下

检查完整 command：

~~~text
[lin_vel_x, lin_vel_y, ang_vel_z]
~~~

不要只看 lin_vel_x。如果 lin_vel_y 或 ang_vel_z 不为零，机器人仍然有运动目标。

---

## 16. 推荐实验顺序

### 实验 A：验证课程逻辑

~~~python
grid_num_x=3
grid_num_yaw=3
min_cell_visits=10
success_window_size=20
max_new_cells_per_update=1
success_rate_threshold=0.8
~~~

目的是快速确认：

~~~text
采样 → 结算 → 统计 → 扩展
~~~

### 实验 B：稳定初始速度

~~~python
grid_num_x=5
grid_num_yaw=5
min_cell_visits=30
success_window_size=50
max_new_cells_per_update=2
~~~

### 实验 C：当前推荐 Go2 配置

~~~python
grid_num_x=9
grid_num_yaw=7
min_cell_visits=50
success_window_size=100
max_new_cells_per_update=4
success_rate_threshold=0.8
resampling_time_range=(8.0, 12.0)
~~~

### 实验 D：增强随机化和高速能力

在实验 C 已稳定后，再逐步扩大：

~~~text
摩擦随机化
质心偏移
推力强度
lin_vel_x 最高速度
ang_vel_z 最高速度
~~~

每次只改变一组因素，并记录：

~~~text
grid_success
grid_active_cells
grid_total_visits
mean_episode_length
fell_over
illegal_contact
~~~

---

## 17. 总结

当前 Grid Adaptive 的完整逻辑可以概括为：

~~~text
将 lin_vel_x × ang_vel_z 划分成网格
        ↓
从 initial_cell 开始
        ↓
只在 active cell 中均匀采样
        ↓
按命令段统计速度跟踪和摔倒情况
        ↓
使用累计访问数 + 近期成功率判断能力
        ↓
有限数量地扩展四连通邻居
        ↓
将课程状态保存到 checkpoint
~~~

主要优点是课程边界由机器人表现决定，能够避免训练早期直接面对完整速度范围。

当前最需要继续改进的部分是命令切换动态过程：机器人不仅要学会保持 2 m/s，还必须学会从 2 m/s 平稳减速到零速度。因此训练中需要出现真实的 mid-episode 命令切换，并建议后续加入 command ramp 和 transition-aware 成功统计。

