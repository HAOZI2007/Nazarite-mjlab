# WTW 步态行为设计教程：behavior、phase 与独立设计 gait

> 本文解释当前 Nazarite-mjlab 的 WTW 表示法。默认训练 gait 已是 Trot；Pronking、Bounding、Pacing 在本文是可扩展模板，而不是当前 checkpoint 自动具有的能力。

## 1. 一句话理解 behavior 和 phase

```text
behavior 描述“要做什么样的运动”
phase 描述“该运动现在进行到周期的哪里”
reward 描述“这个周期位置下每只脚应满足什么物理状态”
```

因此它们不是两套互不相关的指令：

```text
behavior 的 theta 定义四腿 phase 的相对结构
behavior 的 frequency 定义 phase 的推进速度
phase 把抽象 gait 展开成每一个控制周期的时序参考
```

例如，`trot + 2.2 Hz` 不是“机器狗收到一个 trot 标签”那么简单，而是“对角两组腿保持半周期相差，公共时钟每秒推进 2.2 圈”。

## 2. 当前 behavior 向量

```text
behavior = [theta1, theta2, theta3, frequency,
            body_height_offset, body_pitch,
            stance_width, foot_swing_height]
```

| 字段 | 解决的问题 | 会不会改变腿间时序 |
|---|---|---|
| `theta1, theta2, theta3` | 哪些脚同相、哪些脚相差半周期 | 会 |
| `frequency` | 一个 gait 周期多快 | 不改相对关系，只改时间尺度 |
| `body_height_offset` | 身体相对 0.32 m 的目标高度 | 不会 |
| `body_pitch` | 身体前后倾目标 | 不会 |
| `stance_width` | 名义落脚横向宽度 | 不会 |
| `foot_swing_height` | 摆腿抬脚轨迹高度 | 不会 |

最常见的误解是用抬脚高度、机体高度或步频去“变出”另一种 gait。它们只能改变外观和动力学强度；真正决定 Trot / Bound / Pace / Pronking 的是 `theta` 产生的**相对 phase offset**。

## 3. phase 的数学含义

每个环境有一个公共相位：

```text
0 <= base_phase < 1
base_phase_next = (base_phase + frequency × dt) mod 1
```

它是归一化周期进度，不是秒、弧度或真实接触状态。

| phase | 意义 |
|---:|---|
| 0.00 | 周期起点 |
| 0.25 | 周期四分之一 |
| 0.50 | 周期中点 |
| 0.75 | 周期四分之三 |
| 1.00 | 回到 0.00 |

频率给它赋予真实时间尺度：`2 Hz` 的周期为 `0.5 s`，`2.4 Hz` 的周期约为 `0.417 s`。在当前 50 Hz 控制频率下，分别约为 25 和 21 个控制步。

四足相位通过：

```text
phase_leg = (base_phase + offset_leg) mod 1
```

得到。项目腿顺序必须保持为 `[FL, FR, RL, RR]`。当前 `_theta_to_phase_offsets()` 已把论文的腿顺序转换到该顺序；新增传感器、site 或 reward 时都必须沿用它。

## 4. 内置 gait 如何映射到四腿

当前模板：

```python
GAIT_THETA = {
    "pronking": (0.0, 0.0, 0.0),
    "trot": (0.5, 0.0, 0.0),
    "bounding": (0.0, 0.5, 0.0),
    "pacing": (0.0, 0.0, 0.5),
}
```

可直观理解为：

| gait | 同相腿组 | 两组关系 |
|---|---|---|
| Pronking | FL、FR、RL、RR | 四脚同相 |
| Trot | FL+RR；FR+RL | 两个对角组相差 0.5 周期 |
| Bounding | FL+FR；RL+RR | 前后两组相差 0.5 周期 |
| Pacing | FL+RL；FR+RR | 左右两组相差 0.5 周期 |

当前固定 `duty_factor=0.5`，所以每个组支撑约半周期、摆动约半周期。改变 duty factor 才会改变支撑/摆动占空比；它是当前全局时序形状而不是 actor 的 8 维 behavior 输入。

## 5. phase 如何变成接触目标

`wtw_rewards.py` 以平滑周期函数生成：

```text
desired_contact(phase) ∈ [0, 1]
接近 1 -> 期望支撑
接近 0 -> 期望摆动
```

在 `duty_factor=0.5`、`smoothing=0.07` 下，可以近似理解为：

```text
phase 0.0 ~ 0.5 : 支撑
phase 0.5 ~ 1.0 : 摆动
```

边界被高斯 CDF 平滑连接，而不是硬切换。`smoothing` 变大，边界更软、时序更模糊；变小，边界更硬、早期训练对接触延迟更敏感。

关键区别：

```text
phase           = 期望的周期位置
desired_contact = phase 推导出的软目标
actual_contact  = 物理仿真检测到的真实接触
reward          = 促使真实物理接触匹配软目标
```

因此“不在支撑相时没触地”不是直接错误；摆动相本来应该不触地。真正错误是：期望支撑却悬空、或期望摆动却拖地。

## 6. 为什么 actor 输入 sin/cos，而不是裸 phase

actor 接收：

```text
[sin(2πphase_FL..RR), cos(2πphase_FL..RR)]
```

phase `0.99` 与 `0.01` 在周期上非常接近，裸数字却相差 0.98；sin/cos 将它们映射为相近的二维圆周位置。只用 sin 仍会使不同 phase 得到相同值，所以必须同时保留 cos。

当前 actor 的 phase 不做历史堆叠，因为当前帧的 sin/cos 已完整编码周期位置；普通本体观测保留 10 帧，behavior 保留 5 帧。

## 7. 当前的 phase 冻结规则

当前项目不是只要启用 WTW 就不停推进时钟。它会在：

```text
norm([vx, vy]) + abs(wz) <= 0.05
```

时冻结 phase。这样全零速度对应普通站立，而非原地执行 Trot。其后果是：

- 网页 play 从 `2 m/s` 切到 `0 m/s` 时，时序参考会停在当前相位；
- actor 不再收到移动中的相位变化；
- phase 相关奖励在零命令处使用相同阈值被关闭；
- `stand_pose` 负责保持站立姿态。

设计新 gait 时必须决定它是否也遵循这个规则。若任务目标是“原地踩步”，就不应冻结 phase；若目标是正常静止，冻结通常更合适。

## 8. 如何独立设计一个新 gait

按以下顺序设计，先不要动奖励权重。

### 第一步：写接触表

先以一个周期的四个区间写出每条腿的 `S`（支撑）和 `W`（摆动）。例如 Trot：

```text
时间       0-.25  .25-.5  .5-.75  .75-1
FL            S      S       W       W
FR            W      W       S       S
RL            W      W       S       S
RR            S      S       W       W
```

再从表中识别“哪些腿总是同相”和“哪些组相差 0.5”。若能用上述四种对称模板表达，就复用 `GAIT_THETA`；否则需要扩展相位参数化，而不能只改 frequency。

### 第二步：选择 duty factor 和频率

`duty_factor=0.5` 是对称、腾空与支撑时间相等的起点。更稳的慢走常需要大于 0.5 的支撑比例；明显腾空的 gait 可能需要小于 0.5。

新 gait 的第一次训练应固定频率。当前已验证 Trot 的合理起点是约 `2 Hz`；不要一开始给新 gait 一个宽频率范围。确认接触结构后再如当前 Trot 一样开放小范围。

### 第三步：固定风格参数

第一次训练固定高度、pitch、站宽、摆腿高度。这样失败能归因于“相位结构”或“频率”，不会混入风格随机化。

### 第四步：建立正确验收指标

| gait | 最重要的指标 |
|---|---|
| Trot | `contact_schedule_error`、对角组交替、四脚误差对称性 |
| Pronking | schedule + `pronking_sync_error` + 全脚同步离地/落地 |
| Bounding | 前后两组的时序与落地稳定性 |
| Pacing | 左右两组的时序与侧向稳定性 |

`pronking_sync_error` 只适用于四脚期望同步的 gait；用它评估 Trot 会得到近零，但没有任何信息量。

### 第五步：先训练独立任务，再做多 gait

在 `gait_names` 只放新 gait，确认它在整个速度 Grid 内稳定，再将多个 gait 放到一个 behavior 条件策略中。多 gait 任务的优势是部署时一个 checkpoint 可切换，但训练难度更高，且每种 gait 需要足够采样概率。

## 9. 网页 Behavior 面板如何辅助设计

播放 WTW checkpoint 后，网页 `Commands / Behavior` 可覆盖当前选中环境：

```text
Enable override
Gait（仅显示该 checkpoint 训练配置中的 gait）
Frequency
Body height offset
Body pitch
Stance width
Foot swing height
Reset phase
Use training defaults
```

推荐诊断顺序：

1. 将 `Twist` 固定在一个速度，例如 `vx=0.6`；
2. 开启 `Behavior` 覆盖，先设置训练范围内的固定频率；
3. 用 `Reset phase` 观察起步和接触切换；
4. 一次只改一个风格参数，观察高度、足端抬升和稳定性；
5. 超出训练范围的行为只能作为 OOD 压力测试，不能据此判断策略“不会该步态”。

当前单 Trot checkpoint 的下拉框只会显示 Trot，故它不能在网页里切到 Pronking。这是有意保护：不把未训练 gait 伪装成可部署功能。

## 10. 当前 Trot 的实际边界

当前源码默认频率为 `2.0–2.4 Hz`。历史 run `2026-08-30_19-18-04` 曾探索 `2.0–3.0 Hz`，速度和接触时序仍较好，但机体平均高度从目标 `0.32 m` 上漂到约 `0.343 m`。这表明“能跑”不等于“所有风格指标都满足”。

因此下一步优先验证窄范围频率，再根据 `WTW/body_height_signed_error` 决定是否扩大范围或调高度项。每个历史运行的实际范围以其 `params/env.yaml` 为准。
