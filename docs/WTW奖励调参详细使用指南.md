# WTW 奖励调参详细使用指南

> 本文针对 Nazarite-mjlab 当前任务 Nazarite-Velocity-Flat-Go2-WTW 编写。所有 WTW 奖励都是独立 RewardTerm，由 RewardManager 直接累加。调参时直接改对应 term 的 weight 或 params。

## 1. 先确定修改位置

主要配置文件：

~~~
/home/haozi/桌面/Nazarite-mjlab/Train/Nazarite/Nazarite-src/nazarite/config/train_config/base_env_cfg.py
~~~

Go2 任务绑定和 WTW 专用覆盖：

~~~
/home/haozi/桌面/Nazarite-mjlab/Train/Nazarite/Nazarite-src/nazarite/config/train_config/env_cfgs.py
~~~

奖励实现：

~~~
/home/haozi/桌面/Nazarite-mjlab/Train/Nazarite/Nazarite-src/nazarite/mdp/wtw_rewards.py
/home/haozi/桌面/Nazarite-mjlab/Train/Nazarite/Nazarite-src/nazarite/mdp/rewards.py
~~~

当前任务：

~~~bash
uv run train Nazarite-Velocity-Flat-Go2-WTW
uv run play Nazarite-Velocity-Flat-Go2-WTW
~~~

## 2. 当前总奖励结构

RewardManager 对每个 term 计算：

~~~
r_total = Σ(weight_i × term_i)
~~~

所以：

- 正权重通常配合 [0,1] 的“越大越好”奖励；
- 负权重通常配合“越大越坏”的正代价；
- term 的原始数值和 weight 的乘积，才是它对总奖励的实际影响；
- 不要只看 weight 的绝对值，还要看该 term 在 TensorBoard 的平均输出。

当前 WTW 相关结构：

~~~
任务奖励：
  track_linear_velocity       +2.0
  track_angular_velocity      +2.0

通用稳定性：
  upright                     +1.0
  body_ang_vel                -0.02
  dof_pos_limits              -0.2
  joint_acc_l2                -2.5e-7
  joint_torques_l2            -1.0e-4
  action_rate_l2              -0.005
  foot_slip                   -0.05
  soft_landing                -1.0e-5

零速站立：
  stand_pose                  -3.0
  zero_command_stillness      -0.1

WTW 行为：
  wtw_swing_phase_force       -4.0
  wtw_stance_phase_velocity   -4.0
  wtw_body_height              +0.8
  wtw_body_pitch              +0.1
  wtw_stance_width             +0.03
  wtw_foot_swing_height        +0.6
  wtw_raibert_foot_position    +0.2
~~~

WTW 任务移除了 air_time、prolonged_air_time 和 stance_contact。原因是它们没有读取 WTW phase，可能与摆动/支撑时序发出相反信号。

## 3. active mask：为什么零速度时很多 WTW 奖励为零

WTW 行为奖励使用同一个有效命令判断：

~~~
magnitude = norm([vx, vy]) + abs(wz)
active = magnitude > command_threshold
~~~

当前 command_threshold=0.05。

当 active=0：

- phase 冻结；
- 摆动相接触力代价为 0；
- 支撑相足端速度代价为 0；
- body pitch、stance width、swing height、Raibert 行为奖励关闭；
- body height 函数仍计算目标高度奖励，但日志按照 active 过滤；
- 站立使用 stand_pose 和 zero_command_stillness。

这样可以避免零速度站立时，策略被迫继续完成 gait。

如果把 command_threshold 改得太小，微小噪声也会触发 gait；改得太大，低速任务会被误当成站立。通常先保持 0.05，只在实际命令抖动明显时改动。

## 4. 速度任务奖励

### 4.1 track_linear_velocity

作用：让机体平面线速度跟随 twist 的 vx、vy。

当前 WTW 权重为 2.0。它是主任务，不能为了接触时序把它压得太低。

现象与调整：

| 现象 | 调整 |
|---|---|
| 速度追踪差、Grid success 低 | 先检查命令覆盖和随机推力，再小幅增大到 2.5 |
| 速度很好但步态不稳定 | 不要继续增大；改行为奖励和通用稳定性 |
| 低速停不住 | 检查零速采样比例、stand_pose 和 stillness |

### 4.2 track_angular_velocity

作用：跟踪 wz。当前权重为 2.0。

如果直行很好、转弯腿向外撇，优先检查 Raibert 的 yaw 补偿和 stance width，不要直接大幅提高角速度奖励。

## 5. 零速度站立奖励

### 5.1 stand_pose

stand_pose 是零速度下的关节姿态代价，当前 weight=-3.0。它约束“腿回到站立姿态”。

它与 zero_command_stillness 不同：

~~~
stand_pose
  -> 关节位置应接近零速度站立姿态

zero_command_stillness
  -> 机体线速度、角速度、关节速度应接近 0
~~~

如果零速度时机器人保持某个稳定但偏离默认站姿的姿态，增加 stand_pose 可能有帮助；如果加大后晃动变大，说明默认姿态与当前身体高度/动作策略冲突，应先减小而不是继续增加。

### 5.2 zero_command_stillness

当前配置：

~~~python
"zero_command_stillness": RewardTermCfg(
    func=custom_rewards.zero_command_stillness,
    weight=-0.1,
    params={
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "linear_velocity_weight": 1.0,
        "angular_velocity_weight": 1.0,
        "joint_velocity_weight": 0.05,
    },
)
~~~

它返回正 cost，再乘负权重。三项物理含义：

- linear_velocity_weight：抑制机体平动；
- angular_velocity_weight：抑制机体摇摆和旋转；
- joint_velocity_weight：抑制腿部持续抖动。

零速晃动的推荐顺序：

1. 确认 phase 确实冻结；
2. 看 stand_base_lin_vel、stand_base_ang_vel 和 stand_joint_vel；
3. 若机体摇摆为主，先调 angular_velocity_weight；
4. 若腿在快速抖动，调 joint_velocity_weight；
5. 再小幅调 zero_command_stillness.weight。

不要把它一次提高很多。惩罚过强可能让策略不愿意启动或形成僵硬站姿。

## 6. 接触时序奖励

### 6.1 平滑接触目标

phase 先通过 _smooth_contact_target 生成 C：

~~~
C≈1：期望支撑
C≈0：期望摆动
~~~

smoothing 当前为 0.15：

- 增大到 0.20～0.25：边界更软，训练更容易，但时序更模糊；
- 减小到 0.08～0.12：时序更硬，但早期探索更容易不稳定。

固定 trot 第一阶段不建议动 smoothing。只有在接触相位整体对、但摆动/支撑边界过于突兀或过于模糊时才调整。

### 6.2 wtw_swing_phase_force

它惩罚摆动相足端接触力：

~~~
cost = mean((1-C) × [1-exp(-force²/force_std)])
reward contribution = -weight_magnitude × cost
~~~

当前 weight=-4.0，force_std=100.0。

force_std 的作用：

- 越小：相同接触力产生更大 cost，约束更敏感；
- 越大：代价更快饱和前变得不敏感，适合传感器力值较大或早期训练；
- 不能把 force_std 当成 weight。weight 控制整个 term 的强弱，force_std 控制力代价曲线的尺度。

调节顺序：

1. 先看 WTW/swing_phase_force_cost 和四条腿 force 日志；
2. 如果 cost 高且摆动相频繁碰地，先保持 weight，减小 force_std，例如 100 到 60；
3. 如果 cost 已经很低但行为仍不稳定，再小幅增加负权重，例如 -4 到 -5；
4. 如果机器人为了不碰地而跳跃、落脚不稳，减小负权重，或把 force_std 调大。

### 6.3 wtw_stance_phase_velocity

它惩罚期望支撑相的足端水平速度：

~~~
cost = mean(C × [1-exp(-velocity_xy²/velocity_std)])
~~~

当前 weight=-4.0，velocity_std=10.0。

velocity_std 越小，对滑动越敏感；越大，足端需要更快才产生明显代价。调参时：

- 支撑脚明显滑动：先减小 velocity_std 或增大负权重；
- 原地转向被限制、足端需要自然调整：减小负权重；
- 直行稳定、转弯差：优先降低该项或 stance width，而不是提高 tracking reward。

### 6.4 为什么不直接用 schedule_error 作为主奖励

代码保留 schedule_error 作为诊断计算，但当前主要优化的是：

- 摆动相力；
- 支撑相水平速度。

二值接触误差只表示“有没有接触”，不表示摆动相碰撞力多大，也不表示支撑脚是否滑动。因此它更适合作为诊断指标，不应单独替代两个动力学代价。

## 7. 行为参数奖励

### 7.1 wtw_body_height

代码形式：

~~~
reward = exp(-(actual_height - target_height)² / std²)
~~~

当前：

~~~python
weight = 0.80
std = 0.035
~~~

weight 决定策略愿意花多大代价跟随行为高度；std 决定误差容忍度。

- 机体总是低于目标：适度增大 weight，检查实际高度和启动瞬间的高度；
- 高度跟随很好但速度变差：降低 weight；
- reward 接近 0 且动作难学：增大 std，例如 0.035 到 0.045；
- 目标高度本身不合理：先改 behavior 的 body_height_range，不要只改 reward weight。

训练时 body_height_range 固定为 0.32，建议先把 reward 调稳定，再开放到 0.30～0.34。body height 变高会改变关节工作区和落脚几何，不能只看日志平均值。

### 7.2 wtw_body_pitch

当前 weight=0.10，std=0.08。它跟踪 behavior 的 body_pitch。

- pitch 明显偏差：先确认 projected gravity 的 pitch 计算和坐标约定；
- 直行自然、转弯僵硬：降低 weight；
- 只训练 phi=0：这个 term 可以很小，不需要成为主奖励；
- 开放到 -0.05～0.05 后，观察行为日志和实际 pitch，不要立刻扩大到很大的范围。

### 7.3 wtw_stance_width

当前 weight=0.03，std=0.035。它比较机体坐标系中四个足端的横向绝对距离与 sy/2。

该项很容易和转弯、侧向运动冲突，所以保持小权重。出现以下现象时降低它：

- 转弯时脚被拉到固定横向位置；
- 高速时前后腿向不同方向撇；
- 站姿宽度日志很好，但速度追踪变差。

站姿宽度的单位是米，奖励比较的是每只脚到机体中心线的距离，因此目标使用 sy/2。

### 7.4 wtw_foot_swing_height

当前 weight=0.60，std=0.04，smoothing=0.15。它在 first_contact 时评价上一摆动相的峰值高度。

注意它不是“脚越高越好”，而是“峰值接近 hfz 最好”。

- 摆腿高度不足：先确认 WTW/swing_height_error，适度增大 weight 或增大 hfz；
- 脚抬得过高、动作像跳跃：降低 weight，或降低 hfz；
- 日志长期约 0：检查 contact sensor 的 first_contact、height sensor 和落脚事件；
- 四条腿差异大：检查 sensor/site 顺序和每腿的缓存 reset。

过大的摆腿奖励会让策略为了追踪高度制造不必要的腿部运动，因此通常不先把它调到很高。

### 7.5 wtw_raibert_foot_position

当前 weight=0.20，std=0.08。该项只在期望摆动相比较目标落点。

它使用：

- reset 后足端名义位置；
- stance_width；
- 速度命令与当前机体速度误差；
- frequency；
- yaw 角速度。

它不是完整动力学控制器。调参原则：

- 直行已稳定但落脚前后偏差大：小幅增加 weight；
- 转弯出现横向撇腿：降低 weight，检查 yaw 补偿；
- 前腿和后腿朝相反方向偏：先检查腿映射和 phase，不要先改 weight；
- 步频变化后落点误差变大：检查 0.5/frequency 的尺度是否合理。

## 8. 通用奖励的调参边界

WTW 不是只由 WTW term 组成。以下通用奖励仍会影响表现：

| term | 作用 | 调参风险 |
|---|---|---|
| upright | 防止身体倾倒 | 太低容易摔，太高可能僵硬 |
| body_ang_vel | 抑制机体角速度 | 太高会限制自然动态 |
| action_rate_l2 | 平滑动作 | 太高导致小碎步或不愿启动 |
| foot_slip | 抑制接触脚滑动 | 太高会限制转弯 |
| soft_landing | 抑制冲击 | 太高会限制快速落脚 |
| joint_torques_l2 | 降低能耗 | 太高会导致无力 |
| joint_acc_l2 | 降低高频关节加速度 | 太高会导致动作迟钝 |
| dof_pos_limits | 防止越界 | 一般不作为步态主调参项 |

WTW 已经将 pose weight 设为 0，避免默认姿态奖励压过行为奖励。不要在 WTW 任务中无意恢复一个强固定 pose。

## 9. 推荐调参顺序

### 阶段 A：固定行为和 trot

~~~python
frequency_range=(2.0, 2.0)
body_height_range=(0.32, 0.32)
body_pitch_range=(0.0, 0.0)
stance_width_range=(0.21, 0.21)
foot_swing_height_range=(0.07, 0.07)
gait_names=("trot",)
~~~

目标：速度能跟踪、零速能站稳、接触时序不明显错乱。

### 阶段 B：先调任务和时序

1. track_linear_velocity；
2. track_angular_velocity；
3. wtw_swing_phase_force；
4. wtw_stance_phase_velocity；
5. upright 和 body_ang_vel。

### 阶段 C：再调行为参数跟随

1. wtw_body_height；
2. wtw_foot_swing_height；
3. wtw_body_pitch；
4. wtw_stance_width；
5. wtw_raibert_foot_position。

### 阶段 D：最后调零速静止和随机推力

1. phase 是否冻结；
2. stand_pose；
3. zero_command_stillness；
4. push_robot 的强度和间隔；
5. Grid Adaptive 的课程速度。

一次只改一个奖励族。每次训练都记录改动前后的日志路径、checkpoint 和关键权重。

## 10. 按现象调参

| 现象 | 首要判断 | 推荐动作 |
|---|---|---|
| 零速度机体晃动 | phase 是否冻结；看 base/angular/joint vel | 先调 stillness 的对应子权重，再小幅调 weight |
| 加大 stand_pose 后更晃 | 位置目标与行为/高度冲突 | 降低 stand_pose，检查 body height |
| 2 m/s 启动时下沉 | 高度目标或动态支撑不足 | 看 body_height_actual/error，适度提高高度项 |
| 高频小碎步 | 步频、动作平滑、时序边界 | 固定 frequency，检查 action_rate 和 swing height |
| 摆动脚经常碰地 | swing force cost 高 | 减小 force_std 或适度增加负权重 |
| 支撑脚滑 | stance velocity cost 高 | 减小 velocity_std 或适度增加负权重 |
| 转弯困难 | 支撑脚/站姿约束过强 | 降低 stance velocity、stance width 或 Raibert |
| 前腿向左后腿向右 | 映射错误概率高 | 检查 [FL,FR,RL,RR]、theta offset、site 顺序 |
| 脚抬得过高 | swing height 权重或目标过大 | 降低 weight 或 hfz |
| body height 日志很好但动作差 | 该奖励压过速度和接触 | 降低 body height weight |
| Grid success 上升但 gait 差 | Grid 只衡量速度任务 | 查看 WTW cost/score 和 play |
| 推力后容易倒 | 扰动覆盖不足或 upright 太弱 | 保持 play 同推力，检查训练 push 和 upright |

## 11. 参数范围建议

第一阶段固定：

~~~text
frequency = 2.0
body_height = 0.32
body_pitch = 0.0
stance_width = 0.21
swing_height = 0.07
~~~

稳定后逐一开放：

~~~text
frequency:        1.5 到 3.0 Hz
body_height:      0.30 到 0.34 m
body_pitch:      -0.05 到 0.05 rad
stance_width:     0.18 到 0.24 m
swing_height:     0.05 到 0.10 m
~~~

不要同时开放全部范围并改变大量奖励。行为范围扩大后，原来合适的一个 weight 可能不再适合所有行为。

## 12. TensorBoard 判断流程

先看任务是否学会：

~~~text
Metrics/twist/grid_success
Metrics/twist/grid_active_cells
Metrics/twist/grid_total_visits
Metrics/twist/error_vel_xy
Metrics/twist/error_vel_yaw
~~~

再看行为是否被采样：

~~~text
Metrics/behavior/wtw_frequency
Metrics/behavior/wtw_body_height
Metrics/behavior/wtw_swing_height
Metrics/behavior/wtw_gait_id
~~~

再看时序和行为：

~~~text
WTW/swing_phase_force_cost
WTW/swing_phase_force_score
WTW/stance_phase_velocity_cost
WTW/stance_phase_velocity_score
WTW/body_height_actual
WTW/body_height_error
WTW/swing_height_error
WTW/foot_FL_swing_force
WTW/foot_FR_swing_force
WTW/foot_RL_swing_force
WTW/foot_RR_swing_force
~~~

最后看零速：

~~~text
WTW/stand_base_lin_vel
WTW/stand_base_ang_vel
WTW/stand_joint_vel
WTW/stand_still_cost
Episode_Reward/stand_pose
Episode_Reward/zero_command_stillness
~~~

解读原则：

- cost 下降、score 上升，通常表示该动力学目标改善；
- 奖励 term 的 episode contribution 变得更负，不一定是坏事，要结合 cost 和任务表现；
- 四条腿某一条明显偏高，优先检查传感器顺序和相位；
- 零速度日志不应和运动时日志混为一个平均值。

## 13. 修改示例

只提高高度奖励，不改变其他项：

~~~python
rewards["wtw_body_height"].weight = 1.0
~~~

让摆动接触力更敏感：

~~~python
rewards["wtw_swing_phase_force"].params["force_std"] = 70.0
~~~

降低支撑相滑动约束：

~~~python
rewards["wtw_stance_phase_velocity"].weight = -3.0
~~~

提高零速时关节静止约束：

~~~python
rewards["zero_command_stillness"].params["joint_velocity_weight"] = 0.10
~~~

修改后要重新训练；已经保存的 checkpoint 不会自动使用新 reward 配置。

## 14. 训练前检查清单

- 任务名是否为 Nazarite-Velocity-Flat-Go2-WTW；
- behavior 和 phase 是否同时进入 actor；
- actor/critic 历史是否仍为 10/3；
- behavior 是否只采样 trot；
- 零速比例是否足够；
- phase threshold 是否为 0.05；
- WTW 奖励是否全部独立注册；
- air_time、prolonged_air_time、stance_contact 是否未被重新加入；
- WTW 任务是否移除了固定 base_height；
- play 是否保留 push_robot；
- reward 和 command 参数是否与本次实验记录一致。

## 15. 实机前验收

至少完成：

~~~text
固定 0 m/s：站立 30 秒不持续晃动
0 -> 1 m/s：启动不突然下蹲
1 -> 0 m/s：平稳刹停并保持站立
前进、后退、侧移：腿序对称
不同 yaw：不因 Raibert/stance width 锁死
随机推力：能恢复
~~~

实机部署时只使用 actor 可获得的本体观测。行为参数和 phase 必须由同样的顺序、单位和阈值生成，且行为切换应做平滑插值。

