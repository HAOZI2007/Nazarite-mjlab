# WTW：从零手写 Walk These Ways

> 本文按 Nazarite-mjlab 当前源码解释 WTW 的设计、数据流和实现步骤。当前任务以 Go2 平地 trot 为第一阶段目标，并与 Grid Adaptive 速度课程一起工作。

## 1. 当前项目状态

当前代码已经实现：

- 任务名：Nazarite-Velocity-Flat-Go2-WTW；
- twist 命令：GridAdaptiveVelocityCommand；
- behavior 命令：WTWBehaviorCommand；
- actor 历史：10 帧；critic 历史：3 帧；
- 有效速度指令时 phase 推进，零速度指令时 phase 冻结；
- WTW 速度、摆动相、支撑相、身体高度、俯仰、站姿宽度、摆腿高度和 Raibert 均为独立 RewardTerm；
- WTW 移除了 air_time、prolonged_air_time 和 stance_contact，避免它们与 phase 时序冲突；
- WTW play 保留训练中的随机推力。

核心文件：

~~~
/home/haozi/桌面/Nazarite-mjlab/Train/Nazarite/Nazarite-src/nazarite/
├── mdp/wtw.py
├── mdp/wtw_rewards.py
├── mdp/rewards.py
├── mdp/commands.py
├── config/train_config/base_env_cfg.py
├── config/train_config/env_cfgs.py
└── __init__.py
~~~

## 2. WTW 解决什么问题

普通速度策略为：

~~~
a_t = pi(o_t, c_t)
~~~

o 是本体观测，c=[vx, vy, wz] 是速度命令，a 是关节动作。同一个速度存在很多可行步态，纯速度奖励不会要求策略使用指定风格。

WTW 增加行为条件：

~~~
a_t = pi(o_t, c_t, b_t, p_t)
~~~

- c：任务要求，回答“走多快、转多快”；
- b：行为参数，回答“身体多高、步频多大、站多宽”；
- p：四条腿的相位参考，回答“现在每条腿处于周期的什么位置”。

WTW 的目标是一个策略根据 behavior 输出不同运动方式，而不是为每一种风格都训练独立网络。

## 3. 与 Grid Adaptive 的关系

当前数据流：

~~~
GridAdaptiveVelocityCommand
    -> twist=[vx, vy, wz]

WTWBehaviorCommand
    -> behavior=[theta1,theta2,theta3,f,hz,phi,sy,hfz]
    -> phase=[FL,FR,RL,RR]

actor = 本体观测 + twist + behavior + phase + 历史
    -> policy
    -> 12 个关节动作
~~~

Grid Adaptive 管任务速度空间，WTW 管行为空间。零速度环境仍然应该采样，因为它训练停止、刹停和静止站立；但零速度时不应该继续要求机器人执行 gait 周期。

## 4. 八维行为参数

项目在 mdp/wtw.py 中固定顺序：

| 索引 | 名称 | 含义 | 单位 |
|---:|---|---|---|
| 0 | theta1 | 相位模板参数 1 | 周期比例 |
| 1 | theta2 | 相位模板参数 2 | 周期比例 |
| 2 | theta3 | 相位模板参数 3 | 周期比例 |
| 3 | frequency | 步频 | Hz |
| 4 | body_height | 目标机体高度 | m |
| 5 | body_pitch | 目标机体俯仰角 | rad |
| 6 | stance_width | 目标站姿宽度 | m |
| 7 | foot_swing_height | 目标摆腿峰值高度 | m |

对应常量：

~~~python
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
~~~

训练、日志、策略导出和实机部署必须保持这个顺序。

## 5. gait 模板与腿顺序

当前模板：

~~~python
GAIT_THETA = {
    "pronking": (0.0, 0.0, 0.0),
    "trot": (0.5, 0.0, 0.0),
    "bounding": (0.0, 0.5, 0.0),
    "pacing": (0.0, 0.0, 0.5),
}
~~~

四条腿固定为：

~~~
[FL, FR, RL, RR]
~~~

WTWBehaviorCommand._theta_to_phase_offsets() 将 theta 转换成四条腿偏移。这个顺序必须与 GO2_FOOT_SITES 和 feet_ground_contact 的顺序一致。腿映射错误会表现为前后腿朝不同方向撇、接触时序错位等问题。

## 6. CommandTerm 生命周期

WTWBehaviorCommand 主要有：

1. reset 时调用 _resample_command；
2. 一个 resampling 周期内保持 behavior；
3. 每个仿真步调用 compute；
4. compute 根据 twist 决定是否推进 phase。

采样流程：

~~~
选择 gait
  -> 写入 theta
  -> 采样 frequency、height、pitch、width、swing height
  -> 随机初始化 base phase
  -> 生成四条腿 phase
~~~

当前 base_env_cfg.py 的配置：

~~~python
"behavior": WTWBehaviorCommandCfg(
    entity_name="robot",
    resampling_time_range=(30.0, 30.0),
    frequency_range=(2.0, 2.0),
    body_height_range=(0.32, 0.32),
    body_pitch_range=(0.0, 0.0),
    stance_width_range=(0.21, 0.21),
    foot_swing_height_range=(0.07, 0.07),
    gait_names=("trot",),
)
~~~

固定范围只表示当前先验证单一 trot，不代表 WTW 只能使用一种风格。

## 7. phase 的数学和冻结逻辑

公共变量 base_phase 在 [0,1) 内循环。每条腿：

~~~
phase_leg = (base_phase + phase_offset_leg) mod 1
~~~

移动时：

~~~
base_phase_next = (base_phase + frequency * dt) mod 1
~~~

当前有效命令判断：

~~~
norm([vx,vy]) + abs(wz) > 0.05
~~~

有效时推进；否则冻结。代码逻辑：

~~~python
command = env.command_manager.get_command("twist")
command_magnitude = (
    torch.linalg.vector_norm(command[:, :2], dim=1)
    + torch.abs(command[:, 2])
)
phase_active = command_magnitude > 0.05

next_phase = (self._base_phase + frequency * dt_batch) % 1.0
self._base_phase = torch.where(
    phase_active,
    next_phase,
    self._base_phase,
)
~~~

冻结是当前项目的工程化处理：零速度对应普通站立，避免 phase 持续推进诱发原地小碎步。论文原始仓库的 phase 生命周期并非完全如此，做论文对照时要记录这一差异。

## 8. phase observation

behavior 是静态参数，phase 是时间参考。当前 phase observation 返回：

~~~python
phase_angle = 2.0 * math.pi * term.phase
torch.cat((torch.sin(phase_angle), torch.cos(phase_angle)), dim=-1)
~~~

形状为 [N,8]，顺序是 [sin_FL, sin_FR, sin_RL, sin_RR,
cos_FL, cos_FR, cos_RL, cos_RR]。正弦和余弦共同表示完整周期位置，且
phase 从 1 回到 0 时输入连续。

base_env_cfg.py 中：

~~~python
"behavior": ObservationTermCfg(
    func=mdp.generated_commands,
    params={"command_name": "behavior"},
),
"phase": ObservationTermCfg(
    func=custom_wtw.wtw_phase_reference,
    params={"command_name": "behavior"},
),
~~~

## 9. actor、critic 和部署可行性

actor 观测：

~~~
base_ang_vel、projected_gravity、joint_pos、joint_vel、actions、
twist、behavior、phase
~~~

这些量可以由 IMU、关节编码器、上一动作和部署端的命令生成器提供。critic 额外看到：

~~~
base_lin_vel、base_height、foot_height、foot_air_time、
foot_contact、foot_contact_forces
~~~

这是 asymmetric actor-critic。critic 的 privileged 量只帮助训练，不能误放入 actor，否则实机没有这些量时会失配。

当前历史采用逐项配置：

~~~text
actor 普通观测：10 帧
actor behavior：5 帧
actor phase sin/cos：0 帧，只保留当前帧

critic 普通和特权观测：3 帧
critic behavior：5 帧
critic phase sin/cos：0 帧，只保留当前帧
~~~

当前两个 ObservationGroupCfg 的 history_length 都是 None，避免组级设置覆盖
behavior=5 和 phase=0。历史由各个 observation term 独立管理。

## 10. 期望接触相位

wtw_rewards.py 的 _smooth_contact_target 将 phase 映射为 C：

~~~
C in [0,1]
C 接近 1：期望支撑
C 接近 0：期望摆动
~~~

它用带周期边界连接的高斯 CDF 乘积，而不是简单的二值阈值，使 phase 从 1 回到 0 时目标连续。smoothing=0.15；增大它会让边界更柔和，减小它会让时序更硬。

## 11. 两个独立接触奖励

### 11.1 摆动相接触力代价

摆动相希望脚离地：

~~~
cost_swing = (1-C) * [1-exp(-force^2/force_std)]
~~~

返回正代价，因此 weight 使用负数。当前：

~~~python
weight = -4.0
force_std = 100.0
smoothing = 0.15
~~~

函数还记录：

~~~
WTW/foot_FL_swing_force
WTW/foot_FR_swing_force
WTW/foot_RL_swing_force
WTW/foot_RR_swing_force
~~~

### 11.2 支撑相足端水平速度代价

支撑相希望脚贴地不滑：

~~~
cost_stance = C * [1-exp(-velocity_xy^2/velocity_std)]
~~~

当前：

~~~python
weight = -4.0
velocity_std = 10.0
smoothing = 0.15
~~~

二者都通过 _active_mask 在零速度时关闭。不要把零速度站立误判为接触时序失败。

## 12. 行为辅助奖励和 RewardManager

当前所有 WTW 行为项都独立注册，RewardManager 直接累加：

~~~
r_total = sum(weight_i * term_i)
~~~

相关 term：

| term | 物理目标 | 当前权重 |
|---|---|---:|
| wtw_swing_phase_force | 摆动相减小接触力 | -4.0 |
| wtw_stance_phase_velocity | 支撑相减小足端水平速度 | -4.0 |
| wtw_body_height | 跟随 hz | 0.80 |
| wtw_body_pitch | 跟随 phi | 0.10 |
| wtw_stance_width | 跟随 sy | 0.03 |
| wtw_foot_swing_height | 跟随 hfz 峰值 | 0.60 |
| wtw_raibert_foot_position | 合理摆动落点 | 0.20 |

调参时直接修改 base_env_cfg.py 中相应 RewardTerm 的 weight 或 params。

## 13. 身体高度、pitch 和站姿宽度

### 13.1 body height

实际机体高度与 behavior 的 hz 比较：

~~~python
reward = torch.exp(
    -torch.square(actual_height - target_height) / std**2
)
~~~

Go2 WTW 会移除 baseline 的固定 base_height，避免固定高度目标和行为高度目标竞争。当前 std=0.035，权重为 0.80。

### 13.2 body pitch

从 projected gravity 估计 pitch，与 behavior 的 phi 比较：

~~~text
reward = exp(-(pitch - target_pitch)^2 / std^2)
~~~

当前固定 trot 的 phi 为 0，权重 0.10。过大时会限制自然前倾。

### 13.3 stance width

将足端位置变换到机体坐标系，比较实际横向距离与 sy/2。当前权重 0.03，故只作为轻辅助。过大会限制转弯和横向运动。

## 14. 摆腿高度状态机

当前不是每帧直接奖励高度，而是：

~~~
期望摆动相 -> 累积每条腿的 peak height
first_contact -> 比较 peak height 和 hfz
计算奖励 -> 清空该腿缓存
~~~

这更接近摆腿峰值，也避免摆动刚开始时因为脚还没抬起而被过早惩罚。当前 std=0.04，权重 0.60。

## 15. 简化 Raibert

wtw_raibert_foot_position 是适配当前项目的二维工程启发式，不是完整全身动力学控制器：

1. reset 后记录足端名义机体坐标；
2. 用 stance_width 修正左右落点；
3. 用速度命令与当前速度误差修正前后/侧向落点；
4. 用 yaw 角速度加入旋转补偿；
5. 只在期望摆动相比较。

近似形式：

~~~
p_target = p_nominal
         + velocity_error * (0.5 / frequency)
         + stance_width_offset
         + yaw_rate_offset
~~~

当前权重 0.20，std=0.08。出现腿向不同方向撇时，优先检查腿顺序和 phase 映射，不要先盲目增大 Raibert 权重。

## 16. 为什么移除部分 baseline 奖励

air_time、prolonged_air_time 和 stance_contact 没有使用 WTW phase，可能和 phase 时序提供相反信号。例如 phase 要求摆动时，接触奖励仍可能鼓励落地。WTW 当前将其移除，接触时序交给 phase-conditioned 的两个独立项。

foot_slip、soft_landing、动作平滑、扭矩和关节限制仍可保留，它们约束安全和 sim-to-real，而不是第二个步态时钟。

## 17. Grid Adaptive 配置要点

当前最终范围：

~~~
lin_vel_x = (-2.0, 2.0) m/s
lin_vel_y = (-1.0, 1.0) m/s
ang_vel_z = (-0.7, 0.7) rad/s
~~~

课程关键参数：

~~~
grid_num_x = 9
grid_num_yaw = 7
initial_cell = (3, 3)
min_cell_visits = 100
success_window_size = 100
max_new_cells_per_update = 1
success_rate_threshold = 0.8
velocity_error_threshold = 0.35
yaw_error_threshold = 0.35
rel_standing_envs = 0.3
~~~

Grid cell 记录速度任务成功情况，不等于 gait 已学会。gait 还要结合接触力、支撑速度、摆腿高度和 play 判断。

## 18. 从零手写的推荐顺序

### 第一步：behavior command

先完成 WTWBehaviorCommandCfg、WTWBehaviorCommand、固定 trot、行为采样、base_phase 和 phase。验证 behavior 为 [N,8]。

### 第二步：phase observation

加入 wtw_phase_reference，验证输出为 [N,4]，检查四条腿 offset。

### 第三步：冻结 phase

零速度连续多个 step 时 phase 不变；非零速度后恢复推进。

### 第四步：接触奖励

先保证两个奖励返回有限值，再看 cost/score。确认零速度自动关闭。

### 第五步：行为参数奖励

建议顺序：body height -> swing height -> body pitch -> stance width -> Raibert。每加入一项做一次固定速度 play。

### 第六步：历史、推力和课程

确认 actor 10、critic 3，WTW play 保留 push，再逐渐恢复完整 Grid Adaptive。

## 19. 训练和 play

在 Train/Nazarite 目录运行：

~~~bash
uv run train Nazarite-Velocity-Flat-Go2-WTW
uv run play Nazarite-Velocity-Flat-Go2-WTW
~~~

play 未提供 checkpoint 时，需要按 play.py 的要求提供 wandb_run_path 或 checkpoint_file。

WTW play 与训练一样保留 push_robot。重点测试：

~~~
全零速度：静止站立
低速：是否高频小碎步
高速：是否前后腿不对称
2 m/s 切到 0：是否平稳刹停
随机推力：是否能恢复
~~~

## 20. TensorBoard

运行：

~~~bash
tensorboard --logdir /home/haozi/桌面/Nazarite-mjlab/Train/Nazarite/logs/rsl_rl
~~~

重点查看：

~~~
Metrics/twist/grid_success
Metrics/twist/grid_active_cells
Metrics/twist/grid_total_visits

Metrics/behavior/wtw_frequency
Metrics/behavior/wtw_body_height
Metrics/behavior/wtw_swing_height
Metrics/behavior/wtw_gait_id

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

WTW/stand_base_lin_vel
WTW/stand_base_ang_vel
WTW/stand_joint_vel
WTW/stand_still_cost
~~~

每个奖励贡献在 Episode_Reward 下单独记录。不要只看总 reward。

## 21. 常见问题

| 现象 | 优先检查 |
|---|---|
| 零速度晃动 | phase 是否冻结、stand_pose、zero_command_stillness |
| 启动时下蹲 | body_height、实际高度日志、推力强度 |
| 前后腿撇向不同方向 | [FL,FR,RL,RR]、phase offset、Raibert |
| 高频小碎步 | frequency、swing height、是否仍有默认姿态竞争 |
| 支撑脚滑 | stance phase velocity、foot_slip |
| 摆动脚碰地 | swing phase force、swing height |
| behavior 变化但动作不变 | actor 是否收到 behavior/phase，奖励权重是否有效 |
| Grid 扩张快但速度差 | success 阈值、窗口、零速比例、推力 |

## 22. 行为开放路线

第一阶段固定 trot：

~~~python
gait_names=("trot",)
frequency_range=(2.0, 2.0)
body_height_range=(0.32, 0.32)
body_pitch_range=(0.0, 0.0)
stance_width_range=(0.21, 0.21)
foot_swing_height_range=(0.07, 0.07)
~~~

稳定后，一次只开放一个变量：

~~~python
frequency_range=(1.5, 3.0)
foot_swing_height_range=(0.05, 0.10)
body_height_range=(0.30, 0.34)
body_pitch_range=(-0.05, 0.05)
stance_width_range=(0.18, 0.24)
~~~

最后再逐步加入 pronking、bounding 和 pacing。一个条件策略适合在线切换；多个 checkpoint 更容易隔离风险。实机建议用行为预设，不要直接输入 8 个裸参数。

## 23. 复杂地形和实机

当前任务是平地任务。建议路线：

~~~
平地学会多行为
    -> 复杂地形零样本测试
    -> 比较不同 behavior
    -> 再决定是否加入 terrain training
~~~

复杂地形训练还需要 terrain generator、terrain curriculum、actor terrain scan、可靠的足端高度定义和新的接触/终止调参。没有视觉的实机 actor 不能依赖 terrain scan。

部署端的 phase 可以由步频、速度有效性和本地时钟维护；behavior 由遥控器、上层规划器或预设生成。切换高度、频率和摆腿高度时做 0.5～1.0 秒插值。

## 24. 验收标准

~~~
相同速度 + 不同 gait
    -> 接触时序不同

相同 gait + 不同 frequency
    -> 实际步频不同

相同 gait + 不同 swing height
    -> 摆腿峰值不同

相同速度 + 不同 body height
    -> 实际机体高度不同

零速度
    -> phase 冻结并稳定站立
~~~

总结：

~~~
Grid Adaptive 学习任务速度空间
WTW behavior 学习行为空间
phase 提供时间结构
独立 RewardTerm 约束行为
本体观测保证部署可行
~~~
