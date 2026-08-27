# WTW 步态行为设计教程：理解 behavior、phase，并从零设计一个步态

> 本文专门解释一个问题：在 Nazarite-mjlab 的 WTW 中，behavior 和 phase 到底是什么关系？如果不直接照抄 trot、pacing 等模板，应该怎样自己设计一个步态？
>
> 阅读本文时要先记住一句话：
>
> ~~~
> behavior 描述“想要什么样的步态”
> phase 描述“这个步态现在进行到哪里”
> reward 负责把 phase 变成对脚的支撑/摆动要求
> policy 根据三者产生关节动作
> ~~~

## 1. 先建立正确的整体图像

WTW 策略不是直接接收一个“trot 开关”，而是接收：

~~~
actor_input
  = 本体观测
  + twist 速度命令
  + behavior 行为参数
  + phase 周期参考
  + 历史帧
~~~

其中：

| 输入 | 回答的问题 | 是否随时间变化 |
|---|---|---|
| twist | 机器人要走多快、向哪边走 | 可以变化 |
| behavior | 希望采用什么运动风格 | 通常一个 episode 内保持 |
| phase | 当前每条腿处于周期的哪个位置 | 持续变化 |
| 本体观测 | 机器人当前真实状态是什么 | 持续变化 |

因此，behavior 和 phase 不是两个相互独立的命令：

~~~
behavior 提供 phase 的“结构”
phase 提供 behavior 的“时间展开”
~~~

例如：

~~~text
behavior = trot + frequency=2 Hz
phase    = FL 当前处于支撑后半段、FR 当前处于摆动前半段
~~~

behavior 告诉策略“这是 2 Hz 的 trot”，phase 告诉策略“2 Hz trot 的当前时刻已经走到这里”。

## 2. behavior 具体包含什么

当前项目的 behavior 是 8 维向量：

~~~
behavior = [
    theta1,
    theta2,
    theta3,
    frequency,
    body_height,
    body_pitch,
    stance_width,
    foot_swing_height,
]
~~~

每一维的职责：

| 参数 | 作用 | 是否直接决定腿的相对时序 |
|---|---|---|
| theta1、theta2、theta3 | 生成四条腿的相位偏移 | 是 |
| frequency | 决定一个周期有多快 | 否，但决定时间尺度 |
| body_height | 目标机体高度 | 否 |
| body_pitch | 目标机体俯仰 | 否 |
| stance_width | 目标站姿宽度 | 否 |
| foot_swing_height | 目标摆腿峰值 | 否 |

所以设计一个步态时，首先要把问题拆成两层：

### 层 1：设计时序

决定：

~~~
哪几条腿同时支撑？
哪几条腿同时摆动？
一条腿相对于另一条腿晚多少？
每条腿支撑多久、摆动多久？
~~~

这部分由 theta、phase offset 和接触目标完成。

### 层 2：设计运动外观

决定：

~~~
机体多高？
步频多大？
脚抬多高？
站得多宽？
身体是否前倾？
~~~

这部分由 frequency、body_height、body_pitch、stance_width 和 foot_swing_height 完成。

不要把“脚的相对时序”和“脚抬多高”混为一件事。增加 swing height 不会自动把 trot 变成 bounding；改变 theta 才会改变腿之间的相对时序。

## 3. phase 到底是什么

phase 是一个归一化的周期时钟，范围为：

~~~
0 <= phase < 1
~~~

它不是角度，不是秒，也不是 contact 的布尔值。它表示一个 gait 周期的进度：

~~~
phase = 0.00  -> 周期起点
phase = 0.25  -> 周期四分之一
phase = 0.50  -> 周期中点
phase = 0.75  -> 周期四分之三
phase = 1.00  -> 回到 0.00
~~~

频率将 phase 转换为真实时间：

~~~
phase_next = (phase + frequency * dt) mod 1
~~~

例如：

~~~text
frequency = 2 Hz
周期 T = 1 / 2 = 0.5 秒

frequency = 3 Hz
周期 T = 1 / 3 ≈ 0.333 秒
~~~

在 50 Hz 控制下：

~~~text
2 Hz：一个周期约 25 个控制步
3 Hz：一个周期约 17 个控制步
~~~

因此 frequency 不是“每秒抬脚多少次”的模糊参数，而是 phase 时钟每秒推进多少个完整周期。

## 4. 一个公共 phase 和四个腿部 phase

当前 WTW 使用两个层次的 phase。

### 4.1 公共 base phase

WTWBehaviorCommand 内部维护：

~~~
_base_phase: [N]
~~~

每个并行环境有一个公共时钟。它只负责表示“整个 gait 走到哪里”。

### 4.2 四条腿的 phase

四条腿有不同的相位偏移：

~~~
phase_leg = (base_phase + phase_offset_leg) mod 1
~~~

当前形状：

~~~
phase: [N, 4]
腿顺序：[FL, FR, RL, RR]
~~~

例如一个简单的对角 trot 可以是：

~~~
base_phase = 0.20
offset = [0.5, 0.0, 0.0, 0.5]

phase = [0.70, 0.20, 0.20, 0.70]
~~~

这意味着：

~~~text
FL 和 RR 同时处于同一个周期位置
FR 和 RL 同时处于另一个周期位置
两组相差半个周期
~~~

这里真正定义 trot 的不是 base_phase=0.20，而是相对偏移：

~~~
[0.5, 0.0, 0.0, 0.5]
~~~

base_phase 整体加上 0.1，只是让整套步态从另一个时刻开始，不会改变步态类型。

## 5. phase 如何变成支撑和摆动

phase 本身只有一个连续数值，奖励需要把它解释成：

~~~
这一时刻脚应该支撑，还是应该摆动？
~~~

当前 wtw_rewards.py 使用平滑接触目标 C：

~~~
C(phase) ∈ [0, 1]

C 接近 1 -> 期望支撑
C 接近 0 -> 期望摆动
~~~

当前 smoothing=0.15 时，可以近似理解为：

~~~
phase 在 0 到 0.5 附近 -> 主要是支撑相
phase 在 0.5 到 1.0 附近 -> 主要是摆动相
~~~

边界附近不是硬切换，而是平滑过渡。这样 phase 从 1 回到 0 时，接触目标不会突然跳变。

这个设计非常重要：

~~~
phase != actual contact
phase = 期望的周期位置
actual contact = 仿真中脚是否真的接触
reward = 让 actual contact 尽量符合 phase 的期望
~~~

所以 phase 不是“告诉网络当前脚一定接触”，而是“告诉网络当前脚应该趋向支撑还是摆动”。

## 6. 当前项目的 phase、奖励和动作关系

每个控制周期可以理解为：

~~~
1. WTWBehaviorCommand 维护 base_phase
2. 根据 theta 生成四条腿 phase
3. phase observation 经过 sin 编码输入 actor
4. wtw_rewards.py 根据 phase 生成期望支撑/摆动目标
5. actor 结合真实关节状态和 phase 输出动作
6. 仿真得到真实接触、足端速度和机体状态
7. RewardManager 根据误差计算奖励
8. 下一步继续推进 phase
~~~

这形成闭环：

~~~
behavior -> phase -> desired contact
                    ↑
actual contact <- robot action
                    ↑
                 policy
~~~

behavior 不是直接控制关节角，phase 也不是低层轨迹生成器。它们都是给策略的条件和参考，最终动作仍由策略学习产生。

## 7. 当前项目 phase 的特殊规则：零速度冻结

当前 Nazarite 不是任何时候都推进 phase：

~~~
norm([vx, vy]) + abs(wz) > 0.05
    -> phase 推进

否则
    -> phase 冻结
~~~

这样做的目的：

- 全零速度时进入普通站立；
- 不因为 gait 时钟继续变化而原地踏步；
- 速度从 0 变为非 0 时，从冻结位置恢复推进；
- 速度从非 0 变为 0 时，phase 停在刹停时刻。

这也意味着当前步态设计必须同时考虑“运动”和“停止”：

~~~text
不能只让 gait 在 1 m/s 下漂亮，
还要让它从任意 phase 停下来后能保持稳定。
~~~

冻结 phase 是当前项目的工程策略；如果以后要完全复现官方行为，需要单独实现官方的 phase 生命周期，并重新处理零速度站立奖励。

## 8. 当前 theta 到四腿 phase 的转换

当前代码使用：

~~~python
phase_offsets = [
    theta1 + theta3,
    theta2 + theta3,
    theta2,
    theta1,
] % 1.0
~~~

顺序是：

~~~
[FL, FR, RL, RR]
~~~

也就是：

~~~
offset_FL = theta1 + theta3
offset_FR = theta2 + theta3
offset_RL = theta2
offset_RR = theta1
~~~

这三个 theta 只提供三个相对自由度。它不是任意四条腿 phase offset 的完整表达，因此不是所有你画出来的接触表都能直接编码成一个 theta。

### 8.1 反向求 theta

如果你设计了期望偏移：

~~~
[o_FL, o_FR, o_RL, o_RR]
~~~

按照当前公式，可以先取：

~~~
theta1 = o_RR
theta2 = o_RL
theta3 = o_FL - o_RR
~~~

然后检查：

~~~
o_FR == theta2 + theta3  mod 1
~~~

如果不满足，说明这个四腿相位组合不能由当前三 theta 结构表达。此时有两个选择：

1. 修改期望接触时序，使它符合当前参数化；
2. 修改代码，直接支持四条腿独立 phase offset。

### 8.2 为什么整体相位不重要

下面两组相位：

~~~
[0.0, 0.5, 0.5, 0.0]
[0.5, 0.0, 0.0, 0.5]
~~~

在 duty factor=0.5 的情况下，往往只是把整个动作整体平移半个周期。真正重要的是腿与腿之间的差值：

~~~
relative_offset_i_j = (offset_i - offset_j) mod 1
~~~

设计新步态时应先看相对 offset，不要被 reset 时随机的 base_phase 影响。

## 9. 重要提醒：不要只相信 gait 名称

当前代码中的 GAIT_THETA 是：

~~~python
GAIT_THETA = {
    "pronking": (0.0, 0.0, 0.0),
    "trot": (0.5, 0.0, 0.0),
    "bounding": (0.0, 0.5, 0.0),
    "pacing": (0.0, 0.0, 0.5),
}
~~~

代入当前 phase 映射后，得到：

| 名称 | theta | 当前 offsets，顺序 FL FR RL RR |
|---|---|---|
| pronking | [0, 0, 0] | [0, 0, 0, 0] |
| trot | [0.5, 0, 0] | [0.5, 0, 0, 0.5] |
| bounding | [0, 0.5, 0] | [0, 0.5, 0.5, 0] |
| pacing | [0, 0, 0.5] | [0.5, 0.5, 0, 0] |

这里有一个必须注意的工程事实：

~~~
当前 bounding 的相对分组与 trot 相同，只是整体 phase 反向平移；
当前 pacing 形成前腿一组、后腿一组的相位关系。
~~~

因此，当前模板的名字不能单独证明实际学到的是传统意义上的 bounding 或 pacing。加入多步态前，必须：

1. 打印实际 phase offsets；
2. 打印每条腿的期望支撑 mask；
3. 记录实际 contact；
4. 在 play 中观察腿对；
5. 确认命名、phase 映射和传感器顺序一致。

如果你希望得到传统定义的 bounding，通常需要重新检查 theta 到腿顺序的映射或直接使用显式四腿 offset，而不是只把字符串改成 bounding。

## 10. 如何从零设计一个步态

建议先不要写代码，先做一张归一化周期表。

### 第一步：确定运动目的

例如：

| 目标 | 可能的步态倾向 |
|---|---|
| 平地稳定前进 | 对角 trot |
| 快速前进 | 更高频 trot 或前后分组 |
| 越过障碍 | 较高摆腿、较低速度 |
| 原地弹跳 | pronking |
| 侧向稳定 | 左右分组或定制时序 |
| 节省能量 | 较低频率、较长支撑 |

先明确你要解决的是速度、稳定、越障还是外观。步态不是只按名字分类。

### 第二步：画一周期的接触表

使用归一化时间 t∈[0,1)：

~~~text
S = 支撑
W = 摆动
~~~

例如对角两组：

~~~text
时间        0-.25   .25-.5   .5-.75   .75-1
FL            S        S        W        W
RR            S        S        W        W
FR            W        W        S        S
RL            W        W        S        S
~~~

这个表表达的是：

- FL/RR 同组；
- FR/RL 同组；
- 两组相差 0.5；
- duty factor 约为 0.5。

### 第三步：把接触表转换为 phase offset

选择一个腿组作为 offset=0，再给另一组 offset=0.5：

~~~text
FL = 0.5
FR = 0.0
RL = 0.0
RR = 0.5
~~~

得到：

~~~python
offsets = [0.5, 0.0, 0.0, 0.5]
~~~

再根据第 8 节的反解公式尝试求 theta。求出来后一定要重新代回代码验证。

### 第四步：单独决定 frequency

frequency 只决定时间尺度：

~~~text
offset 决定谁先动、谁后动
frequency 决定整套时序多快
~~~

不要为了修正腿对关系而修改 frequency。腿对错是 offset 问题，不是频率问题。

第一阶段建议：

~~~text
固定 frequency=2.0 Hz
固定所有身体行为参数
只验证相位关系
~~~

### 第五步：再决定身体行为参数

时序稳定后，再逐步决定：

~~~text
body_height       0.32 m 附近
body_pitch        0.0 rad 附近
stance_width      0.21 m 附近
foot_swing_height 0.07 m 附近
~~~

原则：

- 先保证脚对和支撑时序；
- 再让身体高度跟随行为；
- 再调摆腿高度；
- 最后调站姿宽度和 Raibert 落点。

如果时序尚未正确，过早调 body_height 或 swing height 会把问题掩盖。

## 11. 设计新步态的两种实现方式

### 方式 A：继续使用 theta 参数化

适用于：

- 你的步态能用三 theta 表示；
- 四腿相位关系比较规则；
- 希望复用当前 behavior 维度；
- 希望兼容现有 checkpoint 和观测结构。

改动位置：

~~~text
Train/Nazarite/Nazarite-src/nazarite/mdp/wtw.py
~~~

在 GAIT_THETA 中加入新名字：

~~~python
GAIT_THETA["my_gait"] = (theta1, theta2, theta3)
~~~

再在配置中：

~~~python
gait_names=("trot", "my_gait")
~~~

但这只是让命令能采样新模板。你还必须验证：

- 生成的 offsets 是否符合接触表；
- phase 的支撑/摆动顺序是否正确；
- WTW 的两个接触奖励是否能区分该步态；
- 行为策略是否真的使用了新 gait。

### 方式 B：显式四腿 phase offset

适用于：

- 想设计任意接触序列；
- 三 theta 无法表达；
- 想加入不同 duty factor；
- 想让每条腿拥有不同支撑时长；
- 想设计不对称步态或特殊恢复步态。

概念上可写成：

~~~python
GAIT_PHASE_OFFSETS = {
    "my_gait": torch.tensor([o_fl, o_fr, o_rl, o_rr]),
}
~~~

然后让 _update_phase 直接读取该 offset。若要实现不同 duty factor，还需要扩展 behavior 或 gait 配置，例如：

~~~python
GAIT_DUTY_FACTOR = {
    "my_gait": 0.6,
}
~~~

并让 _smooth_contact_target 接收 duty factor。当前项目还没有把 duty factor 作为 behavior 维度，因此这是结构性扩展，不是单纯改一个奖励权重。

## 12. duty factor 为什么重要

duty factor 是一条腿在一个周期内处于支撑的比例：

~~~
duty_factor = 支撑时间 / 一个完整周期
~~~

当前项目的接触目标约等于 0.5 duty factor。它影响：

- 同时接触的腿数；
- 支撑稳定性；
- 摆动时间；
- 脚抬起的频率；
- 机器人能否在低速时保持稳定。

粗略理解：

| duty factor | 运动特点 |
|---:|---|
| 0.5 附近 | 标准交替步态 |
| 大于 0.5 | 支撑更久、更稳，摆动更短 |
| 小于 0.5 | 摆动更久、更动态，稳定性要求更高 |

如果你需要 crawl 类步态，通常需要 duty factor 大于 0.5，并且四条腿的相位错开，而当前固定约 0.5 的接触目标不够表达它。此时应先扩展 phase target，而不是仅增加 stance reward。

## 13. 步态设计中的稳定性检查

### 13.1 静态支撑多边形

当机器人低速或短暂停留时，至少要考虑：

~~~text
是否有足够的脚同时支撑？
支撑脚是否分布在机体两侧？
质心投影是否落在支撑区域附近？
~~~

如果某个步态在任意时刻只剩一只脚支撑，它对本体状态、动作延迟和随机推力会非常敏感。

### 13.2 动态稳定性

高速时不能只看静态支撑。还要看：

- 摆动脚是否能在下一次接触前到达落点；
- 支撑脚是否发生滑动；
- 机体速度和期望速度误差是否可控；
- 前后方向的落点是否合理；
- yaw 转向是否需要横向落点补偿。

这就是为什么当前项目还需要 stance phase velocity 和简化 Raibert，而不只需要 phase schedule。

### 13.3 速度与频率的匹配

相同 frequency 下，速度越高，每一步需要覆盖的距离越大：

~~~
步长 ≈ 前进速度 / 步频
~~~

例如：

~~~text
1 m/s、2 Hz -> 每周期约 0.5 m
2 m/s、2 Hz -> 每周期约 1.0 m
~~~

速度提高而 frequency 不变时，可能出现：

- 摆腿时间不够；
- 落点过远；
- 机体下沉；
- 接触脚滑动；
- 需要更大的 Raibert 补偿。

所以如果开放速度到 2 m/s 以上，通常还要考虑 frequency 的范围是否足够，而不是只加速度追踪奖励。

## 14. 相位边界、平滑和历史

phase 从 1 回到 0 是周期边界。当前使用 sin/cos phase reference：

~~~python
[sin(2 * pi * phase), cos(2 * pi * phase)]
~~~

sin 和 cos 共同表示完整周期位置，比单个 sin 更不容易产生 phase 歧义。
当前 actor 的历史配置是：

~~~text
普通本体观测：10 帧
behavior：5 帧
phase sin/cos：0 帧
~~~

phase 不堆叠历史是有意设计：sin/cos 当前帧已经包含完整周期位置，避免
重复输入 10 个相同结构的 phase 历史。behavior 保留 5 帧，则可以帮助策略
识别行为参数是否刚刚发生平滑切换。新增 cos、改变 phase 历史和修改 behavior
历史都会改变 actor 输入维度，旧 checkpoint 不能直接加载，必须重新训练。

## 15. phase 初始化和 resampling 的注意事项

当前每个环境 reset 时都会随机初始化 base_phase：

~~~
base_phase = Uniform(0,1)
~~~

这样做的好处：

- 不让所有并行环境同时进入同一相位；
- 减少策略只记住固定时间点动作；
- 覆盖完整 gait 周期。

behavior 在当前配置中 30 秒重新采样一次。时间太短会导致：

- behavior 尚未学会就改变；
- phase 目标突然变化；
- 身体高度和步频瞬间切换。

时间太长则可能降低行为范围覆盖速度。训练阶段先保持较长 resampling time，部署切换要使用插值。

## 16. 如何验证一个新步态，而不是只看总奖励

### 16.1 先做离线数学验证

写一个小测试，输入 theta，输出：

~~~text
四条腿 phase offsets
各个离散 phase 的 desired_contact
一个周期内每条腿的支撑/摆动表
~~~

至少验证：

- offset 在 [0,1)；
- phase 按 frequency 单调推进并正确回绕；
- 相位差符合设计；
- 同组腿的 phase 差接近 0；
- 交替组 phase 差接近 0.5。

### 16.2 再看 TensorBoard

查看：

~~~text
WTW/swing_phase_force_cost
WTW/swing_phase_force_score
WTW/stance_phase_velocity_cost
WTW/stance_phase_velocity_score
WTW/foot_FL_swing_force
WTW/foot_FR_swing_force
WTW/foot_RL_swing_force
WTW/foot_RR_swing_force
~~~

四条腿的摆动相力不应长期只有某一条腿特别高。

### 16.3 最后在 play 中看实际接触

固定速度和固定 behavior，观察：

- 实际 contact；
- 实际足端高度；
- 机体高度；
- 每条腿的摆动时刻；
- 启动和刹停；
- 推力后的恢复。

不要仅通过“看起来像 trot”判断。最好保存一个周期内的接触序列。

## 17. 一个完整的新步态设计例子

假设要设计一个“前后分组”的步态：

### 目标

~~~text
FL、FR 同组
RL、RR 同组
前后两组相差半个周期
duty factor 约 0.5
frequency 先固定 2 Hz
~~~

### 接触表

~~~text
时间        0-.25   .25-.5   .5-.75   .75-1
FL            S        S        W        W
FR            S        S        W        W
RL            W        W        S        S
RR            W        W        S        S
~~~

### 期望 offset

~~~text
[FL, FR, RL, RR] = [0.0, 0.0, 0.5, 0.5]
~~~

### 检查当前 theta 参数化

代入反解：

~~~text
theta1 = offset_RR = 0.5
theta2 = offset_RL = 0.5
theta3 = offset_FL - theta1 = -0.5 mod 1 = 0.5
~~~

再检查 FR：

~~~text
theta2 + theta3 = 0.5 + 0.5 = 1.0 mod 1 = 0.0
~~~

因此这个 offset 组合可以被当前三 theta 结构表达，候选 theta 为：

~~~text
[theta1, theta2, theta3] = [0.5, 0.5, 0.5]
~~~

但不要直接认为它已经是正确步态。还要把它代入代码，输出四条腿 phase，并确认实际接触表与设计一致。

### 配置方式

~~~python
GAIT_THETA = {
    "my_front_hind": (0.5, 0.5, 0.5),
}
~~~

再先只训练：

~~~python
gait_names=("my_front_hind",)
~~~

这样能把“新步态是否正确”和“多步态竞争”分开。

## 18. 新步态的训练顺序

推荐按下面顺序，不要一步开放全部变量：

### 阶段 A：只验证数学映射

~~~text
固定 theta
固定 frequency=2 Hz
固定 body_height=0.32
固定 pitch=0
固定 stance_width=0.21
固定 swing_height=0.07
固定低速 twist
~~~

目标：phase、接触目标和腿顺序正确。

### 阶段 B：验证速度能力

~~~text
先使用低速前进
再加入后退、侧移和 yaw
最后恢复完整 Grid Adaptive
~~~

目标：新步态不是只在单一速度下成立。

### 阶段 C：开放一个行为维度

顺序建议：

~~~text
frequency
  -> foot_swing_height
  -> body_height
  -> body_pitch
  -> stance_width
~~~

每次只打开一个范围，并比较 behavior 日志和实际运动是否同步变化。

### 阶段 D：与 trot 做 A/B

保持：

- 相同速度采样；
- 相同随机推力；
- 相同奖励；
- 相同训练步数；
- 只改变 gait_names。

比较：

~~~text
速度追踪
接触时序
零速站立
受推力恢复
机体高度
足端滑动
~~~

## 19. 常见设计错误

### 错误 1：只修改 gait 名字

字符串变了，不代表 phase 变了。必须修改 theta 或 phase offset。

### 错误 2：把 frequency 当成步态类型

frequency 只改变时间尺度。两个步态可以同频，也可以同一个步态有不同频率。

### 错误 3：只看 policy 的动作，不看期望 phase

动作奇怪时，先画：

~~~text
base_phase
phase_FL、phase_FR、phase_RL、phase_RR
desired_contact
actual_contact
~~~

### 错误 4：把 phase 当作真实接触

phase 是参考，真实接触由仿真决定。两者差异大时应检查奖励和策略，而不是强行把 phase 改成 actual contact。

### 错误 5：忽略 duty factor

想做 crawl，却仍然使用固定 0.5 支撑/摆动比例，通常无法得到真正的 crawl。

### 错误 6：只增加接触奖励权重

如果 phase 映射错了，增大奖励只会更强地惩罚正确动作。先验证相位，再调权重。

### 错误 7：同时开放新 gait、Grid 全范围和大域随机化

出现问题时无法判断是 gait 映射、速度能力、随机化还是奖励冲突。新 gait 第一轮应尽量固定其他变量。

## 20. 设计一个新步态前的检查表

~~~text
[ ] 明确目标：速度、越障、稳定、能量还是外观
[ ] 画出一周期四条腿 S/W 接触表
[ ] 确定 duty factor
[ ] 计算相对 phase offsets
[ ] 按当前 [FL,FR,RL,RR] 顺序排列
[ ] 反解并验证 theta，或决定使用显式 offset
[ ] 固定 frequency 做数学验证
[ ] 验证 desired_contact，而不是只看 gait 名称
[ ] 检查四条腿 site/contact sensor 顺序
[ ] 固定低速训练新 gait
[ ] 再逐步开放速度和行为参数
[ ] 做与 trot 的 A/B 对比
[ ] 测试 0 -> motion -> 0 的冻结与刹停
[ ] 测试随机推力
~~~

## 21. 最终理解

可以用一个类比理解：

~~~text
behavior = 乐谱中规定的曲风和节拍
phase    = 乐曲当前播放到哪一拍
contact reward = 判断演奏是否按节拍
policy   = 根据乐谱、当前拍子和身体状态实际演奏
~~~

在 Nazarite-mjlab 中：

~~~text
theta 决定四条腿的相对节奏
frequency 决定节奏速度
phase 把节奏推进到当前时刻
body_height 等参数决定运动外观
reward 让真实接触接近期望接触
policy 学习如何用关节动作完成这一切
~~~

所以，独立设计一个步态的核心不是先调奖励，而是：

~~~text
先设计接触时序
再设计 phase offset
再验证 theta/offset 映射
再加入身体行为参数
最后用奖励训练策略遵循它
~~~
