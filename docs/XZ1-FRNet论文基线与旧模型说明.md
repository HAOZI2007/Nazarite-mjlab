# XZ1 FR-Net 论文基线与旧模型说明

> 2026-11-01：本文保留迁移前的 FR-Net 实现记录。默认入口 `Nazarite-FRNet-Recovery-XZ1` 已切换为 Playground 普通 PPO 起身方案，参见同目录《XZ1-FRNet平地恢复训练与使用.md》。本文的旧模型应使用 `Nazarite-FRNet-Paper-Recovery-XZ1`；下文历史命令中的旧任务名需要替换为该名称。环境与 RL 配置分别迁至 `xz1/frnet_legacy_env_cfg.py`、`xz1/frnet_legacy_rl_cfg.py`。

本任务在 XZ1 上训练侧翻、仰倒、趴地后的恢复起身，同时训练少量站立保持。任务 ID 为 `Nazarite-FRNet-Recovery-XZ1`，日志目录为 `Train/Nazarite/logs/rsl_rl/xz1_frnet_recovery_flat/`。本阶段使用 plane，不含障碍物、地形课程或 HIM 策略切换。

## 代码入口

以下路径均相对仓库根目录：

| 文件 | 作用 |
| --- | --- |
| `Train/Nazarite/Nazarite-src/nazarite/config/train_config/xz1/frnet_env_cfg.py` | XZ1 资产、传感器、奖励、随机化、终止条件 |
| `Train/Nazarite/Nazarite-src/nazarite/config/train_config/xz1/frnet_rl_cfg.py` | FR-Net Actor、Critic 与 PPO 参数 |
| `Train/Nazarite/Nazarite-src/nazarite/mdp/xz1/frnet.py` | 碰撞几何安全重置、位置动作、采样与成功指标 |
| `Train/Nazarite/Nazarite-src/nazarite/mdp/xz1/frnet_curriculum.py` | 按姿态成功率升级的 reset 课程、惩罚权重阶段及统计 |
| `Train/Nazarite/Nazarite-src/nazarite/rl/xz1/frnet_runner.py` | 保存和恢复课程等级、统计窗口及阶段起点 |
| `Train/Nazarite/Nazarite-src/nazarite/mdp/fr_net/paper_rewards.py` | 论文表 I、公式 (8)–(10) 的奖励核函数 |
| `Train/Nazarite/Nazarite-src/nazarite/rl/fr_net/frnet_actor.py` | 历史编码器、质量/接触预测、变分 latent、下一帧解码器和策略网络 |
| `Train/Nazarite/Nazarite-src/nazarite/rl/fr_net/frnet_ppo.py` | PPO、MCP 监督、下一帧配对、重建与 KL 损失 |
| `Train/Nazarite/Nazarite-src/nazarite/scripts/xz1/evaluate_frnet_recovery.py` | XZ1 四类初始姿态的独立评测 |

XZ1 有自己的配置和重置逻辑，使用当前 XZ1 MJCF、PD 参数与 12 个实际关节，膝关节名为 `shank_joint`。这里没有替换或修改 XZ1-HIM，也没有修改外置 mjlab。

## 网络与论文的对应关系

Actor 的单帧输入为 42 维：角速度 3、投影重力 3、相对默认站姿的关节角 12、关节速度 12、上一帧原始动作 12。每项保存五帧，ObservationManager 输出 210 维 term-major 历史，FRNetActor 还原为五帧时间序列。

MCP 输出质量预测 4、接触预测 13、latent 16。Actor 使用当前 42 维观测和这些预测，得到 75 维增强输入；策略 MLP 为 512/256/128，输出 12 维关节动作。

XZ1 的四维质量标签顺序为机身、四个髋部合计、四个大腿合计、四个小腿合计，数值为当前质量与标称质量之比。它沿用论文的身体部件分组，并使用比例改善数值尺度。XZ1 独立的四个足端 body 保持标称惯量，不纳入本次质量随机化和四维标签。原有 Go2 默认的四条腿质量标签语义继续保留。

13 维接触顺序为：机身，四髋，四大腿，四小腿。每个四维块均按 `FL、FR、RR、RL` 排列，与 XZ1 的 MJCF 顺序一致。接触传感器使用全局净力，质量/接触真值只进入 Critic 与辅助监督组，Actor 在推理时仅使用本体历史。

Critic 输入为 270 维：原有 83 维当前状态/特权信息，加上 187 维局部高度图。Actor 仍然只接收 210 维本体历史，不会收到高度图或质量、接触真值。Critic 并没有逐项复制论文的全部特权状态（例如 PD 增益、摩擦和质心偏置），不要将 270 维称为论文完整的 Critic 维数。

现在 XZ1 默认启用论文公式 (3)–(7) 的 MCP/VAE 结构：历史编码器产生质量预测、接触 logits、16 维 latent 均值与 log-variance；latent 使用标准正态先验。训练时通过重参数化 `z = mean + exp(0.5 * logvar) * noise` 采样，解码器使用 `[mass_prediction, sigmoid(contact_logits), z]` 这 33 维信息预测下一帧 42 维本体观测。解码器隐藏层为 128/256；log-variance 限制在 [-10, 10] 以控制指数运算范围。

当前总优化目标为：

```text
L = L_PPO + value_coef * L_value - entropy_coef * entropy
    + 0.25 * (MSE_mass + BCE_contact + MSE_next_proprio + 0.01 * KL_latent)
KL_latent = mean_batch[0.5 * sum_latent(mean² + exp(logvar) - 1 - logvar)]
```

这里的 0.25、1.0 重建权重和 0.01 KL 权重是本项目的初始配置，不是论文明确给出的数值；KL 对 latent 维度求和，重建 MSE 对 42 个通道取均值。权重可在 `xz1/frnet_rl_cfg.py` 中调整。

Actor 使用每通道共享的 42 维在线均值/方差归一化，历史五帧和重建目标采用同一组统计量；Critic 也启用在线观测归一化。归一化统计随 checkpoint 保存，并包含在部署 Actor 导出中。策略分支使用 latent 均值，随机采样只用于解码重建，保证 PPO 重新计算动作概率时不因重新采样 latent 而改变策略条件。该选择是明确的工程适配；论文未给出足以逐项核对的 PPO/VAE 采样实现。

`frnet_proprio` 是新增的训练监督组：按 Actor 单帧的相同顺序，输出无附加观测噪声的 42 维当前观测，保留相同编码器偏置语义。采样时把 `history_t` 与 `frnet_proprio_(t+1)` 配对保存，不能在 minibatch 随机打乱后用相邻样本猜下一帧。rollout 最后一帧也有正确的下一帧目标。凡环境返回 done（包括超时），其下一观测可能已经是新回合 reset 观测，因此屏蔽该样本的重建损失；当前帧质量、接触监督和 latent KL 仍可计算。全批 done 时重建损失为有限的零。

本次只在 XZ1 默认打开 `use_vae=True`。共享的 FRNetActor 默认仍为 `False`，保留既有 Go2 Actor 的参数键；Go2 不会自动增加解码器。XZ1 的网络、归一化和 Critic 输入均已变化，应重新训练，不能直接加载上一版 83 维 Critic 的 XZ1 checkpoint。

## 高度传感器核查

XZ1 现在有三组不同用途的高度射线，均只命中地形 geom group 0，采用 yaw 对齐，机器人翻到背面时射线仍向世界下方：

| 传感器 | 配置 | 用途 |
| --- | --- | --- |
| `base_height_scan` | 机身原点附近半径 0.03 m 的 4 条射线，取最小净高 | 恢复质量指标、评测及穿地保护；不参与奖励 |
| `foot_height_scan` | 四足 site 各 4 条射线，半径 0.02 m | Critic 足端高度 |
| `frnet_terrain_height_map` | 1.6 × 1.0 m、间距 0.1 m，17 × 11 = 187 点 | Critic 特权高度图 |

前两组上一版已经存在，第三组本次补齐。平地上图内每个点的地面 world-Z 都相同，但传感器输出是 `base_z - ground_z`，随起身高度变化，并非固定零。187 点数量来自论文，高度图的具体覆盖尺寸是本项目选定的配置。高度图裁剪到 [-1, 1] 后进入 Critic。它不是部署时新增的外部感知要求。

另核查到旧 Go2 平地工厂确有一处遗漏：Critic 保留高度图观测项，却仅在复杂地形分支创建同名传感器。已改为只要 `use_critic_height_map=True`，平地和复杂地形都创建传感器。Go2 原有地图采样尺寸保留不变。

## 初始姿态与穿地预防

最终难度（level 5）下，当前等级样本的 reset 概率为：

| 初始姿态 | 概率 |
| --- | --- |
| 左侧倒地 | 20% |
| 右侧倒地 | 20% |
| 仰倒，四腿朝上 | 20% |
| 趴地，膝关节折叠、腿外展 | 20% |
| 半倒地 | 15% |
| 站立保持 | 5% |

初始化环境时，以真实 XZ1 碰撞体预计算每个课程等级、每类各 64 个姿态。每个姿态都计算旋转后的 box、cylinder、sphere 等碰撞体最低点，再整体抬升至最低点距 plane 8 mm。随机 yaw 不改变最低点的高度，因此运行时可以在设备上快速采样姿态并随机朝向。关节角同时落在实际关节范围内。

训练姿态库的侧躺/仰卧类别中，35% 使用折腿起点，其余使用默认站姿附近的关节角。各阶段关节噪声幅度从 ±0.12 rad 增至 ±0.35 rad，避免只练习“默认站姿整体旋转后倒地”。每一种关节配置都单独重新计算离地间隙。Play/独立评测保留原先完整倒地、默认站姿附近关节角的基础姿态库，便于与旧模型比较；并不跟随课程变简单。这还不是对任意折腿、自碰撞姿态的泛化评测。

这一机制避免用统一 root 高度放置不同朝向的机器人，也避免从 0.5 m 高空下落代替贴地起身。它保证初始机器人与地面的几何间隙，不保证整个动态恢复过程绝无接触穿透；MuJoCo 的软约束仍可能产生小量正常穿透。

Play 中只采样四类完全倒地姿态，每类 25%，每回合 350 个控制步，即当前 50 Hz 下的 7 秒。训练和 Play 均不会在起身成功后提前终止，可以观察剩余时间内是否重新倒下或持续蹦跳。网页中的 Reset Environment 会重新采样；7 秒超时也会开始下一回合。

## 动作与控制频率

物理步长 2 ms、decimation 10，控制周期为 20 ms，即 50 Hz。动作目标为：

```text
q_desired = q_default + 0.7 * action - encoder_bias
```

目标角每控制步最多变化 0.15 rad，并限制在实际 soft joint limits。控制周期内十个物理子步保持相同目标，不逐子步累加相对当前位置的偏移。

倒地 reset 后前六个控制步保持 reset 关节目标，随后六步逐渐接管，相当于 120 ms 保持和 120 ms 渐进接管。动作项用独立计数器记录时间，避免训练入口随机化 `episode_length_buf` 后跳过接管过程。部分环境 reset 时，仅清理对应环境的动作目标与计数器。

## 从 reset 到站立的实际时序

| 时间段 | 实际行为 |
| --- | --- |
| reset 瞬间 | 直接选取倒地/半倒地/站立姿态，设置机身姿态、关节角及速度，并留 8 mm 初始地面间隙 |
| 0–0.12 s | 倒地样本的 PD 目标保持初始关节角，让重力与地面接触自然落稳；不冻结机体 |
| 0.12–0.24 s | 策略目标与初始目标的混合比例从 1/6 增到 1，并继续受每步 0.15 rad 变化限制 |
| 约 0.24 s 起 | 完全由策略目标接管；实际关节仍受力矩、PD 和目标变化速度限制 |
| 满足站立条件后 | 连续维持 0.8 s 后记为成功指标，继续控制到 350 步超时；不增加奖励、不触发重置 |

保持与接管期间策略也在计算动作，仿真、观测和奖励照常进行；保持的是关节目标，不是整个身体静止。站立保持样本跳过上述 hold/ramp，立即按策略控制。独立评测的恢复时间包含这段控制接管时间，不额外扣除。

因此本任务不是“先静止几秒，再触发一次摔倒恢复”，也不是“先通过脚本起身，再训练站立”。起身动作全部由策略学习，没有预设翻身动作序列，也没有起身后切换 HIM。120 ms 保持、120 ms 接管是每次倒地 reset 的控制时序；下面的课程则是在许多回合之间逐步调整训练分布，两者不同。

## 平地恢复课程（2026-11-01）

旧训练发现趴卧能起身，但侧躺常滚成仰卧后停住，因此现在启用姿态难度与动作正则的联合课程。参考来源为：

- `mjlab_playground-main/src/mjlab_playground/getup/getup_env_cfg.py`：早期较弱的动作变化/关节速度惩罚，后期逐步收紧；其 reset 事件还使用随机四元数和宽范围随机关节角。
- `FR-Net-main/legged_gym/envs/go2/go2_recovery/go2_recovery_env.py`：随机四元数 reset，以及从低趴到高站姿的 1500 次迭代目标插值。该仓库实际的 orientation 惩罚使用完整重力误差，和论文表 I 的 `gx²+gy²` 不相同，不能把其成功直接归因于课程。

本项目采用“先探索、后平滑”的思路，并增加 XZ1 专用的成功门控姿态课程。没有照搬 Go2 的目标关节角或改变 XZ1 的站姿奖励目标；这不是声称两个参考仓库已经实现了下表的课程。

| 等级 | 左/右倾角绝对值 | 仰卧类别倾角 | 左/右/背/腹/半倒/站立采样比例 | 五项正则权重乘数 |
| --- | --- | --- | --- | --- |
| 0 | 25–45° | 不采样 | 20/20/0/50/5/5% | 0.10 |
| 1 | 45–75° | 不采样 | 25/25/0/40/5/5% | 0.20 |
| 2 | 75–100° | 100–130° | 22.5/22.5/15/30/5/5% | 0.35 |
| 3 | 80–100° | 130–160° | 22.5/22.5/20/25/5/5% | 0.55 |
| 4 | 80–100° | 160–180° | 22.5/22.5/25/20/5/5% | 0.75 |
| 5 | 83–97° | 173–187° | 20/20/20/20/15/5% | 1.00 |

角度是 reset 时的 roll 范围，另有小幅 pitch 扰动；最终 180° 附近即完全仰卧。早期 `left_side`/`right_side` 代表向对应侧倾斜的练习类别，不表示一开始就完整侧躺；`back` 在 level 2 才加入，从斜仰逐渐走到完全仰卧。

每一级至少停留 12000 个控制步，当前每次 PPO rollout 24 步，相当于约 500 次迭代。与此同时，当前等级实际采样的四个倒地类别中，**每个类别**都至少收集 128 个结束回合，曾站稳成功率 ≥70%、最终站稳率 ≥60%，才允许升级。左右侧分别判定；开始采样背部类别后，它也必须单独通过。半倒、站立、旧等级复习样本及零步 reset 不参与升级。未通过则重开统计窗口，避免最初探索失败永久拉低成功率；不会仅因训练时间增长而强制升级，也不会自动降级。

level >0 时，每次 reset 有 20% 概率从已通过的较低等级中均匀选择一级复习，其余 80% 使用当前等级。上表是每个等级内部的采样比例，实际总体分布还包含这些复习样本。因此即使达到 level 5，也会保留一些简单姿态来维持自起身能力。

课程只缩放 `joint_velocity_limit`、`joint_acc`、`joint_vel`、`action_rate`、`joint_torques` 五项惩罚；限位和接触安全等项全程保留最终权重。奖励形式、正奖励、ε=0.15、XZ1 目标站姿、7 秒回合时长及成功标准保持原定义；在 level 5 五项惩罚也恢复下表的完整论文权重。这意味着**训练早期不是固定论文权重基线**，是明确加入课程的适配版本。

TensorBoard 记录 `Curriculum/recovery/level`、`penalty_scale`、`stage_steps` 以及每类的 `*_trials`、`*_success_rate`、`*_final_stable_rate`。每类率的分母只包含该类、当前等级的已结束回合，避免被腹部起身或站立样本稀释。统计窗口重开后先显示上个窗口的率；新等级还没有该类有效统计时显示 -1，表示未知，不是负成功率。

课程等级、阶段开始步数和统计窗口随 checkpoint 保存。恢复新 checkpoint 会还原课程；加载旧 checkpoint 时即使原训练迭代数很大，也从 level 0 课程重新开始并重新等待最短阶段时间，不能凭迭代数跳过翻身练习。Play 和独立评测完全关闭课程，始终测试完整倒地姿态。

XZ1-HIM 的标称站姿作为恢复目标。当前任务不附加 HIM 专用的站姿伺服前馈补偿，策略需要学习抵消重力负载。FR-Net 动作配置中的 offset/scale 不应与 HIM checkpoint 的动作解释混用。

## 奖励与成功定义

当前任务使用论文表 I 的 13 项奖励。表中列的是核函数和最终权重，训练早期五项正则乘以上一节的阶段系数；按 mjlab 惯例，整项统一乘控制周期 `dt=0.02 s` 后相加。

| 配置名 | 核函数 / 含义 | 权重 |
| --- | --- | --- |
| `base_orientation` | `gx² + gy²` | -0.5 |
| `upright` | `exp(-(gz+1)² / (2 ε²))` | 6.0 |
| `stand_pose` | `exp(-sum((q-qstand)²)) * 1[abs(gz+1) < ε]` | 4.0 |
| `foot_contacts` | 四足接触指示器之和 | 0.3 |
| `body_contact` | 任意非足部件接触地形的单个二值指示器 | -0.2 |
| `safety_force` | 四个膝部对应 body 的世界 XY 接触力 L2 范数之和 | -0.01 |
| `body_bias` | 相对本回合 reset 机身 XY 位置的欧氏距离，截断到 [0, 4] m | -0.1 |
| `joint_limits` | 超出实际关节位置上下限的关节数量 | -1.0 |
| `joint_velocity_limit` | `sum(max(abs(qdot)-0.8, 0))` | -0.1 |
| `joint_acc` | 关节加速度平方和 | -2.5e-6 |
| `joint_vel` | 关节速度平方和 | -0.01 |
| `action_rate` | 原始策略动作相邻两步差值的平方和 | -0.01 |
| `joint_torques` | 执行器力矩平方和 | -5e-4 |

向量平方项采用按关节求和；论文式中的 `||.||_2` 按欧氏范数解释，不额外平方。没有额外机身高度奖励、稳定高度支撑、安静支撑或 +20 成功奖励。不启用总奖励非负截断或乘法组合，负奖励保留。

论文未充分说明的实现约定在本项目中显式固定如下，不能把这些默认值都称为论文原值：

- `ε=0.15`，直立高斯与目标站姿门槛使用同一个参数。数值沿用参考仓库直立高斯默认值；论文未报告 ε 的数值，参考仓库自身的目标姿态实现也并非完整论文基线。
- `qstand` 使用 XZ1 的标称默认站姿，不套用 Go2 关节角。目标姿态采用关节平方误差之和，不除以额外的标准差。
- 接触指示器直接使用当前仿真 contact 的 `found > 0`，不加力阈值或跨控制步滤波；一只脚有多个 contact slot 时仍只计一次。身体接触是任意机身、髋、大腿、小腿触地时取 1，不按接触部件数量累加，也不包含机器人自碰撞。
- XZ1 的 `shank_link` 从膝关节开始，四个小腿 body 的地形接触净力作为膝部力的对应测量。使用 `netforce` 提供的世界坐标力，去掉 Z 分量后计算 L2 范数，不做 `tanh` 饱和或力尺度归一化；这会包含该 body 上的小腿接触，并非只在膝关节单点测力。
- 表 I 的 Angular Velocity Limit 公式写的是 `qdot`，因此约束关节速度，阈值按表中字面值 0.8 rad/s 实现；不是机身角速度，也不是电机速度上限的 80%。参考仓库的 `max_velocity` 函数也采用这个解释。
- 位置越界指示器检查模型实际关节限位；控制器仍单独使用软限位限制动作目标。仅改变动作目标并不代替奖励中的实际关节位置检查。
- XY 参考位置在每次 reset 时记录实际采样到的机身位置；部分环境 reset 仅更新对应环境，其他环境的参考位置不变，不会把每步位置改写为新参考。

取消成功提前终止后，站立质量继续用独立状态化指标观察。连续满足以下条件 0.8 秒，指标 `recovery_success` 才为真：

- 投影重力 Z ≤ -0.90；
- 相对地面高度 ≥ 0.25 m；
- 至少三足接触，各足接触力 ≥ 5 N；
- 机体角速度范数 ≤ 1.5 rad/s；
- 全部关节相对默认站姿的平均绝对误差 ≤ 0.35 rad。

达到上述成功条件不改变奖励、不重置环境。论文报告每回合 350 个时间步；本任务将它作为 350 个控制步，在当前控制频率下为 7 秒。倒地、机身接触和大角度本身不会终止任务；除超时外，保留无效状态和明显穿地的仿真保护终止。

`Episode_Metrics/recovery_success` 用回合最大值，表示本回合曾连续稳定 0.8 秒；`recovery_final_stable` 用回合末值，表示结束时仍连续稳定 0.8 秒。两者有差距说明可能站起后又失稳。`fallen_recovery_success` 把站立样本成功置零，其均值的分母仍是所有回合，因此不是严格的“倒地样本条件成功率”。比较恢复能力应同时使用下面的四类姿态独立评测。

## 域随机化

- 每次 reset 对机身、髋、大腿、小腿的正定惯量随机化，质量比例约为 0.85–1.15；质量标签从实际仿真模型读取。
- 启动时随机化机器人碰撞体摩擦系数，范围 0.5–1.2。
- 启动时 PD 增益乘以 0.9–1.1。
- 启动时编码器偏置 ±0.01 rad。
- 训练 Actor 的角速度、重力、关节角和关节速度有观测噪声；play 和独立评测关闭观测噪声。
- 本阶段执行器延迟固定为零，无周期推倒事件；动力学随机化在 play/评测仍保留。

## 启动命令

在训练项目目录执行。建议本次从头训练，便于验证课程能否学到翻身，避免继续沿用旧策略的仰卧局部最优。此次课程没有改变网络维数，同架构的旧 XZ1 FR-Net checkpoint 可以加载，但会从 level 0 重新开始课程；Go2 FR-Net 或 XZ1-HIM checkpoint 不具有本任务的标签和动作语义。

```bash
cd /home/haozi/桌面/Nazarite-mjlab/Train/Nazarite
uv run train Nazarite-FRNet-Recovery-XZ1 --env.scene.num-envs 1024
```

先跑小规模端到端检查可使用：

```bash
uv run train Nazarite-FRNet-Recovery-XZ1 \
  --env.scene.num-envs 64 --agent.max-iterations 10
```

如果当前终端已激活项目 `.venv`，也可直接使用 `train` / `play`，或使用 `.venv/bin/python -m mjlab.scripts.train` / `.venv/bin/python -m mjlab.scripts.play` 调用相同入口。

选择实际生成的 checkpoint 后播放：

```bash
uv run play Nazarite-FRNet-Recovery-XZ1 \
  --checkpoint-file logs/rsl_rl/xz1_frnet_recovery_flat/<run>/model_<iteration>.pt \
  --num-envs 1 --viewer viser
```

四类姿态独立评测：

```bash
uv run python -m nazarite.scripts.xz1.evaluate_frnet_recovery \
  --checkpoint-file logs/rsl_rl/xz1_frnet_recovery_flat/<run>/model_<iteration>.pt \
  --num-envs 64 --horizon-s 10 --stable-duration-s 1
```

评测沿用训练的重置姿态库定义，并只使用左侧、右侧、仰倒和趴地四类，无站立保持样本。默认每类 64 次，成功要求持续稳定 1 秒，禁用自动重置并运行完整 10 秒，同时记录最终稳定率、平均恢复时间、最大位移和角速度以及无效状态率。结果默认保存到 checkpoint 所在目录的 `recovery_eval.json`，可用 `--output-file` 改位置。恢复时间从 episode reset 开始计时，包含接管阶段；总平均恢复时间按成功试验数加权。

目前评测与训练使用相同的 64 个基础姿态/类别，不是未见姿态泛化测试；yaw、动力学与采样可因 seed 不同而变化。建议至少对多个 seed 评测，再考虑新增未见姿态库和更强随机化。

## 如何判断训练是否有效

优先观察 `Episode_Metrics/recovery_success`、`Episode_Metrics/recovery_final_stable` 与高度、足接触、站姿、角速度分项通过率；再结合独立评测的 `success_rate` 和 `final_stable_rate`。当前没有 `Episode_Termination/recovery_success`，因为成功只记录指标，不提前结束回合。

单纯 upright 奖励变高但高度/支撑不通过，往往表示仍在趴地；只有某一类成功可能表示学会了单侧翻身；成功率高但最终稳定率低，表示站起来后仍会失稳。训练平均 reward 或 MCP 接触预测准确率高，均不能单独证明已经学会起身。

辅助学习新增 `Loss/mcp_reconstruction`、`Loss/mcp_vae_kl` 和 `Loss/mcp_next_valid_fraction`。重建误差应在一致的归一化尺度下观察；KL 不是越小越好，持续接近零且重建不改善可能是 latent 失效。`next_valid_fraction` 是未跨 reset 的重建样本占比，不是恢复成功率。原有 `mcp_mass`、`mcp_contact` 和 `mcp_contact_accuracy` 继续记录。

确认平地四类姿态都能稳定恢复后，再增加摩擦/质量范围、延迟或复杂地形，并另开实验目录比较。

## 本次实现验证

论文奖励、XZ1 FR-Net 与任务注册测试共 17 项通过。检查包含已知姿态下的奖励值、接触点去重、世界 XY 力范数的单位与非饱和计算、真实关节限位、速度门槛、位移截断、独立成功指标的保持/部分重置，以及 350 步回合长度。另覆盖 MuJoCo 原生几何距离核验初始地面间隙、接触通道顺序、真实模型质量标签、部分环境 reset 的 XY 参考位置、每控制步动作限幅、MCP/VAE 参数更新、下一帧时序与 reset 屏蔽、三组高度传感器真实射线命中、checkpoint 保存/加载、确定性 TorchScript 导出一致性，以及四类姿态评测和 JSON 输出。

Ruff、相关文件 Pyright 和 `git diff --check` 通过。另用标准 `train` CLI 在 4 个 CPU 环境中完成一轮含论文奖励与 VAE 的训练，日志、配置快照、模型保存和导出均可用；单轮更新仅验证训练链路，不是恢复效果基准。旧 Go2 平地此前也通过实际 CPU reset/步进检查，Critic 为 146 维，地图为 63 点。

另在 2 个 CPU 环境中从站立姿态执行零动作，连续运行 350 个控制步，验证成功指标不会触发提前重置，超时严格在第 350 步生效，并逐步核对实际总奖励等于 13 项加权和乘 `dt`。这项检查没有使用训练好的策略，不能当作恢复成功率评测。

当前执行环境无法连接 NVIDIA 驱动，因此未验证 GPU 运行和长期训练收敛效果；这些检查仅说明训练任务接线正确，不能证明已学会恢复。
