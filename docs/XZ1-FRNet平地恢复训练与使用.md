# XZ1 平地起身：Playground 迁移版

## 当前任务

从 2026-11-01 起，`Nazarite-FRNet-Recovery-XZ1` 使用 `mjlab_playground-main` 的 Go1 getup 方案适配 XZ1。保留旧任务名是为了沿用入口；**默认任务现在是普通 PPO 起身基线，不再是论文 FR-Net 网络**。日志改存到 `Train/Nazarite/logs/rsl_rl/xz1_getup_flat/`。

旧 MCP/VAE、论文奖励和成功门控课程保留在 `Nazarite-FRNet-Paper-Recovery-XZ1`，可回放旧 checkpoint，详见《XZ1-FRNet论文基线与旧模型说明.md》。新旧网络和动作语义不同，checkpoint 不能互用；新入口遇到旧模型会提示正确任务名。

参考源码位于 `mjlab_playground-main/src/mjlab_playground/getup/`：`getup_env_cfg.py`、`config/go1/{env_cfgs,rl_cfg}.py`、`mdp/{actions,events,rewards,terminations}.py`。Apache-2.0 许可证和来源说明位于 `third_party/mjlab_playground/`。没有安装或修改参考仓库，也没有修改外置 mjlab。

| 部分 | 当前实现 |
| --- | --- |
| Actor | 42 维单帧本体观测，512/256/128 ELU MLP，输出 12 维动作，不做观测归一化 |
| Critic | 本体状态 + 3 维线速度，共 45 维，启用观测归一化 |
| 算法 | 普通 PPO，无 MCP/VAE、质量接触监督或高度图输入 |
| reset | 60% 随机朝向及宽范围关节角下落，40% 默认站姿 |
| Play reset | 100% 随机下落；可指定标准姿态用于评测 |
| 动作 | 每物理子步 `q_current + 0.6 * action - encoder_bias` |
| 接管 | 倒地样本前 25 控制步（0.5 秒）忽略动作偏移，站姿样本立即接管 |
| 回合 | 最长 6 秒；超功率或无效状态提前终止；成功不提前终止 |

Actor 观测顺序：角速度 3、投影重力 3、关节角相对默认站姿 12、关节速度 12、上一帧动作 12。训练噪声分别为 ±0.2、±0.05、±0.03、±1.5；Critic 关闭噪声且使用真实关节角，Play 关闭 Actor 噪声。

PPO 沿用参考参数：动作初始标准差 1.0，学习率 1e-3、自适应 KL 调度、entropy 0.01、gamma 0.99、lambda 0.95、clip 0.2、5 epoch、4 minibatch、rollout 24。不额外裁剪动作。默认训练 3000 次迭代，比参考 Go1 的 2000 次更长，给 XZ1 留出最终约束下的训练时间。

## 此次迁移原因

`2026-11-01_11-25-23` 到约 4226 次迭代仍停留在 level 0，五项正则一直只有完整权重的 10%。末段 200 个日志点的曾站稳指标约 13.2%、最终站稳约 0.46%。这些是训练指标，不是独立测试成功率。课程没有推进，平滑约束也没有收紧，与站起后不稳定的表现相符。

迁移采用参考仓库按步数收紧正则的课程，不再等待姿态成功门槛。是否解决翻身和抖动需要完整训练验证，功能测试不能证明收敛。

## reset 与 XZ1 适配

参考方案采用随机单位四元数、软限位内随机关节角、0.5 m 高度释放。本实现启动时预计算 2048 个随机姿态，每次抽取后叠加随机 yaw；各姿态根据 XZ1 真实碰撞几何计算安全高度，root 放在 `max(0.5 m, 安全高度)`，使初始碰撞体距 plane 至少 8 mm。root 六维速度、关节速度随机为 ±0.5。站姿使用 XZ1 默认关节角，检查站姿高度后额外抬高 2 cm。

这是有限姿态库适配，并非每次 reset 重新连续采样所有关节角。换 seed 会生成新姿态库。它保证初始地面间隙，不保证随机姿态无自碰撞，也不保证动态软接触绝无小量穿透。

保留 XZ1 的实际 PD、力矩限幅和 2 ms 物理步长，decimation 10；参考 Go1 为 5 ms ×4，两者控制频率均为 50 Hz。动作不再使用旧版默认站姿偏置、目标变化限速或渐进混合，也不跨子步累积目标。物理关节限位和力矩限幅仍生效。

倒地前 0.5 秒忽略动作偏移，让下落/接触自然发生，不冻结机体。settle 用独立控制步计数器，避免 runner 初始随机回合长度跳过接管；部分 reset 只清理对应环境。

## 奖励与平滑课程

| 项目 | 核函数 / 含义 | 权重 |
| --- | --- | --- |
| orientation | `exp(-2 * sum((g-[0,0,-1])²))` | 1 |
| torso_height | `(exp(min(h,0.30))-1)/(exp(0.30)-1)` | 1 |
| posture | 正立误差 <0.01 时，`exp(-mean(((q-q_default)/std)²))` | 1 |
| action_rate_l2 | 相邻原始动作差值平方和 | -0.01 → -0.10 |
| joint_vel_l2 | 关节速度平方和 | 0 → -0.01 |
| joint_torques_l2 | 实际执行器力矩平方和 `sum(tau²)` | -2e-6 → -1e-5 |
| feet_support | 起身后同时支撑的足端比例平方 `(n_contact/4)²` | 0.2 |
| synchronized_landing | 四足在 0.12 s 窗口内落地，且随后同时支撑；每个恢复回合最多一次 | 0.1，一次性奖励 |
| dof_pos_limits | 超出软限位的距离和 | -1 |
| self_collisions | 超过 10 N 自碰撞的子步计数，按 4 子步等效尺度归一 | -0.1 |

高度 h 使用当前地面净高，目标改成 XZ1 的 0.30 m，而非 Go1 的 0.275 m。posture std 对 hip/thigh/shank 分别为 0.05/0.10/0.15 rad，沿用参考设置适配关节名；XZ1 hip 轴不同，这些不是已验证的最优参数。奖励有符号相加并乘 dt=0.02，不做非负截断或额外成功奖金。超过目标高度不再增加高度奖励。

足端奖励和力矩惩罚是本项目在 Playground 基线上的新增项。两项足端奖励均要求 gz≤-0.90、相对地面高度≥0.25 m、机身角速度≤1.5 rad/s，并且已结束被动 settle。接触使用足端与地面的力模长≥5 N 判定，不包括自碰撞。共同支撑奖励在 0/1/2/3/4 足接触时分别为 0/0.0625/0.25/0.5625/1，再乘 0.2；鼓励起身后保持四足支撑，趴低时不给这项奖励。

同步落地奖励只用于倒地恢复样本。每只足必须在 settle 结束后先连续离地至少 0.04 s，再产生有效接触；四足最近落地时间距当前均不超过 0.12 s，且当前全部接触地面时，奖励一次。按 50 Hz 控制步采样，不要求四足在完全相同的物理子步撞地，也不奖励冲击大小。已有接触不会在 reset 后被当作新落地；每回合最多发放一次，反复蹦跳不能重复获得此奖金。每个环境的离地时间、落地时间和发奖标记在其 reset 时单独清理。核函数返回 `1/dt` 脉冲，所以经过 RewardManager 的 dt 缩放后，这次事件实际加 0.1，而非 0.002。

力矩惩罚与机械功率终止不同：`sum(tau²)` 在关节静止但持续大力顶住时仍然生效。采用较小权重并逐步加强，以减少持续高力矩而不直接限制翻身动作。在最终权重下，若 12 个关节都为 10 N·m，惩罚率为 -0.012；都达到 31.7 N·m 时约 -0.121，明显小于三项主奖励的最大总和 3。权重是初始调参值，需用恢复率、最终站稳率及力矩 RMS 验证。

| 迭代阈值（rollout=24） | 动作变化权重 | 关节速度权重 | 力矩权重 | 功率终止阈值 |
| --- | --- | --- | --- | --- |
| 0 | -0.01 | 0 | -2e-6 | 不限制 |
| 500 | -0.05 | -0.005 | -5e-6 | 1000 W |
| 900 | -0.08 | -0.008 | -8e-6 | 700 W |
| 1200 | -0.10 | -0.010 | -1e-5 | 400 W |

阈值实际使用累计控制步 `迭代数 ×24`，不是并行环境样本总数；更改 rollout 长度会改变对应迭代数。功率为 `sum(abs(torque * qdot))`，单位 W，不是累计能量，settle 前 25 步豁免。阈值沿用 Go1，需观察 XZ1 是否因必要翻身动作频繁提前终止。

课程随 checkpoint 的累计步数恢复，load 后立即同步有效参数。Play 无课程，使用最终惩罚和 400 W 阈值。独立评测关闭终止与自动 reset，完整运行指定时长。

本次只增改奖励和诊断，不改观测、网络和动作语义，已有 `xz1_getup_flat` 模型可以直接续训。恢复超过 1200 轮的检查点后，力矩惩罚直接采用最终权重，不会重新走弱惩罚课程；新奖励按当前代码生效，原检查点与历史日志不会被改写。

## 碰撞与随机化

仅本任务覆盖碰撞参数：足端 condim 6、其余碰撞体 condim 3，solref=(0.01,1)、优先级 1，视觉 mesh 不碰撞，impratio=10、elliptic 摩擦锥。内部非排除碰撞对参与自碰撞；保留 MJCF 的显式排除对。自碰撞计数乘 4/10，以避免 10 个子步相比参考 4 个子步放大惩罚。

启动随机化包括：编码器偏置 ±0.015 rad；机身 COM XYZ 偏移 ±0.025/±0.025/±0.03 m；所有碰撞体滑动摩擦 0.3–1.5；足端旋转摩擦 1e-4–2e-2、滚动摩擦 1e-5–5e-3（后两项 log-uniform）。关闭本任务执行器延迟，不改 HIM；不接入旧 FR-Net 的质量/PD 随机化及标签。

## 训练和回放

从头训练新任务：

```bash
cd /home/haozi/桌面/Nazarite-mjlab/Train/Nazarite
uv run train Nazarite-FRNet-Recovery-XZ1 --env.scene.num-envs 1024
```

从已验证能起身的模型继续训练新增奖励，例如额外 10000 轮：

```bash
uv run train Nazarite-FRNet-Recovery-XZ1 \
  --env.scene.num-envs 1024 \
  --agent.resume \
  --agent.load-run 2026-11-01_12-51-13 \
  --agent.load-checkpoint model_2999.pt \
  --agent.max-iterations 10000 \
  --agent.run-name feet_torque_10k
```

短流程检查可加 `--env.scene.num-envs 64 --agent.max-iterations 10`。新模型播放：

```bash
uv run play Nazarite-FRNet-Recovery-XZ1 \
  --checkpoint-file logs/rsl_rl/xz1_getup_flat/<run>/model_<iteration>.pt \
  --num-envs 1 --viewer viser
```

旧模型回放：

```bash
uv run play Nazarite-FRNet-Paper-Recovery-XZ1 \
  --checkpoint-file logs/rsl_rl/xz1_frnet_recovery_flat/2026-11-01_11-25-23/model_4200.pt \
  --num-envs 1 --viewer viser
```

## 评测恢复能力和抖动

训练 `getup_success` 沿用参考仓库，只要求曾达到近正立和目标高度，不要求持续站稳。训练含 40% 站姿样本，不能把均值当成倒地恢复成功率。

保留严格的 `recovery_success` 和 `recovery_final_stable`：gz≤-0.90、高度≥0.25 m、至少三足各≥5 N、机身角速度≤1.5 rad/s、关节平均站姿误差≤0.35 rad，持续 0.8 秒。前者表示曾站稳，后者表示回合结束仍站稳。

增加 `joint_speed_rms`、`joint_torque_rms`、`action_delta_rms`、`base_angular_speed`、`peak_power_w`。这些训练统计含翻身/下落，不等同站稳后的抖动。TensorBoard 的 `Episode_Reward/feet_support`、`Episode_Reward/synchronized_landing`、`Episode_Reward/joint_torques_l2` 可观察新增奖励贡献；日志按回合时长归一，一次性落地奖金对应的日志值不一定等于 0.1。不要只追求奖励上涨，仍需确认各倒地姿态的恢复率与最终站稳率。独立评测仅在“正立且高度≥0.25 m”时额外统计关节速度 RMS、机身角速度 RMS、动作变化 RMS；没有站立采样时为 null，不能当成零抖动。

```bash
uv run python -m nazarite.scripts.xz1.evaluate_frnet_recovery \
  --checkpoint-file logs/rsl_rl/xz1_getup_flat/<run>/model_<iteration>.pt \
  --num-envs 64 --horizon-s 10 --stable-duration-s 1
```

每类 64 次，测试左侧/右侧/仰卧/趴卧四组标准贴地姿态，以及一组随机下落。前四组沿用旧姿态库便于比较；后者检查训练分布。总成功率按所有组等权汇总；比较旧模型时单独比较共同四组，或加 `--no-include-random-drops`。旧模型评测指定 `--task Nazarite-FRNet-Paper-Recovery-XZ1`。

同 seed 使用同一随机姿态库，换 seed 才能检查新库；单次结果不是无限姿态泛化证明。应同时比较各组成功率、最终站稳率和站立阶段抖动指标。

## 代码入口

- `Train/Nazarite/Nazarite-src/nazarite/config/train_config/xz1/frnet_env_cfg.py`：当前 Playground 迁移配置。
- `Train/Nazarite/Nazarite-src/nazarite/config/train_config/xz1/frnet_rl_cfg.py`：普通 PPO 参数。
- `Train/Nazarite/Nazarite-src/nazarite/mdp/xz1/getup.py`：安全随机 reset、相对动作、奖励及诊断。
- `Train/Nazarite/Nazarite-src/nazarite/rl/xz1/getup_runner.py`：误载检查与课程恢复。
- `Train/Nazarite/Nazarite-src/nazarite/scripts/xz1/evaluate_frnet_recovery.py`：新旧方案评测。
- `Train/Nazarite/Nazarite-src/nazarite/config/train_config/xz1/frnet_legacy_{env,rl}_cfg.py`：旧 FR-Net 配置。
