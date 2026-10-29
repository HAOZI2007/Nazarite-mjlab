# XZ1 WMP 第一批：特权地形教师与固定评测

实现日期：2026-10-26。本批不是完整视觉学生，也不意味着已训练出新的越障能力。

## 本批新增任务

| 注册名 | 用途 | actor 的地形输入 |
| --- | --- | --- |
| `Nazarite-WMP-HIM-Teacher-XZ1` | HIM 热启动的特权地形教师 PPO | 132 维真值地形扫描，经 MLP 压成 32 维 |
| `Nazarite-WMP-HIM-Eval-XZ1` | 固定地形/难度的名义条件评测与查看 | 同教师；独立评测脚本还支持原盲 HIM checkpoint |

原 `Nazarite-HIM-Complex-Terrain-XZ1` 和 Go2 任务不变。未加入 AMP、深度相机、World Model、学生蒸馏或实机部署代码。名称中的 WMP 表示训练路线，不表示本批教师已经使用 WMP 世界模型。

## 与参考仓库一致的部分和必要适配

- 主要参考 `/home/haozi/桌面/extreme-parkour-main/rsl_rl/rsl_rl/modules/actor_critic.py` 的 `Actor.scan_encoder`：`132 → 128 → 64 → 32`，中间 ELU、末层 Tanh。没有自行换成 CNN、注意力或额外的潜变量算法。
- 扫描点使用 Extreme Parkour `legged_robot_config.py` 的 12×11 个坐标；x 为 −0.45～1.20m，y 为 −0.75～0.75m，间距 0.15m，按 x 外层/y 内层展平。这里不是原 XZ1 critic 的 187 点扫描。
- 扫描值沿用其 `legged_robot.py` 中 `clip(base_z - 0.3 - terrain_z, -1, 1)`。不经过 HIM 历史归一化，也不二次归一化。参考默认的水平扫描噪声也是 0。
- 原仓库从高度场取值；这里适配为 MuJoCo yaw 对齐的垂直 raycast，只扫地形 group 0、排除机器人。射线起点上移 1m，以读到高于机身的墙顶，输出仍按机身高度计算。没有移植高度场三邻点最小值滤波；几何边沿采样与参考实现不完全相同。
- HIM 替代参考教师里的历史编码/估计路径，这是用户选定的 HIM 路线适配，不宣称完全复现 Extreme Parkour 整套 actor。
- 保留当前 XZ1 的关节顺序 FL/FR/RR/RL、动作尺度/偏置、PD、50Hz 控制、47 维 actor 帧、6 帧历史、244 维 critic、50 维 HIM 监督，以及现有奖励、终止、训练课程和域随机化。没有把 Go2 的动作或奖励整包继承过来。

actor 历史输入仍为 `[N,6,47]`，内部转换成当前帧在前的 282 维历史。

最终 actor MLP 输入为：`47 当前帧 + 3 HIM速度 + 16 HIM潜变量 + 32 地形潜变量 = 98` 维。critic 保持 244 维。HIM 和扫描编码器都参与训练，HIM 维持自身监督/对比损失更新，扫描编码器由 PPO 更新。

## 热启动和续训不是同一件事

`--agent.initial-checkpoint` 用于从旧 XZ1-HIM **仅迁移权重**：actor、HIM、critic、归一化统计和动作分布参数一并保留；actor 首层由 66 扩到 98，新增 32 列置零。因此同一历史下，训练开始前的动作保持一致（浮点运算误差范围内）。扫描编码器正常初始化；学习第一步时先更新其与 actor 的连接，随后编码器开始学习。

不加载旧 PPO/HIM Adam 动量，不沿用旧迭代号或环境步计数；教师课程重新从当前 XZ1 起始等级开始。数值/历史契约、参数键和形状不符合要求时直接报错，不用 `strict=False` 忽略不匹配。

`--agent.resume True` 才是教师 checkpoint 的续训：恢复模型、两套优化器、迭代与已有环境课程状态。不与 `initial-checkpoint` 同用。与现有 runner 一样，这不是逐位恢复随机数和所有仿真器状态。

## 当前地形范围

沿用现有 XZ1 已训练的七类：平地、金字塔楼梯、随机网格、宽实心梯形坡、高墙、0.8m 窄楼梯、0.8m 窄坡。先控制变量验证教师能利用地形信息，再扩展 gap、梅花桩、独木桥、轮胎和限高；本批没有擅自改赛道或奖励。

训练保留原 XZ1 的窄路中心线辅助速度指令。**固定评测改为普通恒定速度指令，没有使用地形真值纠偏**，因此窄路评测可能比训练更难，报告不能混为一谈。

## 启动教师训练

所有指令在 `/home/haozi/桌面/Nazarite-mjlab/Train/Nazarite` 执行。第一轮先检查 1000 次迭代，不一次性无人值守长训。800 环境只是与已有 XZ1 运行规模对齐，可按显存减少。

```bash
cd /home/haozi/桌面/Nazarite-mjlab/Train/Nazarite
uv run --no-sync train Nazarite-WMP-HIM-Teacher-XZ1 \
  --env.scene.num-envs 800 \
  --agent.initial-checkpoint /home/haozi/桌面/Nazarite-mjlab/Train/Nazarite/logs/rsl_rl/xz1_him_complex/2026-10-25_22-04-31/model_31300.pt \
  --agent.max-iterations 1000 \
  --agent.run-name stage1_scan_teacher
```

新日志写入 `logs/rsl_rl/xz1_wmp_him_teacher/`，不覆盖原 HIM 日志。没有额外调整奖励/学习率；先看基础行走是否保持、HIM 损失/动作裁剪/数值恢复是否异常，再看相同固定难度下完成度是否提升。

教师续训，替换真实的新训练目录名和 checkpoint 名：

```bash
uv run --no-sync train Nazarite-WMP-HIM-Teacher-XZ1 \
  --env.scene.num-envs 800 \
  --agent.resume True \
  --agent.load-run '<教师训练目录名>' \
  --agent.load-checkpoint model_900.pt \
  --agent.max-iterations 1000
```

查看已训练教师：

```bash
uv run --no-sync play Nazarite-WMP-HIM-Teacher-XZ1 \
  --checkpoint-file '<教师 checkpoint 的绝对路径>' \
  --num-envs 1
```

固定任务 `play Nazarite-WMP-HIM-Eval-XZ1` 默认是难度 0 的窄楼梯、0.4m/s；play 会自动 reset，便于查看。正式计分使用下面的脚本，脚本关闭 auto-reset，先读取终止状态再手动 reset，防止把 reset 后的位置计作通过。

## 固定评测

先评原 HIM 建立对照：

```bash
uv run --no-sync python -m nazarite.scripts.evaluate_xz1_wmp \
  --policy him \
  --checkpoint /home/haozi/桌面/Nazarite-mjlab/Train/Nazarite/logs/rsl_rl/xz1_him_complex/2026-10-25_22-04-31/model_31300.pt \
  --terrains flat high_wall narrow_stairs narrow_ramp \
  --difficulties 0 0.5 1 \
  --episodes 20 --num-envs 4 --seed 42 --speed 0.4 --device cuda:0 \
  --output logs/evaluation/xz1_him_31300_nominal.json
```

教师使用同一命令，将 `--policy` 改为 `teacher`、checkpoint 改为教师路径、output 改为新文件名。输出文件已存在时拒绝覆盖。CPU 检查可用 `--device cpu`。

- 固定每个地形的难度与随机种子；关闭课程、推扰、物理随机化、传感器噪声和执行/观测延迟。不是保留训练随机化的鲁棒性评估。
- 成功要求：四脚均超过障碍远端 5cm，机身超过远端 35cm，至少两脚承重，倾斜不超过 45°，连续保持 0.2s；任何失败终止优先于成功。
- 运行中横向越出评测路线直接失败，不能从障碍旁绕行再回来刷通过；窄路沿用实际宽度和 XZ1 的 12cm 机身边界余量。
- 记录逐回合通过/失败、失败原因、持续时间、前向距离、速度 MAE，严格计满指定回合数，不因并行环境同帧结束而超计。
- 平地和网格是固定距离行走；金字塔从中心顶台出发，明确标记为**下楼梯评测**，不能用它宣称上楼能力。窄楼梯才覆盖完整上/平台/下。
- 本版关闭随机化且起点固定，重复回合不是独立随机场景；20/20 不能解释为可靠的实机成功率或统计置信度。
- 目前是单一正向速度的课程完成度测试，不替代 P1 中前后左右、转向、站立的完整速度网格验收。

## 导出和部署边界

教师保存时导出双输入 ONNX：`obs_history[N,282]`（当前帧在前）、`teacher_scan[N,132]`。元数据标明特权扫描，不会把扫描悄悄删掉导出成盲策略。TorchScript 同样有两路输入。**教师不可直接部署为深度相机策略**；后续视觉学生才以真实相机/WMP 特征替代特权地形编码。

## 验证

新增单元测试覆盖扫描顺序、HIM 权重/归一化等价迁移、扫描梯度、双输入导出、配置隔离、目标边界和落脚判定。CPU 集成覆盖实际观测维度、奖励/终止/指标、两次 PPO 更新、HIM 更新、两套优化器恢复、ONNX Runtime 对齐、并行评测计数。

最终针对性回归共 **36 项通过**（11 项新增参数化测试用例及原 XZ1 窄路、姿态、终止观测回归），新增生产模块 Pyright 为 0 错误、0 警告，新增文件 Ruff 检查通过。剩余测试警告来自已有 PyTorch TorchScript/ONNX 导出接口弃用提示。

```bash
uv run --no-sync pytest -c mjlab/pyproject.toml \
  Nazarite-src/tests/test_xz1_teacher.py \
  mjlab/tests/test_xz1_teacher_smoke.py -q -p no:cacheprovider
```

另已运行真实 GPU CLI：8 环境、2 次迭代、每次 8 步，成功加载 `model_31300.pt` 并完成 PPO/HIM 更新、checkpoint 和 ONNX 保存；动作裁剪比例与数值恢复计数均为 0。测试日志仅写入 `/tmp/xz1_wmp_teacher_first_batch_gpu_smoke_20261026/`，没有启动长训。

原 HIM 的固定低难度（0）短基线中，平地、5cm 高墙、窄楼梯、窄坡各完成 1/1 次；这只说明最小评测流程和基本行走可用，不代表高难度成功率。完整报告为 `/tmp/xz1_him_31300_teacher_first_batch_baseline_20261026.json`。

这些检查验证训练闭环和接口正确，不代表长训收敛、800 环境 GPU 规模性能或实机安全已经验证。
