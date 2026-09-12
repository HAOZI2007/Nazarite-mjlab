# Nazarite Training

Nazarite 的强化学习训练项目，基于本地 mjlab 和 RSL-RL 构建。

## 当前开发状态

当前主线是：

- Go2 平地速度控制：普通 Grid Adaptive baseline 和 WTW Trot 任务；
- FR-Net 平地/复杂地形摔倒恢复。

SMP 相关实现目前已经**停止更新**。代码、工具、数据目录和已生成产物仍保留，
仅用于复现实验、检查历史结果和阅读实现；后续不再新增 SMP 功能、修复 SMP
训练问题或继续围绕 SMP 调参。独立跳跃任务已经移除，不再是当前训练入口。

## 目录结构

~~~text
Train/Nazarite/
├── pyproject.toml
├── mjlab/                         # 通用 MuJoCo 强化学习基础库
│   ├── src/mjlab/
│   ├── RSL-RL/
├── .venv/                         # 当前开发环境
└── Nazarite-src/nazarite/         # Nazarite 自定义训练包
    ├── __init__.py                # 任务发现和注册入口
    ├── config/robot_config/        # 机器人与执行器配置
    ├── config/train_config/
    │   ├── env_cfgs/              # 通用 Go2 环境配置
    │   ├── frnet_config/          # FR-Net 环境与 PPO 配置
    │   ├── smp_config/            # 已冻结的 SMP 教师/下游环境与 PPO 配置
    │   └── train_algorithm/smp/   # 已冻结的 SMP prior、GSI、guidance 和教师算法
    └── mdp/                       # 自定义观测、奖励、命令等
├── tools/smp_tools/               # 已冻结的 3DDogs → Go2 离线工具
├── tools/smp_dataset/             # 已冻结的 SMP 数据子集与本地扩散 prior
└── output/                        # 本地生成产物，禁止提交
~~~

详细的依赖关系、任务注册链路和当前缺口见 [DEPENDENCIES.md](DEPENDENCIES.md)。

## 环境与 import 检查

推荐从本目录执行 `uv run`，由项目配置管理运行环境：

~~~text
uv run python
~~~

项目依赖包括 MuJoCo、PyTorch、tensordict、RSL-RL、Pyright 和 Ruff。

从本项目目录执行基础库检查：

~~~bash
cd mjlab
uv sync
uv run pyright -p pyproject.toml src/mjlab
~~~

当前 mjlab 的 188 个源码文件已经通过 import/type 检查。

检查 Nazarite 自定义包：

~~~bash
uv run ruff check Nazarite-src/nazarite
uv run pyright -p pyproject.toml Nazarite-src/nazarite
~~~

VS Code 工作区在上一级仓库目录打开时，使用：

~~~text
Train/Nazarite/mjlab/.venv/bin/python
~~~

对应的工作区配置是仓库根目录的 pyrightconfig.json 和 .vscode/settings.json。

## 训练与播放

当前已注册的任务：

| ID | 说明 |
|---|---|
| `Nazarite-Velocity-Flat-Go2` | 主线：Grid Adaptive 平地速度 baseline。 |
| `Nazarite-Velocity-Flat-Go2-WTW` | 主线：Grid Adaptive + WTW Trot 条件速度策略。 |
| `Nazarite-FRNet-Recovery-Go2` | 主线：平地 FR-Net 摔倒恢复。 |
| `Nazarite-FRNet-Recovery-Terrain-Go2` | 主线：复杂地形 FR-Net 摔倒恢复。 |
| `Nazarite-SMP-Teacher-Go2` | 已冻结：SMP 参考动作物理闭环教师。 |
| `Nazarite-SMP-Forward-Go2` | 已冻结：冻结 SMP prior 的前向速度任务。 |

从本目录执行：

~~~bash
uv sync
uv run list-envs
uv run train Nazarite-Velocity-Flat-Go2-WTW
uv run play Nazarite-Velocity-Flat-Go2-WTW \
  --checkpoint_file logs/rsl_rl/go2_flat_wtw_independent/<run>/model_2400.pt
~~~

`play` 不传 `--checkpoint_file` 时需要 `wandb_run_path`，因此本地检查已训练模型时建议显式传入 checkpoint。WTW play 会保留随机推力作为独立抗扰动检查；网页 `Commands / Behavior` 面板可临时覆盖当前选中环境的行为参数。

WTW 当前默认是固定 Trot，行为频率范围为 `2.0–3.0 Hz`，Grid Adaptive 使用
5×5 的 x/yaw 网格；actor 普通观测堆叠 10 帧，critic 堆叠 3 帧，behavior
堆叠 5 帧，phase 的 sin/cos 不堆叠历史。具体参数以
`Nazarite-src/nazarite/config/train_config/base_env_cfg.py` 为准。

## SMP 冻结说明

SMP 代码不再更新，但为可复现性保留。默认 prior、数据处理、GSI、扩散 prior
训练入口和 SMP 两个注册任务都属于历史实验资产。除非需要复现旧结果，否则
不要把 SMP 加入新的奖励组合、课程或部署流程；`smp-pretrain` 也仅保留用于
旧实验复现。

SMP 历史流程见 [Nazarite-SMP 全链路数据处理与训练指南](../../docs/Nazarite-SMP全链路数据处理与训练指南.md)。

## 配置约定

- robot_config 只负责机器人 XML/MJCF、实体、执行器、初始状态和尺度参数。
- train_config 负责组装 ManagerBasedRlEnvCfg 与 RslRlOnPolicyRunnerCfg。
- train_config/train_algorithm 负责训练任务专属算法实现；SMP 代码统一放在其
  `smp/` 子包中。
- mdp 负责 Nazarite 专属的观测、奖励、命令、事件、终止和课程函数。
- 任务通过 mjlab.tasks entry point 暴露给 mjlab 的 train 和 play 命令。
- `config/train_config/base_env_cfg.py` 是 WTW 默认行为、Grid、观测历史和奖励组合的唯一组装位置；不要在多个任务文件中分散覆盖同一组参数。
- `output/`、训练日志和模型权重不得提交，完整规则见仓库根目录
  `CONTRIBUTING.md`。
