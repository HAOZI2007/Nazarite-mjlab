# Nazarite Training

Nazarite 的强化学习训练项目，基于本地 mjlab 和 RSL-RL 构建。

## 当前开发状态

当前主线是：

- Go2 平地速度控制：普通 Grid Adaptive baseline 和 WTW Trot 任务；
- FR-Net 平地/复杂地形摔倒恢复。

DELTA 和 SMP 专用任务、算法及部署实现已移除。XZ1/Go2 的其余 12 个任务继续保留，
任务 ID 不变；代码分类和迁移说明见 [框架目录管理说明](../../docs/框架目录管理说明.md)。

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
    │   ├── go2/                  # Go2 环境与算法配置
    │   ├── xz1/                  # XZ1 环境与算法配置
    │   └── shared/               # 共享基础环境
    ├── tasks/                    # go2/、xz1/ 任务注册
    ├── rl/                       # him/、xz1/、wmp/、fr_net/
    ├── scripts/                  # go2/、xz1/ 工具入口
    └── mdp/                       # 自定义观测、奖励、命令等
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
| `Nazarite-HIM-Complex-Terrain-Go2` | HIM 复杂地形速度控制。 |
| `Nazarite-WMP-HIM-Flat-Go2` | WMP + HIM 平地。 |
| `Nazarite-WMP-HIM-Complex-Terrain-Go2` | WMP + HIM 复杂地形。 |
| `Nazarite-Velocity-Flat-XZ1-WTW` | XZ1 WTW 平地。 |
| `Nazarite-HIM-Complex-Terrain-XZ1` | XZ1 HIM 障碍地形。 |
| `Nazarite-WMP-HIM-Collect-XZ1` | XZ1 教师数据采集。 |
| `Nazarite-WMP-HIM-Teacher-XZ1` | XZ1 特权地形教师。 |
| `Nazarite-WMP-HIM-Eval-XZ1` | XZ1 地形评估。 |

从本目录执行：

~~~bash
uv sync
uv run list-envs
uv run train Nazarite-Velocity-Flat-Go2-WTW
uv run play Nazarite-Velocity-Flat-Go2-WTW \
  --checkpoint_file logs/rsl_rl/go2_flat_wtw_independent/<run>/model_2400.pt
~~~

在 AutoDL 上训练时，使用 [autodl/README.md](autodl/README.md) 的上传、环境准备、持久化日志和断点续训流程；实际启动命令由 `autodl/train.sh` 转发到同一个 `train` 入口。

`play` 不传 `--checkpoint_file` 时需要 `wandb_run_path`，因此本地检查已训练模型时建议显式传入 checkpoint。WTW play 会保留随机推力作为独立抗扰动检查；网页 `Commands / Behavior` 面板可临时覆盖当前选中环境的行为参数。

WTW 当前默认是固定 Trot，行为频率范围为 `2.0–3.0 Hz`，Grid Adaptive 使用
5×5 的 x/yaw 网格；actor 普通观测堆叠 10 帧，critic 堆叠 3 帧，behavior
堆叠 5 帧，phase 的 sin/cos 不堆叠历史。具体参数以
`Nazarite-src/nazarite/config/train_config/base_env_cfg.py` 为准。

sim2sim 的运行、模型接口、观测维度、WTW phase 和手柄控制说明见
[sim2sim/README.md](sim2sim/README.md)。


## 配置约定

- robot_config 只负责机器人 XML/MJCF、实体、执行器、初始状态和尺度参数。
- train_config 负责组装 ManagerBasedRlEnvCfg 与 RslRlOnPolicyRunnerCfg。
- mdp 负责 Nazarite 专属的观测、奖励、命令、事件、终止和课程函数。
- 任务通过 mjlab.tasks entry point 暴露给 mjlab 的 train 和 play 命令。
- `config/train_config/base_env_cfg.py` 是 WTW 默认行为、Grid、观测历史和奖励组合的唯一组装位置；不要在多个任务文件中分散覆盖同一组参数。
- `output/`、训练日志和模型权重不得提交，完整规则见仓库根目录
  `CONTRIBUTING.md`。
