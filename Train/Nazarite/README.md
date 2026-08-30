# Nazarite Training

Nazarite 的强化学习训练项目，基于本地 mjlab 和 RSL-RL 构建。

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
    ├── config/train_config/        # 环境和 PPO 配置
    └── mdp/                       # 自定义观测、奖励、命令等
~~~

详细的依赖关系、任务注册链路和当前缺口见 [DEPENDENCIES.md](DEPENDENCIES.md)。

## 环境与 import 检查

当前虚拟环境位于：

~~~text
.venv/bin/python
~~~

基础库依赖已经安装，包括 MuJoCo、PyTorch CUDA 12.8、tensordict、RSL-RL、Pyright 和 Ruff。

从本项目目录执行基础库检查：

~~~bash
cd mjlab
uv sync
uv run pyright -p pyproject.toml src/mjlab
~~~

当前 mjlab 的 188 个源码文件已经通过 import/type 检查。

VS Code 工作区在上一级仓库目录打开时，使用：

~~~text
Train/Nazarite/mjlab/.venv/bin/python
~~~

对应的工作区配置是仓库根目录的 pyrightconfig.json 和 .vscode/settings.json。

## 训练与播放

当前已注册的任务：

| ID | 说明 |
|---|---|
| `Nazarite-Velocity-Flat-Go2` | Grid Adaptive 平地速度 baseline。 |
| `Nazarite-Velocity-Flat-Go2-WTW` | Grid Adaptive + WTW Trot 条件速度策略。 |

从本目录执行：

~~~bash
uv sync
uv run list-envs
uv run train Nazarite-Velocity-Flat-Go2-WTW
uv run play Nazarite-Velocity-Flat-Go2-WTW \
  --checkpoint_file logs/rsl_rl/go2_flat_wtw_independent/<run>/model_2400.pt
~~~

`play` 不传 `--checkpoint_file` 时需要 `wandb_run_path`，因此本地检查已训练模型时建议显式传入 checkpoint。WTW play 会保留随机推力作为独立抗扰动检查；网页 `Commands / Behavior` 面板可临时覆盖当前选中环境的行为参数。

## 配置约定

- robot_config 只负责机器人 XML/MJCF、实体、执行器、初始状态和尺度参数。
- train_config 负责组装 ManagerBasedRlEnvCfg 与 RslRlOnPolicyRunnerCfg。
- mdp 负责 Nazarite 专属的观测、奖励、命令、事件、终止和课程函数。
- 任务通过 mjlab.tasks entry point 暴露给 mjlab 的 train 和 play 命令。
- `config/train_config/base_env_cfg.py` 是 WTW 默认行为、Grid、观测历史和奖励组合的唯一组装位置；不要在多个任务文件中分散覆盖同一组参数。
