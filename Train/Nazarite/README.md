# Nazarite Training

Nazarite 的强化学习训练项目，基于本地 mjlab 和 RSL-RL 构建。

## 目录结构

~~~text
Train/Nazarite/
├── pyproject.toml
├── mjlab/                         # 通用 MuJoCo 强化学习基础库
│   ├── src/mjlab/
│   ├── RSL-RL/
│   └── .venv/                     # 当前开发环境
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
mjlab/.venv/bin/python
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

## 训练命令

自定义任务层目前仍在搭建中。完成机器人模型、环境配置、MDP 函数和任务注册后，从本目录执行：

~~~bash
uv sync
uv run list-envs
uv run train <task-id>
uv run play <task-id>
~~~

## 配置约定

- robot_config 只负责机器人 XML/MJCF、实体、执行器、初始状态和尺度参数。
- train_config 负责组装 ManagerBasedRlEnvCfg 与 RslRlOnPolicyRunnerCfg。
- mdp 负责 Nazarite 专属的观测、奖励、命令、事件、终止和课程函数。
- 任务通过 mjlab.tasks entry point 暴露给 mjlab 的 train 和 play 命令。
