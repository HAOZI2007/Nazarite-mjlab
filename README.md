# Nazarite-mjlab

Nazarite 轮腿机器人 MuJoCo 强化学习与 sim-to-real 工程。

## 当前目录

~~~text
.
├── Train/
│   └── Nazarite/
│       ├── mjlab/                 # 通用 MuJoCo 强化学习基础库
│       │   └── .venv/             # 当前已安装的 Python 环境
│       ├── Nazarite-src/nazarite/ # 自定义训练包
│       ├── pyproject.toml         # 项目依赖和任务 entry point
│       └── DEPENDENCIES.md        # 训练框架依赖说明
├── pyrightconfig.json             # 工作区级 Python 解析配置
├── .vscode/settings.json          # VS Code 解释器和额外解析路径
└── Sim2Real/                      # sim-to-real 代码预留目录
~~~

训练项目说明见 [Train/Nazarite/README.md](Train/Nazarite/README.md)，详细依赖关系见 [Train/Nazarite/DEPENDENCIES.md](Train/Nazarite/DEPENDENCIES.md)。

## 当前环境

当前使用的解释器是：

~~~text
Train/Nazarite/mjlab/.venv/bin/python
~~~

该环境已经安装 mjlab 的主要依赖，包括 MuJoCo、PyTorch CUDA 12.8、tensordict、RSL-RL 和 Pyright。

VS Code 工作区级配置位于：

- [pyrightconfig.json](pyrightconfig.json)
- [.vscode/settings.json](.vscode/settings.json)

## 基础库检查

从基础库目录执行：

~~~bash
cd Train/Nazarite/mjlab
uv sync
uv run pyright -p pyproject.toml src/mjlab
~~~

当前 mjlab 源码已通过 188 个文件的 import/type 检查。

## 自定义任务

Nazarite 自定义任务仍在搭建中，需要继续完成机器人模型、环境配置、MDP 函数和任务注册后，再执行：

~~~bash
cd Train/Nazarite
uv run train <task-id>
uv run play <task-id>
~~~
