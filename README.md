# Nazarite-mjlab

## 鸣谢

- 本项目极大程度参考了山东华宇工学院 HYNova 战队轮足机器人项目的开源仓库，在这里十分感谢HYNova战队对本框架的大力支持。https://github.com/zeitvex/RC_WheelLeg/tree/main

Nazarite东莞理工学院行者实验室 轮腿机器人 MuJoCo 强化学习与 sim-to-real 工程。

## 当前目录

~~~text
.
├── Train/
│   └── Nazarite/
│       ├── mjlab/                 # 通用 MuJoCo 强化学习基础库
│       ├── .venv/                 # 当前已安装的 Python 环境
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

该环境已经安装 mjlab 的主要依赖，包括 MuJoCo、PyTorch CUDA 12.8、tensordict、RSL-RL、Pyright 和 Ruff。

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

## 当前可用任务

项目当前提供两项 Go2 平地速度任务，二者都使用 Grid Adaptive 速度课程：

| 任务 ID | 用途 |
|---|---|
| `Nazarite-Velocity-Flat-Go2` | 不使用 WTW 的速度跟踪 baseline，用于 A/B 对照。 |
| `Nazarite-Velocity-Flat-Go2-WTW` | WTW 条件策略：Trot 行为、phase 时序和独立行为辅助奖励。 |

WTW 当前默认训练阶段：速度范围 `[-1.0, 1.0] m/s`、3 个 x 方向 Grid cell、固定 Trot，频率在 `2.0–2.4 Hz` 小范围采样；横向和 yaw 指令固定为零。详情见 [WTW 从零实现说明](docs/WTW-从零手写Walk-These-Ways.md)。

~~~bash
cd Train/Nazarite
uv run list-envs
uv run train Nazarite-Velocity-Flat-Go2-WTW
uv run play Nazarite-Velocity-Flat-Go2-WTW \
  --checkpoint_file logs/rsl_rl/go2_flat_wtw_independent/<run>/model_2400.pt
~~~

网页 play 的 `Commands` 区域同时提供速度控制器和 WTW `Behavior` 面板。打开 `Enable override` 后，可以实时修改当前选中环境的频率、机体高度偏移、pitch、步宽和摆腿高度；这仅用于验证 checkpoint，不会改变训练配置。

## 引用区
