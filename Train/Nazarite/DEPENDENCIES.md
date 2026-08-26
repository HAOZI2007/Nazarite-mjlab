# Nazarite Train 目录与依赖关系

## 1. 当前工程结构

训练项目根目录是 Train/Nazarite：

~~~text
Train/Nazarite/
├── pyproject.toml                 # 训练包依赖和 entry point
├── uv.lock                        # 由 uv sync 生成的项目锁文件
├── mjlab/                         # 通用 MuJoCo 强化学习基础库
│   ├── src/mjlab/                 # mjlab Python 包
│   ├── RSL-RL/                    # 本地 RSL-RL 依赖
│   └── uv.lock                    # mjlab 自身的锁文件
└── Nazarite-src/
    └── nazarite/                  # Nazarite 自定义训练包
        ├── __init__.py            # 任务发现和注册入口
        ├── config/
        │   ├── robot_config/       # 机器人、执行器和模型资源配置
        │   └── train_config/       # 环境配置和 RL/PPO 配置
        └── mdp/                   # 自定义观测、奖励、命令、事件等
~~~

仓库工作区根目录还包含编辑器解析配置：

~~~text
Nazarite-mjlab/
├── pyrightconfig.json             # 工作区级 Pyright 配置
└── .vscode/settings.json          # VS Code 默认解释器和额外路径
~~~

当前编辑器使用的虚拟环境是 Train/Nazarite/mjlab/.venv，而不是 Train/Nazarite/.venv。

mjlab 是基础库，nazarite 是使用基础库的项目插件。推荐的依赖方向是：

~~~text
nazarite 自定义训练包
    ├── 依赖 mjlab 的环境、场景、传感器、manager 和 RL 接口
    ├── 依赖 mjlab 内部的 RSL-RL runner / algorithm
    └── 依赖 Nazarite 自己的 robot_config、train_config 和 mdp
                         │
                         ▼
                  mjlab 基础库
~~~

mjlab 不应反向导入 Nazarite 的机器人代码。它通过 mjlab.tasks entry point 发现自定义包，这属于插件发现机制。

## 2. mjlab 基础库

Train/Nazarite/mjlab 是一个独立的 mjlab Python 包，依赖定义在 mjlab/pyproject.toml，主要提供：

| 路径 | 职责 |
| --- | --- |
| mjlab/src/mjlab/envs | Manager-Based RL 环境和环境配置基类 |
| mjlab/src/mjlab/scene、entity | 场景、机器人实体、关节和执行器 |
| mjlab/src/mjlab/sim | MuJoCo/MuJoCo-Warp 仿真 |
| mjlab/src/mjlab/sensor | 接触、射线、相机等传感器 |
| mjlab/src/mjlab/managers | action、observation、command、reward、event、termination、curriculum 和 metric manager |
| mjlab/src/mjlab/envs/mdp | 通用 MDP 函数和域随机化函数 |
| mjlab/src/mjlab/rl | RSL-RL 配置、向量环境 wrapper 和 runner 接口 |
| mjlab/src/mjlab/tasks/registry.py | 任务注册、查询和配置加载 |
| mjlab/src/mjlab/scripts/train.py | train <task-id> 训练入口 |
| mjlab/src/mjlab/scripts/play.py | play <task-id> 推理和可视化入口 |
| mjlab/RSL-RL | 本地 RSL-RL 源码依赖 |

基础库负责如何运行强化学习，不负责 Nazarite 的 XML、关节名称、奖励设计和任务名称。

## 3. Nazarite-src/nazarite 自定义包

自定义包通过 Train/Nazarite/pyproject.toml 安装。当前 entry point 是：

~~~toml
[project.entry-points."mjlab.tasks"]
nazarite = "nazarite"
~~~

因此 Nazarite-src/nazarite 必须是可导入的 Python 包。Nazarite-src 是源码根目录，不是 Python 包名。

### 3.1 nazarite/__init__.py

基础库自动发现任务时的调用链是：

~~~text
mjlab.__init__
    └── 读取 mjlab.tasks entry points
          └── 导入 nazarite
                └── 导入配置工厂
                      └── register_mjlab_task(...)
                            └── mjlab.tasks.registry._REGISTRY
~~~

当前任务入口从 `config.train_config.env_cfgs.go2_env_cfgs` 导入环境配置，从
`config.train_config.rl_cfg` 导入训练配置；机器人本体配置位于
`config.robot_config`。目录结构和导入路径应保持一致。

完成配置后，还需要在 __init__.py 中调用 register_mjlab_task(...)，否则 entry point 虽然能被发现，但任务不会进入 registry。

### 3.2 config/robot_config

建议只放机器人本体相关配置：

- MuJoCo XML/MJCF 和 mesh 路径；
- joint、body、actuator 的名称或正则表达式；
- position/velocity actuator 参数；
- 初始位置、初始关节角和 reset 状态；
- 碰撞、摩擦和关节限制；
- action scale、关节顺序和控制频率。

环境配置通过 get_go2_cfg() 等工厂函数使用它，不要把 XML 解析和奖励逻辑混在一起。

### 3.3 config/train_config

环境配置负责组装 ManagerBasedRlEnvCfg：

~~~text
robot entity
  + scene / terrain
  + sensors
  + actions
  + commands
  + observations
  + events / domain randomization
  + rewards
  + terminations
  + curriculum / metrics
  + timestep / decimation / episode length
~~~

RL 配置负责组装 RslRlOnPolicyRunnerCfg，包括 actor/critic、observation normalization、PPO 超参数、rollout、保存频率、logger、checkpoint 和 resume 设置。

### 3.4 mdp

这里放 Nazarite 专属的 MDP 逻辑，例如：

- observations.py：机器人观测；
- rewards.py：速度跟踪、姿态、轮腿协同和能耗奖励；
- commands.py：速度或航向指令；
- events.py：质量、摩擦、推力和执行器随机化；
- terminations.py：摔倒、碰撞和越界终止；
- curriculums.py：地形或任务难度递进；
- actions.py：动作延迟、滤波或自定义动作处理。

这些函数由环境配置通过 ObservationTermCfg、RewardTermCfg、EventTermCfg 等配置项引用，再由基础库 manager 在环境 reset/step 时调用。

## 4. 训练调用链

当前基础库环境位于 mjlab/.venv，基础库检查从 Train/Nazarite/mjlab 目录执行：

~~~bash
cd Train/Nazarite/mjlab
uv sync
uv run pyright -p pyproject.toml src/mjlab
~~~

完成 Nazarite 自定义包后，集成项目命令从 Train/Nazarite 目录执行：

~~~bash
cd Train/Nazarite
uv sync
uv run list-envs
uv run train <task-id>
uv run play <task-id>
~~~

注意：在 Train/Nazarite 执行 uv sync 会按照外层 pyproject.toml 管理集成项目环境；当前已经验证通过的开发环境仍是 mjlab/.venv。VS Code 的工作区配置已明确指向后者。

运行时调用链：

~~~text
uv run train <task-id>
        │
        ▼
mjlab.scripts.train.main()
        │
        ├── mjlab 读取 mjlab.tasks entry points
        ├── 导入 nazarite
        └── nazarite 注册任务
        ▼
registry.load_env_cfg(task_id)
registry.load_rl_cfg(task_id)
        │
        ├── train_config 组装 ManagerBasedRlEnvCfg
        │      ├── 引用 robot_config
        │      ├── 引用 Nazarite mdp
        │      └── 引用 mjlab 的 manager/sensor/terrain
        │
        └── train_config 组装 RslRlOnPolicyRunnerCfg
        ▼
ManagerBasedRlEnv
        ▼
RslRlVecEnvWrapper
        ▼
RSL-RL PPO runner
        ▼
checkpoint、TensorBoard/W&B 日志
~~~

## 5. 当前状态

| 项目 | 当前状态 | 说明 |
| --- | --- | --- |
| mjlab 基础库 | 已存在 | 包含源码、RSL-RL 和自身 uv.lock |
| mjlab/.venv | 已存在 | 已安装 mujoco、torch、tensordict、RSL-RL、pyright 等依赖 |
| Train/Nazarite/pyproject.toml | 已存在 | 声明本地 mjlab、CUDA 依赖和 entry point |
| Nazarite-src/nazarite/__init__.py | 已存在 | 有配置导入，但尚未完成任务注册 |
| config/robot_config/ | 目录已建立 | 尚无机器人配置文件和 XML/MJCF 资源 |
| config/train_config/ | 目录已建立 | 尚无环境配置和 RL 配置文件 |
| mdp/ | 目录已建立 | 尚无自定义 MDP 文件 |
| 项目根 uv.lock | 尚未发现 | 执行 uv sync 后生成，建议提交 |
| 工作区 Pyright 配置 | 已存在 | 根目录 pyrightconfig.json 和 .vscode/settings.json 已指向 mjlab/.venv |
| 可发现 task | 尚未完成 | 需要补齐配置和 register_mjlab_task 调用 |

当前项目已经完成依赖、虚拟环境和包布局的骨架；mjlab 基础库已经可以通过 import/type 检查，但 Nazarite 自定义任务层还没有达到可训练状态。

## 6. 推荐实现顺序

1. 统一 __init__.py 与 train_config 的导入路径。
2. 完成 robot_config，先让机器人 XML、实体和执行器能够单独加载。
3. 完成一个最小 flat 环境配置。
4. 完成最小 rl_cfg，并在 __init__.py 注册 Nazarite-Flat-v0。
5. 执行 uv sync 和 uv run list-envs。
6. 进行少量 iteration 的训练和 checkpoint play 测试。
7. 再逐步加入自定义奖励、噪声、域随机化、rough terrain 和 curriculum。

每次新增 observation 或 action 后，都应固定并检查维度、顺序、scale、history 和控制频率，保证后续 sim2sim 与真机接口一致。

## 7. Import 解析检查

当前已经验证以下依赖能够被 mjlab/.venv 解析：

- torch 和 torch.nn；
- tensordict；
- rsl_rl.modules；
- rsl_rl.models.cnn_model；
- rsl_rl.models.mlp_model；
- mjlab 内部模块。

对 mjlab/src/mjlab 的 188 个源码文件运行 Pyright，结果为 0 errors、0 warnings、0 条 import diagnostics：

~~~bash
cd Train/Nazarite/mjlab
uv run pyright -p pyproject.toml src/mjlab
~~~

如果 VS Code 仍显示“无法解析导入”，请确认打开的是仓库根目录 Nazarite-mjlab，并选择：

~~~text
Train/Nazarite/mjlab/.venv/bin/python
~~~

随后执行 Python: Restart Language Server 或 Developer: Reload Window。
