# 当前未提交改动总结

> 盘点对象：`/home/haozi/桌面/mjlab`。
>
> 本文记录生成前工作区中的全部未提交改动。生成本文后，本文自身也会作为一个新的未提交文件出现。

## 1. 工作区概况

生成本文前的工作区状态：

- 25 个已跟踪文件被修改；
- 1 个未跟踪文件：`.codex/hooks.json`；
- Git diff 统计：约 834 行新增、251 行删除；
- 没有创建 commit，也没有回退或覆盖其他已有修改。

改动主要围绕 Go2 velocity 任务模板、奖励、域随机化、历史观测、速度课程和动作延迟展开，同时包含少量格式化整理。

## 2. 通用速度任务模板

文件：

- `/home/haozi/桌面/mjlab/src/mjlab/tasks/velocity/velocity_env_cfg.py`

`make_velocity_env_cfg()` 现在支持 `asymmetric: bool = True`，并统一提供：

- terrain raycast 与脚部高度传感器；
- Actor/Critic 观测组；
- Critic 特权观测：基座线速度、脚部高度、腾空时间、接触状态和接触力；
- Actor 10 帧历史、Critic 3 帧历史；
- time-major 历史展开；
- IMU 角速度、关节位置和关节速度的 1–3 步随机观测延迟；
- 摩擦、编码器偏置、质心、质量/惯量、关节阻尼、PD 增益、力矩限制和随机推力等通用域随机化；
- 通用奖励注册：`action_rate_l2`、`joint_acc_l2`、`torques_l2`、`dof_vel`、`stance_contact` 和 `prolonged_air_time`；
- 通用 commands、events、rewards、terminations 和仿真骨架。

G1 和 Go1 rough 配置改为调用 `make_velocity_env_cfg(asymmetric=False)`，使 Actor 也能看到 terrain height scan。

## 3. Go2 velocity 配置

文件：

- `/home/haozi/桌面/mjlab/src/mjlab/tasks/velocity/config/go2/env_cfgs.py`
- `/home/haozi/桌面/mjlab/src/mjlab/tasks/velocity/config/go2/__init__.py`
- `/home/haozi/桌面/mjlab/src/mjlab/tasks/velocity/config/go2/rl_cfg.py`

### 3.1 动作延迟

Go2 velocity 的 actuator 配置被复制后设置为：

```text
delay_min_lag = 1
delay_max_lag = 3
delay_resample_on_reset = True
```

rough、flat、stand 和 stairs 都会继承该设置：每个环境 reset 时独立采样 1–3 个物理步，并在当前 episode 内保持不变。延迟作用于策略输出到 actuator 实际控制命令之间。

### 3.2 Rough terrain

rough 配置增加或调整了：

- Go2 脚部 slide、spin、roll 三轴摩擦随机化；
- `base_link` 质心随机化；
- 仅对具有有效正定 pseudo-inertia 的 Go2 body 做质量/惯量随机化，跳过零质量空壳 body 和 foot 惯量矩阵不满足正定条件的 body；
- 基座 `-1~+2 kg` payload 随机化；
- 脚部、躯干、大腿和小腿接触传感器；
- terrain raycast 绑定 `base_link`，脚部高度传感器绑定脚底 sites；
- 粗糙地形碰撞、非法接触、姿态、脚部滑移和腾空相关配置；
- 粗糙地形下的平顺性惩罚：

```text
action_rate_l2 = -0.01
joint_acc_l2   = -0.0001
torques_l2     = -0.0002
dof_vel        = -0.001
```

### 3.3 Flat terrain

flat 配置继续继承 rough 基础配置，然后：

- 切换为 plane terrain；
- 保留 `terrain_scan`，供 height scan 和 upright 使用；
- 移除不适合平地的细分碰撞传感器和碰撞奖励；
- 保留 `illegal_contact`，使用 70 度姿态终止；
- 增加站立命令比例；
- 训练命令范围调整为 `lin_vel_x=(-1.0, 4.0)`、`lin_vel_y=(-2.0, 2.0)`、`ang_vel_z=(-1.0, 1.0)`；
- 使用逐环境、逐速度维度的自适应速度课程。

### 3.4 Stand、stairs 和 RL 配置

- stand 调整站立姿态容差、关节限位惩罚、平顺性惩罚和推力事件；
- stairs 基于 rough 配置切换到 stairs terrain，保留地形课程，关闭速度课程，并重新配置楼梯速度范围、奖励和 play 模式；
- `/home/haozi/桌面/mjlab/src/mjlab/tasks/velocity/config/go2/rl_cfg.py` 主要是格式化整理；当前 flat Go2 使用 `MLPModel`，RNN 配置仍保留给显式选择 RNN 的任务。

## 4. 奖励与奖励管理器

文件：

- `/home/haozi/桌面/mjlab/src/mjlab/tasks/velocity/mdp/rewards.py`
- `/home/haozi/桌面/mjlab/src/mjlab/managers/reward_manager.py`

改动包括：

- `feet_air_time` 改为只在脚落地时，根据上一段完整腾空时间计算奖励；
- 新增 `prolonged_air_time`，惩罚脚超过最大腾空时间；
- 新增 `feet_stance_contact`，站立时惩罚缺少有效接触力的脚；
- `feet_clearance` 只惩罚脚离地过低造成的拖脚，不再惩罚抬得过高；
- `RewardManager` 新增按 episode 查询累计奖励的 `get_episode_sum()`。

## 5. 自适应速度课程

文件：

- `/home/haozi/桌面/mjlab/src/mjlab/tasks/velocity/mdp/curriculums.py`
- `/home/haozi/桌面/mjlab/src/mjlab/tasks/velocity/mdp/velocity_command.py`

新增 `AdaptiveVelocityCommand`：

- 对前进、侧向和 yaw 分别维护课程上限；
- 使用各维度独立 tracking error；
- 使用 EMA 平滑 tracking quality；
- 根据 episode 存活率和跟踪质量升降课程；
- warmup 阶段禁止降级；
- 将课程结果写回 command term 的每环境速度上限。

`UniformVelocityCommand` 增加逐轴误差统计，以及 `lin_vel_x_max`、`lin_vel_y_max`、`ang_vel_z_max` 等每环境上限，并按这些上限重新采样命令。

## 6. 历史观测和导出

文件：

- `/home/haozi/桌面/mjlab/src/mjlab/managers/observation_manager.py`
- `/home/haozi/桌面/mjlab/src/mjlab/rl/exporter_utils.py`
- `/home/haozi/桌面/mjlab/docs/source/observations.rst`

观测管理器新增：

```python
history_ordering="term" | "time"
```

- `term`：同一观测项的历史帧连续排列；
- `time`：每一帧先排列所有观测项，例如 `[A_t0, B_t0, A_t1, B_t1]`。

Go2 velocity 使用 time-major 方式，以便 MLP 直接消费固定长度的时间堆叠向量。ONNX 导出元数据新增 `observation_history_ordering`。

## 7. 执行器延迟底层支持

文件：

- `/home/haozi/桌面/mjlab/src/mjlab/utils/buffers/delay_buffer.py`
- `/home/haozi/桌面/mjlab/src/mjlab/actuator/actuator.py`
- `/home/haozi/桌面/mjlab/src/mjlab/actuator/builtin_group.py`
- `/home/haozi/桌面/mjlab/src/mjlab/actuator/fused_group.py`

新增 `delay_resample_on_reset` / `resample_on_reset`：

- 默认 `False`，保持原有动态重采样行为；
- 开启后 reset 时采样 lag，episode 内不再自动改变；
- 支持全量 reset 和部分环境 reset；
- 延迟配置加入 builtin/fused actuator 分组 key，避免不同延迟语义共享 buffer；
- builtin、fused ideal-PD/DC motor 和普通 actuator 都能传递该配置。

当前 mjlab Go2 物理步长是 `0.005 s`，因此 1–3 个物理步对应约 `5–15 ms`。参考文档中“1000 Hz / 1 ms”的描述与其 `0.005 s` 配置矛盾，本次没有把整个仿真频率改为 1 kHz。

## 8. 测试改动

文件：

- `/home/haozi/桌面/mjlab/tests/test_delay_buffer.py`
- `/home/haozi/桌面/mjlab/tests/test_observation_history.py`
- `/home/haozi/桌面/mjlab/tests/test_velocity_task.py`

新增覆盖：

- time-major 历史排列；
- Go2 Actor/Critic 历史长度、奖励注册和动作延迟配置；
- DelayBuffer reset 重新采样；
- 部分环境 reset 只影响对应环境；
- episode 内 lag 保持不变。

## 9. Go2 资产和其他整理

文件 `/home/haozi/桌面/mjlab/src/mjlab/asset_zoo/robots/unitree_go2/go2_constants.py`：

- 调整 Go2 前后腿初始姿态匹配；
- 调整左右髋关节初始角度符号；
- 其余为注释和格式整理。

以下文件主要是格式化、导入顺序或注释整理，未引入明显新功能：

- `/home/haozi/桌面/mjlab/src/mjlab/asset_zoo/robots/__init__.py`；
- `/home/haozi/桌面/mjlab/src/mjlab/scripts/train.py`；
- `/home/haozi/桌面/mjlab/src/mjlab/tasks/velocity/config/go2/__init__.py`；
- `/home/haozi/桌面/mjlab/src/mjlab/tasks/velocity/config/go2/rl_cfg.py`；
- `/home/haozi/桌面/mjlab/third_party/rsl_rl/rsl_rl/storage/rollout_storage.py`。

注意：Go2 environment 配置文件同时包含功能性修改和大量格式/注释整理，不能仅按 diff 行数判断功能改动规模。

## 10. 文档与变更记录

文件 `/home/haozi/桌面/mjlab/docs/source/changelog.rst` 的 Upcoming version 新增了：

- Go2 velocity 域随机化；
- time-major 历史观测；
- Go2 动作延迟；
- 站立接触奖励；
- 通用速度任务平顺性奖励注册；
- actuator delay fusion 行为说明。

## 11. 未跟踪文件

文件：`/home/haozi/桌面/mjlab/.codex/hooks.json`

该文件配置了 `PostToolUse` hook，在文件写入或编辑后自动执行 `uv run ruff format`。

## 12. 验证结果

已执行：

- `make check`：通过；
- ruff format / ruff check：通过；
- ty：通过；
- pyright：0 errors，只有项目原有的 3 个 wildcard import warning；
- 相关延迟、执行器分组、速度任务测试：`55 passed`；
- 完整测试：`1149 passed, 13 skipped`。

完整测试中仍有 2 个失败：

- `third_party/rsl_rl/tests/models/test_rnn_model.py::... [gru]`；
- `third_party/rsl_rl/tests/models/test_rnn_model.py::... [lstm]`。

失败发生在 PyTorch 2.9 新 ONNX exporter 对 RNN hidden state 的 `aten::unbind` 转换处，属于现有 RNN ONNX 导出兼容问题。Go2 flat 使用 MLP，不涉及该失败路径。

Go2 flat play 环境已在 CPU 上实际创建并 reset 成功，actuator lag 位于配置的 1–3 范围内。

## 13. 相关文件总表

### 已跟踪修改文件

```text
/home/haozi/桌面/mjlab/docs/source/changelog.rst
/home/haozi/桌面/mjlab/docs/source/observations.rst
/home/haozi/桌面/mjlab/src/mjlab/actuator/actuator.py
/home/haozi/桌面/mjlab/src/mjlab/actuator/builtin_group.py
/home/haozi/桌面/mjlab/src/mjlab/actuator/fused_group.py
/home/haozi/桌面/mjlab/src/mjlab/asset_zoo/robots/__init__.py
/home/haozi/桌面/mjlab/src/mjlab/asset_zoo/robots/unitree_go2/go2_constants.py
/home/haozi/桌面/mjlab/src/mjlab/managers/observation_manager.py
/home/haozi/桌面/mjlab/src/mjlab/managers/reward_manager.py
/home/haozi/桌面/mjlab/src/mjlab/rl/exporter_utils.py
/home/haozi/桌面/mjlab/src/mjlab/scripts/train.py
/home/haozi/桌面/mjlab/src/mjlab/tasks/velocity/config/g1/env_cfgs.py
/home/haozi/桌面/mjlab/src/mjlab/tasks/velocity/config/go1/env_cfgs.py
/home/haozi/桌面/mjlab/src/mjlab/tasks/velocity/config/go2/__init__.py
/home/haozi/桌面/mjlab/src/mjlab/tasks/velocity/config/go2/env_cfgs.py
/home/haozi/桌面/mjlab/src/mjlab/tasks/velocity/config/go2/rl_cfg.py
/home/haozi/桌面/mjlab/src/mjlab/tasks/velocity/mdp/curriculums.py
/home/haozi/桌面/mjlab/src/mjlab/tasks/velocity/mdp/rewards.py
/home/haozi/桌面/mjlab/src/mjlab/tasks/velocity/mdp/velocity_command.py
/home/haozi/桌面/mjlab/src/mjlab/tasks/velocity/velocity_env_cfg.py
/home/haozi/桌面/mjlab/src/mjlab/utils/buffers/delay_buffer.py
/home/haozi/桌面/mjlab/tests/test_delay_buffer.py
/home/haozi/桌面/mjlab/tests/test_observation_history.py
/home/haozi/桌面/mjlab/tests/test_velocity_task.py
/home/haozi/桌面/mjlab/third_party/rsl_rl/rsl_rl/storage/rollout_storage.py
```

### 未跟踪文件

```text
/home/haozi/桌面/mjlab/.codex/hooks.json
```
