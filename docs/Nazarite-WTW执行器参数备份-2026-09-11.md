# Nazarite WTW 执行器参数备份

备份时间：2026-09-11

本文件记录修改前 `Nazarite-mjlab` 当前 WTW 使用的执行器参数。

来源：

- `Train/Nazarite/Nazarite-src/nazarite/config/robot_config/go2_cfg.py`
- `Train/Nazarite/sim2sim/config.py`
- `Train/Nazarite/sim2sim/mujoco_io.py`

## 训练配置

当前使用 MuJoCo 内置位置执行器（`BuiltinPositionActuatorCfg`）。训练执行器的实际增益为名义增益乘以 2：

| 关节组 | 名义 Kp | 名义 Kd | Kp 倍率 | Kd 倍率 | 实际 Kp | 实际 Kd | torque limit | armature |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hip / thigh | 15.8952426532 | 1.0119225760 | 2.0 | 2.0 | 31.7904853065 | 2.0238451520 | 23.7 | 0.004026312 |
| calf | 35.7642959698 | 2.2768257960 | 2.0 | 2.0 | 71.5285919396 | 4.5536515920 | 35.55 | 0.009059202 |

Action scale 仍基于未乘 2 的名义 Kp 计算：

```text
hip/thigh = 0.25 * 23.7 / 15.8952426532
calf      = 0.25 * 35.55 / 35.7642959698
```

因此 action 到目标关节位置的映射没有随增益倍率改变。

## sim2sim 配置

sim2sim 使用与训练相同的实际增益：

| 关节组 | Kp | Kd | torque limit |
|---|---:|---:|---:|
| hip / thigh | 31.7904853065 | 2.0238451520 | ±23.7 |
| calf | 71.5285919396 | 4.5536515920 | ±35.55 |

仿真步长为 `0.002 s`，decimation 为 `10`，策略控制周期为 `0.02 s`（50 Hz）。

