# XZ1 HIM / Robocon2027

视觉学生 `stage4_nine_v1` 的完整 HIM + WMP + WM 推理入口见
[XZ1 WMP sim2sim](XZ1_WMP_ROBOCON.md)。下文是基础 HIM 策略入口。

从仓库根目录启动（先连接手柄或无线接收器）：

```bash
./Train/Nazarite/run_xz1_robocon.sh
```

脚本默认加载训练目录 `xz1_him_complex/2026-10-25_22-04-31` 中的
`model_31300.pt`。固定 checkpoint 避免仍在训练时 `policy.onnx` 被自动覆盖。
场景是桌面 `Robocon2027.xml` 的原样副本，位于 `sim2sim/assets/`，不依赖桌面文件。
启动区 1 的出生位置为 `(0, 0, 0.30)`，朝向 +X。

- 左摇杆：前后、横移；右摇杆 X：转向。
- Start：复位机器人和策略历史；键盘 R 也可复位。
- 关闭 MuJoCo 窗口：退出。
- 接收器断开后，沿用现有 evdev 控制器的零速度命令处理。
- 指令范围默认各为 ±1（平移 m/s，转向 rad/s）；可通过
  `--max-vx 0.5 --max-vy 0.5 --max-yaw 0.5` 调低。

列出手柄或指定输入设备：

```bash
./Train/Nazarite/run_xz1_robocon.sh --list-gamepads
./Train/Nazarite/run_xz1_robocon.sh --gamepad-device /dev/input/event21
```

如提示未检测到手柄，检查连接以及当前用户对 `/dev/input/event*` 的读取权限。
轴名可用 `--axis-vx ABS_Y --axis-vy ABS_X --axis-yaw ABS_RX` 覆盖。

使用其他模型、原始桌面场景，或者启动区 2：

```bash
./Train/Nazarite/run_xz1_robocon.sh \
  --policy /absolute/path/to/model.pt
./Train/Nazarite/run_xz1_robocon.sh \
  --scene /home/haozi/桌面/Robocon2027.xml
./Train/Nazarite/run_xz1_robocon.sh --spawn -6.2 -2 0.66 --spawn-yaw 0
```

`--policy` 也接受同一训练任务导出的 `policy.onnx`，会校验观测和关节顺序。
启动脚本使用现有的 `/home/haozi/miniforge3/envs/mjlab_env/bin/uv`，可通过
`UV_BIN` 环境变量改为其他 uv 路径；不会自动安装或同步依赖。

无需手柄的离屏验证（在 `Train/Nazarite` 下执行）：

```bash
/home/haozi/miniforge3/envs/mjlab_env/bin/uv run --no-sync --project mjlab \
  python -m sim2sim.xz1_him --headless --duration 10 --vx 0.3
```

部署接口：47 维单帧 × 6 帧，当前帧在前；关节顺序 FL/FR/RR/RL，每腿
hip/thigh/shank。初始姿态与关节位置观测基准为 `0 / 0.83 / -1.7404`。
动作目标使用训练保存的承重补偿偏置（FL/FR/RR/RL）：

| 关节 | FL | FR | RR | RL |
| --- | --- | --- | --- | --- |
| 髋 | 0 | 0 | 0 | 0 |
| 大腿 | 0.8483 | 0.8481 | 0.8478 | 0.8480 |
| 小腿 | -1.5640 | -1.5624 | -1.5427 | -1.5450 |

目标关节角 = 上表偏置 + 0.25 × action，单位 rad。补偿仅用于执行器目标，
不能代替观测基准；遗漏补偿会使该模型的静止站高从约 0.305 m 降到约 0.239 m。
复位时关节位置仍为初始姿态，执行器目标使用补偿偏置。保留 checkpoint
中的观测归一化。髋/大腿 kp=32、kd=1.9，小腿 kp=35、kd=1.8，力矩限制
31.7 Nm，armature=0.025392063。物理步长 2 ms，控制周期 20 ms，
implicitfast 求解器。推理使用确定性观测和执行器，不注入训练时的随机延迟与噪声。

已验证 checkpoint 与该次 ONNX 导出推理一致、10 秒站立与前进，以及模拟手柄指令
和 Start 复位。完整赛道越障能力仍需实际操作评估；实现手柄输入不代表策略已经
能通过所有障碍。
