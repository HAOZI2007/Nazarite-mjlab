# XZ1 特权地形教师 / Robocon2027 sim2sim

默认加载本次 stage2 训练的最后一个 checkpoint：

```text
logs/rsl_rl/xz1_wmp_him_teacher/2026-10-27_17-48-42_stage2_fixed_curriculum/model_2499.pt
```

使用与基础 HIM、视觉学生相同的 `sim2sim/assets/Robocon2027.xml` 场地、
XZ1 机器人和执行器参数，便于对比各阶段策略。启动时打印完整模型和场景路径。

## 手柄运行

在仓库根目录执行：

```bash
cd /home/haozi/桌面/Nazarite-mjlab
./Train/Nazarite/run_xz1_teacher_robocon.sh
```

左摇杆控制前后、横移，右摇杆 X 控制转向；Start 或键盘 R 复位机器人、
步态相位、观测历史和上一动作。没有摔倒自动复位。关闭主窗口退出。
默认出生点 `(0, 0, 0.30)`，朝向 +X；指令范围各 ±1，单位为 m/s、rad/s。

```bash
./Train/Nazarite/run_xz1_teacher_robocon.sh --max-vx 0.5 --max-vy 0.3 --max-yaw 0.5
./Train/Nazarite/run_xz1_teacher_robocon.sh --list-gamepads
./Train/Nazarite/run_xz1_teacher_robocon.sh --gamepad-device /dev/input/event21
./Train/Nazarite/run_xz1_teacher_robocon.sh --spawn -6.2 -2 0.66 --spawn-yaw 0
```

轴映射可用 `--axis-vx ABS_Y --axis-vy ABS_X --axis-yaw ABS_RX` 覆盖。
脚本使用现有 uv 环境，不自动同步依赖；`UV_BIN` 可覆盖 uv 路径。

切换这次训练的其他 checkpoint：

```bash
./Train/Nazarite/run_xz1_teacher_robocon.sh \
  --policy /home/haozi/桌面/Nazarite-mjlab/Train/Nazarite/logs/rsl_rl/xz1_wmp_him_teacher/2026-10-27_17-48-42_stage2_fixed_curriculum/model_1500.pt
```

也可传入其他采用相同 XZ1 教师观测契约的 `.pt` checkpoint。

## 固定指令和记录

无需手柄，在 `Train/Nazarite` 下直接运行 Python 入口：

```bash
cd /home/haozi/桌面/Nazarite-mjlab/Train/Nazarite
/home/haozi/miniforge3/envs/mjlab_env/bin/uv run --no-sync \
  python -m sim2sim.xz1_teacher --vx 0.3
```

无窗口运行并保存轨迹：

```bash
/home/haozi/miniforge3/envs/mjlab_env/bin/uv run --no-sync \
  python -m sim2sim.xz1_teacher --headless --duration 10 --vx 0.3 \
  --record logs/evaluations/xz1_stage2_teacher_my_test.json
```

目标 JSON 必须不存在。记录每个控制步的指令、动作、输入高度扫描及执行后的
根状态。默认 CPU 两线程，可用 `--device cuda:0`、`--threads` 调整。
`--no-camera-follow` 关闭观看相机跟随。

## 输入与控制对齐

该模型是 **HIM + 特权地形扫描教师**，没有深度相机或 World Model 输入。
因此本入口没有学生的深度预览窗口。

- 本体观测为 47×6；保留训练 actor 的归一化、HIM 网络和地形编码器。
  sim2sim 当前帧在前的历史会转换为原生训练 actor 所需的排列。
- 地形为 12×11 个点，x 范围 -0.45～1.20 m、y 范围 -0.75～0.75 m，
  x 优先展开。网格跟随机器人航向，忽略机身俯仰和横滚。
- 从机身高度上方 1 m 向下发射射线，最大距离 5 m，只包含 group 0 地形，
  排除相机/扫描附着机身。输入为 `clip(base_z - 0.3 - terrain_z, -1, 1)`，
  缺测填 1。每个控制步更新，保持训练的特权扫描语义。
- 控制 50 Hz，物理 500 Hz；关节顺序、PD、动作缩放、承重补偿偏置沿用
  [XZ1 HIM 控制接口](XZ1_ROBOCON.md)。不注入训练扰动和随机延迟。

2026-10-28 已通过四项测试：扫描坐标/方向/分组/缺测、训练 Warp 扫描一致性、
原生 actor 与当前帧优先导出接口的动作一致性、模拟手柄 Start 复位。
另已运行 GUI 3 秒、站立 10 秒、0.3 m/s 前进 10 秒；前进末端位置约
`(3.420, 0.051, 0.322)` m。这些是接口与基础运行验证，完整障碍通过率需另测。

此入口用于教师能力对照，使用仿真真值高度；视觉学生入口见
[XZ1 WMP sim2sim](XZ1_WMP_ROBOCON.md)。
