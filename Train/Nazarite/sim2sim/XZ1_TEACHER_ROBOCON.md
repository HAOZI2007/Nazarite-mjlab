# XZ1 特权地形教师 / Robocon2027 sim2sim

2026-10-31 默认单帧 10 Hz 模型 `model_13100.pt` 已通过三个 CPU 测试，覆盖
扫描几何、采样保持、推理一致性和模拟手柄复位；与保存的 ONNX 连续 21 步
动作一致。站立 15 秒末端站高约 0.317 m；0.3 m/s 前进 10 秒末端位置约
`(3.312, 0.002, 0.362)` m，轨迹检查确认高程在每五控制步之间保持。
完整赛道和本模型的实物手柄操作尚未验证。

默认加载本次单帧、10 Hz 高程训练的固定 checkpoint：

```text
logs/rsl_rl/xz1_wmp_him_teacher/2026-10-30_10-01-18_height_single_10hz_from_him/model_13100.pt
```

使用与基础 HIM、视觉学生相同的 `sim2sim/assets/Robocon2027.xml` 场地、
XZ1 机器人和执行器参数，便于对比各阶段策略。启动时打印完整模型和场景路径。

## 手柄运行

在仓库根目录执行：

```bash
cd /home/haozi/桌面/Nazarite-mjlab
./Train/Nazarite/run_xz1_teacher_robocon.sh
```

如果当前目录已是 `Train/Nazarite`，直接运行 `./run_xz1_teacher_robocon.sh`。

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
  --policy /home/haozi/桌面/Nazarite-mjlab/Train/Nazarite/logs/rsl_rl/xz1_wmp_him_teacher/2026-10-30_10-01-18_height_single_10hz_from_him/model_13000.pt
```

也可传入其他采用相同 XZ1 教师观测契约的 `.pt` checkpoint。
入口根据 checkpoint 自动识别高程历史帧数和采样频率，启动时打印实际配置。
旧单帧/六帧 checkpoint 保持 50 Hz；2026-10-30 新训练入口为单帧、10 Hz，
此前三帧 10 Hz checkpoint 也继续按保存契约加载。
新模型的训练及导出接口见
[高程历史训练说明](../../../docs/XZ1高程历史训练与部署-2026-10-29.md)。

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
根状态。`teacher_scan` 是策略实际使用的最新采样帧，`teacher_scan_history`
是最旧帧在前的完整历史；采样间隔内保持不变。
默认 CPU 两线程，可用 `--device cuda:0`、`--threads` 调整。
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
  缺测填 1。策略按 checkpoint 保存的采样周期接收更新，采样间隔内
  保持整个历史，包含采样时的相对高度。
- 新单帧 10 Hz 模型每五控制步更新一次扫描，共 32 维地形特征，不堆叠历史。
  此前三帧 10 Hz 模型为 96 维地形特征，三次采样跨度 0.2 秒。
  旧六帧 50 Hz 模型仍保留六次扫描，共 192 维地形特征。
  最旧帧到最新帧输入原生 actor，每帧输出 32 维；首次采样填满历史，复位时清空。
  原 stage2 单帧模型仍只使用当前扫描。
- 控制 50 Hz，物理 500 Hz；关节顺序、PD、动作缩放、承重补偿偏置沿用
  [XZ1 HIM 控制接口](XZ1_ROBOCON.md)。不注入训练扰动和随机延迟。

2026-10-28 已通过四项测试：扫描坐标/方向/分组/缺测、训练 Warp 扫描一致性、
原生 actor 与当前帧优先导出接口的动作一致性、模拟手柄 Start 复位。
另已运行 GUI 3 秒、站立 10 秒、0.3 m/s 前进 10 秒；前进末端位置约
`(3.420, 0.051, 0.322)` m。这些是接口与基础运行验证，完整障碍通过率需另测。

此入口使用仿真真值高度；替换雷达高程图时需要匹配坐标、数值和采样时间契约。

2026-10-29 六帧模型 `model_5000.pt` 已验证扫描几何、连续高程历史推理与复位、
模拟手柄 Start 复位，以及与该次保存的 ONNX 连续八步动作一致性。
无窗口站立 15 秒末端站高约 0.320 m；0.3 m/s 前进 10 秒末端位置约
`(3.415, 0.530, 0.337)` m，存在侧向偏移。尚未验证本模型的完整赛道通过率。

视觉学生入口见
[XZ1 WMP sim2sim](XZ1_WMP_ROBOCON.md)。

2026-10-30 三帧 10 Hz 模型 `model_6000.pt` 已通过三个 CPU 测试，覆盖扫描几何、
按五控制步更新并保持历史、推理一致性及模拟手柄复位。另与本次保存的 ONNX
核对了连续 21 个控制步的动作；站立和前进轨迹也检查了实际高程历史保持。
站立 15 秒末端站高约 0.314 m；0.3 m/s 前进 10 秒末端位置约
`(3.518, 0.136, 0.334)` m。完整赛道及实物手柄操作尚未在本模型上验证。
