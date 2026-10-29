# XZ1 HIM + WMP 视觉学生 / Robocon2027 sim2sim

默认策略是 `logs/rsl_rl/xz1_wmp_student/stage4_nine_v1/student_3000.pt`，
场地复用 `sim2sim/assets/Robocon2027.xml`。该入口直接运行原生 CPU MuJoCo，
不创建训练环境，不加载教师或特权地形扫描。

## 启动

从仓库根目录连接手柄后执行：

```bash
./Train/Nazarite/run_xz1_wmp_robocon.sh
```

左摇杆控制前进和横移，右摇杆 X 控制转向；Start 或键盘 R 复位机器人、
步态相位、六帧观测、实际动作队列和 WM 隐状态。关闭窗口退出。
不会因摔倒自动复位。出生点默认 `(0, 0, 0.30)`，朝向 +X。

默认同时打开独立的深度相机窗口，显示策略实际采集的 64×64 米制深度图，
以 10 Hz 与 WM 同步更新。采用最近邻放大，近处为白色，远处或缺测为黑色，
右侧色条单位为米，固定范围 0–2 m。这是同一张网络预处理前的深度图，
与 checkpoint 的安装姿态、视场角和截断距离匹配；显示窗口不会额外采图。

深度窗口中也可按 R 复位；Esc 或关闭该窗口只关闭预览，仿真继续运行。
关闭主 MuJoCo 窗口则退出并关闭预览。使用以下参数关闭预览；`--headless`
也会自动禁用所有窗口：

```bash
./Train/Nazarite/run_xz1_wmp_robocon.sh --no-depth-window
```

限制手柄速度、列出设备或指定设备：

```bash
./Train/Nazarite/run_xz1_wmp_robocon.sh --max-vx 0.5 --max-vy 0.3 --max-yaw 0.5
./Train/Nazarite/run_xz1_wmp_robocon.sh --list-gamepads
./Train/Nazarite/run_xz1_wmp_robocon.sh --gamepad-device /dev/input/event21
```

指令默认限幅为各 ±1，平移单位 m/s，转向单位 rad/s。
轴名可用 `--axis-vx ABS_Y --axis-vy ABS_X --axis-yaw ABS_RX` 指定。
脚本使用现有 uv 环境，不同步或安装依赖；可用 `UV_BIN` 覆盖 uv 路径。

切换同一 run 的 900 轮学生，或更换出生点：

```bash
./Train/Nazarite/run_xz1_wmp_robocon.sh \
  --policy /home/haozi/桌面/Nazarite-mjlab/Train/Nazarite/logs/rsl_rl/xz1_wmp_student/stage4_nine_v1/student_900.pt
./Train/Nazarite/run_xz1_wmp_robocon.sh --spawn -6.2 -2 0.66 --spawn-yaw 0
```

不接手柄时直接使用 Python 入口。以下命令在 `Train/Nazarite` 下执行：

```bash
cd /home/haozi/桌面/Nazarite-mjlab/Train/Nazarite
/home/haozi/miniforge3/envs/mjlab_env/bin/uv run --no-sync \
  python -m sim2sim.xz1_wmp --vx 0.3
```

无窗口运行及记录（记录文件必须尚不存在）：

```bash
/home/haozi/miniforge3/envs/mjlab_env/bin/uv run --no-sync \
  python -m sim2sim.xz1_wmp --headless --duration 10 --vx 0.3 \
  --record logs/evaluations/xz1_wmp_sim2sim_my_test.json
```

JSON 保存每个控制步的指令、根状态、动作、WM 更新标记及耗时汇总；同名
`.depth.npy` 保存最后一张 64×64 原始米制深度图。`--device cuda:0` 可改为
GPU 推理，默认 CPU、两线程；`--no-camera-follow` 关闭观看相机跟随。

## 学生和 WM 打包

已生成包含 actor、HIM 估计器、WM 权重及输入配置的推理包：

```text
logs/deploy/xz1_wmp_stage4_nine_v1/student_3000_deploy.pt
```

运行此包不再读取原始 student、WM checkpoint、教师或数据集，但仍需要本项目的
Python 推理代码和依赖。它是 PyTorch 推理包，不是单独的 ONNX actor。

```bash
./Train/Nazarite/run_xz1_wmp_robocon.sh \
  --policy /home/haozi/桌面/Nazarite-mjlab/Train/Nazarite/logs/deploy/xz1_wmp_stage4_nine_v1/student_3000_deploy.pt
```

为其他学生生成包（在 `Train/Nazarite` 下执行；目标文件不能已存在）：

```bash
/home/haozi/miniforge3/envs/mjlab_env/bin/uv run --no-sync \
  python -m sim2sim.xz1_wmp \
  --policy logs/rsl_rl/xz1_wmp_student/stage4_nine_v1/student_900.pt \
  --export-bundle logs/deploy/xz1_wmp_stage4_nine_v1/student_900_deploy.pt
```

直接读取原 student 时，会校验其配套 WM 的 SHA256；若移动了 WM 文件，可用
`--world-model /absolute/path/to/wm_1000.pt` 指定相同权重的新位置。
包内没有优化器状态；加载时校验输入、相机、关节控制配置及有限权重。

## 推理与训练对齐

- 物理 500 Hz、actor 50 Hz、WM 10 Hz。每五个控制步采集一次实际深度；
  WM 输入包含之前五个已执行动作，复位后立即重新采图。
- 本体观测每帧 47 维、共六帧。沿用 HIM 观测构造，并在送入原生训练 actor
  前转换为其所需的历史排列。保留 checkpoint 的观测归一化和 HIM 网络。
- WM 输入为 64×64 深度、33 维本体状态和 5×12 动作；actor 接收 WM 的
  512 维 `deter` 特征。部署不需要教师或特权扫描。
- 相机配置直接来自学生 checkpoint：安装位置 `(0.2, 0, 0.08)`，
  四元数 wxyz `(0.579228, 0.405580, -0.405580, -0.579228)`，垂直视场角
  58°，截断距离 2 m。深度为相机光轴 Z 距离；采用原生 MuJoCo 像素射线，
  无需 OpenGL 生成网络输入，地形可见组与训练一样为 group 0。
- 缺测和无效深度填充为远端距离，保留训练时的半精度转换、归一化和 8 bit
  量化流程。模型输入处理和 WM 状态更新已与训练 `XZ1VisualPolicy` 对照。
- 执行器参数、关节顺序、承重补偿偏置沿用 [XZ1 控制接口](XZ1_ROBOCON.md)。
  不添加训练中的域随机化、观测噪声或随机延迟。

这里的跟随观看相机不影响网络相机。切换 viewer 视角不会改变策略输入。

## 验证范围与部署选择

2026-10-28 验证了几何深度与训练 Warp 渲染器一致（逐像素容差 2e-5）、
网络推理与训练运行时一致、手柄复位、打包前后输出一致，以及 Robocon 场景内
10 秒站立和 0.3 m/s 前进；12 项测试通过，GUI 已完成 3 秒启动退出检查。
前进末端位置约 `(3.200, 0.235, 0.316)` m；
这表明仍有横向偏移，不代表已经通过整场障碍。

最终实机应部署这一类**视觉学生 + HIM + WM** 推理系统，不需要携带教师。
当前 `student_3000.pt` 是 sim2sim 候选版本；此前测评暴露的沟壑能力和路线保持
问题仍然存在。最终采用哪个 checkpoint，应在后续场地测试和实机输入对齐后确定。
