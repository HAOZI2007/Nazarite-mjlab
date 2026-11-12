# Sim2Real 使用指南（XZ1-HIM 测试版）

这份指南面向刚开始接触 C++ 的使用者。正常使用时只需在终端复制命令；
参数查看、模型冒烟测试、手柄/IMU读取、单电机测试和四路异步电机控制接口都已有独立模块。

当前项目是**测试版组件集合**：已有四路 IM6014 异步传输层，每路管理一条腿的三个电机；
但它尚未接成“读取传感器、运行策略、执行 URDF 角度换算并发出动作”的完整机器人闭环。
请先做离线测试，不能把它当作可直接让机器人行走的程序。

## 1. 目录里有什么

| 位置 | 用途 |
|---|---|
| `Sim2Real/xz1/him/` | XZ1-HIM 观测构造、动作目标换算、ONNX Runtime C++ 推理 |
| `Sim2Real/xz1/imu/` | DM BMI088 串口帧解析和读取程序 |
| `Sim2Real/xz1/motor_control/` | IM6014/N6014 协议、单电机测试和四路异步电机接口 |
| `Sim2Real/xz1/remote_controler/` | Linux evdev 手柄读取和测试程序 |
| `Sim2Real/xz1/tests/` | 不接硬件的 C++ 单元测试 |
| `Sim2Real/sdk/onnxruntime-linux-x64-1.31.1/` | 已下载的 ONNX Runtime C++ SDK |
| `Sim2Real/xz1_him/` | Python 版 ONNX 推理耗时/CPU 测试脚本 |
| `Sim2Real/output/` | 之前保存的本机推理、CPU 和内存测试结果 |

关节索引固定为 `FL → FR → RR → RL`，每条腿依次是 `hip → thigh → shank`：

| action 索引 | 关节 |
|---:|---|
| 0–2 | FL 髋、前腿大腿、前腿小腿 |
| 3–5 | FR 髋、前腿大腿、前腿小腿 |
| 6–8 | RR 髋、后腿大腿、后腿小腿 |
| 9–11 | RL 髋、后腿大腿、后腿小腿 |

控制数组仍按 action 索引排列：0–2 为 FL、3–5 为 FR、6–8 为 RR、9–11 为 RL，每条腿
依次是髋、大腿、小腿。四路 USB-485 模块每一路对应一条腿。按你的规划，电机 ID 默认与
action 索引相同，使用全局 ID 0–11；若实际电机仍配置成每路重复 ID 0–2，可在探测程序
中指定 `per-leg`。实际电机 ID、方向、编码器零位和安装方向都要在实机核对/标定。

## 2. 第一次编译

打开终端，在仓库根目录执行：

```bash
cd /home/haozi/桌面/Nazarite-mjlab
cmake -S Sim2Real/xz1 -B Sim2Real/xz1/build
cmake --build Sim2Real/xz1/build -j2
ctest --test-dir Sim2Real/xz1/build --output-on-failure
```

仓库中已经有 C++ SDK 时，CMake 会自动查找它。配置完成时如果看到
`ONNX Runtime C++ SDK not found`，说明没有发现 SDK；可以明确指定路径重新配置：

```bash
cmake -S Sim2Real/xz1 -B Sim2Real/xz1/build \
  -DONNXRUNTIME_ROOT="$PWD/Sim2Real/sdk/onnxruntime-linux-x64-1.31.1"
cmake --build Sim2Real/xz1/build -j2
```

成功后，可执行程序都在 `Sim2Real/xz1/build/`。`ctest` 全绿表示离线协议和 HIM
测试通过，不代表传感器、电机或机器人已经通过实机测试。

## 3. 不接硬件先检查 ONNX 和参数

先查看当前编译进去的 HIM 参数：

```bash
Sim2Real/xz1/build/xz1_him_config
```

它会输出控制周期、步态相位周期，以及每个关节的默认姿态、动作偏置、动作比例、
PD 增益和仿真关节范围。前几列参数含义如下：

- `default_q_rad`：观测中计算关节位置偏差的参考姿态。
- `action_offset_rad`：策略动作变成目标角度前添加的偏置。
- `action_scale`：策略动作乘的比例；当前目标公式为
  `target_rad = action_offset_rad + action_scale * action`。
- `kp`、`kd`：随目标提供的比例/微分增益。
- `lower_rad`、`upper_rad`：当前配置使用的仿真关节范围，不等同于实机安全范围。

全零输入 ONNX 冒烟测试并测 50 Hz 推理：

```bash
Sim2Real/xz1/build/xz1_onnx_test \
  Train/Nazarite/logs/rsl_rl/xz1_him_complex/2026-11-01_21-00-14/policy.onnx \
  100 1500 50
```

参数依次是 ONNX 文件、预热次数、计时推理次数和目标频率；`1500 50` 约运行 30 秒。
结果会显示模型加载/首次推理耗时、平均/P50/P95/P99/最大推理耗时、实际频率、超过
20 ms 的次数及进程 CPU 占用。频率填 `0` 会尽可能连续推理，例如：

```bash
Sim2Real/xz1/build/xz1_onnx_test \
  Train/Nazarite/logs/rsl_rl/xz1_him_complex/2026-11-01_21-00-14/policy.onnx \
  100 1000 0
```

这个程序输入的是 282 个零，连重力项也为零，因此只用于测试 ONNX Runtime 推理性能；
输出动作不能用于控制机器人。CPU 数值包含进程推理开销，不包含 IMU、手柄、观测构造、
电机通信或完整控制循环。

## 4. 怎样修改参数

XZ1-HIM 部署参数集中在
[`deployment_config.hpp`](xz1/him/deployment_config.hpp)：默认姿态、动作偏置/比例、
PD 增益、关节范围、控制周期和相位周期都在这里。编辑前先备份文件；改完后重新编译，
再运行参数查看器和离线测试：

```bash
cmake --build Sim2Real/xz1/build -j2
ctest --test-dir Sim2Real/xz1/build --output-on-failure
Sim2Real/xz1/build/xz1_him_config
```

这些数值来自指定的 XZ1-HIM 训练运行及仿真配置。修改它们可能造成模型输入/输出契约
不匹配，或者让目标超出真实机构允许范围。若换了 ONNX 模型，需同时核对模型元数据和
对应训练配置；不要靠试数值来调实机。

其他模块的可调项目前位于源码中：

- IMU 波特率在 `imu_reader.cpp` 中设为 `921600`；最后一个命令行参数是“原始陀螺数值
  乘多少变成 rad/s”，默认按度/秒乘 `0.0174532925`。若设备直接输出 rad/s 才传 `1`。
- 手柄最大速度和 deadzone 是 `EvdevGamepad` 构造函数的默认参数，位于
  `remote_controler/evdev_gamepad.hpp`。目前没有单独的配置文件或命令行参数。
- 单电机测试的运动上限位于 `motor_control/motor_test.cpp`：相对移动不超过
  `0.2 rad`，`Kp ≤ 5`、`Kd ≤ 1`、时长不超过 `5 s`。不要为了方便绕过这些限制。

改源码中的参数后都要重新编译；不是编辑后立即生效。

## 5. 分模块运行

### 手柄

列出系统识别到的模拟摇杆设备：

```bash
Sim2Real/xz1/build/xz1_gamepad_test --list
```

找到 `/dev/input/eventX` 后运行：

```bash
Sim2Real/xz1/build/xz1_gamepad_test /dev/input/eventX
```

它每 20 ms 打印一行 `vx,vy,yaw_rate,reset`。默认左摇杆 Y 控制前进/后退、左摇杆 X
控制横移、右摇杆 X 控制偏航；`BTN_START` 表示复位。按 `Ctrl+C` 退出。这里的
速度比例是程序默认值，尚未对你的手柄布局和机器人速度做标定。

### IMU

先找稳定设备路径：

```bash
ls -l /dev/serial/by-id/ /dev/ttyACM* /dev/ttyUSB* 2>/dev/null
```

然后把设备替换成实际路径：

```bash
Sim2Real/xz1/build/xz1_imu_reader /dev/serial/by-id/你的设备路径
```

程序输出角速度和四元数。使用前要确认 BMI088 输出已开启陀螺和四元数帧，并核对
设备的轴方向、四元数顺序/方向、陀螺单位和 IMU 安装角度。当前读取程序打印传感器
原始坐标系的结果；安装到机体后的方向变换函数尚未接入实时部署主循环。

### 单个 IM6014 电机

先确认电机 ID、串口路径、供电和 RS-485 接线。`--probe` 会向指定电机发送 STOP 帧，
读取并打印反馈：

```bash
Sim2Real/xz1/build/xz1_motor_test /dev/ttyUSB0 6000000 1 --probe
```

`6000000` 是波特率，`1` 是电机 ID。参考程序也支持 `4000000`。运动测试会真实驱动
电机，只能在输出轴空载、周围清空且能够立即断开电源时使用：

```bash
Sim2Real/xz1/build/xz1_motor_test /dev/ttyUSB0 6000000 1 \
  --move 0.05 1.0 0.2 2
```

参数依次是相对移动弧度、`Kp`、`Kd`、持续秒数。程序有硬编码幅度上限并在结束时发送
STOP；软件 STOP 不是硬件急停，必须确保有可触及的电源断开装置。电机协议通过离线
编码/CRC 测试，但尚未在实际电机上验证。

### 通道一三电机：非阻塞 0 → 50 → 0 rad 测试

独立程序 `xz1_motor_sequence` 复用 `AsyncMotorSystem`，只启用模块通道一，管理 ID
`0、1、2`，固定使用 `6000000 bps`。其余三路不需要连接或存在。主线程以 50 Hz
提交三台电机的目标，后台线程在同一 485 总线上依次收发，不在主线程等待串口反馈。

**本测试仅适用于你确认过的三台独立、空载、输出轴可以自由多圈旋转的电机。**
这里所有角度都是减速器**输出轴绝对角度**，`50 rad` 约为 `7.96` 圈；不是 `50°`，
也不是转子角度。`0 rad` 指当前电机反馈坐标的零位，不执行编码器清零或机械零位标定。
请固定电机本体、保持输出轴周围清空，并准备好断电开关。

在小电脑上同步最新的 `Sim2Real/xz1/` 源码后，以下命令统一在
`~/桌面/xz1-sim2real`（包含 `Sim2Real` 的项目根目录）执行：
只同步源码，不要从原电脑复制 `build/` 编译目录。

```bash
cd ~/桌面/xz1-sim2real
cmake -S Sim2Real/xz1 -B Sim2Real/xz1/build
cmake --build Sim2Real/xz1/build -j2
ctest --test-dir Sim2Real/xz1/build --output-on-failure
```

确认通道一对应的设备路径，再只发 STOP 探测三台电机（设备不一定是 `/dev/ttyUSB0`）：

```bash
Sim2Real/xz1/build/xz1_motor_sequence /dev/ttyUSB0 --probe
```

探测会持续约 2 秒，输出 `serial_sweep_hz`，表示后台每秒完整访问三台电机的轮数。
它与主线程提交目标的 50 Hz 是两件事。`serial_ms`/`max_serial_ms` 是一轮三电机收发
的耗时，不包含后台等待下一个周期的时间。

确认能读到三台电机的 `id`、`q`（输出轴角度，rad）和 `v`（rad/s）后，执行完整运动：

```bash
Sim2Real/xz1/build/xz1_motor_sequence /dev/ttyUSB0 --run
```

程序首先读取 STOP 反馈，从各自当前角度平滑回到 `0 rad`；三台电机都到位并稳定
1 秒后，平滑转到 `50 rad`，稳定 1 秒，再平滑回到 `0 rad`，最后稳定 1 秒并发送 STOP。
每个阶段都要求三台电机同时满足位置误差不超过 `0.05 rad`、速度不超过 `0.1 rad/s`。
它以三次曲线生成目标，默认最大目标速度 `1 rad/s`，最大目标加速度 `0.5 rad/s²`，
因此 `0 → 50 rad` 和返程各约 `75 秒`，全程约 `2.5 分钟`，另加初次回零和到位等待。

默认 `Kp=2 Nm/rad`、`Kd=0.3 Nm/(rad/s)`，速度前馈和力矩前馈均为零。可以调整
以下四个参数；它们只影响这个独立测试程序，不修改 HIM 模型配置：

```bash
Sim2Real/xz1/build/xz1_motor_sequence /dev/ttyUSB0 --run \
  --speed 0.5 --kp 2 --kd 0.3
```

允许范围：`--speed` 为 `0.1–5 rad/s`，`--accel` 为 `0.1–2 rad/s²`（默认 0.5），
`--kp` 为 `0.1–10`，`--kd` 为 `0–1`。
`--speed` 限制的是目标轨迹变化速度，实际电机速度仍需看反馈。默认轨迹下，跟踪误差
超过 `0.75 rad`、实际速度超过 `2 × speed + 0.5 rad/s`、反馈超过 100 ms 未更新、
电机报告故障/警告或阶段在轨迹结束 15 秒后仍未到位，程序都会中止并尝试向三台电机
发送 STOP。异步驱动在目标超过 100 ms 未更新时锁定故障并停止，需要重启程序。
测试程序默认拒绝从绝对角度超过 `±100 rad` 的反馈开始位置控制，避免在未核对编码器
参考时进行过长的初次回零。可用 `--max-start-rad` 显式设置起始绝对角度上限，
范围为 `(0,1000] rad`，只影响位置模式；它不会重设编码器或修改零点。

速度测试会累积多圈绝对角度。若随后读到约 `102 rad`，且三台输出轴仍为空载、可自由
多圈转动，可以将起始范围扩大到 `±200 rad`，再返回原来的绝对零位。
以下命令从 **Sim2Real 目录** 执行：

```bash
cmake --build xz1/build -j2
./xz1/build/xz1_motor_sequence /dev/ttyUSB0 --zero \
  --max-start-rad 200 --speed 5 --accel 2 --kp 10 --kd 0.5
```

程序从各自实测角度平滑接管，不会直接跳发零角度目标。约 `102 rad → 0` 的轨迹
需约 `31 秒`，另加到位等待；从正角度回零时会反向转约 `16 圈`。
若使用 `--speed 0.5`，同样距离的轨迹约需 `306 秒`。
这是实际转回绝对零位，不是把当前角度清零，也不对角度取模。原有跟踪、反馈和 STOP
保护继续生效。

若要在 50 rad 单程中达到 `5 rad/s`，使用：

```bash
Sim2Real/xz1/build/xz1_motor_sequence /dev/ttyUSB0 --run \
  --speed 5 --accel 2 --kp 10 --kd 0.5
```

这时每个 50 rad 行程的轨迹约为 `15 秒`，加上初次回零和各阶段等待，完整往返约
`34 秒`（以实际到位时间为准）。5 rad/s 是三次轨迹中途的峰值目标速度，起止速度为零，
全程平滑加减速。若只改 `--speed 5`，沿用默认 `--accel 0.5`，则受加速度限制，单程约
24.5 秒且峰值目标速度约 3.06 rad/s。`--preview` 和运行日志中的 `peak_target_speed`
会显示轨迹实际安排的峰值；实机速度仍以反馈 `v` 为准。

按 `Ctrl+C` 中止也会尝试发送 STOP；中止时不会自动返回零位。STOP 会退出驱动保持，
所以最后的零位不会在程序退出后继续保持。串口断开时不能保证 STOP 送达，物理断电仍
是最终停止手段。

不接硬件预览阶段和预计时间：

```bash
Sim2Real/xz1/build/xz1_motor_sequence /dev/ttyUSB0 --preview
Sim2Real/xz1/build/xz1_motor_sequence --help
```

预览不打开串口，预计时间假设初始角度为零。运行时每约 0.5 秒输出反馈，并报告后台
完成的轮询次数 `cycles`、错过的周期数 `missed`。阶段完成时的 `max_submit_us` 是主线程
提交命令所花的最大时间，不包括串口往返时间。源码参数和阶段位于
[`motor_sequence.cpp`](xz1/motor_control/motor_sequence.cpp)，平滑轨迹位于
[`position_ramp.hpp`](xz1/motor_control/position_ramp.hpp)。

#### 回零停在约 0.15–0.24 rad，或实际通信只有约 16 Hz

实测默认 `Kp=2` 时三台电机在零位附近仍可能有明显残差，因此无法通过 `±0.05 rad`
到位检查，程序不会继续到 50 rad。增益换算与原 Python 程序一致；低增益下的静摩擦或
外部负载可能造成这种现象，需要结合反馈力矩判断，不能据此修改编码器偏置。

先只做回零测试，显式使用原 Python 单电机测试默认的增益，并降低目标速度：

```bash
Sim2Real/xz1/build/xz1_motor_sequence /dev/ttyUSB0 --zero \
  --speed 0.5 --kp 5 --kd 0.5
```

`--zero` 只运行“当前角度 → 0 rad”阶段，之后发送 STOP，绝不进入 50 rad 阶段。
日志为每台电机输出 `cmd`（目标角度）、`err=cmd-q` 和 `torque`（反馈输出力矩，Nm）。
`Kp=5` 不保证消除残差；若仍不到位，需要继续根据力矩和误差分析摩擦、负载与控制参数。
保持原到位条件，不通过放宽容差来掩盖回零误差。

后续实测 `Kp=5` 时残差缩小到约 `0.061、0.088、0.061 rad`，通信稳定在 50 Hz，
静止反馈力矩约 `0.35–0.48 Nm`。这一结果与摩擦/负载造成的 PD 稳态误差相符。
独立三电机程序的可选 Kp 上限因此扩到 10，默认仍为 2；可继续只做回零测试：

```bash
Sim2Real/xz1/build/xz1_motor_sequence /dev/ttyUSB0 --zero \
  --speed 0.5 --kp 10 --kd 0.5
```

若其他条件相同，提高 Kp 通常会缩小此类残差，但不保证精确到零。仍须等待三台电机
都满足原到位条件，出现明显振荡时停止测试。这一修改只作用于独立三电机测试的参数
校验；单电机小幅测试上限和 HIM 部署增益不受影响。

当前这块 FTDI 模块曾读到 `ftdi_sio` 驱动的 `latency_timer=16 ms`，USB 短包上报延迟
会拖慢三电机轮询。6 Mbps 是串口线速，不能消除 USB 缓存等待。停止正在运行的测试后，
检查并把通道一对应的驱动延迟设为 1 ms（此操作不发送电机命令）：

```bash
cat /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
printf '1\n' | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
cat /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
Sim2Real/xz1/build/xz1_motor_sequence /dev/ttyUSB0 --probe
```

这项设置需要系统管理员权限，只对当前设备生效，拔插或重启后应重新检查。若换了通道，
同步替换 `ttyUSB0`；若没有 `latency_timer` 文件，不要创建它，需要按实际 USB 驱动分析。
是否达到 50 Hz 以新的 `serial_sweep_hz` 和串口耗时为准。

这个测试程序是 C++ 可执行文件，不依赖 Conda；终端中 `conda` 命令不可用不会影响它。

### 通道一三电机：速度环 0 → 10 → 0 rad/s 测试

仍使用 `xz1_motor_sequence` 和同一个异步驱动，通道一挂 ID `0、1、2`，固定 6 Mbps、
50 Hz。新增 `--velocity` 是独立速度控制模式：`Kp=0`，目标位置和力矩前馈均为零，
命令的 `speed_rad_s` 为目标速度，`Kd` 为电机内部速度反馈的比例系数，即
`torque ≈ Kd × (v_target - v_feedback)`。它不使用位置轨迹来间接实现速度控制，
也没有在主机额外加入积分器。

在项目根目录运行：

```bash
Sim2Real/xz1/build/xz1_motor_sequence /dev/ttyUSB0 --velocity \
  --speed 10 --accel 2 --kd 1 --hold 3
```

如果终端已经在 `Sim2Real/` 内，则运行：

```bash
./xz1/build/xz1_motor_sequence /dev/ttyUSB0 --velocity \
  --speed 10 --accel 2 --kd 1 --hold 3
```

以上数值也是速度模式的默认值。阶段如下：

1. 读取三台电机 STOP 反馈，平滑将速度目标降到零并稳定 1 秒。
2. 平滑从零加速到 `10 rad/s`，目标加速度最多 `2 rad/s²`，加速轨迹约 `7.5 秒`。
3. 三台电机实际速度都进入目标的允许误差范围后，连续保持 `3 秒`。
4. 速度目标从 10 rad/s 平滑减速到零，减速约 `7.5 秒`；三台实际速度均不超过 `0.1 rad/s`
   并稳定 1 秒后，发送 STOP。

从静止开始总计约 `20–21 秒`，实际时间取决于反馈和到位等待。`10 rad/s` 是减速器
输出轴速度，约 `95.5 rpm`。这是速度测试，输出轴会持续转动，最终只回到零速度，
不返回原来的角度。电机本体需固定，输出轴周围保持清空。

默认正速度阶段允许误差为 `max(0.5 rad/s, 目标速度的 10%)`，在 `10 rad/s` 时为
`±1 rad/s`；零速度阶段为 `±0.1 rad/s`。这是测试的通过条件，不代表实际速度精确为
10；应查看 `v` 和 `v_err`，摩擦可能使这种纯速度比例控制存在稳态误差。

允许参数：`--speed 0.1–10 rad/s`、`--accel 0.1–2 rad/s²`、`--kd (0,1]`、
`--hold 0.5–10 秒`。`--hold` 只设置正速度阶段稳定保持的时间；两个零速度阶段固定
保持 1 秒。速度模式不接受 `--kp`，避免意外混入位置保持。

日志输出各电机 `q`、`v`、`torque`、`v_cmd`（目标速度）和 `v_err=v_cmd-v`。
实际速度绝对值超过设定峰值加 `2 rad/s`、速度跟踪误差超过 `3 rad/s` 持续 1 秒、
反馈/目标超时、电机故障/警告、阶段超时，都会中止并尝试 STOP。按 `Ctrl+C` 时直接
尝试 STOP，不继续执行减速轨迹；断电开关保持可触及。

预览阶段和预计时间，不打开串口：

```bash
./xz1/build/xz1_motor_sequence /dev/ttyUSB0 --velocity-preview
```

协议接口 [`velocity_command()`](xz1/motor_control/im6014.hpp) 可用于后续速度控制，
平滑速度轨迹位于 [`velocity_ramp.hpp`](xz1/motor_control/velocity_ramp.hpp)。

### 四路异步电机控制

模块手册说明，单根 USB 线接入电脑后会枚举四个串口设备，通常为 `/dev/ttyUSB0` 至
`/dev/ttyUSB3`；每个 UART 是独立的 485 通道。本程序按腿分配四个后台线程，每个线程只
轮询对应通道上的三台电机。每一路内部仍是半双工通信，三台电机依次收发；四路之间可以
并行。默认目标周期为 20 ms（50 Hz），能否按时完成一轮需要在实际设备上测量。
只测试部分腿时，`LegDevices` 中的空字符串会禁用对应串口；至少保留一路。不能向被禁用
的腿提交使能命令。

你已经使用现有 IM6014 控制程序和这块 4 路模块完成过通信测试，因此这里按实测可用的
硬件链路设计。C++ 程序默认设置为 `6000000 bps`，与参考程序默认配置一致，也支持
`4000000 bps`；运行时应沿用你已经验证成功的波特率和串口映射。模块手册的标称值与这套
实测配置不同，说明书参数不作为当前部署的阻断条件。

先确认四个串口路径和电机 ID，再用纯 STOP 探测。以下命令按你的规划，电机 ID 对应
action 索引 0–11：

```bash
ls /dev/ttyUSB*
Sim2Real/xz1/build/xz1_motor_system_probe \
  /dev/ttyUSB0 /dev/ttyUSB1 /dev/ttyUSB2 /dev/ttyUSB3 6000000 action 2
```

如果电机实际使用每路重复 ID 0–2，把参数 `action` 改为 `per-leg`。探测程序只发 STOP
帧，不发运动目标。

程序内调用方式如下。示例使用全局 ID；若实际采用每路重复 ID 0–2，改用
`kPerLegMotorIds`。`calibrated_target` 必须先做完方向和零偏置到 URDF 角度的换算：

```cpp
nazarite::xz1::motor::LegDevices ports{
    "/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyUSB2", "/dev/ttyUSB3"};
nazarite::xz1::motor::AsyncMotorSystem bus(
    ports, 6000000, nazarite::xz1::motor::kActionIndexMotorIds);
nazarite::xz1::motor::MotorCommands commands{};
for (std::size_t i = 0; i < commands.size(); ++i) {
  commands[i] = nazarite::xz1::motor::position_command(
      static_cast<std::uint8_t>(i), calibrated_target[i], kp[i], kd[i]);
}
bus.submit_commands(commands);  // 不等待串口反馈
const auto latest_state = bus.snapshot();
```

启用的电机命令必须带 1 秒通信超时保护。目标 100 ms 未更新、通信失败或电机故障时，
四个线程都会尽力停止各自通道的三台电机。`bus.stop()` 会等待四路 STOP 扫描完成；
硬件断电开关仍要保持可触及。

## 6. Python 空推理性能测试

之前的 Python 脚本可在 `mjlab_env` 中运行；它同样不连接硬件：

```bash
conda activate mjlab_env
cd /home/haozi/桌面/Nazarite-mjlab
python -m Sim2Real.xz1_him --warmup 100 --iterations 1000
```

按 50 Hz 等周期运行 30 秒并测 CPU：

```bash
python -m Sim2Real.xz1_him --hz 50 --iterations 1500
```

测进程 RSS 内存的脚本目前是一次性测试工具，不是 C++ 部署主程序的一部分。
已保存结果在 `Sim2Real/output/`。这些数字是本机空输入测试，不代表目标部署电脑、
实机观测构造、串口通信和完整控制循环的 CPU/内存或实时性。

## 7. 常见问题

- **CMake 找不到源文件**：在仓库根目录重新执行第 2 节命令。CMakeLists 已按当前
  `motor_control/`、`remote_controler/` 文件夹配置。
- **提示找不到 ONNX Runtime C++ SDK**：确认下面两个文件都存在，或使用第 2 节的
  `-DONNXRUNTIME_ROOT=...` 参数：
  `Sim2Real/sdk/onnxruntime-linux-x64-1.31.1/include/onnxruntime_cxx_api.h` 和
  `Sim2Real/sdk/onnxruntime-linux-x64-1.31.1/lib/libonnxruntime.so`。
- **串口或手柄 Permission denied**：这是 Linux 设备权限问题，确认用户属于相应的
  `dialout`（串口）或 `input`（手柄）组后重新登录系统。
- **模型文件不存在**：检查 ONNX 命令中的模型路径是否和你本机训练输出一致。
- **程序能编译但机器人不能动**：总线传输层还没有接入完整 HIM 实时控制循环，且实机
  的方向、零位、关节限制、总线拓扑和硬件急停尚未验证。

## 8. 建议的上手顺序

先编译并跑 `ctest`，再运行 `xz1_him_config` 和零输入 ONNX 测试。然后分别检查手柄、
IMU 输出，最后在安全夹具/空载条件下测试单个电机。待确认机器人驱动接口、关节映射、
传感器坐标系和急停方式后，再开发并验证 12 路闭环控制程序。
