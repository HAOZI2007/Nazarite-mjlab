# Nazarite Go2 Sim2Sim

`sim2sim/` 是 Nazarite Go2 策略的独立 MuJoCo 推理与可视化程序。它不重新
创建 mjlab 的 ManagerBased 环境，而是直接读取 Go2 XML，构造一个简化平面，
加载已经训练好的 actor，然后以和训练一致的控制频率驱动机器人。

当前支持五种模式：

| 模式 | 策略 | 观测维度 | 用途 |
|---|---|---:|---|
| `baseline` | `Nazarite-Velocity-Flat-Go2` | 45 | 普通 Grid Adaptive 速度策略 |
| `wtw` | `Nazarite-Velocity-Flat-Go2-WTW` | 498 | Grid Adaptive + WTW Trot 策略 |
| `wtw_delta_residual` | `go2_wtw_delta_residual/model_5650.pt` | 2223 | WTW prior + DELTA residual 踏石策略 |
| `wtw_delta_direct` | `go2_wtw_delta_direct/model_9950.pt` | 2223 | Direct 任务的相机 BEV + 动作融合 |
| `him` | `go2_him/policy.onnx` | 282 | HIM 历史观测策略和专用障碍场景 |

当前 WTW sim2sim 配置针对已经训练好的 Trot 策略，默认 behavior 为 2.5 Hz、
0 偏移机体高度、0 pitch、0.25 m 步宽和 0.06 m 摆腿高度。它不是通用的多步态
部署器；如果加载只训练过 Trot 的模型，不要在运行时把 phase 改成 Pronking、
Bounding 或 Pacing。

## 1. 目录结构

```text
Train/Nazarite/sim2sim/
├── main.py                 # 命令行入口和 MuJoCo 主循环
├── config.py               # XML、关节顺序、控制频率、PD 增益和 action scale
├── wtw_delta_direct/       # Direct 任务相机 BEV、观测和策略兼容加载
├── mujoco_io.py            # 加载 XML、创建位置执行器、读写仿真状态
├── scene.py                # 添加平面和可视化网格
├── policy_runner.py        # 加载 ONNX/.pt actor 并执行推理
├── observation.py          # 公共本体观测
├── baseline/
│   ├── config.py           # baseline 模型路径和 45D 观测契约
│   └── observation.py      # baseline 观测拼接
├── wtw/
│   ├── config.py           # WTW 模型路径、behavior 和 command 限幅
│   ├── command.py          # WTW command 限幅
│   └── observation.py      # WTW 历史帧和 phase 构造
├── wtw_delta_residual/
│   ├── config.py           # model_5650.pt 和 2223D 输入契约
│   ├── observation.py      # 498D WTW + 61D DELTA proprioception
│   ├── policy.py           # WTW + DELTA .pt actor 推理
│   └── terrain.py          # 踏石场景和 16x26x4 BEV 射线地图
├── him/
│   ├── config.py           # HIM 训练 run 和 282D 观测契约
│   ├── observation.py      # 6 帧、每帧 47D 的当前优先历史
│   ├── policy.py           # HIM .pt/ONNX 推理
│   └── scene.py            # 实体楼梯和 320x800x100 mm 墙
├── gamepad.py              # Linux evdev 手柄读取
├── camera.py               # 跟随机器人相机
└── math_utils.py           # action 到关节目标的转换
```

运行时默认会从下面的位置寻找 Go2 XML：

```text
Train/Nazarite/MJCF-Manager/Robots/GO2/xmls/go2.xml
```

默认模型路径写在：

```text
sim2sim/baseline/config.py
sim2sim/wtw/config.py
```

如果换了新的训练 run，推荐通过 `--policy` 显式传入模型，不要频繁修改源码
中的默认路径。

## 2. 环境准备

所有命令从 `Train/Nazarite` 执行：

```bash
cd /home/haozi/桌面/Nazarite-mjlab/Train/Nazarite
uv sync
```

检查关键依赖：

```bash
uv run python -c "import mujoco, numpy, onnxruntime; print('sim2sim dependencies: OK')"
```

如果当前环境提示 `ModuleNotFoundError: No module named 'onnxruntime'`，说明
项目环境还没有安装 ONNX 推理后端。项目现在已将它列为 sim2sim 运行依赖，
先同步环境：

```bash
uv sync
```

如果只想临时补装，也可以使用：

```bash
uv pip install onnxruntime
```

`.pt` 策略不使用 ONNX Runtime，但 `.onnx` 策略必须安装它。这不会改变训练任务配置。

其中：

- `mujoco`：运行物理仿真和 viewer；
- `numpy`：构造观测、指令和动作；
- `onnxruntime`：加载 ONNX actor；
- `torch`：当直接加载 `.pt` actor 时使用；
- `evdev`：使用 Linux 手柄时读取 `/dev/input/event*`。

## 3. 最小运行方式

### 3.1 baseline 固定指令

```bash
uv run python -m sim2sim.main \
  --mode baseline \
  --policy logs/rsl_rl/go2_flat_baseline/<run>/<run>.onnx \
  --vx 0.5 \
  --vy 0.0 \
  --yaw 0.0
```

如果默认路径中的模型存在，可以省略 `--policy`：

```bash
uv run python -m sim2sim.main --mode baseline --vx 0.5
```

### 3.2 WTW 固定指令

```bash
uv run python -m sim2sim.main \
  --mode wtw \
  --policy logs/rsl_rl/go2_flat_wtw_independent/<run>/model_14950.pt \
  --vx 0.5 \
  --vy 0.0 \
  --yaw 0.0
```

当前 WTW 策略使用 `.pt` 也可以直接运行。`PolicyRunner` 会从
`actor_state_dict` 重建 MLP，并检查输入维度是否为 498、输出维度是否为 12。
baseline 常用 `.onnx`，加载时还会检查 ONNX metadata 中的观测名称和关节顺序。

### 3.3 WTW + DELTA 踏石策略

不指定 `--policy` 时，`wtw_delta_residual` 会自动加载：

```text
logs/rsl_rl/go2_wtw_delta_residual/2026-09-22_18-53-22/model_5650.pt
```

直接测试踏石场景：

```bash
./mjlab/.venv/bin/python -m sim2sim.main \
  --mode wtw_delta_residual \
  --vx 0.25 \
  --vy 0.0 \
  --yaw 0.0 \
  --no-camera-follow
```

这个模式的输入不是普通 498D WTW 输入，而是：498D WTW 历史、61D 当前
DELTA 状态和 16x26x4 的局部地形地图，总计 2223D。地图由 MuJoCo 踏石和
坑底的碰撞几何实时射线采样得到。

### 3.4 HIM 楼梯和墙场景

HIM 模式默认使用：

```text
logs/rsl_rl/go2_him/2026-09-29_19-00-44/policy.onnx
```

场景沿机器人前进方向依次放置实体楼梯、平台、平地和墙。墙的实际尺寸为：
高度 `0.32 m`、横向宽度 `0.80 m`、前进方向厚度 `0.10 m`。

固定速度测试：

```bash
./mjlab/.venv/bin/python -m sim2sim.main \
  --mode him \
  --policy logs/rsl_rl/go2_him/2026-09-29_19-00-44/model_11700.pt \
  --vx 0.25 \
  --vy 0.0 \
  --yaw 0.0 \
  --no-camera-follow
```

使用同一 run 导出的 `policy.onnx` 也可以运行。手柄版本如下：

```bash
./mjlab/.venv/bin/python -m sim2sim.main \
  --mode him \
  --gamepad \
  --max-vx 0.5 \
  --max-vy 0.5 \
  --max-yaw 0.5 \
  --no-camera-follow
```

### 3.5 调整相机

```bash
uv run python -m sim2sim.main \
  --mode wtw \
  --policy /absolute/path/to/policy.pt \
  --vx 0.5 \
  --camera-distance 3.0 \
  --camera-azimuth 135 \
  --camera-elevation -25 \
  --camera-height 0.12
```

关闭自动跟随：

```bash
uv run python -m sim2sim.main --mode baseline --no-camera-follow
```

## 4. 手柄控制

程序使用 Linux `evdev`，不依赖品牌专用驱动。先查看系统发现的设备：

```bash
uv run python -m sim2sim.main --list-gamepads
```

使用自动发现的第一个手柄：

```bash
./mjlab/.venv/bin/python -m sim2sim.main \
  --mode wtw_delta_residual \
  --gamepad
```

不使用镜头跟随，同时限制踏石测试速度：

```bash
./mjlab/.venv/bin/python -m sim2sim.main \
  --mode wtw_delta_residual \
  --gamepad \
  --max-vx 0.5 \
  --max-vy 0.5 \
  --max-yaw 0.5 \
  --no-camera-follow
```

在 DELTA 模式下，WTW behavior 仍可通过手柄调整：右摇杆 Y 调频率，LB/RB
调 body height，LT/RT 调 body pitch，X/B 调 stance width，A/Y 调 swing
height，Select 恢复默认值。Trot 的相位关系保持不变。

默认映射为：

| 输入 | 指令 | 默认最大值 |
|---|---|---:|
| 左摇杆 Y | `vx`，向前为正 | 1.0 m/s |
| 左摇杆 X | `vy`，向左/右为正负取决于手柄坐标 | 1.0 m/s |
| 右摇杆 X | `yaw` | 1.0 rad/s |

程序会对摇杆施加 `0.08` 的中心死区。可以指定设备和轴：

```bash
uv run python -m sim2sim.main \
  --mode baseline \
  --policy /absolute/path/to/policy.onnx \
  --gamepad \
  --gamepad-device /dev/input/event21 \
  --max-vx 1.0 \
  --max-vy 0.0 \
  --max-yaw 0.5 \
  --deadzone 0.10 \
  --axis-vx ABS_Y \
  --axis-vy ABS_X \
  --axis-yaw ABS_RX
```

如果手柄接收器断开，程序会把指令置零，而不会保留最后一个运动指令。

## 5. 控制循环逻辑

主循环在 `main.py` 中按下面顺序执行：

```text
读取固定指令或手柄指令
        ↓
WTW 模式进行 command 限幅
        ↓
从 MuJoCo 状态构造 actor observation
        ↓
构造 WTW 历史、DELTA proprioception 和局部 BEV 地图
        ↓
ONNX/.pt actor 推理，得到 12 维归一化 action
        ↓
target_q = default_q + action * ACTION_SCALE
        ↓
写入 12 个位置执行器
        ↓
执行 10 个 MuJoCo physics step
        ↓
WTW 模式推进 phase
```

当前控制参数在 `sim2sim/config.py` 中：

```python
PHYSICS_DT = 0.002       # MuJoCo 物理步长
DECIMATION = 10          # 每次策略推理对应的物理步数
CONTROL_DT = 0.02        # 策略控制周期
ACTION_SCALE = 0.25      # 12 个关节的目标角度缩放
```

sim2sim 使用位置目标执行器：髋/大腿/小腿的刚度为 `20.0`，阻尼为 `0.5`，
单关节力限制为 `45.0`。这些参数与当前 sim2sim 配置一致，但它们不等于真实
Go2 电机的底层驱动接口；部署到实机前还需要单独完成执行器映射、限幅和安全验证。

## 6. 观测契约

### 6.1 baseline：45 维

baseline 按下面顺序拼接当前帧：

| 顺序 | 观测项 | 维度 |
|---:|---|---:|
| 1 | `base_ang_vel` | 3 |
| 2 | `projected_gravity` | 3 |
| 3 | `joint_pos`（相对默认关节角） | 12 |
| 4 | `joint_vel` | 12 |
| 5 | `actions`（上一帧 action） | 12 |
| 6 | `command=[vx, vy, yaw]` | 3 |

总维度为 `3 + 3 + 12 + 12 + 12 + 3 = 45`。

### 6.2 WTW：498 维

WTW 当前训练配置使用不同的历史长度，sim2sim 必须复现同样的拼接方式：

| 观测项 | 单帧维度 | 历史长度 | 展平后维度 |
|---|---:|---:|---:|
| `base_ang_vel` | 3 | 10 | 30 |
| `projected_gravity` | 3 | 10 | 30 |
| `joint_pos` | 12 | 10 | 120 |
| `joint_vel` | 12 | 10 | 120 |
| `actions` | 12 | 10 | 120 |
| `command` | 3 | 10 | 30 |
| `behavior` | 8 | 5 | 40 |
| `phase` | 8 | 0 | 8 |
| **合计** |  |  | **498** |

历史顺序是“最旧帧 → 最新帧”，与 mjlab 的 `CircularBuffer.buffer` 一致。程序
第一次运行时使用 reset 后的第一帧填满历史，之后每个控制周期追加一帧。

### 6.3 WTW behavior

`sim2sim/wtw/config.py` 中的 behavior 顺序固定为：

```text
[theta1, theta2, theta3,
 frequency,
 body_height_offset,
 body_pitch,
 stance_width,
 foot_swing_height]
```

当前值为：

```python
[0.5, 0.0, 0.0, 2.5, 0.0, 0.0, 0.25, 0.06]
```

前三个值 `[0.5, 0.0, 0.0]` 对应当前 Trot。`body_height` 是相对基础高度
`0.32 m` 的偏移量，因此 `0.0` 对应目标高度 `0.32 m`。

behavior 必须与 checkpoint 训练分布一致。比如只训练了固定 Trot 的模型，不能
只改前三个 theta 就认为模型获得了 Pronking 能力；不同步态需要独立训练或在
同一个训练任务中显式覆盖对应 gait 分布。当前 `WTWObservationBuilder` 的
`phase_reference()` 也固定按 Trot 偏移构造 phase，切换其他步态时必须同步修改
phase 逻辑并使用对应训练过的模型。

## 7. WTW phase 逻辑

WTW phase 在 `wtw/observation.py` 中维护：

```python
base_phase = (base_phase + frequency * CONTROL_DT) % 1.0
```

当前规则是：

- 初始 phase 为 `0`；
- `frequency=2.5 Hz`、`CONTROL_DT=0.02 s` 时，每个控制周期前进 `0.05` 个周期；
- 当 `|vx| + |vy| + |yaw| <= 0.05` 时冻结 phase；
- 重新给出有效运动指令后，phase 从冻结位置继续推进；
- 当前 Trot 的四条腿 phase 偏移为 `[0.5, 0.0, 0.0, 0.5]`，腿顺序是
  `[FL, FR, RL, RR]`；
- phase 以 8 维 `[sin(2πphase_1...4), cos(2πphase_1...4)]` 输入 actor；
- phase 不堆叠历史帧。

因此，sim2sim 的零速度行为和训练任务一致：机器人停止运动时，phase 不会继续
推进，避免停下后仍被新的步态相位驱动。

## 8. 模型和路径管理

推荐把用于 sim2sim 的模型放在对应训练日志目录中，并通过绝对路径或从
`Train/Nazarite` 出发的相对路径传入：

```bash
uv run python -m sim2sim.main \
  --mode wtw \
  --policy logs/rsl_rl/go2_flat_wtw_independent/2026-09-11_20-39-13/model_14950.pt
```

加载时会进行以下检查：

- 文件存在；
- ONNX 输入是固定维度；
- baseline 输入必须为 45 维；
- WTW 输入必须为 498 维；
- actor 输出必须为 12 维；
- ONNX metadata 中的观测名称与关节顺序必须匹配；
- action、observation 不能含 NaN 或 Inf。

这些检查用于尽早发现“模型和 sim2sim 观测契约不一致”，不代表模型一定已经
在物理上学会了稳定运动。

## 9. WTW + DELTA 深度相机旁路诊断

WTW sim2sim 可以挂载一台与训练侧 D435 几何一致的 MuJoCo 深度相机，并实时显示
四个面板：原始深度图、BEV 高程、原始 support 和 hole-fill 后的 confidence。
该功能是旁路诊断：深度图和 BEV **不会进入 WTW observation，也不会改变 WTW
动作**。它适合先检查相机视角、分辨率和投影质量，再决定是否让视觉参与控制。

启动 WTW 并打开诊断面板：

```bash
uv run python -m sim2sim.main \
  --mode wtw \
  --policy /absolute/path/to/wtw.onnx \
  --vx 0.3 \
  --depth-debug
```

诊断面板默认直接嵌入 MuJoCo 原生 viewer，以紧凑四格显示。按 `1/2/3/4`
会将对应面板放大；按 `0` 或 `Esc` 恢复四格。需要旧版独立 matplotlib 窗口时，
可以额外传入 `--depth-external-window`。按 `H` 可以临时隐藏/显示诊断面板，
只观察机器人运动。

默认配置在 `sim2sim/wtw_delta_residual/depth_camera.py` 的
`DepthCameraConfig` 和 `BevConfig` 中。也可以直接命令行覆盖，例如把相机改到
机体前方、水平安装、竖向分辨率，并扩大 BEV：

```bash
uv run python -m sim2sim.main \
  --mode wtw --policy /absolute/path/to/wtw.onnx --depth-debug \
  --depth-width 72 --depth-height 128 \
  --depth-pos-x 0.40 --depth-pos-y 0.0 --depth-pos-z 0.05 \
  --depth-pitch 0 --depth-roll 90 \
  --bev-height 26 --bev-width 16 \
  --bev-x-min 0.0 --bev-x-max 2.5 \
  --bev-y-min -0.8 --bev-y-max 0.8 \
  --bev-fill-kernel 5
```

相机位置单位为 Go2 `base_link` 坐标系米；pitch 是向下俯仰角，roll 是绕光轴
的图像旋转角。只修改分辨率时，D435 内参会按宽高比例缩放；如果有该分辨率的
实测标定值，应同时传入 `--depth-fx/--depth-fy/--depth-cx/--depth-cy`。当
`roll=±90` 且宽度小于高度时，程序会自动交换横竖方向的 D435 内参，适配竖装
相机；显式传入四个内参后则完全以用户参数为准。

BEV 的行方向是 `+y -> -y`，列方向是近处 `x_min` -> 远处 `x_max`。support
表示原始深度像素实际落入的网格，confidence 还包含局部 hole filling，因此
稀疏点云和插值区域不会混为一谈。

## 10. Nazarite-WTW-Delta-Direct-Go2 sim2sim

Direct 模式放在 `sim2sim/wtw_delta_direct/`，不会改变旧的
`wtw_delta_residual` 模式。它和训练任务使用相同的输入契约：D435 深度图先
投影为 `16x26x5` 的 `(x, y, z, support, confidence)` BEV，再送入
`WtwDeltaDirectActionModel`。相机 BEV 会真正参与动作计算；`--depth-debug` 只是
额外显示同一帧的四个诊断面板。

启动 Direct checkpoint：

```bash
uv run python -m sim2sim.main \
  --mode wtw_delta_direct \
  --policy logs/rsl_rl/go2_wtw_delta_direct/2026-09-24_22-11-00/model_9950.pt \
  --vx 0.3 \
  --depth-debug \
  --gamepad
```

Direct 默认与训练配置一致：相机 `72x128`、位置 `(0.40, 0.0, 0.05)`、水平
朝前、BEV `x=0..2.0 m`、`y=-0.6..0.6 m`、3x3 局部填洞。相机和 BEV 参数仍可
用现有的 `--depth-*`、`--bev-*` 参数覆盖；BEV 高度和宽度必须保持 `16x26`，
否则无法满足 checkpoint 的输入维度。

当前 Direct 日志中的旧 checkpoint 使用单层 `action_fusion` 和旧版 DELTA
几何 token。`wtw_delta_direct/policy.py` 会根据 checkpoint key 自动选择兼容
模型；以后用当前源码重新训练产生的新版 Direct checkpoint 也会自动选择新版
模型。这项兼容逻辑只位于 sim2sim 文件夹，不会改变训练代码。

Direct 模式的动作链路是：

```text
D435 depth -> camera BEV (16x26x5) -> DELTA encoder
WTW observation -> WTW prior
[prior, DELTA proprio, terrain latent] -> action fusion -> 12 joint actions
```

不要把 Direct 模式写成 `wtw_delta_residual`：后者仍使用旧的向下射线
`build_delta_map()` 和 4 通道 residual checkpoint。

## 11. 常见问题

### 10.1 `Default policy not found`

默认模型路径是某一次实验的固定路径。换了 run 后请显式指定：

```bash
uv run python -m sim2sim.main \
  --mode baseline \
  --policy /absolute/path/to/your_policy.onnx
```

### 10.2 `Expected 498D policy input`

这通常表示：

- 把 baseline 模型放到了 `--mode wtw`；
- WTW 模型的历史长度与当前 sim2sim 不一致；
- 训练时启用了 observation normalization、RNN 或其他模型结构；
- 导出的模型不是当前 `base_env_cfg.py` 对应的 checkpoint。

不要通过随意补零来解决。应先核对训练日志中的 `params/env.yaml`、观测历史
长度和 checkpoint 来源。

### 10.3 机器人一启动就倒地

按下面顺序排查：

1. 确认 baseline/WTW mode 与模型类型匹配；
2. 确认使用了正确的 Go2 XML；
3. 确认 action scale、默认关节角和控制频率没有被改动；
4. 先用 `--vx 0.0 --vy 0.0 --yaw 0.0` 检查站立；
5. 再从 `--vx 0.2` 开始逐渐增加速度；
6. WTW 模式确认 behavior 与训练配置一致。

### 10.4 手柄找不到

```bash
uv run python -m sim2sim.main --list-gamepads
```

运行时必须显式加入 `--gamepad`，例如：

```bash
uv run python -m sim2sim.main \
  --mode wtw \
  --policy /absolute/path/to/wtw.onnx \
  --gamepad \
  --depth-debug
```

当前读取是非阻塞的，不推动摇杆时不会卡住 MuJoCo 控制循环。启动日志会打印
实际解析出的 `vx/vy/yaw` 轴；如果接收器没有 `ABS_RX`，yaw 会自动尝试
`ABS_RZ/ABS_Z`，都没有时只禁用 yaw，不影响前向和横向控制。

如果设备没有被发现，检查 Linux 用户是否有读取 `/dev/input/event*` 的权限，
并用 `--gamepad-device` 指定实际设备。不同接收器可能暴露不同轴名，可用
`--axis-vx`、`--axis-vy` 和 `--axis-yaw` 覆盖默认值。

## 11. sim2sim 的边界

当前实现适合做以下事情：

- 快速查看训练 actor 的动作效果；
- 对比 baseline 和 WTW 策略；
- 检查观测维度、历史堆叠和 phase 是否接线正确；
- 用固定速度或手柄做平面上的定性验证；
- 在进入实机前检查模型输出是否异常。

当前实现不等价于：

- mjlab 中完整的随机摩擦、质量、推力和传感器噪声训练环境；
- WTW 的网页 play 行为参数实时覆盖面板；
- 真实 Go2 电机、状态估计器和安全控制器；
- 复杂地形 sim2sim；
- 真机部署程序。

特别是 WTW sim2sim 当前只重建 actor 所需的本体观测、behavior、command 和
phase，不重建训练时的奖励、Grid 课程或 domain randomization。它的作用是
验证已训练策略的输入输出接口和基础运动效果。

另外，当前 sim2sim 使用的 WTW command 限幅为 `vx∈[-2, 2] m/s`、
`vy∈[-1, 1] m/s`、`yaw∈[-1, 1] rad/s`，它比当前训练配置的范围更宽。为了
避免测试超出策略训练分布，建议实际运行时仍限制在训练范围：
`vx∈[-1, 1]`、`vy∈[-0.5, 0.5]`、`yaw∈[-1, 1]`。

## 12. 相关配置和文档

- 训练任务总览：[Train/Nazarite/README.md](../README.md)
- WTW 实现说明：[WTW-从零手写Walk-These-Ways](../../../docs/WTW-从零手写Walk-These-Ways.md)
- WTW 步态行为设计：[WTW-步态行为设计教程](../../../docs/WTW-步态行为设计教程.md)
- WTW 奖励调参：[WTW奖励调参详细使用指南](../../../docs/WTW奖励调参详细使用指南.md)
- 当前 WTW 训练配置：`../Nazarite-src/nazarite/config/train_config/base_env_cfg.py`
- 当前 Go2 机器人配置：`../Nazarite-src/nazarite/config/robot_config/go2_cfg.py`
