# XZ1-HIM 测试部署

当前提供全零输入 ONNX 推理基准和不依赖训练框架的部署核心。
仅依赖 NumPy / ONNX Runtime，不导入 PyTorch、MuJoCo 或 mjlab。

## 在 Conda 环境运行

在仓库根目录执行：

```bash
conda activate mjlab_env
cd /home/haozi/桌面/Nazarite-mjlab
python -m Sim2Real.xz1_him --warmup 100 --iterations 1000 \
  --output Sim2Real/output/xz1_him_benchmark.json
```

默认使用 `2026-10-25_22-04-31/policy.onnx`。也可以传入
`--policy /绝对路径/policy.onnx`，或修改 `xz1_him/config.json`。
配置中的模型相对路径以配置文件所在目录为基准。

输入始终为 float32 `[1, 282]` 全零张量，输出为 `[1, 12]`。
该输入仅用于测速，不代表机器人站立状态（实际静止重力是 `[0,0,-1]`）。
程序打印模型加载、首次推理及预热后的平均/P50/P95/P99/最大耗时，
与 50 Hz 的 20 ms 周期比较。`session.run` 的计时包含 Python 调用与
输入输出拷贝，不包含观测构造、电机通信、调度或模型加载；逐轮验证在计时外。
使用 CPU 后端、单线程。结果是本机测试值，不是实机实时性保证。
没有硬件连接或电机发送操作。

测量 50 Hz 空推理的 CPU 占用（预热后运行 30 秒）：

```bash
python -m Sim2Real.xz1_him --hz 50 --iterations 1500 \
  --output Sim2Real/output/xz1_him_cpu_50hz.json
```

`--hz 0`（默认）表示不等待，尽可能连续推理。
CPU 占用用进程累计 CPU 时间 / 墙钟时间计算，包含全部进程线程和
测试循环开销，排除模型加载和预热。`process_cpu_percent_one_core`
以一个逻辑核满载为 100%（类似 Linux top），`process_cpu_percent_whole_machine`
再除以整机逻辑核数。50 Hz 模式使用 sleep 等待周期，不忙等；
满速连续推理的 CPU 占用不代表 50 Hz 控制时的占用。

## 部署核心接口

`runtime.py` 提供 `RobotState`、`ObservationBuilder`、`ONNXPolicy`、
`Controller` 和待实现的 `RobotInterface` 协议。当前尚未实现实机通信与控制循环。

- 关节顺序：FL、FR、RR、RL，每条腿 hip、thigh、shank。
- 角度 rad、角速度 rad/s；IMU 角速度在训练机体系，四元数为
  **body-to-world、wxyz**，机体系 x 前、y 左、z 上。
- 一帧 47 维：角速度 3、重力 3、速度指令 3、步态相位 2、
  相对默认关节位置 12、关节速度 12、上一轮裁剪后策略动作 12。
- 六帧按当前帧在前拼接，首次使用当前帧填满历史；相位周期 0.6 s，
  指令范数小于 0.1 时相位观测置零，内部相位仍每轮推进。
- 目标位置 = `action_offset + 0.25 * action`，策略动作裁剪至 ±10。
  观测位置参考 `default_q` 与动作偏置不同，保留训练中的负载补偿。
  ONNX 中已经包含观测归一化，不要在外部重复归一化。
- 配置冻结了本次训练的动作偏置、PD 参数与 MJCF 关节范围；
  ONNX 元数据未包含完整动作偏置，替换其他训练模型时须核对其 env.yaml。
  配置的关节范围来自仿真模型，尚未完成实机标定。
  `Controller` 对超范围目标抛异常，调用方应停止控制并复位历史。

接入硬件时，在适配器完成电机 ID/符号/零位与 IMU 安装方向变换，
实现新鲜状态超时、使能/急停、失联看门狗和电机限制，再接入控制循环。
当前 PD 参数用于保持训练契约，不代表已经通过实机验证。

## 验证

```bash
python -m unittest discover -s Sim2Real/tests -v
```
