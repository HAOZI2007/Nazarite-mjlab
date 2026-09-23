# DELTA 论文复现实施指南（Nazarite / mjlab / Go2）

本文档给出在当前仓库中从零实现 DELTA 的工程路线。目标不是只写出一个
Transformer，而是复现论文的完整闭环：本体感知、高程图、可变形局部注意力、
PPO、稀疏地形课程、仿真评估和 sim-to-real 接口。

> 适用项目：`/home/haozi/桌面/Nazarite-mjlab/Train/Nazarite`。
> 当前仓库已经有 Go2、MuJoCo/mjlab、RSL-RL、RayCast 高度传感器和粗糙地形基础，
> 但目前没有 DELTA 编码器和论文中的 sparse-terrain 任务，需要新增代码。

## 1. 先明确复现目标

建议分三阶段，不要一开始就同时接入真实相机和所有论文地形。

| 阶段 | 目标 | 验收标准 |
|---|---|---|
| P0 | 复现网络前向 | 固定输入下输出形状、数值有限；单元测试通过 |
| P1 | 仿真端到端训练 | 平地、坡地、楼梯、间隙、踏脚石可训练；能达到稳定成功率 |
| P2 | 论文级实验与部署 | 26×16/41×25 对比、消融、TensorRT/真实高程图接口 |

论文的关键基线是 AME，而不是普通 MLP。建议先实现 `MLP / AME / DELTA` 三个
可切换编码器，确保三者使用相同 actor、critic、动作和奖励，只有 terrain encoder
不同。

## 2. 论文规格与项目内规格

论文中最重要的设置如下：

- 本体感知维度：48；
- 地图通道：机器人基座坐标系的 `(x, y, z)`；
- 地图物理范围：`2.0 m × 1.2 m`；
- 标准地图：`26 × 16 × 3`；高分辨率地图：`41 × 25 × 3`；
- DELTA 层数 `L=3`，表示维度 `D=64`；
- 注意力头数 `Nh=4`，每头采样数 `K=8`；
- scout patch：`3×3`；最终 patch：`9×9`；
- patch 步长约 `0.08 m`，偏移尺度 `γ=0.25`；
- actor 隐藏层 `[256, 128]`，critic 隐藏层 `[512, 256, 128]`；
- 关节动作：`q_des = q_nom + 0.1 * action`；
- PD：`Kp=100, Kd=1`，并进行力矩限幅；
- 训练：PPO、200 个并行环境、仿真 400 Hz、策略 100 Hz、episode 最长 4 s。

当前项目中可复用的相关基础：

- Go2 机器人和执行器：
  [`go2_cfg.py`](/home/haozi/桌面/Nazarite-mjlab/Train/Nazarite/Nazarite-src/nazarite/config/robot_config/go2_cfg.py)
- 平地/粗糙地形环境：
  [`go2_env_cfgs.py`](/home/haozi/桌面/Nazarite-mjlab/Train/Nazarite/Nazarite-src/nazarite/config/train_config/env_cfgs/go2_env_cfgs.py)、
  [`go2_rough_env_cfgs.py`](/home/haozi/桌面/Nazarite-mjlab/Train/Nazarite/Nazarite-src/nazarite/config/train_config/env_cfgs/go2_rough_env_cfgs.py)
- 状态观测和 `base_lin_vel`：
  [`mjlab/envs/mdp/observations.py`](/home/haozi/桌面/Nazarite-mjlab/Train/Nazarite/mjlab/src/mjlab/envs/mdp/observations.py)
- RayCast/地形高度传感器：
  [`terrain_height_sensor.py`](/home/haozi/桌面/Nazarite-mjlab/Train/Nazarite/mjlab/src/mjlab/sensor/terrain_height_sensor.py)
- 任务注册入口：
  [`nazarite/__init__.py`](/home/haozi/桌面/Nazarite-mjlab/Train/Nazarite/Nazarite-src/nazarite/__init__.py)

## 3. 推荐新增的文件结构

```text
Nazarite-src/nazarite/
├── delta/
│   ├── __init__.py
│   ├── encoder.py              # DELTA 主网络
│   ├── sampling.py             # 有界坐标、双线性 patch、采样工具
│   ├── tokens.py               # scout/final patch 编码
│   └── diagnostics.py          # 采样可行率、未来落脚注意力统计
├── mdp/
│   └── delta_observations.py   # 地图和论文 48 维 proprioception 组合
├── config/train_config/
│   ├── delta_rl_cfg.py         # actor/critic/runner 配置
│   └── env_cfgs/delta_go2_env_cfgs.py
└── scripts/
    ├── train_delta.py
    └── evaluate_delta.py
```

不要把 DELTA 的网络实现塞进 `rewards.py` 或通用 `observations.py`。这样可以避免
后续和现有 WTW/HIM/FR-Net 任务耦合。

## 4. 第一步：定义统一数据接口

DELTA 的 forward 接口建议固定为：

```python
terrain_repr = encoder(
    proprioception=pt,  # [B, 48]
    elevation_map=Mt,   # [B, H, W, 3]
)
# terrain_repr: [B, 64]
```

地图必须统一为机器人基座坐标系。推荐约定：

- `+x`：机器人前方；
- `+y`：机器人左方；
- `+z`：竖直向上；
- 地图中心固定在机器人基座投影附近；
- 采样坐标使用米，不使用像素索引。

归一化必须和论文一致：

```python
x = x / 1.25
y = y / 0.75
z = torch.clamp(z, -0.8, 0.8) / 0.6
```

本体感知至少应包含：投影重力、基座线/角速度、关节位置、关节速度、上一动作
（或上一期望关节目标）和速度指令。最终维度必须通过日志打印确认是 48；不要凭
感觉拼接，因为动作历史和命令维度在现有项目中可能不同。

## 5. 第二步：实现 DELTA 编码器

### 5.1 初始 query

```python
q = layer_norm(query_mlp(pt))  # [B, 64]
```

### 5.2 每层的状态条件采样

每层、每个 head、每个采样点预测二维偏移。推荐张量布局：

```text
q                         [B, D]
offset                    [B, Nh, K, 2]
raw_reference             [B, Nh, K, 2]
bounded_location          [B, Nh, K, 2]
```

有界函数：

```python
def bound_location(raw, limit):
    return limit * torch.tanh(raw / limit)
```

第一层 reference 从固定基础网格加小随机扰动初始化；后续层沿用上一层的 refined
reference。`tanh` 偏移必须保留，不能让网络直接访问地图外部。

### 5.3 双线性提取 scout patch

使用 `torch.nn.functional.grid_sample` 或项目中已有的等价双线性采样工具。重点是：

- 坐标归一化到 `[-1, 1]`；
- `align_corners` 训练和部署必须一致；
- 越界使用边界值或零值要固定；
- patch 中心采用相对高度，避免把绝对基座高度直接当作地形形状。

scout patch 展平后送入共享 `ScoutMLP`，输出 `D-3=61` 维，再拼接采样中心的
归一化三维坐标，得到 `g ∈ R^64`。

### 5.4 scout-guided refinement

```python
delta = gamma * torch.tanh(refine_mlp(torch.cat([q, scout_token], dim=-1)))
refined_raw = raw_reference + delta
location = bound_location(refined_raw, patch_center_limit)
```

注意：scout 的作用是用局部几何修正采样位置，而不是产生最终注意力 token。应分别
记录初始位置和 refined 位置，便于复现论文中的消融和可视化。

### 5.5 自适应最终 patch

前 `L-1` 层使用小 patch；最后一层使用 `9×9` patch，并拆成中心和上下文：

```python
c = center_mlp(center_patch)       # [B, 61]
u = context_mlp(context_patch)     # [B, 61]
gate = torch.sigmoid(gate_mlp(torch.cat([c, u], dim=-1)))
phi = c + gate * context_projection(u)
token = torch.cat([phi, center_xyz], dim=-1)  # [B, 64]
```

中心/上下文分离不能省略。论文消融显示，去掉自适应 patch 后，已知踏脚石性能仍高，
但未见混合课程成功率会显著下降。

### 5.6 多头交叉注意力和 query 更新

每个 head 只对 `K=8` 个 token 做 attention，而不是对整张地图所有网格做 attention：

```python
q_h = q.view(B, Nh, D // Nh)
k = key_proj(tokens)
v = rms_norm(value_proj(tokens))
attn = softmax((q_h.unsqueeze(-2) * k).sum(-1) / sqrt(d_h), dim=-1)
context = (attn.unsqueeze(-1) * v).sum(-2)
q = layer_norm(q + output_proj(context.reshape(B, D)))
q = layer_norm(q + ffn(q))
```

论文特别使用 value-only RMS normalization，目的是防止低注意力但数值范数很大的
token 主导结果。不要直接把普通 LayerNorm 误当成同一操作。

## 6. 第三步：接入 actor、critic 和动作控制

建议共享一个 DELTA encoder：

```text
actor noisy observation ─┐
                         ├─ shared DELTA ─ actor MLP ─ action distribution
critic clean observation┘       └───────── critic MLP ─ value
```

actor 输入为 `[proprioception, terrain_repr]`。critic 可以使用论文的干净观测，
但必须保证共享 encoder 的梯度和 batch 语义正确。

动作输出继续沿用项目的关节位置动作接口：

```python
q_des = q_nom + 0.1 * action
tau = kp * (q_des - q) - kd * qdot
tau = torch.clamp(tau, -tau_max, tau_max)
```

不要同时重复应用项目已有的 `GO2_ACTION_SCALE=0.25` 和论文 `0.1`。复现论文时应
明确选择一个尺度，并在日志中打印最终 `q_des` 范围。

## 7. 第四步：构建仿真高程图

### 7.1 第一版：直接使用 MuJoCo RayCast

为了快速验证网络，可以不用真实深度相机，先在 MuJoCo 中用规则射线生成高度图：

- 在基座坐标系建立 `H×W` 网格；
- 每个网格点沿竖直方向或指定方向 raycast；
- 得到地面交点 `(x,y,z)`；
- 没有命中时写入 NaN 或显式无效值；
- 在送入网络前按论文规则裁剪、归一化。

当前仓库已经有 RayCast 传感器和 `height_scan`，但论文的 DELTA 输入是二维局部地图的
`(x,y,z)` 三通道，不应直接把现有脚端环形高度扫描当成完整替代。可以新增一个
`DeltaElevationMapSensorCfg`，或先在环境 observation term 中生成 `[B,H,W,3]`。

### 7.2 第二版：模拟真实深度相机

加入以下随机化：深度噪声、NaN、尖峰、遮挡、外参误差、时间延迟、地图更新延迟。
否则 sim-to-real 时，网络可能只适应理想高度图。

### 7.3 真实部署接口

论文使用两个 Intel RealSense D430：前置一个、后置一个，融合成一个机器人中心
高程图。论文未公开精确俯仰角，因此应通过视场覆盖和标定优化，而不是照抄未知角度。

点云流程：

```text
depth camera(s)
  → point cloud
  → camera-to-base extrinsic transform
  → timestamp synchronization
  → 2.0×1.2 m grid aggregation
  → NaN/spike filtering
  → [41,25,3] DELTA input
```

基座线速度由 IMU、关节编码器、接触检测和腿式里程计/EKF 等状态估计器提供，必须
转换到基座坐标系；不能用速度指令替代真实线速度。

## 8. 第五步：稀疏地形与课程

建议先复用 `go2_rough_env_cfgs.py` 的地形生成器，再新增论文风格地形：

- 连续：flat、hills、slope、stairs、steps、rough ground；
- 离散：gaps、stepping stones、grid stones；
- 细粒度踏脚石：半径减半；
- 混合评估：训练中不出现的地形排列和几何变体。

每个地形设置 1–10 难度阶段：连续成功两次升级，连续失败三次降级。动态提高
表现较差地形的采样概率。训练和评估随机种子必须分开保存。

奖励至少包含：速度跟踪、姿态稳定、足端摆动高度、支撑期滑移、非期望接触、
长时间支撑惩罚和失败终止惩罚。论文使用失败终止惩罚 `-50`；实际权重需要在
当前 Go2 执行器和时间步长下重新标定。

## 9. PPO 配置建议

新增 `delta_rl_cfg.py`，先沿用项目现有 runner，再替换 actor/critic model class。
最小配置检查项：

```text
num_envs              200（先用 64/128 做 smoke test）
physics_dt            0.0025 s（400 Hz）
policy_dt             0.01 s（100 Hz）
episode_length        4.0 s
PPO                    gamma=0.99, lambda=0.95, clip=0.2
terrain encoder        L=3, D=64, Nh=4, K=8
```

先用 64 个环境跑 200–500 次迭代确认无 NaN，再扩大到 GPU 能承受的并行数。论文的
200 环境是参考值，不是必须值。

## 10. 实施顺序和每步命令

### Step A：网络单元测试

测试内容：

1. `[B,48] + [B,H,W,3] → [B,64]`；
2. `26×16` 和 `41×25` 输出形状相同；
3. 采样位置始终位于有效 patch 中心范围；
4. 地图全 NaN、全平地、极端高度时输出有限；
5. 反向传播后参数梯度非 NaN。

### Step B：平地端到端 smoke test

```bash
cd /home/haozi/桌面/Nazarite-mjlab/Train/Nazarite
uv sync
uv run ruff check Nazarite-src/nazarite
uv run pyright -p pyproject.toml Nazarite-src/nazarite
uv run train Nazarite-Delta-Go2 --num_envs 64 --max_iterations 500
```

具体 CLI 参数以当前 mjlab runner 支持情况为准；如果 runner 不接受覆盖参数，写入
专用配置函数，不要修改已有 WTW 默认任务。

### Step C：标准地形训练

依次加入 flat → rough/slope → stairs/steps → gaps → stepping stones。每加入一种
地形，只改变地形分布和对应奖励，不同时修改网络、动作尺度和 PPO 学习率。

### Step D：AME/DELTA 对比

固定：机器人、随机化、奖励、PPO、训练迭代、随机种子和评估 episode 数量。只替换
encoder，并分别记录：成功率、达到 90% SR 的迭代数、单次迭代时间、encoder FLOPs。

### Step E：高分辨率和混合课程

先在 `26×16` 验证，再切换 `41×25`。检查 DELTA encoder FLOPs 是否保持基本不变；
检查 AME FLOPs 是否随候选数增大。之后加入训练未见过的混合地形。

### Step F：真实地图和部署

先离线录制深度数据，使用同一套地图构建器回放，确认网络输出稳定，再接入实时相机。
最后才导出 TorchScript/ONNX/TensorRT，并在 Jetson 上验证 100 Hz 延迟。

## 11. 必须记录的日志和可视化

除了普通 reward 和成功率，建议每个 episode 记录：

- 初始采样位置和 refined 采样位置；
- 最终层采样点落在可踩表面的比例；
- attention 权重分布；
- 未来 100 ms 内落脚点位置的平均注意力；
- 地图有效率和 NaN 比例；
- actor/critic 输入均值、标准差和最大值；
- encoder forward 时间和总 policy 时间；
- 每种地形、每个难度阶段的成功率。

这些指标可以直接支持论文中的表 II–VI 风格复现，并能快速定位“地图坏了”、
“采样偏移发散”或“策略只会记住某一种地形”等问题。

## 12. 常见失败原因

### 12.1 维度或坐标系错误

症状：平地也无法站稳、attention 全部集中在边界、训练很快 NaN。检查地图布局、
`grid_sample` 坐标、基座坐标系、速度坐标系和归一化常数。

### 12.2 重复动作缩放

症状：腿部动作过小或力矩饱和。检查 `GO2_ACTION_SCALE` 与论文动作尺度是否重复。

### 12.3 地图时间不同步

症状：仿真成功、真实部署时脚总踩在地图边缘。检查相机时间戳、状态估计延迟和
机器人位姿补偿。

### 12.4 只训练一种地形

症状：单一踏脚石成功率高，混合地形失败。必须使用地形组合、未见几何和动态课程，
并保留 adaptive patch 与 scout refinement。

### 12.5 直接把 height scan 当作 DELTA 地图

脚端高度扫描可用于奖励，但不能自动等价为论文的机器人中心 `(x,y,z)` 局部地图。
二者的空间布局、通道数和坐标定义不同。

## 13. 最低可交付版本（MVP）

如果希望尽快看到结果，按以下最小版本执行：

1. MuJoCo RayCast 生成 `26×16×3` 地图；
2. 只实现 `L=3, D=64, Nh=4, K=8` 的 DELTA；
3. 暂时不做真实相机；
4. 使用平地、坡地、楼梯、间隙、踏脚石五类地形；
5. 使用现有 Go2 关节位置动作和 PD 控制；
6. 用 PPO 先跑通 64 环境；
7. 再扩展到 200 环境、41×25、混合课程和真实地图。

完成 MVP 后，再做 AME 对照和消融。这样可以把“网络实现错误”和“奖励/课程不合适”
分开诊断。

## 14. 最终验收标准

达到下面条件，才算完成论文级复现：

- DELTA、AME、MLP 三个 encoder 可通过配置切换；
- 标准分辨率下 DELTA 最终成功率接近 AME；
- 高分辨率下 DELTA encoder 计算量基本不变；
- 细粒度踏脚石上高分辨率 DELTA 明显优于低分辨率；
- 未见混合课程上 DELTA 明显优于 AME；
- 去掉 scout、adaptive patch、deformable sampling 后性能按预期下降；
- 能输出采样位置和注意力可视化；
- 深度相机离线回放和仿真地图接口一致；
- actor 的线速度来自状态估计，且坐标系、频率、延迟与训练一致；
- 导出模型在目标计算平台满足策略频率要求。

论文结果不应被当作无需调参的保证。最需要针对你的机器人重新标定的部分是：
动作尺度、PD 增益、脚端/接触奖励、地图有效范围、障碍高度、相机外参和状态估计噪声。

---

# 15. 代码级实现（从这里开始照着写）

前面的章节说明“做什么”；本章说明“代码应该怎样组织”。以下代码是可运行的
PyTorch 设计骨架，省略的只有项目特定 import 和配置胶水。建议先把它复制到
`Nazarite-src/nazarite/delta/`，让单元测试通过，再接入 PPO。

## 15.1 先理解 RSL-RL 的数据流

当前 RSL-RL 的 `PPO.construct_algorithm()` 会根据配置中的 `class_name` 实例化
actor/critic，并把环境 observation 作为 `TensorDict` 传进去。普通 `MLPModel`
只会把 observation group 拼成一个向量；DELTA 需要同时拿到一个 `[B,48]` 的
proprioception 和一个 `[B,H,W,3]` 的 map，因此不要把 map 先 flatten 后再试图
恢复空间结构。

推荐增加两个 observation term，并关闭该 group 的自动拼接：

```python
# delta_go2_env_cfgs.py（示意）
actor_terms = {
    "delta_proprio": ObservationTermCfg(func=delta_proprioception),
    "delta_map": ObservationTermCfg(func=delta_elevation_map),
}
cfg.observations["actor_delta"] = ObservationGroupCfg(
    terms=actor_terms,
    concatenate_terms=False,
    enable_corruption=True,
)
```

实际使用的 key 取决于 observation manager 的输出命名。第一次运行时必须打印：

```python
print(obs.keys())
print(obs["delta_proprio"].shape)  # [B, 48]
print(obs["delta_map"].shape)     # [B, H, W, 3]
```

如果当前 manager 不支持非拼接 group，最稳妥的改法是新增一个
`DeltaObservationGroup`，只在模型内部拆分；不要用 `view(B,H,W,3)` 猜测，因为
term 的排列和历史帧可能已经改变。

## 15.2 DELTA 的最小类设计

建议把网络拆成四个小类，便于单测和消融：

```text
DeltaEncoder
 ├─ QueryMLP
 ├─ DeltaAttentionLayer × L
 │   ├─ StateOffsetHead
 │   ├─ ScoutMLP
 │   ├─ RefineMLP
 │   ├─ PatchEncoder
 │   ├─ MultiHeadCrossAttention
 │   └─ QueryUpdate(Residual + LN + FFN)
 └─ diagnostics（可选，不参与 forward 输出）
```

所有张量约定如下（`B` 是并行环境数）：

| 名称 | shape | 含义 |
|---|---|---|
| `pt` | `[B,48]` | 本体感知 |
| `Mt` | `[B,H,W,3]` | 基座坐标系高程图 |
| `q` | `[B,64]` | 当前 query |
| `raw_ref` | `[B,4,8,2]` | 未约束采样坐标（米） |
| `loc` | `[B,4,8,2]` | 约束后的采样坐标（米） |
| `scout` | `[B,4,8,64]` | scout token（61 维特征+3 维中心坐标） |
| `token` | `[B,4,8,64]` | terrain evidence token |
| `attn` | `[B,4,8]` | 每个 head 的注意力 |
| `et` | `[B,64]` | DELTA 输出 |

## 15.3 地图米制坐标到 `grid_sample` 坐标

这是最容易写错的部分。假设地图中心为 `(0,0)`，x 范围为
`[-x_limit,+x_limit]`，y 范围为 `[-y_limit,+y_limit]`，则米制位置转为
`grid_sample` 的 `[-1,1]` 坐标：

```python
def meters_to_grid(location_xy, x_limit, y_limit):
    # location_xy: [B, Nh, K, 2], 最后一维是 (x, y)
    x = location_xy[..., 0] / x_limit
    y = location_xy[..., 1] / y_limit
    # grid_sample 的最后一维是 (column, row)，row 正方向向下。
    return torch.stack((x, -y), dim=-1).clamp(-1.0, 1.0)
```

如果你的地图 tensor 第一维对应 y、第二维对应 x，上式适用；如果栅格建立时
采用了 `[x,y]` 排列，必须交换两个分量。用一个平面测试验证：给地图设置
`z=x+2y`，在 `(0.1,0.2)` 采样，结果应接近 `0.5`（按你的归一化规则换算）。

patch 的网格坐标应由中心坐标加固定米制步长生成，而不是在像素上固定步长：

```python
def patch_grid(center_xy, size, step, x_limit, y_limit):
    # center_xy: [B, Nh, K, 2]
    radius = (size - 1) / 2
    axis = torch.arange(-radius, radius + 1, device=center_xy.device) * step
    dy, dx = torch.meshgrid(axis, axis, indexing="ij")
    offsets = torch.stack((dx, dy), dim=-1)                 # [P,P,2]
    xy = center_xy.unsqueeze(-2).unsqueeze(-2) + offsets   # [B,Nh,K,P,P,2]
    return meters_to_grid(xy, x_limit, y_limit)
```

## 15.4 双线性采样函数

`grid_sample` 要求输入为 `[N,C,H,W]`、grid 为 `[N,out_h,out_w,2]`。下面的函数
把 `B×Nh×K` 个 patch 展平为 batch，采样后再恢复形状：

```python
import torch
import torch.nn.functional as F

def bilinear_patches(map_bhwc, centers, size, step, x_limit, y_limit):
    B, H, W, C = map_bhwc.shape
    _, Nh, K, _ = centers.shape
    grid = patch_grid(centers, size, step, x_limit, y_limit)
    # [B,Nh,K,P,P,2] -> [B*Nh*K,P,P,2]
    grid = grid.reshape(B * Nh * K, size, size, 2)
    image = map_bhwc.permute(0, 3, 1, 2)
    image = image[:, None].expand(B, Nh * K, C, H, W)
    image = image.reshape(B * Nh * K, C, H, W)
    patch = F.grid_sample(
        image, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    return patch.reshape(B, Nh, K, C, size, size)
```

训练和部署必须固定 `align_corners=True`、padding 规则和坐标方向。NaN 不能直接
送进 `grid_sample`；在采样前用有效值或 0 替换，并额外提供一个 valid mask，避免
网络把“缺失”误认为真实的平面高度。

## 15.5 可直接实现的 `DeltaAttentionLayer`

下面是核心 forward 的伪代码，变量名与论文公式一一对应：

```python
class DeltaAttentionLayer(nn.Module):
    def __init__(self, dim=64, heads=4, samples=8,
                 scout_size=3, final_size=9, final=False):
        super().__init__()
        self.dim, self.heads, self.samples = dim, heads, samples
        self.final = final
        self.head_dim = dim // heads
        assert dim % heads == 0
        self.offset = nn.Linear(dim, heads * samples * 2)
        self.scout_mlp = nn.Sequential(nn.Linear(3*scout_size*scout_size, 64),
                                       nn.ELU(), nn.Linear(64, dim-3))
        self.refine = nn.Sequential(nn.Linear(2*dim, 64), nn.ELU(),
                                    nn.Linear(64, heads*samples*2))
        self.center_mlp = nn.Sequential(nn.Linear(final_size*final_size, 128),
                                         nn.ELU(), nn.Linear(128, 61))
        self.context_mlp = nn.Sequential(nn.Linear(final_size*final_size, 128),
                                          nn.ELU(), nn.Linear(128, 61))
        self.context_proj = nn.Linear(61, 61)
        self.gate = nn.Linear(122, 1)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 256), nn.ELU(),
                                 nn.Linear(256, dim))

    def forward(self, q, terrain, raw_ref, *, x_limit, y_limit, step, gamma):
        B, D = q.shape
        Nh, K = self.heads, self.samples
        # 1) state-conditioned coarse offset
        d0 = gamma * torch.tanh(self.offset(q)).view(B, Nh, K, 2)
        scout_center = bound_location(raw_ref + d0, (x_limit, y_limit))
        scout_patch = bilinear_patches(terrain, scout_center, 3, step,
                                       x_limit, y_limit)
        scout_flat = scout_patch.flatten(start_dim=3)  # [B,Nh,K,3*3*3]
        scout_feat = self.scout_mlp(scout_flat)
        xyz = torch.cat((scout_center, torch.zeros_like(scout_center[..., :1])), -1)
        scout_token = torch.cat((scout_feat, xyz), -1)
        # 2) scout-guided refinement
        q_expand = q[:, None, None].expand(B, Nh, K, D)
        dr = gamma * torch.tanh(self.refine(torch.cat((q_expand, scout_token), -1)))
        refined = bound_location(raw_ref + d0 + dr, (x_limit, y_limit))
        # 3) local evidence token
        size = 9 if self.final else 3
        patch = bilinear_patches(terrain, refined, size, step,
                                 x_limit, y_limit)
        # 这里的 center/context mask 应按 patch 索引预先注册为 buffer
        center, context = split_center_context(patch, final=self.final)
        c = self.center_mlp(center.flatten(start_dim=3))
        u = self.context_mlp(context.flatten(start_dim=3))
        eta = torch.sigmoid(self.gate(torch.cat((c, u), -1)))
        phi = c + eta * self.context_proj(u)
        token = torch.cat((phi, normalized_xyz(refined, x_limit, y_limit)), -1)
        # 4) single-query multi-head cross attention
        qh = q.view(B, Nh, self.head_dim)
        tk = self.key(token).view(B, Nh, K, self.head_dim)
        tv = rms_norm(self.value(token).view(B, Nh, K, self.head_dim))
        score = (qh[:, :, None] * tk).sum(-1) / (self.head_dim ** 0.5)
        attn = score.softmax(-1)
        context = (attn[..., None] * tv).sum(-2).reshape(B, D)
        q = self.norm1(q + self.out(context))
        q = self.norm2(q + self.ffn(q))
        return q, refined, attn
```

实现时先不要追求完全优化。先让它在 batch=2、`26×16` 地图上运行；确认输出
和梯度都正确后，再把 `B×Nh×K` 展平、预分配 patch offset、使用 `torch.compile`
或 TensorRT 优化。

## 15.6 `DeltaEncoder` 完整 forward

```python
class DeltaEncoder(nn.Module):
    def __init__(self, proprio_dim=48, dim=64, layers=3,
                 heads=4, samples=8, map_extent=(2.0, 1.2)):
        super().__init__()
        self.dim, self.heads, self.samples = dim, heads, samples
        self.x_limit, self.y_limit = map_extent[0] / 2, map_extent[1] / 2
        self.query = nn.Sequential(nn.Linear(proprio_dim, 64), nn.ELU(),
                                   nn.Linear(64, dim), nn.LayerNorm(dim))
        self.layers = nn.ModuleList([
            DeltaAttentionLayer(dim, heads, samples,
                final=(i == layers-1)) for i in range(layers)
        ])
        # [Nh,K,2]：固定基础网格；建议注册为 buffer
        base = make_base_reference_grid(heads, samples, self.x_limit, self.y_limit)
        self.register_buffer("base_reference", base)

    def forward(self, proprio, elevation_map, return_aux=False):
        q = self.query(proprio)
        B = proprio.shape[0]
        raw_ref = self.base_reference[None].expand(B, -1, -1, -1).clone()
        aux = []
        for layer in self.layers:
            q, raw_ref, attn = layer(
                q, elevation_map, raw_ref,
                x_limit=self.x_limit, y_limit=self.y_limit,
                step=0.08, gamma=0.25,
            )
            if return_aux:
                aux.append({"locations": raw_ref, "attention": attn})
        return (q, aux) if return_aux else q
```

这里的 `base_reference` 是 `[Nh,K,2]` 而不是每次 forward 随机生成。论文中的
小随机扰动应在初始化时加入并固定；否则 rollout 和 PPO update 对同一 observation
会看到不同采样位置，训练噪声会明显增大。

## 15.7 如何接入 RSL-RL actor/critic

不要修改通用 `MLPModel`。新增一个继承它的类，例如：

```python
class DeltaActorModel(MLPModel):
    def __init__(self, obs, obs_groups, obs_set, output_dim, **kwargs):
        # 先初始化 distribution，复用 MLPModel 的 Gaussian 行为；
        # 实际项目中可复制 MLPModel.__init__ 的 distribution 部分。
        super().__init__(obs, obs_groups, obs_set, output_dim,
                         hidden_dims=kwargs.pop("hidden_dims", (256,128)),
                         distribution_cfg=kwargs.pop("distribution_cfg", None))
        self.delta = DeltaEncoder(proprio_dim=48)
        self.policy_mlp = MLP(64 + 48, self.mlp.output_dim, (256,128), "elu")

    def get_latent(self, obs, masks=None, hidden_state=None):
        pt = obs["delta_proprio"]
        Mt = obs["delta_map"]
        et = self.delta(pt, Mt)
        return torch.cat((pt, et), dim=-1)
```

真实代码需要注意一个细节：父类 `MLPModel.forward()` 会调用 `self.mlp`，所以有
两种实现选择：

1. 最简单：继承 `MLPModel`，在 `__init__` 后用正确输入维度重建 `self.mlp`；
2. 更清晰：复制 `MLPModel` 的 distribution/forward 逻辑，写一个独立
   `DeltaModel`，避免父类先按 flatten observation 计算错误维度。

推荐第二种。critic 使用完全相同的 `DeltaEncoder` 结构，但实例是否共享参数要
明确决定：论文描述 actor/critic 共享 DELTA 参数，工程上可在构造 runner 时把同一
个 module 注入两边；如果 RSL-RL 的 actor、critic 是独立构造，则先实现“不共享”
版本跑通，再修改 runner 让两者引用同一个 encoder，并确认 optimizer 不重复注册
同一参数。

配置中使用完整限定名，避免 resolver 找不到自定义类：

```python
actor=RslRlModelCfg(
    class_name="nazarite.delta.rsl_model:DeltaActorModel",
    hidden_dims=(256, 128),
    distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0,
                      "std_type": "log"},
)
```

如果 runner 的 observation group 仍要求 `("actor",)`，将 `delta_proprio`、
`delta_map` 放入 actor group，并在自定义 model 中按 key 读取。配置中的
`obs_dim` 只用于初始化检查，不应再把 map 当作一维向量送进普通 MLP。

## 15.8 地图 observation term 的代码逻辑

仿真版最小实现可以直接从 RayCast 结果构建地图：

```python
def delta_elevation_map(env, sensor_name="delta_terrain_scan"):
    sensor = env.scene[sensor_name]
    # hit_pos_b 应为 [B,H,W,3]；若传感器返回 [B,N,3]，按固定网格 reshape
    xyz = sensor.data.hit_pos_b
    xyz = xyz.reshape(env.num_envs, 26, 16, 3)
    xyz[..., 0] /= 1.25
    xyz[..., 1] /= 0.75
    xyz[..., 2] = xyz[..., 2].clamp(-0.8, 0.8) / 0.6
    return torch.nan_to_num(xyz, nan=0.0, posinf=0.0, neginf=0.0)
```

如果现有 RayCast 只暴露高度标量而没有交点坐标，就按已知网格中心补回 x、y：

```python
xy = make_robot_centered_grid(B, H=26, W=16, extent=(2.0,1.2), device=env.device)
z = sensor.data.dist.reshape(B, H, W)
xyz = torch.cat((xy.expand(B,-1,-1,-1), z[...,None]), dim=-1)
```

这里的 `dist` 是否等于高度必须通过平地单元测试确认；射线方向不是竖直时，
距离不能直接当作 z。

## 15.9 单元测试：按这个顺序写

```python
def test_delta_shapes():
    enc = DeltaEncoder()
    pt = torch.randn(2, 48)
    m1 = torch.randn(2, 26, 16, 3)
    m2 = torch.randn(2, 41, 25, 3)
    assert enc(pt, m1).shape == (2, 64)
    assert enc(pt, m2).shape == (2, 64)

def test_delta_finite_and_grad():
    enc = DeltaEncoder()
    pt = torch.randn(2, 48, requires_grad=True)
    m = torch.randn(2, 26, 16, 3, requires_grad=True)
    y = enc(pt, m)
    y.square().mean().backward()
    assert torch.isfinite(y).all()
    assert torch.isfinite(pt.grad).all()

def test_sampling_stays_inside_patch_center():
    enc = DeltaEncoder()
    _, aux = enc(torch.randn(4,48), torch.randn(4,26,16,3), return_aux=True)
    for item in aux:
        loc = item["locations"]
        assert loc[...,0].abs().max() <= 1.0
        assert loc[...,1].abs().max() <= 0.6
```

再补两个有意义的几何测试：

- 平面地图：所有 patch token 在平移采样中心后应几乎不变（中心坐标 token 除外）；
- 单个高台：scout/refined location 应比初始 reference 更靠近高台（训练后再测，
  随机初始化阶段不能要求它已经学会）。

## 15.10 训练链路的实际接线顺序

按下面的顺序提交代码，每一步都能独立运行：

1. `delta/sampling.py`：米制坐标、bound、patch grid、`grid_sample`；
2. `delta/tokens.py`：center/context split 和 gate；
3. `delta/encoder.py`：先关闭 refinement，只用固定 reference 跑通 forward；
4. 加入 query-conditioned offset；
5. 加入 scout patch 和 refine offset；
6. 加入 cross-attention 和 FFN；
7. 新增 `delta_proprio`、`delta_map` observation term；
8. 新增 `DeltaActorModel`，只在平地训练 100–500 iterations；
9. critic 接入并确认 PPO 的 value loss 正常下降；
10. 加入 flat/slope/stairs，再加入 gaps/stepping stones；
11. 加入 `return_aux=True` 的离线诊断，不要在正式训练每步保存所有 patch；
12. 最后实现 AME 对照、消融和高分辨率地图。

## 15.11 你在当前项目中需要实际新增/修改什么

最小文件清单如下：

```text
新增：
Nazarite-src/nazarite/delta/__init__.py
Nazarite-src/nazarite/delta/encoder.py
Nazarite-src/nazarite/delta/sampling.py
Nazarite-src/nazarite/delta/tokens.py
Nazarite-src/nazarite/delta/rsl_model.py
Nazarite-src/nazarite/mdp/delta_observations.py
Nazarite-src/nazarite/config/train_config/delta_rl_cfg.py
Nazarite-src/nazarite/config/train_config/env_cfgs/delta_go2_env_cfgs.py
Nazarite/mjlab/tests/test_delta_encoder.py

修改：
Nazarite-src/nazarite/__init__.py       # 注册 Nazarite-Delta-Go2
```

第一版不要修改已有的 `go2_env_cfgs.py`、WTW 或 FR-Net 配置；复制并派生环境配置，
这样 DELTA 失败时不会污染当前主线任务。等 DELTA 任务稳定后，再考虑提取公共的
高程图 sensor 和奖励函数。

## 15.12 首次运行时应该看到什么

启动时打印并保存以下信息：

```text
DeltaActorModel input: proprio=[B,48], map=[B,26,16,3]
DeltaEncoder output: [B,64]
actor latent: [B,112]
action output: [B,12]
map valid ratio: ...
delta sample range x/y: ...
```

平地 smoke test 的合理现象是：前几百次迭代成功率可能很低，但不能出现 NaN；
采样点可以暂时随机，随着训练应逐渐集中到有效地形。若 actor latent 不是 112、
action 不是 12，先不要调奖励或学习率，优先修正 observation/model 接线。

## 15.13 当前仓库已经落地的第一版

当前实现已经新增：

```text
Nazarite-src/nazarite/delta/sampling.py
Nazarite-src/nazarite/delta/encoder.py
Nazarite-src/nazarite/delta/rsl_model.py
Nazarite-src/nazarite/mdp/delta_observations.py
Nazarite-src/nazarite/config/train_config/delta_rl_cfg.py
Nazarite-src/nazarite/config/train_config/env_cfgs/delta_go2_env_cfgs.py
mjlab/tests/test_delta_encoder.py
```

并注册了任务：

```text
Nazarite-Delta-Go2
```

该任务的设计是：从现有 Go2 actor observation 中删除 `base_lin_vel`，因此当前
本体感知为 45 维；新增 `delta_map`，使用 MuJoCo RayCast 生成 `26×16×3` 地图，
并把地图作为 actor group 的最后一项。`DeltaModel` 再从拼接向量中切出前 45 维
和最后 `26×16×3` 维，恢复地图空间结构后送入 `DeltaEncoder`。

首次运行建议：

```bash
cd /home/haozi/桌面/Nazarite-mjlab/Train/Nazarite
uv run list-envs
uv run train Nazarite-Delta-Go2
```

如果环境中还没有 `uv`，先按项目现有方式安装/激活 `.venv`。当前代码已经通过
Python 语法编译检查；运行训练前还需要在装有项目依赖的环境中运行
`pytest mjlab/tests/test_delta_encoder.py`，确认 PyTorch、mjlab 和 MuJoCo 版本
一致。第一版故意没有修改原有 WTW/HIM/FR-Net 任务。
