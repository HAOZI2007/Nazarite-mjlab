# WTW+DELTA 参考 AME 的改进实施计划

## 1. 目标与范围

本计划针对当前 WTW+DELTA Direct 任务，目标是在不修改原始 WTW 任务的前提下，逐步获得“平地保持 WTW trot、复杂地形使用深度图调整步态”的策略。AME 仓库只作为实现参考，不直接复制其代码或修改其仓库。

当前任务采用冻结 WTW 策略加 DELTA 动作残差：

```text
WTW proprioception -> frozen WTW MLP -> prior action
camera BEV map + DELTA proprioception -> deformable attention -> residual action
prior action + bounded residual -> final action
```

这个结构能保护已有的平地运动能力，但也带来两个限制：残差幅度过小时无法改变越障行为；如果残差没有 gait behavior/phase，就无法知道当前哪条腿处于摆动相，也难以调整抬腿高度和落脚位置。因此改进采用“先修复输入和作用强度，再做 AME 风格两阶段训练，最后视结果升级融合结构”的顺序。

## 2. AME 对当前项目的可复用结论

AME 的视觉输入本质上是规则的局部高程图，而不是直接把原始 D435 深度像素送进策略。其典型配置为 `33 x 21` 的局部地图、约 `5 cm` 网格间距，并将地图坐标和高程一起编码。当前项目已经有 D435 深度图到 BEV 的投影、有效性通道和 DELTA deformable sampling，这部分可以继续保留，但需要监控稀疏投影造成的空洞率。

AME 使用本体特征作为 query、地形 CNN 特征作为 key/value，并进行端到端训练；当前项目的残差方案是冻结 WTW 直接叠加动作。因此当前阶段不应立即解冻全部 WTW，而应先让 DELTA 获得足够的动作预算，确认视觉分支确实改善越障后，再考虑解冻最后一层或改为 latent fusion。

AME 还采用明显的两阶段流程：第一阶段在混合粗糙地形上学习通用视觉 locomotion；第二阶段固定为目标地形、缩小速度和随机化范围，专门提升目标地形成功率。当前项目的课程应沿用这一思想，而不是一开始同时打开最大地形难度、全速度范围和全部随机化。

## 3. 分阶段实施路线

### 阶段 0：可观测性与基线

每次训练都记录以下指标，并按地形类型分别观察：

- `DELTA/residual_to_prior_ratio`：残差 RMS 与 WTW prior RMS 的比例。
- `DELTA/residual_gate_mean`、`DELTA/residual_gate_open_ratio`：门控是否长期关闭。
- `DELTA/map_valid_ratio`、`DELTA/attention_valid_ratio`：D435 投影后的地图是否有足够有效点。
- `DELTA/attention_entropy`、`DELTA/attention_x_mean`：注意力是否塌缩到单点或地图外区域。
- 速度跟踪误差、足端/小腿碰撞、摔倒率、episode 成功距离。

判据：如果 `residual_to_prior_ratio` 长期低于约 `0.05`，先不要调地形；如果 `map_valid_ratio` 长期很低，先修相机投影和空洞填充；如果残差有幅度但碰撞不下降，再检查奖励冲突和课程难度。

### 阶段 1：最小结构修复（本次执行）

1. DELTA proprioception 加入 WTW `behavior` 和四足 `phase`。这样 DELTA 能使用步频、目标机体高度、目标摆腿高度、步态相位等信息，输出相位相关的残差。
2. 将残差幅度从 `0.02 -> 0.08` 调整为约 `0.05 -> 0.25`，在前 `1500` 次 PPO iteration 内线性增加；将门控 bias 从 `-2.0` 调整为 `-0.5`，避免初始门控几乎完全关闭。
3. 学习率从 `1e-4` 提高到 `2e-4`，保持 WTW prior 冻结，先只训练 DELTA encoder、residual head 和 gate。
4. 只在 WTW+DELTA 的深拷贝配置中降低 body-height、foot-clearance、Raibert 落点约束的权重，保留相位接触奖励和速度跟踪，避免复杂地形被平地几何目标锁死。原始 WTW 配置不变。
5. 用单元测试验证新增 behavior/phase 输入、4 通道 BEV 地图和残差边界。

阶段 1 完成标准：平地速度跟踪不明显退化；残差门控平均值不再接近 0；复杂地形上的小腿碰撞和成功距离相对基线改善。

### 阶段 2：AME 风格两阶段课程

**Stage 1 通用训练**

- 混合平地、缓坡、随机网格、低难度 gaps、低难度 stepping stones。
- 速度覆盖 `x/y/yaw`，但先限制最大速度和地形高度差。
- 保留适度传感器噪声、摩擦、质量和推扰动随机化。
- 初始化为稳定 WTW checkpoint，训练 DELTA 分支和残差。

**Stage 2 目标地形微调**

- 只保留目标地形，例如 stepping stones 或 zebra gaps。
- 速度先使用正向 `x=0.3~0.6 m/s`，`y=0`、`yaw=0`，再逐步放宽。
- 降低与目标地形无关的随机化，减少观测噪声，稳定视觉输入分布。
- 使用更细的地形难度课程，按成功距离和摔倒率升级。

### 阶段 3：从动作残差升级为 latent fusion

如果阶段 1/2 证明视觉分支有效，但动作残差仍无法改变机体高度、步频或整体 gait，则改为：

```text
WTW proprio latent + DELTA terrain latent
                  -> fusion MLP / attention
                  -> action head
```

用 WTW checkpoint 初始化 proprio encoder 和 action head，先冻结 WTW 主体，只训练融合层；稳定后仅解冻 action head 的最后一层。该方案比直接相加动作更接近 AME 的端到端策略，也能让 DELTA 影响整套动作输出，而不只是每个关节的局部修正。

## 4. 不建议现在做的事情

- 不要直接解冻全部 WTW 网络，否则已有 trot 很容易被视觉分支破坏。
- 不要同时提高地形难度、速度范围、推扰动和相机噪声；否则无法判断失败来源。
- 不要把原始 WTW 的奖励和配置全局改掉；所有 Direct 实验参数应只放在 `wtw_delta_direct_env_cfg.py` 和 `wtw_delta_direct_rl_cfg.py`。
- 不要仅凭 attention 可视化判断策略学会越障；必须结合残差比例、地图有效率、碰撞和成功距离。

## 5. 每轮实验的验收表

| 项目 | 必查指标 | 通过条件 |
| --- | --- | --- |
| 平地 locomotion | 速度误差、机体高度、摔倒率 | 不明显低于 WTW checkpoint |
| 视觉输入 | map valid ratio、空洞率 | 有效地图稳定，不能长期接近 0 |
| DELTA 作用 | residual/prior、gate open ratio | 残差有可控幅度且不是长期饱和 |
| 越障能力 | 小腿碰撞、足端碰撞、成功距离 | 相对当前基线改善 |
| 泛化 | 不同速度、不同地形等级 | 不只在单一固定地图有效 |
| 部署准备 | 相机内参、外参、延迟、深度范围 | 仿真与 D435 的投影约定一致 |

本次先执行阶段 1；阶段 2 需要阶段 1 的日志确认残差确实在工作后再开始，避免盲目增加训练成本。

## 6. 阶段 1 日志后的实际改进（本次已实现）

`2026-09-22_12-12-53` 的训练表明，残差已经达到 prior 动作约 14%，但地图有效率约 25%，stepping stones 和 gaps 课程仍停留在 0 级。因此本次修改集中解决课程过难和视觉输入稀疏问题，未继续增大残差网络。

### 6.1 速度与 episode

WTW+DELTA 现在使用四段全局控制步数课程：

| 控制步数 | x 速度 | y 速度 | yaw 速度 |
| ---: | --- | --- | --- |
| 0 | `[-0.30, 0.30]` | `[0, 0]` | `[-0.15, 0.15]` |
| 72000 | `[-0.50, 0.50]` | `[0, 0]` | `[-0.25, 0.25]` |
| 192000 | `[-0.80, 0.80]` | `[-0.35, 0.35]` | `[-0.35, 0.35]` |
| 360000 | `[-1.00, 1.00]` | `[-0.50, 0.50]` | `[-0.50, 0.50]` |

`stepping_stones` 仍然覆盖为正向 x、零 y/yaw；其他地形在后续阶段保留正负速度。episode 已恢复为 `10 s`。

### 6.2 稀疏地形初始难度

本次 WTW+DELTA 深拷贝配置中的初始稀疏地形改为：

- stepping stones：石块 `0.80~1.00 m`、间距 `0.05~0.12 m`、高度 `0.06 m`。
- zebra gaps：支撑条 `0.75~0.95 m`、gap `0.04~0.10 m`。

难度仍由 terrain level 递增，原始 DELTA 和 WTW 任务不受影响。

### 6.3 BEV 置信度与奖励门控

BEV 范围从 `3.0 x 2.0 m` 收缩为机器人前方 `2.5 x 1.6 m`，保持 `16 x 26 x 5` 的网络输入尺寸，提高每个网格获得真实 D435 点的概率。五个通道依次为 `x/y/z/support/confidence`：`support` 表示该格是否有真实深度支撑，`confidence` 表示投影/局部填充的可信度。可视化坐标同步使用新范围。

DELTA residual 新增地图置信度：

```text
confidence = clamp(map_valid_ratio / 0.45, 0.50, 1.0)
effective_residual = residual * confidence
```

地图有效率低时仍保留梯度，但不会让稀疏或错误的深度图施加完整动作修正。`delta_swing_clearance` 只有在地图有效率至少 `0.35` 且检测到至少 `0.025` 的 obstacle relief 时才启用，平地或空洞图不再持续惩罚普通摆腿。

### 6.4 下一轮验收指标

新训练不应只看总 reward，需要同时检查：

- `DELTA/map_valid_ratio >= 0.40`
- `DELTA/attention_valid_ratio >= 0.70`
- `DELTA/residual_saturation_ratio < 0.20`
- `Episode_Termination/illegal_contact < 0.15`
- `mean_episode_length` 达到 episode 最大控制步数的 80% 以上
- stepping stones 至少达到 1 级，gaps 至少开始脱离 0 级

只有这些指标同时改善，才进入 AME 风格的目标地形微调阶段。

## 7. 历史 teacher-student 方案（已移除）

仅增大动作 residual 不能解决 stepping stones 的落脚时序问题，因此本轮新增了一个与旧任务隔离的 privileged teacher/depth student 结构：

```text
训练 rollout：本体 + privileged 向下 raycast 地图 -> teacher DELTA -> WTW prior + residual -> action
辅助训练：本体 + 4 帧 D435 BEV -> DELTA + GRU -> student latent
                                             └─ latent/action distillation -> teacher latent/action
play/eval：本体 + 4 帧 D435 BEV -> student -> 同一个 residual action head
```

实现位置：

- 历史版本曾包含 teacher、视觉历史 GRU、动作合成和蒸馏损失；相关实现已删除，不再作为当前任务的训练链路。
- `wtw_delta_env_cfg.py`：新增 `delta_privileged_scan`、`delta_privileged_map` 和 `delta_map_history`，只作用于 WTW+DELTA。
- 当前使用 `wtw_delta_direct_rl_cfg.py`，WTW checkpoint 仍冻结加载；旧 residual checkpoint 不可直接续训 Direct 结构。

actor 的输入组不包含 `delta_privileged_map`；它只存在于训练时的完整 TensorDict 中供 teacher 使用，导出/部署的 actor 输入仍然是 `wtw_proprio + delta_proprio + delta_map + delta_map_history`，因此真实机器人端不需要伪造 privileged raycast 地图。

privileged map 使用 `5.0 x 1.5 m`、`0.1 m` 分辨率的向下 raycast，再裁剪到前方 `2.5 x 1.6 m` 的 `16 x 26 x 5` 地图。向下 raycast 的最大距离设为 `1.5 m`，避免深坑底部被错误当成可支撑地面。训练时 teacher 看到该地图，部署时只保留真实 D435 深度历史，因此不会把仿真特权信息带到机器人上。

### 7.1 本轮验收结果

- 5 通道 `DeltaEncoder`：原有 3/4 通道测试与新 5 通道路径均通过。
- WTW+DELTA 环境配置构造通过，传感器、观测组、地形比例和 10 秒 episode 均能实例化。
- CPU smoke test 已构造 `VelocityOnPolicyRunner` 和 Direct actor，训练模式和 eval/play 模式 action 均为有限值。
- CPU smoke test 的 Warp 编译需要将 kernel cache 指向可写目录；这不是任务代码错误，GPU 训练时使用默认 Warp cache 即可。

### 7.2 新架构训练规则

新结构只能从 WTW prior checkpoint 初始化，不能把旧的 `WtwDeltaResidualModel` 完整 checkpoint 当作严格兼容的续训点。第一次启动建议不加 `--resume`，观察 500~1000 iteration 的 teacher 训练和 distillation 指标；确认 `distillation_latent` 下降、`map_valid_ratio` 稳定后，再从该新 run 续训。验收时同时比较：平地 WTW trot、stepping stones 成功距离/terrain level、`illegal_contact`、`DELTA/residual_to_prior_ratio` 和 student/teacher action 差异。

## 8. 8 GB GPU 显存配置（已实现）

历史 teacher-student 方案同时保存 WTW 历史、当前 BEV、4 帧视觉历史和 privileged map，显存明显高于原始 WTW。该方案已移除；当前 Direct 任务只保留单帧 actor map 和 privileged critic map。

- `scene.num_envs = 64`，原始 WTW 和其他任务不变。
- PPO `num_mini_batches = 8`，每个 minibatch 为 `64 x 24 / 8 = 192` 个样本。
- student distillation epochs 从 4 降为 2。

这只降低并行吞吐，不改变观测、奖励或地形难度。启动时可以额外开启 PyTorch 的可扩展显存段：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 uv run train Nazarite-WTW-Delta-Direct-Go2 --gpu-ids 0
```

如果 Direct 任务仍然 OOM，优先将其 `scene.num_envs` 从 512 降到 256 或 128；不要修改原始 WTW 任务的并行环境数。

## 9. 2026-09-23 训练反馈后的改进（已实现）

`2026-09-23_10-16-55` 训练中，普通粗糙地形已经推进，但 stepping stones 平均 level 仍约为 0、gaps 为 0，且深度 BEV 的 raw occupancy 约 `0.15`、填充后 confidence 约 `0.26`。本轮针对两个直接问题进行修复：

1. Teacher-student 的 teacher forward 之前绕过了 `WtwDeltaResidualModel.forward()`，导致 residual/gate/attention 诊断没有写入 TensorBoard。现在动作合成共用 `_update_diagnostics()`，训练 teacher 和 eval student 都记录：

   ```text
   DELTA/residual_to_prior_ratio
   DELTA/residual_gate_mean
   DELTA/residual_saturation_ratio
   DELTA/map_confidence
   DELTA/attention_entropy
   DELTA/attention_valid_ratio
   ```

2. WTW+DELTA 的 D435 BEV 投影启用 `fill_kernel_size=7`，原始 support 通道保持不变，只扩大局部 confidence 填充范围。原始 DELTA 任务仍使用默认 `fill_kernel_size=5`。这样可以提高稀疏深度图的连续性，同时让 reward 和诊断仍能区分真实命中与填充值。

CPU 环境 smoke test 中，新的 5 通道地图 confidence 在单环境初始场景约为 `0.38~0.40`；teacher action、student action 和环境 step 均保持有限值。该结果不是训练集统计，下一轮训练必须重新观察 raw occupancy、filled confidence 和 `attention_valid_ratio`，不能仅凭 confidence 上升判断视觉真正有效。

## 10. Direct-action 版本（当前执行）

本轮按“无 teacher-student、DELTA 直接进入动作头”的要求新增独立任务：
`Nazarite-WTW-Delta-Direct-Go2`。此前的 residual、standalone DELTA 和 Oracle 实验入口已移除，原始 WTW 任务保持不变。

### 10.1 底层链路

```text
D435 depth (72x128 portrait for Direct, calibrated K)
  -> camera-frame pinhole back-projection
  -> body-frame x/y/z BEV (16x26x5)
  -> DELTA x/y/z + support + confidence attention encoder
  -> terrain latent (64)

WTW proprio (498) -> frozen WTW actor -> prior action (12)
DELTA proprio history (5x61=305) + terrain latent -> action_fusion MLP
[prior action, DELTA proprio history, terrain latent] -> tanh(delta action)
final action = prior action + delta action
```

`action_fusion` 为 `256 -> 128 -> 12` 的非线性 MLP，最后一层零初始化：训练初始输出严格等于 WTW checkpoint，之后 PPO 可以学习 `[-1, 1]` 范围内的每关节地形修正。这里没有固定 `residual_scale`，也没有把 confidence 作为动作后处理门控。WTW 网络仍冻结，避免训练初期破坏已有 trot。

训练使用非对称 PPO：actor 只看 D435 生成的地图；critic 看同范围的干净 raycast 地图和无噪本体观测。这不是师生蒸馏，critic 的地形特权信息不参与动作推理，部署时不需要 raycast。

### 10.2 相机投影和可视化

- 原始 DELTA/WTW+DELTA 任务继续使用 64x36，内参从 `d435.py` 的 848x480 标定按比例缩放为 `fx=31.7385, fy=31.5401, cx=32.3089, cy=17.8522`。
- Direct 任务当前使用 D435 竖装方案，使用 `72x128` portrait 图像；对应旋转后的内参为 `fx=63.0803, fy=63.4770, cx=35.7045, cy=64.6179`。两种分辨率保持相同的 D435 FOV，旧任务的 observation/checkpoint contract 不变。
- 竖装方案只改变成像平面方向：相机安装在 `base_link` 前方较低位置 `pos=(0.40, 0.0, 0.05)`，光轴水平（`camera_pitch_deg=0`），MuJoCo 外参使用 `quat=(-0.5,-0.5,0.5,0.5)` 将 `-Z_cam` 对准机器人前方 `+X_body`，将图像右侧 `+X_cam` 对准机体右侧 `-Y_body`，并将图像上方 `+Y_cam` 对准机体上方 `+Z_body`。因此 portrait 长轴在空间中确实竖直，投影使用 `camera_roll_deg=0`。该位置是仿真的前置安装假设，真实部署时应按 D435 相对机体的实测外参替换。
- 深度先按 pinhole 模型得到光学坐标，再按相机固定外参转换到机体坐标；原始 WTW+DELTA 使用 `x=0..2.5 m, y=-0.8..0.8 m`，direct 版使用 `x=0..2.0 m, y=-0.6..0.6 m`。
- 5 个通道是 `x/y/z/support/confidence`。`support` 只表示真实深度命中，`confidence` 允许局部 3x3 填洞但不会把窄 gap 标成真实支撑。
- play 时 Viser 同步显示 `raw depth / BEV z / support / confidence`，并叠加 DELTA 的采样点。训练时不缓存整幅图，避免额外显存。

### 10.3 8 GB 显卡启动和验收

Direct actor 不保存视觉历史，也不构造 teacher/student 蒸馏状态；训练 critic 仅使用一帧 privileged raycast。当前配置使用 512 个并行环境；先运行 200--500 iterations 检查显存和 FPS，出现 OOM 时通过命令行把 `scene.num_envs` 降到 256 或 128。不要把这项调整应用到原有 WTW+DELTA 任务。

```bash
cd /home/haozi/桌面/Nazarite-mjlab/Train/Nazarite
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run train Nazarite-WTW-Delta-Direct-Go2 --gpu-ids 0
```

第一阶段重点观察：

- `DELTA/direct_action_delta_ratio` 从 0 逐步上升但不爆炸；
- `DELTA/map_valid_ratio`、`DELTA/raw_map_occupancy_ratio` 和 `DELTA/filled_map_ratio` 是否稳定；
- 平地速度跟踪和 WTW trot 是否保持；
- stepping stones 的成功距离、terrain level、`illegal_contact` 是否改善；
- 原始深度、BEV、support 中的 gap 位置是否一致。

如果地图可视化中的 support 与地形位置不一致，应先修正外参、坐标轴或深度单位，再继续训练；不要用增大网络或奖励权重掩盖投影错误。

## 11. 2026-09-24 投影诊断改进

Direct 任务先使用近场 BEV 进行投影验收：`x=0..2.0m`、`y=-0.6..0.6m`、网格仍为 `16(y)×26(x)`。这样不会改变 DELTA 的张量尺寸，只提高近场像素落入网格的概率。训练日志新增：

```text
DELTA/depth_valid_ratio
DELTA/bev_inside_ratio
DELTA/projected_x_mean / projected_y_mean
DELTA/projected_x_span / projected_y_span
DELTA/support_x_near_ratio / support_x_far_ratio
DELTA/support_y_left_ratio / support_y_right_ratio
```

这些指标用于区分“相机没有返回深度”和“深度点被投影到 BEV 范围外”，也能直接判断前向远端是否比近端稀疏、左右视场是否不均匀。Direct 任务的 Viser 面板也保持深度和 BEV 的纵横比，并明确标注 x 向右、y 向下；原有任务仍使用原 BEV 范围。

## 12. 2026-09-25 架构审计后的按顺序实施

本轮不再同时修改相机、控制、奖励和课程，而是先用真值地图分离“控制学不会”和“视觉看不清”。已完成的实施顺序如下。

### 12.1 DELTA 数学和几何修正

- scout 只编码相对于采样中心的高程，不再让绝对地面高度支配特征。
- token 使用采样中心真实的归一化 `x/y/z`，不再将 `z` 写死为 0。
- 层间传播未被截断的 raw reference，只在实际取 patch 时限制到合法中心范围。
- `K=8` 的初始参考点改为 `4x2` 二维网格并加入一次性小扰动，避免全部初始在 `y=0` 上。
- attention 用 confidence 表示观测可信度，support 仍保留为物理深度命中标记。

### 12.2 控制和奖励信息边界

- DELTA 使用 5 帧可部署本体历史，总维度为 305，不包含真值基座线速度。
- actor 使用 D435 地图，critic 使用干净 raycast 地图；这是非对称 PPO，没有 teacher/student 网络和蒸馏 loss。
- `delta_swing_clearance` 从仿真真值地图计算，脚下 support cost 从每脚 4 条向下射线计算，相机投影质量不再能关闭核心地形奖励。
- 稀疏地形只发布正向 x 速度，连续地形保留 x/y/yaw 正负向指令。课程使用分地形成功率 EMA，热身后增加弱项地形的采样概率。

### 12.3 Direct 任务验收顺序

1. 直接从 WTW prior 初始化训练 `Nazarite-WTW-Delta-Direct-Go2` 3000--5000 iterations。
2. 对比 D435 与同步真值的 `map_height_mae_m`、`map_support_iou`、`map_support_precision/recall` 和 `map_gap_recall`。
3. 对同一 checkpoint 分别运行正常、`--delta-map-ablation zero` 和 `--delta-map-ablation shuffle` 的 play。如果三者成功率无显著差异，表示 actor 仍未因果使用视觉。
4. 只有投影指标和因果消融均通过后，才实施时序高程图融合；不在当前单帧架构尚未验证时继续增加网络复杂度。

这次 actor/critic 结构和 DELTA 输入维度都已变化，不能用旧 Direct checkpoint 续训；两个任务都应新建 run，仅由配置加载冻结的 WTW prior checkpoint。

```bash
cd /home/haozi/桌面/Nazarite-mjlab/Train/Nazarite
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run train Nazarite-WTW-Delta-Direct-Go2 --gpu-ids 0
```
