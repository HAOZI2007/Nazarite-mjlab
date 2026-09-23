# WTW+DELTA 参考 AME 的改进实施计划

## 1. 目标与范围

本计划针对当前 `go2_wtw_delta_residual` 任务，目标是在不修改原始 WTW、原始 DELTA 任务的前提下，逐步获得“平地保持 WTW trot、复杂地形使用深度图调整步态”的策略。AME 仓库只作为实现参考，不直接复制其代码或修改其仓库。

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
- 不要把原始 WTW 的奖励和配置全局改掉；所有实验参数应只放在 `wtw_delta_env_cfg.py` 和 `wtw_delta_rl_cfg.py`。
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

BEV 范围从 `3.0 x 2.0 m` 收缩为机器人前方 `2.5 x 1.6 m`，保持 `16 x 26 x 4` 的网络输入尺寸，提高每个网格获得真实 D435 点的概率。可视化坐标同步使用新范围。

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
