# Go2 WMP + HIM 迁移设计

## 目标

在 Nazarite/mjlab 新增独立任务 `Nazarite-WMP-HIM-Complex-Terrain-Go2`，复用现有 Go2 HIM 环境和 HIM PPO，实现：

- 参考 WMP 的前视深度相机、CNN 编码器、RSSM World Model、深度重建损失及 rollout 时的在线状态更新。
- 使用现有 HIM 历史估计器估计基座线速度与运动 latent；不引入 AMP、动作专家数据、判别器或 AMP replay buffer。
- 将 WMP 任务奖励系数迁移到 mjlab 现有 reward terms，并为 sparse terrain 项使用 mjlab 传感器实现等价/近似信号。
- 保持现有 Go2-HIM 和 XZ1-HIM 配置独立，避免覆盖工作区已有的未提交修改。

## 数据与策略接口

Go2 HIM 单帧本体观测按实际 manager 输出固定为 47 维，历史长度为 6；World Model 使用从本体观测独立抽取的 33 维 prop（角速度 3、重力投影 3、速度指令 3、关节位置 12、关节速度 12）。前置校验应确保 manager 的维度与模型配置完全相等，避免沿用 Go2 HIM 配置中的过期 49 维声明。

```text
64×64×1 depth → CNN(4096) + prop MLP → RSSM(512 deter + stochastic state)
                                                       │
                                                       └→ actor-side MLP → 32维视觉 latent
6×47 proprio → HIM → 3维速度估计 + 16维 HIM latent
当前47维 proprio + HIM(3+16) + WMP(32) → PPO actor → 12维关节动作
```

Actor MLP 输入为 `47 + 3 + 16 + 32 = 98` 维。Critic 使用特权观测并额外接收 512 维 RSSM deterministic feature。原 WMP 的 World Model CNN/RSSM/decoder 配置优先保持不变；mjlab 中以每个控制周期更新相机和 RSSM，RSSM 动作输入改为当前 12 维动作，以匹配新的采样周期。

相机深度范围裁剪为 0–2 m，先缩放到 `[-0.5, 0.5]`；遵循源仓库的 `ConvEncoder` 内再次减 0.5 的输入约定。训练及部署预处理必须一致。深度图仅留在压缩后的专用 World Model replay 中，不放入 PPO rollout TensorDict。

## World Model 训练

- 新建与 HIM 分离的 WMP runner 生命周期：当前图像/prop 更新 posterior RSSM，确定性状态写入 `world_model_feature` observation group，再由既有 HIM+PPO actor 采样动作。
- 将图像量化为 uint8 保存到按环境分段的有限 replay；prop、action、reward、episode-first 标志按连续序列采样。
- 训练目标复用源 WMP World Model：image reconstruction + RSSM KL/dynamics/representation loss；AMP 项完全不存在。
- checkpoint 增加 World Model 和 World Model optimizer 状态；保留 HIM estimator checkpoint 状态。
- 第一阶段不迁移 source depth predictor：mjlab 可直接渲染所有训练环境的真实 depth。只有相机渲染吞吐不足时再考虑 privileged-height-to-depth 预测器。

## 奖励迁移

源文件 `legged_gym/envs/a1/a1_amp_config.py` 的活动权重按下表迁移到 Go2：

| 源奖励 | 权重 | mjlab 适配 |
|---|---:|---|
| tracking_lin_vel | 1.0 | 平面速度跟踪，按源 `tracking_sigma=0.25`；仅在对应源障碍的沟壑、高墙、独木桥上使用 `lin_vel_clip=0.1` |
| tracking_ang_vel | 0.5 | HIM 角速度跟踪 |
| torques | -1e-4 | `joint_torques_l2` |
| dof_acc | -2.5e-7 | `joint_acc_l2` |
| action_rate | -0.03 | `action_rate_l2` |
| smoothness | -0.01 | `action_acc_l2` |
| collision | -1.0 | Go2 大腿/小腿碰撞 contact sensor，排除髋部以匹配源 `penalize_contacts_on` |
| dof_pos_limits | -10.0 | 围绕关节范围中心使用源 `soft_dof_pos_limit=0.9` 软限位 |
| dof_vel_limits | -10.0 | Go2 适配为 30 rad/s 标称限速的 0.9，即 27 rad/s，并沿用每关节最大 1 rad/s 超限惩罚 |
| lin_vel_z | -1.0 | 基座垂直速度平方惩罚 |
| feet_air_time | 1.0 | 落足事件上的腾空时间项，保留源 0.5 s threshold 和运动指令门控 |
| feet_stumble | -1.0 | 任一足端水平冲击大于 4 倍垂直冲击时惩罚 |
| feet_edge | -1.0 | 由足端支撑射线估计落脚边缘风险；仅在沟壑和高墙上启用，并按源配置在迭代 4000–10000 将权重系数从 0.1 线性升至 1.0 |
| cheat | -1.0 | 在沟壑及其后对应的障碍技能地形上惩罚偏离前进朝向，不作用于上/下楼梯 |
| stuck | -1.0 | 有前向指令但前向速度低于 0.1 m/s 时惩罚 |
| stool_base_height | -1.0 | 当前 Go2 terrain 无 stool/crawl 地形，不启用；迁移到对应地形后再启用 |

源配置中注释掉/权重为零的 termination、姿态、站立姿势等项不额外启用。`stool_base_height` 仅对源 stool 地形生效；九类 Go2 课程没有 stool，因此不启用。Legged Gym 与当前 mjlab 都按仿真步长缩放活动奖励；传感器近似项（尤其 `feet_edge`）仍需通过单步日志检查数值量级。

## 2027 障碍赛训练地形课程

WMP Go2 使用独立的 `wmp_terrain.py` 课程，不修改共享 Go2-HIM/XZ1-HIM 地形。参考赛规和 `ROBOCON2027.xml` 的障碍尺寸，将障碍拆为 9 类技能地形：

| 地形列 | 障碍模型 | 难度 1 对齐的关键尺寸 |
|---|---|---|
| `stairs_up` / `stairs_down` | 上楼梯 / 下楼梯 | 最高难度台阶深 0.32 m、高 0.12 m、宽 0.80 m |
| `low_frame` | 限高框 | 通行净空约 0.80 × 0.80 × 0.29 m |
| `high_wall` | 高墙 | 高 0.30 m、宽 0.80 m、厚 0.08 m |
| `gap` | 沟壑 | 平台高 0.36 m、平台长 0.80 m、沟宽 0.36 m |
| `tire_array` | 轮胎阵 | 2 × 4，外/内径 0.55/0.33 m，宽 0.165 m，节距 0.55 m |
| `single_beam` | 独木桥 | 3.0 × 0.40 × 0.10 m |
| `plum_blossom` | 梅花桩 | 0.80 m 平台与交错 0.20 m 方块 |
| `ramp` | 斜坡 | 宽 0.80 m，最高难度坡度 20° |

训练生成器按参考仓库采用 8 × 8 m 子地形块、10 行难度、九类各占一列，并从第 0 行开始。课程行的原始难度严格使用 `row / num_rows`；为使末行达到赛规尺寸，几何参数再按 `row / (num_rows - 1)` 归一化到 0–1。各类型在低难度行缩小高度/跨度、加宽净空或减少障碍数量；最高行按规则尺寸收敛。地形块之间不再加连续侧墙围成走廊；play 也按类型列、难度行组织网格，所有块固定为规则尺寸。轮胎环、沟底等采用适合 MuJoCo 的静态几何近似。

课程推进严格沿用参考仓库的通用距离规则，不再按障碍类型设置额外清障线或阈值封顶：机器人相对地形原点前进超过半个 8 m 地块（4 m）则升一级；否则若实际距离小于 `norm(command_xy) * episode_length_s * 0.5` 则降一级；最高等级毕业后随机回到已有等级，最低裁剪为 0。初始化 reset 不触发升降级。所有课程地形只采样正向速度，reset 固定朝向 +x；出生横向扰动限制为 ±0.20 m（相对 8 m 子地形块中心），高度扰动设为 0，使用 Go2 标称 0.32 m 根部高度，避免额外悬空出生。`play=True` 保留按类型列组织的参考网格，并将所有难度行固定为规则尺寸，适合检查技能迁移。

相机使用 64×64 深度图、58°视场、前置 0.28 m、下俯约 20°，只渲染地形组 0，与参考仓库的 terrain-only Warp 深度相机一致。MuJoCo 相机输出正向米制深度，预处理到 `[-0.5, 0.5]`，不套用 Isaac Gym 深度图的负号处理。base 接触传感器每个控制周期保留 10 个物理子步的净接触力历史，终止阈值仍对应参考仓库的 1 N 接触力判据。

建议先确认各单障碍列能够稳定通过，再单独制作含多个赛项障碍及比赛顺序的综合赛道用于最终验证；目前不能把训练课程本身视为完整比赛场景。

## 实施边界与验收

1. 独立新增配置、模型扩展、replay、runner 与任务注册；不得改动 AMP 源仓库，也不得在新任务中实例化 AMP 相关类或读取 AMP motion 文件。
2. 单元测试覆盖 World Model 图像/prop/RSSM tensor 形状、深度归一化、回放序列及 reset mask、Actor 98 维接口、checkpoint 保存/载入。
3. 运行配置构造、单步奖励调用与小规模 headless 训练 smoke test；硬件相机吞吐、长时收敛、ONNX 导出及真实 Go2 部署另行验收。
