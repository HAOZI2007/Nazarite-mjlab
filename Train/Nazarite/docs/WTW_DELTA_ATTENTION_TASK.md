# Nazarite-WTW-Delta-Go2

这是新的 WTW+DELTA 几何注意力任务，和 `Nazarite-WTW-Delta-Direct-Go2`
保持独立。

## 观测契约

```python
obs_groups = {
    "actor": ("wtw_proprio", "delta_map"),
    "critic": ("critic_privileged", "delta_privileged_map"),
}
```

`wtw_proprio` 为 498 维：普通本体项保留 10 帧，behavior 保留 5 帧，phase
保留最新一帧。DELTA 查询上下文从最新一帧切出 61 维：

```text
base_ang_vel(3) + projected_gravity(3) + joint_pos(12)
+ joint_vel(12) + actions(12) + command(3)
+ behavior(8) + phase(8)
```

策略链路为：

```text
WTW proprio history + dense local elevation map
  -> DELTA x/y/z tokens + deformable attention
  -> WTW prior action + terrain latent + query context
  -> complete action fusion head
  -> 12 joint actions
```

新模型位于 `nazarite/delta/wtw_delta_attention_model.py`。初始化时默认加载
WTW checkpoint，并从 WTW 动作先验开始；DELTA 融合层零初始化，避免第一轮
训练破坏已有 trot。该任务会联合微调 WTW 和 DELTA，checkpoint 只是稳定的
策略初始化，不是永久冻结的 teacher。

## 高程传感器和局部地图

当前任务已经移除虚拟深度相机。`delta_privileged_scan` 使用密集垂直射线直接
生成机体坐标系下的局部高程地图，扫描范围为 `3.0×1.0 m`、分辨率 `0.04 m`，
再裁剪到前方 `x=[0.0, 1.5] m`、`y=[-0.5, 0.5] m`。

actor 和 critic 使用相同尺寸和坐标范围，地图不堆叠历史帧。默认地图为
`25×40×3`，也保留 `16×26×3` 消融选项：

- `25x40x3`：默认训练配置；
- `16x26x3`：低分辨率消融配置。

3 个通道依次为 `x/y/z`，两组地图都调用 `delta_privileged_terrain_map`，因此
当前任务先验证 DELTA 编码器和奖励，再单独处理深度相机 sim-to-real。
训练地形只有四类：`plum_stones`（规则均匀细柱梅花桩）、`grid_rough`（随机网格起伏）、
`stairs_up`（上楼梯）和 `stairs_down`（下楼梯）。梅花桩由固定 x/y 间距的圆柱组成，
前方保留一块起步平台；难度主要通过减小柱半径和增加很小的高度扰动实现。

躯干终止使用 `sustained_illegal_contact`：躯干接触力超过 10 N 且连续两个传感器
子步成立才终止。单个碰撞冲量仍由 `collision` 奖励惩罚，但不会立即结束 episode。
`TerrainHeightSensor.data.heights` 始终是相对 clearance；射线 miss 返回
`max_distance`，不再错误地使用世界坐标的 `frame_z`。
高程栅格的 z 聚合使用负无穷作为无效单元初值，因此基座坐标系下正常地面的负 z
不会被错误截成 0；无命中单元才在聚合后填为 0。

## 地形奖励

任务恢复速度指令，不再生成显式目标点。奖励使用论文的正负组合：

```text
r_t = r_plus * exp(0.1 * r_minus)
```

其中正奖励包括速度/yaw 跟踪和 WTW 的俯仰项；身体高度是负平方误差，因此放入
负奖励衰减项。其余负奖励包括
相位/接触约束、关节与动作正则化、足端打滑/软着陆、边缘落脚、绊脚、
身体碰撞、姿态和垂向速度等。失败 episode 额外施加 `-50`，timeout 不施加
失败惩罚。

当前启用的复杂地形相关项：

| 奖励 | 权重 | 作用 |
|---|---:|---|
| `track_velocity_x/y` | `+1.0/+1.0` | 跟踪速度指令的前向/侧向分量 |
| `track_yaw_velocity` | `+2.0` | 跟踪 yaw 角速度指令 |
| `feet_edge` | `-0.5` | 脚下支撑射线不足时惩罚边缘落脚 |
| `delta_swing_clearance` | `-0.35` | 高程地图检测到起伏时，摆动脚 clearance 不足的惩罚 |
| `feet_stumble` | `-0.25` | 足端水平撞击垂直障碍时惩罚 |
| `collision` | `-1.0` | 汇总髋、大腿、小腿、躯干碰撞 |
| `orientation` | `-0.5` | roll/pitch 姿态约束 |
| `lin_vel_z` | `-0.2` | 抑制跳跃/坠落的垂向速度 |

`wtw_shank_contact` 置零，因为它已被 `collision` 汇总，避免小腿碰撞双重惩罚。
WTW phase/contact、摆腿高度和 Raibert 项只保留较弱权重，用于维持基本 trot 时序，
不再强行限制复杂地形上的步态变化。

## 构造另一套地图

当前任务注册默认使用 `25x40`。要做低分辨率消融，从 Python 工厂创建配置并将
runner 配置同步切换：

策略动作高斯分布的初始 `std=0.20`，但不设置 `std_range`，因此标准差由 PPO
自由学习；其他任务仍保留各自的默认范围。

```python
env_cfg = Nazarite_Wtw_Delta_Go2(bev_variant="16x26")
rl_cfg = wtw_delta_go2_runner_cfg(bev_variant="16x26")
```

不要只修改环境或只修改 runner；两者的地图尺寸必须一致。

通过任务注册直接训练时，也可以用环境变量快速切换，不需要改注册代码：

```bash
NAZARITE_WTW_DELTA_BEV=25x40 \
uv run train Nazarite-WTW-Delta-Go2 --gpu-ids 0
```

可用变量：`NAZARITE_WTW_DELTA_BEV=16x26|25x40`。环境和 runner 必须使用同一个值。
