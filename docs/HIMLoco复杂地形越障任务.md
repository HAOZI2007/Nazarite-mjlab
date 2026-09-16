# HIMLoco 复杂地形越障任务

任务 ID：`Nazarite-HIM-Complex-Terrain-Go2`

该任务把参考仓库的 HIMLoco 组件迁入 Nazarite 的本地 RSL-RL：历史编码器、速度/潜变量估计器、HIM 对比学习目标、专用 rollout storage、HIMPPO 和终端状态 runner 均位于 `Train/Nazarite/mjlab/RSL-RL/rsl_rl/`。环境沿用 Nazarite 的 Go2 粗糙地形生成器，包含波浪、金字塔坡面、随机粗糙面、上/下楼梯和离散障碍物，并使用地形等级课程推进。

训练：

```bash
cd Train/Nazarite
uv run train Nazarite-HIM-Complex-Terrain-Go2 --device cuda:0
```

可视化检查：

```bash
cd Train/Nazarite
uv run play Nazarite-HIM-Complex-Terrain-Go2 --checkpoint <checkpoint.pt>
```

HIM actor 的每帧本体观测为 45 维，堆叠 6 帧；critic 的前 45 维作为估计器目标观测，紧随其后的 3 维真实基座线速度作为速度监督目标。这样策略在执行时只依赖历史本体观测，同时在复杂地形上使用估计到的运动状态和潜变量完成越障。
