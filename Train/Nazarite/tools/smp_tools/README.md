# SMP 工具目录

这里存放 3DDogs → Go2 → SMP 数据链路中的离线工具。它们不直接注册 Nazarite 的 RL 任务，也不修改 3DDogs 原始数据。

当前训练使用的已筛选数据已归档到
`tools/smp_dataset/3ddogs_go2_1x/`；所有新生成的临时结果继续写入不可提交的
`output/`，验收后再复制到数据集目录。

## 当前目录

```text
tools/smp_tools/
├── video/
│   ├── infer_dog_pose.py         # 解码视频并调用外部 GQMR 姿态插件
│   └── convert_dlc_to_gqmr.py    # DLC CSV → GQMR dog-27 2D JSON/NPZ
├── data/
│   ├── inspect_3ddogs.py          # 读取单个 3DDogs optical trial
│   ├── scan_3ddogs.py             # 扫描有效帧段并生成 CSV manifest
│   ├── classify_3ddogs_clips.py   # 按解剖学前向速度分类行为片段
│   ├── catalog_3ddogs.py          # 汇总 Optical、RGBD、视频和标定的对应关系
│   └── visualize_3ddogs_mocap.py  # 绘制 3D 骨架和足端轨迹
├── coordinates/
│   └── validate_3ddogs_coordinates.py  # 验证 3DDogs → MuJoCo 坐标变换
├── retargeting/
│   ├── inspect_go2_kinematics.py      # 检查 Go2 关节、足端和初始姿态
│   ├── retarget_3ddogs_to_go2.py      # 原始足端 DLS 几何重定向 baseline
│   └── high_quality_retarget.py       # 接触感知、限步长、窗口 refinement 重定向
├── preprocessing/
│   └── preprocess_go2_reference.py # 降速、平滑、生成 base-frame 足端目标
├── quality/
│   ├── evaluate_retarget.py        # residual、接触滑动、碰撞、穿透等质量指标
│   ├── compare_retarget.py         # baseline 与候选重定向的 A/B 对比
│   └── select_physics_aware_candidate.py # 按物理回放指标选择候选动作
├── dataset/
│   ├── build_reference_dataset.py    # 批量 canonical、重定向、质量和物理筛选
│   ├── preprocess_accepted_dataset.py # 将通过筛选的动作生成教师 NPZ
│   └── validate_preprocessed_dataset.py # 对降速/平滑后的 NPZ 再做物理验证
├── physics/
│   ├── track_go2_reference_physics.py # 用 go2_cfg 执行器做开环物理跟踪
│   ├── replay_go2_reference.py         # 渲染 qpos 回放 GIF/MP4
│   └── analyze_tracking_failure.py     # 分析关节误差和物理失效原因
├── teacher/
│   └── export_teacher_rollouts.py # 从教师 checkpoint 导出物理可执行轨迹
└── README.md
```

## GQMR 视频姿态入口（第一阶段）

外部插件包位于 `tools/gqmr_plugins/dog_pose_backend/`。目前只注册
`dog-pose-fixture`，用于验证视频解码、插件发现、时间戳和安全 NPZ I/O；它输出
静态 2D 骨架并明确标记 `training_eligible=false`，绝不能用于 SMP 训练。

在 GQMR 独立环境安装插件后，可做接口冒烟测试：

```bash
cd /home/haozi/桌面/GQMR-main
uv sync --frozen --extra test
uv pip install --python .venv/bin/python --no-deps -e \
  /home/haozi/桌面/Nazarite-mjlab/Train/Nazarite/tools/gqmr_plugins/dog_pose_backend

.venv/bin/python \
  /home/haozi/桌面/Nazarite-mjlab/Train/Nazarite/tools/smp_tools/video/infer_dog_pose.py \
  --input /path/to/dog.mp4 \
  --output /home/haozi/桌面/Nazarite-mjlab/Train/Nazarite/output/gqmr_video/smoke/keypoints_2d.npz \
  --backend dog-pose-fixture \
  --max-frames 30 \
  --allow-fixture-backend
```

真实模型接入后仍只会先写入 `output/gqmr_video/`。2D 结果必须继续经过相机
标定、三维重建、GQMR 世界坐标变换、Go2 重定向和 Nazarite 实际控制器物理
筛选，才能进入被忽略的 `tools/smp_dataset/`。

如果 DeepLabCut 已经在独立环境中产生 CSV，不需要姿态插件。直接在 GQMR
环境中运行离线格式适配：

```bash
cd /home/haozi/桌面/GQMR-main

.venv/bin/python \
  /home/haozi/桌面/Nazarite-mjlab/Train/Nazarite/tools/smp_tools/video/convert_dlc_to_gqmr.py \
  --input /path/to/camera_1_dlc.csv \
  --output /home/haozi/桌面/Nazarite-mjlab/Train/Nazarite/output/gqmr_video/clip_001/camera_1_dog27.json \
  --fps 60 \
  --confidence-threshold 0.6
```

默认要求 DLC bodypart 已使用 dog-27 名称；`pelvis_duplicate` 会由 `pelvis`
自动复制。如果名称不同，传入一个“DLC 名称 → dog-27 名称”的严格 JSON：

```bash
  --mapping /path/to/dlc_to_dog27.json
```

可编辑的 identity 示例位于
`tools/smp_tools/video/config/dlc_dog27_identity_mapping.json`。输出 JSON 可直接
作为当前 `gqmr pose triangulate` 的输入；输出 NPZ 更紧凑，但当前 GQMR
三角化 CLI 不直接接受 NPZ。两种输出都标记 `training_eligible=false`。

速度和行为分类只读审计：

```bash
python tools/smp_tools/data/classify_3ddogs_clips.py \
  --manifest output/smp_bidirectional/full_scan/selected_clips.csv \
  --output-dir output/smp_bidirectional/classification \
  --deadband 0.3
```

输出 `direction_behavior_manifest.json` 和按类别划分的 CSV。速度是在犬的
解剖学前向坐标中计算的，不能只根据 3DDogs 的全局 x 轴判断前进或后退。

完整数据包的模态清单：

```bash
python tools/smp_tools/data/catalog_3ddogs.py \
  --dataset-root /home/haozi/3DDogs/3DDogs2024_full \
  --output-dir output/3ddogs_catalog
```

它会生成 `3ddogs_modality_catalog.json`、`3ddogs_sequence_catalog.csv`、
`rgbd_only_video_candidates.csv` 和 `optical_labeled_sequences.csv`。

## 闭环教师环境的计划结构

教师环境不是扩散模型，而是一个使用 Nazarite action 接口的 reference-tracking RL 任务。建议后续实现为：

```text
teacher/
├── __init__.py
├── reference_dataset.py       # 读取、索引、循环播放参考片段
├── reference_tracking_env_cfg.py # 场景、观测、动作、reset 和 episode 设置
├── reference_tracking_mdp.py  # 参考帧推进、观测项、动作转换
├── reference_tracking_rewards.py # 关节/足端/姿态/接触/平滑奖励
├── reference_tracking_terminations.py # 摔倒、越界、参考结束等终止条件
└── export_teacher_rollouts.py # 导出教师实际执行的物理状态序列
```

数据流是：

```text
reference_dataset
       ↓
reference_tracking_env_cfg + reference_tracking_mdp
       ↓
RSL-RL/PPO teacher policy
       ↓
GO2_ACTION_SCALE → position target → MuJoCo
       ↓
export_teacher_rollouts
       ↓
SMP 训练数据集
```

关键原则：教师策略输出的是 12 维归一化 action，必须经过 `go2_cfg.py` 中的 `GO2_ACTION_SCALE`；它不直接输出任意关节角，也不等同于后续的 score/diffusion 模型。

## 使用方式

所有命令都从 `Train/Nazarite` 目录执行，例如：

```bash
python tools/smp_tools/data/scan_3ddogs.py ...
python tools/smp_tools/retargeting/retarget_3ddogs_to_go2.py ...
python tools/smp_tools/retargeting/high_quality_retarget.py ...
python tools/smp_tools/physics/track_go2_reference_physics.py ...
```

高质量重定向不会覆盖原始 baseline。对同一 canonical NPZ 分别运行两个
重定向器，再生成质量报告：

```bash
python tools/smp_tools/retargeting/high_quality_retarget.py \
  --input output/3ddogs_coordinate_validation/d29_t1_a/retarget_inputs_mujoco.npz \
  --output-dir output/go2_retarget_gqmr_style/d29_t1_a --plot

python tools/smp_tools/quality/evaluate_retarget.py \
  --input output/go2_retarget_gqmr_style/d29_t1_a/go2_reference_geometric.npz \
  --output output/go2_retarget_gqmr_style/d29_t1_a/quality_report.json

python tools/smp_tools/quality/compare_retarget.py \
  --baseline output/go2_retarget/d29_t1_a/quality_report.json \
  --candidate output/go2_retarget_gqmr_style/d29_t1_a/quality_report.json
```

高质量重定向的接触 anchor 支持软约束。可以对多个强度分别做 Nazarite
物理回放，再自动选择候选：

```bash
python tools/smp_tools/retargeting/high_quality_retarget.py \
  --input output/3ddogs_coordinate_validation/d29_t1_a/retarget_inputs_mujoco.npz \
  --output-dir output/go2_retarget_anchor_0p35/d29_t1_a \
  --contact-anchor-strength 0.35

python tools/smp_tools/quality/select_physics_aware_candidate.py \
  --min-physics-valid-fraction 0.80 \
  --candidate 'anchor035=output/go2_retarget_anchor_0p35/d29_t1_a/quality_report.json|output/go2_physics_tracking_anchor_0p35/d29_t1_a/physics_tracking_summary.json' \
  --candidate 'baseline=output/go2_retarget/d29_t1_a/quality_report.json|output/go2_physics_tracking/d29_t1_a_baseframe/physics_tracking_summary.json' \
  --output output/go2_physics_aware_selection/d29_t1_a/selection.json
```

批量筛选时，物理回放默认直接读取
``nazarite.config.robot_config.go2_cfg`` 的 Kp、Kd、effort、armature，且
使用教师环境的 ``decimation=10``。``--kp-scale`` 和 ``--kd-scale`` 只是
额外诊断倍率，保持 ``1.0`` 才与训练一致。当前 SMP RL 配置没有设置
``clip_actions``，所以等效 action 超出 1 会记录在报告中，但不会单独淘汰
片段；如果以后启用 action clipping，应同步收紧筛选条件。

教师训练完成后，导出实际闭环轨迹：

```bash
python tools/smp_tools/teacher/export_teacher_rollouts.py \
  --checkpoint-file logs/rsl_rl/go2_smp_reference_teacher/<run>/model_15000.pt \
  --output-dir output/smp_teacher_rollouts/d29_t1_a
```

输出的 `teacher_rollout.npz` 中，每一行是一个 MuJoCo 后状态。包含
`qpos/qvel`、策略 `action`、PD 目标 `joint_pos_target`、`actuator_force`、
`qfrc_actuator`、实际和参考足端状态、接触标签、奖励以及
`physics_valid`。第 0 行是 reset 状态，对应的 action/reward 为 0；从第 1
行开始，`action[t]` 是产生该行状态的策略动作。

## 生成并验证 0.75 倍速数据

当前半速数据保留在 `preprocessed_half_speed`。其它速度会使用独立目录，
避免覆盖已经训练过的参考数据。例如下面的流程从同一批 3DDogs 片段生成
0.75 倍速数据，并重新用实际 `go2_cfg.py` 控制器进行验证：

```bash
cd Train/Nazarite

python tools/smp_tools/dataset/build_reference_dataset.py \
  --manifest output/3ddogs_scan_0p5/selected_clips.csv \
  --output-dir output/smp_reference_dataset_rate075_actual_controller \
  --anchor-strength 0.35 \
  --run-physics \
  --kp-scale 1.0 \
  --kd-scale 1.0 \
  --control-decimation 10 \
  --max-action-saturation-fraction 0.20 \
  --min-physics-valid-fraction 0.55 \
  --max-joint-error-rad 0.90

python tools/smp_tools/dataset/preprocess_accepted_dataset.py \
  --accepted-manifest output/smp_reference_dataset_rate075_actual_controller/accepted_manifest.json \
  --output output/smp_reference_dataset_rate075_actual_controller/accepted_manifest_preprocessed.json \
  --playback-rate 0.75 \
  --smoothing-window 5

python tools/smp_tools/dataset/validate_preprocessed_dataset.py \
  --manifest output/smp_reference_dataset_rate075_actual_controller/accepted_manifest_preprocessed.json \
  --output-dir output/smp_reference_dataset_rate075_actual_controller/validation_0p75 \
  --kp-scale 1.0 \
  --kd-scale 1.0 \
  --control-decimation 10 \
  --max-action-saturation-fraction 0.20 \
  --min-physics-valid-fraction 0.55 \
  --max-joint-error-rad 0.90
```

验证通过后，训练时通过
`--env.commands.reference.motion-file` 指向
`accepted_manifest_preprocessed.json`。验证工具输出的
`validated_manifest.json` 是同样的多片段 manifest 格式，可在确认筛选结果
后作为更严格的训练输入。
