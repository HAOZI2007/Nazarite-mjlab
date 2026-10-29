# XZ1 WMP 第三批：视觉学生蒸馏

本批接通“扫描教师 → 冻结 WMP 世界模型 → 深度视觉学生”，包含在线蒸馏、续训、固定地形评测和专用 play。尚未进入 PPO 微调和实机部署。没有自动启动正式训练。

## 实现和参考边界

- WM 直接使用已迁移的 `wmp_go1-main/dreamer`，加载 `xz1_wm_v1_resume5000/wm_5000.pt`，不改网络、不更新参数。原版 RSSM 在 eval 下仍采样，保留此行为。
- 动作监督对应 Extreme Parkour 的 `OnPolicyRunner.learn_vision` 和 `PPO.update_depth_actor`：`mean(||teacher_action - student_action||₂)`。未加入它未启用的 latent 对齐损失。XZ1 当前观测没有该实现中的两维目标 yaw，因此没有另造 yaw 预测头。
- 继承教师的 actor MLP、HIM 和归一化统计，移除 scan encoder，增加 `512 → 128 ELU → 32 Tanh` 的视觉投影。只优化视觉投影与 actor MLP；教师、WM、HIM 和归一化全部冻结，critic 不参与蒸馏。
- actor 输入仍为 **47 当前观测 + 3 HIM 速度估计 + 16 HIM latent + 32 视觉 latent = 98 维**，输出 12 维动作。HIM 使用 6×47 历史。学生不读取 teacher scan 或 critic。WM 的 prop33 从 estimator50 中明确去掉相位、上一步动作和真实基座速度。
- 学生实际行走，教师在相同状态标注。默认从第一步由学生执行，与参考 `num_pretrain_iter=0` 一致。可选 `--teacher-warmup` 和 `--teacher-mix-iterations` 是项目适配项，默认均为 0；启用后按环境/控制步抽样执行教师动作，再逐渐降至零。WM 始终接收实际执行的动作。
- 地形、课程、奖励、终止和域随机化继承当前 XZ1 扫描教师配置，保持七类地形。奖励只用于监测与原有课程，不作为本阶段的优化损失。课程等级不能代替独立越障成功率。

## 时序与保存

控制为 50 Hz，深度和 WM 后验为 10 Hz。每次相机更新输入的是通向当前图像的五个已执行动作，按旧到新折叠成 60 维。两次相机更新间保持 512 维 deter；不使用未来图像。

每个环境独立维护 RSSM、动作历史和相机采样相位。reset 后强制获取新帧、动作块清零、`is_first=True`，不把上一回合记忆带入新回合。深度缺失射线映射到 far clip；按离线数据的 float16 保存和 uint8 量化处理后，仅进行一次归一化。

相机直接读取 WM 训练数据 manifest 中的配置，不在学生阶段另设一套外参。会检查 WM/数据契约、manifest SHA256、教师 SHA256、教师数值约定。当前相机仍是仿真假定安装位姿，未经实机标定。

checkpoint 包含学生、优化器、训练轮次、地形类型/等级、DELTA 课程统计、全局控制步、Torch 随机状态和教师/WM 溯源。续训恢复这些状态，但物理回合、相机历史和 RSSM 从新回合开始，不保证与不中断运行逐步一致。续训需保留相同环境数和超参数；`--iterations` 是总目标轮次，可增加。每轮默认 24 控制步，每 100 轮保存一次。

学生 checkpoint 引用原 WM 文件；WM 加载还需核对原数据 manifest。请保留原教师、WM 和 `logs/wmp_data/xz1_teacher_v1/manifest.json` 路径。部署导出暂未实现，禁止通过继承的 HIM 导出接口生成遗漏视觉状态的 actor-only 模型。

## 执行命令

所有命令从以下目录运行，`--output` 必须是新目录/新文件，避免覆盖：

```bash
cd /home/haozi/桌面/Nazarite-mjlab/Train/Nazarite
```

### 1. 第一轮蒸馏：1000 轮

这是独立的监督训练入口，不走 PPO 的 `train <task-id>`。32 个环境已在本机 GPU 做过流程验证。

```bash
uv run --no-sync python -m nazarite.scripts.distill_xz1_wmp \
  --teacher logs/rsl_rl/xz1_wmp_him_teacher/2026-10-26_19-23-47_stage1_scan_teacher/model_999.pt \
  --world-model logs/wmp_pretrain/xz1_wm_v1_resume5000/wm_5000.pt \
  --output logs/rsl_rl/xz1_wmp_student/student_v1 \
  --iterations 1000 \
  --num-envs 32 \
  --steps-per-iteration 24 \
  --learning-rate 1e-4 \
  --save-interval 100 \
  --device cuda:0
```

先在 200、500 轮检查 play；1000 轮做固定地形评测。若显存不足，首次运行前把环境数改为 16。续训时需保持这个环境数一致。监督学习率 `1e-4` 是本阶段保守起点，不是声称参考仓库的唯一正确值。

TensorBoard：

```bash
uv run --no-sync tensorboard --logdir logs/rsl_rl/xz1_wmp_student
```

关注 `Distill/action_l2`、`Distill/grad_norm`、`Episode/length_steps`、`Episode/reward`、`Termination/*`、`Curriculum/terrain_level`。启用执行混合时还要看 `Distill/teacher_probability`；概率未降为零前的表现不能作为纯学生能力。

### 2. 专用 play

不要用普通 HIM 播放入口加载这个 checkpoint；专用入口会维护完整的相机、WM 和 reset。

```bash
uv run --no-sync python -m nazarite.scripts.play_xz1_wmp_student \
  --checkpoint logs/rsl_rl/xz1_wmp_student/student_v1/student_500.pt \
  --terrain narrow_stairs --difficulty 0.5 \
  --speed 0.4 --seed 46 --device cuda:0 --viewer native
```

逐一切换 `flat high_wall narrow_stairs narrow_ramp trapezoid_ramp random_grid pyramid_stairs`。`--difficulty 0 / 0.5 / 1` 显式控制难度，避免只看一种地形。无桌面可用 `--viewer viser`；自动化流程可用 `--headless-steps 100`。

默认 `--command-mode velocity` 没有特权路线纠偏。`--command-mode course` 在窄地形使用教师已有的特权纠偏，只用于对照，不能将其结果当作不带特权引导的越障表现。

### 3. 独立评测

```bash
uv run --no-sync python -m nazarite.scripts.evaluate_xz1_wmp \
  --policy student \
  --checkpoint logs/rsl_rl/xz1_wmp_student/student_v1/student_1000.pt \
  --terrains flat high_wall narrow_stairs narrow_ramp trapezoid_ramp random_grid pyramid_stairs \
  --difficulties 0 0.5 1 \
  --seeds 46 47 --speeds 0.3 0.4 \
  --episodes 8 --num-envs 4 \
  --reset-x 0.03 --reset-y 0.02 --reset-yaw 0.05 \
  --command-mode velocity --vision normal --device cuda:0 \
  --output logs/rsl_rl/xz1_wmp_student/student_v1/eval_1000_normal.json
```

保持其他参数一致，再分别改 `--vision far`、`--vision shuffled` 和输出文件名。far 将图像替换为最远深度，shuffled 将当前帧像素顺序随机打散；二者都是评测干预，不是训练增强。若正常视觉与干预后的越障成功率基本相同，不能宣称策略已学会利用视觉。

沿用教师评测标准：四足越过障碍远端、落地有至少两足承重、姿态稳定持续 0.2 秒；失败优先。评测不生成 teacher scan，不启用训练噪声、时延和域随机化。`pyramid_stairs` 当前出生配置对应下楼方向，报告会注明，不能解释为上下楼都通过。

### 4. 续训到 3000 轮

```bash
uv run --no-sync python -m nazarite.scripts.distill_xz1_wmp \
  --teacher logs/rsl_rl/xz1_wmp_him_teacher/2026-10-26_19-23-47_stage1_scan_teacher/model_999.pt \
  --world-model logs/wmp_pretrain/xz1_wm_v1_resume5000/wm_5000.pt \
  --resume logs/rsl_rl/xz1_wmp_student/student_v1/student_1000.pt \
  --output logs/rsl_rl/xz1_wmp_student/student_v1_resume3000 \
  --iterations 3000 --num-envs 32 --steps-per-iteration 24 \
  --learning-rate 1e-4 --save-interval 100 --device cuda:0
```

先检查 1000 轮效果再决定是否续训，避免仅看蒸馏 loss 下降就盲目加轮次。

## 验证记录（2026-10-27）

- 学生及相关教师/WM/数据回归测试累计 41 项通过；新增文件和修改的评测入口通过 Ruff、Pyright。
- 真实 `model_999.pt` + `wm_5000.pt`：32 环境训练 2 次更新，再从 checkpoint 续训到第 3 次；缩短回合以反复触发 timeout/reset。控制步计数恢复为 48→72，优化器有状态，冻结 HIM/归一化在续训前后逐元素保持一致。
- 固定 flat/high_wall 的学生评测完成，专用 play 的 far 深度干预完成无界面 12 步检查。GPU 流程输出在 `/tmp/xz1_student_final_*`。
- 这些都是短流程验证，尚无正式学生训练成功率。0.3 秒评测只用于检查调用/reset，不足以跨越障碍，不能作为能力结论。native/viser 交互窗口未在本轮人工验收。

主要代码：`nazarite/rl/xz1_student.py`、`scripts/distill_xz1_wmp.py`、`scripts/play_xz1_wmp_student.py`、扩展的 `scripts/evaluate_xz1_wmp.py`、`tests/test_xz1_student.py`（均位于 `Train/Nazarite/Nazarite-src/`）。
