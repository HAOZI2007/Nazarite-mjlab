# XZ1 WMP 第二批：教师深度采集与独立 WM 预训练

2026-10-28 更新：九类地形的正式采集请使用
[最新采集说明](XZ1九类教师数据采集-2026-10-28.md)，其中包含配额修复、完整等级覆盖、
第四阶段教师命令和短序列处理。本页下方保留首批实现及历史命令说明。

实现日期：2026-10-26。基于第一轮教师 `2026-10-26_19-23-47_stage1_scan_teacher/model_999.pt`。

## 本批入口

| 入口 | 用途 |
| --- | --- |
| `Nazarite-WMP-HIM-Collect-XZ1` | 新注册的带深度相机教师环境，可用于 play 查看相机 |
| `python -m nazarite.scripts.collect_xz1_wmp` | 实际数据采集循环，显式处理终止与 reset |
| `python -m nazarite.scripts.pretrain_xz1_wmp` | 离线训练 WM，保存验证指标和可续训 checkpoint |
| `python -m nazarite.scripts.evaluate_xz1_wmp` | 扩展现有固定评测，支持多种子/速度与起点扰动 |

采集器冻结教师，只做推理，不更新 PPO/HIM；离线预训练只更新 WM。没有实现视觉学生、蒸馏或学生 PPO 微调。`train Nazarite-WMP-HIM-Collect-XZ1` 不是采集入口，请使用上述采集脚本；教师续训仍使用原 Teacher 任务。

## WM 已改为直接迁移参考源码

根据后续要求，移除了原有紧凑 WM，直接引入 `wmp_go1-main/dreamer/models.py`、`networks.py`、`tools.py` 和 `configs.yaml`，保存于 `nazarite/rl/wmp/reference_dreamer/`。训练直接调用参考的 `WorldModel._train()`；外围 `world_model.py` 仅适配本项目的数据、维度、设备和验证接口。

保留原版：通道 LayerNorm 的深度 CNN、本体 symlog MLP、图像和本体解码头、`learned initial`、`unimix=0.01`、原 GRU/RSSM、分布负对数似然、动态/表征 KL 和原优化器。图像损失按像素**求和**；本体使用 `symlog_mse`；奖励头使用255桶 `symlog_disc`，但 loss scale 仍按参考设为0。

默认64×64深度、CNN depth=32、本体编码/解码 MLP 5×1024、RSSM deter=512、32×32离散状态、hidden=512，KL free=1、dyn=0.5、rep=0.1；Adam lr=1e-4、eps=1e-8、grad clip=1000。编码5120维、完整RSSM特征1536维，供策略/后续学生使用的仍是参考的**512维 deter**。XZ1五步动作合并后60维，完整默认模型25,236,129个参数。CLI默认 batch=16、length=64，8GB机器建议先用 batch=4。

源码文件保留上游许可和归属，原始SHA256见 `reference_dreamer/README.md`。`models.py`、`networks.py` 和 YAML 未改算法；`tools.py` 仅更新 GradScaler API 并添加非有限 loss/梯度保护。验证使用与参考 `_train()` 一致的只读损失计算，不执行优化；随机RSSM沿用参考行为，固定验证随机种子且恢复训练RNG。

**旧紧凑WM权重不能续训。** 新文件记录 `world_model_implementation=wmp_go1_dreamer_v1`；离线入口和Go2 WMP runner均会明确拒绝旧实现的checkpoint。教师HIM/扫描教师权重、已采集的数据契约不变，可复用原始数据重新预训练WM。

参考 WMP 在线联合学习，本项目仍按已定路线拆成独立采集/预训练；这属于训练阶段安排的适配。地形潜变量的 ridge probe 只是诊断，不是新策略结构或 WM 的训练损失。教师沿用前批 Extreme Parkour 132→128→64→32 扫描编码器。

## 数据与时间契约

- 策略 50Hz、采样流 10Hz，每个正常相邻图像间包含五次控制命令，按最早→最新展平为 60 维。图像 `t` 对应从上一图像到 `t` 的动作，不能使用图像后将执行的动作预测它。
- 首帧 t=0、`is_first=True`、动作分块全零。每回合独立文件；中途终止不足五步时只记录终止元数据，不拼接不完整动作分块，更不拿 reset 后画面代替终止画面。
- 每个采样点和 reset 首帧显式强制刷新相机。`Simulation.sense(force_render=True)` 不推进物理、观测历史或命令时钟；可能增加实际渲染次数，因此“数据流10Hz”不意味着底层严格只渲染10次/秒。异步 reset 时也可能额外渲染，尚未做大规模吞吐基准。
- 保存的 `action` 是实际下发给环境的裁剪后策略命令，范围 ±10，位于关节缩放、偏置、执行延迟和 PD 之前。保留训练随机化时有执行延迟，不能把它解释为瞬时实测电机位置或力矩。
- `prop[33]` 为干净 estimator 中的角速度3、重力3、命令3、关节位置项12、关节速度项12，沿用其观测尺度。从 XZ1 的 50 维组中剔除相位 `[9:11]`、前一动作 `[35:47]`、真实速度 `[47:50]`，没有套用 Go2 的 48 维切片。
- 保存 `actor_history[6,47]`，顺序仍是环境的最旧帧在前；教师内部转换逻辑不变。动作/关节顺序仍为 FL、FR、RR、RL。
- 深度文件存正值米制 float16，裁剪至 2m；MuJoCo Warp 的未命中值0及无效值先映射到远裁剪距离，避免把天空解释成近墙。训练读取时转换为本项目 replay 的 uint8 表示，再由适配层恢复至 `[-0.5,0.5]`，传入原版WM。不对已经归一化的相机图再做米制归一化。参考 CNN 内部另减 0.5 的中心偏移照原源码保留。
- 教师动作、132维扫描及32维 latent 作为标签保存；WM 的 encoder 只读取 image 和 prop，地形标签不会成为学生必须依赖的推理输入。
- `time_s` 是回合控制时钟，`global_time_s` 是当前种子/地形/难度 case 的控制时钟；每个 case 从0开始，二者都不是主机墙上时间。

读取数据会验证形状、有限值、10Hz 时间连续性、首帧隔离、动作裁剪，以及“当前分块第一个动作等于上一图像对应的教师动作”，用于捕获一帧错位。

## 相机配置

配置位于 `Nazarite-src/nazarite/config/train_config/XZ1_env_cfgs/collect_env_cfg.py` 的 `XZ1DepthCfg`。默认挂载在 `robot/base_link`，位置 `(0.20,0,0.08)m`，前视下俯约20°，垂直 FOV=58°，64×64，只渲染地形 group0，沿用参考的地形专用视觉方式。

**位置和视角是仿真初值，不是实测 XZ1 相机外参。** 采集 CLI 支持 `--camera-pos x y z`、`--camera-fovy angle`；四元数可在配置中调整。正式视觉训练前应按实机安装和内参校准；变更后另建数据目录，元数据记录完整相机参数。

## 扩展教师评测

所有命令从 `/home/haozi/桌面/Nazarite-mjlab/Train/Nazarite` 执行。输出文件/目录必须是新路径，避免覆盖实验。

```bash
cd /home/haozi/桌面/Nazarite-mjlab/Train/Nazarite
uv run --no-sync python -m nazarite.scripts.evaluate_xz1_wmp \
  --policy teacher \
  --checkpoint logs/rsl_rl/xz1_wmp_him_teacher/2026-10-26_19-23-47_stage1_scan_teacher/model_999.pt \
  --terrains high_wall narrow_stairs narrow_ramp random_grid \
  --difficulties 0.5 1 \
  --seeds 42 43 44 --speeds 0.3 0.4 0.6 \
  --reset-x 0.1 --reset-y 0.025 --reset-yaw 0.08 \
  --command-mode velocity \
  --episodes 20 --num-envs 4 --device cuda:0 \
  --output logs/evaluation/xz1_teacher999_grid_v2.json
```

`--command-mode velocity` 为普通恒定机身速度，无赛道真值纠偏；`course` 仅在窄楼梯/窄坡启用原训练的真值中心线纠偏，JSON 明确标记 `privileged_course_steering=true`。两种模式应分别输出和比较，后者不能当作策略已经具备独立规划能力。固定种子参数 `--seed`、单速 `--speed` 继续兼容；复数选项优先。

每回合新增 `crossed_obstacle`、`max_stable_landing_seconds`、`max_lateral_deviation_m`。原成功判定保持：四脚过远端、机身过远端35cm、路线内、至少两脚承重、倾斜≤45°且连续0.2s，失败优先。`crossed_obstacle=true` 仍可能落地后越界而判失败；金字塔仍只有下楼梯验收。

## 采集教师数据

```bash
uv run --no-sync python -m nazarite.scripts.collect_xz1_wmp \
  --checkpoint logs/rsl_rl/xz1_wmp_him_teacher/2026-10-26_19-23-47_stage1_scan_teacher/model_999.pt \
  --output logs/wmp_data/xz1_teacher_v1 \
  --terrains flat high_wall narrow_stairs narrow_ramp trapezoid_ramp random_grid pyramid_stairs \
  --difficulties 0 0.5 1 \
  --seeds 42 43 44 45 46 --validation-seeds 46 \
  --episodes-per-case 8 --num-envs 8 --episode-seconds 20 \
  --device cuda:0
```

默认保留教师训练的奖励、随机化、观测/执行延迟及速度命令，包括窄路真值纠偏；只将地形固定为当前种子/种类/难度并关闭课程，均衡覆盖七类地形。单地形生成时同步过滤命令中的地形名称，避免仍引用不存在的高墙列。

加 `--nominal --speed 0.4 --command-mode velocity` 可采集无随机化的固定正向对照；`--command-mode` 和 `--speed` 只影响 nominal 模式。默认训练模式仍按原教师采样速度，不能误读为固定0.4m/s。

环境数量先用8，在本机测显存和采集时间后再增大。默认保留所有完整终止回合，包括失败；达到 case 配额时丢弃其他尚未结束的并行回合。`manifest.json` 保存教师文件 SHA256、相机参数、数据契约和逐回合终止信息。每个回合落盘后更新 manifest；异常退出标为 interrupted，离线训练拒绝把未完成采集当作完整数据集。本批没有断点追加采集功能。

验证种子必须是总种子的非空真子集，同一地形种子下所有难度和回合进入同一集合。读取时再次检查训练/验证种子不重叠，避免把同一场景的相邻帧拆给验证集。

## 独立预训练

先以1000次更新检查验证趋势，8GB显存机器可从 batch=4 开始；模型维度仍沿用默认参考配置。

```bash
uv run --no-sync python -m nazarite.scripts.pretrain_xz1_wmp \
  --dataset logs/wmp_data/xz1_teacher_v1 \
  --output logs/wmp_pretrain/xz1_wm_v1 \
  --updates 1000 --batch-size 4 --batch-length 64 \
  --eval-interval 100 --validation-batches 4 \
  --device cuda:0
```

输出包括 TensorBoard events、`config.json`、`metrics.json`、`wm_100.pt` 等 checkpoint。保存模型配置、数据 manifest 哈希、WM 权重、Adam 状态、步数和验证指标；并未导出机器人 actor。

```bash
uv run --no-sync python -m nazarite.scripts.pretrain_xz1_wmp \
  --dataset logs/wmp_data/xz1_teacher_v1 \
  --resume logs/wmp_pretrain/xz1_wm_v1/wm_1000.pt \
  --output logs/wmp_pretrain/xz1_wm_v2 \
  --updates 2000 --batch-size 4 --batch-length 64 \
  --eval-interval 100 --validation-batches 4 --device cuda:0
```

`--updates` 是总目标，以上继续1000次。续训要求模型配置、数据契约及 manifest 哈希一致；恢复 Adam 和迭代号，但没有保存随机采样器/RNG位置，不宣称逐位复现实验轨迹。

序列不会跨回合；每个采样窗口重置 RSSM，首个动作置零。长度不足64帧的短失败回合不能参与 length=64 训练，启动时会输出分地形可用/短回合数量，并写入配置。若需覆盖早期失败可另做更短序列实验，不能把“采集到了失败”当成这些失败都已进入长序列训练。采样对所有有效窗口均匀，长回合贡献更多窗口。

## TensorBoard 验收

- `Train/image_loss`、`Validation/image_loss`：同时看，训练下降而验证上升说明泛化没有同步改善。现在是参考的像素和，不能和旧紧凑版的像素均值直接比较；64×64单通道时除以4096可换算为图像均方误差。
- `Train/prop_loss`、`Validation/prop_loss`：新增原版本体重建损失，检查其与图像、KL的相对量级。`reward_loss`虽然仍记录，默认权重为0，不影响总损失。
- `Validation/future_image_mse` 与 `persistence_mse`：前者用前5帧建立状态，未来最多5帧只给动作、不给未来图像/本体；后者重复最后已见图像。`future_to_persistence_ratio < 1` 才优于该简单基线，但低运动片段可能让基线特别强。
- `Validation/terrain/<地形名>/future_image_mse` 和对应 `persistence_mse`：单独观察被验证批次采样到的地形，避免平地掩盖障碍问题；未出现某类 tag 不代表该类已验收，可增加 `--validation-batches` 扩大覆盖。
- `Validation/feature_std` 和全部有限值：检查状态是否塌缩或产生非有限值。
- `Validation/terrain_probe_mse` 对比 `terrain_mean_baseline_mse`：线性 ridge probe 只在训练种子拟合，验证种子计分。用于观察地形标签可解码性；本体也能解释部分标签，低 probe loss 不等于已经证明视觉有用。
- `Train/kl`、`kl_dyn_rep`、`grad_norm`：观察动态是否开始学习；本模块 free-bits 下限使 `kl_dyn_rep=2` 可能正常，不能单独当作故障。

先确认验证预测优于基线、地形信息和 reset 正确，再实现冻结教师的视觉学生蒸馏。固定一组验证窗口贯穿训练，原始指标和基线同时保留；不会根据短流程的 loss 自动判定模型已合格。

## 同步修复与验证

已有 Go2 WMP runner 保留 pending reset，直到下一次 WM 更新消费；发生 reset 时立即清理对应环境的 RSSM 状态、缓存特征和动作历史，支持推理的局部 reset。动作历史记录裁剪后命令。此次共享WM也切换到原版Dreamer，权重结构已经变化；HIM/actor架构不变。WM更新遇到非有限 loss/梯度时停止，避免写坏权重。

本批验证包括数据时间对应/种子隔离/序列边界、WM更新与续训、相机强制刷新、原教师PPO/HIM更新及ONNX导出回归。真实GPU已用 `model_999.pt` 采集平地和高墙：8个8秒回合、每回合81帧，共648帧；完整默认规格WM完成2次更新和独立验证，并额外验证 batch=4、length=64 的一次更新。另用原训练随机化采集窄楼梯，刻意在五步相机周期中间终止并 reset。

以下为早先紧凑WM的历史检查，不能作为此次原版模型的训练结果：两次更新验证 image loss 约0.672，未来预测差于保持上一帧基线。历史权重不再适用于当前实现，原始数据仍有效。

最终GPU短流程产物：`/tmp/xz1_wmp_second_batch_gpu_smoke_20261026_v2/`。其中 `data/` 为名义采集，`data_training_resets/` 为训练随机化/reset测试，`wm/` 为两次更新结果，`wm_length64/` 为64帧序列测试。评测入口短流程另存 `/tmp/xz1_wmp_second_batch_gpu_smoke_20261026_v1/eval_course.json`；其回合仅0.12秒，超时结果不能用于评价教师越障成功率。

原版Dreamer GPU检查另存 `/tmp/xz1_wmp_reference_gpu_smoke_20261026/`：复用上述原始数据，以完整默认模型、batch=4、length=64完成两次更新、独立验证及checkpoint保存。训练总损失约1045→919，未来/保持上一帧误差比约209→185，仍未达到可供学生使用的训练质量。这只证明原版模型能在本机运行；没有启动正式长训，没有修改教师权重、奖励或地形。

另直接加载 `/home/haozi/桌面/wmp_go1-main/dreamer` 原始源码，对照28个映射配置字段；在小尺寸模型、固定输入和随机种子下，对比原版和迁移版图像/本体/奖励损失、KL、一次Adam更新后的所有权重与优化器状态，逐项完全一致。该检查证明迁移数值一致性，不代表XZ1越障或学生效果已经验收。
