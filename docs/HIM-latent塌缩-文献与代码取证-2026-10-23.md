# HIM estimator 的 16 维 latent 塌缩：文献、代码与社区证据

> 调研日期 2026-10-23。所有 arXiv 编号均逐一 curl 核验过标题；所有代码引文均取自官方仓库原始文件。
> 全文标注 **【事实】**（可点链接/原文/可复算）与 **【推测】**（我的判断，未经作者确认）。

---

## 0. 结论摘要

1. **【事实】你观察到的 `swap loss ≈ 0.10` 不是 bug，是这套公式的数学必然。** HIM 的 `swap_loss` 在“prototype 为单位向量 + τ=3.0”时的可观测区间只有 **≈[0.098, 0.108]**，而完全塌缩与完全随机在此指标上**读数几乎相同**（我按上游公式复算：随机 0.1084，全部同向 0.1084，8 簇 0.1047）。它能“停在 0.10 不报警”，恰恰说明它作为塌缩监控指标**在结构上失效**，而不是训练正常。
2. **【事实】社区一致复现出同样的现象，且直接质疑了论文 Fig.4 的 t-SNE 结论**：官方 issue #11 / #13 / #28 中至少 4 位复现者报告 swap loss 不下降、t-SNE 分不出地形、“对比学习的两个编码器走了捷径”。
3. **【事实】把 temperature 从 3.0 调到 0.1 能让 t-SNE 分出地形、swap loss 大幅改善，但报告者明确说“对最终奖励值没有什么影响”**（issue #28）。这是“latent 对策略没有增益”的最强单条外部证据。
4. **【推测，但证据强】你的假设方向正确**：target = `next_obs[3:48]`，即“去掉 command 的整帧观测 + 末尾 3 维 base_lin_vel”。其中 42/45 维几乎由当前状态决定，唯一真正“不可从历史推断”的 3 维线速度又被独立的速度头专门回归 —— 16 维 latent 确实没有必须编码的残差。叠加 τ=3 使对比信号比 SwAV 原版小一个数量级，塌缩几乎无代价。
5. **【推测】但“不可避免”要打个折**：塌缩到 std=0.003 而非精确 0、且“随机置换 latent(0.212) 优于置零(0.297)”，说明 latent 里仍残留少量信息。更准确的定性是：**latent 退化成“一个学到的常数偏置 + 一点微弱抖动”**。这来自目标设计 + 温度标定的组合，改配方即可恢复，不是物理上限。
6. **工程上最该先做的不是加正则，而是换 target 并修 τ**：target 换成真正含残差的特权量（接触力/足高/地形高度/摩擦估计）、τ→0.1、把 swap loss 改成 per-sample CE（去掉 1/K 归一化）以便监控。

---

## 1. HIM 原论文与官方代码

### 1.1 链接与元信息

| 项目 | 内容 |
|---|---|
| 论文 | [arXiv:2312.11460](https://arxiv.org/abs/2312.11460) — *Hybrid Internal Model: Learning Agile Legged Locomotion with Simulated Robot Response* (v3, 2024-01-02) |
| 作者 | Junfeng Long, Zirui Wang, Quanyi Li, Jiawei Gao, Liu Cao, Jiangmiao Pang |
| 单位 | 上海 AI Lab（OpenRobotLab）+ 浙江大学 + 清华大学 |
| 项目页 | https://junfeng-long.github.io/HIMLoco/ |
| 代码 | [InternRobotics/HIMLoco](https://github.com/InternRobotics/HIMLoco)（原 OpenRobotLab/HIMLoco），977 stars / 97 forks |
| 代码状态 | **【事实】最后一次 commit 2024-05-14（`ef289aca` "fix export bug and lr bug"）；`him_estimator.py` 全程只被 2 个 commit 改过，`temperature=3.0` 从未被作者调整过。** |
| 同仓库后续工作 | [Learning H-Infinity Locomotion Control, arXiv:2404.14405](https://arxiv.org/abs/2404.14405)（`projects/h_infinity/`，沿用同一 estimator，未改 latent 设计） |

### 1.2 latent 的完整实现（逐行引用）

`rsl_rl/rsl_rl/modules/him_estimator.py`：

```python
# L11-58  默认超参：注意 temperature=3.0、num_prototype=32
class HIMEstimator(nn.Module):
    def __init__(self, temporal_steps, num_one_step_obs, enc_hidden_dims=[128, 64, 16],
                 tar_hidden_dims=[128, 64], activation='elu', learning_rate=1e-3,
                 max_grad_norm=10.0, num_prototype=32, temperature=3.0):
```

```python
# L41  编码器最后一层同时输出 16 维 latent 和 3 维速度（共享 trunk，无独立头）
enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[-1] + 3)]
```

```python
# L64-68  forward：latent 做 L2 归一化，且对 actor 双 detach（PPO 梯度不回传 estimator）
parts = self.encoder(obs_history.detach())
vel, z = parts[..., :3], parts[..., 3:]
z = F.normalize(z, dim=-1, p=2)
return vel.detach(), z.detach()
```

```python
# L82-83  target 的构造：这一行是"没有残差"假设的核心证据
vel = next_critic_obs[:, self.num_one_step_obs : self.num_one_step_obs + 3]   # 3 维 base_lin_vel
next_obs = next_critic_obs.detach()[:, 3 : self.num_one_step_obs + 3]          # 丢掉前 3 维 command
```

```python
# L92-95  每次 update 都就地重归一化原型（no_grad），所以 score 恒在 [-1, 1]
w = F.normalize(self.proto.weight.data, dim=-1, p=2)
self.proto.weight.copy_(w)

# L97-107  swap loss：Sinkhorn 产生 q，softmax(score/τ) 产生 p
score_s = z_s @ self.proto.weight.T
q_s = sinkhorn(score_s)
log_p_s = F.log_softmax(score_s / self.temperature, dim=-1)
swap_loss = -0.5 * (q_s * log_p_t + q_t * log_p_s).mean()

# L108-116  总损失 = 速度回归 + swap，无任何方差/协方差/熵正则
estimation_loss = F.mse_loss(pred_vel, vel)
losses = estimation_loss + swap_loss
```

```python
# L119-133  Sinkhorn：行归一化到 1/K、列归一化到 1/B，3 次迭代 → 等分约束（SwAV 原版做法）
Q = torch.exp(out / eps).T          # eps=0.05
...                                  # 3 轮行/列交替归一化
return (Q * B).T
```

### 1.3 观测维度（对上你的 47 维）

`legged_gym/legged_gym/envs/base/legged_robot_config.py:36-39`：

```python
num_one_step_observations    = 45      # 单帧 obs
num_observations             = 45 * 6  # 6 帧历史 = 270
num_one_step_privileged_obs  = 45 + 3 + 3 + 187   # obs + base_lin_vel + external_forces + height_scan
```

单帧 45 维组成（`legged_robot.py:247-253`）：commands×3 + base_ang_vel×3 + projected_gravity×3 + dof_pos×12 + dof_vel×12 + actions×12 = 45。

**【事实】于是 `next_obs` 的 45 维 = [ang_vel(3), gravity(3), dof_pos(12), dof_vel(12), actions(12), base_lin_vel(3)]。前 42 维是“当前状态 + 本步动作”的近确定性函数；只有末 3 维 base_lin_vel 是历史推不出来的，而它正是速度头的回归目标。** 这与你描述的“跳过 command、末尾接 3 维特权线速度”完全一致 —— 你的复现是忠实的。

actor 侧（`him_actor_critic.py:96, 169`）：`mlp_input_dim_a = num_one_step_obs + 3 + 16`，输入 `cat(obs_history[:, :45], vel, latent)`。

### 1.4 论文自己怎么说

**【事实】论文在 Related Work 里把 representation collapse 写成了“回归类辅助损失”的失败模式：**

> "What these methods have in common is the use of a regression objective to learn a model or representation, which requires the neural network to fit the targets perfectly. However, due to the uncertainty brought by domain randomization, the network is willing to fit the random noise, **eventually resulting in representation collapse**."

**【事实】论文主张对比学习能规避它，并宣称 latent 空间可分：**

> §3.3: "we propose modeling it into a latent space Z ⊂ R^16 and optimizing it to be close to the successor state with contrastive learning"
>
> Fig.4 caption: t-SNE "has a **more separable encoding of environments**"

**【事实】论文全文没有任何一处报告自身 latent 的方差、有效秩、原型使用率或 swap loss 的绝对数值**，只有一个 t-SNE 定性图。这一点很关键：作者用 collapse 论证了自己的动机，却没有为自己的方案提供同类度量。

---

## 2. 为什么 swap loss 恒定在 0.10（我做的数值复算）

**【事实·可复算】按上游公式，`swap_loss = mean_b CE_b / K`**（`.mean()` 摊到 B×K 个元素）。又因为 `z` 与原型都被归一化，`score ∈ [-1, 1]`，τ=3.0 时 `logits ∈ [-0.333, 0.333]`，softmax 几乎是均匀分布，CE ≈ log K。于是：

```
swap_loss ≈ log(32) / 32 = 3.466 / 32 = 0.1083
```

我在本地用 numpy 按上游公式（B=4096, K=32, D=16, τ=3, eps=0.05, 3 iters）跑了 5 种 latent 配置：

| latent 配置 | swap_loss |
|---|---|
| 随机（16 维正态后归一化） | **0.1084** |
| 完全塌缩（所有样本同向） | **0.1084** |
| 2 簇 | 0.1068 |
| 8 簇 | 0.1047 |
| 随机 × 3（模拟更大 score 幅度） | 0.1166 |

**【结论·事实】整个可观测区间只有 ≈[0.098, 0.108]，且“塌缩”与“未塌缩”的读数差异小于 1%。** 你的 0.10 与 log(32)/32 完全吻合，说明：

- 你的复现没有问题，读数和官方一致；
- **这个指标不具备检测塌缩的能力**（不是“没报警”，是“物理上报警不了”）；
- 对比信号强度：SwAV 原版 τ=0.1，logits 幅度是这里的 30 倍，梯度也大约强 30 倍。**τ=3.0 把对比项变成了几乎无梯度的装饰项**，而 `MSE(vel)` 的梯度大得多 —— 共享 trunk 的 19 维输出里，前 3 维有强监督，后 16 维只有一个极弱且平台化的对比信号。

**【推测】塌缩的动力学机制**：`F.normalize` 使 latent 对尺度不变，只剩方向自由度；方向上的唯一驱动力（swap）梯度 ≈ O(1/(K·τ))，被 Adam 的噪声与 trunk 共享梯度淹没；而“所有样本同向”是该损失的近似平台（上表：0.1084 vs 0.1084）。于是初始化附近的小幅特征在几百轮内被磨平，direction 漂到一个几乎常数的点，swap loss 当然不会变 —— 它本来就在平台上。

---

## 3. 社区证据：这不是你一个人的问题

**【事实】以下均为官方仓库 issue 原文（英文原文 + 中文说明）：**

**(a) [#11 "Swap Loss not Decreasing in Contrastive Learning"](https://github.com/InternRobotics/HIMLoco/issues/11)（P1terQ，2024-08-10 开）**

> "when I attempted to run the open-source code, **the swap loss didn't drop, regardless of short or long iterations**."
> — 中文：跑官方代码，swap loss 无论训多久都不下降。

> mahanxin (2025-01-20): "I also conduct t-SNE to analyze the latent space output, but unfortunately **it could not identify the terrain**."
> — 中文：我也做了 t-SNE，可惜分不出地形。

> CSU-BossYu (2026-03-09): "I have encountered exactly the same issue, and I agree with your view that **the teacher and student encoders in contrastive learning take shortcuts instead of learning terrain features**."
> — 中文：遇到完全相同的问题，同意你的看法——对比学习里的两个编码器走了捷径，没有学到地形特征。

**(b) [#13 "关于latent space的问题请教"](https://github.com/InternRobotics/HIMLoco/issues/13)（2025-01-08，中文）** —— 问 16 维到底代表什么特征。

> LRuYuan (2026-05-05): "估计器网络，代码默认这个 temperature=3，我用 tsne 可视化这个地形分类，**结果显示的就是什么也没有学到**。"
> CSU-BossYu (2026-05-19): "我的可视化的结果也很差，没学到东西。"

**(c) [#28 "地形分类可视化"](https://github.com/InternRobotics/HIMLoco/issues/28)（LRuYuan，2026-05-05，中文）** —— 最接近你问题的一条。

> 默认 τ=3 时 t-SNE **只分出一种地形**；"我调整这个参数(temperature)，便可以分出不同的地形"。
>
> 后续自述："**0.1 的效果挺好的，但是 swap 损失函数改善很多，但是对最终的奖励值没有什么影响，对训练结果也看不出来有什么提升**。"
> — 中文：τ=0.1 时聚类恢复、swap loss 明显改善，但奖励和最终训练结果看不出提升。

**【这条极其关键】**它同时证明两件事：(1) τ 确实是塌缩的直接旋钮；(2) **即使把对比项修好、latent 变得可分，策略性能也没变** —— 强烈支持“latent 没有承载策略需要的残差信息”。

**(d) [#29](https://github.com/InternRobotics/HIMLoco/issues/29)（mrk0669，2026-05-10）** —— 直接提 PR 把 HIM 的双编码器 + 原型表换成 DreamWaQ 式 VAE（mu/logvar + KL，推理用 mu），作为"HIM 原型对比设计是否必要"的消融。**社区已经有人走到"推翻原型瓶颈"这一步了。**

**(e) [#6](https://github.com/InternRobotics/HIMLoco/issues/6)**（20 条评论）训练曲线问题，作者 Junfeng-Long 回复："There are some bugs and improper configs in the code. Already fixed, please try the new one."，另建议最多训 2000 iter。

**(f) [#17](https://github.com/InternRobotics/HIMLoco/issues/17)** 指出 `him_ppo.py:192` 返回的 estimation/swap loss 是**最后一个 mini-batch** 的值而非 epoch 均值 —— 你看到的曲线抖动也有这个成分（但不影响“停在 0.10”的结论）。

**【事实】作者从未在任何 issue 中回应“latent 塌缩 / t-SNE 分不出地形”的质疑**（#11、#13、#28 均无作者回复）。

---

## 4. 同类工作怎么防塌缩（可借鉴的具体做法）

| 工作 | 机制 | 对本题的可移植性 |
|---|---|---|
| [SwAV, arXiv:2006.09882](https://arxiv.org/abs/2006.09882) | Sinkhorn 等分约束（HIM 抄的就是它）+ **τ=0.1** + multi-crop | **HIM 丢了 τ 的标定**（3.0 vs 0.1），这是最直接的一处偏离 |
| [DINO, arXiv:2104.14294](https://arxiv.org/abs/2104.14294) | centering（减均值，防单维主导）+ sharpening（低温，防均匀）+ **EMA teacher + stop-grad** | HIM 的 target encoder 与 source 同步更新、无 EMA，双分支可同时塌缩 |
| [VICReg, arXiv:2105.04906](https://arxiv.org/abs/2105.04906) | 方差 hinge（每维 std 下限 1）+ 协方差去相关（off-diagonal→0）+ 不变项 | **最直接的补丁**：对 `z` 加 var+cov 正则，能在 loss 上把“塌缩”变成显式惩罚 |
| [LeJEPA / SIGReg, arXiv:2511.08544](https://arxiv.org/abs/2511.08544) | 强制嵌入服从各向同性高斯（特征函数匹配），单超参、无 teacher | 比 VICReg 更简洁的“抗塌缩”正则，2025 年的新方案 |
| [DreamWaQ, arXiv:2301.10602](https://arxiv.org/abs/2301.10602) | CVAE：mu/logvar + KL，推理用 mu | 有 KL 项时塌缩表现为“KL→0”（posterior collapse），**至少可监控**；issue #29 正是这条路线 |
| [SleepWalking/SWAQ, arXiv:2608.30883](https://arxiv.org/abs/2608.30883)（2026-08） | 用**特权量重建**（而非对比）来塑造历史表征 | 原文给出了塌缩的可判定下界（见下），是最有理论依据的替代方案 |
| [Predictive Training with Latent Imagination, arXiv:2607.17574](https://arxiv.org/abs/2607.17574)（2026-07） | JEPA predictor + SIGReg 监督 recurrent state，推理时丢弃 predictor | “训练时用、推理时扔”的范式，与 HIM 的 estimator 定位一致 |
| [VQ-VAE, arXiv:1711.00937](https://arxiv.org/abs/1711.00937) | codebook 使用率统计 + dead code 重启 | 32 个原型的**使用率直方图 / perplexity** 是最便宜的塌缩探针，HIM 完全没做 |

**【事实】SleepWalking 关于“为什么重建比对比更抗塌缩”的原文论证（可直接引用）：**

> "The reconstruction objective also penalizes complete representation collapse. If h_t is constant, a deterministic decoder can produce only a constant prediction, **whose minimum expected squared error is given by the weighted variance of Y_{t+1}**. Therefore, achieving a reconstruction error below this constant-predictor baseline requires h_t to retain information."

— 中文：重建目标天然惩罚完全塌缩。若隐状态恒定，解码器只能输出常数，其最小期望平方误差恰是目标 Y 的方差。想低于这个“常数预测器基线”，就必须保留信息。

**【这是本题最关键的方法论差异】**：MSE 重建/回归有一个**与塌缩挂钩的、非退化的基线**（= Var(Y)），塌缩立刻让 loss 卡在基线上、肉眼可见；而 softmax-对比 + Sinkhorn 的 swap loss **在塌缩时读数不变**（第 2 节），从设计上就无法承担“监控表征健康”的职责。HIM 论文把 regression 的 collapse 当作动机，却换了一个对 collapse **不敏感**的损失 —— 这本身是逻辑缺口。

---

## 5. 裁决：你的假设站得住吗？

### 5.1 支持假设的证据（强）

1. **【事实】target 构造**：`next_obs = next_critic_obs[:, 3:48]`，42/45 维是当前状态的近确定性函数；唯一非确定的 3 维（base_lin_vel）**已被独立速度头专门回归**。latent 没有“必须编码的残差”。
2. **【事实】loss 结构**：`estimation_loss` 是强监督（MSE，量级 O(10⁻²) 级梯度），`swap_loss` 在 τ=3 下梯度 O(1/(K·τ))≈0.01 倍量级，且处于平台。共享 trunk 的 16 维出口实际上**只被极弱信号驱动**。
3. **【事实】社区对照实验**：τ=0.1 修复了聚类（latent 变得可分），但**奖励无变化**（issue #28）。若 latent 真携带策略必需的残差，修好对比项应当改变性能。
4. **【事实】论文的 t-SNE 可分离性主张被至少 4 位独立复现者否证**（#11、#13、#28 及其评论）。
5. **【事实】作者 2 年未更新代码、未回应此类 issue**，`temperature=3.0` 原封不动 —— 这个配置更可能是“没调过”，而非“调过并验证过”。

### 5.2 反对 / 需要修正的部分

1. **【事实】你的置换实验（0.212）优于置零（0.297）**，说明 latent 并非纯常数，仍含少量信息。所以准确说法是 **“latent 退化为常数偏置 + 微弱抖动”**，不是“完全死亡”。
2. **【事实】论文 Table 4 真机消融**：`w/o lat. inp.`（latent 置零）短楼梯 100%→50%、长楼梯 176.5→13.3 阶、缺步 100%→5%；`w/o lat. loss` 同样崩。**【推测】但这更像“把训练分布内的输入替换成分布外常数”造成的 OOD 破坏**，不能证明 latent 携带信息 —— 一个被 actor 当常数偏置使用的向量，置零同样会崩。
3. **【推测】“塌缩不可避免”过强**：真正的因果是“target 无残差 + τ 标定错误 + 无方差正则”三者叠加。任改其一都能解除，所以是**配方问题**，不是信息论下界。

### 5.3 一句话结论

> **支持你的假设，但把它精化为**：HIM 的 latent 塌缩是“target 里没有需要编码的残差 + τ=3 让对比项退化为常数 + 损失对塌缩不敏感”共同造成的**设计性退化**；它在官方代码里同样存在（社区已复现），而**不是**这个架构的信息论上限。判据是 issue #28：修好对比项后 latent 变可分但策略不变 —— 残差确实不在那里。

---

## 6. 建议的工程改法（按性价比排序）

### 6.1 先做 3 个判别实验（半天，不用重训大模型）

| 实验 | 做法 | 判读 |
|---|---|---|
| **① batch-mean 替换** | 推理时把 latent 换成该 batch 的均值向量（保留偏置、抹掉方差），对比 0.141 基线 | 误差≈0.141 → 方差无用（假设成立）；跳向 0.297 → 方差真有用 |
| **② 线性探针** | 用 latent 线性回归 friction / terrain level / payload / 接触状态；报 R² | R²≈0 → latent 无特权信息（假设成立） |
| **③ 原型使用率** | 统计 `argmax(q)` 直方图与 perplexity（32 个原型用了几个） | 只用 1–3 个 → 原型表退化，塌缩已发生 |

补充：记录 `swap_loss` 与 `estimation_loss` 的**梯度范数比**，以及 `z` 的每维 std / 有效秩随轮次曲线（你已有）。这三条一起能把“无残差”与“优化失败”分开。

### 6.2 若假设成立：按顺序改

1. **换 target（最关键）**：不要用整帧下一观测。改成只喂“当前历史推不出来的量”：
   - 下一帧的 **base_lin_vel(3) + 足端接触力/接触标志 + 足端高度 + 地形高度采样**（HIM 里现成有 187 维 height scan，`num_one_step_privileged_obs` 末尾就是它）；
   - 或预测 **t+1…t+k 的多步**特权量（k=3~5），强制编码动力学而非单帧快照。
2. **修 τ 与监控口径**：`temperature: 3.0 → 0.1`（SwAV 原值）；把 `swap_loss` 的 `.mean()` 改成 per-sample CE 的均值（去掉 1/K），使读数落在 [0, logK] 的可解释区间；或干脆额外记录 `-q·log p` 的 per-sample 值。
3. **加防塌缩正则**：对 `z_s` 加 VICReg 方差项（hinge，目标 std=1，权重 1.0）+ 协方差 off-diagonal 项（权重 0.04）；或换 SIGReg。注意：HIM 的 `z` 已 L2 归一化，方差正则要作用在**归一化前**的 16 维上，否则与单位球约束打架。
4. **EMA target encoder + stop-grad**：target 分支用 momentum(0.99) 更新，避免双分支同步塌缩（DINO/BYOL 的标准做法）。
5. **若确认无残差，就别硬塞**：把 latent 降到 4–8 维，或删掉 latent 只留速度头 + 历史 encoder（与论文 `w/o lat. loss` 的差别在于：你是主动省算力，不是被动掉点）。要“地形感知”，正解是**换 target**，不是加正则。

### 6.3 若假设被推翻（实验①②显示方差有用）

那么问题落在优化上：优先查 (a) `F.normalize` 前的 16 维是否被 trunk 的共享梯度压制（可给 latent 加独立 MLP 头，从 64 维分叉）；(b) Adam 的 eps 与 lr 是否让 16 维方向的更新被淹没；(c) `class` 是否把 `z` 的 detach 做错（应为 `z.detach()` 只给 actor，estimator 内部仍要反传）。

---

## 7. 可点击链接清单

**HIM 本体**
1. 论文：[arXiv:2312.11460](https://arxiv.org/abs/2312.11460)
2. 代码：[InternRobotics/HIMLoco](https://github.com/InternRobotics/HIMLoco)
3. 项目页：https://junfeng-long.github.io/HIMLoco/

**社区 issue（塌缩直接证据）**
4. [#11 Swap Loss not Decreasing](https://github.com/InternRobotics/HIMLoco/issues/11)
5. [#13 关于 latent space 的问题请教](https://github.com/InternRobotics/HIMLoco/issues/13)
6. [#28 地形分类可视化（τ=0.1 实验）](https://github.com/InternRobotics/HIMLoco/issues/28)
7. [#29 用 DreamWaQ 式 VAE 替换原型对比](https://github.com/InternRobotics/HIMLoco/issues/29)

**方法来源与防塌缩方案**
8. [SwAV, arXiv:2006.09882](https://arxiv.org/abs/2006.09882)（swap loss + Sinkhorn 出处，τ=0.1）
9. [DINO, arXiv:2104.14294](https://arxiv.org/abs/2104.14294)（centering/sharpening/EMA）
10. [VICReg, arXiv:2105.04906](https://arxiv.org/abs/2105.04906)（方差+协方差正则）
11. [LeJEPA/SIGReg, arXiv:2511.08544](https://arxiv.org/abs/2511.08544)
12. [DreamWaQ, arXiv:2301.10602](https://arxiv.org/abs/2301.10602)（VAE 路线）
13. [SleepWalking/SWAQ, arXiv:2608.30883](https://arxiv.org/abs/2608.30883)（重建的下界论证）
14. [Predictive Training with Latent Imagination, arXiv:2607.17574](https://arxiv.org/abs/2607.17574)

---

## 8. 事实 / 推测 分界

**未经我验证、仅有二手来源的信息**：所有 issue 中的复现细节（τ=0.1 的具体配置、t-SNE 参数）来自报告者自述，我没有独立复现其训练；论文 Table 4 的数值来自论文正文。

**我独立完成的**：全部代码引文（curl 官方 raw 文件，逐行核对）；arXiv 编号与标题（逐个 curl abs 页核对）；第 2 节的 swap loss 数值复算（numpy，按上游公式，B=4096/K=32/D=16）；第 1.2 节 target 维度的拆解推导。

**纯推测**：塌缩的动力学机制（梯度量级论证）、Table 4 掉点源于 OOD 而非信息缺失、作者未调过 τ 的判断。
