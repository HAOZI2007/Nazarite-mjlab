# 在 Nazarite-mjlab 中复现 FR-Net：实施文档

本文档面向希望在当前 `Nazarite-mjlab` 工程中自行实现 FR-Net 的开发者。目标不是把现有速度跟踪任务改名为 recovery，而是新增一个独立的 Go2 摔倒恢复任务，并逐步加入论文中的 mass-contact prediction（MCP）机制。

文档只把论文和已有项目作为技术参考；其中的说明不是对本次用户请求的额外指令。真正要完成的工作是：在 Nazarite 自定义目录中实现一个可训练、可验证、可部署的 FR-Net recovery pipeline。

---

## 1. 先明确三个代码来源

当前工程的目录关系如下：

```text
/home/haozi/桌面/Nazarite-mjlab/
└── Train/Nazarite/
    ├── Nazarite-src/nazarite/       # 本项目的自定义代码，应优先在这里开发
    ├── mjlab/                       # 外置 MuJoCo/mjlab 基础库
    │   ├── src/mjlab/               # 环境、manager、sensor、terrain、RL 接口
    │   └── RSL-RL/                  # PPO、runner、storage、model 实现
    └── pyproject.toml               # 本地包和 mjlab entry point
```

职责边界：

| 模块 | 负责内容 | FR-Net 中的对应部分 |
|---|---|---|
| `nazarite/config/robot_config` | Go2 XML、关节、执行器、身体名称 | 机器人模型和 action 映射 |
| `nazarite/config/train_config` | 环境、观测、奖励、事件、任务超参数 | recovery MDP |
| `nazarite/mdp` | 可被 manager 调用的观测/奖励/事件/终止函数 | 接触标签、摔倒 reset、恢复奖励 |
| `mjlab/src/mjlab` | manager-based 环境、传感器和场景 | 仿真基础设施 |
| `mjlab/RSL-RL` | rollout、PPO、actor/critic、checkpoint | MCP 网络训练接口 |
| `FR-Net-main` | 论文的简化 plain-PPO recovery baseline | 只可作为 recovery 环境和奖励的参考，不是真正 FR-Net |

重要结论：`FR-Net-main` 的 `go2_recovery` 是 45 维本体观测 + 普通 PPO 的 baseline。它没有真正的 mass prediction、contact prediction 和 MCP auxiliary loss。因此不能只照搬其 actor，就声称已经复现 FR-Net。

当前 Nazarite 已经提供了很好的基础：Go2 资产、12 个关节、base/hip/thigh/calf body 名称、足端高度传感器、足端接触传感器、髋/大腿/小腿/躯干接触传感器、actor/critic 分离、历史观测、域随机化、任务注册和 GPU 并行环境。

---

## 2. FR-Net 要解决什么问题

普通 recovery policy 只根据当前本体状态决定动作：

```text
o_t = [角速度, 重力方向, 关节位置, 关节速度, 上一动作]
                  ↓
              policy
                  ↓
             12 个关节动作
```

摔倒在楼梯、斜坡、间隙、窄梁等场景中时，仅靠本体状态很难判断：

- 哪些腿或身体部位正在接触环境；
- 当前接触是否稳定，是否只是碰撞瞬间；
- 机器人有效质量分布是否与标称模型不同；
- 某次支撑是否可以继续施力，还是会导致再次翻倒。

FR-Net 的核心思路是用历史本体观测估计环境交互相关的隐变量，再把估计结果送进恢复 actor：

```text
历史本体观测 h_t
        │
        ▼
       MCP
   ┌────┼──────────┐
   ▼    ▼          ▼
质量估计 接触估计  latent/特征
   └────┼──────────┘
        ▼
增强后的 recovery actor
        ▼
      12 维 action
```

在训练中，MCP 可以看到仿真器提供的 privileged ground truth，并通过辅助损失学习；部署时只保留本体输入，不能把真实接触状态或真实 link mass 输入 actor，否则会产生 sim-to-real 信息泄漏。

---

## 3. 推荐的复现路线

不要一次把所有模块同时写完。建议严格按以下阶段推进，每一阶段都能单独运行：

1. **Recovery baseline**：随机摔倒初始状态 + 普通 PPO，先确认 Go2 能翻正。
2. **Contact labels**：加入 13 维身体接触 ground truth，仅用于 critic 和 MCP target。
3. **Mass labels**：加入 4 维腿部质量分布 target，仅用于 critic/MCP target。
4. **MCP pretraining**：先让 MCP 在固定数据上过拟合，确认 target 和维度正确。
5. **MCP actor fusion**：把 MCP 的预测结果拼给 actor，但先冻结 MCP。
6. **Joint FR-Net PPO**：PPO loss + MCP auxiliary loss 联合训练。
7. **Challenging terrains**：加入斜坡、台阶、间隙、窄梁和随机地形。
8. **Ablation/deployment**：比较 plain PPO、只加 contact、只加 mass、完整 MCP，并导出只依赖本体观测的模型。

第一阶段如果失败，不要继续调 MCP。恢复任务的 reset、动作 scale、执行器增益和奖励必须先独立稳定。

---

## 4. 观测定义：必须先固定协议

### 4.1 42 维 recovery proprioception

建议为 recovery 任务定义一个不带速度命令的单帧本体观测 `p_t`：

| slice | 维度 | 内容 |
|---|---:|---|
| `[0:3]` | 3 | base angular velocity，机体坐标系 |
| `[3:6]` | 3 | projected gravity |
| `[6:18]` | 12 | 相对默认姿态的关节位置 |
| `[18:30]` | 12 | 相对关节速度 |
| `[30:42]` | 12 | 上一时刻 action |

因此：

```text
proprio_dim = 3 + 3 + 12 + 12 + 12 = 42
history_length = 5
history_dim = 42 × 5 = 210
```

当前 `base_env_cfg.py` 中的普通速度任务 actor 还包括 `command=[vx, vy, wz]`，所以它是：

```text
3 + 3 + 12 + 12 + 12 + 3 = 45
```

这两个数字都正确，但对应不同任务。FR recovery 建议移除 `command`，避免策略学习速度跟踪而不是恢复。若你决定保留 command，则所有 `42/210` 都应统一改成 `45/225`，不能混用。

### 4.2 13 维 contact target

推荐固定顺序如下，训练、日志、checkpoint 和部署都不能改变：

```text
[base,
 FL_hip, FR_hip, RL_hip, RR_hip,
 FL_thigh, FR_thigh, RL_thigh, RR_thigh,
 FL_calf, FR_calf, RL_calf, RR_calf]
```

接触 target 是二值标签：

```python
contact_target = (normal_force > contact_force_threshold).float()
```

建议阈值初始取 `5 N`，再通过日志检查不同地形和仿真步长下的接触力分布。不要直接用 `found` 作为唯一真值，因为极短接触、接触槽位和数值噪声会造成抖动；应使用最近 2～4 个 physics substeps 的最大值或平滑值。

### 4.3 4 维 mass target

第一版推荐预测四条腿的相对质量，而不是直接预测绝对 kg：

```text
mass_target = [m_FL, m_FR, m_RL, m_RR] / m_nominal_leg
```

质量 target 必须和质量随机化使用同一个定义。如果训练中只随机四条腿的总质量，可以用四维比例；如果随机的是每个 link 的质量，则先按腿聚合：

```text
m_FL = m_FL_hip + m_FL_thigh + m_FL_calf + m_FL_foot
...  # 其余三腿同理
```

建议先将 target 归一化到约 `[0.5, 1.5]`，输出层使用线性层，loss 前做 clamp 或标准化。不要在网络里直接用 `Softplus` 后拿输出和未归一化的 kg 比较，否则 auxiliary loss 的量级容易压倒 PPO。

### 4.4 actor 输入维度

完整 FR-Net 的推荐输入为：

```text
42              当前本体观测
4               MCP mass prediction
13              MCP contact prediction
16              MCP latent
----------------
75              enhanced actor input
```

MCP 的内部输入是 5 帧历史，即 `210` 维。若使用 `contact logits`，actor 可以接收 `sigmoid(logits)`；若使用 binary prediction，必须在训练时避免 hard threshold，否则会阻断梯度。推荐使用概率值，并把预测值限制在 `[0, 1]`。

---

## 5. 现有 Go2 配置如何复用

参考文件：

- [go2_cfg.py](/home/haozi/桌面/Nazarite-mjlab/Train/Nazarite/Nazarite-src/nazarite/config/robot_config/go2_cfg.py)
- [base_env_cfg.py](/home/haozi/桌面/Nazarite-mjlab/Train/Nazarite/Nazarite-src/nazarite/config/train_config/base_env_cfg.py)
- [go2_env_cfgs.py](/home/haozi/桌面/Nazarite-mjlab/Train/Nazarite/Nazarite-src/nazarite/config/train_config/env_cfgs/go2_env_cfgs.py)

`go2_cfg.py` 中可以直接复用：

- `get_go2_cfg()`；
- `GO2_ACTION_SCALE`；
- `GO2_BASE_BODY`；
- `GO2_HIP_BODIES`、`GO2_THIGH_BODIES`、`GO2_CALF_BODIES`；
- 足端 geom/site 名称；
- 当前 PD actuator stiffness、damping、effort limit。

不要为了 recovery 复制一份 XML。FR-Net 的质量随机化应通过事件修改实体/模型参数；只有在确认 mjlab 的模型参数不能在运行时安全修改时，才考虑生成多种 XML。

现有 `base_env_cfg.py` 已经提供：

- actor/critic 分离；
- actor 有噪声，critic 无噪声；
- event manager 的 reset、push、摩擦、encoder bias、base COM randomization；
- reward manager；
- termination manager；
- `ObservationTermCfg.history_length` 和 flatten history。

但 recovery 不应直接继承速度任务的 command、velocity tracking、WTW behavior 和 gait reward。应新建 recovery config，显式列出 recovery 需要的项。

---

## 6. 新增文件结构

建议最终结构如下：

```text
Nazarite-src/nazarite/
├── mdp/
│   ├── frnet_observations.py       # 42 维本体、contact/mass target、MCP 输入
│   ├── frnet_rewards.py            # 翻正、站立、接触、动作平滑奖励
│   ├── frnet_events.py             # 摔倒 reset、质量/摩擦/地形随机化
│   └── frnet_terminations.py       # timeout、成功、失败终止
├── rl/
│   ├── frnet_actor.py              # MCP + enhanced actor
│   ├── frnet_ppo.py                # PPO + auxiliary loss
│   └── frnet_storage.py            # 若不修改外置 storage，可放 wrapper
└── config/train_config/
    ├── env_cfgs/frnet_recovery_env_cfgs.py
    └── frnet_rl_cfg.py
```

如果希望保持 Nazarite 与外置库的边界清晰，优先把网络和 PPO 放在 `nazarite/rl`；只有当 `mjlab` 的 runner/storage 无法通过现有接口扩展时，才在外置库中做最小、可记录的修改。

---

## 7. 实现 `frnet_observations.py`

### 7.1 本体观测函数

不要直接在一个函数里拼接所有数据后再让 observation manager 猜 shape。更容易排查的方式是每个 term 返回固定 shape：

```python
def recovery_proprioception(env):
    ang_vel = ...       # [N, 3]
    gravity = ...       # [N, 3]
    joint_pos = ...     # [N, 12]
    joint_vel = ...     # [N, 12]
    last_action = ...   # [N, 12]
    return torch.cat((ang_vel, gravity, joint_pos, joint_vel, last_action), dim=-1)
```

更推荐分别注册五个 observation terms，这样 manager 可以分别加噪声和历史：

```python
actor_terms = {
    "base_ang_vel": ObservationTermCfg(..., history_length=5),
    "projected_gravity": ObservationTermCfg(..., history_length=5),
    "joint_pos": ObservationTermCfg(..., history_length=5),
    "joint_vel": ObservationTermCfg(..., history_length=5),
    "actions": ObservationTermCfg(..., history_length=5),
}
```

要确认 mjlab 的 group history 是按 term 还是按 group 展平。当前 Nazarite 的 WTW 配置已经使用逐 term 的 `history_length` 和 `flatten_history_dim=True`，因此可直接沿用这一模式。首次接入时打印：

```python
print(env.observation_manager.group_obs_dim)
print(obs["actor"].shape)
```

### 7.2 接触标签

当前 Go2 环境已经配置了：

- `hip_ground_touch`：4 个 hip body；
- `thigh_ground_touch`：4 个 thigh body；
- `shank_ground_touch`：4 个 calf body；
- `trunk_ground_touch`：base body；
- `feet_ground_contact`：4 个足端 geom。

FR-Net 的 13 维 target 应从这四类 body/contact sensor 读取并按固定顺序拼接。建议单独写：

```python
def contact_target(env, sensor_names, force_threshold=5.0):
    # 每个 sensor 取 force 或 found，整理成 [N, 1]/[N, 4]
    base = ...
    hips = ...
    thighs = ...
    calves = ...
    return torch.cat((base, hips, thighs, calves), dim=-1)
```

实现时要先检查每个 sensor 的实际 tensor shape。`reduce="none"`、`num_slots=1`、`history_length=4` 可能产生 `[N, 4, 1, ...]` 或带历史维度的 shape；不能凭名称猜。统一在函数中 `squeeze`/`amax`，最终强制返回 `[N, 13]`。

训练中存在两份 contact：

1. `contact_target`：privileged label，只供 loss、critic 或日志；
2. `contact_prediction`：MCP 输出，只供 actor 和辅助 loss。

绝不能把 `contact_target` 注册进 actor observation，否则 actor 直接读取仿真真值，部署时会失效。

### 7.3 质量随机化与质量 target

新增 `frnet_events.randomize_leg_mass`。事件需要完成两件事：

1. 修改当前环境中四条腿对应 link 的质量；
2. 将本次随机化得到的四维归一化质量保存到 `env.frnet_mass_target`。

伪代码：

```python
def randomize_leg_mass(env, env_ids, ranges, asset_cfg):
    mass_ratio = sample_uniform(ranges, (len(env_ids), 4), env.device)
    apply_mass_to_leg_links(env, env_ids, mass_ratio, asset_cfg)
    env.frnet_mass_target[env_ids] = mass_ratio
```

重点是随机化必须在 reset/startup 时发生，且 target 必须与实际生效参数完全一致。若 MuJoCo/MuJoCo-Warp 不允许逐环境修改 `body_mass`，应在 scene 创建时生成多个质量 variant，或者先固定一组质量做 MCP 单元测试；不要伪造 label。

### 7.4 MCP 的输入缓冲

最稳妥的形式是给环境增加一个 `[num_envs, 5, 42]` 的 ring buffer：

```python
env.frnet_obs_history = torch.zeros(
    env.num_envs, 5, 42, device=env.device
)
```

每个 step：

```python
history[:, :-1] = history[:, 1:].clone()
history[:, -1] = current_proprio
mcp_input = history.reshape(env.num_envs, 5 * 42)
```

reset 时必须把对应环境的历史清零，或用 reset 后第一帧重复填充；不要保留上一条 episode 的状态。推荐第一版使用重复填充，避免 reset 后 MCP 输入全零造成 distribution shift：

```python
history[env_ids] = current_proprio[env_ids, None, :].repeat(1, 5, 1)
```

---

## 8. MCP 网络设计

### 8.1 推荐结构

建议先采用一个简单、可解释的 MLP 版本：

```text
input:  [N, 5, 42]
flatten or temporal encoder
shared MLP: 210 -> 256 -> 128
            ├── mass head:    128 -> 4
            ├── contact head: 128 -> 13 logits
            ├── latent head:  128 -> 16 mean/log_std
            └── decoder:       128 + 4 + 13 + 16 -> 42
```

第一版用 flatten MLP 最容易验证；之后可以替换为 GRU/1D temporal encoder。不要一开始同时引入 CNN、GRU、VAE 和 terrain image，否则出现问题时无法判断是时间对齐、网络结构还是 label 错误。

### 8.2 输出和 loss

```python
mass_pred = mass_head(h)                 # [N, 4]
contact_logits = contact_head(h)         # [N, 13]
contact_prob = torch.sigmoid(contact_logits)
latent_mu, latent_logvar = latent_head(h)
latent = latent_mu                       # actor 使用确定性路径
next_proprio_pred = decoder(torch.cat(
    (h, mass_pred, contact_prob, latent), dim=-1
))
```

推荐辅助损失：

```text
L_mass    = MSE(mass_pred, mass_target)
L_contact = BCEWithLogits(contact_logits, contact_target)
L_recon   = MSE(next_proprio_pred, next_proprio_target)
L_KL      = -0.5 * mean(1 + logvar - mu² - exp(logvar))
L_MCP     = λm L_mass + λc L_contact + λr L_recon + β L_KL
```

`next_proprio_target` 是下一时刻的 42 维本体观测，而不是当前观测。时间错位会让 decoder loss 看似下降但不能学到动力学信息；因此 storage 至少需要保存 `next_observations` 或在采样时显式保存下一帧。

初始超参数可以从以下范围开始，而不是直接追求论文最终数值：

```text
λm = 1.0
λc = 1.0
λr = 0.1
β  = 1e-4 ~ 1e-3
λ_aux = 0.1 ~ 0.5       # L_total 中 MCP 的总权重
```

所有 target 先标准化。每个 loss 的 scalar 都写入 TensorBoard，并监控：

```text
MCP/mass_mse
MCP/contact_bce
MCP/reconstruction_mse
MCP/kl
MCP/contact_accuracy
MCP/contact_f1
MCP/mass_relative_error
```

### 8.3 不要把 MCP 写成普通 observation function

`ObservationTermCfg(func=mcp_observation)` 只能让 manager 调用网络并拿到输出；如果这个网络没有注册在 PPO optimizer 里，它的参数不会被训练。正确做法是：

```text
FRNetActor.forward()
    └── self.mcp(...)
    └── self.policy(...)

optimizer = Adam(
    list(actor.parameters()) + list(critic.parameters()), ...
)
```

然后在 PPO update 中把 `L_MCP` 加到总 loss。MCP 是 actor 的子模块，而不是一个脱离 optimizer 的环境回调。

---

## 9. Recovery 环境配置

新增 [frnet_recovery_env_cfgs.py]，不要直接修改 `Nazarite_Velocity_Flat_Go2`。配置大致如下：

```python
def Nazarite_FRNet_Recovery_Go2(play=False):
    cfg = make_base_env_cfg(enable_wtw=False)
    cfg.scene.num_envs = 4096
    cfg.scene.entities = {"robot": get_go2_cfg()}

    # recovery 没有速度命令，删除 twist observation/command
    cfg.commands.pop("twist", None)
    cfg.observations["actor"].terms.pop("command", None)
    cfg.observations["critic"].terms.pop("command", None)

    # 重新设置 recovery actor/critic terms、sensor、events、rewards
    ...
    return cfg
```

实际代码中需按当前 `ManagerBasedRlEnvCfg` 的字段类型补齐，不能照抄普通 Python dict 后跳过 config validation。

### 9.1 传感器

至少保留：

- `hip_ground_touch`；
- `thigh_ground_touch`；
- `shank_ground_touch`；
- `trunk_ground_touch`；
- `feet_ground_contact`。

必要时增加四足 foot force 的独立 sensor，以便 reward 和 contact target 使用同一物理量。每个 sensor 的 `fields`、`reduce`、`num_slots` 和历史长度都要在 shape test 中验证。

### 9.2 Reset：随机摔倒状态

恢复训练不应从正常站立状态开始，否则策略很可能学成普通 locomotion。建议 reset 分布分阶段扩大：

```text
阶段 A：roll/pitch 接近倒地，z 保持安全高度
阶段 B：任意 yaw + 较大 roll/pitch
阶段 C：随机四元数 + 小范围位置/线速度/角速度
阶段 D：随机地形上的侧翻、仰翻、部分悬空
```

四元数必须归一化；base z 不能低到让模型初始就严重穿透地面。仿真初始穿透会产生巨大接触力，MCP label、reward 和梯度都会污染。

reset event 需要同步清理：

```text
last_action
MCP history
next-observation cache
contact smoothing buffer
success/failure counters
```

### 9.3 Termination

Recovery 训练和 play 的 termination 应分开：

- 训练：timeout、明显失控、长时间穿模/异常接触可 reset；
- play：不要在刚摔倒时立即因 bad orientation reset，否则看不到恢复过程；可只保留 timeout 和明确成功后的手动 reset。

建议使用“持续时间”而不是单帧角度判定：

```text
bad_orientation = projected_gravity_z > threshold
bad_counter += bad_orientation
terminate = bad_counter > patience
```

若每个倒地 episode 都在失败前强制 reset，策略会学不到翻身后半段。另一方面，若永远不终止严重穿模 episode，奖励和 contact label 会变成异常值；应设置 force/height/NaN 安全终止。

### 9.4 Recovery rewards

建议从 `FR-Net-main/docs/go2_recovery.md` 的 baseline reward 思路开始，再移植到 Nazarite 的 `RewardTermCfg`。第一版推荐项：

```text
orientation penalty             -0.5
upright Gaussian bonus           6.0
base height bonus                1.0
foot contact bonus               0.1
stand-pose/curriculum bonus      8.0
base angular velocity penalty   -0.05
joint torque penalty             -2e-4
joint acceleration penalty      -1e-6
joint velocity penalty           -2e-3
joint limit penalty              -10.0
action L2                       -1e-2
action rate                     -0.02
second-order smoothness         -0.05
foot stumble                    -1.0
```

这些不是必须的最终论文超参数，只是迁移 baseline 的起点。Nazarite 的 reward manager 会按 `term_value * weight * dt` 累加，因此不能把 Isaac Gym 版本的 scale 不经换算直接当成最终结果。

恢复奖励的关键是奖励翻正和支撑形成，而不是只奖励静态高度：

```python
orientation = exp(-||projected_gravity - target||² / sigma²)
height = exp(-(base_height - target_height)² / sigma_h²)
foot_contact = contact_probability_of_four_feet.sum(dim=-1)
```

若只加 upright reward，机器人可能通过跳跃短暂经过直立姿态；应同时约束角速度、动作平滑、足端接触和稳定持续时间。

---

## 10. Critic 和 privileged information

推荐 asymmetric actor-critic：

```text
actor:
    42-dim proprio history -> MCP prediction -> enhanced actor -> action

critic:
    proprio history
    + true mass target
    + true contact target
    + base linear velocity
    + terrain height/terrain identity（可选）
    -> value
```

critic 可以使用仿真器真值，因为部署时只运行 actor。但要明确 privileged 信息的边界：如果 actor observation 中出现 `mass_target`、`contact_target`、terrain height ground truth 或 MuJoCo body contact force，就不是真实 FR-Net 部署结构。

第一版建议 critic 直接读真值 mass/contact，actor 读 MCP prediction。这有助于降低 value function 的方差；之后可以做消融，测试 critic 也使用预测量的影响。

---

## 11. 修改 actor/model 接口

当前 `rl_cfg.py` 使用：

```python
RslRlModelCfg(class_name="MLPModel", hidden_dims=(512, 256, 128))
```

因此需要让 mjlab/RSL-RL 的 model factory 能找到自定义 `FRNetActor`。有两种方式：

### 方式 A：注册自定义 class（推荐）

在 Nazarite 中定义 `FRNetActor`，并把 class 注册到 mjlab 使用的 model registry/factory，配置写成：

```python
actor=RslRlModelCfg(
    class_name="FRNetActor",
    hidden_dims=(512, 256, 128),
    custom_kwargs={
        "proprio_dim": 42,
        "history_length": 5,
        "mass_dim": 4,
        "contact_dim": 13,
        "latent_dim": 16,
    },
)
```

具体字段名要以当前 `mjlab/src/mjlab/rl` 和 `mjlab/RSL-RL` 的 cfg/parser 为准；如果 `RslRlModelCfg` 没有 `custom_kwargs`，就扩展该配置类或使用专门的 `FRNetModelCfg`。

### 方式 B：自定义 runner 中直接构造

编写 `FRNetOnPolicyRunner`，继承当前 velocity runner 的 rollout/update 流程，只替换 actor-critic 构造和 update。该方式更容易控制 MCP loss，但需要理解 runner 的 checkpoint、日志、resume 和 play 接口。

actor 的 forward 必须能够返回：

```text
action distribution
value estimate
mass_pred
contact_logits
latent
reconstruction_pred
```

采样时只需要 distribution/value；update 时需要 auxiliary outputs。不要在采样和 update 中使用两套不同的 history 对齐规则。

---

## 12. PPO 与 MCP auxiliary loss

标准 PPO 的 clipped objective 保持不变：

```text
L_PPO = L_policy + c_v L_value - c_e entropy
```

FR-Net 使用：

```text
L_total = L_PPO + λ_aux L_MCP
```

推荐 update 顺序：

```python
for epoch in range(num_learning_epochs):
    for batch in storage.mini_batch_generator(...):
        action_dist, value, aux = actor_critic.evaluate(batch.obs)

        ppo_loss = compute_ppo_loss(...)
        mcp_loss = compute_mcp_loss(
            aux,
            batch.mass_target,
            batch.contact_target,
            batch.next_proprio,
        )
        loss = ppo_loss + aux_coef * mcp_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
```

注意：若 actor 使用的是 `contact_prob`、`mass_pred`、`latent`，MCP loss 的梯度会通过 actor 输入路径回传。可以按阶段控制：

```text
阶段 1：只训练 MCP，actor 输入 ground-truth 或冻结 MCP
阶段 2：冻结 MCP，actor 学习适应预测接口
阶段 3：解冻 MCP，联合训练，aux_coef 从 0.1 开始
```

如果一次 update 中 PPO 梯度和 MCP 梯度量级相差 100 倍以上，应先做 loss normalization 或调 `aux_coef`，不要盲目增大学习率。

### 12.1 rollout storage 必须新增的字段

如果外置 storage 只存 observation/action/reward/value/log_prob，至少扩展：

```text
mass_targets       [T, N, 4]
contact_targets    [T, N, 13]
next_proprio       [T, N, 42]
```

建议直接保存 `next_observations`，避免在 episode boundary 上错误地把下一条 episode 的第一帧当成上一帧的 next state：

```python
Transition.next_observations = next_obs
Batch.next_observations = storage.next_observations
```

在 [mjlab/RSL-RL] 目录中做修改前，先查清当前版本的 `Transition`、`RolloutStorage.add()`、mini-batch generator、checkpoint state dict 和 runner interface。不要直接复制 upstream rsl_rl 的 storage，因为当前工程使用的是本地 RSL-RL 依赖，字段和 API 可能不同。

如果暂时不实现 decoder，可以先删除 `next_proprio` 和 `L_recon`，只做 mass/contact auxiliary prediction；这是合理的中间版本，但文档和实验名称应写成 `FRNet-MCP-no-reconstruction`，不要把它误记为完整模型。

---

## 13. 域随机化与困难地形

### 13.1 质量随机化

至少随机：

- 四条腿质量比例；
- base COM offset；
- 足端摩擦系数；
- encoder bias；
- actuator stiffness/damping（若硬件模型允许）；
- 初始 roll/pitch/yaw、线速度和角速度。

质量随机化范围要逐步扩大，例如：

```text
warm-up:  [0.95, 1.05]
middle:   [0.80, 1.20]
final:    [0.60, 1.40]
```

真实 robot 的质量并不会在每个 episode 改变；这里的 randomization 是为了产生可识别的动力学变化，让 MCP 学习从历史响应中估计有效质量。

### 13.2 地形课程

课程顺序建议：

```text
flat ground
  -> low slope / uneven height
  -> stairs
  -> gaps / missing support
  -> narrow beam / partial support
  -> mixed challenging terrain
```

每一级都必须有足够的 fall pose 和接触标签覆盖。只把 terrain height scan 加到 critic，并不会让 FR-Net 获得 contact prediction；MCP 的 contact target 仍然必须由身体-地形接触计算得到。

困难地形中需要检查 `upright` 的参考系。当前 `rewards.py` 支持相对于 world up 或 terrain surface normal 判断；斜坡上应明确选择一种定义，并在实验记录中固定，否则同一个姿态在不同地形上会得到不一致的 orientation reward。

---

## 14. 任务注册和启动

在 `nazarite/__init__.py` 中加入独立 task：

```python
register_mjlab_task(
    task_id="Nazarite-FRNet-Recovery-Go2",
    env_cfg=Nazarite_FRNet_Recovery_Go2(),
    play_env_cfg=Nazarite_FRNet_Recovery_Go2(play=True),
    rl_cfg=frnet_go2_runner_cfg(),
    runner_cls=FRNetOnPolicyRunner,
)
```

不要覆盖现有：

- `Nazarite-Velocity-Flat-Go2`；
- `Nazarite-Velocity-Flat-Go2-WTW`。

这样可以保留 baseline，便于公平消融。训练命令以当前 mjlab 的 CLI 为准，先在 `Train/Nazarite/mjlab` 环境中运行 `train --help` 和 `play --help` 确认参数名。工程 README 当前示例使用 `uv run` 和 `--checkpoint_file`，而不同版本脚本可能使用 `--checkpoint`，不要混用命令行参数。

建议实验名：

```text
go2_recovery_plain
go2_recovery_contact_only
go2_recovery_mass_only
go2_recovery_mcp
go2_recovery_mcp_terrain
```

---

## 15. 最小测试清单

### 15.1 import/config smoke test

```bash
cd /home/haozi/桌面/Nazarite-mjlab/Train/Nazarite
uv run python -c "import nazarite; print('nazarite import ok')"
```

验证任务已注册、Go2 XML 可找到、所有 body/site/geom pattern 都能解析。

### 15.2 单环境 reset/play test

```text
num_envs = 1
headless = false
episode_length_s = short
```

人工检查：

- 初始姿态确实是摔倒，而不是站立；
- 没有明显穿模或 NaN；
- 关节 action 方向正确；
- action scale 不会让腿瞬间撞击地面；
- contact sensor 的四肢顺序正确；
- reset 后 MCP history 已清零/重复填充。

### 15.3 shape test

必须打印并断言：

```text
current_proprio.shape == [N, 42]
mcp_input.shape       == [N, 210]
mass_target.shape     == [N, 4]
contact_target.shape  == [N, 13]
actor_input.shape     == [N, 75]
action.shape          == [N, 12]
```

如果保留 command，则相应改为 45、225 和 78，并在整个工程统一。

### 15.4 MCP overfit test

固定一小批数据（例如 256 个 transition），关闭 PPO，只训练 MCP。正确结果应该是：

- mass MSE 明显下降；
- contact BCE 和 F1 明显改善；
- decoder MSE 下降；
- 同一 batch 重复训练时输出稳定。

如果 overfit 失败，优先排查 target shape、body 顺序、时间对齐和 mass randomization 是否真正生效，而不是增加网络层数。

### 15.5 PPO smoke test

```text
num_envs = 64
num_steps_per_env = 8 或 24
max_iterations = 2
```

确认：

- loss 没有 NaN；
- action distribution 的 std 有限；
- MCP loss 被记录且确实变化；
- checkpoint 可以保存和恢复；
- play 模式不会要求 `mass_target`/`contact_target`。

### 15.6 success metric

不要只看 episode reward。至少记录：

```text
翻正成功率
稳定站立持续时间
从倒地到 upright 的时间
四足接触恢复率
再次翻倒率
不同质量范围的成功率
不同地形的成功率
MCP mass/contact prediction 指标
```

---

## 16. 部署和导出

部署 actor 的真实输入应该只有：

```text
IMU angular velocity
IMU/projected gravity
12 joint positions
12 joint velocities
12 previous actions
```

部署端维护 5 帧 history，运行 MCP，再运行 policy。它不能依赖：

- MuJoCo contact sensor；
- 仿真器 body mass；
- privileged terrain height；
- critic 输入；
- training-only target buffer。

导出前做两次 forward 对比：

1. PyTorch eager actor；
2. TorchScript/ONNX actor。

检查 action 最大绝对误差、MCP mass/contact 输出误差和 history reset 行为。对于 GRU，导出时还要显式处理 hidden state；第一版用 flatten MLP 会更容易导出和部署。

---

## 17. 常见错误和对应处理

### 错误 1：把 FR-Net 写成一个 observation function

结果是 actor 可能拿到 MCP 输出，但 MCP 参数没有进入 optimizer，也没有 auxiliary loss。修复：MCP 必须是 actor/algorithm 的 `nn.Module` 子模块。

### 错误 2：actor 偷看 contact ground truth

仿真表现会非常好，部署后立即失效。修复：真值只给 critic、target 和 loss；actor 只能收到预测结果。

### 错误 3：42 维和 45 维混用

当前速度任务带 3 维 command 是 45 维；无 command 的 recovery 是 42 维。所有 history、网络输入、storage 和导出脚本必须采用同一协议。

### 错误 4：contact sensor 顺序不固定

模型可能学到错误的腿对应关系。修复：集中定义 `CONTACT_BODY_ORDER`，所有拼接和日志只引用这一常量。

### 错误 5：next observation 错一帧

decoder loss 不代表真实预测能力。修复：在环境 step 后保存当前 transition 的真实 next proprio；遇到 done 时不要跨 episode 取 next frame。

### 错误 6：reset 后历史没有清理

新 episode 会继承上一条 episode 的运动状态，MCP 看起来“有记忆”但实际上是数据污染。修复：reset event 同步清理/初始化 history。

### 错误 7：直接沿用 locomotion reward

velocity tracking、air-time、stance gait reward 会和翻身动作冲突。recovery 配置应显式删除 `twist` command 和 WTW gait terms。

### 错误 8：直接复制 FR-Net-main 的 Isaac Gym 配置

Isaac Gym 的 tensor shape、PD torque、episode reset 和 reward scale 与 mjlab 不同。应迁移“任务思想”和奖励项，再按 Nazarite 的 manager API 重新实现。

### 错误 9：MCP loss 压过 PPO

策略只会预测质量/接触，不会恢复。修复：记录每项 loss 的梯度/数值，降低 `aux_coef`，或者先冻结 MCP/actor 分阶段训练。

### 错误 10：困难地形和随机倒地同时开启

训练早期信号过于稀疏。修复：先 flat recovery，再增加坡度、台阶、间隙和窄梁，并用 curriculum 控制比例。

---

## 18. 推荐的实际开发顺序

### 第 1 天：恢复 baseline

完成独立 `Nazarite-FRNet-Recovery-Go2`，只用 42 维本体观测和普通 PPO。确认 64 环境运行 2 iteration 无 NaN，单环境能看到倒地姿态和动作响应。

### 第 2 天：接触标签和质量随机化

先不让 actor 使用预测量，只实现 target、日志和 critic privileged 输入。手动检查 13 个 contact channel 与实际身体对应；检查四维 mass ratio 与仿真模型一致。

### 第 3 天：MCP overfit

固定 rollout 数据，独立训练 MCP。必须先得到可解释的 contact F1 和 mass error，再进入 PPO。

### 第 4 天：接入 actor

actor 输入变成 75 维。先冻结 MCP 训练 policy，再逐步解冻。保留 plain PPO checkpoint 作为初始化和对照。

### 第 5 天以后：完整联合训练与困难地形

加入 decoder、联合 auxiliary loss、地形 curriculum、domain randomization 和部署导出。每次只改变一个主要因素，并保留 config/hash/checkpoint。

---

## 19. 最终验收标准

可称为“在 Nazarite 中实现了 FR-Net recovery”的最低标准是：

1. recovery task 是独立注册的任务，不破坏已有速度/WTW 任务；
2. actor 部署输入不包含任何 privileged ground truth；
3. MCP 从本体历史预测 4 维 mass 和 13 维 contact；
4. MCP 参数确实被 optimizer 更新，并有独立 auxiliary loss；
5. 训练 storage 正确保存 target 和 next observation；
6. flat、坡面和至少一种困难地形上都有 recovery success metric；
7. plain PPO、contact-only、mass-only 和 full MCP 至少完成一组消融；
8. 导出的 actor 在没有仿真接触真值的情况下仍能前向运行。

在达到这些标准前，建议把实验称为 `recovery baseline`、`MCP prototype` 或 `FRNet partial`，这样后续结果和论文复现结论会更准确。

---

## 附：建议优先阅读的现有文件

- [任务注册](/home/haozi/桌面/Nazarite-mjlab/Train/Nazarite/Nazarite-src/nazarite/__init__.py)
- [Go2 机器人配置](/home/haozi/桌面/Nazarite-mjlab/Train/Nazarite/Nazarite-src/nazarite/config/robot_config/go2_cfg.py)
- [通用环境配置](/home/haozi/桌面/Nazarite-mjlab/Train/Nazarite/Nazarite-src/nazarite/config/train_config/base_env_cfg.py)
- [Go2 速度任务配置](/home/haozi/桌面/Nazarite-mjlab/Train/Nazarite/Nazarite-src/nazarite/config/train_config/env_cfgs/go2_env_cfgs.py)
- [现有奖励实现](/home/haozi/桌面/Nazarite-mjlab/Train/Nazarite/Nazarite-src/nazarite/mdp/rewards.py)
- [依赖与 entry point 说明](/home/haozi/桌面/Nazarite-mjlab/Train/Nazarite/DEPENDENCIES.md)
- [FR-Net-main 的 baseline 说明](/home/haozi/桌面/FR-Net-main/docs/go2_recovery.md)
