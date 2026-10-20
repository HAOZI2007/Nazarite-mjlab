# AMP 接入调研：bytedance/WMP 与 haozhang04/wmp_go1

面向"把 AMP 塞进已有 PPO + world-model runner"的实现笔记。只讲代码事实与落地点，不涉及论文本身。

**取证对象（钉住 commit）**

| 仓库 | 分支 | HEAD commit | 日期 |
| --- | --- | --- | --- |
| `bytedance/WMP` | `master` | `c232c115ada4517453ebded5019078ba055456de` | 2024-12-18 |
| `haozhang04/wmp_go1` | `main` | `65cf1afd13e4792bce7b9def6f44372ecbb3cdf0` | 2026-10-09 |

下文路径均相对仓库根目录；同一路径在两仓都有时，前缀 `WMP/`、`GO1/` 区分。GitHub 链接前缀分别为
`https://github.com/bytedance/WMP/blob/master/` 与 `https://github.com/haozhang04/wmp_go1/blob/main/`。

**先说结论（后面每条都有证据）**

1. 判别器只看 **30 维/状态**，输入层是 `concat(s, s')` = **60 维**；61 维的帧只是"大容器"，其中 12 维足端位置被静默丢弃。
2. 风格奖励 = `0.01 * clamp(1 - (d-1)²/4, 0)`，然后 **直接加到** 任务奖励上；`amp_task_reward_lerp` 只当开关用，注释里的凸组合是假的。
3. `reference_state_initialization` 出厂配置里是 `False`（两仓都是），配的 `reference_state_initialization_prob = 0.85` 是**死键**，全仓无人读取；一旦打开就是 100% 复位都从动作片段采样。
4. 两仓的 AMP 内核（`amp_ppo.py` / `amp_discriminator.py` / `motion_loader.py`）**逐字节相同**，差异全在配置与数据。
5. 动作片段的 12 维关节块顺序是 **FL, FR, RL, RR**，而 A1/Go1 的 IsaacGym DOF 顺序是 **FR, FL, RR, RL**（URDF 声明序），代码里没做置换 → 左右腿对调；Go2 的 MJCF 顺序恰是 **FL, FR, RL, RR**，与片段一致，**这反而意味着 Go2 移植时"不置换"才是对的**。

---

## 0 速查表：一个 AMP 判别器牵涉的全部接口

| 概念 | 形状 / 值 | 出处 |
| --- | --- | --- |
| AMP 观测（单状态） | 30 = 12+3+3+12 | `WMP/legged_gym/envs/base/legged_robot.py:481-491` |
| 判别器输入 | 60 = `concat(s, s')` | `WMP/rsl_rl/runners/wmp_runner.py:112-116` |
| 判别器隐层 | `[1024, 512]`，ReLU，输出 1 维无激活 | `rsl_rl/algorithms/amp_discriminator.py:49-66`；`WMP/legged_gym/envs/a1/a1_amp_config.py:328` |
| 风格奖励系数 | `0.5 * 0.02 = 0.01` | `WMP/legged_gym/envs/a1/a1_amp_config.py:324` |
| `amp_task_reward_lerp` | `0.3`（只当 `>0` 开关） | `a1/a1_amp_config.py:327`；`amp_discriminator.py:96-97` |
| 归一化器 | `Normalizer(30)`，`eps=1e-4`，`clip=10` | `rsl_rl/utils/utils.py:114-131`；`wmp_runner.py:111` |
| 策略侧 replay buffer | `1e6 × 30`（config），构造默认 `1e5` | `a1/a1_amp_config.py:312`；`rsl_rl/storage/replay_buffer.py:39-50` |
| 专家预加载 | 2 000 000 条 (s, s') | `a1/a1_amp_config.py:326`；`motion_loader.py:141-152` |
| 每次 update 的 batch | 24576 = 4096×24/4，real 与 fake 各一份 | `amp_ppo.py:182-190` |
| 判别器优化器 | 与策略**共用**一个 Adam | `amp_ppo.py:91-97` |
| 梯度惩罚 | `lambda_=10`，目标 0，只对专家（**未归一化**） | `amp_discriminator.py:68-84`；`amp_ppo.py:262-263` |
| 训练频率 | 与策略同步，50 Hz（`dt=0.005×4`） | `a1/a1_amp_config.py:137` |
| 部署是否需要判别器 | **不需要** | `legged_gym/scripts/play.py:127-131` |

---

## 1 判别器看到的到底是什么：30 维

### 1.1 环境侧：30 维，且是"绝对关节角"

```python
    def get_amp_observations(self):
        joint_pos = self.dof_pos
        # foot_pos = self.foot_positions_in_base_frame(self.dof_pos).to(self.device)
        base_lin_vel = self.base_lin_vel
        base_ang_vel = self.base_ang_vel
        joint_vel = self.dof_vel
        # z_pos = self.root_states[:, 2:3]
        # if (self.cfg.terrain.measure_heights):
        #     z_pos = z_pos - torch.mean(self.measured_heights, dim=-1, keepdim=True)
        # return torch.cat((joint_pos, foot_pos, base_lin_vel, base_ang_vel, joint_vel, z_pos), dim=-1)
        return torch.cat((joint_pos, base_lin_vel, base_ang_vel, joint_vel), dim=-1)
```

`WMP/legged_gym/envs/base/legged_robot.py:481-491`，`GO1/legged_gym/envs/base/legged_robot.py:534-544`（同样 12 行，逐字相同）。
12+3+3+12 = 30，顺序 **`[joint_pos(12), base_lin_vel(3), base_ang_vel(3), joint_vel(12)]`**。

三个容易踩的点：

* `joint_pos` 是 `self.dof_pos`，即**仿真器里的绝对关节角**，不是策略观测里那种相对量（策略观测用的是
  `(self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos`，`legged_robot.py:435`）。移植时如果误用相对角，判别器与专家数据永远不会同分布。
* `base_lin_vel` / `base_ang_vel` 是机体系量（`legged_gym` 里由 `quat_rotate_inverse(self.base_quat, root_states[:,7:13])` 得到），与片段里的 31:37 列同系（见 1.2 / 5.2 的实测）。
* 这里**没有加观测噪声**，也没有历史堆叠（`noise.add_noise=False`）。整个 AMP 观测是单帧。
* 同文件里还有 `get_full_amp_observations()`（`legged_robot.py:493-501`），返回 61 维（pos+rot+joint_pos+foot_pos+lin_vel+ang_vel+joint_vel+foot_vel），但**全仓无人调用**，是死代码（`grep -rn get_full_amp_observations` 只命中定义处）。

### 1.2 动作文件侧：61 维帧布局

`rsl_rl/datasets/motion_loader.py:37-68` 用一组常量描述每帧 61 个数：

| 列区间 | 常量 | 语义 | 维度 |
| --- | --- | --- | --- |
| 0:3 | `ROOT_POS_START_IDX/END_IDX` | 根位置（片段自带的世界坐标） | 3 |
| 3:7 | `ROOT_ROT_*` | 根四元数 (x,y,z,w) | 4 |
| 7:19 | `JOINT_POSE_*` | 关节角，**绝对角** | 12 |
| 19:31 | `TAR_TOE_POS_LOCAL_*` | 足端位置（机体系） | 12 |
| 31:34 | `LINEAR_VEL_*` | 机体线速度 | 3 |
| 34:37 | `ANGULAR_VEL_*` | 机体角速度 | 3 |
| 37:49 | `JOINT_VEL_*` | 关节速度 | 12 |
| 49:61 | `TAR_TOE_VEL_LOCAL_*` | 足端速度（机体系） | 12 |

实测校验（本机解析 `trot1.txt` 等文件）：第 31 列在标注 `0.4mps` 的片段里均值 +0.42、在 `backward_0.4mps` 里 −0.41，第 34:37 列在 `turn_left_0.5radps` 里 z 分量均值 +0.53 → 与上表语义一致。

### 1.3 61 → 42 → 30：真正的切片代码

**第一刀**：载入时只保留 7:49（丢掉根位置 0:3 和足端速度 49:61，并顺带丢掉根位置）：

```python
                # Remove first 7 observation dimensions (root_pos and root_orn).
                self.trajectories.append(torch.tensor(
                    motion_data[
                        :,
                        AMPLoader.ROOT_ROT_END_IDX:AMPLoader.JOINT_VEL_END_IDX
                    ], dtype=torch.float32, device=device))
                self.trajectories_full.append(torch.tensor(
                        motion_data[:, :AMPLoader.JOINT_VEL_END_IDX],
                        dtype=torch.float32, device=device))
```

`motion_loader.py:115-123`。`trajectories` = 列 7:49 = 42 列 = `[joint_pos(12), toe_pos(12), lin_vel(3), ang_vel(3), joint_vel(12)]`；
`trajectories_full` = 列 0:49 = 49 列，供"参考状态初始化"用。
注意这行注释本身就承认了"丢掉前 7 维"，却把 `observation_dim` 的 `-12` 留给足端位置，是理解这段代码最容易卡住的地方。

**第二刀**：`observation_dim` 语义 —— 它减掉的 12 是**足端位置**，不是根位置：

```python
    @property
    def observation_dim(self):
        """Size of AMP observations."""
        return self.trajectories[0].shape[1] - 12
```

`motion_loader.py:364-367`：42 − 12（`TAR_TOE_POS_LOCAL_SIZE`）= **30**。

**第三刀**：训练时的真实取样，直接按常量拼"关节角 + 31:49"，跳过 19:31：

```python
            s = self.preloaded_s[idxs, AMPLoader.JOINT_POSE_START_IDX:AMPLoader.JOINT_POSE_END_IDX]
            s = torch.cat([s, self.preloaded_s[idxs, AMPLoader.LINEAR_VEL_START_IDX:AMPLoader.JOINT_VEL_END_IDX]], dim=-1)
```

`motion_loader.py:342-349`（`s_next` 同法处理 `preloaded_s_next`）。
拼出来 = 列 7:19（12）+ 列 31:49（3+3+12 = 18）= **30**，顺序恰好等于
`[joint_pos, lin_vel, ang_vel, joint_vel]`，与环境侧 `get_amp_observations()` 的 `cat` 顺序**完全一致**。

### 1.4 矛盾的消解

| 数字 | 是什么 | 谁产生 |
| --- | --- | --- |
| 61 | 磁盘上每帧的宽度（含根位姿、足端位置、足端速度） | 数据文件 `Frames` |
| 49 | `trajectories_full` 宽度（留给参考状态初始化：根位姿+关节+足端+速度） | `motion_loader.py:121-123` |
| 42 | `trajectories` 宽度（丢掉根位置与足端速度） | `motion_loader.py:116-120` |
| 30 | 判别器真正的单状态维度（再丢 12 维足端位置） | `motion_loader.py:342-349` + `:364-367` |
| 60 | 判别器输入层维度 `concat(s, s')` | `wmp_runner.py:112-116` |

所以 30 与 61 并不冲突：61 是"容器"，30 是"取出的四个块"。`observation_dim` 那个 `- 12` 容易误读成"减掉根位置 3 维"，实际减的是 `TAR_TOE_POS_LOCAL_SIZE`。

### 1.5 这段代码里两个潜伏 bug（移植时别照抄）

1. **非预加载分支维度不一致。** 同一个 `feed_forward_generator` 的 `else` 分支走
   `get_frame_at_time()`（`motion_loader.py:216-224`）→ 返回 `self.trajectories[...]` 的 42 列，
   而 `observation_dim` 声明 30、判别器输入层 60。一旦把 `preload_transitions` 关掉，这个分支会在
   `torch.cat([s, s_next])` 处因 42+42≠60 直接报错。WMP 训练路径固定 `preload_transitions=True`，
   所以从未触发。依据：`wmp_runner.py:106-110`。
2. `observation_dim` 与 `get_frame_at_time` 的宽度语义不一致（30 vs 42），说明这两条路径是两个人写的。

---

## 2 奖励塑形

### 2.1 原始代码（全文）

```python
    def predict_amp_reward(
            self, state, next_state, task_reward, normalizer=None):
        with torch.no_grad():
            self.eval()
            if normalizer is not None:
                state = normalizer.normalize_torch(state, self.device)
                next_state = normalizer.normalize_torch(next_state, self.device)

            d = self.amp_linear(self.trunk(torch.cat([state, next_state], dim=-1)))
            reward = self.amp_reward_coef * torch.clamp(1 - (1/4) * torch.square(d - 1), min=0)
            if self.task_reward_lerp > 0:
                reward = self._lerp_reward(reward, task_reward.unsqueeze(-1))
            self.train()
        return reward.squeeze(), d
```

```python
    def _lerp_reward(self, disc_r, task_r):
        # r = (1.0 - self.task_reward_lerp) * disc_r + self.task_reward_lerp * task_r
        r = disc_r + task_r
        return r
```

`rsl_rl/algorithms/amp_discriminator.py:86-104`（两仓逐字节相同）。

### 2.2 抵达 PPO 的精确公式

```
d        = D(concat(norm(s), norm(s')))                 # 线性输出，无 sigmoid
r_style  = 0.01 * max(0, 1 - (d - 1)^2 / 4)             # ∈ [0, 0.01]，d=1 时取满
r_ppo    = r_task + r_style                             # 加法，不是凸组合
```

* 乘在判别器输出项上的是 `amp_reward_coef`；`d` 本身以 `(d-1)²/4` 进入，最优目标 `d→1`。
* `r_task` 是环境 `rew_buf`（各任务奖励项加权和），在 runner 里被**整体替换**：

```python
            next_amp_obs_with_term = torch.clone(next_amp_obs)
            next_amp_obs_with_term[reset_env_ids] = terminal_amp_states
            rewards = self.alg.discriminator.predict_amp_reward(
                amp_obs, next_amp_obs_with_term, rewards, normalizer=self.alg.amp_normalizer)[0]
            amp_obs = torch.clone(next_amp_obs)
            self.alg.process_env_step(rewards, dones, infos, next_amp_obs_with_term)
```

  `WMP/rsl_rl/runners/wmp_runner.py:318-324`。注意这里是把 `rewards` 这个变量**重新赋值**为
  `predict_amp_reward(...)` 的返回值——因为 `lerp>0` 时函数内部已经把任务奖励加进去了，
  所以最终是"加法"；若 `amp_task_reward_lerp=0`，则 `r_ppo = r_style`，任务奖励被**彻底丢弃**（这是个陷阱）。
* `amp_task_reward_lerp` 的数值（0.3）**不参与任何运算**：它只在 `if self.task_reward_lerp > 0:` 里当
  布尔门。注释 `# r = (1.0 - lerp) * disc_r + lerp * task_r` 与实现 `r = disc_r + task_r` 明确不一致，
  应以代码为准（加法）。

### 2.3 量级

`amp_reward_coef = 0.5 * 0.02`（`WMP/legged_gym/envs/a1/a1_amp_config.py:324`，`GO1/...:289`）= 0.01。
任务奖励是 `rewards.scales` 各项加权（量级 O(0.1~1) 每项，多项目标和 O(1)），因此风格项最多贡献约
**1%** 的奖励幅度；早期 `d≈0` 时 `1-(0-1)²/4 = 0.75` → 0.0075。也就是说 AMP 在这里是"温和的正则"，
不是主导信号。想让风格项真正起作用，要么提 `amp_reward_coef`，要么按 2.2 改成凸组合（见第 7 节建议）。

### 2.4 终止状态的处理

```python
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        terminal_amp_states = self.get_amp_observations()[env_ids]
        self.reset_idx(env_ids)
```

`WMP/legged_gym/envs/base/legged_robot.py:288-290`；随后 `post_physics_step` 把它 return 出去
（`:319`）→ `step()`（`:201`）→ runner（`:254-256`）。runner 用它把那些"已 done"的环境的
`next_amp_obs` 覆写成**复位前的观测**，于是 `(s_t, s_terminal)` 会被当成合法迁移送进判别器与 replay buffer。
效果是：终止状态不会被当成"未来状态缺失"，而是被当作一个仍需与专家分布比较的状态（跌倒姿态因此拿低风格奖励）。

---

## 3 参考状态初始化（reference state initialization）

### 3.1 开关与装载

```python
        if self.cfg.env.reference_state_initialization:
            self.amp_loader = AMPLoader(motion_files=self.cfg.env.amp_motion_files, device=self.device, time_between_frames=self.dt)
```

`WMP/legged_gym/envs/base/legged_robot.py:132-133`（`GO1/...:164-165` 同）。
注意这里**没有**传 `preload_transitions`，用的是默认 `False` → 走"按时间采样 + 插值"的路径，
产出 49 列完整帧（`motion_loader.py:289-297` → `:241-260`）。

配置侧：

| 键 | WMP | GO1 | 说明 |
| --- | --- | --- | --- |
| `env.reference_state_initialization` | `False`（`a1_amp_config.py:52`） | `False`（`a1_amp_config.py:66`） | 基类默认也是 `False`（`legged_robot_config.py:48`） |
| `env.reference_state_initialization_prob` | `0.85`（`a1_amp_config.py:53`） | 不存在 | **死键**：全仓仅此一处出现，无任何读取 |

### 3.2 机械过程

复位分支：

```python
        # reset robot states
        if self.cfg.env.reference_state_initialization:
            frames = self.amp_loader.get_full_frame_batch(len(env_ids))
            self._reset_dofs_amp(env_ids, frames)
            self._reset_root_states_amp(env_ids, frames)
        else:
            self._reset_dofs(env_ids)
            self._reset_root_states(env_ids)
```

`legged_robot.py:355-362`（`GO1/...:400-406`）。

```python
    def _reset_dofs_amp(self, env_ids, frames):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
            frames: AMP frames to initialize motion with
        """
        self.dof_pos[env_ids] = AMPLoader.get_joint_pose_batch(frames)
        self.dof_vel[env_ids] = AMPLoader.get_joint_vel_batch(frames)
```

`legged_robot.py:732-746`。**docstring 是错的**（照抄普通复位的说明）：实际既不是 0.5:1.5 随机，速度也不是 0，
而是把片段里的关节角与关节速度**原样复制**。

```python
    def _reset_root_states_amp(self, env_ids, frames):
        ...
        # base position
        root_pos = AMPLoader.get_root_pos_batch(frames)
        root_pos[:, :2] = root_pos[:, :2] + self.env_origins[env_ids, :2]
        self.root_states[env_ids, :3] = root_pos
        root_orn = AMPLoader.get_root_rot_batch(frames)
        self.root_states[env_ids, 3:7] = root_orn
        self.root_states[env_ids, 7:10] = quat_rotate(root_orn, AMPLoader.get_linear_vel_batch(frames))
        self.root_states[env_ids, 10:13] = quat_rotate(root_orn, AMPLoader.get_angular_vel_batch(frames))
```

`legged_robot.py:800-819`。要点：

* 根位置：x,y 叠到该环境的 `env_origins` 上，**z 直接用片段里的高度**（不叠加地形高度）；
* 姿态：片段四元数原样；
* 速度：片段里的**机体系**线速度/角速度经 `quat_rotate(root_orn, ·)` 旋到世界系再写进 `root_states[:,7:13]`。

### 3.3 与普通复位的差异

| 维度 | 普通 `_reset_*` | `_reset_*_amp` |
| --- | --- | --- |
| 采样来源 | `init_state` + `env_origins`，关节角 = `default_dof_pos × U(0.5,1.5)` | 从动作片段按时间均匀采样一帧（含 `slerp` 插值） |
| 关节速度 | 置 0 | 复制片段 `joint_vel` |
| 根速度 | `torch_rand_float(-0.5, 0.5, (n,6))` | 复制片段线/角速度（旋到世界系） |
| 根高度 | `init_state.pos[2]`（+地形偏移） | 片段 `root_pos[2]`，不贴地形 |
| 时机 | 每次 `reset_idx` | 同样只在 `reset_idx`（含回合内终止），**不**在 episode 中途注入 |
| 概率 | — | 无概率：开关一开就是 100% |

### 3.4 四个坑

1. `reference_state_initialization_prob = 0.85` 是**死的**，改了没有任何效果（`grep -rn reference_state_initialization_prob` 只命中定义）。
2. 出厂是 `False`：论文里的"参考状态初始化"在这两个仓库里**默认没开**，所以"这套代码跑出来的曲线"并不含该特性。
3. 打开后是 100% 复位都从片段采样，且 z 不贴地形——在起伏地形上会直接悬空/穿地。移植到有地形的环境必须改成
   "按 `reset_z` 采样 + 地形高度对齐"，并按 `prob` 混合。
4. 因为片段里 12 维关节块是 FL,FR,RL,RR（见 5.1），而 A1/Go1 的 `dof_pos` 是 FR,FL,RR,RL，这条路径同样**没有做置换**——和判别器数据是同一个错误。

---

## 4 数据管线

### 4.1 磁盘格式

单个 `.txt` 文件，内容是一个 JSON 对象：

| 键 | 出现情况 | 实测值 | 谁在读 |
| --- | --- | --- | --- |
| `Frames` | 必有 | `N × 61` 浮点数组 | `motion_loader.py:100` |
| `FrameDuration` | 必有 | 全部 `0.02`（50 Hz） | `motion_loader.py:127` |
| `MotionWeight` | 必有 | 全部 `1.0` | `motion_loader.py:126`，`:136` 归一化成权重 |
| `LoopMode` | 必有 | 全部 `"Wrap"` | **已抓取源码子集里零命中**（`grep -rn LoopMode wmp/ go1/` 无输出） |
| `EnableCycleOffsetPosition` / `EnableCycleOffsetRotation` | 仅 `go2_motion/*` 有 | `true` / `true` | **已抓取源码子集里零命中** |

`traj_len = (shape[0] - 1) * frame_duration`（`motion_loader.py:129-130`）；`LoopMode=Wrap` 的语义靠"采样时间对
`traj_len` 取模"隐式实现（`motion_loader.py:197-201`）。

其它已核实的行为：

* 载入时**不归一化四元数**：`pose3d.QuaternionNormalize` / `motion_util.standardize_quaternion` 两行都是注释（`motion_loader.py:107-108`）。
* `reorder_from_pybullet_to_isaac`（`motion_loader.py:154-184`）**存在但调用被注释**（`motion_loader.py:101-102`），注释里写"PyBullet [FR,FL,RR,RL] → IsaacGym [FL,FR,RL,RR]"，方向与本节 5.1 的实测正好相反——别信这段注释，用 FK 验证。
* 插值：`slerp` 名字骗人，实际只对根四元数做 `quaternion_slerp`，其余列一律线性插值（`motion_loader.py:209-210`、`:299-333`）。
* 采样：`subst = time_between_frames + frame_duration`，`t = traj_len * U(0,1) - subst` 后截到 ≥0（`motion_loader.py:197-201`），所以 `s' = 采样点 + env.dt`。`env.dt = 0.005×4 = 0.02` 正好等于 `FrameDuration` → (s, s') 恰好相邻一帧。

### 4.2 配置真正加载的文件

WMP（glob，吃到目录里 4 个文件）：

```python
MOTION_FILES = glob.glob('datasets/mocap_motions/*')
```

`WMP/legged_gym/envs/a1/a1_amp_config.py:38`；用于 `env.amp_motion_files`（`:54`）与 `runner.amp_motion_files`（`:325`）。

wmp_go1（显式 9 个，A1 莫卡普 + Go2 片段**混用**）：

```python
        # MOTION_FILES = glob.glob('datasets/mocap_motions/*') + glob.glob('datasets/go2_motion/*')
        # MOTION_FILES = glob.glob('datasets/mocap_motions/*')
        MOTION_FILES = [
            'datasets/mocap_motions/trot1.txt',
            'datasets/mocap_motions/trot2.txt',
            'datasets/go2_motion/go2_backward_0.4mps.txt',
            'datasets/go2_motion/go2_backward_0.7mps.txt',
            'datasets/go2_motion/go2_left_0.5mps.txt',
            'datasets/go2_motion/go2_right_0.5mps.txt',
            'datasets/go2_motion/go2_stance_0.0mps.txt',
            'datasets/go2_motion/go2_turn_left_0.5radps.txt',
            'datasets/go2_motion/go2_turn_right_0.5radps.txt',
        ]
```

`GO1/legged_gym/envs/a1/a1_amp_config.py:37-52`。被注释掉的是 `datasets/mocap_motions_collected/{bound,flat,hop}/*`
以及 `go2_motion` 里的 forward/hop 片段。**这就是一个现成的坏味道**：判别器同时在拟合 A1 莫卡普（机身 0.29 m、
关节角 0.95/−1.79）和 Go2 片段（0.31 m、0.79/−1.70），两个分布的"专家"在 30 维空间里并不重合。

### 4.3 两仓 `datasets/` 树

WMP（只有 4 个文件）：

```
datasets/
└── mocap_motions/
    ├── hop1.txt   hop2.txt   trot1.txt   trot2.txt      # 4 × 501 帧
```

wmp_go1（40 个文件，三层）：

```
datasets/
├── mocap_motions/                 hop1 hop2 trot1 trot2 .txt                (4, 与 WMP 同名同字节数)
├── go2_motion/                    17 个 go2_*.txt
│     backward_{0.4,0.7,1.0}mps, forward_{0.4,0.8,1.3}mps,
│     left_{0.2,0.5}mps, right_{0.2,0.5}mps, stance_0.0mps,
│     turn_left_{0.2,0.5,0.8}radps, turn_right_{0.2,0.5,0.8}radps
└── mocap_motions_collected/
      ├── bound/  7 个
      ├── flat/   6 个
      └── hop/    6 个
      （backward_0.6mps, forward_0.6mps, forward_1.0mps[仅 bound], left_0.4mps,
        right_0.4mps, turn_ccw_0.8radps, turn_cw_0.8radps）
```

### 4.4 逐文件实测（帧数 / 时长 / root z / 前向速度）

`go2_motion/`（全部 `FrameDuration=0.02`, `LoopMode=Wrap`, `MotionWeight=1.0`）：

| 文件 | 字节 | 帧数 | 时长 s | mean root_z | mean vx (col 31) |
| --- | --- | --- | --- | --- | --- |
| go2_backward_0.4mps | 249344 | 149 | 2.96 | 0.312 | −0.41 |
| go2_backward_0.7mps | 415750 | 250 | 4.98 | 0.309 | −0.69 |
| go2_backward_1.0mps | 414890 | 250 | 4.98 | 0.315 | −1.04 |
| go2_forward_0.4mps | 249241 | 149 | 2.96 | 0.319 | +0.42 |
| go2_forward_0.8mps | 412782 | 249 | 4.96 | 0.320 | +0.85 |
| go2_forward_1.3mps | 411719 | 249 | 4.96 | 0.326 | +1.33 |
| go2_left_0.2mps | 249418 | 149 | 2.96 | 0.314 | −0.00 |
| go2_left_0.5mps | 415559 | 250 | 4.98 | 0.312 | +0.00 |
| go2_right_0.2mps | 249419 | 149 | 2.96 | 0.314 | +0.00 |
| go2_right_0.5mps | 415468 | 250 | 4.98 | 0.313 | +0.01 |
| go2_stance_0.0mps | 255850 | 147 | 2.92 | 0.311 | −0.00 |
| go2_turn_left_0.2radps | 418360 | 249 | 4.96 | 0.310 | +0.01 |
| go2_turn_left_0.5radps | 419015 | 250 | 4.98 | 0.310 | +0.00 |
| go2_turn_left_0.8radps | 418718 | 250 | 4.98 | 0.311 | +0.00 |
| go2_turn_right_0.2radps | 420223 | 250 | 4.98 | 0.310 | +0.01 |
| go2_turn_right_0.5radps | 419750 | 250 | 4.98 | 0.310 | +0.00 |
| go2_turn_right_0.8radps | 419063 | 250 | 4.98 | 0.311 | +0.00 |

`mocap_motions/`（两仓同字节数）：

| 文件 | 字节 | 帧数 | root_z | mean vx |
| --- | --- | --- | --- | --- |
| hop1.txt | 541995 | 501 | 0.312 | +0.78 |
| hop2.txt | 542033 | 501 | 0.324 | +0.97 |
| trot1.txt | 544949 | 501 | 0.290 | +0.39 |
| trot2.txt | 544540 | 501 | 0.285 | +0.80 |

`mocap_motions_collected/`：18 个文件全部 500 帧 / 9.98 s，字节 606~615 KB，root_z 0.347~0.372，vx 与文件名
大体吻合（`flat/forward_0.6mps` 实测 mean vx +0.80，与文件名 0.6 不符——见 8 节）。

**归属判定**：站高 0.29~0.33 m 且关节角均值 `thigh≈0.77~1.0 / calf≈−1.58~−1.81` 的两组——`mocap_motions/`
是 **A1**（与 A1 配置 `init_state z=0.35`、A1 腿长一致），`go2_motion/` 是 **Go2**（站高 0.31，与 Go2 默认
0.32 一致，且文件名就是 go2）；`mocap_motions_collected/` 站高 0.35~0.37，明显比 Go2 高，**既非 Go2 也非 A1**
（最像 Go1，但未核实，见 8 节）。两仓都**没有** Go1 专用片段。

---

## 5 移植到 Go2 的硬伤

### 5.1 三处关节顺序，两套约定

| 载体 | 顺序 | 证据 |
| --- | --- | --- |
| A1 URDF（IsaacGym 资产） | FR, FL, RR, RL | `<joint name=...>` 声明序：`FR_hip` in `WMP/resources/robots/a1/urdf/a1.urdf:139`、`FL_hip:261`、`RR_hip:383`、`RL_hip:505` |
| Go1 URDF | FR, FL, RR, RL | `GO1/resources/robots/go1/urdf/go1.urdf`，12 个 revolute 依次为 FR×3, FL×3, RR×3, RL×3 |
| Go2 MJCF（本仓 `MJCF-Manager/Robots/GO2/xmls/go2.xml`） | **FL, FR, RL, RR** | `<joint name="FL_hip_joint">:82`、`FR:117`、`RL:153`、`RR:189`；`mujoco` 读入后 `mj_joint(i).name` 同序（实测 `nq=19, njnt=13`） |
| 动作片段的 12 维关节块 | **FL, FR, RL, RR** | 见 5.2 FK 实测 |

A1 与 Go1 的 URDF 关节几何几乎相同（`a1.urdf`：hip origin `(±0.183, ∓0.047, 0)`、thigh `(0, ∓0.08505, 0)`、
calf `(0,0,−0.2)`；`go1.urdf`：`(±0.1881, ∓0.04675, 0)`、`(0, ∓0.08, 0)`、`(0,0,−0.213)`；两者限位都是
hip ±0.802851、thigh [−1.047198, 4.188790]、calf [−2.696534, −0.916298]）。

### 5.2 FK 实测：片段到底按什么顺序排

用 `a1.urdf` 前向运动学复算 `trot1.txt` 的每帧足端位置，与被存储的 19:31 列比对。因为足端点的
精确偏移未知，对每条腿用"沿小腿轴的最小二乘偏移 `d`"消掉这个未知量——**顺序正确时四腿解出同一个 `d` 且残差≈0**：

| 解释 | 第 0 帧拟合 `d` (m) | 第 0 帧残差 (mm) | 第 100 帧 `d` (m) | 第 100 帧残差 (mm) |
| --- | --- | --- | --- | --- |
| **[FL,FR,RL,RR]** | 0.2002 / 0.2099 / 0.2106 / 0.2071 | 12.9 / 8.4 / 7.4 / 10.8 | 0.2117 / 0.2045 / 0.2054 / 0.2124 | 5.6 / 12.1 / 11.8 / 3.7 |
| [FR,FL,RR,RL] | 0.1891 / 0.2027 / 0.2061 / 0.2043 | 265.9 / 261.7 / 266.2 / 262.1 | 0.2080 / 0.1879 / 0.1957 / 0.2094 | 266.1 / 260.8 / 266.1 / 261.3 |

结论：**12 维关节块与 12 维足端块都是 FL, FR, RL, RR**。第二种解释的残差集中在 y 方向（≈±260 mm），
正是"左右腿对调"的特征。旁证：`go2_motion/go2_forward_0.4mps.txt` 第 0 帧足端 y 依次为
`+0.103, −0.098, +0.079, −0.086`（左、右、左、右），同一约定。

于是：

* A1/Go1 的 `dof_pos` 顺序（FR,FL,RR,RL）与片段顺序不一致 → **判别器的 policy 半边与 expert 半边左右腿互换**，
  而 `_reset_dofs_amp` 也把片段关节角直接塞进错误的槽位。代码里没有任何置换（`reorder_from_pybullet_to_isaac`
  被注释，且其注释描述的方向是反的）。
* Go2 的 MJCF 顺序（FL,FR,RL,RR）与片段一致 → 移植到 Go2 时**保持原样（不置换）反而是自洽的**。
  这一条很容易被"照着 WMP 抄"抄错：有人会热心加一个置换，结果把 Go2 弄成错的那一半。

### 5.3 关节数 / 限位 / 默认角

| 项 | A1（配置） | Go2 片段实测均值 | Go2 MJCF 限位 | 本仓 Go2 默认（用户给） |
| --- | --- | --- | --- | --- |
| 关节数 | 12 | 12 | 12 | 12 |
| hip | ±0.1（FL/RL +0.1，FR/RR −0.1） | −0.12 ~ −0.18（FL、RL），+0.13 ~ +0.22（FR、RR） | ±1.0472 | — |
| front thigh | 0.8 | 0.71 ~ 0.90 | [−1.5708, 3.4907] | 0.8 |
| rear thigh | 1.0 | 0.80 ~ 0.95 | [−0.5236, 4.5379] | 1.0 |
| calf | −1.5 | −1.56 ~ −1.77 | [−2.7227, −0.83776] | −1.5 |
| 站高 | 0.35（`init_state`） | 0.309 ~ 0.326 | — | 0.32 |

A1 侧数值：`WMP/legged_gym/envs/a1/a1_amp_config.py:91-106`（`GO1/...:116-133` 相同）；Go2 MJCF 限位：
`go2.xml:11-24`（`abduction ±1.0472`、`front_hip [−1.5708,3.4907]`、`back_hip [−0.5236,4.5379]`、`knee [−2.7227,−0.83776]`）。

### 5.4 姿态 gap：把片段"放"到 Go2 上量一遍

用本仓 Go2 模型（足端球半径 0.022 m）把不同姿态摆上去，量足端离地高度（负数=穿地）：

| 姿态 | base z | FL / FR / RL / RR 离地 (mm) |
| --- | --- | --- |
| Go2 `go2_forward_0.4` 关节均值 | 0.319 | +5.8 / +6.3 / +3.1 / +2.1 |
| A1 `trot1` 关节均值（自己的高度） | 0.290 | **−14.2 / −9.1 / −5.7 / −6.5** |
| A1 `trot1` 关节均值 | 0.320 | +15.8 / +20.9 / +24.3 / +23.5 |
| A1 `hop1` 关节均值 | 0.312 | −15.1 / −15.6 / −21.7 / −20.8 |
| 本仓 Go2 默认（0.8/1.0/−1.5, hip ±0.1） | 0.320 | **−3.5 / −3.5** / +6.1 / +6.1 |
| MJCF `home` keyframe（0.9/−1.8, hip 0） | 0.270 | −18.4 ×4 |

推论：

* Go2 自己的片段与 Go2 模型高度自洽（站高 0.31~0.32 时足端离地 2~6 mm）→ **片段本身不需要尺度缩放**，
  它已经是 Go2 的形态。
* A1 莫卡普的姿态放到 Go2 上，要么把基座压到 0.28~0.29（前足穿地 1.4 cm），要么离地 1.5~2.4 cm。
  也就是说 **A1 片段不能原样喂给 Go2 的判别器**：不是"缩放"能解决的，是腿部构型不同（A1 的 calf 均值 −1.79
  vs Go2 −1.66，thigh 0.95 vs 0.79），需要重定向（retargeting）或干脆换数据。
* 反过来，本仓 Go2 默认角（0.8/1.0/−1.5 + hip ±0.1）本身就让**前足穿地 3.5 mm**，后足悬空 6 mm；
  这个默认值不是自洽站姿，接 AMP 前建议先校准（0.9/−1.8、hip 0、z 0.27 的 `home` 又是 −18 mm，另一个极端）。

### 5.5 移植到 Go2 究竟要改什么

| # | 事项 | 结论 |
| --- | --- | --- |
| 1 | 12 维关节块顺序 | Go2 MJCF 与片段**一致**（FL,FR,RL,RR），**不要**加左右置换；但如果你的框架把动作空间排成 FR,FL,RR,RL（照抄 A1），就必须在 AMP 观测处把 12 维重排回 FL,FR,RL,RR |
| 2 | 足端 12 维 | 判别器用不到（被丢）；仅参考初始化用不到（只用根位姿+关节+速度）→ 无需移植足端公式 |
| 3 | 数值缩放 | **不需要**。角度单位与片段相同（rad），速度是机体系 m/s、rad/s，root z 0.31~0.32 与 Go2 一致 |
| 4 | 常值偏置 | 片段 hip 是 −0.12(左)/+0.13(右)、thigh 0.79/0.85、calf −1.70；若你的默认角是 ±0.1 / 0.8/1.0 / −1.5，则策略要学会一个约 0.2 rad 的 hip 偏置与 0.2 rad 的膝偏置。两种做法：(a) 把默认角改成片段均值（推荐，AMP 观测是绝对角）；(b) 保持默认角，接受策略额外负担 |
| 5 | 数据混用 | 别学 wmp_go1 把 A1 莫卡普与 Go2 片段混在一个判别器里；Go2 只用 `go2_motion/*` |
| 6 | 参考初始化 | 片段 z 是 0.31 左右，若地形有高度，必须改成 `z = terrain_height + 片段相对高度`，并按 `prob` 混合 |
| 7 | 站高 | 你的 0.32 与片段一致，无需改；但默认关节角要让足端落在 z=0（见 5.4） |

### 5.6 AMP 影响部署吗

**不影响：判别器只在训练期存在，推理与导出都不需要它。**

* 导出：`legged_gym/scripts/play.py:128` 只导出 actor（`export_policy_as_jit(ppo_runner.alg.actor_critic, path)`）。
* 推理：`play.py:123` `policy = ppo_runner.get_inference_policy(...)` → `wmp_runner.py:584-588` 返回
  `self.alg.actor_critic.act_inference`，前向签名只有 `(obs, history, wm_feature)`，没有 AMP 分支。
* `play.py:77` 还把 `train_cfg.runner.amp_num_preload_transitions = 1` 压到 1，说明推理时那段代码干脆跑不起来也不想跑。
* 但要注意：`get_amp_observations()` 里的 `base_lin_vel` 是**特权量**（仿真真值）。所以即使你想在部署时
  算风格奖励也不可能——除非用状态估计器。这也解释了为什么 AMP 的奖励、判别器、归一化器都留在训练里。

---

## 6 其它容易漏的点

| 主题 | 事实 | 出处 |
| --- | --- | --- |
| checkpoint 内容 | 保存 `model_state_dict / optimizer_state_dict / world_model_dict / wm_optimizer_state_dict / depth_predictor / iter / infos`；**`discriminator_state_dict` 与 `amp_normalizer` 两行被注释掉** | `WMP/rsl_rl/runners/wmp_runner.py:558-568` |
| load 侧 | 判别器与归一化器的加载同样被注释；`optimizer_state_dict` 照常恢复（`strict=False` 加载 actor-critic） | `wmp_runner.py:571-582` |
| 后果 | **断点续训会让判别器与归一化器从随机/初始状态重来**，而 Adam 里仍留着旧判别器参数的一阶/二阶矩 → 静默劣化。移植时务必把这两个 state 存进去 | 同上 |
| 优化器 | 策略与判别器**共用一个 Adam**：`actor_critic`（无 wd）、`amp_trunk`（`weight_decay=10e-4`）、`amp_head`（`weight_decay=10e-2`） | `rsl_rl/algorithms/amp_ppo.py:91-97` |
| 判别器损失 | LSGAN 形式：`expert_loss = MSE(D(exp), +1)`、`policy_loss = MSE(D(pol), −1)`、`amp_loss = 0.5*(两者和)`；**不是 BCE、无 sigmoid** | `amp_ppo.py:255-261` |
| 总损失 | `surrogate + vel_predict_coef*vel + value_loss_coef*value − entropy_coef*entropy + amp_loss + grad_pen_loss`（AMP 两项权重都是 1） | `amp_ppo.py:266-271` |
| 梯度裁剪 | 只裁 actor-critic：`clip_grad_norm_(self.actor_critic.parameters(), ...)`，判别器不裁 | `amp_ppo.py:276` |
| 梯度惩罚 | 只对专家数据，`lambda_=10`，目标 `‖∇‖ → 0`（不是 WGAN-GP 的 1） | `amp_discriminator.py:68-84` |
| **梯度惩罚用的是未归一化数据** | 四个张量被就地重绑为归一化版本后，`compute_grad_pen(*sample_amp_expert, ...)` 仍展开**原始 tuple** | `amp_ppo.py:249-263` |
| **归一化器的统计用"已归一化数据"更新** | `policy_state/expert_state` 在第 251/253 行已被 `normalize_torch` 重绑，第 283-284 行拿它们去 `update()` → 统计量自我喂食、向 (0,1) 漂移 | `amp_ppo.py:249-254, 282-284`；`utils.py:114-131` |
| real/fake 配比 | 两个 `feed_forward_generator` 用同一个 `mini_batch_size = num_envs*num_steps//num_mini_batches`（4096×24/4 = 24576），`zip` 成 1:1 | `amp_ppo.py:182-190` |
| 策略侧采样 | `ReplayBuffer.feed_forward_generator` 用 `np.random.choice(num_samples, size)`（默认**有放回**）；早期 `num_samples < batch` 时大量重复 | `rsl_rl/storage/replay_buffer.py:39-60` |
| 专家侧采样 | 从 2e6 条预加载 (s,s') 里均匀随机 + 时间维 `slerp` 插值 | `motion_loader.py:141-152, 335-349` |
| 终止迁移 | `amp_storage.insert` 写入**所有**迁移，包括 done 的（其 next 被换成 `terminal_amp_states`）；`not_done_idxs = (dones == False).nonzero().squeeze()` 算了但**没用** | `amp_ppo.py:152-154` |
| 历史堆叠 | **无**。AMP 观测是单帧 30 维；`(s, s')` 才是判别器的输入 | `amp_ppo.py:141`；`amp_discriminator.py:94` |
| 观测噪声 | AMP 观测取的是原始 `dof_pos/dof_vel/base_vel`，不经过 `add_noise` | `legged_robot.py:481-491` |
| 调用频率 | 每个策略步一次（50 Hz），与片段帧率一致；`time_between_frames=self.env.dt` | `wmp_runner.py:106-108`；`a1_amp_config.py:137` |
| `update()` 返回 | 7 个值，顺序：`mean_value_loss, mean_surrogate_loss, mean_vel_predict_loss, mean_amp_loss, mean_grad_pen_loss, mean_policy_pred, mean_expert_pred`；最后两个是**判别器输出均值 d**，不是损失；全部除以 `num_learning_epochs*num_mini_batches` | `amp_ppo.py:294-304`；消费处 `wmp_runner.py:353` |
| 策略观测 vs AMP 观测 | 策略用相对角 `(dof_pos − default_dof_pos)*scale`，AMP 用**绝对角** | `legged_robot.py:435` vs `:482` |
| `min_normalized_std` | `[0.05,0.02,0.05]*4` 乘以 `dof_pos_limits` 跨度，用于夹 `std` 下界 | `a1_amp_config.py:330`；`wmp_runner.py:120-122` |
| 片段里的 `MotionWeight` | 归一化成权重，但两个仓库所有文件都是 1.0 → 实际等价于均匀选 clip | `motion_loader.py:127,136` |

---

## 7 落地清单（给 PPO + world-model runner）

按依赖顺序，最小改动面：

1. **观测函数**（env 侧，新增 30 维）：`cat(dof_pos[12], base_lin_vel[3], base_ang_vel[3], dof_vel[12])`，
   用**绝对**关节角；不要加噪声、不要 stack 历史。每步算一次，同时保留"复位前的值"作为
   `terminal_amp_obs`（对应 `legged_robot.py:288-290`）。
2. **数据加载器**：解析 JSON（`Frames` 61 列 / `FrameDuration` / `MotionWeight`），
   存三份视图：`full(49)` 供参考初始化、`traj(42)`、以及训练用的 30 维切片（7:19 + 31:49）。
   预加载 (s, s') 表（2e6 条）比"在线插值"省事且与 WMP 行为一致；若要保留在线分支，务必把
   `get_frame_at_time` 也切成 30 维，别留 42 维的坑（1.5 节）。
3. **归一化器**：`Normalizer(30, eps=1e-4, clip=10)`，训练时用它归一化 s 与 s'。
   建议**修掉**原实现的 bug：用归一化**之前**的数据 `update()`。
4. **判别器**：输入 `2×30=60`，MLP `[1024,512]` + 线性输出 1（**不要**加 sigmoid）。
   若你改用 BCE，奖励公式必须跟着换成 `-log(1-D)` 一类，`(d-1)²/4` 会失效。
5. **奖励**：`r += coef * clamp(1 - (d-1)²/4, 0)`，`coef=0.01`；**别**照抄 `_lerp_reward` 的注释。
   若要真正的凸组合，写成 `r = (1-λ)·r_style_scaled + λ·r_task` 并重新调 `coef`，否则风格项只有 1% 量级。
6. **replay buffer**：`(30, 1e6)`，每步 `insert(s, s_next)`，done 的环境用 `terminal_amp_obs` 当 `s_next`。
7. **训练**：把判别器参数挂进**同一个** Adam（trunk wd 1e-3、head wd 1e-1），
   每个 minibatch 用 1:1 的 real/fake 各一份，损失 `0.5*(MSE(+1)+MSE(−1)) + 10·‖∇D(exp)‖²`；
   梯度只裁 policy 那半边；统计量回传要区分"损失"与"判别器输出均值"（第 6 节）。
8. **checkpoint**：存 `discriminator_state_dict` 与 `amp_normalizer`（`state_dict`），加载时恢复；
   否则续训就是重新开始。这条是这两个仓库最值得"对着改"的地方。
9. **参考初始化（可选）**：`prob` 生效的伯努利混合 + 地形高度对齐；
   采样源要用**完整帧**（root pos/rot/vel + joint pos/vel），且顺序按 5.1 处理好。
10. **部署**：什么都不用做；导出 actor 即可（`play.py:123-128`）。

建议偏离原实现的 5 处（都是明确的 bug/坑，不是口味问题）：

| # | 原实现 | 建议 |
| --- | --- | --- |
| 1 | `_lerp_reward` 注释与代码矛盾 | 改成真凸组合并重调系数，或删掉 `lerp` 参数只留加法 |
| 2 | 归一化器用归一化后数据更新 | 用原始数据更新 |
| 3 | 梯度惩罚用未归一化专家数据 | 改用归一化后数据（与 D 的输入一致） |
| 4 | checkpoint 不存判别器/归一化器 | 存 |
| 5 | A1/Go1 的关节块左右腿对调、`prob` 死键 | Go2 上确认顺序一致（5.1），`prob` 实现或删除 |

---

## 8 未核实

* **IsaacGym 的 DOF 顺序是否严格等于 URDF 声明序**：本机无 IsaacGym，5.1 的比较基于"URDF 声明序 = IsaacGym
  `dof_names` 顺序"这一通行假设。若某版本 IsaacGym 会重排，则"左右腿对调"的结论需要重验。可直接打印
  `env.dof_names` 一秒确证。
* **`a1_amp_config.py` 里 asset 指向的 `go1_new_2.urdf` 未下载核对**：`GO1/legged_gym/envs/a1/a1_amp_config.py:187`
  用的是 `go1_new_2.urdf`，本报告解析的是同目录的 `go1.urdf`。两者关节顺序/几何若有差异（例如髋偏置不同），
  5.1 / 5.3 的 Go1 列需要更新。
* **`mocap_motions_collected/*` 属于哪台机器**：站高 0.35~0.37、`thigh 0.63~0.90 / calf −1.24~−1.40`，
  比 A1/Go2 都"高且直"，最像 Go1 或 A1 的另一套标定；文件 `flat/forward_0.6mps` 实测 vx +0.80 与文件名
  的 0.6 不符（可能是采集时的目标速度而非实际速度）。未找到任何配置引用这 18 个文件（`MOTION_FILES`
  里相关行被注释）。
* **`LoopMode` / `EnableCycleOffset*` 是否真被别处读取**：grep 只覆盖抓下来的源码子集（`rsl_rl/`、`legged_gym/envs/`、`scripts/`、URDF），未遍历整仓；若有别的脚本读这两个键，本报告会漏。
* **片段的原始来源**：`reorder_from_pybullet_to_isaac` 的注释暗示片段来自 PyBullet，但两个仓库都没有采集脚本，
  片段是否经过 A1→Go2 的重定向无法从代码确认。5.4 的结论只基于几何复算。
* **AMP 是否在该仓库里真的"起作用"**：没有 ablation、没有训练曲线可查；`amp_task_reward_lerp=0.3`
  与 `amp_reward_coef=0.01` 的组合下风格项上限约为任务奖励的 1%，是否足以塑形无法从代码判断。
* **`world_model` 与 AMP 的耦合**：`amp_ppo.py` 的 `update()` 里同时有 `vel_predict_loss` 与 value/surrogate，
  但 world model 的 latent 前向（`wm_feature`）是喂给 actor-critic 的，与判别器无任何数据交换——
  本报告只确认了"没有耦合"，未进一步追 world model 侧的训练代码。
* **奖励量级的具体数值**：任务奖励各项权重在 `rewards.scales`（`a1_amp_config.py:220-262` 一带），
  本报告只用了"O(1)"的粗描述，未逐项求和。
