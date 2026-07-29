"""HIMLoco Actor-Critic 策略网络模块。

HIMLoco 框架的核心创新之一是 **Actor-Estimator-Critic 三者分离架构**，
抛弃了传统的单一 Actor-Critic 网络设计。本模块实现了 :class:`HIMActorCritic`，
将 HIMEstimator 嵌入 Actor-Critic 结构中。

架构概览
--------

.. code-block:: text

                    obs_history (历史观测堆叠)
                           │
                    ┌──────┴──────┐
                    │  Estimator  │  ← 情报分析官
                    │  (Encoder)  │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │ vel (3)    │ latent (16)│
              └────────────┼────────────┘
                           │
                    ┌──────┴──────┐  + current_obs (单步观测)
                    │    Actor    │  ← 战术执行官
                    └──────┬──────┘
                           │
                       action (12维关节目标)

              privileged_obs (特权观测)
                       │
              ┌────────┴────────┐
              │     Critic      │  ← 上帝视角教练
              └────────┬────────┘
                       │
                  value (价值估计)

关键设计要点：
-----------
1. **Actor 输入分离**: Actor 不直接处理长历史序列，而是接收 Estimator
   提取的19维"情报"（3维速度 + 16维 Latent）+ 当前单步观测。这种
   设计避免了 Actor 因直接处理长历史而导致的梯度混乱问题。

2. **Critic 特权通道**: Critic 使用完整的特权观测（privileged
   observation），包含真实的速度、地形高度、摩擦力等环境状态信息。
   在训练时 Critic 可以获得"上帝视角"来给出更准确的价值估计，而在
   推理时 Critic 被丢弃，仅保留 Actor + Estimator。

3. **梯度截断**: :meth:`act` 中通过 Estimator 获取速度和 Latent 时
   使用 ``.detach()`` 截断梯度。这确保 PPO 的更新不会扭曲 Estimator
   的表征（防止表征坍缩）。

4. **动态归一化**: :class:`RunningMeanStd` 和 :class:`Normalization`
   提供在线更新的均值和方差统计，用于观测数据的实时标准化，
   提升训练稳定性。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

# HIMEstimator 已经存放在同级目录的 him_estimator.py 中
from mjlab.rl.modules.him_estimator import HIMEstimator


class RunningMeanStd:
    """在线计算均值和方差（Welford 算法变体），用于数据流的实时归一化。

    与离线统计（在整个数据集上预计算均值/方差）不同，此类维护一个
    动态更新的均值和方差，每次调用 :meth:`update` 时增量式更新统计量。
    这在强化学习中非常重要，因为观测分布会随着策略的改进而不断变化。

    实现基于 Welford 在线算法，通过维护总数 n、均值 mean 和方差 var
    三个统计量，能够在不存储全部历史数据的情况下精确计算均值和方差。

    Parameters
    ----------
    shape : tuple[int, ...]
        待归一化数据的形状（不含批次维度）。
    device : torch.device | str
        张量所在的设备。
    """

    def __init__(
        self, shape: tuple[int, ...], device: torch.device | str
    ) -> None:
        # n 初始化为一个很小的正值（1e-4），避免除零错误
        # 同时也起到一定的平滑作用，让初始几步的归一化更稳定
        self.n = 1e-4
        self.uninitialized = True
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)

    def update(self, x: torch.Tensor) -> None:
        """使用新批次数据增量更新均值和方差。

        基于 Welford 在线算法，将新批次数据与当前统计量合并。
        时间复杂度 O(N)，空间复杂度 O(1)（指额外空间）。

        Args:
            x: 新批次数据，shape ``(batch_size, *shape)``。
        """
        count = self.n
        batch_count = x.size(0)
        tot_count = count + batch_count

        old_mean = self.mean.clone()
        # 新批次均值与旧均值的差值
        delta = torch.mean(x, dim=0) - old_mean

        # 增量更新全局均值：
        # new_mean = old_mean + delta * batch_count / tot_count
        self.mean = old_mean + delta * batch_count / tot_count

        # 增量更新全局方差（基于平方和累积量 M2）：
        # M2_new = M2_old + M2_batch + delta² * n_old * n_batch / n_total
        m_a = self.var * count  # 旧数据的平方和累积量
        m_b = x.var(dim=0) * batch_count  # 新批次数据的平方和
        M2 = m_a + m_b + torch.square(delta) * count * batch_count / tot_count

        self.var = M2 / tot_count
        self.n = tot_count


class Normalization(nn.Module):
    """基于 :class:`RunningMeanStd` 的在线归一化层。

    在前向传播时，可选择是否使用当前批次数据更新统计量。
    通常的用法是：
    - 训练时调用 ``forward(x, update=True)``，同时更新统计量和归一化。
    - 推理时调用 ``forward(x, update=False)``，仅使用已记录的统计量。

    Parameters
    ----------
    shape : tuple[int, ...]
        待归一化数据的形状（不含批次维度）。
    device : torch.device | str
        张量所在的设备，默认 ``'cuda:0'``。
    """

    def __init__(
        self, shape: tuple[int, ...], device: torch.device | str = "cuda:0"
    ) -> None:
        super().__init__()
        self.running_ms = RunningMeanStd(shape=shape, device=device)

    def forward(
        self, x: torch.Tensor, update: bool = False
    ) -> torch.Tensor:
        """对输入数据进行归一化。

        Args:
            x: 输入数据，shape ``(B, *shape)``。
            update: 是否用当前批次更新统计量。默认 False。

        Returns:
            归一化后的数据，shape 与输入相同。
            公式: ``(x - mean) / (std + 1e-4)``
        """
        if update:
            self.running_ms.update(x)
        return (x - self.running_ms.mean) / (
            torch.sqrt(self.running_ms.var) + 1e-4
        )


class HIMActorCritic(nn.Module):
    """HIMLoco Actor-Critic 策略网络。

    将 :class:`HIMEstimator` 嵌入 Actor-Critic 结构，形成
    **Estimator-Actor-Critic 三者分离**的架构。

    角色分工（类比军事指挥链）：
    - **Estimator（情报分析官）**: 从历史观测中提取19维"情报"
      （3维速度 + 16维 Latent），提供给 Actor 使用。
    - **Actor（战术执行官）**: 接收 Estimator 的情报 + 当前单步观测，
      输出动作（12维关节目标位置）。不使用特权信息，仅在推理期保留。
    - **Critic（上帝视角教练）**: 直接使用完整的特权观测
      （包括真实速度、环境状态等），给出准确的价值估计 V(s)。
      仅在训练期使用，推理时丢弃。

    Parameters
    ----------
    num_actor_obs : int
        Actor 观测的总维度。包含堆叠的历史帧：
        ``num_actor_obs = num_one_step_obs * history_size``
    num_critic_obs : int
        Critic（特权）观测的维度。包含单步 proprioceptive 观测 +
        特权信息（真实速度、外部扰动等）。
    num_one_step_obs : int
        单步观测的维度（去除了历史堆叠后的一帧特征数）。
        用于从观测历史中切出当前帧和计算历史步数。
    num_actions : int
        动作空间的维度。对于 Go2 四足机器人，通常为 12（12个关节）。
    actor_hidden_dims : list[int]
        Actor MLP 的隐藏层维度。默认 ``[512, 256, 128]``。
        共三层，从高维逐渐压缩。
    critic_hidden_dims : list[int]
        Critic MLP 的隐藏层维度。默认 ``[512, 256, 128]``。
    init_noise_std : float
        动作分布对数标准差的初始值。较大的值（如 1.0）鼓励
        训练初期的探索行为。默认 ``1.0``。
    obs_term_dims : list[int]
        单步观测中各项特征的维度列表，用于从按术语优先排列的
        历史观测中正确提取当前帧。默认 ``[3, 3, 3, 12, 12, 12, 3]``，
        对应 velocity 任务的标准观测结构。
    """

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_one_step_obs: int,
        num_actions: int,
        actor_hidden_dims: list[int] | None = None,
        critic_hidden_dims: list[int] | None = None,
        init_noise_std: float = 1.0,
        obs_term_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        if actor_hidden_dims is None:
            actor_hidden_dims = [512, 256, 128]
        if critic_hidden_dims is None:
            critic_hidden_dims = [512, 256, 128]
        if obs_term_dims is None:
            obs_term_dims = [3, 3, 3, 12, 12, 12, 3]

        self.num_one_step_obs = num_one_step_obs
        # 历史堆叠步数 = 总 Actor 观测维度 / 单步观测维度
        self.history_size = int(num_actor_obs / num_one_step_obs)
        history_size = self.history_size
        self.obs_term_dims = obs_term_dims

        # ==================== Estimator（情报分析官） ====================
        # 从 obs_history 中提取 3 维速度和 16 维潜在特征
        # temporal_steps=history_size: 使用与 Actor 相同的历史长度
        self.estimator = HIMEstimator(
            temporal_steps=history_size,
            num_one_step_obs=num_one_step_obs,
        )

        # ==================== Actor（战术执行官） ====================
        # 输入维度: 单步观测 + 3维速度 + 16维 Latent
        # Actor 不直接接触长历史序列，由 Estimator 负责"阅读理解"历史
        mlp_input_dim_a = num_one_step_obs + 3 + 16
        actor_layers: list[nn.Module] = []
        curr_dim = mlp_input_dim_a
        for hidden_dim in actor_hidden_dims:
            actor_layers.extend(
                [nn.Linear(curr_dim, hidden_dim), nn.ELU()]
            )
            curr_dim = hidden_dim
        # 最终线性层：输出动作均值（每个关节一个目标位置）
        actor_layers.append(nn.Linear(curr_dim, num_actions))
        self.actor = nn.Sequential(*actor_layers)

        # ==================== Critic（上帝视角教练） ====================
        # 输入维度: 完整的特权观测
        # 输出: 单一标量价值 V(s)
        critic_layers: list[nn.Module] = []
        curr_dim = num_critic_obs
        for hidden_dim in critic_hidden_dims:
            critic_layers.extend(
                [nn.Linear(curr_dim, hidden_dim), nn.ELU()]
            )
            curr_dim = hidden_dim
        # 最终线性层：输出单一价值
        critic_layers.append(nn.Linear(curr_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        # ==================== 动作分布参数 ====================
        # 对数标准差作为可学习参数（独立于观测输入）
        # 初始化为较大的值（init_noise_std=1.0），鼓励前期探索
        # 学习过程中会自动缩小以产生更确定性的动作
        self.std = nn.Parameter(
            init_noise_std * torch.ones(num_actions)
        )

    def _extract_current_obs(
        self, obs_history: torch.Tensor
    ) -> torch.Tensor:
        """从按术语优先排列的历史观测中提取最新一帧的单步观测。

        ObservationManager 使用术语优先（term-major）方式堆叠历史：
        每个术语的 ``feature_dim × history_size`` 个元素连续排列，
        在术语内部按时间步分组（t0 的全部维度, t1 的全部维度, …）。
        例如 base_lin_vel(3) × 15 → 45 维，布局为：
        ``[x_t0, y_t0, z_t0, x_t1, y_t1, z_t1, ..., x_t14, y_t14, z_t14]``

        Returns:
            最新时间步的单步观测，shape ``(B, num_one_step_obs)``。
        """
        B = obs_history.shape[0]
        latest_parts: list[torch.Tensor] = []
        offset = 0
        for d in self.obs_term_dims:
            block_size = d * self.history_size
            block = obs_history[:, offset : offset + block_size]  # (B, d*H)
            # (B, d*H) → (B, H, d)，取最后一步 → (B, d)
            block = block.view(B, self.history_size, d)
            latest_parts.append(block[:, -1, :])
            offset += block_size
        return torch.cat(latest_parts, dim=-1)  # (B, num_one_step_obs)

    def act(
        self, obs_history: torch.Tensor
    ) -> torch.Tensor:
        """Actor 前向传播：根据观测历史生成动作。

        此方法在 Rollout 采集阶段被调用，执行以下步骤：
        1. 从 obs_history 中切出当前帧（最新一步）观测
        2. 通过 Estimator 提取速度和潜在特征（梯度截断）
        3. 拼接当前帧 + 情报 -> Actor MLP -> 动作均值
        4. 从高斯分布中采样动作

        执行完毕后，以下属性被更新：
        - ``self.action_mean``: 动作均值
        - ``self.action_std``: 动作标准差
        - ``self.action``: 采样得到的动作
        - ``self.action_log_prob``: 动作的对数概率
        - ``self.entropy``: 动作分布的熵

        Args:
            obs_history: 堆叠的历史观测，shape ``(B, num_actor_obs)``。
                按术语优先排列（term-major ordering），见
                :meth:`_extract_current_obs` 文档。

        Returns:
            采样得到的动作，shape ``(B, num_actions)``。
        """
        # 从按术语优先排列的历史中提取最新一帧观测
        current_obs = self._extract_current_obs(obs_history)

        # 从 Estimator 获取情报（梯度截断，防止 Actor 影响 Estimator）
        # est_vel:  (B, 3) - 预测的线速度
        # latent:   (B, 16) - 隐式潜在特征
        est_vel, latent = self.estimator.sample_latent(obs_history)

        # 拼接 Actor 输入：[当前观测, 预测速度, 潜在特征]
        actor_input = torch.cat([current_obs, est_vel, latent], dim=-1)

        # Actor 前向传播：输出 12 维动作均值（Go2 的 12 个关节目标）
        self.action_mean = self.actor(actor_input)
        # 【核心修复】：强行截断 std，使其最小不能低于 0.001，防止梯度更新时穿透 0 导致崩溃
        safe_std = torch.clamp(self.std, min=0.001)
        self.action_std = safe_std.expand_as(self.action_mean)

        # 构造高斯分布并采样
        dist = Normal(self.action_mean, self.action_std)
        self.action = dist.sample()
        # 对数概率（对各维度求和，得到联合对数概率）
        self.action_log_prob = dist.log_prob(self.action).sum(dim=-1)
        # 熵（用于 PPO 的熵正则化项，鼓励探索）
        self.entropy = dist.entropy().sum(dim=-1)

        return self.action

    def evaluate(self, critic_obs: torch.Tensor) -> torch.Tensor:
        """Critic 价值估计：评估给定特权观测下的状态价值 V(s)。

        仅在训练阶段使用。Critic 使用完整的特权观测（包括真实速度、
        环境参数等），给出"上帝视角"的价值判断，用于 GAE 计算和
        PPO 的 Value Loss。

        Args:
            critic_obs: Critic 特权观测，shape ``(B, num_critic_obs)``。

        Returns:
            状态价值 V(s)，shape ``(B, 1)``。
        """
        return self.critic(critic_obs)

    def get_actions_log_prob(
        self, actions: torch.Tensor
    ) -> torch.Tensor:
        """计算给定动作在当前策略下的对数概率。

        在 PPO 更新时使用：用旧策略的均值/标准差重新计算
        当前策略下旧动作的对数概率，用于计算重要性采样比率。

        Args:
            actions: 动作张量，shape ``(B, num_actions)``。

        Returns:
            各维度联合对数概率，shape ``(B,)``。
        """
        dist = Normal(self.action_mean, self.action_std)
        return dist.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_dict) -> torch.Tensor:
        """评估/推断模式下的动作输出接口，完美兼容 mjlab play 脚本的 TensorDict"""

        # 1. 从可能的字典或 TensorDict 中提取数据
        if hasattr(obs_dict, "keys"):
            if "actor" in obs_dict.keys():  # type: ignore
                raw_data = obs_dict["actor"]  # type: ignore
            else:
                raw_data = list(obs_dict.values())[0]  # type: ignore
        else:
            raw_data = obs_dict

        # 2. 强行转换成纯 PyTorch Tensor，用 isinstance 让 Pylance 闭嘴
        if isinstance(raw_data, torch.Tensor):
            clean_tensor = raw_data.clone().detach().to(self.std.device)
        else:
            clean_tensor = torch.tensor(raw_data, dtype=torch.float32, device=self.std.device)

        # 3. 补齐 batch 维度
        if clean_tensor.ndim == 1:
            clean_tensor = clean_tensor.unsqueeze(0)

        # 4. 提取当前单步观测并让 Estimator 提取隐特征与速度
        current_obs = self._extract_current_obs(clean_tensor)
        est_vel, latent = self.estimator.sample_latent(clean_tensor)

        # 5. 拼接并送入 Actor 输出确定性动作 (不加随机噪声，步态更稳)
        actor_input = torch.cat([current_obs, est_vel, latent], dim=-1)
        actions = self.actor(actor_input)

        return actions
