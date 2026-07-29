"""HIMLoco 情报估计器模块 (HIM Estimator)。

HIMLoco 抛弃了传统的 Teacher-Student 蒸馏，改用对比学习（Swap Loss +
Sinkhorn-Knopp 软分配）从观测历史中恢复被部分可观测性（POMDP）所隐藏的
关键状态信息。本模块实现了 Estimator 的三个核心组件：

1. **Encoder（编码器）**: 接收堆叠的历史观测（obs_history），输出3维显式
   速度预测（Velocity）和16维隐式潜在特征（Latent）。
2. **Target（目标网络）**: 接收单步特权观测（包含真实未来状态），生成目标
   潜在特征，仅在训练时使用（推理时丢弃）。
3. **Prototype（原型聚类空间）**: 可学习的 Embedding 码本，通过 Sinkhorn-
   Knopp 算法将 Encoder 和 Target 的输出软分配到聚类中心，支撑 Swap Loss
   的交叉对比学习。

训练损失组合：
- **Estimation Loss (MSE)**: 监督显式速度预测的物理锚点损失。
- **Swap Loss (CE)**: 通过跨时空的交叉验证，使历史特征与未来特征在同一
  原型空间中互预测，驱动隐式特征的语义对齐。

References
----------
.. [1] Long, J. et al. "HIMLoco: Hierarchical Implicit Modeling for
   Generalizable Locomotion." 2024.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


@torch.no_grad()
def sinkhorn(
    out: torch.Tensor, eps: float = 0.05, iters: int = 3
) -> torch.Tensor:
    """Sinkhorn-Knopp 算法：将相似度矩阵转化为软聚类分配。

    给定样本与原型之间的（log）相似度矩阵，通过迭代行列归一化，输出一个
    近似双随机的软分配矩阵 Q。与硬分配（argmax）相比，Sinkhorn 能够保证
    每个原型被均匀使用（避免聚类坍缩），同时每个样本获得平滑的分配权重。

    该算法应用于 HIMLoco 的 Swap Loss：Encoder 和 Target 分别经过
    Sinkhorn 得到软标签 q_s 和 q_t，然后通过交叉熵进行互预测。

    Args:
        out: 相似度分数矩阵，shape ``(B, K)``，其中 B 为批次大小，
            K 为原型数量。
        eps: 熵正则化强度。值越小分配越尖锐（趋近硬分配）；值越大
            分配越平滑。默认 0.05。
        iters: Sinkhorn 迭代次数。3 次即可达到良好的近似，更多迭代
            会提高双随机精度但增加计算开销。默认 3。

    Returns:
        软分配矩阵 ``Q``，shape ``(B, K)``。每行和为 1/K（每个样本
        等权重分配给所有原型），每列和为 1/B（每个原型等权重覆盖
        所有样本），满足双随机约束。
    """
    Q = torch.exp(out / eps).T  # 转置后 shape: (K, B)
    K, B = Q.shape[0], Q.shape[1]
    Q /= Q.sum()  # 全局归一化初始化
    for _ in range(iters):
        # 行归一化：每个原型（聚类中心）获得相等的总质量
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K
        # 列归一化：每个样本获得相等的总质量
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B
    return (Q * B).T  # 还原为 (B, K)


class HIMEstimator(nn.Module):
    """HIMLoco 情报估计器：从观测历史中恢复隐藏状态信息。

    Estimator 是 HIMLoco 框架的核心创新。它接收堆叠的历史观测序列
    （如过去 N 帧的关节角度、角速度、IMU 等 proprioceptive 数据），
    通过 Encoder 预测显式的速度信息（3维线速度）和隐式的潜在特征
    （16维 Latent），然后将这些"情报"传递给 Actor 用于决策。

    这种设计避免了 Actor 直接处理长历史序列带来的梯度混乱问题，
    将"理解历史"与"执行动作"解耦为两个独立的网络模块。

    核心设计要点：
    ----------
    1. **物理锚点（Velocity MSE）**: 显式监督速度预测，为隐式特征的
       学习提供稳定的物理约束，防止表征坍缩。
    2. **对比学习（Swap Loss + Sinkhorn）**: Encoder 和历史观测的
       原型分配应该能预测 Target 对真实未来状态的分配，反之亦然。
       这种跨时空的交叉验证迫使 Encoder 学到真正与未来相关的特征。
    3. **梯度截断**: :meth:`sample_latent` 中对 Encoder 输出做了
       ``detach()``，确保 Actor 的梯度不会反向传播到 Estimator，
       防止 Actor 为了获得高分而扭曲 Estimator 的表征（表征坍缩）。

    Parameters
    ----------
    temporal_steps : int
        历史观测堆叠的时间步数（即堆叠了多少帧）。
    num_one_step_obs : int
        单步观测的维度（一帧 proprioceptive 观测的特征数）。
    enc_hidden_dims : list[int]
        Encoder MLP 的隐藏层维度。最后一层决定潜在维度。
        默认 ``[128, 64, 16]``，即最终输出 16 维 Latent + 3 维速度。
    tar_hidden_dims : list[int]
        Target MLP 的隐藏层维度。默认 ``[128, 64]``，输出与 Encoder
        相同的 Latent 维度，以在同一原型空间中进行对比。
    learning_rate : float
        Adam 优化器的学习率。默认 ``1e-3``。
    max_grad_norm : float
        梯度裁剪的最大范数。Estimator 与 PPO 共享更新步骤中的学习率
        调整，较大的裁剪值（默认 ``10.0``）给 Estimator 更多优化空间。
    num_prototype : int
        原型聚类空间的聚类中心数量。更多的原型可以提供更细粒度的
        表征，但也增加计算开销。默认 ``32``。
    temperature : float
        Swap Loss 中 log-softmax 的温度系数。较高的温度产生更平滑
        的概率分布，有助于训练的稳定性。默认 ``3.0``。
    """

    def __init__(
        self,
        temporal_steps: int,
        num_one_step_obs: int,
        enc_hidden_dims: list[int] | None = None,
        tar_hidden_dims: list[int] | None = None,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 10.0,
        num_prototype: int = 32,
        temperature: float = 3.0,
    ) -> None:
        super().__init__()
        if enc_hidden_dims is None:
            enc_hidden_dims = [128, 64, 16]
        if tar_hidden_dims is None:
            tar_hidden_dims = [128, 64]

        self.temporal_steps = temporal_steps
        self.num_one_step_obs = num_one_step_obs
        self.num_latent = enc_hidden_dims[-1]  # 潜在特征的维度（默认 16）
        self.max_grad_norm = max_grad_norm
        self.temperature = temperature

        # ==================== 1. Encoder 网络 ====================
        # 输入: 堆叠的历史观测，维度为 temporal_steps * num_one_step_obs
        # 输出: [3维速度预测, num_latent维潜在特征]
        # 设计: MLP + ELU 激活，最后一层聚合速度和隐式特征
        enc_input_dim = temporal_steps * num_one_step_obs
        enc_layers: list[nn.Module] = []
        for hidden_dim in enc_hidden_dims[:-1]:
            enc_layers.extend(
                [nn.Linear(enc_input_dim, hidden_dim), nn.ELU()]
            )
            enc_input_dim = hidden_dim
        # 最终线性层：同时输出速度（3维）和潜在特征（num_latent 维）
        enc_layers.append(
            nn.Linear(enc_input_dim, self.num_latent + 3)
        )
        self.encoder = nn.Sequential(*enc_layers)

        # ==================== 2. Target 网络 ====================
        # 输入: 单步特权观测（包含真实的下一个状态信息）
        # 输出: num_latent 维目标潜在特征
        # 设计: 仅在训练时使用，为 Encoder 提供对比学习的目标
        tar_input_dim = num_one_step_obs
        tar_layers: list[nn.Module] = []
        for hidden_dim in tar_hidden_dims:
            tar_layers.extend(
                [nn.Linear(tar_input_dim, hidden_dim), nn.ELU()]
            )
            tar_input_dim = hidden_dim
        tar_layers.append(nn.Linear(tar_input_dim, self.num_latent))
        self.target = nn.Sequential(*tar_layers)

        # ==================== 3. Prototype 原型聚类空间 ====================
        # 可学习的 Embedding 码本，每个原型是一个 num_latent 维向量
        # 在训练过程中，Encoder 和 Target 的输出被软分配到这些原型上
        # Sinkhorn-Knopp 算法保证每个原型被均匀使用，避免坍缩
        self.proto = nn.Embedding(num_prototype, self.num_latent)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def sample_latent(
        self, obs_history: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """供 Actor 调用的推理接口：提取速度与潜在特征（梯度截断）。

        此方法在 Rollout 采集和 PPO 更新时由 Actor 调用，用于获取
        Estimator 从历史观测中提取的"情报"（速度 + Latent）。

        **关键设计**：对 Encoder 的输入做了 ``.detach()`` 处理，
        确保 Actor 的梯度不会反传到 Estimator。这防止了：
        - Actor 为了获得更高奖励而扭曲 Estimator 的表征
        - 表征坍缩（Representation Collapse）

        Args:
            obs_history: 堆叠的历史观测，shape 为
                ``(B, temporal_steps * num_one_step_obs)``。

        Returns:
            元组 ``(velocity, latent)``：
            - **velocity**: shape ``(B, 3)``，预测的线速度（物理锚点）。
            - **latent**: shape ``(B, num_latent)``，L2归一化后的隐式
              潜在特征，捕捉无法被速度显式表达的隐藏状态信息。
        """
        parts = self.encoder(obs_history.detach())
        vel, z = parts[..., :3], parts[..., 3:]
        # L2 归一化：将潜在特征投影到单位超球面上，
        # 使得与原型之间的相似度计算等价于余弦相似度
        z = F.normalize(z, dim=-1, p=2)
        return vel.detach(), z.detach()

    def update(
        self,
        obs_history: torch.Tensor,
        next_critic_obs: torch.Tensor,
        lr: float | None = None,
    ) -> tuple[float, float]:
        """执行一次 Estimator 的参数更新。

        在 PPO 的每个 mini-batch 更新循环中被调用。同时计算：
        1. **速度估计损失（Estimation Loss）**: MSE(pred_vel, vel_true)
        2. **交换预测损失（Swap Loss）**: 基于 Sinkhorn 软分配和
           原型聚类的交叉对比学习损失

        Args:
            obs_history: 堆叠的历史观测，shape
                ``(B, temporal_steps * num_one_step_obs)``。
            next_critic_obs: 下一步的 Critic（特权）观测，shape
                ``(B, num_critic_obs)``。从中切出真实速度（用于 MSE）
                和下一步 proprioceptive 观测（用于 Target 网络）。
            lr: 可选的学习率覆盖。如果为 None，使用优化器当前学习率。
                在 HIMPPO 中，该参数会根据自适应 KL 散度动态调整。

        Returns:
            元组 ``(estimation_loss, swap_loss)``，均为 Python float。
        """
        # ---- 动态学习率调整 ----
        # 与 PPO 的自适应 KL 散度机制联动：当 KL 散度过大/过小时，
        # PPO 调整学习率并同步传给 Estimator，让两者"双向奔赴"
        if lr is not None:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

        # ---- 从特权观测中提取真实标签 ----
        # critic_obs 的结构 (前48维 = actor单步观测):
        #   [base_lin_vel(3), base_ang_vel(3), projected_gravity(3),
        #    joint_pos(12), joint_vel(12), actions(12), command(3)]
        #   + privileged: [foot_height(4), foot_air_time(4),
        #                  foot_contact(4), foot_contact_forces(12)]
        # vel_true:  真实线速度 (前3维)，用于监督 Encoder 的速度预测
        # next_obs:  下一步的 actor 单步观测 (前48维)，输入 Target 网络
        vel_true = next_critic_obs[:, :3].detach()
        next_obs = next_critic_obs[
            :, : self.num_one_step_obs
        ].detach()

        # ---- 前向传播 ----
        # Encoder: 历史 -> 速度预测 + 潜在特征
        z_s = self.encoder(obs_history)  # (B, latent+3)
        # Target: 真实下一步 -> 目标潜在特征
        z_t = self.target(next_obs)  # (B, latent)
        pred_vel, z_s = z_s[..., :3], z_s[..., 3:]

        # L2 归一化：将潜在特征投影到单位超球面，
        # 使内积等价于余弦相似度
        z_s = F.normalize(z_s, dim=-1, p=2)
        z_t = F.normalize(z_t, dim=-1, p=2)

        # ---- 原型归一化 ----
        # 在每次更新前对原型 Embedding 做 L2 归一化，
        # 将所有原型约束在单位超球面上，保持均匀分布
        with torch.no_grad():
            w = F.normalize(self.proto.weight.data.clone(), dim=-1, p=2)
            self.proto.weight.copy_(w)

        # ---- 原型相似度打分 ----
        # 计算每个样本与每个原型之间的余弦相似度（因已归一化，内积=余弦）
        score_s = z_s @ self.proto.weight.T  # (B, num_prototype)
        score_t = z_t @ self.proto.weight.T  # (B, num_prototype)

        # ---- Sinkhorn 软标签生成 ----
        # 通过 Sinkhorn-Knopp 算法将相似度转换为双随机软分配
        # q_s, q_t 作为交叉熵损失的"软标签"
        with torch.no_grad():
            q_s = sinkhorn(score_s)  # Encoder 侧的软分配
            q_t = sinkhorn(score_t)  # Target 侧的软分配

        # ---- Log-Softmax 概率分布 ----
        log_p_s = F.log_softmax(score_s / self.temperature, dim=-1)
        log_p_t = F.log_softmax(score_t / self.temperature, dim=-1)

        # ==================== Swap Loss (交换预测损失) ====================
        # 核心思想：跨越时空的交叉验证
        # - Encoder 的软分配 q_s 应该能预测 Target 的概率分布 p_t
        # - Target 的软分配 q_t 应该能预测 Encoder 的概率分布 p_s
        # 对称化处理 (-0.5 * (方向1 + 方向2)) 确保两个方向同等重要
        # 这种方式迫使 Encoder 学到的表征真正与未来状态相关
        swap_loss = -0.5 * (q_s * log_p_t + q_t * log_p_s).mean()

        # ==================== Estimation Loss (速度估计损失) ====================
        # 显式监督速度预测，提供稳定的物理锚点
        # 这是防止整个隐式特征学习过程"漂移"或"坍缩"的关键
        estimation_loss = F.mse_loss(pred_vel, vel_true)

        # 总损失 = 速度锚点 + 对比学习
        loss = estimation_loss + swap_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return estimation_loss.item(), swap_loss.item()
