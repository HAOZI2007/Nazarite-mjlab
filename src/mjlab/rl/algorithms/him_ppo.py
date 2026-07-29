import torch
import torch.nn as nn
import torch.optim as optim

class HIMRolloutStorage:
    def __init__(self, num_envs, num_transitions, obs_shape, privileged_obs_shape, actions_shape, device='cuda:0'):
        self.device = device
        self.num_transitions = num_transitions
        self.num_envs = num_envs
        self.step = 0

        # 原有 PPO 容器
        self.observations = torch.zeros(num_transitions, num_envs, *obs_shape, device=self.device)
        self.critic_observations = torch.zeros(num_transitions, num_envs, *privileged_obs_shape, device=self.device)
        self.actions = torch.zeros(num_transitions, num_envs, *actions_shape, device=self.device)
        self.rewards = torch.zeros(num_transitions, num_envs, 1, device=self.device)
        self.dones = torch.zeros(num_transitions, num_envs, 1, device=self.device).byte()
        
        # [HIMLoco 新增]：用于存储真实的下一步特权信息
        self.next_critic_observations = torch.zeros(num_transitions, num_envs, *privileged_obs_shape, device=self.device)
        
        # PPO 相关的概率与优势值
        self.actions_log_prob = torch.zeros(num_transitions, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions, num_envs, *actions_shape, device=self.device)

    def add_transitions(self, obs, critic_obs, actions, rewards, dones, values, actions_log_prob, mu, sigma, next_critic_obs):
        if self.step >= self.num_transitions:
            raise AssertionError("Rollout buffer overflow")
        
        self.observations[self.step].copy_(obs)
        self.critic_observations[self.step].copy_(critic_obs)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards.view(-1, 1))
        self.dones[self.step].copy_(dones.view(-1, 1))
        self.values[self.step].copy_(values)
        self.actions_log_prob[self.step].copy_(actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(mu)
        self.sigma[self.step].copy_(sigma)
        
        # [HIMLoco 新增]：存入下一步特权观测
        self.next_critic_observations[self.step].copy_(next_critic_obs)
        
        self.step += 1

    def clear(self):
        self.step = 0

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        """生成供 PPO 和 Estimator 同步更新的 Mini-batch 数据"""
        batch_size = self.num_envs * self.num_transitions
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        obs = self.observations.flatten(0, 1)
        critic_obs = self.critic_observations.flatten(0, 1)
        next_critic_obs = self.next_critic_observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                yield (
                    obs[batch_idx], critic_obs[batch_idx], actions[batch_idx],
                    next_critic_obs[batch_idx], values[batch_idx], advantages[batch_idx],
                    returns[batch_idx], old_actions_log_prob[batch_idx],
                    old_mu[batch_idx], old_sigma[batch_idx]
                )


class RolloutTransition:
    """临时包裹：用于存放每一次 env.step 前后产生的碎片化数据"""
    def __init__(self):
        self.observations = None
        self.critic_observations = None
        self.actions = None
        self.rewards = None
        self.dones = None
        self.values = None
        self.actions_log_prob = None
        self.action_mean = None
        self.action_sigma = None

    def clear(self):
        self.__init__()


class HIMPPO:
    def __init__(self, actor_critic, storage, clip_param=0.2, desired_kl=0.01, learning_rate=1e-3, value_loss_coef=1.0, entropy_coef=0.01, gamma=0.99, lam=0.95, max_grad_norm=1.0):
        self.actor_critic = actor_critic
        self.storage = storage
        self.clip_param = clip_param
        self.desired_kl = desired_kl
        self.learning_rate = learning_rate
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.lam = lam
        
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        
        # 实例化临时数据包裹，解决 Pylance 找不到 transition 属性的报错
        self.transition = RolloutTransition()

    def act(self, obs, critic_obs):
        """生成动作，并将前向传播的结果暂存进 transition 包裹中"""
        self.transition.observations = obs.clone()
        self.transition.critic_observations = critic_obs.clone()
        
        with torch.inference_mode():
            self.transition.actions = self.actor_critic.act(obs).detach()
            self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
            self.transition.actions_log_prob = self.actor_critic.action_log_prob.detach()
            self.transition.action_mean = self.actor_critic.action_mean.detach()
            self.transition.action_sigma = self.actor_critic.action_std.detach()
            
        return self.transition.actions

    def process_env_step_him(self, rewards, dones, infos, next_critic_obs):
        """将从 Runner 收到的步进数据（包含下一帧特权态）打包装入缓冲区"""
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones.clone()
        
        self.storage.add_transitions(
            self.transition.observations,
            self.transition.critic_observations,
            self.transition.actions,
            self.transition.rewards,
            self.transition.dones,
            self.transition.values,
            self.transition.actions_log_prob,
            self.transition.action_mean,
            self.transition.action_sigma,
            next_critic_obs  # 将拦截到的真实下一帧特权观测塞进去
        )
        self.transition.clear() # 清空包裹，准备下一步

    def compute_returns(self, last_critic_obs):
        """使用 GAE (Generalized Advantage Estimation) 算法反向计算所有步骤的优势值"""
        with torch.inference_mode():
            last_values = self.actor_critic.evaluate(last_critic_obs).detach()
            
        advantage = 0
        # 从最后一步往前倒推（时间逆序）
        for step in reversed(range(self.storage.num_transitions)):
            if step == self.storage.num_transitions - 1:
                next_values = last_values
            else:
                next_values = self.storage.values[step + 1]
            
            # 如果下一步触发了 done，那么不考虑未来的奖励（掩码设为 0）
            next_is_not_terminal = 1.0 - self.storage.dones[step].float()
            
            # TD 误差公式: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = self.storage.rewards[step] + next_is_not_terminal * self.gamma * next_values - self.storage.values[step]
            
            # GAE 优势值递归公式: A_t = delta_t + gamma * lambda * A_{t+1}
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            self.storage.returns[step] = advantage + self.storage.values[step]
            
        # 计算全局优势值并做标准化处理（提高训练稳定性）
        self.storage.advantages = self.storage.returns - self.storage.values
        self.storage.advantages = (self.storage.advantages - self.storage.advantages.mean()) / (self.storage.advantages.std() + 1e-8)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_estimation_loss = 0
        mean_swap_loss = 0

        generator = self.storage.mini_batch_generator(num_mini_batches=4, num_epochs=5)

        for (obs_batch, critic_obs_batch, actions_batch, next_critic_obs_batch,
             target_values_batch, advantages_batch, returns_batch,
             old_actions_log_prob_batch, old_mu_batch, old_sigma_batch) in generator:

            self.actor_critic.act(obs_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # 自适应 KL 散度调参
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.e-5) +
                    (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / 
                    (2.0 * torch.square(sigma_batch)) - 0.5, dim=-1)
                kl_mean = torch.mean(kl)

                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate

            # 估计器更新
            estimation_loss, swap_loss = self.actor_critic.estimator.update(
                obs_history=obs_batch, 
                next_critic_obs=next_critic_obs_batch, 
                lr=self.learning_rate
            )

            # PPO 代理损失
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = -torch.min(surrogate, surrogate_clipped).mean()

            value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_estimation_loss += estimation_loss
            mean_swap_loss += swap_loss

        self.storage.clear()
        return mean_value_loss, mean_surrogate_loss, mean_estimation_loss, mean_swap_loss

    # ---------------------------------------------------------
    # 在 HIMPPO 类中补充：模型保存、加载与策略获取接口
    # ---------------------------------------------------------
    def save(self):
        """保存模型与优化器状态（使用特有键名以避开 mjlab 原生的模型切割机制）"""
        return {
            'him_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }

    def load(self, loaded_dict, load_cfg=None, strict=True):
        """加载模型与优化器状态"""
        if 'him_state_dict' in loaded_dict:
            self.actor_critic.load_state_dict(loaded_dict['him_state_dict'], strict=strict)
        if 'optimizer_state_dict' in loaded_dict:
            self.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        return True

    def get_policy(self):
        """兼容原生的 ONNX 导出探测，防止获取不到 policy 报错"""
        return self.actor_critic.actor
    def eval_mode(self):
        """将网络切换到评估模式（禁用 dropout、batchnorm 更新等），兼容 rsl_rl 接口"""
        self.actor_critic.eval()