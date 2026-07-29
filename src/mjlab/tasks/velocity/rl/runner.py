import wandb
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
    attach_metadata_to_onnx,
    get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner

import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import torch
import os
from typing import Any, Dict

# 导入我们手写的 HIM 算法模块
from mjlab.rl.algorithms.him_ppo import HIMPPO, HIMRolloutStorage
from mjlab.rl.modules.him_actor_critic import HIMActorCritic


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
    env: RslRlVecEnvWrapper

    def save(self, path: str, infos=None):
        super().save(path, infos)
        policy_dir, filename, onnx_path = self._get_export_paths(path)
        try:
            self.export_policy_to_onnx(str(policy_dir), filename)
            run_name: str = (
                wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
            )  # type: ignore[assignment]
            metadata = get_base_metadata(self.env.unwrapped, run_name)
            attach_metadata_to_onnx(str(onnx_path), metadata)
            if self.logger.logger_type in ["wandb"] and self.cfg["upload_model"]:
                wandb.save(str(onnx_path), base_path=str(policy_dir))
        except Exception as e:
            print(f"[WARN] ONNX export failed (training continues): {e}")


class HIMVelocityOnPolicyRunner(VelocityOnPolicyRunner):
    """专为 HIMLoco 适配的 OnPolicy Runner，继承并保留原有 ONNX 导出等机制"""

    log_dir: str | None
    writer: SummaryWriter | None
    num_steps_per_env: int
    current_learning_iteration: int
    save_interval: int
    him_alg: Any  # 使用 Any 告诉 Pylance 别多管闲事

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir, device)
        
        # 动态绑定属性
        self.log_dir = log_dir
        self.writer = None
        self.num_steps_per_env = int(train_cfg.get("num_steps_per_env", 24))
        self.save_interval = int(train_cfg.get("save_interval", 50))
        
        _init_obs_dict = self.env.get_observations()
        num_actor_obs = int(_init_obs_dict["actor"].shape[-1])
        
        if "critic" in _init_obs_dict.keys():
            num_critic_obs = int(_init_obs_dict["critic"].shape[-1])
        else:
            num_critic_obs = num_actor_obs
            
        num_actions = int(self.env.num_actions)
        
        history_length = 15
        num_one_step_obs = num_actor_obs // history_length
        
        actor_critic = HIMActorCritic(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_one_step_obs=num_one_step_obs,
            num_actions=num_actions,
            init_noise_std=float(train_cfg.get("actor", {}).get("distribution_cfg", {}).get("init_std", 1.0)),
        ).to(self.device)
        
        storage = HIMRolloutStorage(
            num_envs=self.env.num_envs,
            num_transitions=self.num_steps_per_env,
            obs_shape=[num_actor_obs],
            privileged_obs_shape=[num_critic_obs],
            actions_shape=[num_actions],
            device=self.device
        )
        
        # 实例化 HIMPPO
        self.him_alg = HIMPPO(
            actor_critic=actor_critic,
            storage=storage,
            clip_param=float(train_cfg.get("algorithm", {}).get("clip_param", 0.2)),
            desired_kl=float(train_cfg.get("algorithm", {}).get("desired_kl", 0.01)),
            learning_rate=float(train_cfg.get("algorithm", {}).get("learning_rate", 1e-3)),
            value_loss_coef=float(train_cfg.get("algorithm", {}).get("value_loss_coef", 1.0)),
            entropy_coef=float(train_cfg.get("algorithm", {}).get("entropy_coef", 0.01)),
            gamma=float(train_cfg.get("algorithm", {}).get("gamma", 0.99)),
            lam=float(train_cfg.get("algorithm", {}).get("lam", 0.95)),
            max_grad_norm=float(train_cfg.get("algorithm", {}).get("max_grad_norm", 1.0)),
        )
        # 兼容父类接口
        self.alg = self.him_alg  # type: ignore

    def get_inference_policy(self, device: str | None = None):
        """返回 HIM 的完整推理管线（Estimator + Actor）。

        HIM 的 get_policy() 只返回 actor MLP，缺少 Estimator 的速度/隐变量
        提取步骤。推理时必须走 act_inference 完整管线。
        """
        self.him_alg.eval_mode()
        if device is not None:
            self.him_alg.actor_critic.to(device)

        return self.him_alg.actor_critic.act_inference

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        log_dir = self.log_dir 
        
        if log_dir is not None and self.writer is None:
            if not hasattr(self, 'logger') or getattr(self.logger, "logger_type", None) != "wandb":
                self.writer = SummaryWriter(log_dir=log_dir, flush_secs=10)
                
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs_dict = self.env.get_observations()
        obs = obs_dict["actor"].to(self.device)
        critic_obs = obs_dict["critic"].to(self.device) if "critic" in obs_dict.keys() else obs

        self.him_alg.actor_critic.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()

            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.him_alg.act(obs, critic_obs)
                    
                    next_obs_dict, rewards, dones, infos = self.env.step(actions)  # type: ignore
                    
                    next_obs = next_obs_dict["actor"].to(self.device)
                    next_critic_obs = next_obs_dict["critic"].to(self.device) if "critic" in next_obs_dict.keys() else next_obs
                    rewards, dones = rewards.to(self.device), dones.to(self.device)

                    real_next_critic_obs = next_critic_obs.clone()
                    
                    if "terminal_observation" in infos:
                        term_obs = infos["terminal_observation"]
                        if isinstance(term_obs, dict) and "critic" in term_obs:
                            term_critic = term_obs["critic"].to(self.device)
                            term_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                            if len(term_ids) > 0:
                                real_next_critic_obs[term_ids] = term_critic[term_ids]

                    # type: ignore 让 Pylance 忽略参数数量和类型的检查
                    if hasattr(self.him_alg, 'process_env_step_him'):
                        self.him_alg.process_env_step_him(rewards, dones, infos, real_next_critic_obs)  # type: ignore
                    else:
                        self.him_alg.process_env_step(rewards, dones, infos)  # type: ignore

                    obs = next_obs
                    critic_obs = next_critic_obs

                    if 'episode' in infos:
                        ep_infos.append(infos['episode'])
                    cur_reward_sum += rewards
                    cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0

            stop = time.time()
            collection_time = stop - start

            start = stop
            self.him_alg.compute_returns(critic_obs)  # type: ignore
            
            mean_value_loss, mean_surrogate_loss, mean_estimation_loss, mean_swap_loss = self.him_alg.update()
            
            stop = time.time()
            learn_time = stop - start

            if log_dir is not None:
                # 1. 计算平均奖励与回合长度
                mean_reward = float(torch.tensor(list(rewbuffer)).mean().item()) if len(rewbuffer) > 0 else 0.0
                mean_length = float(torch.tensor(list(lenbuffer)).mean().item()) if len(lenbuffer) > 0 else 0.0
                
                # 2. 控制台打印进度 (每 10 次迭代打印一次)
                if it % 10 == 0:
                    print(f"Iter: {it}/{tot_iter} | Reward: {mean_reward:.3f} | "
                          f"Val Loss: {mean_value_loss:.3f} | Est Loss: {mean_estimation_loss:.4f} | Swap Loss: {mean_swap_loss:.4f}")

                # 3. 汇总所有的核心指标
                metrics: Dict[str, float] = {
                    "Train/mean_reward": mean_reward,
                    "Train/mean_episode_length": mean_length,
                    "Loss/value_function": float(mean_value_loss),
                    "Loss/surrogate": float(mean_surrogate_loss),
                    "Loss/estimation_loss": float(mean_estimation_loss),
                    "Loss/swap_loss": float(mean_swap_loss),
                }

                # 4. 写入 Tensorboard (添加 type: ignore 避免 str 传入参数名的警告)
                if self.writer is not None:
                    for k, v in metrics.items():
                        self.writer.add_scalar(str(k), v, it)  # type: ignore
                elif hasattr(self, 'logger') and hasattr(self.logger, "writer") and self.logger.writer is not None:
                    for k, v in metrics.items():
                        self.logger.writer.add_scalar(str(k), v, it)  # type: ignore

                # 5. 写入 WandB
                if getattr(getattr(self, "logger", None), "logger_type", None) in ["wandb", "WandbLogWriter"]:
                    import wandb
                    wandb.log(metrics, step=it)

            if log_dir is not None and it % self.save_interval == 0:
                self.save(os.path.join(log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        if log_dir is not None:
            self.save(os.path.join(log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))