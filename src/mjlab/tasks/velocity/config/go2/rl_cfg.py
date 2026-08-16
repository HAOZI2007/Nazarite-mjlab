"""RL configuration for Unitree Go1 velocity task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_go2_RNN_ppo_runner_cfg(
  experiment_name: str = "go2_velocity",
) -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree Go2 velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      class_name="RNNModel",
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
      # 🌟 新增：激活 RSL-RL 原生 RNN 支持
      rnn_type="lstm",  # 使用 LSTM 网络
      rnn_hidden_dim=512,  # LSTM 的隐藏层大小
      rnn_num_layers=1,  # LSTM 的层数
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      class_name="RNNModel",
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
      # 🌟 新增：Critic 也同样可以配置 RNN
      rnn_type="lstm",
      rnn_hidden_dim=512,
      rnn_num_layers=1,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name=experiment_name,
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=100_000,
  )


def unitree_go2_normal_ppo_runner_cfg(
  experiment_name: str = "go2_velocity",
) -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree Go2 velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      class_name="MLPModel",
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "log",
      },
    ),
    critic=RslRlModelCfg(
      class_name="MLPModel",
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-5,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name=experiment_name,
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=15_000,
  )


def unitree_go2_ASYM_ppo_runner_cfg(
  experiment_name: str = "go2_velocity",
) -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree Go2 velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      class_name="MLPModel",
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      class_name="RNNModel",
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
      # 🌟 新增：Critic 也同样可以配置 RNN
      rnn_type="lstm",
      rnn_hidden_dim=512,
      rnn_num_layers=1,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name=experiment_name,
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=100_000,
  )
