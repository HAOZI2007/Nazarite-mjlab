"""RSL-RL configuration for the Nazarite FR-Net recovery task."""

from __future__ import annotations

from dataclasses import dataclass

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


@dataclass
class FRNetActorCfg(RslRlModelCfg):
  """Additional constructor arguments consumed by :class:`FRNetActor`."""

  history_length: int = 5
  proprio_dim: int = 42
  mass_dim: int = 4
  contact_dim: int = 13
  latent_dim: int = 16
  mcp_hidden_dims: tuple[int, ...] = (256, 128)
  class_name: str = "nazarite.rl.fr_net.frnet_actor:FRNetActor"


@dataclass
class FRNetPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """Additional auxiliary-loss weights consumed by :class:`FRNetPPO`."""

  mcp_loss_coef: float = 0.25
  mass_loss_coef: float = 1.0
  contact_loss_coef: float = 1.0
  class_name: str = "nazarite.rl.fr_net.frnet_ppo:FRNetPPO"


def frnet_go2_recovery_runner_cfg(
  experiment_name: str = "go2_frnet_recovery",
) -> RslRlOnPolicyRunnerCfg:
  """Create the actor-critic and PPO settings for FR-Net recovery."""
  return RslRlOnPolicyRunnerCfg(
    actor=FRNetActorCfg(
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
    algorithm=FRNetPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-4,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      mcp_loss_coef=0.25,
      mass_loss_coef=1.0,
      contact_loss_coef=1.0,
    ),
    obs_groups={"actor": ("actor",), "critic": ("critic",)},
    experiment_name=experiment_name,
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=15_000,
  )
