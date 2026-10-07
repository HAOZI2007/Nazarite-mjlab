"""PPO runner configuration for the XZ1 WTW task."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from nazarite.config.train_config.rl_cfg import unitree_go2_normal_ppo_runner_cfg


def xz1_wtw_runner_cfg(
    experiment_name: str = "xz1_flat_wtw",
) -> RslRlOnPolicyRunnerCfg:
    """Return the standard WTW PPO setup with an XZ1 experiment name."""
    return unitree_go2_normal_ppo_runner_cfg(experiment_name=experiment_name)
