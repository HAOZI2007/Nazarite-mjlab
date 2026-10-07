"""HIM runner configuration for the XZ1 47-D observation task."""

from mjlab.rl import RslRlModelCfg
from nazarite.config.train_config.him_cfg import (
    RslRlHimActorCfg,
    RslRlHimAlgorithmCfg,
    RslRlHimRunnerCfg,
)


def xz1_him_runner_cfg(
    experiment_name: str = "xz1_him_complex",
) -> RslRlHimRunnerCfg:
    """Return a HIM runner whose one-step actor frame is exactly 47-D."""
    actor = RslRlHimActorCfg(
        num_one_step_obs=47,
        history_term_dims=(3, 3, 3, 2, 12, 12, 12),
        action_observation_slice=(35, 47),
        obs_normalization=True,
    )
    algorithm = RslRlHimAlgorithmCfg(
        estimator_target_slices=((0, 6), (9, 47), (47, 50)),
        estimator_velocity_slice=(47, 50),
    )
    return RslRlHimRunnerCfg(
        actor=actor,
        critic=RslRlModelCfg(hidden_dims=(512, 256, 128), obs_normalization=True),
        algorithm=algorithm,
        obs_groups={"actor": ("actor",), "critic": ("critic",)},
        logger="tensorboard",
        experiment_name=experiment_name,
    )


__all__ = ["xz1_him_runner_cfg"]
