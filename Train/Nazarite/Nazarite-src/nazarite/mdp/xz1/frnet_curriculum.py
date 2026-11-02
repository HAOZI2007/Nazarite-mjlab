"""Success-gated pose practice and staged FR-Net regularization for XZ1.

The regularization schedule follows mjlab_playground's explore-then-smooth
approach. Pose stages are an XZ1 adaptation, not a claimed paper curriculum.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.manager_base import ManagerTermBase

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.curriculum_manager import CurriculumTermCfg


@dataclass(frozen=True)
class RecoveryStage:
    """Angles in degrees; family weights follow FALL_CASES in frnet.py."""

    side_roll: tuple[float, float]
    back_roll: tuple[float, float]
    probabilities: tuple[float, ...]
    penalty_scale: float
    joint_noise: float


RECOVERY_STAGES = (
    RecoveryStage((25, 45), (25, 45), (0.20, 0.20, 0, 0.50, 0.05, 0.05), 0.10, 0.12),
    RecoveryStage((45, 75), (45, 75), (0.25, 0.25, 0, 0.40, 0.05, 0.05), 0.20, 0.15),
    RecoveryStage(
        (75, 100), (100, 130), (0.225, 0.225, 0.15, 0.30, 0.05, 0.05), 0.35, 0.20
    ),
    RecoveryStage(
        (80, 100), (130, 160), (0.225, 0.225, 0.20, 0.25, 0.05, 0.05), 0.55, 0.25
    ),
    RecoveryStage(
        (80, 100), (160, 180), (0.225, 0.225, 0.25, 0.20, 0.05, 0.05), 0.75, 0.30
    ),
    RecoveryStage(
        (83, 97), (173, 187), (0.20, 0.20, 0.20, 0.20, 0.15, 0.05), 1.0, 0.35
    ),
)

# Physical joint limits and contact safety retain their full weights throughout.
REGULARIZED_REWARDS = (
    "joint_velocity_limit",
    "joint_acc",
    "joint_vel",
    "action_rate",
    "joint_torques",
)


class RecoveryCurriculum(ManagerTermBase):
    """Require each practiced fall family to recover before increasing difficulty.

    The manager calls this before reset events, while terminal episode metrics
    and the episode's sampled level still exist. Rehearsal episodes never
    contribute to promotion at a harder level. Standing episodes never count.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        self.level = 0
        self.stage_started_at = env.common_step_counter
        self.trials = torch.zeros(4, device=env.device)
        self.successes = torch.zeros_like(self.trials)
        self.final_stable = torch.zeros_like(self.trials)
        self.success_rates = torch.full_like(self.trials, -1)
        self.final_rates = torch.full_like(self.trials, -1)
        self._final_weights = {
            name: env.reward_manager.get_term_cfg(name).weight
            for name in REGULARIZED_REWARDS
        }
        self.apply_stage()

    def restart(self) -> None:
        """Old checkpoints carry no evidence of mastering any pose stage."""
        self.level = 0
        self.stage_started_at = self._env.common_step_counter
        self.trials.zero_()
        self.successes.zero_()
        self.final_stable.zero_()
        self.success_rates.fill_(-1)
        self.final_rates.fill_(-1)
        self.apply_stage()

    def apply_stage(self) -> None:
        """Always derive weights from final values, never multiply cumulatively."""
        self._env.extras["frnet_curriculum_level"] = self.level
        scale = RECOVERY_STAGES[self.level].penalty_scale
        for name, weight in self._final_weights.items():
            self._env.reward_manager.get_term_cfg(name).weight = weight * scale

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | slice | None,
        min_episodes_per_case: int = 128,
        min_stage_steps: int = 12000,
        promotion_success_rate: float = 0.70,
        promotion_final_stable_rate: float = 0.60,
    ) -> dict[str, float]:
        if min_episodes_per_case < 1 or min_stage_steps < 0:
            raise ValueError("Invalid recovery curriculum sample/step minimum")
        if not (0 < promotion_final_stable_rate <= promotion_success_rate <= 1):
            raise ValueError("Invalid recovery curriculum success thresholds")
        if isinstance(env_ids, slice):
            ids = torch.arange(env.num_envs, device=env.device)[env_ids]
        else:
            ids = resolve_env_ids(env, env_ids)
        # Zero-step construction/reset and unfinished manual resets are not trials.
        terminal = getattr(env, "reset_buf", None)
        if terminal is not None and "frnet_episode_steps" in env.extras:
            valid = (
                (env.extras["frnet_episode_steps"][ids] > 0)
                & terminal[ids]
                & (env.extras["frnet_reset_level"][ids] == self.level)
            )
            kinds = env.extras["frnet_reset_kind"][ids]
            for case in range(4):
                selected = ids[valid & (kinds == case)]
                self.trials[case] += len(selected)
                self.successes[case] += (
                    env.extras["frnet_episode_success"][selected].float().sum()
                )
                self.final_stable[case] += (
                    env.extras["frnet_final_stable"][selected].float().sum()
                )

        # Belly success cannot hide a failure to roll from either side or back.
        required = torch.tensor(
            RECOVERY_STAGES[self.level].probabilities[:4], device=env.device
        ).gt(0)
        age = env.common_step_counter - self.stage_started_at
        enough = bool((self.trials[required] >= min_episodes_per_case).all())
        if enough and age >= min_stage_steps:
            self.success_rates = torch.where(
                self.trials > 0,
                self.successes / self.trials.clamp_min(1),
                -torch.ones_like(self.trials),
            )
            self.final_rates = torch.where(
                self.trials > 0,
                self.final_stable / self.trials.clamp_min(1),
                -torch.ones_like(self.trials),
            )
            mastered = bool(
                (self.success_rates[required] >= promotion_success_rate).all()
                & (self.final_rates[required] >= promotion_final_stable_rate).all()
            )
            if mastered and self.level < len(RECOVERY_STAGES) - 1:
                self.level += 1
                self.stage_started_at = env.common_step_counter
                self.success_rates.fill_(-1)
                self.final_rates.fill_(-1)
            # Fresh assessment windows avoid failures from early exploration
            # dominating the estimated rate for the rest of training.
            self.trials.zero_()
            self.successes.zero_()
            self.final_stable.zero_()
        self.apply_stage()
        result = {
            "level": float(self.level),
            "penalty_scale": RECOVERY_STAGES[self.level].penalty_scale,
            "stage_steps": float(env.common_step_counter - self.stage_started_at),
        }
        for case, name in enumerate(("left_side", "right_side", "back", "belly")):
            result[f"{name}_trials"] = self.trials[case].item()
            # Until the first completed assessment, report the current window.
            seen = self.trials[case] > 0
            success = self.successes[case] / self.trials[case].clamp_min(1)
            final = self.final_stable[case] / self.trials[case].clamp_min(1)
            result[f"{name}_success_rate"] = torch.where(
                seen, success, self.success_rates[case]
            ).item()
            result[f"{name}_final_stable_rate"] = torch.where(
                seen, final, self.final_rates[case]
            ).item()
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "level": self.level,
            "stage_started_at": self.stage_started_at,
            **{
                name: getattr(self, name).detach().cpu().clone()
                for name in (
                    "trials",
                    "successes",
                    "final_stable",
                    "success_rates",
                    "final_rates",
                )
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        level = int(state["level"])
        if state.get("version") != 1 or not 0 <= level < len(RECOVERY_STAGES):
            raise ValueError("Unsupported XZ1 recovery curriculum checkpoint")
        self.level = level
        self.stage_started_at = int(state["stage_started_at"])
        for name in (
            "trials",
            "successes",
            "final_stable",
            "success_rates",
            "final_rates",
        ):
            value = torch.as_tensor(state[name], device=self.device)
            if value.shape != (4,) or not torch.isfinite(value).all():
                raise ValueError(f"Invalid recovery curriculum tensor: {name}")
            getattr(self, name).copy_(value)
        self.apply_stage()
