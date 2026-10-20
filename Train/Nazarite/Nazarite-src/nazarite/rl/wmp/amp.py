"""Adversarial motion prior: the discriminator, its reward and its update step.

The reference task guarantees a natural trot with this instead of a list of hand
tuned gait rewards: a small network learns to tell reference transitions from
policy ones, and the policy is paid for looking like the reference.  Two details
are copied from the reference implementation on purpose, because they change the
sign and scale of the gradient:

* the style reward is the discriminator's own score mapped through
  ``1 - (d - 1)^2 / 4`` and added to the task reward, not a probit or a logit;
* the discriminator is trained with an LSGAN objective (regression to +1 for
  reference transitions and -1 for policy ones) rather than a sigmoid classifier.

Two details are deliberately *not* copied: the discriminator gets its own
optimiser (the reference shares the policy's Adam, which drags it at policy speed)
and its state is checkpointed (the reference leaves it out, so a resumed run
quietly restarts the discriminator from scratch).
"""

from __future__ import annotations

import torch
from torch import nn

from nazarite.rl.wmp.motion import (
    AMP_OBSERVATION_DIM,
    AMPStateNormalizer,
    MotionDataset,
    load_dataset,
)

DEFAULT_HIDDEN_DIMS = (1024, 512)
DEFAULT_STYLE_COEFFICIENT = 0.02
DEFAULT_BATCH_SIZE = 256
DEFAULT_LEARNING_RATE = 1.0e-4
DEFAULT_GRAD_PENALTY = 10.0


class AMPDiscriminator(nn.Module):
    """Score one transition: ``+1`` for reference motion, ``-1`` for policy motion."""

    def __init__(
        self,
        observation_dim: int = AMP_OBSERVATION_DIM,
        hidden_dims: tuple[int, ...] = DEFAULT_HIDDEN_DIMS,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        if observation_dim <= 0:
            raise ValueError(f"observation_dim must be positive, got {observation_dim}")
        layers: list[nn.Module] = []
        width = 2 * int(observation_dim)
        for dim in hidden_dims:
            if dim <= 0:
                raise ValueError(f"hidden dims must be positive, got {hidden_dims}")
            layers.extend([nn.Linear(width, int(dim)), activation()])
            width = int(dim)
        # Linear output: the LSGAN target is +-1, so the head must not saturate.
        layers.append(nn.Linear(width, 1))
        self.hidden_dims = tuple(int(dim) for dim in hidden_dims)
        self.observation_dim = int(observation_dim)
        self.net = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor, next_states: torch.Tensor) -> torch.Tensor:
        pair = torch.cat([states, next_states], dim=-1)
        return self.net(pair).squeeze(-1)


class AdversarialMotionPrior:
    """Everything the runner needs to turn reference clips into a style reward."""

    def __init__(
        self,
        device: str | torch.device = "cpu",
        dataset: MotionDataset | None = None,
        motion_dir: str | None = None,
        hidden_dims: tuple[int, ...] = DEFAULT_HIDDEN_DIMS,
        style_coefficient: float = DEFAULT_STYLE_COEFFICIENT,
        batch_size: int = DEFAULT_BATCH_SIZE,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        grad_penalty: float = DEFAULT_GRAD_PENALTY,
        seed: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.motion_dir = motion_dir
        self.dataset = dataset if dataset is not None else load_dataset(motion_dir)
        self.dataset = MotionDataset(
            [clip.to(self.device) for clip in self.dataset.clips]
        )
        self.style_coefficient = float(style_coefficient)
        self.batch_size = int(batch_size)
        if self.batch_size < 2:
            raise ValueError(f"batch_size must be at least 2, got {self.batch_size}")
        self.grad_penalty = float(grad_penalty)
        self.generator = torch.Generator(device="cpu")
        if seed is not None:
            self.generator.manual_seed(int(seed))

        # Freeze the statistics on the reference states once. The reference
        # implementation keeps updating them from live data, which makes the
        # discriminator chase a moving normalisation while both networks learn.
        self.normalizer = AMPStateNormalizer(AMP_OBSERVATION_DIM, device=self.device)
        self.normalizer.update(self.dataset.all_states())

        self.discriminator = AMPDiscriminator(
            AMP_OBSERVATION_DIM, tuple(hidden_dims)
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=float(learning_rate)
        )
        self.bad_values = 0

    def _normalise(self, states: torch.Tensor) -> torch.Tensor:
        return self.normalizer.normalize(states).detach()

    def style_reward(
        self, states: torch.Tensor, next_states: torch.Tensor
    ) -> torch.Tensor:
        """Per-environment style reward in the discriminator's own scale.

        The caller multiplies by ``step_dt`` to match the reward manager's
        ``scale_rewards_by_dt`` convention, so this number stays comparable with
        a reward weight rather than a per-step return.
        """
        with torch.inference_mode():
            score = self.discriminator(
                self._normalise(states), self._normalise(next_states)
            )
            if not bool(torch.isfinite(score).all()):
                self.bad_values += int((~torch.isfinite(score)).sum())
                score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
            return self.style_coefficient * (
                1.0 - 0.25 * (score - 1.0).square()
            ).clamp_min(0.0)

    def update(
        self, policy_states: torch.Tensor, policy_next_states: torch.Tensor
    ) -> dict[str, float]:
        """One LSGAN step on a balanced batch of reference and policy transitions."""
        if policy_states.shape[0] < 2:
            raise ValueError("need at least two policy transitions to update AMP")
        half = self.batch_size // 2
        expert_states, expert_next = self.dataset.sample(half, self.generator)
        expert_states = expert_states.to(self.device)
        expert_next = expert_next.to(self.device)

        policy_states = policy_states.to(self.device)
        policy_next = policy_next_states.to(self.device)
        index = torch.randint(0, policy_states.shape[0], (half,), device=self.device)
        policy_states = policy_states[index]
        policy_next = policy_next[index]

        expert_pair = torch.cat(
            [self._normalise(expert_states), self._normalise(expert_next)], dim=-1
        )
        policy_pair = torch.cat(
            [self._normalise(policy_states), self._normalise(policy_next)], dim=-1
        )

        expert_score = self.discriminator(
            expert_pair[:, :AMP_OBSERVATION_DIM], expert_pair[:, AMP_OBSERVATION_DIM:]
        )
        policy_score = self.discriminator(
            policy_pair[:, :AMP_OBSERVATION_DIM], policy_pair[:, AMP_OBSERVATION_DIM:]
        )
        # LSGAN: regress reference to +1 and policy to -1.
        loss = (expert_score - 1.0).square().mean() + (
            policy_score + 1.0
        ).square().mean()
        penalty = self._gradient_penalty(expert_pair)
        total = loss + self.grad_penalty * penalty

        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        self.optimizer.step()

        metrics = {
            "loss": float(loss.detach()),
            "grad_pen": float(penalty.detach()),
            "d_expert": float(expert_score.detach().mean()),
            "d_policy": float(policy_score.detach().mean()),
        }
        # Same convention as the world model metrics: never let a NaN reach the
        # logger, and count it so a stuck run is visible instead of silent.
        for key, value in metrics.items():
            if not torch.isfinite(torch.tensor(value)):
                metrics[key] = 0.0
                self.bad_values += 1
        return metrics

    def _gradient_penalty(self, expert_pair: torch.Tensor) -> torch.Tensor:
        """Penalise the discriminator's gradient on reference transitions only."""
        expert_pair = expert_pair.detach().requires_grad_(True)
        score = self.discriminator(
            expert_pair[:, :AMP_OBSERVATION_DIM], expert_pair[:, AMP_OBSERVATION_DIM:]
        )
        gradient = torch.autograd.grad(
            outputs=score.sum(),
            inputs=expert_pair,
            create_graph=True,
            retain_graph=True,
        )[0]
        return gradient.square().sum(dim=-1).mean()

    def state_dict(self) -> dict[str, object]:
        return {
            "discriminator": self.discriminator.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "normalizer": self.normalizer.state_dict(),
            "bad_values": self.bad_values,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.discriminator.load_state_dict(state["discriminator"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.normalizer.load_state_dict(state["normalizer"])
        self.bad_values = int(state.get("bad_values", 0))
