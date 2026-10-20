"""AMP reference motions for the Go2 WMP-HIM tasks.

The clips are the reference repository's `go2_motion` set: a flat-ground trot
family recorded from a Go2 policy at several speeds plus turn-in-place and a
standing pose. Each file is a JSON object whose `Frames` entry holds 61 scalars
per frame:

======  ============  ==================================================
slice   width         content
======  ============  ==================================================
0:3     3             root position
3:7     4             root quaternion
7:19    12            joint positions
19:31   12            toe positions in the root frame
31:34   3             linear velocity
34:37   3             angular velocity
37:49   12            joint velocities
49:61   12            toe velocities
======  ============  ==================================================

The discriminator only ever sees a *state*, and a state is the same 30 numbers
the environment feeds it: joint positions, base linear velocity, base angular
velocity and joint velocities, in that order. The toe columns are dropped rather
than reordered, which is why ``AMP_STATE_SLICES`` keeps them out instead of
slicing a single contiguous range.

Clip joint order is the Go2 MJCF order (FL, FR, RL, RR) and must not be
permuted. The A1 and Go1 environments of the reference report their joints as
FR, FL, RR, RL, so only those robots need the left/right swap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

AMP_FRAME_DIM = 61
"""Width of a raw clip frame."""

AMP_OBSERVATION_DIM = 30
"""Width of one AMP state: joints 12, base linear 3, base angular 3, joints 12."""

AMP_STATE_SLICES: tuple[slice, ...] = (
    slice(7, 19),
    slice(31, 34),
    slice(34, 37),
    slice(37, 49),
)

DEFAULT_MOTION_DIR = (
    Path(__file__).resolve().parent.parent.parent / "amp_data" / "go2_motion"
)


@dataclass(frozen=True)
class MotionClip:
    """One reference clip already reduced to AMP states."""

    name: str
    states: torch.Tensor
    frame_duration: float = 0.02
    loop_mode: str = "Wrap"
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.states.ndim != 2 or self.states.shape[1] != AMP_OBSERVATION_DIM:
            raise ValueError(
                f"{self.name}: expected [frames, {AMP_OBSERVATION_DIM}] states, "
                f"got {tuple(self.states.shape)}"
            )
        if self.states.shape[0] < 2:
            raise ValueError(f"{self.name}: a clip needs at least two frames")


def state_from_frames(frames: torch.Tensor) -> torch.Tensor:
    """Reduce raw 61-wide clip frames to 30-wide AMP states."""
    if frames.ndim != 2 or frames.shape[1] != AMP_FRAME_DIM:
        raise ValueError(
            f"expected [frames, {AMP_FRAME_DIM}] clip frames, got {tuple(frames.shape)}"
        )
    return torch.cat([frames[:, part] for part in AMP_STATE_SLICES], dim=1)


def load_clip(path: Path) -> MotionClip:
    """Load one JSON clip and reduce it to AMP states."""
    payload = json.loads(Path(path).read_text())
    frames = torch.tensor(payload["Frames"], dtype=torch.float32)
    duration = float(payload.get("FrameDuration", 0.02))
    if duration <= 0.0:
        raise ValueError(f"{path}: FrameDuration must be positive, got {duration}")
    return MotionClip(
        name=Path(path).stem,
        states=state_from_frames(frames),
        frame_duration=duration,
        loop_mode=str(payload.get("LoopMode", "Wrap")),
        weight=float(payload.get("MotionWeight", 1.0)),
    )


def load_dataset(directory: Path | None = None) -> MotionDataset:
    """Load every clip in ``directory`` (the bundled Go2 trot set by default)."""
    root = Path(directory) if directory is not None else DEFAULT_MOTION_DIR
    clips = [load_clip(path) for path in sorted(root.glob("*.txt"))]
    if not clips:
        raise FileNotFoundError(f"no .txt motion clips under {root}")
    return MotionDataset(clips)


class MotionDataset:
    """A weighted mixture of clips that yields ``(state, next_state)`` pairs.

    Pairs, not single frames: the discriminator scores a transition, so the
    successor of the last frame of a wrapping clip is its first frame.
    """

    def __init__(
        self, clips: list[MotionClip], device: str | torch.device = "cpu"
    ) -> None:
        if not clips:
            raise ValueError("a motion dataset needs at least one clip")
        self.clips = clips
        self.device = torch.device(device)
        self._states = [clip.states.to(self.device) for clip in clips]
        weights = torch.tensor([clip.weight for clip in clips], dtype=torch.float32)
        if float(weights.sum()) <= 0.0:
            raise ValueError("motion clip weights must sum to a positive value")
        self._cumulative = torch.cumsum(weights.to(self.device) / weights.sum(), dim=0)

    @property
    def num_frames(self) -> int:
        return int(sum(states.shape[0] for states in self._states))

    def sample(
        self, batch_size: int, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw ``(state, next_state)`` pairs, honouring per-clip weights."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        clip_ids = torch.searchsorted(
            self._cumulative,
            torch.rand(batch_size, generator=generator, device=self.device),
        ).clamp_(max=len(self._states) - 1)

        current = torch.empty(batch_size, AMP_OBSERVATION_DIM, device=self.device)
        following = torch.empty(batch_size, AMP_OBSERVATION_DIM, device=self.device)
        for index in range(len(self._states)):
            selected = clip_ids == index
            if not bool(selected.any()):
                continue
            count = int(selected.sum())
            states = self._states[index]
            frames = states.shape[0]
            offsets = torch.randint(
                0, frames, (count,), generator=generator, device=self.device
            )
            current[selected] = states[offsets]
            following[selected] = states[(offsets + 1) % frames]
        return current, following


class AMPStateNormalizer:
    """Running mean/std over AMP states.

    ``update`` must be fed the *unnormalised* states; normalising never changes the
    statistics. The reference implementation normalised first and then accumulated,
    which lets the running mean drift towards zero as the network trains.
    """

    def __init__(
        self,
        dim: int = AMP_OBSERVATION_DIM,
        device: str | torch.device = "cpu",
        epsilon: float = 1.0e-8,
    ) -> None:
        self.epsilon = float(epsilon)
        self.count = 0.0
        self.mean = torch.zeros(dim, device=device)
        self.var = torch.ones(dim, device=device)

    def update(self, states: torch.Tensor) -> None:
        """Fold a batch of raw states into the running statistics."""
        if states.ndim != 2 or states.shape[1] != self.mean.shape[0]:
            width = self.mean.shape[0]
            raise ValueError(
                f"expected [batch, {width}] states, got {tuple(states.shape)}"
            )
        states = states.detach().to(self.mean.device)
        batch_count = float(states.shape[0])
        batch_mean = states.mean(dim=0)
        batch_var = states.var(dim=0, unbiased=False)
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * (batch_count / total)
        spread = (
            self.var * self.count
            + batch_var * batch_count
            + delta.square() * (self.count * batch_count / total)
        )
        self.var = spread / total
        self.count = total

    def normalize(self, states: torch.Tensor) -> torch.Tensor:
        """Normalise with the current statistics, without touching them."""
        return (states - self.mean) / (self.var.sqrt() + self.epsilon)

    def state_dict(self) -> dict[str, torch.Tensor | float]:
        return {"mean": self.mean.clone(), "var": self.var.clone(), "count": self.count}

    def load_state_dict(self, state: dict[str, torch.Tensor | float]) -> None:
        # as_tensor accepts either a tensor or a plain float, so a checkpoint
        # round-tripped through JSON by another tool still loads.
        device = self.mean.device
        self.mean = torch.as_tensor(
            state["mean"], dtype=torch.float32, device=device
        ).clone()
        var = torch.as_tensor(state["var"], dtype=torch.float32, device=device).clone()
        self.var = var
        self.count = float(state["count"])
