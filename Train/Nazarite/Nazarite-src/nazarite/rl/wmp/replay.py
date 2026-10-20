"""Bounded, image-compressed sequence replay for World Model training."""

from __future__ import annotations

import torch


class WorldModelReplay:
    """Per-environment ring buffers that never sample across episode boundaries."""

    def __init__(
        self,
        num_envs: int,
        capacity_per_env: int,
        image_shape: tuple[int, int, int],
        prop_dim: int,
        action_dim: int,
        device: torch.device | str,
    ) -> None:
        if num_envs < 1 or capacity_per_env < 2:
            raise ValueError(
                "Replay dimensions must be positive and capacity_per_env >= 2"
            )
        self.num_envs = int(num_envs)
        self.capacity = int(capacity_per_env)
        self.device = torch.device(device)
        self.image = torch.empty(
            (num_envs, capacity_per_env, *image_shape), dtype=torch.uint8, device="cpu"
        )
        self.prop = torch.empty(
            (num_envs, capacity_per_env, prop_dim), dtype=torch.float32, device="cpu"
        )
        self.action = torch.empty(
            (num_envs, capacity_per_env, action_dim), dtype=torch.float32, device="cpu"
        )
        self.reward = torch.empty(
            (num_envs, capacity_per_env), dtype=torch.float32, device="cpu"
        )
        self.first = torch.zeros(
            (num_envs, capacity_per_env), dtype=torch.bool, device="cpu"
        )
        self.episode = torch.full(
            (num_envs, capacity_per_env), -1, dtype=torch.int64, device="cpu"
        )
        self.position = torch.zeros(num_envs, dtype=torch.long, device="cpu")
        self.size = torch.zeros(num_envs, dtype=torch.long, device="cpu")
        self.episode_id = torch.full((num_envs,), -1, dtype=torch.long, device="cpu")
        self.total_frames = 0

    def add(
        self,
        image: torch.Tensor,
        prop: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        is_first: torch.Tensor,
    ) -> None:
        image_cpu = image.detach().to(device="cpu", dtype=torch.uint8)
        prop_cpu = prop.detach().to(device="cpu", dtype=torch.float32)
        action_cpu = action.detach().to(device="cpu", dtype=torch.float32)
        reward_cpu = reward.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
        first_cpu = is_first.detach().reshape(-1).to(device="cpu", dtype=torch.bool)
        ids = torch.arange(self.num_envs)
        fresh_episode = first_cpu | (self.episode_id < 0)
        self.episode_id += fresh_episode.long()
        self.image[ids, self.position] = image_cpu
        self.prop[ids, self.position] = prop_cpu
        self.action[ids, self.position] = action_cpu
        self.reward[ids, self.position] = reward_cpu
        self.first[ids, self.position] = first_cpu
        self.episode[ids, self.position] = self.episode_id
        self.position = (self.position + 1) % self.capacity
        self.size = torch.clamp(self.size + 1, max=self.capacity)
        self.total_frames += self.num_envs

    def can_sample(self, batch_length: int) -> bool:
        return any(
            self._valid_starts(env_id, batch_length) for env_id in range(self.num_envs)
        )

    def _ordered_slots(self, env_id: int) -> torch.Tensor:
        size = int(self.size[env_id])
        if size < self.capacity:
            return torch.arange(size)
        return (
            int(self.position[env_id]) + torch.arange(self.capacity)
        ) % self.capacity

    def _valid_starts(self, env_id: int, batch_length: int) -> list[torch.Tensor]:
        slots = self._ordered_slots(env_id)
        if slots.numel() < batch_length:
            return []
        episode = self.episode[env_id, slots]
        starts = []
        for start in range(slots.numel() - batch_length + 1):
            seq_episode = episode[start : start + batch_length]
            if torch.all(seq_episode == seq_episode[0]):
                starts.append(slots[start : start + batch_length])
        return starts

    def sample(self, batch_size: int, batch_length: int) -> dict[str, torch.Tensor]:
        if batch_size < 1 or batch_length < 2:
            raise ValueError("batch_size must be positive and batch_length >= 2")
        candidates = [
            (env_id, starts)
            for env_id in range(self.num_envs)
            if (starts := self._valid_starts(env_id, batch_length))
        ]
        if not candidates:
            raise RuntimeError(
                f"No contiguous replay sequence of length {batch_length} is available"
            )
        env_ids: list[int] = []
        sequences: list[torch.Tensor] = []
        for _ in range(batch_size):
            candidate_index = int(torch.randint(len(candidates), (1,)).item())
            env_id, starts = candidates[candidate_index]
            start_index = int(torch.randint(len(starts), (1,)).item())
            slots = starts[start_index]
            env_ids.append(env_id)
            sequences.append(slots)
        env_index = torch.tensor(env_ids, dtype=torch.long)
        slot_index = torch.stack(sequences)
        batch = {
            "image": self.image[env_index[:, None], slot_index].to(
                self.device, non_blocking=True
            ),
            "prop": self.prop[env_index[:, None], slot_index].to(
                self.device, non_blocking=True
            ),
            "action": self.action[env_index[:, None], slot_index].to(
                self.device, non_blocking=True
            ),
            "reward": self.reward[env_index[:, None], slot_index].to(
                self.device, non_blocking=True
            ),
            "is_first": self.first[env_index[:, None], slot_index].to(
                self.device, non_blocking=True
            ),
        }
        # Dreamer/WMP resets the latent at the start of each sampled subsequence.
        batch["is_first"][:, 0] = True
        return batch
