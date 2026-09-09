"""Window dataset and robust quantile normalization for SMP pretraining."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MotionWindowDataset(Dataset[torch.Tensor]):
    def __init__(self, data_dir: str | Path, norm_stats_file: str | Path):
        files = sorted(Path(data_dir).glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"no motion-window NPZ files in {data_dir}")
        chunks: list[np.ndarray] = []
        expected: tuple[int, int] | None = None
        for path in files:
            with np.load(path, allow_pickle=False) as archive:
                if "windows" not in archive.files:
                    raise ValueError(f"{path} has no 'windows' array")
                windows = np.asarray(archive["windows"], dtype=np.float32)
            if windows.ndim != 3 or not np.isfinite(windows).all():
                raise ValueError(f"invalid windows in {path}: shape={windows.shape}")
            shape = (int(windows.shape[1]), int(windows.shape[2]))
            if expected is None:
                expected = shape
            elif shape != expected:
                raise ValueError(
                    f"window shape mismatch in {path}: {shape} != {expected}"
                )
            chunks.append(windows)
        assert expected is not None
        self.window_size, self.feature_dim = expected
        with np.load(norm_stats_file, allow_pickle=False) as stats:
            self.q_low = np.asarray(stats["q_low"], dtype=np.float32)
            self.q_high = np.asarray(stats["q_high"], dtype=np.float32)
        if self.q_low.shape != (self.feature_dim,) or self.q_high.shape != (
            self.feature_dim,
        ):
            raise ValueError("normalization statistics do not match feature dimension")
        span = self.q_high - self.q_low
        if np.any(span <= 0.0):
            raise ValueError("normalization statistics contain a non-positive span")
        data = np.concatenate(chunks, axis=0)
        self.windows = torch.from_numpy(2.0 * (data - self.q_low) / span - 1.0)

    def __len__(self) -> int:
        return int(self.windows.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.windows[index]

    def denormalize(self, value: torch.Tensor) -> torch.Tensor:
        low = torch.as_tensor(self.q_low, device=value.device, dtype=value.dtype)
        high = torch.as_tensor(self.q_high, device=value.device, dtype=value.dtype)
        return (value + 1.0) * 0.5 * (high - low) + low
