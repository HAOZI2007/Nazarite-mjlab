"""HIM inference using the exported ONNX or the training checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..policy_runner import PolicyRunner
from .config import ACTION_DIM, FRAME_DIM, HISTORY_SIZE, OBS_DIM, OBSERVATION_NAMES


class _HIMActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(OBS_DIM, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, 19),
        )
        self.mlp = nn.Sequential(
            nn.Linear(FRAME_DIM + 19, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, ACTION_DIM),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        frames = history.reshape(-1, HISTORY_SIZE, FRAME_DIM).clamp(-100.0, 100.0)
        frames[:, :, 37:49] = frames[:, :, 37:49].clamp(-10.0, 10.0)
        history = frames.flatten(start_dim=1)
        encoded = self.encoder(history)
        latent = F.normalize(encoded[:, 3:], dim=-1, p=2.0)
        return self.mlp(torch.cat((history[:, :FRAME_DIM], encoded[:, :3], latent), dim=-1)).clamp(-10.0, 10.0)


class HIMPolicy:
    def __init__(self, path: str | Path) -> None:
        path = Path(path).expanduser().resolve()
        self.actor: _HIMActor | None = None
        self.onnx: PolicyRunner | None = None
        self.obs_dim = OBS_DIM
        self.last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        if path.suffix.lower() != ".pt":
            self.onnx = PolicyRunner(
                path, expected_obs_dim=OBS_DIM,
                expected_observation_names=OBSERVATION_NAMES,
            )
            if self.onnx.session is None:
                raise RuntimeError("HIM ONNX session was not initialized")
            metadata = self.onnx.session.get_modelmeta().custom_metadata_map
            if metadata.get("him_export_history_order") != "frame_major_current_first":
                raise ValueError("HIM ONNX must accept current-first frame history")
            return

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint.get("actor_state_dict")
        if not isinstance(state, dict):
            raise TypeError("HIM checkpoint is missing actor_state_dict")
        actor = _HIMActor()
        actor.encoder.load_state_dict({
            key.removeprefix("estimator.encoder."): value
            for key, value in state.items() if key.startswith("estimator.encoder.")
        }, strict=True)
        actor.mlp.load_state_dict({
            key.removeprefix("mlp."): value
            for key, value in state.items() if key.startswith("mlp.")
        }, strict=True)
        actor.eval()
        self.actor = actor

    def reset(self) -> None:
        self.last_action.fill(0.0)
        if self.onnx is not None:
            self.onnx.reset()

    def step(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape != (OBS_DIM,) or not np.all(np.isfinite(obs)):
            raise ValueError(f"HIM observation must be finite and {(OBS_DIM,)}")
        if self.onnx is not None:
            action = self.onnx.step(obs)
        else:
            if self.actor is None:
                raise RuntimeError("HIM actor is not loaded")
            with torch.inference_mode():
                action = self.actor(torch.from_numpy(obs[None, :]))[0].numpy()
        if not np.all(np.isfinite(action)):
            raise FloatingPointError("HIM policy produced non-finite action")
        self.last_action = action.copy()
        return action
