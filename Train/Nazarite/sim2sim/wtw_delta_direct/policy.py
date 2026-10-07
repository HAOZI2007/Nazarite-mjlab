"""Torch checkpoint runner for Nazarite-WTW-Delta-Direct-Go2."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from nazarite.delta.direct_action_model import WtwDeltaDirectActionModel

from ..config import ACTION_DIM
from .config import DELTA_PROPRIO_DIM, MAP_CHANNELS, MAP_HEIGHT, MAP_WIDTH, WTW_OBS_DIM
from .legacy_model import LegacyWtwDeltaDirectModel


class WtwDeltaDirectPolicy:
    """Run the exact direct-action actor saved by the Direct task."""

    def __init__(self, policy_path: str | Path):
        path = Path(policy_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Direct policy not found: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or not isinstance(
            checkpoint.get("actor_state_dict"), dict
        ):
            raise TypeError("Direct checkpoint must contain actor_state_dict")
        actor_state = checkpoint["actor_state_dict"]
        delta_proprio_dim = int(actor_state["delta.query.0.weight"].shape[1])
        if delta_proprio_dim not in (61, 305):
            raise ValueError(
                f"Unsupported Direct DELTA proprioception dimension: {delta_proprio_dim}"
            )
        obs = TensorDict(
            {
                "wtw_proprio": torch.zeros(1, WTW_OBS_DIM),
                "delta_proprio": torch.zeros(1, delta_proprio_dim),
                "delta_map": torch.zeros(1, MAP_HEIGHT * MAP_WIDTH * MAP_CHANNELS),
            },
            batch_size=[1],
        )
        # Direct runs before the encoder refactor have a single linear fusion
        # layer (12x137) and 27/216-dimensional geometry tokens. New runs use
        # the current MLP fusion and encoder. Select by checkpoint structure so
        # the sim2sim folder can validate both without changing training code.
        self.legacy = "action_fusion.weight" in actor_state
        if self.legacy:
            self.model = LegacyWtwDeltaDirectModel(
                delta_proprio_dim, MAP_HEIGHT, MAP_WIDTH, MAP_CHANNELS
            )
        else:
            self.model = WtwDeltaDirectActionModel(
                obs,
                {"actor": ["wtw_proprio", "delta_proprio", "delta_map"]},
                "actor",
                ACTION_DIM,
                hidden_dims=(512, 256, 128),
                obs_normalization=False,
                prior_group="wtw_proprio",
                delta_map_group="delta_map",
                delta_proprio_group="delta_proprio",
                delta_proprio_dim=delta_proprio_dim,
                map_height=MAP_HEIGHT,
                map_width=MAP_WIDTH,
                map_channels=MAP_CHANNELS,
                map_extent=(2.0, 1.2),
                map_center=(1.0, 0.0),
                forward_x_threshold=0.0,
                residual_hidden_dims=(256, 128),
                residual_scale=1.0,
                residual_gate_bias=0.0,
                distribution_cfg={
                    "class_name": "GaussianDistribution",
                    "init_std": 0.20,
                    "std_type": "log",
                    "std_range": (0.08, 0.25),
                },
                freeze_wtw=True,
            )
        self.model.load_state_dict(actor_state, strict=True)
        self.model.eval()
        self.delta_proprio_dim = delta_proprio_dim
        self.delta_history_length = delta_proprio_dim // DELTA_PROPRIO_DIM
        self.last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.last_diagnostics: dict[str, float] = {}
        self.policy_path = path

    def reset(self) -> None:
        self.last_action[:] = 0.0
        self.model.reset()

    def step(
        self,
        wtw_proprio: np.ndarray,
        delta_proprio: np.ndarray,
        delta_map: np.ndarray,
    ) -> np.ndarray:
        inputs = {
            "wtw_proprio": self._validate(wtw_proprio, WTW_OBS_DIM),
            "delta_proprio": self._validate(delta_proprio, self.delta_proprio_dim),
            "delta_map": self._validate(delta_map, MAP_HEIGHT * MAP_WIDTH * MAP_CHANNELS),
        }
        obs = TensorDict(
            {name: torch.from_numpy(value[None, :]) for name, value in inputs.items()},
            batch_size=[1],
        )
        with torch.inference_mode():
            output = self.model.act_inference(obs).cpu().numpy()
        action = np.asarray(output[0], dtype=np.float32)
        if action.shape != (ACTION_DIM,) or not np.all(np.isfinite(action)):
            raise FloatingPointError("Direct policy returned invalid actions")
        self.last_action = action.copy()
        self.last_diagnostics = {
            key: float(value.detach().cpu().item())
            for key, value in self.model.last_diagnostics.items()
            if value.numel() == 1
        }
        return action

    @staticmethod
    def _validate(value: np.ndarray, size: int) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        if value.shape != (size,):
            raise ValueError(f"Policy input must be {(size,)}, got {value.shape}")
        if not np.all(np.isfinite(value)):
            raise FloatingPointError("Policy input contains NaN or Inf")
        return value
