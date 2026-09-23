"""Torch checkpoint runner for the WTW + DELTA residual actor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from nazarite.delta.wtw_delta_model import WtwDeltaResidualModel

from ..config import ACTION_DIM
from .config import (
    DELTA_MAP_DIM,
    DELTA_PROPRIO_DIM,
    MAP_CHANNELS,
    MAP_HEIGHT,
    MAP_WIDTH,
    RESIDUAL_JOINT_SCALES,
    WTW_OBS_DIM,
)


class WtwDeltaResidualPolicy:
    """Load and run the exact composite actor saved by rsl_rl."""

    def __init__(self, policy_path: str | Path):
        path = Path(policy_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Policy not found: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Unsupported Torch checkpoint format: {path}")
        actor_state = checkpoint.get("actor_state_dict")
        if not isinstance(actor_state, dict):
            raise TypeError("Torch checkpoint does not contain actor_state_dict")

        obs = TensorDict(
            {
                "wtw_proprio": torch.zeros(1, WTW_OBS_DIM),
                "delta_proprio": torch.zeros(1, DELTA_PROPRIO_DIM),
                "delta_map": torch.zeros(1, DELTA_MAP_DIM),
            },
            batch_size=[1],
        )
        self.model = WtwDeltaResidualModel(
            obs,
            {"actor": ["wtw_proprio", "delta_proprio", "delta_map"]},
            "actor",
            ACTION_DIM,
            hidden_dims=(512, 256, 128),
            obs_normalization=False,
            prior_group="wtw_proprio",
            delta_map_group="delta_map",
            delta_proprio_group="delta_proprio",
            delta_proprio_dim=DELTA_PROPRIO_DIM,
            map_height=MAP_HEIGHT,
            map_width=MAP_WIDTH,
            map_channels=MAP_CHANNELS,
            map_extent=(2.5, 1.6),
            map_center=(1.25, 0.0),
            forward_x_threshold=0.0,
            residual_hidden_dims=(128, 64),
            residual_scale=0.25,
            residual_activation="softsign",
            residual_joint_scales=RESIDUAL_JOINT_SCALES,
            residual_gate_bias=-0.5,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.35,
                "std_type": "log",
                "std_range": (0.15, 0.38),
            },
            freeze_wtw=True,
        )
        self.model.load_state_dict(actor_state, strict=True)
        self.model.eval()
        self.last_action = np.zeros(ACTION_DIM, dtype=np.float32)
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
            "delta_proprio": self._validate(delta_proprio, DELTA_PROPRIO_DIM),
            "delta_map": self._validate(delta_map, DELTA_MAP_DIM),
        }
        tensor_inputs = {
            name: torch.from_numpy(value[None, :])
            for name, value in inputs.items()
        }
        obs = TensorDict(tensor_inputs, batch_size=[1])
        with torch.inference_mode():
            output = self.model.act_inference(obs).cpu().numpy()
        action = np.asarray(output[0], dtype=np.float32)
        if action.shape != (ACTION_DIM,):
            raise RuntimeError(f"Policy output must be {(ACTION_DIM,)}, got {action.shape}")
        if not np.all(np.isfinite(action)):
            raise FloatingPointError("Policy output contains NaN or Inf")
        self.last_action = action.copy()
        return action

    @staticmethod
    def _validate(value: np.ndarray, size: int) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        if value.shape != (size,):
            raise ValueError(f"Policy input must have shape {(size,)}, got {value.shape}")
        if not np.all(np.isfinite(value)):
            raise FloatingPointError("Policy input contains NaN or Inf")
        return value

