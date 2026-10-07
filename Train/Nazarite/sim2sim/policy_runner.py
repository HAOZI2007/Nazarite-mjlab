from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn

try:
    import onnxruntime as ort
except ModuleNotFoundError:  # Optional until an ONNX policy is selected.
    ort: Any = None

try:
    from .config import (
        ACTION_DIM,
        JOINT_NAMES,
    )
except ImportError:  # pragma: no cover - direct script fallback
    from config import (
        ACTION_DIM,
        JOINT_NAMES,
    )


class _TorchActor(nn.Module):
    """The MLP actor layout saved by mjlab's MLPModel checkpoint."""

    def __init__(self, obs_dim: int, hidden_dims: list[int], action_dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.ELU()))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)


class PolicyRunner:
    """Run either an exported ONNX actor or an mjlab ``.pt`` actor checkpoint."""

    def __init__(
        self,
        policy_path: str | Path,
        *,
        expected_obs_dim: int | None = None,
        expected_observation_names: list[str] | None = None,
    ):
        policy_path = Path(policy_path).expanduser().resolve()
        if not policy_path.is_file():
            raise FileNotFoundError(f"Policy not found: {policy_path}")

        self.session: Any | None = None
        self.actor: _TorchActor | None = None
        self.input_name = ""
        self.output_name = ""
        if policy_path.suffix.lower() == ".pt":
            self.obs_dim = self._load_torch_checkpoint(
                policy_path,
                expected_obs_dim,
            )
        else:
            onnxruntime_module: Any = ort
            if onnxruntime_module is None:
                raise RuntimeError(
                    "ONNX policy requested but onnxruntime is not installed. "
                    "Run `uv sync` in Train/Nazarite or `uv pip install onnxruntime`."
                )
            self.session = onnxruntime_module.InferenceSession(
                str(policy_path),
                providers=["CPUExecutionProvider"],
            )
            session: Any = self.session

            # Older onnxruntime typing stubs incorrectly expose these lists as
            # SparseTensor. The cast is limited to metadata returned by the API;
            # actual input/output arrays are converted with np.asarray below.
            input_nodes = cast(Any, session.get_inputs())
            output_nodes = cast(Any, session.get_outputs())
            if len(input_nodes) != 1 or len(output_nodes) != 1:
                raise RuntimeError("Expected an ONNX actor with one input and one output")

            self.input_name = str(cast(Any, input_nodes[0]).name)
            self.output_name = str(cast(Any, output_nodes[0]).name)
            self.obs_dim = self._validate_contract(
                input_nodes[0],
                output_nodes[0],
                expected_obs_dim,
                expected_observation_names,
            )

        self.last_action = np.zeros(ACTION_DIM, dtype=np.float32)

    def _load_torch_checkpoint(
        self,
        policy_path: Path,
        expected_obs_dim: int | None,
    ) -> int:
        checkpoint = torch.load(policy_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Unsupported Torch checkpoint format: {policy_path}")
        actor_state = checkpoint.get("actor_state_dict")
        if not isinstance(actor_state, dict):
            raise TypeError("Torch checkpoint does not contain actor_state_dict")

        weight_keys = sorted(
            (key for key in actor_state if key.startswith("mlp.") and key.endswith(".weight")),
            key=lambda key: int(key.split(".")[1]),
        )
        if not weight_keys:
            raise ValueError("Torch actor checkpoint contains no MLP weights")
        dimensions = [int(actor_state[key].shape[1]) for key in weight_keys]
        dimensions.append(int(actor_state[weight_keys[-1]].shape[0]))
        obs_dim = dimensions[0]
        action_dim = dimensions[-1]
        if expected_obs_dim is not None and obs_dim != expected_obs_dim:
            raise ValueError(
                f"Expected {expected_obs_dim}D policy input, got {obs_dim}D checkpoint"
            )
        if action_dim != ACTION_DIM:
            raise ValueError(
                f"Expected {ACTION_DIM}D policy output, got {action_dim}D checkpoint"
            )

        hidden_dims = [int(actor_state[key].shape[0]) for key in weight_keys[:-1]]
        actor = _TorchActor(obs_dim, hidden_dims, action_dim)
        mlp_state = {
            key: value for key, value in actor_state.items() if key.startswith("mlp.")
        }
        actor.load_state_dict(mlp_state, strict=True)
        actor.eval()
        self.actor = actor
        return obs_dim

    def _validate_contract(
        self,
        input_node: Any,
        output_node: Any,
        expected_obs_dim: int | None,
        expected_observation_names: list[str] | None,
    ) -> int:
        """Reject policies whose dimensions or exported ordering do not match."""
        input_shape = cast(list[Any], input_node.shape)
        output_shape = cast(list[Any], output_node.shape)
        if not input_shape or not isinstance(input_shape[-1], int):
            raise ValueError(f"Expected a fixed-size ONNX input, got {input_shape}")
        obs_dim = int(input_shape[-1])
        if expected_obs_dim is not None and obs_dim != expected_obs_dim:
            raise ValueError(
                f"Expected {expected_obs_dim}D policy input, got {input_shape}"
            )
        if not output_shape or output_shape[-1] != ACTION_DIM:
            raise ValueError(f"Expected {ACTION_DIM}D ONNX output, got {output_shape}")

        if self.session is None:
            raise RuntimeError("ONNX session is not initialized")
        metadata = cast(Any, self.session.get_modelmeta()).custom_metadata_map
        exported_obs = metadata.get("observation_names")
        if (
            expected_observation_names is not None
            and exported_obs
            and exported_obs.split(",") != expected_observation_names
        ):
            raise ValueError(
                "Policy observation order does not match sim2sim: "
                f"{exported_obs}"
            )
        exported_joints = metadata.get("joint_names")
        if exported_joints and exported_joints.split(",") != JOINT_NAMES:
            raise ValueError(
                "Policy joint order does not match sim2sim: "
                f"{exported_joints}"
            )
        return obs_dim

    def reset(self) -> None:
        self.last_action[:] = 0.0

    def step(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape != (self.obs_dim,):
            raise ValueError(f"Policy input must be {(self.obs_dim,)}, got {obs.shape}")

        if self.actor is not None:
            with torch.inference_mode():
                output = self.actor(torch.from_numpy(obs[None, :])).numpy()
        else:
            if self.session is None:
                raise RuntimeError("Policy backend is not initialized")
            # Convert the untyped runtime result to a NumPy array before indexing.
            outputs = cast(Any, self.session.run(
                [self.output_name],
                {self.input_name: obs[None, :]},
            ))
            output = np.asarray(outputs[0], dtype=np.float32)
        if output.ndim != 2 or output.shape[0] != 1:
            raise RuntimeError(f"Expected batched policy output, got {output.shape}")

        action = np.asarray(output[0], dtype=np.float32)
        if action.shape != (ACTION_DIM,):
            raise RuntimeError(f"Policy output must be {(ACTION_DIM,)}, got {action.shape}")
        if not np.all(np.isfinite(action)):
            raise FloatingPointError("Policy output contains NaN or Inf")

        self.last_action = action.copy()
        return action
