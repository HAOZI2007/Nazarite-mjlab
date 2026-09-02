"""Mass-contact prediction actor used by the FR-Net recovery task."""

from __future__ import annotations

import copy

import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP
from tensordict import TensorDict
from torch import nn

from nazarite.mdp.fr_net.frnet_observations import (
  CONTACT_TARGET_DIM,
  HISTORY_LENGTH,
  MASS_TARGET_DIM,
  PROPRIO_DIM,
  PROPRIO_TERM_DIMS,
)


class FRNetActor(MLPModel):
  """Gaussian policy augmented with a mass-contact prediction module.

  The manager supplies one term-major 210-dimensional actor group. This class
  reconstructs its five chronological 42-dimensional frames, predicts the
  training-only mass/contact targets, and gives the policy MLP only the current
  proprioception plus MCP predictions. The privileged targets never enter this
  module, so the exported actor is deployable from proprioception alone.
  """

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict | None = None,
    history_length: int = HISTORY_LENGTH,
    proprio_dim: int = PROPRIO_DIM,
    mass_dim: int = MASS_TARGET_DIM,
    contact_dim: int = CONTACT_TARGET_DIM,
    latent_dim: int = 16,
    mcp_hidden_dims: tuple[int, ...] | list[int] = (256, 128),
  ) -> None:
    super().__init__(
      obs=obs,
      obs_groups=obs_groups,
      obs_set=obs_set,
      output_dim=output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      obs_normalization=obs_normalization,
      distribution_cfg=distribution_cfg,
    )
    if obs_normalization:
      raise ValueError(
        "FRNetActor currently expects obs_normalization=False. Normalize the "
        "five-frame proprioceptive input explicitly before enabling it."
      )
    if len(self.obs_groups) != 1:
      raise ValueError("FRNetActor requires exactly one actor observation group")
    if proprio_dim != sum(PROPRIO_TERM_DIMS):
      raise ValueError(
        f"Expected proprio_dim={sum(PROPRIO_TERM_DIMS)}, got {proprio_dim}"
      )
    if self.obs_dim != history_length * proprio_dim:
      raise ValueError(
        f"FRNetActor expected {history_length * proprio_dim} actor dimensions, "
        f"got {self.obs_dim}. Check ObservationTermCfg.history_length."
      )
    if not mcp_hidden_dims:
      raise ValueError("mcp_hidden_dims must contain at least one hidden layer")

    self.history_length = history_length
    self.proprio_dim = proprio_dim
    self.mass_dim = mass_dim
    self.contact_dim = contact_dim
    self.latent_dim = latent_dim

    mcp_feature_dim = mcp_hidden_dims[-1]
    self.mcp_encoder = MLP(
      history_length * proprio_dim,
      mcp_feature_dim,
      mcp_hidden_dims,
      activation,
    )
    self.mass_head = nn.Linear(mcp_feature_dim, mass_dim)
    self.contact_head = nn.Linear(mcp_feature_dim, contact_dim)
    self.latent_head = nn.Linear(mcp_feature_dim, latent_dim)

    # Replace MLPModel's raw-observation policy with the enhanced policy head.
    policy_input_dim = proprio_dim + mass_dim + contact_dim + latent_dim
    if self.distribution is not None:
      policy_output_dim = self.distribution.input_dim
    else:
      policy_output_dim = output_dim
    self.mlp = MLP(policy_input_dim, policy_output_dim, hidden_dims, activation)
    self._last_mcp_outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = (
      None
    )

  def _unpack_history(self, flat_history: torch.Tensor) -> torch.Tensor:
    """Convert term-major ``[N, 210]`` history into chronological ``[N, 5, 42]``."""
    chunks = []
    offset = 0
    for term_dim in PROPRIO_TERM_DIMS:
      width = self.history_length * term_dim
      chunk = flat_history[:, offset : offset + width]
      chunks.append(chunk.reshape(flat_history.shape[0], self.history_length, term_dim))
      offset += width
    if offset != flat_history.shape[-1]:
      raise RuntimeError(
        f"Unexpected FR-Net actor history width {flat_history.shape[-1]}; "
        f"consumed {offset}."
      )
    return torch.cat(chunks, dim=-1)

  def _mcp_forward(
    self,
    flat_history: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    history = self._unpack_history(flat_history)
    current_proprio = history[:, -1, :]
    feature = self.mcp_encoder(history.flatten(start_dim=1))
    mass_prediction = self.mass_head(feature)
    contact_logits = self.contact_head(feature)
    latent = self.latent_head(feature)
    return current_proprio, mass_prediction, contact_logits, latent

  def get_latent(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
    """Compute the 75-dimensional enhanced policy input and cache MCP outputs."""
    del masks, hidden_state
    flat_history = obs[self.obs_groups[0]]
    current, mass_prediction, contact_logits, latent = self._mcp_forward(flat_history)
    self._last_mcp_outputs = (mass_prediction, contact_logits, latent)
    contact_probability = torch.sigmoid(contact_logits)
    return torch.cat((current, mass_prediction, contact_probability, latent), dim=-1)

  def get_mcp_outputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return predictions from the most recent forward pass for auxiliary loss."""
    if self._last_mcp_outputs is None:
      raise RuntimeError("FR-Net MCP outputs requested before actor forward pass")
    return self._last_mcp_outputs

  def as_jit(self) -> nn.Module:
    """Export the deployable history-to-action actor, including MCP."""
    return _TorchFRNetActor(self)

  def as_onnx(self, verbose: bool) -> nn.Module:
    """Export the deployable history-to-action actor, including MCP."""
    return _OnnxFRNetActor(self, verbose)


class _TorchFRNetActor(nn.Module):
  """TorchScript-safe deterministic version of :class:`FRNetActor`."""

  def __init__(self, model: FRNetActor) -> None:
    super().__init__()
    self.history_length = model.history_length
    self.mcp_encoder = copy.deepcopy(model.mcp_encoder)
    self.mass_head = copy.deepcopy(model.mass_head)
    self.contact_head = copy.deepcopy(model.contact_head)
    self.latent_head = copy.deepcopy(model.latent_head)
    self.policy_mlp = copy.deepcopy(model.mlp)
    self.term_dims = [3, 3, 12, 12, 12]
    self.deterministic_output: nn.Module = (
      copy.deepcopy(model.distribution.as_deterministic_output_module())
      if model.distribution is not None
      else nn.Identity()
    )

  def _unpack_history(self, flat_history: torch.Tensor) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    offset = 0
    for term_dim in self.term_dims:
      width = self.history_length * term_dim
      chunks.append(
        flat_history[:, offset : offset + width].reshape(
          flat_history.shape[0], self.history_length, term_dim
        )
      )
      offset += width
    return torch.cat(chunks, dim=-1)

  def forward(self, flat_history: torch.Tensor) -> torch.Tensor:
    history = self._unpack_history(flat_history)
    feature = self.mcp_encoder(history.flatten(start_dim=1))
    mass_prediction = self.mass_head(feature)
    contact_probability = torch.sigmoid(self.contact_head(feature))
    latent = self.latent_head(feature)
    enhanced = torch.cat(
      (history[:, -1, :], mass_prediction, contact_probability, latent),
      dim=-1,
    )
    return self.deterministic_output(self.policy_mlp(enhanced))

  @torch.jit.export
  def reset(self) -> None:
    pass


class _OnnxFRNetActor(_TorchFRNetActor):
  """ONNX adapter with the metadata expected by the existing mjlab exporter."""

  def __init__(self, model: FRNetActor, verbose: bool) -> None:
    super().__init__(model)
    self.verbose = verbose
    self.input_size = model.history_length * model.proprio_dim

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.input_size),)

  @property
  def input_names(self) -> list[str]:
    return ["proprio_history"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]
