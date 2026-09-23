"""WTW locomotion prior with a DELTA terrain residual policy.

The actor keeps a frozen WTW MLP as the default action policy and adds a
zero-initialized DELTA residual.  This makes a newly created task start at the
WTW behaviour while still allowing PPO to learn terrain-dependent corrections.
"""

from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from rsl_rl.modules import MLP
from rsl_rl.utils import resolve_class
from tensordict import TensorDict
from torch import nn
from mjlab.utils.lab_api.math import quat_apply

from .encoder import DeltaEncoder


class WtwDeltaResidualModel(nn.Module):
  """Actor model: frozen/optional WTW prior plus DELTA action residual."""

  is_recurrent = False

  def __init__(
      self,
      obs: TensorDict,
      obs_groups: dict[str, list[str]],
      obs_set: str,
      output_dim: int,
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
      distribution_cfg=None,
      delta_map_group="delta_map",
      prior_group=None,
      delta_proprio_group=None,
      delta_proprio_dim=45,
      map_height=16,
      map_width=26,
      map_channels=3,
      map_extent=(2.0, 2.0),
      map_center=(1.5, 0.0),
      forward_x_threshold=0.0,
      map_confidence_target=0.45,
      map_confidence_floor=0.50,
      residual_hidden_dims=(128, 64),
      residual_scale=0.1,
      residual_scale_start=None,
      residual_scale_ramp_iters=0,
      residual_joint_scales=None,
      residual_gate_bias=-2.0,
      residual_activation="softsign",
      diagnostics_enabled=True,
      wtw_checkpoint=None,
      freeze_wtw=True,
      **kwargs,
  ):
    super().__init__()
    del kwargs
    if obs_normalization:
      raise NotImplementedError("Use term-aware normalization for WTW+DELTA observations")
    self.obs_groups = list(obs_groups[obs_set])
    if delta_map_group not in self.obs_groups:
      raise ValueError(f"Observation group '{delta_map_group}' is required")
    self.delta_map_group = delta_map_group
    self.prior_group = prior_group or next(
      (group for group in self.obs_groups if group != delta_map_group), None,
    )
    self.delta_proprio_group = delta_proprio_group
    if self.prior_group is None or self.prior_group not in self.obs_groups:
      raise ValueError("prior_group must be included in the selected observation groups")
    if self.delta_proprio_group is None or self.delta_proprio_group not in self.obs_groups:
      raise ValueError("delta_proprio_group must be included in the selected observation groups")
    self.base_groups = [self.prior_group]
    self.map_height, self.map_width = int(map_height), int(map_width)
    self.map_channels = int(map_channels)
    self.map_extent = tuple(float(value) for value in map_extent)
    self.map_center = tuple(float(value) for value in map_center)
    self.map_confidence_target = float(map_confidence_target)
    self.map_confidence_floor = float(map_confidence_floor)
    if self.map_confidence_target <= 0.0:
      raise ValueError("map_confidence_target must be positive")
    if not 0.0 <= self.map_confidence_floor <= 1.0:
      raise ValueError("map_confidence_floor must be in [0, 1]")
    self.map_dim = self.map_height * self.map_width * self.map_channels
    if obs[delta_map_group].shape[-1] != self.map_dim:
      raise ValueError(
        f"Expected {delta_map_group} dimension {self.map_dim}, "
        f"got {obs[delta_map_group].shape[-1]}"
      )
    self.base_obs_dim = int(obs[self.prior_group].shape[-1])
    self.group_sizes = {group: int(obs[group].shape[-1]) for group in self.obs_groups}
    self.obs_dim = sum(self.group_sizes.values())
    self.output_dim = int(output_dim)
    self.delta_proprio_dim = int(obs[self.delta_proprio_group].shape[-1])
    self.residual_scale_end = float(residual_scale)
    self.residual_scale_start = float(
      self.residual_scale_end if residual_scale_start is None else residual_scale_start
    )
    self.residual_scale_ramp_iters = max(0, int(residual_scale_ramp_iters))
    self.training_iteration = 0
    self.residual_scale = self.residual_scale_start
    self.last_diagnostics: dict[str, torch.Tensor] = {}
    self._previous_residual_action: torch.Tensor | None = None
    self.diagnostics_enabled = bool(diagnostics_enabled)
    self.residual_activation = str(residual_activation)
    if self.residual_activation not in ("softsign", "tanh"):
      raise ValueError("residual_activation must be 'softsign' or 'tanh'")
    self.attention_show_all_heads = False
    self.attention_scale = 1.0
    self._attention_viz_state: list[dict[str, torch.Tensor]] = []
    self.obs_normalizer = nn.Identity()
    if residual_joint_scales is None:
      residual_joint_scales = (1.0,) * int(output_dim)
    if len(residual_joint_scales) != int(output_dim):
      raise ValueError("residual_joint_scales must have one value per action")
    if any(float(value) <= 0.0 for value in residual_joint_scales):
      raise ValueError("residual_joint_scales must be positive")
    self.register_buffer(
      "residual_joint_scales",
      torch.as_tensor(residual_joint_scales, dtype=torch.float32),
    )

    # This MLP has exactly the architecture of the existing WTW actor.  A
    # checkpoint can therefore be loaded without changing any old task.
    self.wtw_policy = MLP(self.base_obs_dim, output_dim, hidden_dims, activation)
    if freeze_wtw:
      self.wtw_policy.requires_grad_(False)

    self.delta = DeltaEncoder(
      proprio_dim=self.delta_proprio_dim, dim=64, layers=3, heads=4, samples=8,
      map_channels=self.map_channels, map_extent=self.map_extent,
      map_center=self.map_center,
      forward_x_threshold=float(forward_x_threshold),
    )
    residual_input_dim = 64 + self.delta_proprio_dim
    self.residual_head = MLP(
      residual_input_dim, output_dim, residual_hidden_dims, activation,
    )
    self.residual_gate = MLP(
      residual_input_dim, output_dim, residual_hidden_dims, activation,
    )
    # At initialization the residual is exactly zero, so the composite policy
    # behaves like the WTW prior even before DELTA has learned anything.
    last = self.residual_head[-1]
    if isinstance(last, nn.Linear):
      nn.init.zeros_(last.weight)
      nn.init.zeros_(last.bias)
    gate_last = self.residual_gate[-1]
    if isinstance(gate_last, nn.Linear):
      nn.init.zeros_(gate_last.weight)
      nn.init.constant_(gate_last.bias, float(residual_gate_bias))

    if distribution_cfg is None:
      self.distribution = None
    else:
      cfg = copy.deepcopy(distribution_cfg)
      distribution_class, distribution_args = resolve_class(cfg)
      self.distribution = distribution_class(output_dim, **distribution_args)

    if wtw_checkpoint:
      self._load_wtw_checkpoint(Path(wtw_checkpoint))

  def _load_wtw_checkpoint(self, path: Path) -> None:
    if not path.is_file():
      raise FileNotFoundError(f"WTW checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("actor_state_dict", checkpoint.get("model_state_dict"))
    if state is None:
      raise KeyError(f"Checkpoint {path} has no actor_state_dict/model_state_dict")
    wtw_state = {}
    for key, value in state.items():
      if key.startswith("mlp."):
        wtw_state[key.removeprefix("mlp.")] = value
      elif key.startswith("actor.mlp."):
        wtw_state[key.removeprefix("actor.mlp.")] = value
    missing, unexpected = self.wtw_policy.load_state_dict(wtw_state, strict=False)
    if missing:
      raise RuntimeError(f"WTW checkpoint is incompatible; missing actor keys: {missing}")
    if unexpected:
      raise RuntimeError(f"WTW checkpoint has unexpected actor keys: {unexpected}")

    # Match the prior's stochastic policy as well as its deterministic mean.
    # Starting a pretrained WTW policy at std=1.0 injects roughly three times
    # the exploration noise used by the source checkpoint and can immediately
    # destroy its gait.
    distribution_state = {}
    for key, value in state.items():
      if key.startswith("distribution."):
        distribution_state[key.removeprefix("distribution.")] = value
      elif key.startswith("actor.distribution."):
        distribution_state[key.removeprefix("actor.distribution.")] = value
    if self.distribution is not None and distribution_state:
      missing, unexpected = self.distribution.load_state_dict(
        distribution_state, strict=False,
      )
      if missing or unexpected:
        raise RuntimeError(
          "WTW checkpoint distribution is incompatible; "
          f"missing={missing}, unexpected={unexpected}"
        )

  def _observation(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    base = obs[self.prior_group]
    proprio = obs[self.delta_proprio_group]
    terrain = obs[self.delta_map_group].reshape(
      base.shape[0], self.map_height, self.map_width, self.map_channels,
    )
    return base, proprio, terrain

  def _map_confidence(self, terrain: torch.Tensor) -> torch.Tensor:
    """Convert BEV valid coverage into a bounded residual confidence.

    A sparse camera projection should not be allowed to apply a full action
    correction. The floor keeps early training gradients alive while the
    target makes the correction reach full strength only with a usable map.
    """
    if self.map_channels < 4:
      return terrain.new_ones(terrain.shape[0])
    valid_ratio = terrain[..., 3].clamp(0.0, 1.0).mean(dim=(1, 2))
    return (valid_ratio / self.map_confidence_target).clamp(
      min=self.map_confidence_floor, max=1.0,
    )

  def set_training_iteration(self, iteration: int) -> None:
    """Update the task-local residual curriculum before a rollout."""
    self.training_iteration = max(0, int(iteration))
    if self.residual_scale_ramp_iters <= 0:
      self.residual_scale = self.residual_scale_end
      return
    progress = min(1.0, self.training_iteration / self.residual_scale_ramp_iters)
    self.residual_scale = self.residual_scale_start + progress * (
      self.residual_scale_end - self.residual_scale_start
    )

  def get_latent(self, obs, masks=None, hidden_state=None):
    del masks, hidden_state
    base, proprio, terrain = self._observation(obs)
    terrain_latent = self.delta(proprio, terrain)
    return base, proprio, terrain_latent

  def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
    base, proprio, terrain = self._observation(obs)
    terrain_latent = self.delta(proprio, terrain)
    prior = self.wtw_policy(base)
    residual_input = torch.cat((proprio, terrain_latent), dim=-1)
    residual_raw = self.residual_head(residual_input)
    # Softsign keeps a non-zero gradient for large corrections. The previous
    # tanh head saturated on almost every joint in the first run, so PPO could
    # no longer learn the direction of the terrain correction.
    if self.residual_activation == "softsign":
      residual = residual_raw / (1.0 + residual_raw.abs())
    else:
      residual = torch.tanh(residual_raw)
    gate = torch.sigmoid(self.residual_gate(residual_input))
    residual = gate * residual * self.residual_joint_scales.to(residual.dtype)
    map_confidence = self._map_confidence(terrain)
    residual_action = self.residual_scale * residual * map_confidence.unsqueeze(-1)
    mean = prior + residual_action
    if self.diagnostics_enabled:
      with torch.no_grad():
        prior_rms = prior.square().mean().sqrt()
        residual_rms = residual_action.square().mean().sqrt()
        final_rms = mean.square().mean().sqrt()
        previous = self._previous_residual_action
        if previous is None or previous.shape != residual_action.shape:
          residual_rate_rms = residual_action.new_zeros(())
        else:
          residual_rate_rms = (residual_action - previous).square().mean().sqrt()
        self._previous_residual_action = residual_action.detach()
        self.last_diagnostics = {
          "DELTA/prior_action_rms": prior_rms.detach(),
          "DELTA/residual_rms": residual_rms.detach(),
          "DELTA/residual_max": residual_action.abs().amax().detach(),
          "DELTA/residual_saturation_ratio": (residual.abs() > 0.95).to(residual.dtype).mean().detach(),
          "DELTA/residual_gate_mean": gate.mean().detach(),
          "DELTA/residual_gate_open_ratio": (gate > 0.5).to(gate.dtype).mean().detach(),
          "DELTA/gate_open_penalty": gate.square().mean().detach(),
          "DELTA/residual_rate_rms": residual_rate_rms.detach(),
          "DELTA/final_action_rms": final_rms.detach(),
          "DELTA/residual_to_prior_ratio": (residual_rms / prior_rms.clamp_min(1.0e-6)).detach(),
          "DELTA/residual_scale": mean.new_tensor(self.residual_scale),
          "DELTA/map_confidence": map_confidence.mean().detach(),
          "DELTA/encoder_output_rms": terrain_latent.square().mean().sqrt().detach(),
        }
        for index, value in enumerate((self.residual_scale * residual).square().mean(dim=0).sqrt()):
          self.last_diagnostics[f"DELTA/residual_rms_joint_{index}"] = value.detach()
        self.last_diagnostics.update(self.delta.last_stats)
    if self.distribution is None:
      return mean
    if stochastic_output:
      self.distribution.update(mean)
      return self.distribution.sample()
    return self.distribution.deterministic_output(mean)

  def reset(self, dones=None, hidden_state=None):
    del dones, hidden_state

  def get_hidden_state(self):
    return None

  def detach_hidden_state(self, dones=None):
    del dones

  @property
  def output_mean(self):
    if self.distribution is None:
      raise RuntimeError("This model has no output distribution")
    return self.distribution.mean

  @property
  def output_std(self):
    if self.distribution is None:
      raise RuntimeError("This model has no output distribution")
    return self.distribution.std

  @property
  def output_entropy(self):
    if self.distribution is None:
      raise RuntimeError("This model has no output distribution")
    return self.distribution.entropy

  @property
  def output_distribution_params(self):
    if self.distribution is None:
      raise RuntimeError("This model has no output distribution")
    return self.distribution.params

  def get_output_log_prob(self, outputs):
    if self.distribution is None:
      raise RuntimeError("This model has no output distribution")
    return self.distribution.log_prob(outputs)

  def get_kl_divergence(self, old_params, new_params):
    if self.distribution is None:
      raise RuntimeError("This model has no output distribution")
    return self.distribution.kl_divergence(old_params, new_params)

  def update_normalization(self, obs):
    del obs

  def act_inference(self, obs):
    return self(obs, stochastic_output=False)

  def get_diagnostics(self) -> dict[str, torch.Tensor]:
    """Return detached scalar diagnostics for the training logger."""
    return {key: value.detach() for key, value in self.last_diagnostics.items()}

  def enable_attention_cache(self, enabled: bool = True) -> None:
    """Enable detached attention snapshots for the play visualizer."""
    self.delta.enable_attention_cache(enabled)
    if not enabled:
      self._attention_viz_state = []

  def debug_visualize(self, env, visualizer) -> None:
    """Draw cached WTW+DELTA sampling points on the observed BEV surface."""
    if not getattr(env, "delta_attention_enabled", False):
      return
    aux = self.delta.get_attention()
    elevation_map = self.delta.last_map
    if not aux or elevation_map is None or self.map_channels < 4:
      return
    try:
      env_idx = int(visualizer.env_idx)
      robot = env.scene["robot"]
      base_pos = robot.data.root_link_pos_w[env_idx]
      base_quat = robot.data.root_link_quat_w[env_idx]
    except (AttributeError, IndexError, KeyError, RuntimeError):
      return

    # These limits are the explicit BEV contract in delta_depth_image().
    x0 = self.map_center[0] - 0.5 * self.map_extent[0]
    x1 = self.map_center[0] + 0.5 * self.map_extent[0]
    y0 = self.map_center[1] - 0.5 * self.map_extent[1]
    y1 = self.map_center[1] + 0.5 * self.map_extent[1]
    map_z = elevation_map[env_idx, ..., 2][None, None]
    map_valid = elevation_map[env_idx, ..., 3][None, None]
    colors = ((0.15, 0.45, 1.0), (1.0, 0.75, 0.10), (1.0, 0.20, 0.10))
    next_state: list[dict[str, torch.Tensor]] = []
    for layer_idx, item in enumerate(aux):
      locations = item["locations"][env_idx]
      attention = item["attention"][env_idx]
      if not self.attention_show_all_heads:
        locations = locations.mean(dim=0, keepdim=True)
        attention = attention.mean(dim=0, keepdim=True)
      previous = self._attention_viz_state[layer_idx] if layer_idx < len(self._attention_viz_state) else None
      if previous is not None and previous["locations"].shape == locations.shape:
        locations = 0.90 * previous["locations"] + 0.10 * locations
        attention = 0.90 * previous["attention"] + 0.10 * attention
      next_state.append({"locations": locations, "attention": attention})

      grid = torch.stack((locations[..., 0], -locations[..., 1]), dim=-1).reshape(1, -1, 1, 2)
      terrain_z = F.grid_sample(
        map_z, grid, mode="bilinear", padding_mode="border", align_corners=True,
      ).reshape(locations.shape[:2])
      valid = F.grid_sample(
        map_valid, grid, mode="bilinear", padding_mode="zeros", align_corners=True,
      ).reshape(locations.shape[:2]) > 0.25
      colour = colors[min(layer_idx, len(colors) - 1)]
      for head_idx in range(locations.shape[0]):
        weights = attention[head_idx] / attention[head_idx].max().clamp_min(1.0e-6)
        for sample_idx in range(locations.shape[1]):
          if not bool(valid[head_idx, sample_idx]):
            continue
          location = locations[head_idx, sample_idx]
          local = torch.stack((
            x0 + 0.5 * (location[0] + 1.0) * (x1 - x0),
            y0 + 0.5 * (location[1] + 1.0) * (y1 - y0),
            terrain_z[head_idx, sample_idx] * 0.8 + 0.01,
          ))
          center = base_pos + quat_apply(base_quat, local)
          weight = float(weights[sample_idx].item())
          radius = 0.008 + 0.022 * weight * float(self.attention_scale)
          visualizer.add_sphere(
            center, radius,
            (*colour, 0.25 + 0.75 * min(weight, 1.0)),
            label=f"wtw_delta_l{layer_idx}_h{head_idx}_p{sample_idx}",
          )
    self._attention_viz_state = next_state

  def as_jit(self):
    return _WtwDeltaExport(self)

  def as_onnx(self, verbose=False):
    del verbose
    return _WtwDeltaExport(self)


class _WtwDeltaExport(nn.Module):
  def __init__(self, model: WtwDeltaResidualModel):
    super().__init__()
    self.model = copy.deepcopy(model)
    self.obs_groups = list(model.obs_groups)
    self.obs_dim = model.obs_dim

  def forward(self, obs):
    td = TensorDict(
      {group: obs[..., offset:offset + size] for group, offset, size in self._slices()},
      batch_size=[obs.shape[0]],
    )
    return self.model.act_inference(td)

  def _slices(self):
    offset = 0
    result = []
    for group in self.obs_groups:
      size = self.model.group_sizes[group]
      result.append((group, offset, size))
      offset += size
    return result

  def get_dummy_inputs(self):
    return (torch.zeros(1, self.obs_dim),)

  @property
  def input_names(self):
    return ["obs"]

  @property
  def output_names(self):
    return ["actions"]
