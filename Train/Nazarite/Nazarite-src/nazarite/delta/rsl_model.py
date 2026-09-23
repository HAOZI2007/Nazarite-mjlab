"""RSL-RL model wrapper that preserves DELTA map structure."""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from rsl_rl.modules import MLP
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_class
from tensordict import TensorDict
from torch import nn

from mjlab.utils.lab_api.math import quat_apply

from .encoder import DeltaEncoder


class DeltaModel(nn.Module):
  """Actor/critic model for a flattened observation group.

  The observation manager concatenates terms into ``obs[obs_group]``.  We keep
  the map as the final term and split it here, restoring ``[B,H,W,C]`` before
  calling DeltaEncoder.  This avoids changing the existing PPO storage API.
  """

  is_recurrent = False

  def __init__(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str,
               output_dim: int, hidden_dims=(256, 128), activation="elu",
               obs_normalization=False, distribution_cfg=None,
               proprio_dim=45, map_height=16, map_width=26, map_channels=3,
               map_extent=(2.0, 1.2), map_center=(0.0, 0.0), **kwargs):
    super().__init__()
    del kwargs
    self.obs_groups = list(obs_groups[obs_set])
    self.obs_dim = sum(obs[group].shape[-1] for group in self.obs_groups)
    self.proprio_dim = int(proprio_dim)
    self.map_height, self.map_width = int(map_height), int(map_width)
    self.map_channels = int(map_channels)
    map_dim = self.map_height * self.map_width * self.map_channels
    if self.obs_dim != self.proprio_dim + map_dim:
      raise ValueError(
        f"DELTA expects obs_dim={self.proprio_dim + map_dim}, got {self.obs_dim}; "
        "ensure delta_map is the final observation term"
      )
    self.obs_normalizer = nn.Identity()
    if obs_normalization:
      # Normalizing the concatenated map would destroy the metric semantics;
      # retain the option for API compatibility but leave it disabled by default.
      raise NotImplementedError("Use map-aware normalization for DELTA instead of global normalization")
    if distribution_cfg is None:
      self.distribution = None
      head_out = output_dim
    else:
      cfg = copy.deepcopy(distribution_cfg)
      distribution_class, distribution_args = resolve_class(cfg)
      self.distribution: Distribution = distribution_class(output_dim, **distribution_args)
      head_out = self.distribution.input_dim
    self.delta = DeltaEncoder(
      proprio_dim=self.proprio_dim, dim=64, layers=3, heads=4, samples=8,
      map_channels=self.map_channels, map_extent=tuple(map_extent),
      map_center=tuple(map_center),
    )
    self.attention_show_all_heads = False
    self.attention_scale = 1.0
    self._attention_viz_state: list[dict[str, torch.Tensor]] = []
    self.mlp = MLP(self.proprio_dim + 64, head_out, hidden_dims, activation)
    if self.distribution is not None:
      self.distribution.init_mlp_weights(self.mlp)

  def _observation(self, obs: TensorDict) -> torch.Tensor:
    return torch.cat([obs[group] for group in self.obs_groups], dim=-1)

  def get_latent(self, obs, masks=None, hidden_state=None):
    del masks, hidden_state
    value = self._observation(obs)
    proprio = value[..., : self.proprio_dim]
    terrain = value[..., self.proprio_dim:].reshape(
      value.shape[0], self.map_height, self.map_width, self.map_channels
    )
    encoded = self.delta(
      proprio,
      terrain,
      return_aux=self.delta.attention_cache_enabled,
    )
    if self.delta.attention_cache_enabled:
      encoded = encoded[0]
    return torch.cat((proprio, encoded), dim=-1)

  def enable_attention_cache(self, enabled: bool = True) -> None:
    """Enable snapshots used by the interactive Viser attention overlay."""
    self.delta.enable_attention_cache(enabled)
    if not enabled:
      self._attention_viz_state = []

  def debug_visualize(self, env, visualizer) -> None:
    """Draw DELTA's learned sampling points into an mjlab DebugVisualizer.

    Locations are in the normalized DELTA map frame.  The depth preprocessing
    uses 5.0 m (x), 1.5 m (y), and 0.8 m (z) normalization scales.  The cached map is sampled at
    each location to recover the corresponding terrain height, so points are
    drawn on the observed surface instead of at a fixed height above the base.
    A short EMA keeps the overlay readable while the policy is moving.
    """
    if not getattr(env, "delta_attention_enabled", False):
      return
    aux = self.delta.get_attention()
    elevation_map = self.delta.last_map
    if not aux or elevation_map is None:
      return
    try:
      env_idx = int(visualizer.env_idx)
      robot = env.scene["robot"]
      base_pos = robot.data.root_link_pos_w[env_idx]
      base_quat = robot.data.root_link_quat_w[env_idx]
    except (AttributeError, IndexError, KeyError, RuntimeError):
      return

    # Layer colours progress from blue (coarse) to red (final refinement).
    colors = ((0.15, 0.45, 1.0), (1.0, 0.75, 0.10), (1.0, 0.20, 0.10))
    # Keep this inverse mapping synchronized with delta_depth_image().  The
    # depth map covers x=[0, 5] m in front of the camera, while the encoder's
    # sampling coordinate is the symmetric grid [-1, 1].  Consequently x=0 in
    # DELTA coordinates means the image centre (2.5 m forward), not the robot
    # base.  Lateral y remains centred around zero.
    x_limit, y_limit = self.delta.limits
    map_z = elevation_map[env_idx, ..., 2]
    map_z_image = map_z[None, None]
    next_state: list[dict[str, torch.Tensor]] = []
    for layer_idx, item in enumerate(aux):
      locations = item["locations"][env_idx]
      attention = item["attention"][env_idx]
      if not self.attention_show_all_heads:
        locations = locations.mean(dim=0, keepdim=True)
        attention = attention.mean(dim=0, keepdim=True)
      previous = (
        self._attention_viz_state[layer_idx]
        if layer_idx < len(self._attention_viz_state)
        else None
      )
      if (
        previous is not None
        and previous["locations"].shape == locations.shape
        and previous["attention"].shape == attention.shape
      ):
        # Do not jump to a completely new set of spheres on every 50 Hz
        # policy update.  This also prevents a one-frame sensor glitch from
        # making the overlay appear to blink.
        locations = 0.90 * previous["locations"] + 0.10 * locations
        attention = 0.90 * previous["attention"] + 0.10 * attention
      next_state.append({"locations": locations, "attention": attention})

      grid = torch.stack(
        (locations[..., 0] / self.delta.limits[0],
         -locations[..., 1] / self.delta.limits[1]),
        dim=-1,
      ).reshape(1, -1, 1, 2)
      terrain_z = F.grid_sample(
        map_z_image,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
      ).reshape(locations.shape[:2])
      valid = terrain_z.abs() > 1.0e-5
      colour = colors[min(layer_idx, len(colors) - 1)]
      for head_idx in range(locations.shape[0]):
        weights = attention[head_idx]
        weights = weights / weights.max().clamp_min(1.0e-6)
        for sample_idx in range(locations.shape[1]):
          if not bool(valid[head_idx, sample_idx]):
            continue
          location = locations[head_idx, sample_idx]
          xy = torch.stack(
            (
              (location[0] / x_limit + 1.0) * 2.5,
              location[1] / y_limit * 1.5,
            )
          )
          local_z = terrain_z[head_idx, sample_idx] * 0.8
          local = torch.cat((xy, local_z.reshape(1)))
          center = base_pos + quat_apply(base_quat, local)
          center = center + center.new_tensor((0.0, 0.0, 0.01))
          weight = float(weights[sample_idx].item())
          radius = 0.008 + 0.022 * weight * float(self.attention_scale)
          visualizer.add_sphere(
            center,
            radius,
            (*colour, 0.25 + 0.75 * min(weight, 1.0)),
            label=f"delta_l{layer_idx}_h{head_idx}_p{sample_idx}",
          )
    self._attention_viz_state = next_state

  def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
    del masks, hidden_state
    raw = self.mlp(self.get_latent(obs))
    if self.distribution is None:
      return raw
    if stochastic_output:
      self.distribution.update(raw)
      return self.distribution.sample()
    return self.distribution.deterministic_output(raw)

  def reset(self, dones=None, hidden_state=None):
    del dones, hidden_state

  def get_hidden_state(self):
    return None

  def detach_hidden_state(self, dones=None):
    del dones

  @property
  def output_mean(self):
    return self.distribution.mean

  @property
  def output_std(self):
    return self.distribution.std

  @property
  def output_entropy(self):
    return self.distribution.entropy

  @property
  def output_distribution_params(self):
    return self.distribution.params

  def get_output_log_prob(self, outputs):
    return self.distribution.log_prob(outputs)

  def get_kl_divergence(self, old_params, new_params):
    return self.distribution.kl_divergence(old_params, new_params)

  def update_normalization(self, obs):
    del obs

  def act_inference(self, obs):
    return self(obs, stochastic_output=False)

  def as_jit(self):
    return _DeltaExport(self, self.obs_dim)

  def as_onnx(self, verbose=False):
    del verbose
    return _DeltaExport(self, self.obs_dim)


class _DeltaExport(nn.Module):
  """Export wrapper accepting the same flattened observation as the runner."""

  def __init__(self, model: DeltaModel, obs_dim: int):
    super().__init__()
    # Exporters move the wrapper to CPU before tracing. Keep a detached copy so
    # this temporary device change does not move the live training policy.
    self.model = copy.deepcopy(model)
    self.obs_dim = obs_dim

  def forward(self, obs):
    # Export uses deterministic action output and a single flat input tensor.
    td = TensorDict({group: obs for group in self.model.obs_groups}, batch_size=[obs.shape[0]])
    return self.model.act_inference(td)

  def get_dummy_inputs(self):
    return (torch.zeros(1, self.obs_dim),)

  @property
  def input_names(self):
    return ["obs"]

  @property
  def output_names(self):
    return ["actions"]
