"""Reward manager for computing reward signals."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from prettytable import PrettyTable

from mjlab.managers.manager_base import ManagerBase, ManagerTermBaseCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


@dataclass(kw_only=True)
class RewardTermCfg(ManagerTermBaseCfg):
  """Configuration for a reward term."""

  func: Any
  """The callable that computes this reward term's value."""

  weight: float
  """Weight multiplier for this reward term."""


class RewardManager(ManagerBase):
  """Manages reward computation by aggregating weighted reward terms.

  Reward Scaling Behavior:
    By default, rewards are scaled by the environment step duration (dt). This
    normalizes cumulative episodic rewards across different simulation frequencies.
    The scaling can be disabled via the ``scale_by_dt`` parameter.

    When ``scale_by_dt=True`` (default):
      - ``reward_buf`` (returned by ``compute()``) = raw_value * weight * dt
      - ``_episode_sums`` (cumulative rewards) are scaled by dt
      - ``Episode_Reward/*`` logged metrics are scaled by dt

    When ``scale_by_dt=False``:
      - ``reward_buf`` = raw_value * weight (no dt scaling)

    Regardless of the scaling setting:
      - ``_step_reward`` (via ``get_active_iterable_terms()``) always contains
        the unscaled reward rate (raw_value * weight)
  """

  _env: ManagerBasedRlEnv

  def __init__(
    self,
    cfg: dict[str, RewardTermCfg],
    env: ManagerBasedRlEnv,
    *,
    scale_by_dt: bool = True,
  ):
    self._term_names: list[str] = list()
    self._term_cfgs: list[RewardTermCfg] = list()
    self._class_term_cfgs: list[RewardTermCfg] = list()
    self._scale_by_dt = scale_by_dt

    self.cfg = deepcopy(cfg)
    super().__init__(env=env)
    self._episode_sums = dict()
    for term_name in self._term_names:
      self._episode_sums[term_name] = torch.zeros(
        self.num_envs, dtype=torch.float, device=self.device
      )
    configured_composition = getattr(env.cfg, "reward_composition", None)
    self._reward_composition = (
      configured_composition
      if isinstance(configured_composition, dict)
      else None
    )
    if self._reward_composition:
      self._episode_sums["__paper_combined"] = torch.zeros(
        self.num_envs, dtype=torch.float, device=self.device
      )
    self._reward_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
    self._step_reward = torch.zeros(
      (self.num_envs, len(self._term_names)), dtype=torch.float, device=self.device
    )

  def __str__(self) -> str:
    msg = f"<RewardManager> contains {len(self._term_names)} active terms.\n"
    table = PrettyTable()
    table.title = "Active Reward Terms"
    table.field_names = ["Index", "Name", "Weight"]
    table.align["Name"] = "l"
    table.align["Weight"] = "r"
    for index, (name, term_cfg) in enumerate(
      zip(self._term_names, self._term_cfgs, strict=False)
    ):
      table.add_row([index, name, term_cfg.weight])
    msg += table.get_string()
    msg += "\n"
    return msg

  # Properties.

  @property
  def active_terms(self) -> list[str]:
    return self._term_names

  # Methods.

  def reset(
    self, env_ids: torch.Tensor | slice | None = None
  ) -> dict[str, torch.Tensor]:
    if env_ids is None:
      env_ids = slice(None)
    extras = {}
    for key in self._episode_sums.keys():
      episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
      log_name = "paper_combined" if key == "__paper_combined" else key
      extras["Episode_Reward/" + log_name] = (
        episodic_sum_avg / self._env.max_episode_length_s
      )
      self._episode_sums[key][env_ids] = 0.0
    for term_cfg in self._class_term_cfgs:
      term_cfg.func.reset(env_ids=env_ids)
    return extras

  def compute(self, dt: float) -> torch.Tensor:
    self._reward_buf[:] = 0.0
    scale = dt if self._scale_by_dt else 1.0
    weighted_rates: dict[str, torch.Tensor] = {}
    for term_idx, (name, term_cfg) in enumerate(
      zip(self._term_names, self._term_cfgs, strict=False)
    ):
      if term_cfg.weight == 0.0:
        self._step_reward[:, term_idx] = 0.0
        continue
      value = term_cfg.func(self._env, **term_cfg.params)
      self._check_term_shape(name, value)
      value = value * term_cfg.weight * scale
      # NaN/Inf can occur from corrupted physics state; zero them to avoid policy crash.
      value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
      self._reward_buf += value
      self._episode_sums[name] += value
      self._step_reward[:, term_idx] = value / scale
      weighted_rates[name] = self._step_reward[:, term_idx]

    composition = self._reward_composition
    if composition:
      positive_names = tuple(composition.get("positive_terms", ()))
      negative_names = tuple(composition.get("negative_terms", ()))
      missing = (
        set(positive_names + negative_names)
        - set(weighted_rates)
      )
      if missing:
        raise KeyError(
          "Reward composition references inactive or unknown terms: "
          f"{sorted(missing)}"
        )
      positive = self._sum_rates(weighted_rates, positive_names)
      negative = self._sum_rates(weighted_rates, negative_names)
      beta = float(composition.get("negative_exponent", 0.1))
      # The paper's negative sum is non-positive.  Clamp the exponent only to
      # avoid numerical underflow if a malformed physics state produces a very
      # large penalty; this does not change the ordinary operating range.
      attenuation = torch.exp(torch.clamp(beta * negative, min=-20.0, max=0.0))
      combined_rate = torch.nan_to_num(
        positive * attenuation, nan=0.0, posinf=0.0, neginf=0.0
      )
      self._reward_buf = combined_rate * scale

      terminal_penalty = float(composition.get("terminal_penalty", 0.0))
      if terminal_penalty != 0.0:
        terminated = getattr(self._env, "reset_terminated", None)
        time_outs = getattr(self._env, "reset_time_outs", None)
        if terminated is not None:
          failure = terminated
          if time_outs is not None:
            failure = failure & ~time_outs
          # Terminal penalties are specified in environment reward units, not
          # reward-rate units, so they remain -50 exactly under dt scaling.
          self._reward_buf += terminal_penalty * failure.to(self._reward_buf.dtype)

      if "__paper_combined" in self._episode_sums:
        self._episode_sums["__paper_combined"] += self._reward_buf
      if hasattr(self._env, "extras") and "log" in self._env.extras:
        self._env.extras["log"]["Reward/paper_positive_rate"] = positive.mean()
        self._env.extras["log"]["Reward/paper_negative_rate"] = negative.mean()
        self._env.extras["log"]["Reward/paper_attenuation"] = attenuation.mean()
    return self._reward_buf

  @staticmethod
  def _sum_rates(
    rates: dict[str, torch.Tensor], names: tuple[str, ...]
  ) -> torch.Tensor:
    if not names:
      sample = next(iter(rates.values()), None)
      if sample is None:
        raise RuntimeError("Cannot compose rewards without active reward terms")
      return torch.zeros_like(sample)
    total = torch.zeros_like(rates[names[0]])
    for name in names:
      total = total + rates[name]
    return total

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    """Delegate debug visualization to class-based reward terms."""
    for _, func in self.get_visualizable_terms():
      func.debug_vis(visualizer)

  def get_visualizable_terms(self) -> list[tuple[str, Any]]:
    """Return ``(name, func)`` pairs for class-based terms with debug_vis."""
    results: list[tuple[str, Any]] = []
    for term_cfg in self._class_term_cfgs:
      if not hasattr(term_cfg.func, "debug_vis"):
        continue
      name = next(
        n
        for n, c in zip(self._term_names, self._term_cfgs, strict=False)
        if c is term_cfg
      )
      results.append((name, term_cfg.func))
    return results

  def get_active_iterable_terms(self, env_idx):
    terms = []
    for idx, name in enumerate(self._term_names):
      terms.append((name, [self._step_reward[env_idx, idx].cpu().item()]))
    return terms

  def get_term_cfg(self, term_name: str) -> RewardTermCfg:
    if term_name not in self._term_names:
      raise ValueError(f"Term '{term_name}' not found in active terms.")
    return self._term_cfgs[self._term_names.index(term_name)]

  def _prepare_terms(self):
    for term_name, term_cfg in self.cfg.items():
      term_cfg: RewardTermCfg | None
      if term_cfg is None:
        print(f"term: {term_name} set to None, skipping...")
        continue
      self._resolve_common_term_cfg(term_name, term_cfg)
      self._term_names.append(term_name)
      self._term_cfgs.append(term_cfg)
      if hasattr(term_cfg.func, "reset") and callable(term_cfg.func.reset):
        self._class_term_cfgs.append(term_cfg)
