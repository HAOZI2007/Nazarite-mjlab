"""Observation clipping and pre-clip diagnostics for HIM training."""

from copy import deepcopy

import torch


ACTION_CLIP = 10.0
OBSERVATION_CLIP = 100.0
RAW_OBSERVATION_ABORT = 1000.0


def checked_observation(env, *, source_func, numerics_tag, raw_limit, **params):
  """Record non-finite/out-of-range values before manager clipping."""
  value = source_func(env, **params)
  bad = (~torch.isfinite(value) | (value.abs() > raw_limit)).flatten(1).any(1)
  if not hasattr(env, "_him_numerics_bad"):
    env._him_numerics_bad = torch.zeros(env.num_envs, dtype=torch.bool, device=value.device)
    env._him_numerics_evidence = {}
  env._him_numerics_bad.logical_or_(bad)
  previous = env._him_numerics_evidence.get(numerics_tag)
  if previous is None:
    previous = torch.zeros_like(value)
  mask = bad.reshape((-1,) + (1,) * (value.ndim - 1))
  env._him_numerics_evidence[numerics_tag] = torch.where(mask, value, previous)
  return value


def configure_him_observations(cfg) -> None:
  """Apply the HIM numerical contract to actor and critic terms."""
  for group_name in ("actor", "critic"):
    group = cfg.observations[group_name]
    group.terms = {name: deepcopy(term) for name, term in group.terms.items()}
    for name, term in group.terms.items():
      bound = ACTION_CLIP if name == "actions" else OBSERVATION_CLIP
      term.clip = (-bound, bound)
      term.params = {
        **term.params,
        "source_func": term.func,
        "numerics_tag": f"{group_name}/{name}",
        "raw_limit": RAW_OBSERVATION_ABORT,
      }
      term.func = checked_observation
