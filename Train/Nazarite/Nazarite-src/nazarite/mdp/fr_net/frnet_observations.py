"""FR-Net observations and privileged MCP supervision targets."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import torch

from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.velocity import mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from nazarite.mdp import rewards as custom_rewards

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


PROPRIO_TERM_DIMS = (3, 3, 12, 12, 12)
"""Dimensions of ``base_ang_vel``, gravity, q, dq, and previous action."""

PROPRIO_DIM = sum(PROPRIO_TERM_DIMS)
HISTORY_LENGTH = 5
MASS_TARGET_DIM = 4
CONTACT_TARGET_DIM = 13

_LEG_BODY_GROUPS = (
  ("FL_hip", "FL_thigh", "FL_calf"),
  ("FR_hip", "FR_thigh", "FR_calf"),
  ("RL_hip", "RL_thigh", "RL_calf"),
  ("RR_hip", "RR_thigh", "RR_calf"),
)


def _contact_sensor_label(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  num_primaries: int,
  force_threshold: float,
) -> torch.Tensor:
  """Return a force-thresholded contact label for one body-contact sensor."""
  sensor = env.scene[sensor_name]
  if not isinstance(sensor, ContactSensor):
    raise TypeError(f"Expected ContactSensor for '{sensor_name}', got {type(sensor)}")

  data = sensor.data
  if data.force_history is not None:
    # [N, P * slots, H, 3] -> [N, P * slots]. A brief collision inside a
    # control period is still meaningful for fall recovery, so retain its max.
    magnitude = torch.linalg.vector_norm(data.force_history, dim=-1).amax(dim=-1)
  elif data.force is not None:
    # [N, P * slots, 3] -> [N, P * slots].
    magnitude = torch.linalg.vector_norm(data.force, dim=-1)
  elif data.found is not None:
    magnitude = (data.found > 0).to(dtype=torch.float32)
    force_threshold = 0.5
  else:
    raise RuntimeError(f"Contact sensor '{sensor_name}' has no usable contact field")

  if sensor.cfg.num_slots > 1:
    magnitude = magnitude.reshape(
      env.num_envs,
      num_primaries,
      sensor.cfg.num_slots,
    ).amax(dim=-1)

  expected_shape = (env.num_envs, num_primaries)
  if tuple(magnitude.shape) != expected_shape:
    raise RuntimeError(
      f"'{sensor_name}' returned {tuple(magnitude.shape)}, expected {expected_shape}. "
      "Check the body pattern and ContactSensorCfg.num_slots."
    )
  return (magnitude >= force_threshold).to(dtype=torch.float32)


def frnet_contact_target(
  env: ManagerBasedRlEnv,
  force_threshold: float = 5.0,
) -> torch.Tensor:
  """Return the privileged [N, 13] body-ground contact target.

  The fixed channel order is ``base, FL/FR/RL/RR hip, thigh, calf``. It is
  deliberately defined once here so the target, logging, checkpoints, and MCP
  head always agree on leg ordering.
  """
  base = _contact_sensor_label(env, "trunk_ground_touch", 1, force_threshold)
  hips = _contact_sensor_label(env, "hip_ground_touch", 4, force_threshold)
  thighs = _contact_sensor_label(env, "thigh_ground_touch", 4, force_threshold)
  calves = _contact_sensor_label(env, "shank_ground_touch", 4, force_threshold)
  return torch.cat((base, hips, thighs, calves), dim=-1)


def _sim_body_ids(
  env: ManagerBasedRlEnv,
  body_names: tuple[str, ...],
) -> torch.Tensor:
  """Resolve ordered robot body names into global MuJoCo body IDs."""
  robot = env.scene["robot"]
  local_ids, resolved_names = robot.find_bodies(body_names, preserve_order=True)
  if tuple(resolved_names) != body_names:
    raise RuntimeError(
      f"FR-Net leg body mapping mismatch: expected {body_names}, "
      f"got {tuple(resolved_names)}"
    )
  local_ids_tensor = torch.as_tensor(local_ids, dtype=torch.long, device=env.device)
  return robot.indexing.body_ids[local_ids_tensor]


def frnet_mass_target(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Return the current [N, 4] leg-mass ratios in FL, FR, RL, RR order.

  ``dr.pseudo_inertia`` expands ``body_mass`` per environment and changes it at
  reset. Reading the live model, rather than caching a sampled ratio, ensures
  the supervision target always matches the dynamics actually simulated.
  """
  body_mass = env.sim.model.body_mass
  if body_mass.ndim == 1:
    body_mass = body_mass.unsqueeze(0).expand(env.num_envs, -1)
  default_mass = env.sim.get_default_field("body_mass")

  ratios = []
  for body_names in _LEG_BODY_GROUPS:
    body_ids = _sim_body_ids(env, body_names)
    current_mass = body_mass[:, body_ids].sum(dim=-1)
    nominal_mass = default_mass[body_ids].sum().clamp_min(1.0e-6)
    ratios.append((current_mass / nominal_mass).unsqueeze(-1))
  return torch.cat(ratios, dim=-1)


def frnet_aux_targets(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Return [N, 17] = four mass ratios followed by 13 contact labels."""
  return torch.cat((frnet_mass_target(env), frnet_contact_target(env)), dim=-1)


def make_frnet_observation_terms() -> tuple[
  dict[str, ObservationTermCfg],
  dict[str, ObservationTermCfg],
  dict[str, ObservationTermCfg],
]:
  """Build actor history, asymmetric critic, and training-only MCP labels.

  The actor group is 210-dimensional. Its flattened layout is term-major, not
  frame-major; :class:`FRNetActor` reconstructs it into ``[N, 5, 42]``.
  """
  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
      history_length=HISTORY_LENGTH,
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
      history_length=HISTORY_LENGTH,
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"biased": True},
      noise=Unoise(n_min=-0.01, n_max=0.01),
      history_length=HISTORY_LENGTH,
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
      history_length=HISTORY_LENGTH,
    ),
    "actions": ObservationTermCfg(
      func=mdp.last_action,
      history_length=HISTORY_LENGTH,
    ),
  }

  critic_terms = deepcopy(actor_terms)
  for term in critic_terms.values():
    # The critic only needs a current privileged state, not a long history.
    term.history_length = 1
  critic_terms.update(
    {
      "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel, history_length=1),
      "base_lin_vel": ObservationTermCfg(
        func=mdp.builtin_sensor,
        params={"sensor_name": "robot/imu_lin_vel"},
        history_length=1,
      ),
      "base_height": ObservationTermCfg(
        func=custom_rewards.safe_base_height,
        history_length=1,
      ),
      "foot_height": ObservationTermCfg(
        func=custom_rewards.safe_foot_height,
        params={"sensor_name": "foot_height_scan"},
        history_length=1,
      ),
      "foot_contact": ObservationTermCfg(
        func=custom_rewards.safe_foot_contact,
        params={"sensor_name": "feet_ground_contact"},
        history_length=1,
      ),
      "foot_contact_forces": ObservationTermCfg(
        func=custom_rewards.safe_foot_contact_forces,
        params={"sensor_name": "feet_ground_contact"},
        history_length=1,
      ),
      # Privileged values help the critic estimate return but never reach actor.
      "frnet_mass_target": ObservationTermCfg(
        func=frnet_mass_target,
        history_length=1,
      ),
      "frnet_contact_target": ObservationTermCfg(
        func=frnet_contact_target,
        history_length=1,
      ),
    }
  )

  auxiliary_terms = {
    "targets": ObservationTermCfg(func=frnet_aux_targets, history_length=0),
  }
  return actor_terms, critic_terms, auxiliary_terms
