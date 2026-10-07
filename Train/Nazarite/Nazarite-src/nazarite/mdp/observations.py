"""Observations used by the HIM locomotion policy."""

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def phase(env, period: float, command_name: str) -> torch.Tensor:
  """Return the current gait phase as sine/cosine, disabled while standing."""
  global_phase = (env.episode_length_buf * env.step_dt) % period / period
  result = torch.stack(
    (torch.sin(global_phase * torch.pi * 2.0),
     torch.cos(global_phase * torch.pi * 2.0)),
    dim=-1,
  )
  command = env.command_manager.get_command(command_name)
  if command is not None:
    standing = torch.linalg.norm(command, dim=1) < 0.1
    result = torch.where(standing.unsqueeze(1), torch.zeros_like(result), result)
  return result


def him_behavior_parameters(env, command_name: str = "behavior") -> torch.Tensor:
  """Return the two explicit HIM conditions: height and stance width."""
  command = env.command_manager.get_command(command_name)
  if command is None or command.ndim != 2 or command.shape[1] < 2:
    return torch.zeros((env.num_envs, 2), device=env.device)
  return command[:, :2]


def base_com(env, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  body_ids = asset.indexing.body_ids[asset_cfg.body_ids]
  return env.sim.model.body_ipos[:, body_ids].flatten(start_dim=1)


def foot_contact(env, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return (sensor.data.found > 0).float()
