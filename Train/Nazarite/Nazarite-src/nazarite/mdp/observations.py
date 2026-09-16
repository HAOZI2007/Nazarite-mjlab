"""Observations used by the HIM locomotion policy."""

import torch


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

