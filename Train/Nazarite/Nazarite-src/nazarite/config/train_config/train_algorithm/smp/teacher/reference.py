"""Reference-motion command for the closed-loop Go2 teacher.

The command owns the temporal state of one reference sequence per parallel
environment.  ``motion_file`` may point either to a single preprocessed NPZ
or to a dataset manifest containing multiple preprocessed NPZ files.  The
manifest path is intentionally accepted here instead of in the environment
configuration so the old single-clip workflow remains compatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


SMP_LEG_ORDER: tuple[str, ...] = ("FL", "FR", "RL", "RR")
SMP_JOINT_ORDER: tuple[str, ...] = tuple(
  f"{leg}_{kind}_joint" for leg in SMP_LEG_ORDER for kind in ("hip", "thigh", "calf")
)

_ClipArrayKey = Literal[
  "qpos",
  "root_pos_mujoco",
  "root_quat_wxyz",
  "foot_target_world_m",
  "foot_target_base_m",
]


class _ReferenceClip(TypedDict):
  fps: float
  qpos: np.ndarray
  root_pos_mujoco: np.ndarray
  root_quat_wxyz: np.ndarray
  foot_target_world_m: np.ndarray
  foot_target_base_m: np.ndarray


def _sample_sequence(values: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
  """Linearly sample ``[frames, ...]`` at one floating-point phase per env."""
  clamped = phase.clamp(0.0, values.shape[0] - 1.0)
  lower = torch.floor(clamped).to(dtype=torch.long)
  upper = torch.clamp(lower + 1, max=values.shape[0] - 1)
  alpha = (clamped - lower.to(dtype=clamped.dtype)).reshape(
    (-1,) + (1,) * (values.ndim - 1)
  )
  return values[lower] * (1.0 - alpha) + values[upper] * alpha


def _sample_motion_sequence(
  values: torch.Tensor,
  motion_ids: torch.Tensor,
  time_steps: torch.Tensor,
  frame_counts: torch.Tensor,
) -> torch.Tensor:
  """Sample padded multi-clip data with a per-environment clip and phase.

  ``values`` has shape ``[num_clips, max_frames, ...]``.  Padding is never
  observed because the phase is clamped against the selected clip's own
  frame count before interpolation.
  """
  selected = values[motion_ids]
  selected_counts = frame_counts[motion_ids]
  max_phase = (selected_counts - 1).to(dtype=time_steps.dtype)
  clamped = torch.minimum(torch.clamp(time_steps, min=0.0), max_phase)
  lower = torch.floor(clamped).to(dtype=torch.long)
  upper = torch.minimum(lower + 1, selected_counts - 1)
  alpha = (clamped - lower.to(dtype=clamped.dtype)).reshape(
    (-1,) + (1,) * (values.ndim - 2)
  )
  env_ids = torch.arange(motion_ids.shape[0], device=values.device)
  lower_values = selected[env_ids, lower]
  upper_values = selected[env_ids, upper]
  return lower_values * (1.0 - alpha) + upper_values * alpha


def _finite_difference(values: np.ndarray, fps: float) -> np.ndarray:
  if values.shape[0] < 2:
    return np.zeros_like(values, dtype=np.float32)
  return np.gradient(values, 1.0 / fps, axis=0).astype(np.float32)


def _load_array(archive: np.lib.npyio.NpzFile, key: str, shape_tail: tuple[int, ...]) -> np.ndarray:
  if key not in archive.files:
    raise ValueError(f"reference NPZ is missing '{key}'")
  values = np.asarray(archive[key], dtype=np.float32)
  if values.ndim != len(shape_tail) + 1 or values.shape[1:] != shape_tail:
    raise ValueError(f"reference '{key}' must have shape [frames, {shape_tail}], got {values.shape}")
  if not np.isfinite(values).all():
    raise ValueError(f"reference '{key}' contains non-finite values")
  return values


@dataclass(kw_only=True)
class ReferenceMotionCommandCfg(CommandTermCfg):
  """Configuration for a single-clip or multi-clip reference command."""

  motion_file: str
  entity_name: str = "robot"
  joint_names: tuple[str, ...] = SMP_JOINT_ORDER
  foot_site_names: tuple[str, ...] = SMP_LEG_ORDER
  contact_height_threshold: float = 0.025
  randomize_clip: bool = True
  randomize_start_phase: bool = True
  start_phase_range: tuple[float, float] = (0.0, 1.0)

  def build(self, env: ManagerBasedRlEnv) -> ReferenceMotionCommand:
    return ReferenceMotionCommand(self, env)


class ReferenceMotionCommand(CommandTerm):
  """Time-indexed reference state used by teacher observations and rewards."""

  cfg: ReferenceMotionCommandCfg

  def __init__(self, cfg: ReferenceMotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    motion_path = Path(cfg.motion_file).expanduser().resolve()
    if not motion_path.is_file():
      raise FileNotFoundError(f"reference motion file not found: {motion_path}")

    clip_paths = self._resolve_clip_paths(motion_path)
    clip_arrays = [
      self._load_clip(path, cfg, env.sim.mj_model.nq) for path in clip_paths
    ]
    if not clip_arrays:
      raise ValueError(f"reference manifest contains no clips: {motion_path}")

    fps_values = [clip["fps"] for clip in clip_arrays]
    fps = fps_values[0]
    if any(abs(value - fps) > 1.0e-5 for value in fps_values[1:]):
      raise ValueError(f"all reference clips must have the same fps, got {fps_values}")
    max_frames = max(clip["qpos"].shape[0] for clip in clip_arrays)

    def stack_padded(key: _ClipArrayKey) -> np.ndarray:
      arrays: list[np.ndarray] = [clip[key] for clip in clip_arrays]
      result = np.empty((len(arrays), max_frames) + arrays[0].shape[1:], dtype=np.float32)
      for index, array in enumerate(arrays):
        result[index, : array.shape[0]] = array
        result[index, array.shape[0] :] = array[-1]
      return result

    qpos_all = stack_padded("qpos")
    source_root_pos_all = stack_padded("root_pos_mujoco")
    root_quat_all = stack_padded("root_quat_wxyz")
    foot_world_all = stack_padded("foot_target_world_m")
    foot_base_arrays = [clip["foot_target_base_m"] for clip in clip_arrays]
    foot_base_all = np.empty((len(clip_arrays), max_frames, 4, 3), dtype=np.float32)
    for index, array in enumerate(foot_base_arrays):
      foot_base_all[index, : array.shape[0]] = array
      foot_base_all[index, array.shape[0] :] = array[-1]
    frame_counts = np.asarray([clip["qpos"].shape[0] for clip in clip_arrays], dtype=np.int64)

    joint_ids, resolved_joint_names = self.robot.find_joints(
      cfg.joint_names, preserve_order=True
    )
    if tuple(resolved_joint_names) != cfg.joint_names:
      raise ValueError(
        f"Go2 entity joint order mismatch: expected {cfg.joint_names}, got {resolved_joint_names}"
      )
    site_ids, resolved_site_names = self.robot.find_sites(
      cfg.foot_site_names, preserve_order=True
    )
    if tuple(resolved_site_names) != cfg.foot_site_names:
      raise ValueError(
        f"Go2 entity foot-site order mismatch: expected {cfg.foot_site_names}, got {resolved_site_names}"
      )
    self.joint_ids = torch.as_tensor(joint_ids, dtype=torch.long, device=self.device)
    self.site_ids = torch.as_tensor(site_ids, dtype=torch.long, device=self.device)
    qpos_addresses = self.robot.indexing.joint_q_adr[self.joint_ids]
    self.reference_joint_pos: torch.Tensor = torch.as_tensor(
      qpos_all[:, :, qpos_addresses.cpu().numpy()], device=self.device
    )
    joint_vel_all = np.empty_like(self.reference_joint_pos.cpu().numpy())
    for index, clip in enumerate(clip_arrays):
      joint_pos = clip["qpos"][:, qpos_addresses.cpu().numpy()]
      joint_vel = _finite_difference(joint_pos, fps)
      joint_vel_all[index, : joint_vel.shape[0]] = joint_vel
      joint_vel_all[index, joint_vel.shape[0] :] = joint_vel[-1]
    self.reference_joint_vel: torch.Tensor = torch.as_tensor(
      joint_vel_all,
      device=self.device,
    )
    # Shift only each clip's horizontal path to its local origin.  The vertical
    # component is an absolute base height and must be preserved.
    path_offset = np.zeros((len(clip_arrays), 1, 3), dtype=np.float32)
    path_offset[:, 0, :2] = source_root_pos_all[:, 0, :2]
    root_pos_all = source_root_pos_all - path_offset
    self.reference_root_pos_rel: torch.Tensor = torch.as_tensor(
      root_pos_all, device=self.device
    )
    self.reference_root_quat: torch.Tensor = torch.as_tensor(
      root_quat_all, device=self.device
    )
    root_lin_vel_w_array = np.stack(
      [_finite_difference(root_pos_all[i], fps) for i in range(len(clip_arrays))], axis=0
    )
    root_lin_vel_w_tensor = torch.as_tensor(root_lin_vel_w_array, device=self.device)
    self.reference_root_lin_vel_b: torch.Tensor = quat_apply_inverse(
      self.reference_root_quat, root_lin_vel_w_tensor
    )
    self.reference_root_ang_vel_b: torch.Tensor = torch.zeros_like(
      self.reference_root_lin_vel_b
    )
    self.reference_foot_target_base: torch.Tensor = torch.as_tensor(
      foot_base_all, device=self.device
    )
    self.reference_contact: torch.Tensor = torch.as_tensor(
      foot_world_all[..., 2] <= cfg.contact_height_threshold,
      dtype=torch.float32,
      device=self.device,
    )
    self.reference_fps: float = fps
    self.frame_counts: torch.Tensor = torch.as_tensor(
      frame_counts, dtype=torch.long, device=self.device
    )
    self.num_motions = len(clip_arrays)
    self.motion_ids: torch.Tensor = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self.time_steps: torch.Tensor = torch.zeros(
      self.num_envs, dtype=torch.float32, device=self.device
    )
    # Kept for compatibility with tools that inspect the command.  Runtime
    # termination uses frame_counts[motion_ids], not this maximum value.
    self.frame_count = max_frames
    self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_foot_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_root_vel"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def phase(self) -> torch.Tensor:
    selected_counts = self.frame_counts[self.motion_ids]
    return self.time_steps / torch.clamp((selected_counts - 1).to(torch.float32), min=1.0)

  @property
  def joint_pos(self) -> torch.Tensor:
    return _sample_motion_sequence(
      self.reference_joint_pos, self.motion_ids, self.time_steps, self.frame_counts
    )

  @property
  def joint_vel(self) -> torch.Tensor:
    return _sample_motion_sequence(
      self.reference_joint_vel, self.motion_ids, self.time_steps, self.frame_counts
    )

  @property
  def root_pos_w(self) -> torch.Tensor:
    return _sample_motion_sequence(
      self.reference_root_pos_rel, self.motion_ids, self.time_steps, self.frame_counts
    ) + self._env.scene.env_origins

  @property
  def root_quat_w(self) -> torch.Tensor:
    return _sample_motion_sequence(
      self.reference_root_quat, self.motion_ids, self.time_steps, self.frame_counts
    )

  @property
  def root_lin_vel_b(self) -> torch.Tensor:
    return _sample_motion_sequence(
      self.reference_root_lin_vel_b, self.motion_ids, self.time_steps, self.frame_counts
    )

  @property
  def root_ang_vel_b(self) -> torch.Tensor:
    return _sample_motion_sequence(
      self.reference_root_ang_vel_b, self.motion_ids, self.time_steps, self.frame_counts
    )

  @property
  def foot_target_base(self) -> torch.Tensor:
    return _sample_motion_sequence(
      self.reference_foot_target_base, self.motion_ids, self.time_steps, self.frame_counts
    )

  @property
  def contact(self) -> torch.Tensor:
    return _sample_motion_sequence(
      self.reference_contact, self.motion_ids, self.time_steps, self.frame_counts
    )

  @property
  def command(self) -> torch.Tensor:
    return torch.cat((self.joint_pos, self.joint_vel), dim=-1)

  @property
  def finished(self) -> torch.Tensor:
    return self.time_steps >= (self.frame_counts[self.motion_ids] - 1).to(self.time_steps.dtype)

  def _actual_foot_pos_base(self) -> torch.Tensor:
    root_pos = self.robot.data.root_link_pos_w
    root_quat = self.robot.data.root_link_quat_w
    foot_pos_w = self.robot.data.site_pos_w[:, self.site_ids]
    expanded_quat = root_quat[:, None, :].expand(-1, foot_pos_w.shape[1], -1)
    return quat_apply_inverse(expanded_quat, foot_pos_w - root_pos[:, None, :])

  def _update_metrics(self) -> None:
    self.metrics["error_joint_pos"] = torch.linalg.vector_norm(
      self.joint_pos - self.robot.data.joint_pos[:, self.joint_ids], dim=-1
    )
    self.metrics["error_joint_vel"] = torch.linalg.vector_norm(
      self.joint_vel - self.robot.data.joint_vel[:, self.joint_ids], dim=-1
    )
    self.metrics["error_foot_pos"] = torch.linalg.vector_norm(
      self.foot_target_base - self._actual_foot_pos_base(), dim=-1
    ).mean(dim=-1)
    self.metrics["error_root_vel"] = torch.linalg.vector_norm(
      self.root_lin_vel_b - self.robot.data.root_link_lin_vel_b, dim=-1
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if self.cfg.randomize_clip:
      self.motion_ids[env_ids] = torch.randint(
        self.num_motions, (len(env_ids),), device=self.device
      )
    else:
      self.motion_ids[env_ids] = 0
    if self.cfg.randomize_start_phase:
      low, high = self.cfg.start_phase_range
      if not 0.0 <= low <= high <= 1.0:
        raise ValueError(f"start_phase_range must be inside [0, 1], got {self.cfg.start_phase_range}")
      phase = torch.rand(len(env_ids), device=self.device) * (high - low) + low
    else:
      phase = torch.zeros(len(env_ids), device=self.device)
    selected_counts = self.frame_counts[self.motion_ids[env_ids]]
    self.time_steps[env_ids] = phase * (selected_counts - 1).to(torch.float32)
    root_pos = self.root_pos_w[env_ids]
    root_quat = self.root_quat_w[env_ids]
    root_velocity = torch.zeros((len(env_ids), 6), device=self.device)
    joint_velocity = torch.zeros_like(self.joint_pos[env_ids])
    self.robot.write_root_state_to_sim(
      torch.cat((root_pos, root_quat, root_velocity), dim=-1), env_ids=env_ids
    )
    self.robot.write_joint_state_to_sim(
      self.joint_pos[env_ids], joint_velocity, joint_ids=self.joint_ids, env_ids=env_ids
    )
    self.robot.reset(env_ids=env_ids)

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    # Reset-time compute must leave the command at frame 0.  The per-step path
    # advances by the exact environment dt, allowing 60 Hz references to be
    # interpolated at Nazarite's 50 Hz policy rate.
    if env_ids is None:
      self.time_steps += self._env.step_dt * self.reference_fps

  @staticmethod
  def _resolve_clip_paths(motion_path: Path) -> list[Path]:
    """Resolve an NPZ path or a dataset manifest into preprocessed NPZ paths."""
    if motion_path.suffix.lower() != ".json":
      return [motion_path]
    import json

    with motion_path.open(encoding="utf-8") as stream:
      manifest = json.load(stream)
    clips = manifest.get("clips")
    if not isinstance(clips, list):
      raise TypeError(f"reference manifest must contain a 'clips' list: {motion_path}")
    paths = []
    for index, entry in enumerate(clips):
      if not isinstance(entry, dict) or not entry.get("preprocessed_npz"):
        raise ValueError(f"manifest clip {index} lacks 'preprocessed_npz': {motion_path}")
      path = Path(str(entry["preprocessed_npz"])).expanduser()
      if not path.is_absolute():
        path = motion_path.parent / path
      paths.append(path.resolve())
    return paths

  @staticmethod
  def _load_clip(
    motion_path: Path,
    cfg: ReferenceMotionCommandCfg,
    nq: int,
  ) -> _ReferenceClip:
    if not motion_path.is_file():
      raise FileNotFoundError(f"reference clip not found: {motion_path}")
    with np.load(motion_path, allow_pickle=False) as archive:
      required = {
        "fps", "qpos", "smp_joint_order", "smp_leg_order", "root_pos_mujoco",
        "root_quat_wxyz", "foot_target_world_m",
      }
      missing = sorted(required - set(archive.files))
      if missing:
        raise ValueError(f"reference NPZ {motion_path} lacks: {', '.join(missing)}")
      source_joint_order = tuple(str(name) for name in archive["smp_joint_order"].tolist())
      source_leg_order = tuple(str(name) for name in archive["smp_leg_order"].tolist())
      if source_joint_order != cfg.joint_names:
        raise ValueError(f"reference joint order must be {cfg.joint_names}, got {source_joint_order}")
      if source_leg_order != cfg.foot_site_names:
        raise ValueError(f"reference leg order must be {cfg.foot_site_names}, got {source_leg_order}")
      qpos = _load_array(archive, "qpos", (nq,))
      root_pos = _load_array(archive, "root_pos_mujoco", (3,))
      root_quat = _load_array(archive, "root_quat_wxyz", (4,))
      foot_world = _load_array(archive, "foot_target_world_m", (4, 3))
      if "foot_target_base_m" in archive.files:
        foot_base = _load_array(archive, "foot_target_base_m", (4, 3))
      else:
        root_rotation = matrix_from_quat(torch.as_tensor(root_quat)).numpy()
        path_offset = np.array((root_pos[0, 0], root_pos[0, 1], 0.0), dtype=np.float32)
        root_pos_rel = root_pos - path_offset
        foot_world_rel = foot_world - path_offset[None, None, :]
        foot_base = np.einsum(
          "fki,fij->fkj", foot_world_rel - root_pos_rel[:, None, :], root_rotation
        ).astype(np.float32)
      fps = float(np.asarray(archive["fps"]).item())
    if fps <= 0.0:
      raise ValueError(f"reference fps must be positive: {motion_path}")
    if qpos.shape[0] < 2:
      raise ValueError(f"reference must contain at least two frames: {motion_path}")
    if np.any(np.linalg.norm(root_quat, axis=1) < 1.0e-6):
      raise ValueError(f"reference root quaternions must be non-zero: {motion_path}")
    return {
      "fps": fps,
      "qpos": qpos,
      "root_pos_mujoco": root_pos,
      "root_quat_wxyz": root_quat,
      "foot_target_world_m": foot_world,
      "foot_target_base_m": foot_base,
    }
