from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> dict[str, torch.Tensor]:
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None

  # Compute the distance the robot walked.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
    dim=1,
  )

  # Robots that walked far enough progress to harder terrains.
  move_up = distance > terrain_generator.size[0] / 2

  # Robots that walked less than half of their required distance go to
  # simpler terrains.
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  )
  move_down *= ~move_up

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  # Compute per-terrain-type mean levels.
  levels = terrain.terrain_levels.float()
  result: dict[str, torch.Tensor] = {
    "mean": torch.mean(levels),
    "max": torch.max(levels),
  }

  # In curriculum mode num_cols == num_terrains (one column per type),
  # so the column index directly maps to the sub-terrain name.
  sub_terrain_names = list(terrain_generator.sub_terrains.keys())
  terrain_origins = terrain.terrain_origins
  assert terrain_origins is not None
  num_cols = terrain_origins.shape[1]
  if num_cols == len(sub_terrain_names):
    types = terrain.terrain_types
    for i, name in enumerate(sub_terrain_names):
      mask = types == i
      if mask.any():
        result[name] = torch.mean(levels[mask])

  return result


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  for stage in velocity_stages:
    if env.common_step_counter >= stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
  return {
    "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
  }


class _DimState(TypedDict):
  min: float
  max: float
  step_up: float
  std: float
  cap: torch.Tensor
  metric: str
  ema: torch.Tensor


# Command tracking metric used as the curriculum signal for each velocity
# dimension. yaw uses the existing ``error_vel_yaw`` metric (which reflects the
# heading-controlled command for heading envs) rather than a per-axis
# ``error_vel_z``.
_DIM_METRIC = {
  "lin_vel_x": "error_vel_x",
  "lin_vel_y": "error_vel_y",
  "ang_vel_z": "error_vel_yaw",
}


class AdaptiveVelocityCommand:
  """Per-env adaptive velocity curriculum, one independent dimension per axis.

  Each environment's speed caps (forward ``lin_vel_x``, symmetric ``lin_vel_y``
  and ``ang_vel_z``) are adjusted independently when its episode ends, based on
  two signals from the just-completed episode:

  * **survival**: whether it lived through ``survival_ratio`` of the max
    episode length, or fell early.
  * **tracking**: the EMA of the mean per-axis tracking error, converted to a
    0-1 quality score ``exp(-(mean_err / std)²)``.

  Unlike the previous single-axis version, each dimension uses its *own*
  tracking error (``error_vel_x`` / ``error_vel_y`` / ``error_vel_yaw``) so
  that, e.g., poor lateral tracking no longer holds back the forward-speed
  curriculum. A cap is raised only when the env survived *and* tracks above
  ``up_threshold``, and lowered when it fell early *or* tracks below
  ``down_threshold``. A hysteresis band between the two thresholds keeps the
  cap stable against single-episode noise. A warmup period disables lowering
  so the policy can first learn to walk.

  The per-env caps are written to the command term's ``{dim}_max`` buffers,
  which ``UniformVelocityCommand._resample_command`` uses as sampling bounds.
  This runs before the command manager resamples, so the updated caps take
  effect for the next episode.
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    self._command_name: str = cfg.params["command_name"]
    self._decay: float = cfg.params["decay"]
    self._up_threshold: float = cfg.params["up_threshold"]
    self._down_threshold: float = cfg.params["down_threshold"]
    self._ema: float = cfg.params["ema"]
    self._warmup_steps: int = cfg.params["warmup_steps"]
    self._survival_ratio: float = cfg.params["survival_ratio"]

    command_term = cast(
      UniformVelocityCommand, env.command_manager.get_term(self._command_name)
    )
    self._command_term = command_term
    self._max_command_step: float = (
      command_term.cfg.resampling_time_range[1] / env.step_dt
    )
    self._max_ep_len: int = env.max_episode_length
    # An env is considered to have survived if it lived through at least this
    # fraction of the max episode length (e.g. 0.98 -> 980 of 1000 steps).
    self._survival_steps: int = int(self._survival_ratio * self._max_ep_len)

    # Build per-dimension state. The cap lives on the command term (so
    # _resample_command reads it); the tracking-quality EMA lives here.
    self._dims: dict[str, _DimState] = {}
    for dim, dim_cfg in cfg.params["dims"].items():
      cap = cast(torch.Tensor, getattr(command_term, f"{dim}_max"))
      cap[:] = float(dim_cfg["min"])
      self._dims[dim] = {
        "min": float(dim_cfg["min"]),
        "max": float(dim_cfg["max"]),
        "step_up": float(dim_cfg["step_up"]),
        "std": float(dim_cfg["std"]),
        "cap": cap,
        "metric": _DIM_METRIC[dim],
        # EMA initialized at the up threshold (neutral start).
        "ema": torch.full((env.num_envs,), self._up_threshold, device=env.device),
      }

  def __call__(
    self, env: ManagerBasedRlEnv, env_ids: torch.Tensor, **kwargs: Any
  ) -> dict[str, torch.Tensor]:
    del kwargs  # Params are read from cfg in __init__.
    if isinstance(env_ids, slice):
      env_ids = torch.arange(env.num_envs, device=env.device)

    ep_len = env.episode_length_buf
    survived = ep_len[env_ids] >= self._survival_steps
    result: dict[str, torch.Tensor] = {}

    for dim, state in self._dims.items():
      cap = state["cap"]
      # Mean per-axis tracking error over the finished episode, recovered from
      # the command term's accumulated metric (summed as err / max_command_step).
      metric_sum = cast(torch.Tensor, self._command_term.metrics[state["metric"]])
      mean_err = (
        metric_sum[env_ids] * self._max_command_step / ep_len[env_ids].clamp(min=1)
      )
      quality = torch.exp(-((mean_err / state["std"]) ** 2))

      ema = state["ema"]
      ema[env_ids] = self._ema * ema[env_ids] + (1.0 - self._ema) * quality

      up = survived & (ema[env_ids] > self._up_threshold)
      down = (~survived) | (ema[env_ids] < self._down_threshold)

      new_cap = cap[env_ids].clone()
      new_cap[up] = (new_cap[up] + state["step_up"]).clamp(max=state["max"])
      # Lowering is disabled during warmup so the policy can learn to walk.
      if env.common_step_counter >= self._warmup_steps:
        new_cap[down] = (new_cap[down] * self._decay).clamp(min=state["min"])
      cap[env_ids] = new_cap

      result[f"{dim}_max_mean"] = cap.mean()
      result[f"{dim}_max_max"] = cap.max()
      result[f"{dim}_ema_mean"] = ema[env_ids].mean()

    return result


def _randomization_scale(env: ManagerBasedRlEnv) -> float:
  curriculum = getattr(env, "_velocity_curriculum", None)
  if curriculum is None:
    return 0.0
  return float(curriculum.randomization_scale)


def _scale_symmetric_range(
  value: tuple[float, float], scale: float
) -> tuple[float, float]:
  return value[0] * scale, value[1] * scale


def staged_push_robot(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  velocity_range: dict[str, tuple[float, float]],
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> None:
  scale = _randomization_scale(env)
  if scale <= 0.0:
    return
  scaled_range = {
    name: _scale_symmetric_range(bounds, scale)
    for name, bounds in velocity_range.items()
  }
  envs_mdp.push_by_setting_velocity(
    env, env_ids, velocity_range=scaled_range, asset_cfg=asset_cfg
  )


@requires_model_fields(
  "body_mass",
  "body_ipos",
  "body_inertia",
  "body_iquat",
  recompute=RecomputeLevel.set_const,
)
def staged_pseudo_inertia(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  alpha_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
  distribution: str = "uniform",
) -> None:
  scale = _randomization_scale(env)
  envs_mdp.dr.pseudo_inertia(
    env,
    env_ids,
    alpha_range=_scale_symmetric_range(alpha_range, scale),
    asset_cfg=asset_cfg,
    distribution=distribution,
  )


@requires_model_fields("body_mass", recompute=RecomputeLevel.set_const)
def staged_body_mass(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
  distribution: str = "uniform",
  operation: str = "scale",
) -> None:
  scale = _randomization_scale(env)
  envs_mdp.dr.body_mass(
    env,
    env_ids,
    ranges=_scale_symmetric_range(ranges, scale),
    asset_cfg=asset_cfg,
    distribution=distribution,
    operation=operation,
  )


class StagedVelocityCommand:
  """Global velocity stages with a robustness phase for each stage.

  A stage first trains with no staged push or mass randomization. Once the
  global episode window meets the survival and tracking gates, the same speed
  range is trained with the stage's randomization level. A later successful
  window advances to the next global speed range.
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    self._command_name: str = cfg.params["command_name"]
    self._stages: list[dict[str, Any]] = cfg.params["stages"]
    self._min_phase_episodes: int = int(cfg.params["min_phase_episodes"])
    self._evaluation_window_episodes: int = int(
      cfg.params["evaluation_window_episodes"]
    )
    self._required_windows: int = int(cfg.params["required_windows"])
    self._survival_threshold: float = float(cfg.params["survival_threshold"])
    self._track_threshold: float = float(cfg.params["track_threshold"])
    self._high_speed_track_threshold: float = float(
      cfg.params["high_speed_track_threshold"]
    )
    self._direct_start_threshold: float = float(cfg.params["direct_start_threshold"])
    self._direct_start_success_threshold: float = float(
      cfg.params["direct_start_success_threshold"]
    )
    self._randomization_enabled = bool(cfg.params.get("randomization_enabled", True))
    self._max_invalid_rate = float(cfg.params.get("max_invalid_rate", 0.0))
    self._max_action_outlier_rate = float(
      cfg.params.get("max_action_outlier_rate", self._max_invalid_rate)
    )
    self._max_state_limit_rate = float(
      cfg.params.get("max_state_limit_rate", self._max_invalid_rate)
    )
    self._max_nan_rate = float(cfg.params.get("max_nan_rate", self._max_invalid_rate))

    command_term = cast(
      UniformVelocityCommand, env.command_manager.get_term(self._command_name)
    )
    self._command_term = command_term
    self._stage_index = 0
    self._phase = 0
    self._phase_episodes = 0
    self._window_episodes = 0
    self._successful_windows = 0
    self._survival_sum = 0.0
    self._track_sum = 0.0
    self._high_speed_track_sum = 0.0
    self._high_speed_steps = 0.0
    self._direct_start_attempts = 0.0
    self._direct_start_successes = 0.0
    self._last_survival_rate = 0.0
    self._last_track_quality = 0.0
    self._last_high_speed_track = 0.0
    self._last_direct_start_success = 0.0
    self._last_invalid_rate = 0.0
    self._last_action_outlier_rate = 0.0
    self._last_state_limit_rate = 0.0
    self._last_nan_rate = 0.0
    self._randomization_multiplier = 1.0
    self._apply_stage()
    object.__setattr__(env, "_velocity_curriculum", self)

  @property
  def randomization_scale(self) -> float:
    if self._phase == 0 or not self._randomization_enabled:
      return 0.0
    return float(
      self._stages[self._stage_index]["randomization_scale"]
      * self._randomization_multiplier
    )

  def _apply_stage(self) -> None:
    stage = self._stages[self._stage_index]
    self._command_term.cfg.ranges.lin_vel_x = tuple(stage["lin_vel_x"])
    self._command_term.cfg.ranges.lin_vel_y = tuple(stage["lin_vel_y"])
    self._command_term.cfg.ranges.ang_vel_z = tuple(stage["ang_vel_z"])
    self._command_term.lin_vel_x_max.fill_(stage["lin_vel_x"][1])
    self._command_term.lin_vel_y_max.fill_(stage["lin_vel_y"][1])
    self._command_term.ang_vel_z_max.fill_(stage["ang_vel_z"][1])

  def _reset_window(self) -> None:
    self._window_episodes = 0
    self._survival_sum = 0.0
    self._track_sum = 0.0
    self._high_speed_track_sum = 0.0
    self._high_speed_steps = 0.0
    self._direct_start_attempts = 0.0
    self._direct_start_successes = 0.0

  def _advance_if_ready(self) -> None:
    high_speed_required = self._stages[self._stage_index]["lin_vel_x"][1] >= (
      self._direct_start_threshold
    )
    direct_ok = not high_speed_required or (
      self._direct_start_attempts > 0.0
      and self._last_direct_start_success >= self._direct_start_success_threshold
    )
    high_speed_ok = (
      not high_speed_required
      or self._last_high_speed_track >= self._high_speed_track_threshold
    )
    stability_ok = (
      self._last_invalid_rate <= self._max_invalid_rate
      and self._last_action_outlier_rate <= self._max_action_outlier_rate
      and self._last_state_limit_rate <= self._max_state_limit_rate
      and self._last_nan_rate <= self._max_nan_rate
    )
    ready = (
      self._last_survival_rate >= self._survival_threshold
      and self._last_track_quality >= self._track_threshold
      and direct_ok
      and high_speed_ok
      and stability_ok
    )
    if ready:
      self._successful_windows += 1
    else:
      self._successful_windows = 0

    if self._successful_windows < self._required_windows:
      if self._phase == 1 and not stability_ok:
        # Back off perturbations after a numerical failure, then ramp them back
        # up through clean evaluation windows.
        self._randomization_multiplier = max(0.25, self._randomization_multiplier * 0.5)
      self._reset_window()
      return
    self._successful_windows = 0
    self._phase_episodes = 0
    self._reset_window()
    if self._phase == 0:
      self._phase = 1
      self._randomization_multiplier = 1.0
    elif self._randomization_multiplier < 1.0:
      self._randomization_multiplier = min(1.0, self._randomization_multiplier + 0.25)
    elif self._stage_index < len(self._stages) - 1:
      self._stage_index += 1
      self._phase = 0
      self._randomization_multiplier = 1.0
      self._apply_stage()

  def __call__(
    self, env: ManagerBasedRlEnv, env_ids: torch.Tensor, **kwargs: Any
  ) -> dict[str, torch.Tensor]:
    del kwargs
    if isinstance(env_ids, slice):
      env_ids = torch.arange(env.num_envs, device=env.device)
    if env.common_step_counter > 0 and len(env_ids) > 0:
      command_metrics = self._command_term.metrics
      survival = (
        env.episode_length_buf[env_ids] >= env.max_episode_length * 0.95
      ).float()
      track_steps = command_metrics["track_x_step_count"][env_ids]
      track_quality = torch.where(
        track_steps > 0.0,
        command_metrics["track_x_quality_sum"][env_ids] / track_steps,
        torch.zeros_like(track_steps),
      )
      high_steps = command_metrics["high_speed_step_count"][env_ids]
      high_quality = torch.where(
        high_steps > 0.0,
        command_metrics["high_speed_track_sum"][env_ids] / high_steps,
        torch.zeros_like(high_steps),
      )
      attempts = command_metrics["direct_start_attempt"][env_ids]
      successes = command_metrics["direct_start_success"][env_ids]
      self._phase_episodes += len(env_ids)
      self._window_episodes += len(env_ids)
      self._survival_sum += float(survival.sum().item())
      self._track_sum += float(track_quality.sum().item())
      valid_high = high_steps > 0.0
      self._high_speed_steps += float(valid_high.sum().item())
      self._high_speed_track_sum += float(high_quality[valid_high].sum().item())
      self._direct_start_attempts += float(attempts.sum().item())
      self._direct_start_successes += float(successes.sum().item())
      self._last_survival_rate = self._survival_sum / max(self._window_episodes, 1)
      self._last_track_quality = self._track_sum / max(self._window_episodes, 1)
      self._last_high_speed_track = self._high_speed_track_sum / max(
        self._high_speed_steps, 1.0
      )
      self._last_direct_start_success = self._direct_start_successes / max(
        self._direct_start_attempts, 1.0
      )
      episode_lengths = env.episode_length_buf[env_ids].float().clamp(min=1.0)
      invalid_rate = env.episode_invalid_steps[env_ids].float() / episode_lengths
      action_outlier_rate = (
        env.episode_action_outlier_steps[env_ids].float() / episode_lengths
      )
      state_limit_rate = (
        env.episode_state_limit_steps[env_ids].float() / episode_lengths
      )
      nan_rate = env.episode_nan_steps[env_ids].float() / episode_lengths
      previous_count = self._window_episodes - len(env_ids)
      self._last_invalid_rate = (
        self._last_invalid_rate * previous_count + float(invalid_rate.sum().item())
      ) / max(self._window_episodes, 1)
      self._last_action_outlier_rate = (
        self._last_action_outlier_rate * previous_count
        + float(action_outlier_rate.sum().item())
      ) / max(self._window_episodes, 1)
      self._last_state_limit_rate = (
        self._last_state_limit_rate * previous_count
        + float(state_limit_rate.sum().item())
      ) / max(self._window_episodes, 1)
      self._last_nan_rate = (
        self._last_nan_rate * previous_count + float(nan_rate.sum().item())
      ) / max(self._window_episodes, 1)
      if (
        self._phase_episodes >= self._min_phase_episodes
        and self._window_episodes >= self._evaluation_window_episodes
      ):
        self._advance_if_ready()

    return {
      "stage": torch.tensor(self._stage_index, device=env.device, dtype=torch.float),
      "phase": torch.tensor(self._phase, device=env.device, dtype=torch.float),
      "randomization_level": torch.tensor(
        self.randomization_scale, device=env.device, dtype=torch.float
      ),
      "survival_rate": torch.tensor(
        self._last_survival_rate, device=env.device, dtype=torch.float
      ),
      "track_quality": torch.tensor(
        self._last_track_quality, device=env.device, dtype=torch.float
      ),
      "high_speed_track_quality": torch.tensor(
        self._last_high_speed_track, device=env.device, dtype=torch.float
      ),
      "direct_start_success": torch.tensor(
        self._last_direct_start_success, device=env.device, dtype=torch.float
      ),
      "invalid_rate": torch.tensor(
        self._last_invalid_rate, device=env.device, dtype=torch.float
      ),
      "action_outlier_rate": torch.tensor(
        self._last_action_outlier_rate, device=env.device, dtype=torch.float
      ),
      "state_limit_rate": torch.tensor(
        self._last_state_limit_rate, device=env.device, dtype=torch.float
      ),
      "nan_rate": torch.tensor(
        self._last_nan_rate, device=env.device, dtype=torch.float
      ),
    }
