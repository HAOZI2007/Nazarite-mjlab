from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import torch

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)

if TYPE_CHECKING:
    from mjlab.viewer.debug_visualizer import DebugVisualizer


@dataclass(kw_only=True)
class HimBehaviorCommandCfg(CommandTermCfg):
    """HIM-only morphology conditions: body height and stance width.

    This intentionally does not expose WTW gait, frequency, pitch, phase, or
    swing-height state. The command tensor is exactly two-dimensional.
    """

    entity_name: str
    body_height_range: tuple[float, float] = (-0.10, 0.025)
    stance_width_range: tuple[float, float] = (0.18, 0.29)
    freeze_when_standing: bool = True
    velocity_command_name: str = "twist"
    standing_threshold: float = 0.1

    def build(self, env: ManagerBasedRlEnv) -> HimBehaviorCommand:
        return HimBehaviorCommand(self, env)

    def __post_init__(self) -> None:
        if self.body_height_range[1] < self.body_height_range[0]:
            raise ValueError("body_height_range must be increasing")
        if self.stance_width_range[0] <= 0.0 or self.stance_width_range[1] < self.stance_width_range[0]:
            raise ValueError("stance_width_range must be positive and increasing")


class HimBehaviorCommand(CommandTerm):
    """Resampled [body_height_offset, stance_width] condition."""

    def __init__(self, cfg: HimBehaviorCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._command = torch.zeros(self.num_envs, 2, device=self.device)
        self._command[:, 1] = 0.5 * (cfg.stance_width_range[0] + cfg.stance_width_range[1])
        self._gui_enabled: viser.GuiCheckboxHandle | None = None
        self._gui_sliders: tuple[viser.GuiSliderHandle, viser.GuiSliderHandle] | None = None
        self._gui_get_env_idx: Callable[[], int] | None = None

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        sample_ids = env_ids
        if self.cfg.freeze_when_standing:
            twist = self._env.command_manager.get_command(self.cfg.velocity_command_name)
            standing = torch.linalg.norm(twist[env_ids], dim=1) < self.cfg.standing_threshold
            sample_ids = env_ids[~standing]
        if len(sample_ids) == 0:
            return
        count = len(sample_ids)
        self._command[sample_ids, 0] = torch.empty(count, device=self.device).uniform_(
            *self.cfg.body_height_range
        )
        self._command[sample_ids, 1] = torch.empty(count, device=self.device).uniform_(
            *self.cfg.stance_width_range
        )

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        del env_ids

    def create_gui(
        self,
        name: str,
        server: "viser.ViserServer",
        get_env_idx: Callable[[], int],
        on_change: Callable[[], None] | None = None,
        request_action: Callable[[str, Any], None] | None = None,
    ) -> None:
        """Expose body height and stance width controls in the play viewer."""
        del request_action
        from viser import Icon

        height_low = 0.32 + self.cfg.body_height_range[0]
        height_high = 0.32 + self.cfg.body_height_range[1]
        with server.gui.add_folder(name.capitalize()):
            enabled = server.gui.add_checkbox("Enable", initial_value=True)
            height = server.gui.add_slider(
                "Body height (m)", min=height_low, max=height_high,
                step=0.005, initial_value=0.32,
            )
            stance = server.gui.add_slider(
                "Stance width (m)", min=self.cfg.stance_width_range[0],
                max=self.cfg.stance_width_range[1], step=0.005,
                initial_value=0.5 * sum(self.cfg.stance_width_range),
            )
            neutral = server.gui.add_button("Neutral", icon=Icon.SQUARE_X)

            @neutral.on_click
            def _(_) -> None:
                height.value = 0.32
                stance.value = 0.5 * sum(self.cfg.stance_width_range)

            if on_change is not None:
                height.on_update(lambda _ev: on_change())
                stance.on_update(lambda _ev: on_change())
                enabled.on_update(lambda _ev: on_change())

        self._gui_enabled = enabled
        self._gui_sliders = (height, stance)
        self._gui_get_env_idx = get_env_idx

    def compute(
        self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> None:
        super().compute(dt, env_ids)
        if self._gui_enabled is None or not self._gui_enabled.value:
            return
        assert self._gui_sliders is not None and self._gui_get_env_idx is not None
        idx = self._gui_get_env_idx()
        height, stance = self._gui_sliders
        self._command[idx, 0] = height.value - 0.32
        self._command[idx, 1] = stance.value


@dataclass(kw_only=True)
class TerrainGoalCommandCfg(CommandTermCfg):
    """Explicit 2-D traversal goal for terrain-aware locomotion.

    The goal is command state rather than an actor observation.  This keeps the
    policy observation contract unchanged while making the exact target used by
    rewards available to the viewer and to diagnostics.
    """

    goal_distance_range: tuple[float, float] = (1.2, 1.5)
    goal_lateral_range: tuple[float, float] = (0.0, 0.0)
    marker_height: float = 0.05
    marker_radius: float = 0.06
    avoid_void: bool = True
    """Reject candidates whose downward ray only reaches a deep pit floor."""

    max_void_drop: float = 0.6
    """Maximum support drop relative to the terrain spawn height."""

    max_resample_attempts: int = 12
    """Number of support-aware candidate batches before using a safe fallback."""

    def build(self, env: ManagerBasedRlEnv) -> TerrainGoalCommand:
        return TerrainGoalCommand(self, env)

    def __post_init__(self) -> None:
        if self.goal_distance_range[0] <= 0.0:
            raise ValueError("Terrain goal distance must be positive")
        if self.goal_distance_range[1] < self.goal_distance_range[0]:
            raise ValueError("Terrain goal distance range must be increasing")
        if self.goal_lateral_range[1] < self.goal_lateral_range[0]:
            raise ValueError("Terrain goal lateral range must be increasing")
        if self.marker_height < 0.0 or self.marker_radius <= 0.0:
            raise ValueError("Terrain goal marker dimensions must be positive")
        if self.max_void_drop <= 0.0:
            raise ValueError("max_void_drop must be positive")
        if self.max_resample_attempts < 1:
            raise ValueError("max_resample_attempts must be positive")


class TerrainGoalCommand(CommandTerm):
    """World-frame target point sampled relative to each terrain origin."""

    cfg: TerrainGoalCommandCfg

    def __init__(self, cfg: TerrainGoalCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.cfg = cfg
        self.goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.metrics["distance_to_goal"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["goal_x"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["goal_y"] = torch.zeros(self.num_envs, device=self.device)

        self.robot = env.scene["robot"]

    @property
    def command(self) -> torch.Tensor:
        """Return the explicit world-frame goal ``[x, y, z]``."""
        return self.goal_pos_w

    def _update_metrics(self) -> None:
        delta = self.goal_pos_w[:, :2] - self.robot.data.root_link_pos_w[:, :2]
        self.metrics["distance_to_goal"] = torch.linalg.vector_norm(delta, dim=-1)
        self.metrics["goal_x"] = self.goal_pos_w[:, 0]
        self.metrics["goal_y"] = self.goal_pos_w[:, 1]

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        origins = self._env.scene.env_origins[env_ids]
        base_z = self.robot.data.root_link_pos_w[env_ids, 2]

        # Candidate points are checked against the actual MuJoCo terrain. This
        # matters for gaps/stepping-stones: a ray can hit their deep floor even
        # though the point is not a usable support surface.
        remaining = torch.arange(len(env_ids), device=self.device)
        selected_xy = torch.zeros(len(env_ids), 2, device=self.device)
        selected_z = base_z + self.cfg.marker_height
        for _ in range(self.cfg.max_resample_attempts):
            if len(remaining) == 0:
                break
            count = len(remaining)
            distances = torch.empty(count, device=self.device).uniform_(
                *self.cfg.goal_distance_range
            )
            lateral = torch.empty(count, device=self.device).uniform_(
                *self.cfg.goal_lateral_range
            )
            candidate_origins = origins[remaining]
            candidate_xy = torch.stack(
                (
                    candidate_origins[:, 0] + distances,
                    candidate_origins[:, 1] + lateral,
                ),
                dim=-1,
            )
            support = self._probe_support(candidate_xy, candidate_origins)
            if support is None:
                # CPU ray queries are unavailable in lightweight tests or
                # alternate backends. Preserve the old behavior there.
                valid = torch.ones(count, dtype=torch.bool, device=self.device)
                support_z = base_z[remaining] + self.cfg.marker_height
            else:
                valid, hit_z = support
                support_z = hit_z + self.cfg.marker_height
            if valid.any():
                chosen = remaining[valid]
                selected_xy[chosen] = candidate_xy[valid]
                selected_z[chosen] = support_z[valid]
                remaining = remaining[~valid]

        if len(remaining) > 0:
            # The reset origin is guaranteed to be a support patch for the
            # terrain generators used here. This fallback is intentionally
            # conservative rather than leaving a target in a deep void.
            fallback_xy = origins[remaining, :2].clone()
            fallback_xy[:, 0] += min(float(self.cfg.goal_distance_range[0]), 0.3)
            selected_xy[remaining] = fallback_xy
            selected_z[remaining] = base_z[remaining] + self.cfg.marker_height

        self.goal_pos_w[env_ids, :2] = selected_xy
        self.goal_pos_w[env_ids, 2] = selected_z

    def _probe_support(
        self,
        candidate_xy: torch.Tensor,
        candidate_origins: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Probe candidate points with host-side MuJoCo downward rays.

        The simulation uses Warp for batched stepping, but the static terrain
        geometry is also present in the host MuJoCo model. These infrequent
        reset-time queries avoid adding another actor observation or sensor.
        """
        if not self.cfg.avoid_void:
            return None
        try:
            import mujoco
            import numpy as np

            model = self._env.sim.mj_model
            data = self._env.sim.mj_data
        except (AttributeError, ImportError):
            return None

        valid_values: list[bool] = []
        hit_values: list[float] = []
        for xy, origin in zip(
            candidate_xy.detach().cpu().numpy(),
            candidate_origins.detach().cpu().numpy(),
            strict=True,
        ):
            ray_start = np.array(
                [float(xy[0]), float(xy[1]), float(origin[2]) + 3.0],
                dtype=np.float64,
            )
            ray_dir = np.array([0.0, 0.0, -1.0], dtype=np.float64)
            geom_id = np.array([-1], dtype=np.int32)
            normal = np.zeros(3, dtype=np.float64)
            try:
                distance = mujoco.mj_ray(
                    model,
                    data,
                    ray_start,
                    ray_dir,
                    None,
                    True,
                    -1,
                    geom_id,
                    normal,
                )
            except (RuntimeError, TypeError, ValueError):
                return None
            if distance < 0.0 or geom_id[0] < 0:
                valid_values.append(False)
                hit_values.append(float(origin[2]))
                continue
            hit_z = float(ray_start[2] - distance)
            hit_values.append(hit_z)
            valid_values.append(hit_z >= float(origin[2]) - self.cfg.max_void_drop)
        return (
            torch.tensor(valid_values, dtype=torch.bool, device=self.device),
            torch.tensor(hit_values, dtype=candidate_xy.dtype, device=self.device),
        )

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        del env_ids

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        goal_pos = self.goal_pos_w.detach().cpu().numpy()
        base_pos = self.robot.data.root_link_pos_w.detach().cpu().numpy()
        for batch in env_indices:
            goal = goal_pos[batch].copy()
            base = base_pos[batch].copy()
            # Draw the marker and an explicit direction vector in world frame.
            visualizer.add_sphere(
                center=goal,
                radius=self.cfg.marker_radius,
                color=(1.0, 0.75, 0.05, 0.95),
                label=f"terrain_goal_{batch}",
            )
            arrow_start = base.copy()
            arrow_start[2] += 0.08
            visualizer.add_arrow(
                start=arrow_start,
                end=goal,
                color=(1.0, 0.75, 0.05, 0.85),
                width=0.025,
                label=f"terrain_goal_direction_{batch}",
            )


@dataclass(kw_only=True)
class GridAdaptiveVelocityCommandCfg(UniformVelocityCommandCfg):
    """基于速度网格的自适应命令配置。

    网格的两个维度分别对应前向速度 ``lin_vel_x`` 和偏航角速度
    ``ang_vel_z``。横向速度仍在 ``ranges.lin_vel_y`` 中独立均匀采样。
    """

    grid_num_x: int = 9
    """前向速度方向的网格数量。"""

    grid_num_yaw: int = 7
    """偏航角速度方向的网格数量。"""

    initial_cell: tuple[int, int] | None = None
    """初始激活网格，格式为 ``(x_index, yaw_index)``。"""

    min_cell_visits: int = 20
    """扩展邻居前，一个网格至少需要完成的命令段数量。"""

    success_window_size: int = 100
    """用于判断课程扩展的近期成功率窗口大小。"""

    max_new_cells_per_update: int = 4
    """单个仿真步最多新激活的网格数量。"""

    require_all_active_cells_ready: bool = False
    """是否要求所有已激活 cell 达标后，才允许继续扩展新的邻居。"""

    success_rate_threshold: float = 0.8
    """扩展邻居所需的最低成功率。"""

    velocity_error_threshold: float = 0.4
    """线速度平均误差成功阈值，单位为 m/s。"""

    yaw_error_threshold: float = 0.35
    """偏航角速度平均误差成功阈值，单位为 rad/s。"""

    gait_quality_behavior_command_name: str | None = None
    """可选的 WTW 行为命令名；设置后，课程扩展会额外检查接触质量。"""

    gait_quality_sensor_name: str | None = None
    """用于读取四足接触状态的传感器名。"""

    gait_schedule_error_threshold: float | None = None
    """命令段平均接触时序误差上限；``None`` 表示不作为课程门槛。"""

    gait_sync_error_threshold: float | None = None
    """同步步态的四足接触分歧上限；``None`` 表示不作为课程门槛。"""

    gait_mixed_contact_threshold: float | None = None
    """同步步态高置信相位内的混合接触比例上限；``None`` 表示不检查。"""

    gait_contact_smoothing: float = 0.07
    """构造接触时序目标时使用的高斯 CDF 平滑宽度。"""

    def build(self, env: ManagerBasedRlEnv) -> GridAdaptiveVelocityCommand:
        return GridAdaptiveVelocityCommand(self, env)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.grid_num_x < 1 or self.grid_num_yaw < 1:
            raise ValueError("Grid dimensions must be positive.")
        if self.min_cell_visits < 1:
            raise ValueError("min_cell_visits must be at least 1.")
        if self.success_window_size < 1:
            raise ValueError("success_window_size must be at least 1.")
        if self.max_new_cells_per_update < 1:
            raise ValueError("max_new_cells_per_update must be at least 1.")
        if not 0.0 < self.success_rate_threshold <= 1.0:
            raise ValueError("success_rate_threshold must be in (0, 1].")
        if self.velocity_error_threshold <= 0.0 or self.yaw_error_threshold <= 0.0:
            raise ValueError("Velocity error thresholds must be positive.")
        gait_thresholds = (
            self.gait_schedule_error_threshold,
            self.gait_sync_error_threshold,
            self.gait_mixed_contact_threshold,
        )
        if any(threshold is not None for threshold in gait_thresholds):
            if (
                self.gait_quality_behavior_command_name is None
                or self.gait_quality_sensor_name is None
            ):
                raise ValueError(
                    "Gait quality thresholds require both "
                    "gait_quality_behavior_command_name and gait_quality_sensor_name."
                )
            if any(
                threshold is not None and not 0.0 <= threshold <= 1.0
                for threshold in gait_thresholds
            ):
                raise ValueError("Gait quality thresholds must be in [0, 1].")
        if self.gait_contact_smoothing <= 0.0:
            raise ValueError("gait_contact_smoothing must be positive.")
        if self.heading_command or self.rel_heading_envs != 0.0:
            raise ValueError(
                "GridAdaptiveVelocityCommand requires heading_command=False and "
                "rel_heading_envs=0.0."
            )


@dataclass(frozen=True, kw_only=True)
class ScheduledVelocityStageCfg:
    """Velocity range used after a global environment-step boundary."""

    step: int
    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    ang_vel_z: tuple[float, float]


@dataclass(kw_only=True)
class ScheduledVelocityCommandCfg(UniformVelocityCommandCfg):
    """Velocity command whose range is widened at fixed training milestones."""

    stages: tuple[ScheduledVelocityStageCfg, ...] = ()

    def build(self, env: ManagerBasedRlEnv) -> ScheduledVelocityCommand:
        return ScheduledVelocityCommand(self, env)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.heading_command or self.rel_heading_envs != 0.0:
            raise ValueError(
                "ScheduledVelocityCommand requires heading_command=False and "
                "rel_heading_envs=0.0."
            )
        previous_step = -1
        for stage in self.stages:
            if stage.step < 0 or stage.step < previous_step:
                raise ValueError("Scheduled velocity stage steps must be increasing.")
            previous_step = stage.step
            for name, value in (
                ("lin_vel_x", stage.lin_vel_x),
                ("lin_vel_y", stage.lin_vel_y),
                ("ang_vel_z", stage.ang_vel_z),
            ):
                if value[1] < value[0]:
                    raise ValueError(f"{name} range must be increasing, got {value}.")


class ScheduledVelocityCommand(UniformVelocityCommand):
    """Uniform command sampling with a range selected by global training steps."""

    cfg: ScheduledVelocityCommandCfg

    def __init__(
        self, cfg: ScheduledVelocityCommandCfg, env: ManagerBasedRlEnv
    ) -> None:
        super().__init__(cfg, env)
        self.cfg = cfg
        self.metrics["velocity_stage"] = torch.zeros(self.num_envs, device=self.device)

    def _stage_index(self) -> int:
        index = 0
        for i, stage in enumerate(self.cfg.stages):
            if self._env.common_step_counter >= stage.step:
                index = i
            else:
                break
        return index

    def _active_ranges(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        if not self.cfg.stages:
            return (
                self.cfg.ranges.lin_vel_x,
                self.cfg.ranges.lin_vel_y,
                self.cfg.ranges.ang_vel_z,
            )
        stage = self.cfg.stages[self._stage_index()]
        return stage.lin_vel_x, stage.lin_vel_y, stage.ang_vel_z

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        x_range, y_range, yaw_range = self._active_ranges()
        r = torch.empty(len(env_ids), device=self.device)
        self.vel_command_b[env_ids, 0] = r.uniform_(*x_range)
        self.vel_command_b[env_ids, 1] = r.uniform_(*y_range)
        self.vel_command_b[env_ids, 2] = r.uniform_(*yaw_range)
        self.is_standing_env[env_ids] = (
            r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
        )
        self.is_world_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_world_envs
        self.vel_command_w[env_ids] = self.vel_command_b[env_ids]
        self.is_forward_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_forward_envs
        forward_ids = env_ids[self.is_forward_env[env_ids]]
        if len(forward_ids) > 0:
            self.vel_command_b[forward_ids, 0] = (
                self.vel_command_b[forward_ids, 0].abs().clamp(min=0.3)
            )
            self.vel_command_b[forward_ids, 1:] = 0.0
        self.metrics["velocity_stage"][env_ids] = float(self._stage_index())


@dataclass(kw_only=True)
class TerrainConditionedVelocityCommandCfg(ScheduledVelocityCommandCfg):
    """Scheduled velocity command with terrain-specific forward-only ranges.

    The normal ranges and scheduled stages apply to every terrain.  Environments
    whose terrain column name is listed in ``forward_only_terrain_names`` receive
    a positive forward-x command with zero lateral and yaw velocity instead.
    """

    forward_only_terrain_names: tuple[str, ...] = ()
    """Terrain generator columns that should only be traversed in +x."""
    forward_only_x_min: float = 0.1
    """Lower bound for positive x commands on the selected terrains."""

    def build(self, env: ManagerBasedRlEnv) -> TerrainConditionedVelocityCommand:
        return TerrainConditionedVelocityCommand(self, env)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.forward_only_x_min < 0.0:
            raise ValueError("forward_only_x_min must be non-negative")


class TerrainConditionedVelocityCommand(ScheduledVelocityCommand):
    """Apply forward-only commands only to configured terrain columns."""

    cfg: TerrainConditionedVelocityCommandCfg

    def __init__(
        self, cfg: TerrainConditionedVelocityCommandCfg, env: ManagerBasedRlEnv
    ) -> None:
        super().__init__(cfg, env)
        self.cfg = cfg
        self._forward_only_envs = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._forward_only_type_ids = torch.empty(
            0, dtype=torch.long, device=self.device
        )

        if cfg.forward_only_terrain_names:
            terrain = env.scene["terrain"]
            terrain_cfg = terrain.cfg.terrain_generator
            if terrain_cfg is None:
                raise ValueError(
                    "forward_only_terrain_names requires a generated terrain scene"
                )
            terrain_names = list(terrain_cfg.sub_terrains)
            missing = set(cfg.forward_only_terrain_names) - set(terrain_names)
            if missing:
                raise ValueError(
                    "Unknown forward-only terrain names: " + ", ".join(sorted(missing))
                )
            selected_types = torch.tensor(
                [terrain_names.index(name) for name in cfg.forward_only_terrain_names],
                dtype=terrain.terrain_types.dtype,
                device=self.device,
            )
            self._forward_only_type_ids = selected_types
            self._forward_only_envs = torch.isin(terrain.terrain_types, selected_types)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        # Terrain curricula may reassign an environment to a different terrain type.
        # Refresh the mask at every command resample instead of freezing it at init.
        if self._forward_only_type_ids.numel() > 0:
            terrain = self._env.scene["terrain"]
            self._forward_only_envs[env_ids] = torch.isin(
                terrain.terrain_types[env_ids], self._forward_only_type_ids
            )
        selected_ids = env_ids[self._forward_only_envs[env_ids]]
        if len(selected_ids) == 0:
            return

        x_range, _, _ = self._active_ranges()
        x_low = max(float(self.cfg.forward_only_x_min), 0.0)
        x_high = max(float(x_range[1]), x_low)
        r = torch.empty(len(selected_ids), device=self.device)
        self.vel_command_b[selected_ids, 0] = r.uniform_(x_low, x_high)
        self.vel_command_b[selected_ids, 1:] = 0.0
        self.vel_command_w[selected_ids] = self.vel_command_b[selected_ids]
        self.is_forward_env[selected_ids] = True


class GridAdaptiveVelocityCommand(UniformVelocityCommand):
    """使用激活网格均匀采样，并根据成功率扩展四连通邻居。"""

    def __init__(
        self, cfg: GridAdaptiveVelocityCommandCfg, env: ManagerBasedRlEnv
    ) -> None:
        super().__init__(cfg, env)
        # 父类的 cfg 属性保持 UniformVelocityCommandCfg 类型，避免违反
        # Pylance 对可变属性类型覆盖的检查；自定义字段通过 _grid_cfg 访问。
        self._grid_cfg = cfg

        # 每个环境可以处于不同 cell，但所有并行环境共享一张课程地图。
        self.current_cell = torch.zeros(
            self.num_envs, 2, dtype=torch.long, device=self.device
        )
        self.active_cells = torch.zeros(
            self._grid_cfg.grid_num_x,
            self._grid_cfg.grid_num_yaw,
            dtype=torch.bool,
            device=self.device,
        )
        initial_cell = self._initial_cell()
        self.active_cells[initial_cell[0], initial_cell[1]] = True

        # 全局网格统计量：访问次数和成功次数。
        self.cell_visits = torch.zeros(
            self._grid_cfg.grid_num_x,
            self._grid_cfg.grid_num_yaw,
            dtype=torch.long,
            device=self.device,
        )
        self.cell_successes = torch.zeros_like(self.cell_visits)

        # 每个 cell 保存最近若干个命令段的成败结果。
        # 第一维展平为 cell，便于在不同 cell 上独立维护 ring buffer。
        num_cells = self._grid_cfg.grid_num_x * self._grid_cfg.grid_num_yaw
        self.cell_success_history = torch.zeros(
            num_cells,
            self._grid_cfg.success_window_size,
            dtype=torch.bool,
            device=self.device,
        )
        self.cell_recent_visits = torch.zeros(
            num_cells, dtype=torch.long, device=self.device
        )
        self.cell_recent_successes = torch.zeros_like(self.cell_recent_visits)
        self.cell_history_ptr = torch.zeros_like(self.cell_recent_visits)
        # 防止同一个 common_step_counter 内的多个 reset 批次重复扩展课程。
        self._last_expansion_step = -1

        # 当前命令段的逐环境统计量。
        self._segment_error_xy = torch.zeros(self.num_envs, device=self.device)
        self._segment_error_yaw = torch.zeros(self.num_envs, device=self.device)
        # 接触质量按命令段累积，和速度误差一样仅在命令切换/reset 时结算。
        self._segment_gait_schedule_error = torch.zeros(
            self.num_envs, device=self.device
        )
        self._segment_gait_sync_error = torch.zeros(self.num_envs, device=self.device)
        self._segment_gait_mixed_contact = torch.zeros(
            self.num_envs, device=self.device
        )
        self._segment_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.metrics["grid_error_vel_xy"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["grid_error_vel_yaw"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["grid_success"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["grid_gait_schedule_error"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["grid_gait_sync_error"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["grid_gait_mixed_contact"] = torch.zeros(
            self.num_envs, device=self.device
        )

    def _initial_cell(self) -> tuple[int, int]:
        """获取并检查初始 cell，默认使用网格中心。"""
        cell = self._grid_cfg.initial_cell
        if cell is None:
            cell = (
                self._grid_cfg.grid_num_x // 2,
                self._grid_cfg.grid_num_yaw // 2,
            )
        if not (
            0 <= cell[0] < self._grid_cfg.grid_num_x
            and 0 <= cell[1] < self._grid_cfg.grid_num_yaw
        ):
            raise ValueError(
                f"initial_cell={cell} is outside the grid "
                f"({self._grid_cfg.grid_num_x}, {self._grid_cfg.grid_num_yaw})."
            )
        return cell

    def _grid_edges(
        self, value_range: tuple[float, float], num_bins: int
    ) -> torch.Tensor:
        """根据配置范围生成网格边界。"""
        if value_range[1] < value_range[0]:
            raise ValueError(f"Velocity range must be increasing, got {value_range}.")
        # ``low == high`` 表示该维度被刻意锁死，例如 Pronking 单元阶段固定
        # yaw=0。linspace 会返回重复边界，随后区间采样自然得到唯一常量。
        return torch.linspace(
            value_range[0], value_range[1], num_bins + 1, device=self.device
        )

    def _sample_cells(self, env_ids: torch.Tensor) -> None:
        """从激活 cell 中均匀采样指定环境的 cell。"""
        active_ids = torch.nonzero(
            self.active_cells.flatten(), as_tuple=False
        ).flatten()
        if len(active_ids) == 0:
            raise RuntimeError("Grid Adaptive Curriculum has no active cells.")

        # torch.randint 对 active_ids 等概率取样：P(cell)=1/N_active。
        selected = active_ids[
            torch.randint(len(active_ids), (len(env_ids),), device=self.device)
        ]
        self.current_cell[env_ids, 0] = selected // self._grid_cfg.grid_num_yaw
        self.current_cell[env_ids, 1] = selected % self._grid_cfg.grid_num_yaw

    def _sample_uniform(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        """为每个环境在各自的区间内均匀采样。"""
        return low + (high - low) * torch.rand(len(low), device=self.device)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        # 命令计时器到期和环境 reset 都会走这里。先结算旧命令段，再抽取新 cell。
        self._settle_segments(env_ids)
        self._sample_cells(env_ids)

        x_edges = self._grid_edges(
            self._grid_cfg.ranges.lin_vel_x, self._grid_cfg.grid_num_x
        )
        yaw_edges = self._grid_edges(
            self._grid_cfg.ranges.ang_vel_z, self._grid_cfg.grid_num_yaw
        )
        x_index = self.current_cell[env_ids, 0]
        yaw_index = self.current_cell[env_ids, 1]

        x_low, x_high = x_edges[x_index], x_edges[x_index + 1]
        yaw_low, yaw_high = yaw_edges[yaw_index], yaw_edges[yaw_index + 1]
        self.vel_command_b[env_ids, 0] = self._sample_uniform(x_low, x_high)
        self.vel_command_b[env_ids, 1] = self._sample_uniform(
            torch.full_like(x_low, self._grid_cfg.ranges.lin_vel_y[0]),
            torch.full_like(x_low, self._grid_cfg.ranges.lin_vel_y[1]),
        )
        self.vel_command_b[env_ids, 2] = self._sample_uniform(yaw_low, yaw_high)

        # 保留普通速度命令的 standing/world/forward 逻辑；当前仅启用 standing。
        r = torch.rand(len(env_ids), device=self.device)
        self.is_standing_env[env_ids] = r <= self.cfg.rel_standing_envs
        self.is_world_env[env_ids] = r <= self.cfg.rel_world_envs
        self.vel_command_w[env_ids] = self.vel_command_b[env_ids]
        self.is_forward_env[env_ids] = r <= self.cfg.rel_forward_envs

        # standing 命令是独立的零速度任务，不应被当作当前速度 cell 的样本。
        # _update_command 会在每一步将这些环境的实际命令置零。
        self.metrics["grid_success"][env_ids] = 0.0

        forward_ids = env_ids[self.is_forward_env[env_ids]]
        if len(forward_ids) > 0:
            self.vel_command_b[forward_ids, 0] = (
                self.vel_command_b[forward_ids, 0].abs().clamp(min=0.3)
            )
            self.vel_command_b[forward_ids, 1:] = 0.0

        self._segment_error_xy[env_ids] = 0.0
        self._segment_error_yaw[env_ids] = 0.0
        self._segment_gait_schedule_error[env_ids] = 0.0
        self._segment_gait_sync_error[env_ids] = 0.0
        self._segment_gait_mixed_contact[env_ids] = 0.0
        self._segment_steps[env_ids] = 0

    def _gait_quality_errors(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """计算 WTW 接触质量，避免课程与 reward 使用两套成功定义。

        此处只通过命令/传感器的公开属性访问 WTW，不导入 WTW 类，因而 Grid
        Adaptive 仍可独立用于没有 WTW 的 baseline。返回的三个量依次为：逐脚
        时序 L1 误差、四脚接触分歧、在高置信支撑/摆动区间的混合接触指示。
        """
        if self._grid_cfg.gait_quality_behavior_command_name is None:
            return None
        try:
            behavior = self._env.command_manager.get_term(
                self._grid_cfg.gait_quality_behavior_command_name
            )
            sensor = self._env.scene[self._grid_cfg.gait_quality_sensor_name]
            phase = behavior.phase
            duty_factor = float(getattr(behavior, "duty_factor", 0.5))
            found = sensor.data.found
        except (AttributeError, KeyError, TypeError):
            return None
        if phase is None or found is None:
            return None

        # 与 wtw_rewards._smooth_contact_target 使用相同的周期高斯 CDF 公式。
        sigma = max(float(self._grid_cfg.gait_contact_smoothing), 1.0e-3)
        duty = min(max(duty_factor, 1.0e-3), 1.0 - 1.0e-3)
        root_two = math.sqrt(2.0)

        def normal_cdf(value: torch.Tensor) -> torch.Tensor:
            return 0.5 * (1.0 + torch.erf(value / (sigma * root_two)))

        desired_contact = (
            normal_cdf(phase) * (1.0 - normal_cdf(phase - duty))
            + normal_cdf(phase - 1.0) * (1.0 - normal_cdf(phase - duty - 1.0))
        ).clamp(0.0, 1.0)
        actual_contact = (found > 0).to(dtype=desired_contact.dtype)
        schedule_error = torch.abs(actual_contact - desired_contact).mean(dim=1)

        # 非同步 gait 本来就不应让四脚接触相同，因此只在四腿期望接触几乎
        # 相同的 Pronking 类时计算同步门槛；其他 gait 的这两个量为零。
        synchronous = desired_contact.amax(dim=1) - desired_contact.amin(dim=1) < 1.0e-3
        sync_error = (
            torch.abs(actual_contact[:, 1:] - actual_contact[:, :1]).mean(dim=1)
            * synchronous
        )
        high_confidence = (desired_contact.mean(dim=1) > 0.9) | (
            desired_contact.mean(dim=1) < 0.1
        )
        mixed_contact = (
            1.0 - actual_contact.prod(dim=1) - (1.0 - actual_contact).prod(dim=1)
        ) * (synchronous & high_confidence)
        return schedule_error, sync_error, mixed_contact

    def _update_metrics(self) -> None:
        # 保留 mjlab 原有的 error_vel_xy/error_vel_yaw metrics。
        super()._update_metrics()
        error_xy = torch.norm(
            self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2],
            dim=-1,
        )
        error_yaw = torch.abs(
            self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2]
        )
        self._segment_error_xy += error_xy
        self._segment_error_yaw += error_yaw
        gait_errors = self._gait_quality_errors()
        if gait_errors is not None:
            schedule_error, sync_error, mixed_contact = gait_errors
            self._segment_gait_schedule_error += schedule_error
            self._segment_gait_sync_error += sync_error
            self._segment_gait_mixed_contact += mixed_contact
        self._segment_steps += 1
        steps = self._segment_steps.clamp_min(1).float()
        self.metrics["grid_error_vel_xy"] = self._segment_error_xy / steps
        self.metrics["grid_error_vel_yaw"] = self._segment_error_yaw / steps
        self.metrics["grid_gait_schedule_error"] = (
            self._segment_gait_schedule_error / steps
        )
        self.metrics["grid_gait_sync_error"] = self._segment_gait_sync_error / steps
        self.metrics["grid_gait_mixed_contact"] = (
            self._segment_gait_mixed_contact / steps
        )

    def _settle_segments(self, env_ids: torch.Tensor) -> None:
        """结算命令段并根据成功率更新课程地图。"""
        if len(env_ids) == 0:
            return
        # standing 任务不参与速度网格难度评估，否则大量零速度成功会虚高
        # 非零速度 cell 的成功率，造成 curriculum 过早扩展。
        valid = (self._segment_steps[env_ids] > 0) & (~self.is_standing_env[env_ids])
        # 没有可用于网格统计的 segment 时，清理当前段状态后直接返回。
        if not valid.any():
            self._segment_error_xy[env_ids] = 0.0
            self._segment_error_yaw[env_ids] = 0.0
            self._segment_gait_schedule_error[env_ids] = 0.0
            self._segment_gait_sync_error[env_ids] = 0.0
            self._segment_gait_mixed_contact[env_ids] = 0.0
            self._segment_steps[env_ids] = 0
            return

        settled_ids = env_ids[valid]
        steps = self._segment_steps[settled_ids].float()
        mean_xy = self._segment_error_xy[settled_ids] / steps
        mean_yaw = self._segment_error_yaw[settled_ids] / steps
        mean_schedule_error = self._segment_gait_schedule_error[settled_ids] / steps
        mean_sync_error = self._segment_gait_sync_error[settled_ids] / steps
        mean_mixed_contact = self._segment_gait_mixed_contact[settled_ids] / steps

        # 提前摔倒算失败；正常 time_out 不直接算失败，由跟踪误差判定。
        terminated = getattr(self._env, "reset_terminated", None)
        if terminated is None:
            terminated = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        success = (
            ~terminated[settled_ids]
            & (mean_xy <= self._grid_cfg.velocity_error_threshold)
            & (mean_yaw <= self._grid_cfg.yaw_error_threshold)
        )
        # 可选的 gait-aware gate：只要配置阈值，课程就不会把“速度能跟上、
        # 但仍是错峰 trot”的命令段判成 Pronking 成功。
        if self._grid_cfg.gait_schedule_error_threshold is not None:
            success &= (
                mean_schedule_error <= self._grid_cfg.gait_schedule_error_threshold
            )
        if self._grid_cfg.gait_sync_error_threshold is not None:
            success &= mean_sync_error <= self._grid_cfg.gait_sync_error_threshold
        if self._grid_cfg.gait_mixed_contact_threshold is not None:
            success &= mean_mixed_contact <= self._grid_cfg.gait_mixed_contact_threshold
        self.metrics["grid_success"][settled_ids] = success.float()

        cell_x = self.current_cell[settled_ids, 0]
        cell_yaw = self.current_cell[settled_ids, 1]
        ones = torch.ones_like(cell_x, dtype=torch.long)
        self.cell_visits.index_put_((cell_x, cell_yaw), ones, accumulate=True)
        self.cell_successes.index_put_(
            (cell_x, cell_yaw), success.long(), accumulate=True
        )
        self._record_recent_results(cell_x, cell_yaw, success)
        self._expand_ready_cells()

        # 防止同一个 segment 在 reset 流程中被重复统计。
        self._segment_error_xy[env_ids] = 0.0
        self._segment_error_yaw[env_ids] = 0.0
        self._segment_gait_schedule_error[env_ids] = 0.0
        self._segment_gait_sync_error[env_ids] = 0.0
        self._segment_gait_mixed_contact[env_ids] = 0.0
        self._segment_steps[env_ids] = 0

    def _expand_ready_cells(self) -> None:
        """根据近期表现，有限数量地激活四连通邻居。"""
        current_step = int(self._env.common_step_counter)
        if self._last_expansion_step == current_step:
            return
        self._last_expansion_step = current_step

        recent_visits = self.cell_recent_visits.reshape_as(self.cell_visits)
        recent_successes = self.cell_recent_successes.reshape_as(self.cell_visits)
        recent_success_rate = recent_successes.float() / recent_visits.clamp_min(1.0)
        ready = (
            self.active_cells
            & (self.cell_visits >= self._grid_cfg.min_cell_visits)
            & (recent_visits >= self._grid_cfg.min_cell_visits)
            & (recent_success_rate >= self._grid_cfg.success_rate_threshold)
        )

        # 严格 frontier 模式用于逐 cell 课程：新激活的邻居在完成自己的
        # 访问量和成功率验证前，会阻止已经成熟的旧 cell 继续向外扩张。
        # 否则在大规模并行环境中，中心 cell 往往会在相邻两次 reset 中迅速
        # 打开多个方向，导致“新速度格尚未验证，课程已全部展开”。
        if (
            self._grid_cfg.require_all_active_cells_ready
            and (self.active_cells & ~ready).any()
        ):
            return

        # 使用确定性顺序选取候选 cell，保证相同 checkpoint 能复现实验轨迹。
        candidate_ids: set[tuple[int, int]] = set()
        ready_ids = torch.nonzero(ready, as_tuple=False).tolist()
        for x_index, yaw_index in ready_ids:
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                next_x = x_index + dx
                next_yaw = yaw_index + dy
                if (
                    0 <= next_x < self._grid_cfg.grid_num_x
                    and 0 <= next_yaw < self._grid_cfg.grid_num_yaw
                    and not self.active_cells[next_x, next_yaw]
                ):
                    candidate_ids.add((next_x, next_yaw))

        selected_ids = sorted(candidate_ids)[: self._grid_cfg.max_new_cells_per_update]
        for x_index, yaw_index in selected_ids:
            self.active_cells[x_index, yaw_index] = True

    def _record_recent_results(
        self,
        cell_x: torch.Tensor,
        cell_yaw: torch.Tensor,
        success: torch.Tensor,
    ) -> None:
        """将本批次结果写入各 cell 独立的近期成功率 ring buffer。"""
        flat_cell_ids = cell_x * self._grid_cfg.grid_num_yaw + cell_yaw
        window_size = self._grid_cfg.success_window_size

        # 同一批次内可能有多个环境属于同一个 cell。先按 cell 分组，再一次性
        # 写入该组结果，避免直接 scatter 时多个环境争用同一个 ring 指针。
        for cell_id in torch.unique(flat_cell_ids).tolist():
            group_mask = flat_cell_ids == cell_id
            group_success = success[group_mask].bool()
            num_results = len(group_success)
            ptr = int(self.cell_history_ptr[cell_id].item())
            previous_visits = int(self.cell_recent_visits[cell_id].item())

            if num_results >= window_size:
                # 本批次已经覆盖整个窗口，最终状态只由最后 window_size 个结果决定。
                final_results = group_success[-window_size:]
                start = (ptr + num_results - window_size) % window_size
                positions = (
                    torch.arange(window_size, device=self.device) + start
                ) % window_size
                self.cell_success_history[cell_id, positions] = final_results
                self.cell_recent_successes[cell_id] = final_results.long().sum()
                self.cell_recent_visits[cell_id] = window_size
            else:
                positions = (
                    torch.arange(num_results, device=self.device) + ptr
                ) % window_size
                overwritten = max(0, previous_visits + num_results - window_size)
                removed_successes = 0
                if overwritten > 0:
                    removed_successes = int(
                        self.cell_success_history[cell_id, positions[-overwritten:]]
                        .long()
                        .sum()
                        .item()
                    )
                self.cell_success_history[cell_id, positions] = group_success
                self.cell_recent_successes[cell_id] = (
                    self.cell_recent_successes[cell_id]
                    + group_success.long().sum()
                    - removed_successes
                )
                self.cell_recent_visits[cell_id] = min(
                    window_size, previous_visits + num_results
                )

            self.cell_history_ptr[cell_id] = (ptr + num_results) % window_size

    def curriculum_state_dict(self) -> dict[str, Any]:
        """返回可写入 checkpoint 的课程状态。"""
        state: dict[str, Any] = {
            "version": 3,
            "active_cells": self.active_cells.detach().cpu().clone(),
            "cell_visits": self.cell_visits.detach().cpu().clone(),
            "cell_successes": self.cell_successes.detach().cpu().clone(),
            "cell_success_history": self.cell_success_history.detach().cpu().clone(),
            "cell_recent_visits": self.cell_recent_visits.detach().cpu().clone(),
            "cell_recent_successes": self.cell_recent_successes.detach().cpu().clone(),
            "cell_history_ptr": self.cell_history_ptr.detach().cpu().clone(),
            "current_cell": self.current_cell.detach().cpu().clone(),
            "vel_command_b": self.vel_command_b.detach().cpu().clone(),
            "vel_command_w": self.vel_command_w.detach().cpu().clone(),
            "time_left": self.time_left.detach().cpu().clone(),
            "command_counter": self.command_counter.detach().cpu().clone(),
            "is_standing_env": self.is_standing_env.detach().cpu().clone(),
            "is_world_env": self.is_world_env.detach().cpu().clone(),
            "is_forward_env": self.is_forward_env.detach().cpu().clone(),
            "segment_error_xy": self._segment_error_xy.detach().cpu().clone(),
            "segment_error_yaw": self._segment_error_yaw.detach().cpu().clone(),
            "segment_gait_schedule_error": self._segment_gait_schedule_error.detach()
            .cpu()
            .clone(),
            "segment_gait_sync_error": self._segment_gait_sync_error.detach()
            .cpu()
            .clone(),
            "segment_gait_mixed_contact": self._segment_gait_mixed_contact.detach()
            .cpu()
            .clone(),
            "segment_steps": self._segment_steps.detach().cpu().clone(),
            "last_expansion_step": self._last_expansion_step,
        }
        return state

    def load_curriculum_state_dict(self, state: dict[str, Any]) -> None:
        """从 checkpoint 恢复课程状态；兼容没有近期窗口的旧状态。"""
        # checkpoint 中的字段名保持简洁，但命令段累计量在类中使用下划线表示
        # 内部状态，因此这里显式建立字段名到成员变量的映射。
        # 网格统计是课程的核心状态，形状不一致通常意味着配置发生了变化。
        grid_targets = {
            "active_cells": self.active_cells,
            "cell_visits": self.cell_visits,
            "cell_successes": self.cell_successes,
            "cell_success_history": self.cell_success_history,
            "cell_recent_visits": self.cell_recent_visits,
            "cell_recent_successes": self.cell_recent_successes,
            "cell_history_ptr": self.cell_history_ptr,
        }
        # 逐环境状态在训练和 play 之间可能有不同形状（例如 4096 -> 1），
        # 这种情况下跳过恢复，下一次 reset 会重新生成当前环境的运行状态。
        runtime_targets = {
            "current_cell": self.current_cell,
            "vel_command_b": self.vel_command_b,
            "vel_command_w": self.vel_command_w,
            "time_left": self.time_left,
            "command_counter": self.command_counter,
            "is_standing_env": self.is_standing_env,
            "is_world_env": self.is_world_env,
            "is_forward_env": self.is_forward_env,
            "segment_error_xy": self._segment_error_xy,
            "segment_error_yaw": self._segment_error_yaw,
            "segment_gait_schedule_error": self._segment_gait_schedule_error,
            "segment_gait_sync_error": self._segment_gait_sync_error,
            "segment_gait_mixed_contact": self._segment_gait_mixed_contact,
            "segment_steps": self._segment_steps,
        }

        for name, target in grid_targets.items():
            value = state.get(name)
            if value is None:
                continue
            if not isinstance(value, torch.Tensor) or value.shape != target.shape:
                raise ValueError(
                    f"Invalid Grid Adaptive state for '{name}': "
                    f"expected tensor shape {tuple(target.shape)}."
                )
            target.copy_(value.to(device=self.device, dtype=target.dtype))

        for name, target in runtime_targets.items():
            value = state.get(name)
            if value is None:
                continue
            if not isinstance(value, torch.Tensor) or value.shape != target.shape:
                continue
            target.copy_(value.to(device=self.device, dtype=target.dtype))

        last_expansion_step = state.get("last_expansion_step")
        if last_expansion_step is not None:
            self._last_expansion_step = int(last_expansion_step)

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        # 先结算旧段，再调用父类清理 metrics 并采样新命令。
        assert isinstance(env_ids, torch.Tensor)
        self._settle_segments(env_ids)
        extras = super().reset(env_ids)
        self._segment_error_xy[env_ids] = 0.0
        self._segment_error_yaw[env_ids] = 0.0
        self._segment_gait_schedule_error[env_ids] = 0.0
        self._segment_gait_sync_error[env_ids] = 0.0
        self._segment_gait_mixed_contact[env_ids] = 0.0
        self._segment_steps[env_ids] = 0
        extras["grid_active_cells"] = float(self.active_cells.sum().item())
        extras["grid_total_visits"] = float(self.cell_visits.sum().item())
        return extras
