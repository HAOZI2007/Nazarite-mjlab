"""Overlay managers for Viser viewer orchestration.

These managers intentionally coordinate *when* higher-level updates happen
(env switches, paused/running updates, etc.) while leaving low-level render
handle lifecycle ownership inside :mod:`scene.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import mujoco
import numpy as np
import viser

from mjlab.sensor import CameraSensor
from mjlab.viewer.viser.camera_viewer import ViserCameraViewer
from mjlab.viewer.viser.reward_bar_panel import RewardBarPanel
from mjlab.viewer.viser.term_plotter import ViserTermPlotter


class _EnvProtocol(Protocol):
  @property
  def unwrapped(self) -> Any: ...


class _SceneProtocol(Protocol):
  env_idx: int
  debug_visualization_enabled: bool
  needs_update: bool

  @property
  def show_contact_points(self) -> bool: ...
  @property
  def show_contact_forces(self) -> bool: ...

  def clear_debug_all(self) -> None: ...
  def clear(self) -> None: ...


@dataclass
class ViserTermOverlays:
  """Manage reward/metrics term plot tabs for Viser viewer."""

  server: viser.ViserServer
  env: _EnvProtocol
  scene: _SceneProtocol
  frame_time: float
  reward_bar_max_terms: int = 20
  reward_plotter: ViserTermPlotter | None = None
  reward_bar_panel: RewardBarPanel | None = None
  metrics_plotter: ViserTermPlotter | None = None

  def setup_tabs(self, tabs: Any) -> None:
    """Create rewards/metrics tabs based on available managers."""
    if hasattr(self.env.unwrapped, "reward_manager"):
      with tabs.add_tab("Rewards", icon=viser.Icon.CHART_LINE):
        term_names = [
          name
          for name, _ in self.env.unwrapped.reward_manager.get_active_iterable_terms(
            self.scene.env_idx
          )
        ]
        # Live bar panel (running-mean comparison).
        self.reward_bar_panel = RewardBarPanel(
          self.server,
          term_names,
          update_dt=self.frame_time,
          max_terms=self.reward_bar_max_terms,
        )
        self.reward_plotter = ViserTermPlotter(
          self.server, term_names, name="Reward", env_idx=self.scene.env_idx
        )

    if hasattr(self.env.unwrapped, "metrics_manager"):
      term_names = [
        name
        for name, _ in self.env.unwrapped.metrics_manager.get_active_iterable_terms(
          self.scene.env_idx
        )
      ]
      if term_names:
        with tabs.add_tab("Metrics", icon=viser.Icon.CHART_BAR):
          self.metrics_plotter = ViserTermPlotter(
            self.server, term_names, name="Metric", env_idx=self.scene.env_idx
          )

  def on_env_switch(self) -> None:
    """Clear histories when active environment changes."""
    env_idx = self.scene.env_idx
    if self.reward_plotter:
      self.reward_plotter.clear_histories()
      self.reward_plotter.update_env_idx(env_idx)
    if self.reward_bar_panel:
      self.reward_bar_panel.clear_histories()
    if self.metrics_plotter:
      self.metrics_plotter.clear_histories()
      self.metrics_plotter.update_env_idx(env_idx)

  def update(self, paused: bool) -> None:
    """Update term plots from the selected environment."""
    if (
      self.reward_plotter is not None or self.reward_bar_panel is not None
    ) and not paused:
      terms = list(
        self.env.unwrapped.reward_manager.get_active_iterable_terms(self.scene.env_idx)
      )
      if self.reward_plotter is not None:
        self.reward_plotter.update(terms)
      if self.reward_bar_panel is not None:
        self.reward_bar_panel.update(terms)

    if self.metrics_plotter is not None and not paused:
      terms = list(
        self.env.unwrapped.metrics_manager.get_active_iterable_terms(self.scene.env_idx)
      )
      self.metrics_plotter.update(terms)

  def clear_histories(self) -> None:
    """Clear all overlay histories."""
    self.on_env_switch()

  def cleanup(self) -> None:
    """Cleanup plotter resources."""
    if self.reward_plotter:
      self.reward_plotter.cleanup()
    if self.reward_bar_panel:
      self.reward_bar_panel.cleanup()
    if self.metrics_plotter:
      self.metrics_plotter.cleanup()


@dataclass
class ViserCameraOverlays:
  """Manage camera feed widgets and updates for Viser viewer."""

  server: viser.ViserServer
  env: _EnvProtocol
  mj_model: mujoco.MjModel
  camera_viewers: list[ViserCameraViewer] | None = None
  delta_map_viewer: "ViserDeltaMapViewer | None" = None

  @property
  def has_cameras(self) -> bool:
    """Whether the environment has any camera sensors."""
    return any(
      isinstance(s, CameraSensor) for s in self.env.unwrapped.scene.sensors.values()
    )

  def setup_controls(self) -> None:
    """Create camera feed controls under the active GUI folder."""
    camera_sensors = [
      sensor
      for sensor in self.env.unwrapped.scene.sensors.values()
      if isinstance(sensor, CameraSensor)
    ]
    if not camera_sensors:
      self.camera_viewers = []
      return

    self.camera_viewers = [
      ViserCameraViewer(self.server, sensor, self.mj_model) for sensor in camera_sensors
    ]
    if getattr(self.env.unwrapped, "delta_visual_debug_enabled", False):
      self.delta_map_viewer = ViserDeltaMapViewer(self.server)

  def update(self, sim_data: Any, env_idx: int, scene_offset: Any) -> None:
    """Push latest camera images/frustums to GUI."""
    if not self.camera_viewers:
      return
    for camera_viewer in self.camera_viewers:
      camera_viewer.update(sim_data, env_idx, scene_offset)
    if self.delta_map_viewer is not None:
      self.delta_map_viewer.update(self.env.unwrapped, env_idx)

  def cleanup(self) -> None:
    """Cleanup all camera feed widgets."""
    if not self.camera_viewers:
      return
    for camera_viewer in self.camera_viewers:
      camera_viewer.cleanup()
    if self.delta_map_viewer is not None:
      self.delta_map_viewer.cleanup()


class ViserDeltaMapViewer:
  """Synchronized raw-depth/BEV diagnostic images for DELTA play."""

  def __init__(self, server: viser.ViserServer, display_size: int = 160):
    self._handles: list[viser.GuiImageHandle] = []
    self._labels = (
      "DELTA raw depth (0-5m)",
      "DELTA BEV z (x right, y down)",
      "DELTA observed support (x right, y down)",
      "DELTA confidence (x right, y down)",
    )
    self._display_size = display_size
    for label in self._labels:
      self._handles.append(
        server.gui.add_image(
          np.zeros((display_size, display_size, 3), dtype=np.uint8),
          label=label,
          format="jpeg",
        )
      )

  @staticmethod
  def _resize(
    image: np.ndarray, size: int, *, border: tuple[int, int, int] = (80, 80, 80)
  ) -> np.ndarray:
    """Nearest-neighbor resize with letterboxing and orientation markers."""
    if image.ndim == 2:
      image = image[..., None]
    if image.shape[-1] == 1:
      image = np.repeat(image, 3, axis=-1)
    height, width = image.shape[:2]
    scale = max(1, min(size // max(height, 1), size // max(width, 1)))
    resized = np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)
    canvas = np.empty((size, size, 3), dtype=np.uint8)
    canvas[...] = np.asarray(border, dtype=np.uint8)
    offset_y = (size - resized.shape[0]) // 2
    offset_x = (size - resized.shape[1]) // 2
    canvas[
      offset_y : offset_y + resized.shape[0], offset_x : offset_x + resized.shape[1]
    ] = resized
    # x increases to the right; y decreases down the image because the first
    # BEV row is +y. The border and centerline make that convention visible.
    canvas[offset_y, offset_x : offset_x + resized.shape[1]] = (255, 80, 40)
    canvas[offset_y : offset_y + resized.shape[0], offset_x] = (40, 180, 255)
    if resized.shape[0] > 2 and resized.shape[1] > 2:
      canvas[
        offset_y + resized.shape[0] // 2, offset_x : offset_x + resized.shape[1]
      ] = (
        canvas[offset_y + resized.shape[0] // 2, offset_x : offset_x + resized.shape[1]]
        * 0.75
      ).astype(np.uint8)
    return canvas

  @staticmethod
  def _gray(value: np.ndarray, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    normalized = np.clip((value - low) / max(high - low, 1.0e-6), 0.0, 1.0)
    channel = (normalized * 255.0).astype(np.uint8)
    return np.repeat(channel[..., None], 3, axis=-1)

  @staticmethod
  def _heat(value: np.ndarray, low: float, high: float) -> np.ndarray:
    normalized = np.clip((value - low) / max(high - low, 1.0e-6), 0.0, 1.0)
    # Blue -> cyan -> yellow -> red, without requiring matplotlib in the viewer.
    red = np.clip(2.0 * normalized, 0.0, 1.0)
    green = np.clip(2.0 * (1.0 - np.abs(normalized - 0.5)), 0.0, 1.0)
    blue = np.clip(2.0 * (1.0 - normalized), 0.0, 1.0)
    return (np.stack((red, green, blue), axis=-1) * 255.0).astype(np.uint8)

  def update(self, env: Any, env_idx: int) -> None:
    snapshot = getattr(env, "_delta_visual_debug", None)
    if snapshot is None:
      return
    try:
      depth = snapshot["depth"][env_idx].detach().cpu().numpy()
      bev_z = snapshot["bev_z"][env_idx].detach().cpu().numpy()
      support = snapshot["support"][env_idx].detach().cpu().numpy()
      confidence = snapshot["confidence"][env_idx].detach().cpu().numpy()
    except (KeyError, IndexError, RuntimeError):
      return
    self._handles[0].image = self._resize(
      self._gray(depth, 0.0, 5.0), self._display_size, border=(20, 20, 20)
    )
    self._handles[1].image = self._resize(
      self._heat(bev_z, -0.8, 0.8), self._display_size
    )
    self._handles[2].image = self._resize(self._gray(support), self._display_size)
    self._handles[3].image = self._resize(
      self._heat(confidence, 0.0, 1.0), self._display_size
    )

  def cleanup(self) -> None:
    for handle in self._handles:
      handle.remove()


@dataclass
class ViserDebugOverlays:
  """Manage debug visualization queueing and env-switch behavior."""

  env: _EnvProtocol
  scene: _SceneProtocol

  def on_env_switch(self) -> None:
    """Reset debug visuals when switching selected environment."""
    if self.scene.debug_visualization_enabled:
      self.scene.clear_debug_all()

  def queue(self) -> None:
    """Queue environment debug visualizers for the current frame."""
    if self.scene.debug_visualization_enabled and hasattr(
      self.env.unwrapped, "update_visualizers"
    ):
      self.scene.clear()  # Clear queued arrows from previous frame.
      self.env.unwrapped.update_visualizers(self.scene)


@dataclass
class ViserContactOverlays:
  """Manage contact-visualization orchestration from the viewer layer.

  Note: contact mesh creation/update/removal stays in ``ViserMujocoScene``.
  This manager only requests scene refreshes at the right times.
  """

  scene: _SceneProtocol

  def is_enabled(self) -> bool:
    """Whether any contact visualization is currently enabled."""
    return self.scene.show_contact_points or self.scene.show_contact_forces

  def on_env_switch(self) -> None:
    """Request a scene refresh when switching environments with contacts enabled."""
    if self.is_enabled():
      self.scene.needs_update = True
