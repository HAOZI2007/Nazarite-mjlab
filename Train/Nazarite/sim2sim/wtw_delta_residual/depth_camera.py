"""D435-like camera and BEV diagnostics for sim2sim.

The camera in this module is deliberately a diagnostic sensor.  It is rendered
after the WTW action has been selected and its output is never concatenated
into the WTW observation.  This makes it safe to use while validating camera
mounting and the depth-to-BEV geometry before enabling visual control.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np


def _quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def _axis_angle(axis: tuple[float, float, float], degrees: float) -> np.ndarray:
    vector = np.asarray(axis, dtype=np.float64)
    vector /= np.linalg.norm(vector)
    half_angle = np.deg2rad(degrees) * 0.5
    return np.concatenate(
        ([np.cos(half_angle)], np.sin(half_angle) * vector)
    )


@dataclass(frozen=True)
class DepthCameraConfig:
    """Mounting, stream and calibrated pinhole parameters.

    Position is in metres in the Go2 base frame.  The zero-pitch/zero-roll
    orientation points the optical axis along the robot +X direction.  The
    default stream is a compact D435 128x72 resize; change all four intrinsics
    together when using a different stream or load them from calibration.
    """

    name: str = "sim2sim_delta_depth_camera"
    width: int = 64
    height: int = 36
    pos_b: tuple[float, float, float] = (0.30, 0.0, 0.12)
    pitch_deg: float = 20.0
    roll_deg: float = 0.0
    # D435 848x480 calibration scaled to the default 64x36 stream.
    fx_px: float = 31.738
    fy_px: float = 31.540
    cx_px: float = 32.354
    cy_px: float = 17.852
    max_depth_m: float = 5.0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Depth camera resolution must be positive")
        if any(value <= 0.0 for value in (self.fx_px, self.fy_px, self.max_depth_m)):
            raise ValueError("Camera focal lengths and max depth must be positive")
        if len(self.pos_b) != 3:
            raise ValueError("Camera position must have three components")

    @property
    def quat_b(self) -> tuple[float, float, float, float]:
        """Return MuJoCo [w, x, y, z] camera orientation.

        MuJoCo cameras look along local -Z.  The base orientation maps -Z to
        body +X, +X to body -Y and +Y to body +Z.  Pitch is applied about body
        Y, then roll about the resulting optical axis.
        """

        base = np.array((-0.5, -0.5, 0.5, 0.5), dtype=np.float64)
        pitched = _quat_mul(_axis_angle((0.0, 1.0, 0.0), self.pitch_deg), base)
        # The camera optical axis is the body-frame direction of local -Z.
        rotation = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotation, pitched)
        optical_axis = -(rotation.reshape(3, 3)[:, 2])
        rolled = _quat_mul(
            _axis_angle(tuple(optical_axis), self.roll_deg), pitched
        )
        rolled /= np.linalg.norm(rolled)
        return (
            float(rolled[0]),
            float(rolled[1]),
            float(rolled[2]),
            float(rolled[3]),
        )

    def install(self, spec: mujoco.MjSpec) -> None:
        """Attach this camera to ``robot/base_link`` in an MjSpec."""
        body = spec.body("base_link")
        if body is None:
            raise RuntimeError("Go2 XML does not contain base_link")
        camera = body.add_camera()
        camera.name = self.name
        camera.pos[:] = self.pos_b
        camera.quat[:] = self.quat_b
        camera.resolution[:] = [self.width, self.height]
        # MuJoCo's focal_pixel is in pixels, while principal_pixel is an offset
        # from the image centre (not the absolute cx/cy pixel coordinate).
        # Supplying these values preserves the calibrated pinhole rays instead
        # of relying on an approximate fovy-only camera.
        camera.sensor_size[:] = [self.width, self.height]
        camera.focal_pixel[:] = [self.fx_px, self.fy_px]
        camera.principal_pixel[:] = [
            self.cx_px - 0.5 * self.width,
            self.cy_px - 0.5 * self.height,
        ]


@dataclass(frozen=True)
class BevConfig:
    """Robot-frame BEV grid configuration."""

    height: int = 16
    width: int = 26
    x_range_m: tuple[float, float] = (0.0, 2.5)
    y_range_m: tuple[float, float] = (-0.8, 0.8)
    z_scale_m: float = 0.8
    fill_kernel_size: int = 5

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0:
            raise ValueError("BEV height and width must be positive")
        if not self.x_range_m[1] > self.x_range_m[0]:
            raise ValueError("BEV x_range_m must be increasing")
        if not self.y_range_m[1] > self.y_range_m[0]:
            raise ValueError("BEV y_range_m must be increasing")
        if self.z_scale_m <= 0.0:
            raise ValueError("BEV z_scale_m must be positive")
        if self.fill_kernel_size < 1 or self.fill_kernel_size % 2 == 0:
            raise ValueError("BEV fill_kernel_size must be a positive odd integer")


@dataclass(frozen=True)
class DepthObservation:
    depth_m: np.ndarray
    bev_z: np.ndarray
    support: np.ndarray
    confidence: np.ndarray


def depth_to_bev(
    depth_m: np.ndarray,
    camera: DepthCameraConfig,
    bev: BevConfig,
) -> DepthObservation:
    """Project a depth image into the same x/y/z/support/confidence layout.

    The equations mirror ``nazarite.mdp.delta_observations.delta_depth_image``
    so the visual diagnostic is representative of the training projection.
    """
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.shape != (camera.height, camera.width):
        raise ValueError(
            f"Depth image must be {(camera.height, camera.width)}, got {depth.shape}"
        )
    v, u = np.indices(depth.shape, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 1.0e-3) & (depth < camera.max_depth_m)
    z = np.clip(np.nan_to_num(depth, nan=0.0), 0.0, camera.max_depth_m)
    x_right = (u - camera.cx_px) * z / camera.fx_px
    y_down = (v - camera.cy_px) * z / camera.fy_px

    # MuJoCo camera coordinates are +X right, +Y up and -Z forward.  Build the
    # same camera-frame ray used by the renderer, then rotate it with the exact
    # configured camera quaternion. This avoids separate roll sign conventions
    # for portrait streams and keeps every mounting angle geometrically exact.
    rotation = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, np.asarray(camera.quat_b, dtype=np.float64))
    camera_points = np.stack((x_right, -y_down, -z), axis=-1)
    body_points = camera_points @ rotation.reshape(3, 3).T
    body_x = body_points[..., 0] + camera.pos_b[0]
    body_y = body_points[..., 1] + camera.pos_b[1]
    body_z = body_points[..., 2] + camera.pos_b[2]

    x0, x1 = bev.x_range_m
    y0, y1 = bev.y_range_m
    col = np.floor((body_x - x0) / (x1 - x0) * bev.width).astype(np.int64)
    row = np.floor((y1 - body_y) / (y1 - y0) * bev.height).astype(np.int64)
    inside = (
        valid
        & (col >= 0)
        & (col < bev.width)
        & (row >= 0)
        & (row < bev.height)
    )
    bev_z = np.full((bev.height, bev.width), -1.0, dtype=np.float32)
    support = np.zeros_like(bev_z)
    for r, c, value in zip(row[inside], col[inside], body_z[inside], strict=False):
        bev_z[r, c] = max(bev_z[r, c], float(value / bev.z_scale_m))
        support[r, c] = 1.0

    # Match the training-side local hole fill.  Confidence is kept separate
    # from raw support so gaps remain distinguishable from interpolated cells.
    radius = bev.fill_kernel_size // 2
    filled = bev_z.copy()
    confidence = support.copy()
    for r in range(bev.height):
        for c in range(bev.width):
            if support[r, c] > 0.0:
                continue
            r0, r1 = max(0, r - radius), min(bev.height, r + radius + 1)
            c0, c1 = max(0, c - radius), min(bev.width, c + radius + 1)
            values = bev_z[r0:r1, c0:c1][support[r0:r1, c0:c1] > 0.0]
            if values.size:
                filled[r, c] = float(values.mean())
                confidence[r, c] = 0.5
    return DepthObservation(
        depth_m=depth.copy(),
        bev_z=filled,
        support=support,
        confidence=confidence,
    )


class DepthCameraDiagnostics:
    """Render the attached MuJoCo camera and maintain a live diagnostic plot."""

    def __init__(
        self,
        model: mujoco.MjModel,
        camera: DepthCameraConfig,
        bev: BevConfig,
        *,
        visualize: bool = True,
    ) -> None:
        self.camera = camera
        self.bev = bev
        self.model = model
        # Training-side CameraSensorCfg enables terrain group 0 while hiding
        # the robot's visual/collision groups (2/3). Reproduce that contract so
        # the diagnostic depth image is not filled by the camera carrier itself.
        self.scene_option = mujoco.MjvOption()
        self.scene_option.geomgroup[:] = 0
        self.scene_option.geomgroup[0] = 1
        self.renderer = mujoco.Renderer(model, camera.height, camera.width)
        self.renderer.enable_depth_rendering()
        self.figure: Any = None
        self.axes: Any = None
        self.images: list[Any] = []
        self.selected_panel: int | None = None
        self.native_visible = True
        self._status_text: Any = None
        self._native_viewer: Any = None
        self._native_viewport: tuple[int, int] | None = None
        if visualize:
            self._init_plot()

    def attach_native_viewer(self, viewer: Any) -> None:
        """Attach the MuJoCo viewer for in-window image overlays."""
        self._native_viewer = viewer
        viewport = viewer.viewport
        if viewport is None:
            raise RuntimeError("MuJoCo viewer did not expose a viewport")
        self._native_viewport = (int(viewport.width), int(viewport.height))
        self._update_native_text()

    def handle_native_key(self, key: int) -> bool:
        """Handle panel selection keys received from MuJoCo's GLFW callback."""
        if key in (ord("1"), ord("2"), ord("3"), ord("4")):
            self.selected_panel = key - ord("1")
        elif key in (ord("0"), 256):  # GLFW_KEY_ESCAPE is 256.
            self.selected_panel = None
        elif key in (ord("h"), ord("H")):
            self.native_visible = not self.native_visible
        else:
            return False
        self._update_native_text()
        return True

    @staticmethod
    def _resize_nearest(image: np.ndarray, height: int, width: int) -> np.ndarray:
        rows = np.linspace(0, image.shape[0] - 1, height).round().astype(np.int64)
        cols = np.linspace(0, image.shape[1] - 1, width).round().astype(np.int64)
        return image[rows[:, None], cols[None, :]]

    @staticmethod
    def _colorize(
        value: np.ndarray,
        *,
        vmin: float,
        vmax: float,
        cmap_name: str,
        invalid: np.ndarray | None = None,
    ) -> np.ndarray:
        """Convert a scalar panel to uint8 RGB without opening a plot window."""
        import matplotlib

        array = np.asarray(value, dtype=np.float32)
        normalized = np.clip((array - vmin) / max(vmax - vmin, 1.0e-6), 0.0, 1.0)
        rgb = np.asarray(matplotlib.colormaps[cmap_name](normalized)[..., :3] * 255.0, dtype=np.uint8)
        if invalid is not None:
            rgb[np.asarray(invalid, dtype=bool)] = 0
        return rgb

    def _native_panels(self, observation: DepthObservation) -> list[np.ndarray]:
        depth = observation.depth_m
        panels = [
            self._colorize(
                depth,
                vmin=0.0,
                vmax=self.camera.max_depth_m,
                cmap_name="turbo",
                invalid=depth >= self.camera.max_depth_m,
            ),
            self._colorize(observation.bev_z, vmin=-1.0, vmax=1.0, cmap_name="terrain"),
            self._colorize(observation.support, vmin=0.0, vmax=1.0, cmap_name="gray"),
            self._colorize(observation.confidence, vmin=0.0, vmax=1.0, cmap_name="viridis"),
        ]
        return panels

    def _update_native_text(self) -> None:
        if self._native_viewer is None:
            return
        if not self.native_visible:
            selected = "hidden"
        else:
            selected = "grid" if self.selected_panel is None else f"panel {self.selected_panel + 1}"
        self._native_viewer.set_texts(
            (
                mujoco.mjtFontScale.mjFONTSCALE_100,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                f"WTW+DELTA camera: {selected}",
                "1 depth | 2 BEV | 3 support | 4 confidence | 0/Esc restore | H hide/show",
            )
        )

    def update_native(self, observation: DepthObservation) -> None:
        """Draw depth/BEV images directly inside the MuJoCo viewer window."""
        if self._native_viewer is None or self._native_viewport is None:
            return
        self._update_native_text()
        if not self.native_visible:
            self._native_viewer.set_images([])
            return
        viewport_width, viewport_height = self._native_viewport
        margin = 8
        panel_width = max(1, (viewport_width - 3 * margin) // 2)
        panel_height = max(1, (viewport_height - 3 * margin) // 2)
        panels = self._native_panels(observation)
        images: list[tuple[mujoco.MjrRect, np.ndarray]] = []
        if self.selected_panel is None:
            rects = (
                (margin, margin + panel_height + margin),
                (2 * margin + panel_width, margin + panel_height + margin),
                (margin, margin),
                (2 * margin + panel_width, margin),
            )
            for panel, (left, bottom) in zip(panels, rects, strict=True):
                images.append(
                    (
                        mujoco.MjrRect(left, bottom, panel_width, panel_height),
                        self._resize_nearest(panel, panel_height, panel_width),
                    )
                )
        else:
            images.append(
                (
                    mujoco.MjrRect(
                        margin,
                        margin,
                        max(1, viewport_width - 2 * margin),
                        max(1, viewport_height - 2 * margin),
                    ),
                    self._resize_nearest(
                        panels[self.selected_panel],
                        max(1, viewport_height - 2 * margin),
                        max(1, viewport_width - 2 * margin),
                    ),
                )
            )
        self._native_viewer.set_images(images)

    def _init_plot(self) -> None:
        import matplotlib.pyplot as plt

        plt.ion()
        self.figure, self.axes = plt.subplots(2, 2, figsize=(8, 5), num="WTW + DELTA camera")
        titles = ("Raw depth (m)", "BEV elevation", "Raw support", "BEV confidence")
        for axis, title in zip(self.axes.flat, titles, strict=True):
            axis.set_title(title)
        self.images = [None, None, None, None]
        self.figure.canvas.mpl_connect("key_press_event", self._on_key)
        self.figure.canvas.mpl_connect("button_press_event", self._on_click)
        self._status_text = self.figure.text(
            0.5,
            0.015,
            "Click a panel or press 1-4 to enlarge; press 0/Esc to restore",
            ha="center",
            va="bottom",
            fontsize=9,
        )
        self._apply_layout()

    def _on_key(self, event: Any) -> None:
        if event.key in {"1", "2", "3", "4"}:
            self.selected_panel = int(event.key) - 1
        elif event.key in {"0", "escape"}:
            self.selected_panel = None
        else:
            return
        self._apply_layout()
        if self.figure is not None:
            self.figure.canvas.draw_idle()

    def _on_click(self, event: Any) -> None:
        if self.axes is None or event.inaxes is None:
            return
        for index, axis in enumerate(self.axes.flat):
            if event.inaxes is axis:
                self.selected_panel = None if self.selected_panel == index else index
                self._apply_layout()
                if self.figure is not None:
                    self.figure.canvas.draw_idle()
                return

    def _apply_layout(self) -> None:
        if self.axes is None:
            return
        positions = (
            (0.06, 0.54, 0.42, 0.38),
            (0.55, 0.54, 0.42, 0.38),
            (0.06, 0.10, 0.42, 0.38),
            (0.55, 0.10, 0.42, 0.38),
        )
        for index, axis in enumerate(self.axes.flat):
            if self.selected_panel is None:
                axis.set_visible(True)
                axis.set_position(positions[index])
            elif index == self.selected_panel:
                axis.set_visible(True)
                axis.set_position((0.07, 0.10, 0.86, 0.82))
            else:
                axis.set_visible(False)
        if self._status_text is not None:
            mode = "grid" if self.selected_panel is None else f"panel {self.selected_panel + 1}"
            self._status_text.set_text(
                f"{mode} | click a panel or press 1-4 to enlarge; press 0/Esc to restore"
            )

    def capture(self, data: mujoco.MjData) -> DepthObservation:
        self.renderer.update_scene(
            data, camera=self.camera.name, scene_option=self.scene_option
        )
        # mujoco.Renderer already converts the OpenGL depth buffer to metric
        # distance before returning it. Do not apply a second perspective
        # unprojection here (that would collapse the scene toward the camera).
        depth = np.asarray(self.renderer.render(), dtype=np.float32)
        depth = np.where(np.isfinite(depth), depth, self.camera.max_depth_m)
        depth = np.clip(depth, 0.0, self.camera.max_depth_m).astype(np.float32)
        return depth_to_bev(depth, self.camera, self.bev)

    def update_plot(self, observation: DepthObservation) -> None:
        if self.figure is None or self.axes is None:
            return
        import matplotlib.pyplot as plt

        depth = observation.depth_m.copy()
        depth[depth >= self.camera.max_depth_m] = np.nan
        values = (depth, observation.bev_z, observation.support, observation.confidence)
        for index, (axis, value) in enumerate(zip(self.axes.flat, values, strict=True)):
            if self.images[index] is None:
                if index == 0:
                    image = axis.imshow(value, cmap="turbo", vmin=0.0, vmax=self.camera.max_depth_m)
                elif index == 1:
                    image = axis.imshow(
                        value,
                        cmap="terrain",
                        vmin=-1.0,
                        vmax=1.0,
                        origin="upper",
                        extent=(self.bev.x_range_m[0], self.bev.x_range_m[1], self.bev.y_range_m[0], self.bev.y_range_m[1]),
                        aspect="auto",
                    )
                else:
                    image = axis.imshow(value, cmap="gray" if index == 2 else "viridis", vmin=0.0, vmax=1.0)
                self.images[index] = image
            else:
                self.images[index].set_data(value)
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        plt.pause(0.001)

    def close(self) -> None:
        self.renderer.close()
        if self.figure is not None:
            import matplotlib.pyplot as plt

            plt.close(self.figure)
