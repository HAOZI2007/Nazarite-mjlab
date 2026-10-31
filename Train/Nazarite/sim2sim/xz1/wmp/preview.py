"""Display the exact metric depth frames acquired by the visual policy."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


class WMPDepthPreview:
    def __init__(self, far_clip: float, on_reset: Callable[[], None]) -> None:
        # Import the GUI backend only for interactive simulation.
        import matplotlib.pyplot as plt

        self.plt = plt
        self.closed = False
        self.on_reset = on_reset
        self.figure, axis = plt.subplots(figsize=(5.5, 5.2))
        manager = self.figure.canvas.manager
        if manager is not None:
            manager.set_window_title("XZ1 WMP depth camera")
        self.image = axis.imshow(
            np.full((64, 64), far_clip, dtype=np.float32),
            cmap="gray_r",
            vmin=0.0,
            vmax=far_clip,
            origin="upper",
            interpolation="nearest",
        )
        axis.set_title("Policy depth: 64 x 64, 10 Hz")
        axis.set_axis_off()
        self.figure.colorbar(self.image, ax=axis, label="Optical-axis depth (m)")
        self.status = self.figure.text(
            0.5,
            0.025,
            "Near = white | far / missing = black | R: reset | Esc: close preview",
            ha="center",
            fontsize=8,
        )
        self.figure.tight_layout(rect=(0, 0.05, 1, 1))
        self.figure.canvas.mpl_connect("close_event", self._on_close)
        self.figure.canvas.mpl_connect("key_press_event", self._on_key)
        plt.show(block=False)

    def _on_close(self, event) -> None:
        self.closed = True

    def _on_key(self, event) -> None:
        if event.key in ("r", "R"):
            self.on_reset()
        elif event.key == "escape":
            self.close()

    def update(self, depth_m: np.ndarray) -> None:
        if self.closed:
            return
        # Reuse the policy's cached capture; never cast additional camera rays
        # or resample the network input just to draw the preview.
        self.image.set_data(depth_m)
        self.figure.canvas.draw_idle()

    def poll(self) -> None:
        if not self.closed:
            self.figure.canvas.flush_events()

    def close(self) -> None:
        self.closed = True
        self.plt.close(self.figure)
