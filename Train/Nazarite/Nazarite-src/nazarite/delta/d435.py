"""Calibrated Intel RealSense D435 depth-camera parameters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PinholeIntrinsics:
  """Pinhole intrinsics tied to a specific image resolution."""

  width: int
  height: int
  fx: float
  fy: float
  cx: float
  cy: float

  def scaled_to(self, width: int, height: int) -> PinholeIntrinsics:
    """Scale intrinsics to a resized full-frame image."""
    if width <= 0 or height <= 0:
      raise ValueError("Camera resolution must be positive")
    scale_x = width / self.width
    scale_y = height / self.height
    return PinholeIntrinsics(
      width=width,
      height=height,
      fx=self.fx * scale_x,
      fy=self.fy * scale_y,
      cx=self.cx * scale_x,
      cy=self.cy * scale_y,
    )


# Factory depth-stream intrinsics read from the supplied D435 at 848x480.
# The reported Brown-Conrady coefficients are all zero, so the rectified depth
# stream is represented directly by this pinhole model.
D435_DEPTH_848X480 = PinholeIntrinsics(
  width=848,
  height=480,
  fx=420.5351867675781,
  fy=420.5351867675781,
  cx=428.09356689453125,
  cy=238.02996826171875,
)
D435_DEPTH_SCALE_M_PER_UNIT = 0.0010000000474974513

# Rendering the full D435 image in every parallel environment is unnecessary.
# The full frame is resized without cropping, so scaling K preserves its rays.
DELTA_DEPTH_WIDTH = 64
DELTA_DEPTH_HEIGHT = 36
DELTA_D435_INTRINSICS = D435_DEPTH_848X480.scaled_to(
  DELTA_DEPTH_WIDTH, DELTA_DEPTH_HEIGHT,
)
