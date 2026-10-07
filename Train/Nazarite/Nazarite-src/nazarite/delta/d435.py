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

# The current WTW+DELTA attention task uses the ZED depth-stream field of
# view.  Keep these values separate from the legacy D435 calibration above so
# old DELTA/residual checkpoints retain their original camera contract.
ZED_DEPTH_FOV_HORIZONTAL_DEG = 58.4
ZED_DEPTH_FOV_VERTICAL_DEG = 45.5
ZED_DEPTH_WIDTH = 160
ZED_DEPTH_HEIGHT = 120


def pinhole_from_fov(
    width: int,
    height: int,
    horizontal_fov_deg: float,
    vertical_fov_deg: float,
) -> PinholeIntrinsics:
    """Build centered pinhole intrinsics for a rendered image size and FOV."""
    if width <= 0 or height <= 0:
        raise ValueError("Camera resolution must be positive")
    if not 0.0 < horizontal_fov_deg < 180.0:
        raise ValueError("horizontal_fov_deg must be between 0 and 180")
    if not 0.0 < vertical_fov_deg < 180.0:
        raise ValueError("vertical_fov_deg must be between 0 and 180")
    import math

    fx = width / (2.0 * math.tan(math.radians(horizontal_fov_deg) * 0.5))
    fy = height / (2.0 * math.tan(math.radians(vertical_fov_deg) * 0.5))
    return PinholeIntrinsics(
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=(width - 1) * 0.5,
        cy=(height - 1) * 0.5,
    )

# Rendering the full D435 image in every parallel environment is unnecessary.
# The full frame is resized without cropping, so scaling K preserves its rays.
DELTA_DEPTH_WIDTH = 64
DELTA_DEPTH_HEIGHT = 36
DELTA_D435_INTRINSICS = D435_DEPTH_848X480.scaled_to(
    DELTA_DEPTH_WIDTH,
    DELTA_DEPTH_HEIGHT,
)

# Portrait/vertical camera stream used only by the direct-action diagnostic
# task.  This is a 90-degree optical-axis roll of the 128x72 stream: the
# original 848x480 horizontal focal length becomes the portrait vertical
# focal length, and vice versa.
DELTA_DIRECT_DEPTH_WIDTH = 72
DELTA_DIRECT_DEPTH_HEIGHT = 128
DELTA_DIRECT_D435_INTRINSICS = PinholeIntrinsics(
    width=DELTA_DIRECT_DEPTH_WIDTH,
    height=DELTA_DIRECT_DEPTH_HEIGHT,
    fx=D435_DEPTH_848X480.fy * DELTA_DIRECT_DEPTH_WIDTH / D435_DEPTH_848X480.height,
    fy=D435_DEPTH_848X480.fx * DELTA_DIRECT_DEPTH_HEIGHT / D435_DEPTH_848X480.width,
    cx=D435_DEPTH_848X480.cy * DELTA_DIRECT_DEPTH_WIDTH / D435_DEPTH_848X480.height,
    cy=D435_DEPTH_848X480.cx * DELTA_DIRECT_DEPTH_HEIGHT / D435_DEPTH_848X480.width,
)

# Direct-task mounting hypothesis: the D435 is on the front face of the
# chassis, below the top shell, rather than above the body center.  These
# values are relative to ``robot/base_link`` and must be shared by the MuJoCo
# camera and the depth-to-BEV projection.
DELTA_DIRECT_CAMERA_POS_B = (0.40, 0.0, 0.05)
DELTA_DIRECT_CAMERA_PITCH_DEG = 0.0
DELTA_DIRECT_CAMERA_ROLL_DEG = 0.0
# MuJoCo cameras look along their local -Z axis. This pose maps
# ``-Z_cam -> +X_body`` (forward), ``+X_cam -> -Y_body`` (image right) and
# ``+Y_cam -> +Z_body`` (image up). The long portrait image axis is therefore
# vertical in the robot frame, as it is for a physically upright D435.
DELTA_DIRECT_CAMERA_QUAT = (-0.5, -0.5, 0.5, 0.5)
