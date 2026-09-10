"""Test-only dog-27 pose backend for validating the GQMR video boundary."""

from __future__ import annotations

from typing import Any

import numpy as np
from gqmr.pose import KeypointBatch, PoseBackendInfo, VideoFrameBatch

DOG27_NAMES = (
    "pelvis",
    "pelvis_duplicate",
    "spine",
    "neck",
    "head",
    "muzzle",
    "left_shoulder",
    "left_front_upper",
    "left_front_elbow",
    "left_front_wrist",
    "left_front_toe",
    "right_shoulder",
    "right_front_upper",
    "right_front_elbow",
    "right_front_wrist",
    "right_front_toe",
    "left_hip",
    "left_hind_knee",
    "left_hind_ankle",
    "left_hind_toe",
    "right_hip",
    "right_hind_knee",
    "right_hind_ankle",
    "right_hind_toe",
    "tail_base",
    "tail_mid",
    "tail_tip",
)

# Side-view fixture in normalized image coordinates (x right, y down).  This is
# intentionally static and only verifies schema, timestamps, plugin discovery,
# and serialization.  It does not represent observations from the input video.
_NORMALIZED_TEMPLATE = np.asarray(
    [
        (0.43, 0.42), (0.43, 0.42), (0.50, 0.39), (0.61, 0.37),
        (0.69, 0.33), (0.77, 0.36),
        (0.60, 0.42), (0.62, 0.49), (0.64, 0.57), (0.66, 0.67), (0.69, 0.74),
        (0.58, 0.43), (0.57, 0.50), (0.56, 0.58), (0.55, 0.68), (0.57, 0.75),
        (0.42, 0.45), (0.44, 0.53), (0.47, 0.63), (0.50, 0.74),
        (0.39, 0.46), (0.36, 0.54), (0.34, 0.64), (0.35, 0.75),
        (0.35, 0.40), (0.27, 0.37), (0.19, 0.34),
    ],
    dtype=np.float32,
)


class FixtureDogPoseBackend:
    """Emit a deterministic 2D dog-27 fixture; never valid for training."""

    api_version = 1

    def __init__(self) -> None:
        self._loaded = False
        self._confidence = 0.99

    def describe(self) -> PoseBackendInfo:
        return PoseBackendInfo(
            api_version=1,
            name="Nazarite dog-27 fixture (test only)",
            package="nazarite-gqmr-dog-pose",
            package_version="0.1.0",
            skeleton_ids=("dog-27",),
            dimensions=(2,),
            multi_instance=False,
            batch_range=(1, 100_000),
            devices=("cpu",),
            output_coordinate_frame="image_pixels_x_right_y_down",
        )

    def load(self, config: dict[str, Any]) -> None:
        unknown = set(config) - {"confidence"}
        if unknown:
            raise ValueError(f"unknown fixture backend options: {sorted(unknown)}")
        confidence = float(config.get("confidence", 0.99))
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1]")
        self._confidence = confidence
        self._loaded = True

    def infer(self, batch: VideoFrameBatch, cancel: Any) -> KeypointBatch:
        if not self._loaded:
            raise RuntimeError("fixture backend must be loaded before inference")
        if cancel.cancelled:
            raise RuntimeError("fixture inference was cancelled")

        frame_count, height, width = batch.frames.shape[:3]
        image_scale = np.asarray([width - 1, height - 1], dtype=np.float32)
        points = _NORMALIZED_TEMPLATE * image_scale
        positions = np.broadcast_to(
            points, (frame_count, 1, len(DOG27_NAMES), 2)
        ).copy()
        confidence = np.full(
            (frame_count, 1, len(DOG27_NAMES)),
            self._confidence,
            dtype=np.float32,
        )
        return KeypointBatch(
            timestamps=batch.timestamps,
            keypoint_names=DOG27_NAMES,
            instance_ids=("fixture-dog",),
            positions=positions,
            confidence=confidence,
            valid_mask=np.ones_like(confidence, dtype=np.bool_),
            coordinate_frame="image_pixels_x_right_y_down",
            metadata={
                "backend": "dog-pose-fixture",
                "source_is_fixture": True,
                "training_eligible": False,
                "warning": "Static schema fixture; contains no video-derived pose.",
            },
        )

    def close(self) -> None:
        self._loaded = False
