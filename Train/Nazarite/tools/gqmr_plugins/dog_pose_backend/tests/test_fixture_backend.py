from __future__ import annotations

import numpy as np
from gqmr.pose import VideoFrameBatch
from gqmr_dog_pose.backend import DOG27_NAMES, FixtureDogPoseBackend


class _NotCancelled:
    cancelled = False


def test_fixture_preserves_video_time_axis_and_marks_output_unsafe() -> None:
    backend = FixtureDogPoseBackend()
    backend.load({})
    video = VideoFrameBatch(
        frames=np.zeros((3, 32, 48, 3), dtype=np.uint8),
        pts=np.asarray([0, 2, 4], dtype=np.int64),
        time_base_numerator=1,
        time_base_denominator=60,
    )

    result = backend.infer(video, _NotCancelled())

    assert result.positions.shape == (3, 1, len(DOG27_NAMES), 2)
    assert np.array_equal(result.timestamps, video.timestamps)
    assert result.keypoint_names == DOG27_NAMES
    assert result.metadata["source_is_fixture"] is True
    assert result.metadata["training_eligible"] is False
