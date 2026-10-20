"""Unit tests for the AMP motion loader and normaliser.

These run without a simulator: the loader is plain JSON plus tensor slicing, so a
synthetic clip can pin the column layout and the wrap-around pairing, and the
bundled Go2 set only needs to be checked for shape and plausibility.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from nazarite.rl.wmp import motion

TORCH_SEED = 7


def _frame(offset: float) -> list[float]:
    """A 61-wide frame whose every column carries its own index as a marker."""
    return [float(column) + offset for column in range(motion.AMP_FRAME_DIM)]


def _write_clip(path: Path, offsets: list[float], weight: float = 1.0) -> Path:
    payload = {
        "LoopMode": "Wrap",
        "FrameDuration": 0.02,
        "MotionWeight": weight,
        "Frames": [_frame(offset) for offset in offsets],
    }
    path.write_text(json.dumps(payload))
    return path


def test_state_from_frames_keeps_the_amp_column_order() -> None:
    frames = torch.tensor([_frame(0.0), _frame(1.0)])
    states = motion.state_from_frames(frames)

    assert states.shape == (2, motion.AMP_OBSERVATION_DIM)
    # Joints 7..18, base linear 31..33, base angular 34..36, joint velocities 37..48.
    expected_columns = list(range(7, 19)) + list(range(31, 34)) + list(range(34, 37))
    expected_columns += list(range(37, 49))
    assert states[0].tolist() == [float(column) for column in expected_columns]
    # The toe columns (19..30) must not leak in.
    assert 19.0 not in states[0].tolist()
    assert 30.0 not in states[0].tolist()


def test_state_from_frames_rejects_a_wrong_width() -> None:
    with pytest.raises(ValueError, match="clip frames"):
        motion.state_from_frames(torch.zeros(4, motion.AMP_FRAME_DIM - 1))


def test_load_clip_reads_metadata_and_states(tmp_path: Path) -> None:
    clip = motion.load_clip(_write_clip(tmp_path / "trot.txt", [0.0, 1.0, 2.0]))
    assert clip.name == "trot"
    assert clip.frame_duration == pytest.approx(0.02)
    assert clip.loop_mode == "Wrap"
    assert clip.weight == pytest.approx(1.0)
    assert clip.states.shape == (3, motion.AMP_OBSERVATION_DIM)


def test_load_clip_rejects_negative_frame_duration(tmp_path: Path) -> None:
    path = _write_clip(tmp_path / "bad.txt", [0.0, 1.0])
    payload = json.loads(path.read_text())
    payload["FrameDuration"] = -0.02
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="FrameDuration"):
        motion.load_clip(path)


def test_dataset_pairs_each_state_with_its_successor_and_wraps() -> None:
    clip = motion.MotionClip(
        name="ramp",
        states=torch.arange(4, dtype=torch.float32)
        .repeat(motion.AMP_OBSERVATION_DIM, 1)
        .T,
    )
    dataset = motion.MotionDataset([clip])
    generator = torch.Generator().manual_seed(TORCH_SEED)
    current, following = dataset.sample(64, generator)

    assert current.shape == following.shape == (64, motion.AMP_OBSERVATION_DIM)
    # Every row of this synthetic clip is a constant, so the pairing is visible:
    # a normal step advances by one, and the last frame wraps to frame 0 (-3).
    steps = following - current
    assert set(steps[:, 0].tolist()) <= {1.0, -3.0}
    assert float((steps[:, 0] == 1.0).float().mean()) > 0.5


def test_dataset_honours_clip_weights() -> None:
    first = motion.MotionClip(
        name="a", states=torch.zeros(4, motion.AMP_OBSERVATION_DIM)
    )
    second = motion.MotionClip(
        name="b", states=torch.ones(4, motion.AMP_OBSERVATION_DIM), weight=9.0
    )
    dataset = motion.MotionDataset([first, second])
    current, _ = dataset.sample(2000, torch.Generator().manual_seed(TORCH_SEED))
    share_b = float((current[:, 0] > 0.5).float().mean())
    assert share_b == pytest.approx(0.9, abs=0.03)


def test_dataset_rejects_an_empty_mixture() -> None:
    with pytest.raises(ValueError, match="at least one clip"):
        motion.MotionDataset([])


def test_bundled_go2_clips_load_and_look_plausible() -> None:
    dataset = motion.load_dataset()
    assert len(dataset.clips) == 17
    names = {clip.name for clip in dataset.clips}
    assert "go2_stance_0.0mps" in names
    assert "go2_forward_0.8mps" in names
    # Clips are 3-5 s at 50 Hz; the slowest ones are the shortest.
    assert all(clip.states.shape[0] > 100 for clip in dataset.clips)
    current, following = dataset.sample(256, torch.Generator().manual_seed(TORCH_SEED))
    assert torch.isfinite(current).all()
    assert torch.isfinite(following).all()
    # Joint velocities of a trot are order 1 rad/s, not order 100.
    assert float(current[:, 6:12].abs().max()) < 60.0


def test_normalizer_updates_use_raw_states_and_leave_normalize_pure() -> None:
    normalizer = motion.AMPStateNormalizer()
    batch = torch.randn(512, motion.AMP_OBSERVATION_DIM) * 3.0 + 5.0

    before = normalizer.mean.clone()
    normalized = normalizer.normalize(batch)
    assert torch.equal(normalizer.mean, before)
    assert float(normalized.mean()) != pytest.approx(0.0, abs=0.1)

    normalizer.update(batch)
    assert normalizer.count == pytest.approx(512.0)
    assert torch.allclose(normalizer.mean, batch.mean(dim=0), atol=1e-4)
    assert torch.allclose(normalizer.var, batch.var(dim=0, unbiased=False), atol=1e-3)
    # After the update the same batch normalises to roughly zero mean, unit scale.
    normalized = normalizer.normalize(batch)
    assert float(normalized.mean().abs()) < 1e-3
    assert float(normalized.std()) == pytest.approx(1.0, abs=1e-2)


def test_normalizer_accumulates_across_batches() -> None:
    normalizer = motion.AMPStateNormalizer()
    first = torch.full((100, motion.AMP_OBSERVATION_DIM), 1.0)
    second = torch.full((300, motion.AMP_OBSERVATION_DIM), 3.0)
    normalizer.update(first)
    normalizer.update(second)

    assert normalizer.count == pytest.approx(400.0)
    assert float(normalizer.mean[0]) == pytest.approx(2.5, abs=1e-5)
    # Population variance of {1 x100, 3 x300} is 0.75.
    assert float(normalizer.var[0]) == pytest.approx(0.75, abs=1e-5)


def test_normalizer_state_dict_round_trips() -> None:
    normalizer = motion.AMPStateNormalizer()
    normalizer.update(torch.randn(128, motion.AMP_OBSERVATION_DIM))
    state = normalizer.state_dict()

    restored = motion.AMPStateNormalizer()
    restored.load_state_dict(state)
    assert torch.allclose(restored.mean, normalizer.mean)
    assert torch.allclose(restored.var, normalizer.var)
    assert restored.count == pytest.approx(normalizer.count)
    # The round trip must not alias the source buffers.
    restored.mean[0] = 123.0
    assert float(normalizer.mean[0]) != 123.0
