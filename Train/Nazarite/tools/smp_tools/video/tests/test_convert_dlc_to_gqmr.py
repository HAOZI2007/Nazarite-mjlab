from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from gqmr.skeletons import get_skeleton
from gqmr.sources.files import load_deeplabcut_csv, load_generic_keypoints_json
from tools.smp_tools.video.convert_dlc_to_gqmr import (
    remap_dlc_to_dog27,
    save_generic_keypoints_json,
)


def _write_dlc_csv(path: Path, names: list[str]) -> None:
    header = ["scorer"]
    bodyparts = ["bodyparts"]
    coordinates = ["coords"]
    for name in names:
        header.extend(("net", "net", "net"))
        bodyparts.extend((name, name, name))
        coordinates.extend(("x", "y", "likelihood"))
    rows = [header, bodyparts, coordinates]
    for frame in range(2):
        row: list[str | int | float] = [frame]
        for index in range(len(names)):
            likelihood = 0.4 if frame == 1 and names[index] == "muzzle" else 0.9
            row.extend((10.0 + index + frame, 20.0 + index, likelihood))
        rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)


def test_dlc_conversion_reorders_dog27_and_writes_strict_json(tmp_path: Path) -> None:
    dog27_names = get_skeleton("dog-27").names
    unique_names = [name for name in dog27_names if name != "pelvis_duplicate"]
    source_path = tmp_path / "dlc.csv"
    _write_dlc_csv(source_path, unique_names)
    source = load_deeplabcut_csv(source_path, fps=50.0)

    converted = remap_dlc_to_dog27(
        source,
        mapping={},
        confidence_threshold=0.6,
    )
    output = tmp_path / "dog27.json"
    save_generic_keypoints_json(output, converted)
    restored = load_generic_keypoints_json(output)

    assert restored.keypoint_names == dog27_names
    assert restored.positions.shape == (2, 1, 27, 2)
    pelvis = restored.keypoint_names.index("pelvis")
    duplicate = restored.keypoint_names.index("pelvis_duplicate")
    muzzle = restored.keypoint_names.index("muzzle")
    assert np.array_equal(
        restored.positions[:, 0, pelvis],
        restored.positions[:, 0, duplicate],
        equal_nan=True,
    )
    assert not restored.valid_mask[1, 0, muzzle]
    assert np.isnan(restored.positions[1, 0, muzzle]).all()
    assert restored.metadata["training_eligible"] is False


def test_dlc_conversion_rejects_incomplete_skeleton(tmp_path: Path) -> None:
    source_path = tmp_path / "dlc.csv"
    _write_dlc_csv(source_path, ["pelvis", "muzzle"])
    source = load_deeplabcut_csv(source_path, fps=60.0)

    with pytest.raises(ValueError, match="missing mapped keypoints"):
        remap_dlc_to_dog27(source, mapping={}, confidence_threshold=0.6)
