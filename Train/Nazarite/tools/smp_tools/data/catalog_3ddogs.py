"""Create a read-only catalog of all 3DDogs modalities and sequence matches.

The catalog distinguishes new motion content from alternate representations:
the two in-the-wild MP4 directories are different backgrounds of the same
recordings, while RGB-D-only sequences are additional candidates for a future
video/depth pose-estimation pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _compact_sequence(name: str) -> str:
    """Convert d12_t1_a/d12_t1a to the common d12_t1a spelling."""
    return name.replace("_a", "a").replace("_b", "b")


def _subject_id(sequence: str) -> int | None:
    match = re.match(r"d(\d+)_", sequence)
    return int(match.group(1)) if match else None


def _base_trial(sequence: str) -> str:
    compact = _compact_sequence(sequence)
    return compact[:-1] if compact.endswith(("a", "b")) else compact


def _sequence_dirs(path: Path) -> set[str]:
    return {item.name for item in path.iterdir() if item.is_dir()} if path.is_dir() else set()


def _optical_global(path: Path) -> set[str]:
    return {
        _compact_sequence(item.stem.replace("optical_sync_align_", ""))
        for item in path.glob("optical_sync_align_*.txt")
    }


def _optical_local(path: Path) -> set[str]:
    return {
        _compact_sequence(item.parent.name)
        for item in path.glob("*/101.txt")
        if item.parent.is_dir()
    }


def _wild_sequences(path: Path) -> set[str]:
    return {
        item.parent.parent.name
        for item in path.glob("*/101/inpaint_out.mp4")
    }


def _image_count(path: Path) -> int:
    return sum(1 for _ in path.glob("*.png")) if path.is_dir() else 0


def _read_subject_splits(path: Path) -> dict[str, set[int]]:
    result: dict[str, set[int]] = {}
    for split_path in sorted(path.glob("*.txt")):
        values = {
            int(line.strip())
            for line in split_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        result[split_path.stem] = values
    return result


def _write_sequence_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "sequence_id",
        "subject_id",
        "split",
        "has_optical_global",
        "has_optical_cam_local",
        "rgb_frames_cam101",
        "depth_frames_cam122",
        "has_calibration",
        "has_detections",
        "has_wild_video_00000",
        "has_wild_video_00001",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    optical_root = root / "Data/Optical/Sync_Align_v2023_11_16b"
    rgbd_root = root / "Data/RGBD/Trimmed_Undist_FOV_Opt_2022_01"
    calibration_root = root / "Metadata_Config/RGBD/Calib/v2022_01_17_undistorted_fov_optimized__per_seq"
    detections_root = root / "Processing_Results/Detections"
    wild_root = root / "Processing_Results/InTheWildVersion"
    split_root = root / "Processing_Results/TrainingInfo"

    optical_global = _optical_global(optical_root)
    optical_local = _optical_local(optical_root / "cam_local")
    rgbd_sequences = _sequence_dirs(rgbd_root)
    calibration_bases = _sequence_dirs(calibration_root)
    detection_sequences = {
        _compact_sequence(item) for item in _sequence_dirs(detections_root)
    }
    wild_by_background = {
        background: _wild_sequences(wild_root / background)
        for background in ("00000", "00001")
    }
    splits = _read_subject_splits(split_root)

    sequences = sorted(rgbd_sequences | optical_global | optical_local)
    rows: list[dict[str, Any]] = []
    for sequence in sequences:
        subject = _subject_id(sequence)
        split = next(
            (name for name in ("train", "val", "test") if subject in splits.get(name, set())),
            "unknown",
        )
        rgb_dir = rgbd_root / sequence / "Images/101"
        depth_dir = rgbd_root / sequence / "Images/122"
        rows.append(
            {
                "sequence_id": sequence,
                "subject_id": subject,
                "split": split,
                "has_optical_global": sequence in optical_global,
                "has_optical_cam_local": sequence in optical_local,
                "rgb_frames_cam101": _image_count(rgb_dir),
                "depth_frames_cam122": _image_count(depth_dir),
                "has_calibration": _base_trial(sequence) in calibration_bases,
                "has_detections": sequence in detection_sequences,
                "has_wild_video_00000": sequence in wild_by_background["00000"],
                "has_wild_video_00001": sequence in wild_by_background["00001"],
            }
        )

    rgbd_only = [row for row in rows if not row["has_optical_global"] and row["rgb_frames_cam101"] > 0]
    optical_sequences = [row for row in rows if row["has_optical_global"]]
    report = {
        "dataset_root": str(root),
        "sequence_count_union": len(rows),
        "modality_counts": {
            "optical_global_txt": len(optical_global),
            "optical_cam_local_recordings": len(optical_local),
            "rgbd_sequence_dirs": len(rgbd_sequences),
            "rgbd_sequences_with_rgb_frames": sum(row["rgb_frames_cam101"] > 0 for row in rows),
            "rgbd_sequences_with_depth_frames": sum(row["depth_frames_cam122"] > 0 for row in rows),
            "rgb_frames_cam101": sum(row["rgb_frames_cam101"] for row in rows),
            "depth_frames_cam122": sum(row["depth_frames_cam122"] for row in rows),
            "calibration_trial_bases": len(calibration_bases),
            "detection_sequence_dirs": len(detection_sequences),
            "wild_mp4_00000": len(wild_by_background["00000"]),
            "wild_mp4_00001": len(wild_by_background["00001"]),
        },
        "matched_sequences": {
            "optical_and_rgbd": sum(row["has_optical_global"] and row["rgb_frames_cam101"] > 0 for row in rows),
            "rgbd_without_optical": len(rgbd_only),
            "optical_without_rgbd": sum(row["has_optical_global"] and row["rgb_frames_cam101"] == 0 for row in rows),
            "rgbd_without_optical_with_calibration": sum(row["has_calibration"] for row in rgbd_only),
        },
        "subject_splits": {name: sorted(values) for name, values in splits.items()},
        "interpretation": {
            "rgbd_only_use": "additional video/depth candidates; require pose estimation and physical filtering before SMP",
            "optical_rgbd_use": "supervised/validation pairs for a video pose estimator and synchronized projection checks",
            "wild_video_use": "appearance/background augmentation; same sequence IDs, so not new motion content",
            "cam_local_use": "camera-coordinate labels and 2D projection validation; same motion as global optical",
        },
        "sequences": rows,
    }
    report_path = output_dir / "3ddogs_modality_catalog.json"
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    _write_sequence_csv(output_dir / "3ddogs_sequence_catalog.csv", rows)
    _write_sequence_csv(output_dir / "rgbd_only_video_candidates.csv", rgbd_only)
    _write_sequence_csv(output_dir / "optical_labeled_sequences.csv", optical_sequences)

    print(json.dumps(
        {
            "sequence_count_union": report["sequence_count_union"],
            "modality_counts": report["modality_counts"],
            "matched_sequences": report["matched_sequences"],
        },
        indent=2,
    ))
    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()
