"""Convert a DeepLabCut 2D CSV into GQMR generic dog-27 keypoints.

The output remains a 2D intermediate artifact and is explicitly marked as not
training eligible.  Use JSON when the file will be passed directly to GQMR's
current ``pose triangulate`` CLI; NPZ is available for compact archival.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from gqmr.core.json import loads_strict_json
from gqmr.exporters.common import atomic_write
from gqmr.pose import KeypointBatch
from gqmr.skeletons import get_skeleton
from gqmr.sources.files import load_deeplabcut_csv, save_generic_keypoints_npz

PELVIS_DUPLICATE = "pelvis_duplicate"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="DeepLabCut CSV")
    parser.add_argument("--output", type=Path, required=True, help="Generic .json or .npz")
    parser.add_argument("--fps", type=float, required=True, help="Source video frame rate")
    parser.add_argument(
        "--mapping",
        type=Path,
        help="Optional strict JSON object mapping DLC bodypart names to dog-27 names",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.6,
        help="Observations below this likelihood are marked invalid (default: 0.6)",
    )
    return parser.parse_args(argv)


def load_mapping(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    try:
        document = loads_strict_json(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"cannot read mapping file {path}: {error}") from error
    if not isinstance(document, dict):
        raise TypeError("mapping file must contain a JSON object")
    if any(
        not isinstance(source, str)
        or not source
        or not isinstance(target, str)
        or not target
        for source, target in document.items()
    ):
        raise TypeError("mapping keys and values must be non-empty strings")
    return dict(document)


def remap_dlc_to_dog27(
    source: KeypointBatch,
    *,
    mapping: dict[str, str],
    confidence_threshold: float,
) -> KeypointBatch:
    """Return a dog-27 ordered 2D batch with thresholded validity."""

    if source.dimensions != 2 or len(source.instance_ids) != 1:
        raise ValueError("DeepLabCut conversion requires one 2D animal instance")
    if (
        not np.isfinite(confidence_threshold)
        or confidence_threshold < 0.0
        or confidence_threshold > 1.0
    ):
        raise ValueError("confidence threshold must be finite and in [0, 1]")

    dog27_names = get_skeleton("dog-27").names
    valid_targets = set(dog27_names) - {PELVIS_DUPLICATE}
    unknown_targets = sorted(set(mapping.values()) - valid_targets)
    if unknown_targets:
        raise ValueError(f"mapping contains unknown dog-27 targets: {unknown_targets}")
    unknown_sources = sorted(set(mapping) - set(source.keypoint_names))
    if unknown_sources:
        raise ValueError(f"mapping contains DLC bodyparts not present in CSV: {unknown_sources}")

    source_for_target: dict[str, str] = {}
    for source_name in source.keypoint_names:
        target_name = mapping.get(source_name, source_name)
        if target_name not in valid_targets:
            continue
        previous = source_for_target.get(target_name)
        if previous is not None:
            raise ValueError(
                f"DLC bodyparts {previous!r} and {source_name!r} both map to {target_name!r}"
            )
        source_for_target[target_name] = source_name

    missing = sorted(valid_targets - set(source_for_target))
    if missing:
        raise ValueError(
            "DLC result cannot form dog-27; missing mapped keypoints: "
            f"{missing}"
        )

    name_to_index = {name: index for index, name in enumerate(source.keypoint_names)}
    frames = len(source.timestamps)
    positions = np.full((frames, 1, len(dog27_names), 2), np.nan, dtype=np.float32)
    confidence = np.zeros((frames, 1, len(dog27_names)), dtype=np.float32)
    valid = np.zeros((frames, 1, len(dog27_names)), dtype=np.bool_)
    for target_index, target_name in enumerate(dog27_names):
        lookup_name = "pelvis" if target_name == PELVIS_DUPLICATE else target_name
        source_index = name_to_index[source_for_target[lookup_name]]
        positions[:, 0, target_index] = source.positions[:, 0, source_index]
        confidence[:, 0, target_index] = source.confidence[:, 0, source_index]
        valid[:, 0, target_index] = (
            source.valid_mask[:, 0, source_index]
            & (source.confidence[:, 0, source_index] >= confidence_threshold)
        )
    positions[~valid] = np.nan

    return KeypointBatch(
        timestamps=source.timestamps,
        keypoint_names=dog27_names,
        instance_ids=source.instance_ids,
        positions=positions,
        confidence=confidence,
        valid_mask=valid,
        coordinate_frame="image_pixels_x_right_y_down",
        metadata={
            "format": "deeplabcut_csv_dog27",
            "source": source.metadata,
            "mapping_source_to_dog27": mapping,
            "confidence_threshold": confidence_threshold,
            "pipeline_stage": "pose_2d",
            "training_eligible": False,
            "next_required_step": "calibrated multiview triangulation or validated 3D lifting",
        },
    )


def _json_position_payload(batch: KeypointBatch) -> list[Any]:
    payload = batch.positions.astype(object)
    invalid_coordinates = np.broadcast_to(~batch.valid_mask[..., None], batch.positions.shape)
    payload[invalid_coordinates] = None
    return payload.tolist()


def save_generic_keypoints_json(path: Path, batch: KeypointBatch) -> Path:
    document = {
        "timestamps": batch.timestamps.tolist(),
        "keypoint_names": list(batch.keypoint_names),
        "instance_ids": list(batch.instance_ids),
        "positions": _json_position_payload(batch),
        "confidence": batch.confidence.tolist(),
        "valid_mask": batch.valid_mask.tolist(),
        "coordinate_frame": batch.coordinate_frame,
        "metadata": batch.metadata,
    }
    encoded = (
        json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    return atomic_write(path, lambda stream: stream.write(encoded))


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    source_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"DeepLabCut CSV does not exist: {source_path}")
    if output_path.suffix.lower() not in {".json", ".npz"}:
        raise ValueError("--output must have a .json or .npz suffix")

    mapping = load_mapping(args.mapping.expanduser().resolve() if args.mapping else None)
    source = load_deeplabcut_csv(source_path, fps=args.fps)
    result = remap_dlc_to_dog27(
        source,
        mapping=mapping,
        confidence_threshold=args.confidence_threshold,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".json":
        save_generic_keypoints_json(output_path, result)
    else:
        save_generic_keypoints_npz(output_path, result)

    per_keypoint = {
        name: float(np.mean(result.valid_mask[:, 0, index]))
        for index, name in enumerate(result.keypoint_names)
    }
    summary = {
        "input": str(source_path),
        "output": str(output_path),
        "frames": len(result.timestamps),
        "fps": args.fps,
        "keypoints": len(result.keypoint_names),
        "coordinate_frame": result.coordinate_frame,
        "valid_observation_fraction": float(np.mean(result.valid_mask)),
        "minimum_keypoint_valid_fraction": min(per_keypoint.values()),
        "confidence_threshold": args.confidence_threshold,
        "training_eligible": False,
        "next_required_step": "GQMR multiview triangulation",
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
