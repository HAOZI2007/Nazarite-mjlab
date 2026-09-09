"""Create a read-only manifest of usable 3DDogs optical MoCap clips.

This scanner does not copy or modify the licensed 3DDogs source data.  It
uses the same required root and paw markers as :mod:`inspect_3ddogs`, finds
contiguous valid frame runs, and records runs long enough for later coordinate
validation and Go2 retargeting.

Example:

    cd Train/Nazarite
    uv run python tools/smp_tools/data/scan_3ddogs.py \\
      --input-dir /home/haozi/3DDogs/3DDogs2024_full/Data/Optical/ \\
Sync_Align_v2023_11_16b \\
      --output-dir output/3ddogs_scan --min-duration-s 2.0

The outputs are:

* ``all_valid_runs.csv``: every contiguous run with all required markers;
* ``selected_clips.csv``: only runs meeting ``--min-duration-s``;
* ``scan_summary.json``: settings, aggregate counts, and parse failures.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import numpy as np

if TYPE_CHECKING:
  from tools.smp_tools.data.inspect_3ddogs import (
    _contiguous_runs,
    extract_retarget_inputs,
    load_optical_trial,
  )
else:
  # Keep the documented direct-script command independent from the training
  # package and its simulator task-registration side effects.
  from inspect_3ddogs import (
    _contiguous_runs,
    extract_retarget_inputs,
    load_optical_trial,
  )

RUN_FIELDNAMES: tuple[str, ...] = (
  "source_path",
  "source_relpath",
  "trial_name",
  "fps",
  "num_frames",
  "marker_count",
  "valid_frames",
  "valid_fraction",
  "start_index",
  "end_index_exclusive",
  "start_frame_num",
  "end_frame_num",
  "length_frames",
  "duration_s",
  "meets_min_duration",
)


class RunRow(TypedDict):
  """One continuous valid run recorded in the CSV manifests."""

  source_path: str
  source_relpath: str
  trial_name: str
  fps: float
  num_frames: int
  marker_count: int
  valid_frames: int
  valid_fraction: float
  start_index: int
  end_index_exclusive: int
  start_frame_num: int
  end_frame_num: int
  length_frames: int
  duration_s: float
  meets_min_duration: bool


@dataclass(frozen=True)
class ScanSettings:
  """Inputs that determine reproducible valid-clip selection."""

  input_dir: Path
  file_glob: str
  min_duration_s: float


def _run_rows(input_path: Path, input_dir: Path) -> list[RunRow]:
  """Read a trial and return one manifest row per contiguous valid run."""
  trial = load_optical_trial(input_path)
  extracted = extract_retarget_inputs(trial)
  valid_mask = extracted["valid_mask"]
  valid_frames = int(np.count_nonzero(valid_mask))
  rows: list[RunRow] = []

  for start, end in _contiguous_runs(valid_mask):
    length_frames = end - start
    rows.append(
      {
        "source_path": str(input_path.resolve()),
        "source_relpath": str(input_path.relative_to(input_dir)),
        "trial_name": input_path.stem,
        "fps": trial.fps,
        "num_frames": int(trial.frame_numbers.size),
        "marker_count": len(trial.marker_names),
        "valid_frames": valid_frames,
        "valid_fraction": float(valid_mask.mean()),
        "start_index": start,
        "end_index_exclusive": end,
        "start_frame_num": int(trial.frame_numbers[start]),
        "end_frame_num": int(trial.frame_numbers[end - 1]),
        "length_frames": length_frames,
        "duration_s": length_frames / trial.fps,
        "meets_min_duration": False,
      }
    )
  return rows


def _write_csv(
  output_path: Path,
  rows: Iterable[RunRow],
) -> None:
  with output_path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=RUN_FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)


def scan_dataset(settings: ScanSettings) -> tuple[list[RunRow], dict[str, object]]:
  """Scan matching trials without writing to or changing any source file."""
  input_dir = settings.input_dir.resolve()
  input_paths = sorted(path for path in input_dir.glob(settings.file_glob) if path.is_file())
  if not input_paths:
    raise FileNotFoundError(
      f"no files matched {settings.file_glob!r} under {input_dir}"
    )

  all_rows: list[RunRow] = []
  failures: list[dict[str, str]] = []
  trials_with_valid_frames = 0
  for input_path in input_paths:
    try:
      rows = _run_rows(input_path, input_dir)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
      failures.append({"source_path": str(input_path), "error": str(exc)})
      continue
    all_rows.extend(rows)
    if rows:
      trials_with_valid_frames += 1

  for row in all_rows:
    row["meets_min_duration"] = bool(row["duration_s"] >= settings.min_duration_s)
  all_rows.sort(key=lambda row: (row["source_relpath"], row["start_index"]))
  selected_rows = [row for row in all_rows if row["meets_min_duration"]]
  selected_rows.sort(key=lambda row: row["duration_s"], reverse=True)

  summary: dict[str, object] = {
    "source_directory": str(input_dir),
    "file_glob": settings.file_glob,
    "min_duration_s": settings.min_duration_s,
    "source_files_found": len(input_paths),
    "source_files_parsed": len(input_paths) - len(failures),
    "source_files_failed": len(failures),
    "trials_with_any_valid_run": trials_with_valid_frames,
    "all_valid_runs": len(all_rows),
    "selected_clips": len(selected_rows),
    "selected_duration_s": sum(row["duration_s"] for row in selected_rows),
    "failures": failures,
  }
  return all_rows, summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--input-dir",
    type=Path,
    required=True,
    help="Directory containing optical_sync_align_*.txt source files",
  )
  parser.add_argument(
    "--output-dir", type=Path, required=True, help="Directory for derived manifests"
  )
  parser.add_argument(
    "--file-glob",
    default="optical_sync_align_*.txt",
    help="Glob evaluated directly inside --input-dir",
  )
  parser.add_argument(
    "--min-duration-s",
    type=float,
    default=2.0,
    help="Minimum contiguous fully-valid duration for selected_clips.csv (default: 2.0)",
  )
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
  args = _parse_args(argv)
  if args.min_duration_s <= 0.0:
    raise ValueError("--min-duration-s must be positive")

  input_dir: Path = args.input_dir.expanduser().resolve()
  if not input_dir.is_dir():
    raise NotADirectoryError(input_dir)
  output_dir: Path = args.output_dir.expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

  settings = ScanSettings(
    input_dir=input_dir,
    file_glob=args.file_glob,
    min_duration_s=args.min_duration_s,
  )
  all_rows, summary = scan_dataset(settings)
  selected_rows = [row for row in all_rows if row["meets_min_duration"]]
  selected_rows.sort(key=lambda row: row["duration_s"], reverse=True)

  _write_csv(output_dir / "all_valid_runs.csv", all_rows)
  _write_csv(output_dir / "selected_clips.csv", selected_rows)
  with (output_dir / "scan_summary.json").open("w", encoding="utf-8") as stream:
    json.dump(summary, stream, indent=2)
    stream.write("\n")

  print(json.dumps(summary, indent=2))
  print(f"\nWrote: {output_dir / 'all_valid_runs.csv'}")
  print(f"Wrote: {output_dir / 'selected_clips.csv'}")
  print(f"Wrote: {output_dir / 'scan_summary.json'}")


if __name__ == "__main__":
  main()
