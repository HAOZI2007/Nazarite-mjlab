"""Preprocess accepted geometric references and write a training manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _read(path: Path) -> list[dict[str, Any]]:
  with path.open("r", encoding="utf-8") as stream:
    value = json.load(stream)
  if not isinstance(value, list):
    raise ValueError(f"accepted manifest must be a JSON list: {path}")
  return value


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--accepted-manifest", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--playback-rate", type=float, default=0.5)
  parser.add_argument("--smoothing-window", type=int, default=5)
  return parser.parse_args(argv)


def _preprocess_dir_name(playback_rate: float) -> str:
  """Return a stable, human-readable directory name for one speed variant.

  Keep the original half-speed directory name so existing teacher manifests
  remain valid.  Other speeds must use a separate directory; otherwise a new
  preprocessing run could silently overwrite the half-speed data referenced by
  the current teacher environment.
  """
  if abs(playback_rate - 0.5) < 1.0e-9:
    return "preprocessed_half_speed"
  rate_text = f"{playback_rate:.6f}".rstrip("0").rstrip(".").replace(".", "p")
  return f"preprocessed_rate_{rate_text}"


def main(argv: Sequence[str] | None = None) -> None:
  args = _parse_args(argv)
  if not 0.0 < args.playback_rate <= 1.0:
    raise ValueError("--playback-rate must be in (0, 1]")
  if args.smoothing_window < 1 or args.smoothing_window % 2 == 0:
    raise ValueError("--smoothing-window must be a positive odd integer")
  manifest_path = args.accepted_manifest.expanduser().resolve()
  output_path = args.output.expanduser().resolve()
  entries = _read(manifest_path)
  project_root = Path(__file__).resolve().parents[3]
  preprocess_script = project_root / "tools/smp_tools/preprocessing/preprocess_go2_reference.py"
  updated: list[dict[str, Any]] = []
  failures: list[dict[str, str]] = []
  preprocess_dir_name = _preprocess_dir_name(args.playback_rate)
  for entry in entries:
    clip_root = Path(entry["geometric_npz"]).resolve().parent
    geometric_path = Path(entry["geometric_npz"]).resolve()
    preprocess_root = clip_root / preprocess_dir_name
    try:
      result = subprocess.run([
        sys.executable,
        str(preprocess_script),
        "--input", str(geometric_path),
        "--output-dir", str(preprocess_root),
        "--playback-rate", str(args.playback_rate),
        "--smoothing-window", str(args.smoothing_window),
      ], check=False, capture_output=True, text=True)
      if result.returncode != 0:
        raise RuntimeError(f"{result.stdout}\n{result.stderr}")
      enriched = dict(entry)
      enriched["preprocessed_npz"] = str(preprocess_root / "go2_reference_preprocessed.npz")
      enriched["preprocess_summary"] = str(preprocess_root / "preprocess_summary.json")
      updated.append(enriched)
    except (OSError, RuntimeError) as exc:
      failures.append({"clip_name": str(entry.get("clip_name", "unknown")), "error": str(exc)})

  report = {
    "source_manifest": str(manifest_path),
    "playback_rate": args.playback_rate,
    "smoothing_window": args.smoothing_window,
    "requested_clips": len(entries),
    "preprocessed_clips": len(updated),
    "failed_clips": len(failures),
    "clips": updated,
    "failures": failures,
  }
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.open("w", encoding="utf-8") as stream:
    json.dump(report, stream, indent=2)
    stream.write("\n")
  print(json.dumps({key: report[key] for key in (
    "requested_clips", "preprocessed_clips", "failed_clips", "playback_rate"
  )}, indent=2))
  print(f"\nWrote: {output_path}")


if __name__ == "__main__":
  main()
