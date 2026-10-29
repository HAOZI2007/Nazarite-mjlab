"""Episode files and leakage-free sequence sampling for XZ1 WMP pretraining."""

from __future__ import annotations

import bisect
import json
from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .world_model import WMPWorldModel

CONTRACT = {
    "version": 1,
    "robot": "XZ1",
    "joint_order": "FL_FR_RR_RL",
    "frame_size": 47,
    "history_size": 6,
    "history_order": "oldest_first",
    "prop_dim": 33,
    "action_dim": 12,
    "action_chunk": 5,
    "policy_dt": 0.02,
    "image_shape": [64, 64, 1],
    "depth_storage": "clipped_positive_metres_float16",
    "depth_no_hit": "far_clip",
    "action_semantics": "clipped_policy_command_before_joint_scale_offset_delay",
    "action_alignment": "five_commands_leading_to_current_image_oldest_first",
    "first_action": "zero_no_preceding_transition",
    "scan_latent": "privileged_label_only",
}


def sanitize_xz1_depth(depth: torch.Tensor, far_clip: float = 2.0):
    """MuJoCo Warp writes zero for ray misses; they are far, not near walls."""
    valid = torch.isfinite(depth) & (depth > 0)
    return torch.where(valid, depth.clamp_max(far_clip), far_clip)


def xz1_wm_prop(estimator: torch.Tensor) -> torch.Tensor:
    """Drop phase[9:11], previous action[35:47], and true velocity[47:50]."""
    if estimator.ndim != 2 or estimator.shape[1] != 50:
        raise ValueError("XZ1 WM requires the clean 50-D estimator group")
    return torch.cat((estimator[:, :9], estimator[:, 11:35]), dim=-1)


class EpisodeWriter:
    """One file per complete episode, manifest updated after every commit."""

    def __init__(self, directory: Path, metadata: dict):
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory
        self.manifest = {
            "contract": CONTRACT,
            "metadata": metadata,
            "status": "collecting",
            "episodes": [],
        }
        self.flush()

    def flush(self):
        temporary = self.directory / "manifest.json.tmp"
        temporary.write_text(
            json.dumps(self.manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temporary.replace(self.directory / "manifest.json")

    def add(self, frames: list[dict[str, torch.Tensor]], metadata: dict):
        if not frames:
            raise ValueError("Cannot save an empty episode")
        tensors = {
            key: torch.stack([frame[key] for frame in frames]) for key in frames[0]
        }
        validate_episode(tensors, self.manifest["metadata"]["camera"]["far_clip"])
        filename = f"episode_{len(self.manifest['episodes']):07d}.pt"
        path = self.directory / filename
        temporary = path.with_suffix(".pt.tmp")
        torch.save({"tensors": tensors, "metadata": metadata}, temporary)
        temporary.replace(path)
        self.manifest["episodes"].append(
            {"file": filename, "frames": len(frames), **metadata}
        )
        self.flush()

    def finish(self, status: str = "complete"):
        self.manifest["status"] = status
        self.flush()


def validate_episode(tensors: dict[str, torch.Tensor], far_clip: float):
    count = tensors["depth_m"].shape[0]
    shapes = {
        "depth_m": (count, 64, 64, 1),
        "prop": (count, 33),
        "action": (count, 60),
        "teacher_action": (count, 12),
        "teacher_latent": (count, 32),
        "teacher_scan": (count, 132),
        "actor_history": (count, 6, 47),
        "is_first": (count,),
        "time_s": (count,),
        "global_time_s": (count,),
    }
    for name, shape in shapes.items():
        if tensors[name].shape != shape or not torch.isfinite(tensors[name]).all():
            raise ValueError(f"Invalid episode field {name}; expected {shape}")
    if count < 1 or not tensors["is_first"][0] or tensors["is_first"][1:].any():
        raise ValueError("Each episode must have exactly one initial frame")
    if tensors["action"][0].count_nonzero():
        raise ValueError("Initial frame cannot contain previous-episode actions")
    if count > 1 and not torch.allclose(
        tensors["action"][1:].reshape(-1, 5, 12)[:, 0],
        tensors["teacher_action"][:-1],
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError("Action chunk does not start at the previous camera frame")
    if tensors["depth_m"].amin() < 0 or tensors["depth_m"].amax() > far_clip + 0.002:
        raise ValueError("Depth must be positive metres within the configured clip")
    if (
        tensors["teacher_action"].abs().amax() > 10
        or tensors["action"].abs().amax() > 10
    ):
        raise ValueError("Dataset contains unclipped policy commands")
    for name in ("time_s", "global_time_s"):
        delta = tensors[name][1:] - tensors[name][:-1]
        if delta.numel() and not torch.allclose(
            delta, torch.full_like(delta, 0.1), atol=1e-6, rtol=0
        ):
            raise ValueError("Camera sequence is not contiguous at 10 Hz")


class XZ1SequenceDataset(Dataset):
    """Windows from individual episodes; train/val split by terrain seeds.

    A bounded cache avoids loading the entire depth dataset into RAM. Dataset
    indices are all valid windows. Longer episodes contribute more windows;
    episodes shorter than the requested length are explicitly counted below.
    """

    def __init__(self, directory: Path, split: str, length: int = 64):
        if length < 2 or split not in ("train", "validation"):
            raise ValueError("Need sequence length >=2 and train/validation split")
        self.directory, self.length = directory, length
        self.manifest = json.loads((directory / "manifest.json").read_text())
        if self.manifest["contract"] != CONTRACT:
            raise ValueError("Unsupported XZ1 data contract")
        if self.manifest["status"] != "complete":
            raise ValueError("Collection must finish before offline pretraining")
        episodes = self.manifest["episodes"]
        self.terrain_names = sorted({row.get("terrain", "unknown") for row in episodes})
        split_seeds = {
            part: {row["seed"] for row in episodes if row["split"] == part}
            for part in ("train", "validation")
        }
        if split_seeds["train"] & split_seeds["validation"]:
            raise ValueError("Terrain seed leakage between train and validation")
        self.episodes, self.ends = [], []
        self.stats = {"eligible_episodes": 0, "short_episodes": 0, "by_terrain": {}}
        total = 0
        for row in episodes:
            if Path(row["file"]).name != row["file"]:
                raise ValueError("Episode path must stay inside the dataset")
            windows = row["frames"] - length + 1
            if row["split"] == split:
                key = "eligible_episodes" if windows > 0 else "short_episodes"
                self.stats[key] += 1
                terrain = row.get("terrain", "unknown")
                per_terrain = self.stats["by_terrain"].setdefault(
                    terrain,
                    {"eligible_episodes": 0, "short_episodes": 0, "windows": 0},
                )
                per_terrain[key] += 1
                per_terrain["windows"] += max(0, windows)
            if row["split"] == split and windows > 0:
                self.episodes.append(row)
                total += windows
                self.ends.append(total)
        if not total:
            raise ValueError(f"No {split} episodes have {length} camera frames")
        self._cache: OrderedDict[str, dict] = OrderedDict()

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index: int):
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode = bisect.bisect_right(self.ends, index)
        start = index - (self.ends[episode - 1] if episode else 0)
        filename = self.episodes[episode]["file"]
        if filename not in self._cache:
            data = torch.load(
                self.directory / filename, map_location="cpu", weights_only=True
            )["tensors"]
            validate_episode(data, self.manifest["metadata"]["camera"]["far_clip"])
            if len(data["time_s"]) != self.episodes[episode]["frames"]:
                raise ValueError("Episode length differs from its manifest")
            self._cache[filename] = data
            if len(self._cache) > 2:
                self._cache.popitem(last=False)
        self._cache.move_to_end(filename)
        data = self._cache[filename]
        window = slice(start, start + self.length)
        far = self.manifest["metadata"]["camera"]["far_clip"]
        image = WMPWorldModel.normalize_depth(data["depth_m"][window].float(), far)
        image = ((image + 0.5) * 255).round().clamp(0, 255).to(torch.uint8)
        first = torch.zeros(self.length, dtype=torch.bool)
        first[0] = True
        action = data["action"][window].clone()
        action[0] = 0
        return {
            "image": image,
            "prop": data["prop"][window].float(),
            "action": action,
            "reward": torch.zeros(self.length),
            "is_first": first,
            # This is only used by the evaluation probe, never by WM.encode.
            "teacher_latent": data["teacher_latent"][window].float(),
            "terrain_id": torch.tensor(
                self.terrain_names.index(
                    self.episodes[episode].get("terrain", "unknown")
                )
            ),
        }
