"""Checkpoint/bundle loading and the complete 50 Hz / 10 Hz WMP runtime."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from nazarite.rl.wmp.world_model import WMPWorldModel, WMPWorldModelCfg
from nazarite.rl.wmp.xz1_dataset import CONTRACT, sanitize_xz1_depth
from nazarite.rl.xz1_student import STUDENT_FORMAT, build_student, file_sha256

from ..config import CONTROL_DT, PHYSICS_DT
from ..xz1_him import ACTION_OFFSET, DEFAULT_Q, JOINT_NAMES

BUNDLE_FORMAT = "xz1_wmp_sim2sim_v1"


def control_contract() -> dict:
    return {
        "joint_names": JOINT_NAMES,
        "default_q": DEFAULT_Q.tolist(),
        "action_offset": ACTION_OFFSET.tolist(),
        "action_scale": 0.25,
        "control_dt": CONTROL_DT,
        "physics_dt": PHYSICS_DT,
        "kp": [32, 32, 35] * 4,
        "kd": [1.9, 1.9, 1.8] * 4,
        "effort_limit": 31.7,
        "armature": 0.025392063,
    }


def load_bundle(path: Path, world_model: Path | None = None) -> dict:
    """No teacher checkpoint, training environment, or dataset is needed."""
    path = path.expanduser().resolve()
    source = torch.load(path, map_location="cpu", weights_only=True)
    if source.get("format") == BUNDLE_FORMAT:
        if world_model is not None:
            raise ValueError("An exported bundle already contains its World Model")
        bundle = source
    else:
        if source.get("format") != STUDENT_FORMAT:
            raise ValueError("Expected an XZ1 visual student or sim2sim bundle")
        wm_path = (world_model or Path(source["world_model_path"])).expanduser()
        if file_sha256(wm_path) != source["world_model_sha256"]:
            raise ValueError("World Model SHA256 differs from the student's WM")
        wm = torch.load(wm_path, map_location="cpu", weights_only=True)
        if wm.get("world_model_implementation") != WMPWorldModel.implementation:
            raise ValueError("World Model is not the ported reference WMP")
        if wm.get("contract") != source.get("contract"):
            raise ValueError("Student and World Model contracts differ")
        bundle = {
            "format": BUNDLE_FORMAT,
            "contract": source["contract"],
            "actor_kwargs": source["actor_kwargs"],
            "actor_state_dict": source["actor_state_dict"],
            "camera": source["camera"],
            "world_model_cfg": wm["model_cfg"],
            "world_model_state_dict": wm["world_model"],
            "control": control_contract(),
            "sources": {
                "student": str(path),
                "student_sha256": file_sha256(path),
                "world_model": str(wm_path.resolve()),
                "world_model_sha256": source["world_model_sha256"],
            },
        }
    if bundle.get("contract") != CONTRACT:
        raise ValueError("XZ1 47x6 / prop33 / action5x12 contract differs")
    if bundle.get("control") != control_contract():
        raise ValueError("Bundle servo/observation contract differs from XZ1")
    camera = bundle["camera"]
    if (
        camera["interval"] != 5
        or not np.isfinite(camera["far_clip"])
        or camera["far_clip"] <= 0
        or not 0 < camera["fovy"] < 180
        or len(camera["pos"]) != 3
        or len(camera["quat"]) != 4
        or not np.isfinite([*camera["pos"], *camera["quat"]]).all()
        or not np.isclose(np.linalg.norm(camera["quat"]), 1, atol=1e-4)
    ):
        raise ValueError("Invalid saved depth camera contract")
    cfg = WMPWorldModelCfg(**bundle["world_model_cfg"])
    if (cfg.action_chunk, cfg.prop_dim, cfg.deter_dim) != (5, 33, 512):
        raise ValueError("Expected action60 / prop33 / deter512 WM")
    for group in ("actor_state_dict", "world_model_state_dict"):
        if not all(torch.isfinite(v).all() for v in bundle[group].values()):
            raise ValueError(f"Non-finite {group}")
    return bundle


class WMPStudentPolicy:
    def __init__(self, bundle: dict, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.camera = bundle["camera"]
        obs = TensorDict(
            {"actor": torch.zeros(1, 6, 47, device=self.device)}, batch_size=[1]
        )
        self.actor = build_student(obs, device, bundle["actor_kwargs"])
        self.actor.load_state_dict(bundle["actor_state_dict"], strict=True)
        self.actor.requires_grad_(False).eval()
        self.model = WMPWorldModel(
            WMPWorldModelCfg(**bundle["world_model_cfg"]), device=device
        )
        self.model.load_state_dict(bundle["world_model_state_dict"], strict=True)
        self.model.requires_grad_(False).eval()
        self.reset()

    @torch.no_grad()
    def reset(self) -> None:
        self.tick = 0
        self.actions = torch.zeros(1, 5, 12, device=self.device)
        self.feature = torch.zeros(1, 512, device=self.device)
        self.state = self.model.dynamics.initial(1)
        self.last_action = np.zeros(12, dtype=np.float32)
        self.last_depth: np.ndarray | None = None
        self.image_updates = 0
        self._awaiting_advance = False

    @torch.no_grad()
    def step(
        self, history: np.ndarray, depth_reader: Callable[[], np.ndarray]
    ) -> np.ndarray:
        if self._awaiting_advance:
            raise RuntimeError("Call advance(executed_action) after the control step")
        history = np.asarray(history, dtype=np.float32)
        if history.shape != (282,) or not np.isfinite(history).all():
            raise ValueError("Expected finite current-first 6x47 history")
        # Existing sim2sim builder is current-first. The native training actor
        # consumes oldest-first [batch,6,47] and reverses it internally once.
        frames = torch.as_tensor(history.reshape(6, 47).copy(), device=self.device)
        if self.tick % self.camera["interval"] == 0:
            depth = np.asarray(depth_reader(), dtype=np.float32)
            if depth.shape != (64, 64):
                raise ValueError("Depth must be a 64x64 image in positive metres")
            self.last_depth = depth.copy()
            depth_tensor = torch.as_tensor(depth, device=self.device)[None, ..., None]
            depth_tensor = sanitize_xz1_depth(depth_tensor, self.camera["far_clip"])
            image = self.model.normalize_depth(
                depth_tensor.half().float(), self.camera["far_clip"]
            )
            image = ((image + 0.5) * 255).round().clamp(0, 255) / 255 - 0.5
            # Same clean prop33 as xz1_wm_prop: no phase, last action, or
            # privileged linear velocity. Current commands are already in frame.
            prop = torch.cat((frames[0, :9], frames[0, 11:35]))[None]
            self.state = self.model.observe_step(
                image,
                prop,
                self.actions.flatten(1),
                torch.tensor([self.tick == 0], device=self.device),
                self.state,
            )
            self.feature = self.state["deter"]
            if not all(torch.isfinite(v).all() for v in self.state.values()):
                raise FloatingPointError("Non-finite World Model state")
            self.image_updates += 1
        obs = TensorDict(
            {"actor": frames.flip(0)[None], "world_model_feature": self.feature},
            batch_size=[1],
        )
        action = self.actor(obs).clamp(-10, 10).cpu().numpy()[0].copy()
        if not np.isfinite(action).all():
            raise FloatingPointError("Non-finite student action")
        self._awaiting_advance = True
        return action

    @torch.no_grad()
    def advance(self, executed: np.ndarray) -> None:
        if not self._awaiting_advance:
            raise RuntimeError("advance requires one pending policy step")
        executed = np.asarray(executed, dtype=np.float32)
        if executed.shape != (12,) or not np.isfinite(executed).all():
            raise ValueError("Expected twelve finite executed policy actions")
        self.last_action = np.clip(executed, -10, 10).copy()
        self.actions = torch.roll(self.actions, -1, dims=1)
        self.actions[:, -1] = torch.as_tensor(self.last_action, device=self.device)
        self.tick += 1
        self._awaiting_advance = False
