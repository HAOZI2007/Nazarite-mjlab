"""XZ1 action distillation and the shared stateful visual inference contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch
from tensordict import TensorDict

from nazarite.config.train_config.XZ1_env_cfgs.collect_env_cfg import (
    XZ1DepthCfg,
    attach_xz1_depth,
)
from nazarite.config.train_config.XZ1_env_cfgs.teacher_env_cfg import (
    Nazarite_WMP_HIM_Eval_XZ1,
)
from nazarite.config.train_config.XZ1_env_cfgs.teacher_rl_cfg import (
    xz1_teacher_runner_cfg,
)
from nazarite.rl.wmp.actor import WMPHIMActorModel
from nazarite.rl.wmp.world_model import WMPWorldModel, WMPWorldModelCfg
from nazarite.rl.wmp.xz1_dataset import CONTRACT, sanitize_xz1_depth, xz1_wm_prop
from nazarite.rl.xz1_teacher import XZ1TerrainTeacherActor

STUDENT_FORMAT = "xz1_wmp_action_student_v1"


def student_eval_cfg(checkpoint, **kwargs):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("format") != STUDENT_FORMAT or saved.get("contract") != CONTRACT:
        raise ValueError("Not a compatible XZ1 visual student checkpoint")
    cfg = Nazarite_WMP_HIM_Eval_XZ1(**kwargs)
    # No teacher scan is even generated during student evaluation/play.
    cfg.observations.pop("teacher_scan")
    cfg.scene.sensors = tuple(
        sensor for sensor in cfg.scene.sensors if sensor.name != "teacher_terrain_scan"
    )
    return attach_xz1_depth(cfg, XZ1DepthCfg(**saved["camera"]))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def actor_kwargs():
    cfg = asdict(xz1_teacher_runner_cfg().actor)
    cfg.pop("class_name")
    cfg.pop("scan_group")
    return cfg


class XZ1VisualStudentActor(WMPHIMActorModel):
    """Only actor history and WM deter enter this actor; no oracle scan."""

    def as_jit(self):
        raise NotImplementedError("Export requires the complete camera/RSSM runtime")

    def as_onnx(self, verbose=False):
        raise NotImplementedError("Export requires the complete camera/RSSM runtime")


def build_student(obs, device, kwargs=None):
    actor = XZ1VisualStudentActor(
        obs, {"actor": ["actor"]}, "actor", 12, **(kwargs or actor_kwargs())
    ).to(device)
    actor.requires_grad_(False)
    actor.mlp.requires_grad_(True)
    actor.visual_encoder.requires_grad_(True)
    # Inference normalization is deliberately frozen, including its buffers.
    return actor.eval()


def load_teacher(obs, checkpoint, device):
    source = torch.load(checkpoint, map_location=device, weights_only=False)
    contract = (source.get("infos") or {}).get("him_numerics", {})
    if contract.get("frame_size") != 47 or contract.get("history_size") != 6:
        raise ValueError("Teacher requires the XZ1 47 x 6 HIM contract")
    teacher = XZ1TerrainTeacherActor(
        obs, {"actor": ["actor"]}, "actor", 12, **actor_kwargs()
    ).to(device)
    expected = {
        "version": 1,
        "action_clip": teacher.action_clip,
        "observation_clip": teacher.observation_clip,
        "action_observation_slice": list(teacher.action_observation_slice or ()),
        "history_size": teacher.history_size,
        "frame_size": teacher.num_one_step_obs,
        "export_history_order": "frame_major_current_first",
        "last_action": "clipped_policy_action_before_joint_scale_and_offset",
        "teacher_scan": {
            "points": 132,
            "latent": 32,
            "offset": 0.3,
            "clip": [-1.0, 1.0],
            "order": "x_major_y_minor",
            "privileged_not_camera": True,
        },
    }
    if contract != expected:
        raise ValueError("Teacher numerical/scan contract differs")
    teacher.load_state_dict(source["actor_state_dict"], strict=True)
    if not all(torch.isfinite(v).all() for v in teacher.state_dict().values()):
        raise ValueError("Non-finite teacher weights")
    return teacher.requires_grad_(False).eval()


def warm_start_student(student, teacher):
    source = teacher.state_dict()
    common = {k: v for k, v in source.items() if not k.startswith("scan_encoder.")}
    target = student.state_dict()
    expected = {k for k in target if not k.startswith("visual_encoder.")}
    if set(common) != expected:
        raise ValueError("Teacher/student common parameter contract differs")
    student.load_state_dict({**target, **common}, strict=True)


def load_frozen_world_model(checkpoint, device, teacher_sha256=None):
    source = torch.load(checkpoint, map_location=device, weights_only=True)
    if source.get("world_model_implementation") != WMPWorldModel.implementation:
        raise ValueError("Student requires the original WMP Dreamer implementation")
    if source.get("contract") != CONTRACT:
        raise ValueError("World Model XZ1 data contract differs")
    cfg = WMPWorldModelCfg(**source["model_cfg"])
    if (cfg.action_chunk, cfg.prop_dim, cfg.deter_dim) != (5, 33, 512):
        raise ValueError("Expected action60, prop33, deter512 World Model")
    manifest_path = Path(source["dataset"]) / "manifest.json"
    if file_sha256(manifest_path) != source["manifest_sha256"]:
        raise ValueError("World Model dataset manifest has changed")
    manifest = json.loads(manifest_path.read_text())
    if (
        teacher_sha256 is not None
        and manifest["metadata"]["teacher_sha256"] != teacher_sha256
    ):
        raise ValueError("Teacher differs from the frozen WM dataset teacher")
    camera_dict = manifest["metadata"]["camera"]
    camera_dict = {
        **camera_dict,
        "pos": tuple(camera_dict["pos"]),
        "quat": tuple(camera_dict["quat"]),
    }
    camera = XZ1DepthCfg(**camera_dict)
    model = WMPWorldModel(cfg, device=device)
    model.load_state_dict(source["world_model"], strict=True)
    model.requires_grad_(False).eval()
    return model, camera


class XZ1VisualPolicy:
    """50 Hz actor, 10 Hz posterior, five preceding *executed* commands.

    Call advance once per physics/control step and reset after manual resets.
    Acquisition is forced at each due timestamp, matching the dataset recorder.
    """

    def __init__(self, env, actor, model, camera, vision="normal"):
        if abs(env.step_dt - 0.02) > 1e-8 or camera.interval != 5:
            raise ValueError("Visual student requires policy50Hz/camera10Hz")
        if vision not in ("normal", "far", "shuffled"):
            raise ValueError("Unknown vision ablation")
        self.env, self.actor, self.model, self.camera = env, actor, model, camera
        self.vision = vision
        self.ticks = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.actions = torch.zeros(env.num_envs, 5, 12, device=env.device)
        self.feature = torch.zeros(env.num_envs, 512, device=env.device)
        self.state = model.dynamics.initial(env.num_envs)
        self.pending = torch.ones_like(self.ticks, dtype=torch.bool)

    @torch.no_grad()
    def observations(self, obs):
        ids = (self.pending & (self.ticks % 5 == 0)).nonzero().flatten()
        if ids.numel():
            self.env.sim.sense(force_render=True)
            depth = self.env.scene["xz1_depth_camera"].data.depth
            if depth is None or depth.shape != (self.env.num_envs, 64, 64, 1):
                raise RuntimeError("Invalid XZ1 depth frame")
            depth = sanitize_xz1_depth(depth[ids], self.camera.far_clip)
            if self.vision == "far":
                depth = torch.full_like(depth, self.camera.far_clip)
            elif self.vision == "shuffled":
                # Shuffle pixels, works even with one evaluation environment.
                depth = depth.flatten(1)[:, torch.randperm(4096, device=depth.device)]
                depth = depth.reshape(-1, 64, 64, 1)
            # Match offline float16 storage and uint8 image preprocessing exactly.
            depth = depth.half().float()
            image = self.model.normalize_depth(depth, self.camera.far_clip)
            image = ((image + 0.5) * 255).round().clamp(0, 255) / 255 - 0.5
            state = self.model.observe_step(
                image,
                xz1_wm_prop(obs["estimator"])[ids],
                self.actions[ids].flatten(1),
                self.ticks[ids] == 0,
                {k: v[ids] for k, v in self.state.items()},
            )
            for key, value in state.items():
                self.state[key][ids] = value
            self.feature[ids] = state["deter"]
        self.pending[:] = False
        return TensorDict(
            {
                "actor": obs["actor"].detach().clone(),
                "world_model_feature": self.feature.detach().clone(),
            },
            batch_size=[self.env.num_envs],
        )

    @torch.no_grad()
    def __call__(self, obs):
        return self.actor(self.observations(obs))

    def advance(self, executed):
        self.actions = torch.roll(self.actions, -1, dims=1)
        self.actions[:, -1] = executed.detach().clamp(-10, 10)
        self.ticks += 1
        self.pending[:] = True

    def reset(self, dones):
        self.ticks[dones] = 0
        self.actions[dones] = 0
        self.feature[dones] = 0
        self.pending[dones] = True
        # obs_step's is_first replaces the previous recurrent state, including
        # the learned initial state. No stale action can cross this boundary.


def action_distillation_step(actor, optimizer, observations, labels, max_grad_norm=1.0):
    prediction = actor(observations)
    # Extreme Parkour PPO.update_depth_actor: mean vector L2, not squared MSE.
    loss = (labels.detach() - prediction).norm(p=2, dim=-1).mean()
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite action distillation loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(
        [p for p in actor.parameters() if p.requires_grad],
        max_grad_norm,
        error_if_nonfinite=True,
    )
    optimizer.step()
    return float(loss.detach()), float(norm)


def load_student_policy(env, obs, checkpoint, device, vision="normal"):
    source = torch.load(checkpoint, map_location=device, weights_only=True)
    if source.get("format") != STUDENT_FORMAT or source.get("contract") != CONTRACT:
        raise ValueError("Not a compatible XZ1 visual student checkpoint")
    wm_path = source["world_model_path"]
    if file_sha256(wm_path) != source["world_model_sha256"]:
        raise ValueError("Frozen World Model checkpoint has changed")
    model, camera = load_frozen_world_model(wm_path, device)
    actor = build_student(obs, device, source["actor_kwargs"])
    actor.load_state_dict(source["actor_state_dict"], strict=True)
    if asdict(camera) != source["camera"]:
        raise ValueError("Student camera differs from the frozen WM camera")
    if not all(torch.isfinite(v).all() for v in actor.state_dict().values()):
        raise ValueError("Non-finite student weights")
    return XZ1VisualPolicy(env, actor, model, camera, vision)
