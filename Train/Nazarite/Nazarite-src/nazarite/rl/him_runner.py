"""Nazarite HIM runner with checkpoint and ONNX metadata contracts."""

from __future__ import annotations

import inspect
import os
import tempfile
import warnings

import onnx
import torch

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.rl.runner import MjlabOnPolicyRunner
from rsl_rl.runners.him_on_policy_runner import HIMOnPolicyRunner as _HIMOnPolicyRunner
from rsl_rl.utils.numerics import tensor_items


def _single_file_onnx_kwargs() -> dict:
  try:
    parameters = inspect.signature(torch.onnx.export).parameters
  except (TypeError, ValueError):
    return {}
  if "external_data" in parameters:
    return {"external_data": False}
  if "use_external_data_format" in parameters:
    return {"use_external_data_format": False}
  return {}


class HIMOnPolicyRunner(MjlabOnPolicyRunner, _HIMOnPolicyRunner):
  """Add MjLab environment persistence and metadata to the HIM runner."""

  env: RslRlVecEnvWrapper

  def _numerical_contract(self) -> dict:
    actor = self.alg.get_policy()
    return {
      "version": 1,
      "action_clip": actor.action_clip,
      "observation_clip": actor.observation_clip,
      "action_observation_slice": list(actor.action_observation_slice or ()),
      "history_size": actor.history_size,
      "frame_size": actor.num_one_step_obs,
      "export_history_order": "frame_major_current_first",
      "last_action": "clipped_policy_action_before_joint_scale_and_offset",
    }

  def _validate_numerics(self) -> None:
    actor = self.alg.get_policy()
    wrapper_clip = getattr(self.env, "clip_actions", actor.action_clip)
    if wrapper_clip != actor.action_clip:
      raise ValueError(f"Environment action clip {wrapper_clip} differs from policy {actor.action_clip}")
    for label, model in (("actor", actor), ("critic", self.alg.critic)):
      for name, tensor in model.state_dict().items():
        if not torch.isfinite(tensor).all():
          raise ValueError(f"Invalid {label} tensor: {name}")
      normalizer = getattr(model, "obs_normalizer", None)
      if (
        normalizer is not None
        and hasattr(normalizer, "std")
        and ((normalizer._var < 0).any() or (normalizer._std < 0).any() or normalizer.count < 0)
      ):
        raise ValueError(f"Invalid {label} normalizer statistics")
    for label in ("optimizer", "estimator"):
      owner = getattr(self.alg, label, None)
      optimizer = getattr(owner, "optimizer", owner)
      for name, tensor in tensor_items(getattr(optimizer, "state", {})):
        if not torch.isfinite(tensor).all():
          raise ValueError(f"Invalid {label} optimizer state: {name}")

  def load(self, path, load_cfg=None, strict=True, map_location=None):
    infos = super().load(path, load_cfg, strict, map_location)
    saved_contract = (infos or {}).get("him_numerics")
    if saved_contract is not None and saved_contract != self._numerical_contract():
      raise ValueError("HIM checkpoint numerical contract differs from current configuration")
    self._validate_numerics()
    return infos

  def export_policy_to_onnx(self, path: str, filename: str = "policy.onnx", verbose: bool = False) -> None:
    self._validate_numerics()
    onnx_model = self.alg.get_policy().as_onnx(verbose=verbose)
    onnx_model.to("cpu")
    onnx_model.eval()
    os.makedirs(path, exist_ok=True)
    input_names = list(onnx_model.input_names)
    output_names = list(onnx_model.output_names)
    fd, temporary_path = tempfile.mkstemp(prefix=".policy_", suffix=".onnx", dir=path)
    os.close(fd)
    try:
      torch.onnx.export(
        onnx_model,
        onnx_model.get_dummy_inputs(),
        temporary_path,
        export_params=True,
        opset_version=18,
        verbose=verbose,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={name: {0: "batch"} for name in (*input_names, *output_names)},
        dynamo=False,
        **_single_file_onnx_kwargs(),
      )
      metadata = get_base_metadata(self.env.unwrapped, "local")
      metadata.update({f"him_{key}": str(value) for key, value in self._numerical_contract().items()})
      attach_metadata_to_onnx(temporary_path, metadata)
      onnx.checker.check_model(temporary_path)
      os.replace(temporary_path, os.path.join(path, filename))
    finally:
      if os.path.exists(temporary_path):
        os.unlink(temporary_path)
    self.alg.get_policy().to(self.device)

  def save(self, path: str, infos: dict | None = None) -> None:
    self._validate_numerics()
    super().save(path, {**(infos or {}), "him_numerics": self._numerical_contract()})
    try:
      self.export_policy_to_onnx(os.path.dirname(path), "policy.onnx")
    except (OSError, RuntimeError, ValueError) as exc:
      warnings.warn(f"HIM ONNX export skipped: {exc}")
