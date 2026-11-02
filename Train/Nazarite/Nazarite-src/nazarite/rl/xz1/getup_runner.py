"""Plain getup PPO runner with immediate curriculum restore and version checks."""

from __future__ import annotations

import torch

from mjlab.rl.exporter_utils import attach_metadata_to_onnx
from mjlab.rl.runner import MjlabOnPolicyRunner
from nazarite.mdp.xz1.getup import SettleRelativeJointPositionAction


class XZ1GetupRunner(MjlabOnPolicyRunner):
    def save(self, path: str, infos: dict | None = None) -> None:
        super().save(path, {**(infos or {}), "xz1_getup_version": 1})
        directory, filename, onnx_path = self._get_export_paths(path)
        try:
            self.export_policy_to_onnx(str(directory), filename)
            env = self.env.unwrapped
            action = env.action_manager.get_term("joint_pos")
            assert isinstance(action, SettleRelativeJointPositionAction)
            robot = env.scene["robot"]
            scale = action.scale
            if isinstance(scale, torch.Tensor):
                scale = scale[0].cpu().tolist()
            # The shared velocity exporter assumes nominal-pose offsets. Store
            # this port's relative control contract explicitly for deployment.
            attach_metadata_to_onnx(
                str(onnx_path),
                {
                    "task": "Nazarite-FRNet-Recovery-XZ1",
                    "policy_family": "playground_getup_ppo_v1",
                    "joint_names": list(robot.joint_names),
                    "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
                    "observation_names": list(
                        env.observation_manager.active_terms["actor"]
                    ),
                    "action_scale": scale,
                    "action_semantics": "q_current + scale * action - encoder_bias",
                    "target_update": "every_physics_substep",
                    "control_dt": env.step_dt,
                    "physics_dt": env.physics_dt,
                },
            )
        except (OSError, RuntimeError, ValueError, AssertionError, ImportError) as error:
            print(f"[WARN] Getup ONNX export failed (training continues): {error!r}")

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        if any(
            key.startswith("mcp_encoder.")
            for key in checkpoint.get("actor_state_dict", {})
        ):
            raise ValueError(
                "This is a legacy XZ1 FR-Net checkpoint. Replay it with task "
                "Nazarite-FRNet-Paper-Recovery-XZ1. The Playground getup port "
                "uses a different actor and action semantics; train it from scratch."
            )
        del checkpoint
        infos = super().load(path, load_cfg, strict, map_location)
        # common_step_counter is restored by MjlabOnPolicyRunner. Apply its
        # schedules now, before the next rollout rather than the next timeout.
        self.env.unwrapped.curriculum_manager.compute()
        return infos
