"""RI-4438-style terminal supervision for XZ1 HIM automatic resets."""

from __future__ import annotations

import gc
from copy import deepcopy
from dataclasses import replace

import mujoco
import mujoco_warp as mjwarp
import torch
import warp as wp
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from nazarite.mdp.him.numerics import checked_observation
from nazarite.mdp.shared.observations import phase

ESTIMATOR_TERMS = (
    "base_ang_vel",
    "projected_gravity",
    "command",
    "phase",
    "joint_pos",
    "joint_vel",
    "actions",
    "base_lin_vel",
)


def configure_estimator_observations(cfg: ManagerBasedRlEnvCfg, *, play: bool) -> None:
    """Expose the first 50 critic features without history or terrain scans.

    Actor/critic layouts and HIM target slices remain unchanged. Only the
    independent supervision group is stored as next-step estimator input.
    """
    terms = {
        name: replace(
            deepcopy(cfg.observations["critic"].terms[name]),
            history_length=0,
            delay_min_lag=0,
            delay_max_lag=0,
            delay_hold_prob=0.0,
            delay_update_period=0,
            noise=None,
        )
        for name in ESTIMATOR_TERMS
    }
    for name, term in terms.items():
        if term.func is checked_observation:
            term.params["numerics_tag"] = f"estimator/{name}"
    cfg.observations["estimator"] = ObservationGroupCfg(
        terms=terms,
        concatenate_terms=True,
        enable_corruption=False,
        history_length=0,
    )
    if not play:
        cfg.recorders["him_terminal"] = RecorderTermCfg(func=TerminalEstimatorRecorder)


class TerminalEstimatorRecorder(RecorderTerm):
    """Save fresh terminal targets before reset using only kinematics kernels.

    Integration and interval pushes leave derived sensor data stale by one
    physics step. Refreshing position/velocity kernels resolves that without
    collision solving or raycasting. The regular post-reset forward/sense
    pipeline supplies the observations used for the next policy action.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._refresh_graph = None
        self._source_graph = None
        self._recorded = False
        group = env.observation_manager.cfg["estimator"]
        if group.history_length or any(
            term.history_length or term.delay_max_lag for term in group.terms.values()
        ):
            raise ValueError(
                "Terminal estimator observations must have no history or delay"
            )

        # Custom observation terms may require more than position/velocity
        # sensors. Fall back to a full refresh rather than save stale targets.
        supported = {
            mdp.builtin_sensor,
            mdp.projected_gravity,
            mdp.generated_commands,
            mdp.joint_pos_rel,
            mdp.joint_vel_rel,
            mdp.last_action,
            phase,
        }

        def source(term):
            return (
                term.params["source_func"]
                if term.func is checked_observation
                else term.func
            )

        self._kinematics_only = all(
            source(term) in supported for term in group.terms.values()
        )
        for term in group.terms.values():
            if source(term) is mdp.builtin_sensor:
                sensor = env.sim.mj_model.sensor(term.params["sensor_name"])
                if sensor.type[0] not in (
                    mujoco.mjtSensor.mjSENS_GYRO,
                    mujoco.mjtSensor.mjSENS_VELOCIMETER,
                    # XZ1's existing IMU linear velocity is a frame sensor;
                    # sensor_vel refreshes it with the same kinematic kernels.
                    mujoco.mjtSensor.mjSENS_FRAMELINVEL,
                    mujoco.mjtSensor.mjSENS_FRAMEANGVEL,
                ):
                    self._kinematics_only = False

    def _refresh_kinematics(self) -> None:
        sim = self._env.sim
        mjwarp.kinematics(sim.wp_model, sim.wp_data)
        mjwarp.com_pos(sim.wp_model, sim.wp_data)
        mjwarp.com_vel(sim.wp_model, sim.wp_data)
        mjwarp.sensor_vel(sim.wp_model, sim.wp_data)

    def _refresh_terminal_state(self) -> None:
        sim = self._env.sim
        if not self._kinematics_only:
            sim.forward()
            sim.sense()
            return
        with wp.ScopedDevice(sim.wp_device):
            if not sim.use_cuda_graph:
                self._refresh_kinematics()
                return
            # Domain randomization may replace GPU arrays and recapture the
            # simulation graphs; rebuild our graph when those pointers change.
            if self._refresh_graph is None or self._source_graph is not sim.step_graph:
                self._refresh_graph = None
                self._source_graph = sim.step_graph
                gc_enabled = gc.isenabled()
                gc.disable()
                try:
                    with wp.ScopedCapture(device=sim.wp_device) as capture:
                        self._refresh_kinematics()
                    self._refresh_graph = capture.graph
                finally:
                    if gc_enabled:
                        gc.enable()
            assert self._refresh_graph is not None
            wp.capture_launch(self._refresh_graph)

    def record_pre_reset(self, env_ids: torch.Tensor) -> None:
        env = self._env
        self._refresh_terminal_state()
        # Match the manual-reset terminal command update. Reset subsequently
        # resamples these timers, then the framework advances them with dt=0.
        env.command_manager.compute(dt=env.step_dt, env_ids=env_ids)
        terminal = env.observation_manager.compute_group("estimator")
        assert isinstance(terminal, torch.Tensor)
        env.extras["terminal_observations"] = TensorDict(
            {"estimator": terminal[env_ids].clone()},
            batch_size=[env_ids.numel()],
        )
        self._recorded = True

    def record_post_reset(self, env_ids: torch.Tensor) -> None:
        del env_ids
        # Explicit reset() is not preceded by record_pre_reset().
        if not self._recorded:
            self._env.extras.pop("terminal_observations", None)

    def record_post_step(self) -> None:
        if not self._recorded:
            self._env.extras.pop("terminal_observations", None)
        self._recorded = False
