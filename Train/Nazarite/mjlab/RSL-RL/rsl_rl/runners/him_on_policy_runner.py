"""On-policy runner for TensorDict HIM training."""

from __future__ import annotations

import os
import time

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import HIMPPO
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.utils.numerics import TrainingNumericsGuard


class HIMOnPolicyRunner(OnPolicyRunner):
    """Collect HIM rollouts while preserving terminal observations."""

    alg: HIMPPO

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        if self.is_distributed:
            self.alg.broadcast_parameters()
        self.logger.init_logging_writer()
        unwrapped = getattr(self.env, "unwrapped", None)
        manual_reset = not getattr(getattr(unwrapped, "cfg", None), "auto_reset", True)
        guard = None
        if self.cfg.get("numerics"):
            guard = TrainingNumericsGuard(self.device, self.logger.log_dir, **self.cfg["numerics"])
            self.alg.numerics_guard = guard

        def check(stage, values):
            if guard is None:
                return
            bad = getattr(unwrapped, "_him_numerics_bad", None)
            evidence = getattr(unwrapped, "_him_numerics_evidence", {})
            guard.check(stage, {"observations": values, "pre_clip": evidence},
                        limit=guard.cfg["raw_observation_abort"], extra_bad=bad)
            if bad is not None:
                bad.zero_()
                for value in evidence.values():
                    value.zero_()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            action_peak = torch.zeros((), device=self.device)
            action_counts = torch.zeros(3, device=self.device)
            with torch.inference_mode():
                for step in range(self.cfg["num_steps_per_env"]):
                    if guard is not None:
                        guard.iteration, guard.step = it, step
                        guard.remember(observations=obs)
                    check("before_action", obs)
                    actions = self.alg.act(obs)
                    if guard is not None:
                        guard.remember(actions=actions)
                        guard.check("raw_action", {"actions": actions, "distribution": self.alg.actor.output_distribution_params})
                        action_peak = torch.maximum(action_peak, actions.abs().max())
                        clip = self.alg.actor.action_clip
                        if clip is not None:
                            action_counts[0] += (actions.abs() > clip).sum()
                        action_counts[1] += actions.numel()
                        action_counts[2] += (actions.abs() > guard.cfg["raw_action_warn"]).sum()
                    next_obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    check("after_step", {"next": next_obs, "terminal": extras.get("terminal_observations")})
                    if guard is not None:
                        guard.check("step_result", {"rewards": rewards, "dones": dones})
                    next_obs, rewards, dones = next_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    step_extras = dict(extras)
                    if manual_reset:
                        step_extras["terminal_observations"] = next_obs
                    self.alg.process_env_step(next_obs, rewards, dones, step_extras)
                    reset_extras = {}
                    if manual_reset:
                        reset_ids = dones.reshape(-1).nonzero(as_tuple=False).flatten()
                        if reset_ids.numel():
                            next_obs, reset_extras = self._reset_done_envs(next_obs, reset_ids)
                        check("after_reset", next_obs)
                    log_extras = dict(step_extras)
                    if isinstance(reset_extras.get("log"), dict):
                        log_extras.setdefault("log", {}).update(reset_extras["log"])
                    self.logger.process_env_step(rewards, dones, log_extras, self.alg.intrinsic_rewards)
                    obs = next_obs
                collect_time = time.time() - start
                start = time.time()
                self.alg.compute_returns(obs)
            loss_dict = self.alg.update()
            if guard is not None:
                if self.is_distributed:
                    torch.distributed.all_reduce(action_peak, op=torch.distributed.ReduceOp.MAX)
                    torch.distributed.all_reduce(action_counts)
                loss_dict["numerics/raw_action_max"] = action_peak.item()
                loss_dict["numerics/action_clip_fraction"] = (action_counts[0] / action_counts[1].clamp_min(1)).item()
                loss_dict["numerics/raw_action_warn_fraction"] = (action_counts[2] / action_counts[1].clamp_min(1)).item()
            self.current_learning_iteration = it
            self.logger.log(it=it, start_it=start_it, total_it=total_it,
                            collect_time=collect_time, learn_time=time.time() - start,
                            loss_dict=loss_dict, learning_rate=self.alg.learning_rate,
                            action_std=self.alg.get_policy().output_std, rnd_weight=None)
            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))
        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
            self.logger.stop_logging_writer()

    def _reset_done_envs(self, terminal_obs: TensorDict, reset_ids: torch.Tensor) -> tuple[TensorDict, dict]:
        unwrapped = getattr(self.env, "unwrapped", None)
        if unwrapped is None or not hasattr(unwrapped, "reset"):
            raise RuntimeError("HIM training requires reset(env_ids=...) when auto_reset=False")
        reset_result = unwrapped.reset(env_ids=reset_ids.to(unwrapped.device))
        reset_obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        reset_extras = reset_result[1] if isinstance(reset_result, tuple) else {}
        reset_obs = TensorDict(reset_obs, batch_size=[self.env.num_envs], device=self.device)
        for key in terminal_obs.keys():
            if key in reset_obs:
                terminal_obs[key][reset_ids] = reset_obs[key].to(self.device)[reset_ids]
        return terminal_obs, reset_extras
