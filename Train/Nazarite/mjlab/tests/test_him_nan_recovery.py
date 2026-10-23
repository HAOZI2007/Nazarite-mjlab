"""The HIM runner recovers per-environment MuJoCo blow-ups instead of aborting.

One environment per 10^8 env-steps can produce a NaN state (a limb wedged between
two box corners wedges the solver into an unrecoverable step). The runner used to
let ``TrainingNumericsGuard`` raise on that observation and kill the whole run;
it now abandons those episodes and keeps training.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from rsl_rl.runners.him_on_policy_runner import HIMOnPolicyRunner
from tensordict import TensorDict


class _FakeEnv:
  """Minimal stand-in for ``RslRlVecEnvWrapper`` around a manager based env."""

  def __init__(self, num_envs: int = 4, obs_dim: int = 3) -> None:
    self.num_envs = num_envs
    self.obs_dim = obs_dim
    self.device = torch.device("cpu")
    self.unwrapped = self
    self.reset_calls: list[list[int]] = []
    self.obs_groups: tuple[str, ...] = ("actor",)
    self._him_numerics_bad = torch.zeros(num_envs, dtype=torch.bool)
    self._him_numerics_evidence = {"actor/base_ang_vel": torch.zeros(num_envs, 2)}

  def reset(self, env_ids: torch.Tensor):
    # ManagerBasedRlEnv.reset returns a plain nested dict, not a TensorDict.
    self.reset_calls.append(env_ids.tolist())
    self._him_numerics_bad.zero_()
    obs = {
      group: {"term": torch.full((self.num_envs, self.obs_dim), 7.0)}
      for group in self.obs_groups
    }
    return obs, {}


def _runner(env: _FakeEnv) -> SimpleNamespace:
  return SimpleNamespace(env=env, _nan_recoveries=0)


def _obs(*groups: str, num_envs: int = 4, dim: int = 3) -> TensorDict:
  """Nested observation layout: groups of per-term tensors, as mjlab emits."""

  return TensorDict(
    {
      name: TensorDict({"term": torch.zeros(num_envs, dim)}, batch_size=[num_envs])
      for name in groups
    },
    batch_size=[num_envs],
  )


def test_finite_step_is_left_alone():
  env = _FakeEnv()
  runner = _runner(env)
  obs = _obs("actor")
  rewards = torch.zeros(4)
  dones = torch.zeros(4, dtype=torch.bool)

  HIMOnPolicyRunner._recover_nonfinite_envs(runner, obs, rewards, dones, {})

  assert env.reset_calls == []
  assert runner._nan_recoveries == 0
  assert not dones.any()


def test_nan_environment_is_reset_and_counted():
  env = _FakeEnv()
  env._him_numerics_bad[2] = True
  runner = _runner(env)
  obs = _obs("actor")
  obs["actor", "term"][2] = torch.tensor([float("nan"), 0.0, 1.0])
  rewards = torch.ones(4)
  dones = torch.zeros(4, dtype=torch.bool)
  extras = {"log": {"Episode_Termination/fell_over": 1.0}}

  HIMOnPolicyRunner._recover_nonfinite_envs(runner, obs, rewards, dones, extras)

  assert env.reset_calls == [[2]]
  assert runner._nan_recoveries == 1
  # The episode is abandoned and the bad state is gone before the numerics guard
  # looks at this step, so it cannot raise.
  assert dones.tolist() == [False, False, True, False]
  assert rewards.tolist() == [1.0, 1.0, 0.0, 1.0]
  assert not torch.isnan(obs["actor", "term"]).any()
  assert obs["actor", "term"][2].tolist() == [7.0, 7.0, 7.0]
  assert not env._him_numerics_bad.any()
  assert not env._him_numerics_evidence["actor/base_ang_vel"].any()
  # reset() clears the shared step log; the runner must keep this step's metrics.
  assert extras["log"] == {"Episode_Termination/fell_over": 1.0}


def test_nan_without_the_manager_flag_is_still_recovered():
  """A NaN reached only through the observation check must count as well."""

  env = _FakeEnv()
  env.obs_groups = ("actor", "critic")
  runner = _runner(env)
  obs = _obs("actor", "critic")
  obs["critic", "term"][1, :] = float("inf")
  rewards = torch.ones(4)
  dones = torch.zeros(4, dtype=torch.bool)

  HIMOnPolicyRunner._recover_nonfinite_envs(runner, obs, rewards, dones, {})

  assert env.reset_calls == [[1]]
  assert runner._nan_recoveries == 1


@pytest.mark.parametrize("env_ids", [[0, 3]])
def test_all_bad_environments_are_handled(env_ids):
  env = _FakeEnv()
  runner = _runner(env)
  obs = _obs("actor")
  obs["actor", "term"][env_ids, 0] = float("nan")
  rewards = torch.ones(4)
  dones = torch.zeros(4, dtype=torch.bool)

  HIMOnPolicyRunner._recover_nonfinite_envs(runner, obs, rewards, dones, {})

  assert env.reset_calls == [env_ids]
  assert runner._nan_recoveries == len(env_ids)


def test_plain_dict_observations_are_recovered_too():
  """The training loop's TensorDict and the env's dict share the same paths."""

  env = _FakeEnv()
  runner = _runner(env)
  obs = {"actor": {"term": torch.zeros(4, 3)}}
  obs["actor"]["term"][3, 1] = float("nan")
  rewards = torch.ones(4)
  dones = torch.zeros(4, dtype=torch.bool)

  HIMOnPolicyRunner._recover_nonfinite_envs(runner, obs, rewards, dones, {})

  assert env.reset_calls == [[3]]
  assert obs["actor"]["term"][3].tolist() == [7.0, 7.0, 7.0]


def test_missing_reset_path_is_zeroed_instead_of_aborting():
  """A container mismatch must not leak a NaN into the rollout buffer."""

  env = _FakeEnv()
  runner = _runner(env)
  obs = _obs("actor", "critic")
  obs["critic", "term"][0, 0] = float("nan")
  rewards = torch.ones(4)
  dones = torch.zeros(4, dtype=torch.bool)

  with pytest.warns(UserWarning, match="did not return observation paths"):
    HIMOnPolicyRunner._recover_nonfinite_envs(runner, obs, rewards, dones, {})

  assert env.reset_calls == [[0]]
  assert torch.isfinite(obs["actor", "term"]).all()
  # "critic" is absent from reset()'s dict in this fake, so it falls back to zeroing.
  assert obs["critic", "term"][0].tolist() == [0.0, 0.0, 0.0]
