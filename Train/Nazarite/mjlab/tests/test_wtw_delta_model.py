import torch
from nazarite.delta.critic_model import DeltaPrivilegedCriticModel
from nazarite.delta.direct_action_model import WtwDeltaDirectActionModel
from nazarite.delta.wtw_delta_model import WtwDeltaResidualModel
from tensordict import TensorDict


def _obs() -> TensorDict:
  return TensorDict(
    {
      "wtw_proprio": torch.zeros(2, 498),
      "delta_proprio": torch.zeros(2, 45),
      "delta_map": torch.zeros(2, 16 * 26 * 3),
    },
    batch_size=[2],
  )


def test_wtw_delta_residual_starts_as_prior():
  obs = _obs()
  groups = {"actor": ("wtw_proprio", "delta_proprio", "delta_map")}
  model = WtwDeltaResidualModel(
    obs,
    groups,
    "actor",
    12,
    hidden_dims=(64, 32),
    prior_group="wtw_proprio",
    delta_proprio_group="delta_proprio",
    delta_map_group="delta_map",
    distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
  )
  with torch.no_grad():
    prior = model.wtw_policy(obs["wtw_proprio"])
    output = model(obs)
  assert output.shape == (2, 12)
  assert torch.allclose(output, prior)


def test_wtw_delta_residual_is_trainable_but_prior_can_be_frozen():
  obs = _obs()
  groups = {"actor": ("wtw_proprio", "delta_proprio", "delta_map")}
  model = WtwDeltaResidualModel(
    obs,
    groups,
    "actor",
    12,
    hidden_dims=(64, 32),
    prior_group="wtw_proprio",
    delta_proprio_group="delta_proprio",
    delta_map_group="delta_map",
    freeze_wtw=True,
    distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
  )
  assert not any(parameter.requires_grad for parameter in model.wtw_policy.parameters())
  assert any(parameter.requires_grad for parameter in model.delta.parameters())
  assert any(parameter.requires_grad for parameter in model.residual_head.parameters())


def test_wtw_delta_residual_is_bounded():
  obs = _obs()
  groups = {"actor": ("wtw_proprio", "delta_proprio", "delta_map")}
  model = WtwDeltaResidualModel(
    obs,
    groups,
    "actor",
    12,
    hidden_dims=(64, 32),
    prior_group="wtw_proprio",
    delta_proprio_group="delta_proprio",
    delta_map_group="delta_map",
    residual_scale=0.1,
  )
  with torch.no_grad():
    model.residual_head[-1].bias.fill_(1.0e6)
    prior = model.wtw_policy(obs["wtw_proprio"])
    output = model(obs)
  correction = output - prior
  assert torch.all(correction <= 0.1 + 1.0e-6)
  assert torch.all(correction >= -0.1 - 1.0e-6)


def test_wtw_delta_supports_bev_validity_channel_and_gate():
  obs = TensorDict(
    {
      "wtw_proprio": torch.zeros(2, 498),
      "delta_proprio": torch.zeros(2, 45),
      "delta_map": torch.zeros(2, 16 * 26 * 4),
    },
    batch_size=[2],
  )
  model = WtwDeltaResidualModel(
    obs,
    {"actor": ("wtw_proprio", "delta_proprio", "delta_map")},
    "actor",
    12,
    hidden_dims=(64, 32),
    map_height=16,
    map_width=26,
    map_channels=4,
    prior_group="wtw_proprio",
    delta_proprio_group="delta_proprio",
    delta_map_group="delta_map",
    residual_joint_scales=(0.35, 0.8, 1.0) * 4,
  )
  output = model(obs)
  assert output.shape == (2, 12)
  assert model.get_diagnostics()["DELTA/residual_gate_mean"] < 0.2


def test_wtw_delta_accepts_behavior_and_phase_context():
  """The terrain branch must receive gait context in addition to proprioception."""
  obs = TensorDict(
    {
      "wtw_proprio": torch.zeros(2, 498),
      # 45 base proprio + 8 behavior + 8 sin/cos phase.
      "delta_proprio": torch.zeros(2, 61),
      "delta_map": torch.zeros(2, 16 * 26 * 4),
    },
    batch_size=[2],
  )
  model = WtwDeltaResidualModel(
    obs,
    {"actor": ("wtw_proprio", "delta_proprio", "delta_map")},
    "actor",
    12,
    hidden_dims=(64, 32),
    map_height=16,
    map_width=26,
    map_channels=4,
    prior_group="wtw_proprio",
    delta_proprio_group="delta_proprio",
    delta_map_group="delta_map",
    residual_scale=0.25,
    residual_scale_start=0.05,
    residual_scale_ramp_iters=1500,
    residual_gate_bias=-0.5,
  )
  assert model.delta_proprio_dim == 61
  model.set_training_iteration(0)
  assert model.residual_scale == 0.05
  model.set_training_iteration(1500)
  assert model.residual_scale == 0.25


def test_wtw_delta_scales_residual_with_bev_confidence():
  obs = TensorDict(
    {
      "wtw_proprio": torch.zeros(2, 498),
      "delta_proprio": torch.zeros(2, 61),
      "delta_map": torch.zeros(2, 16 * 26 * 4),
    },
    batch_size=[2],
  )
  model = WtwDeltaResidualModel(
    obs,
    {"actor": ("wtw_proprio", "delta_proprio", "delta_map")},
    "actor",
    12,
    hidden_dims=(64, 32),
    map_height=16,
    map_width=26,
    map_channels=4,
    prior_group="wtw_proprio",
    delta_proprio_group="delta_proprio",
    delta_map_group="delta_map",
    residual_scale=0.25,
    map_confidence_target=0.45,
    map_confidence_floor=0.50,
  )
  with torch.no_grad():
    model.residual_head[-1].bias.fill_(1.0)
    output = model(obs)
  prior = model.wtw_policy(obs["wtw_proprio"])
  # Empty maps use the configured floor rather than disabling gradients.
  correction = (output - prior).abs()
  assert torch.all(correction <= 0.125 + 1.0e-5)
  assert model.get_diagnostics()["DELTA/map_confidence"] == 0.5


def test_wtw_checkpoint_restores_distribution_std(tmp_path):
  obs = _obs()
  groups = {"actor": ("wtw_proprio", "delta_proprio", "delta_map")}
  source = WtwDeltaResidualModel(
    obs,
    groups,
    "actor",
    12,
    hidden_dims=(64, 32),
    prior_group="wtw_proprio",
    delta_proprio_group="delta_proprio",
    delta_map_group="delta_map",
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 0.4,
      "std_type": "log",
    },
  )
  expected_log_std = torch.linspace(-1.4, -0.8, 12)
  checkpoint_state = {
    f"mlp.{key}": value.detach().clone()
    for key, value in source.wtw_policy.state_dict().items()
  }
  checkpoint_state["distribution.log_std_param"] = expected_log_std
  checkpoint = tmp_path / "wtw.pt"
  torch.save({"actor_state_dict": checkpoint_state}, checkpoint)

  restored = WtwDeltaResidualModel(
    obs,
    groups,
    "actor",
    12,
    hidden_dims=(64, 32),
    prior_group="wtw_proprio",
    delta_proprio_group="delta_proprio",
    delta_map_group="delta_map",
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "log",
    },
    wtw_checkpoint=str(checkpoint),
  )
  assert torch.allclose(restored.distribution.log_std_param, expected_log_std)


def test_wtw_delta_direct_action_starts_as_prior_and_uses_delta_latent():
  obs = TensorDict(
    {
      "wtw_proprio": torch.zeros(2, 498),
      "delta_proprio": torch.zeros(2, 61),
      "delta_map": torch.zeros(2, 16 * 26 * 5),
    },
    batch_size=[2],
  )
  model = WtwDeltaDirectActionModel(
    obs,
    {"actor": ("wtw_proprio", "delta_proprio", "delta_map")},
    "actor",
    12,
    hidden_dims=(64, 32),
    map_height=16,
    map_width=26,
    map_channels=5,
    map_extent=(2.5, 1.6),
    map_center=(1.25, 0.0),
    prior_group="wtw_proprio",
    delta_proprio_group="delta_proprio",
    delta_map_group="delta_map",
    distribution_cfg=None,
  )
  with torch.no_grad():
    prior = model.wtw_policy(obs["wtw_proprio"])
    output = model(obs)
  assert torch.allclose(output, prior)
  assert not any(parameter.requires_grad for parameter in model.wtw_policy.parameters())
  assert any(parameter.requires_grad for parameter in model.action_fusion.parameters())

  # The zero output initialization intentionally ignores DELTA on the first
  # rollout. Open a terrain-latent path to verify that the learned correction
  # changes the final action once PPO has updated the fusion MLP.
  with torch.no_grad():
    first = model.action_fusion[0]
    output_layer = model.action_fusion[-1]
    assert isinstance(first, torch.nn.Linear)
    assert isinstance(output_layer, torch.nn.Linear)
    first.weight[:, model.output_dim + model.delta_proprio_dim] = 1.0
    output_layer.weight[:, 0] = 1.0
  changed = obs.clone()
  changed["delta_map"] = torch.randn_like(changed["delta_map"])
  with torch.no_grad():
    changed_output = model(changed)
  assert not torch.allclose(changed_output, output)
  assert "DELTA/direct_action_delta_ratio" in model.get_diagnostics()


def test_delta_privileged_critic_encodes_clean_map() -> None:
  obs = TensorDict(
    {
      "critic_privileged": torch.zeros(2, 120),
      "delta_proprio_critic": torch.zeros(2, 61),
      "delta_privileged_map": torch.zeros(2, 16 * 26 * 5),
    },
    batch_size=[2],
  )
  critic = DeltaPrivilegedCriticModel(
    obs,
    {
      "critic": (
        "critic_privileged",
        "delta_proprio_critic",
        "delta_privileged_map",
      )
    },
    "critic",
    1,
    hidden_dims=(64, 32),
  )

  value = critic(obs)
  value.square().mean().backward()

  assert value.shape == (2, 1)
  assert any(parameter.grad is not None for parameter in critic.delta.parameters())
