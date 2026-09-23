"""PPO configuration for the isolated WTW+DELTA residual task."""

from pathlib import Path

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def _default_wtw_checkpoint() -> str | None:
  repo_root = Path(__file__).resolve().parents[4]
  # This is the latest stable flat-ground WTW run.  Keep the older checkpoint
  # as a fallback so a clean checkout without the new log can still load.
  candidates = (
    repo_root / "logs/rsl_rl/go2_flat_wtw_independent/2026-09-19_23-27-56/model_14999.pt",
    repo_root / "logs/rsl_rl/go2_flat_wtw_independent/2026-09-11_20-39-13/model_14950.pt",
  )
  return next((str(path) for path in candidates if path.is_file()), None)


def wtw_delta_go2_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  common = {
    "class_name": "nazarite.delta.wtw_delta_model:WtwDeltaResidualModel",
    "hidden_dims": (512, 256, 128),
    "obs_normalization": False,
    "prior_group": "wtw_proprio",
    "delta_map_group": "delta_map",
    "delta_proprio_group": "delta_proprio",
    "map_height": 16,
    "map_width": 26,
    "map_channels": 4,
    "map_extent": (2.5, 1.6),
    "map_center": (1.25, 0.0),
    "map_confidence_target": 0.45,
    "map_confidence_floor": 0.50,
    "forward_x_threshold": 0.0,
    "residual_activation": "softsign",
    "residual_hidden_dims": (128, 64),
    # Give DELTA enough action budget to clear obstacles while preserving a
    # short warm-up around the pretrained WTW prior.
    "residual_scale": 0.25,
    "residual_scale_start": 0.05,
    "residual_scale_ramp_iters": 1_500,
    # sigmoid(-0.5) starts near 0.38 open, instead of the old sigmoid(-2)
    # gate which suppressed almost every terrain correction.
    "residual_gate_bias": -0.5,
    "wtw_checkpoint": _default_wtw_checkpoint(),
    "freeze_wtw": True,
  }
  actor = {
    **common,
    "distribution_cfg": {
      "class_name": "GaussianDistribution",
      # Used only when a checkpoint has no distribution state. The configured
      # WTW checkpoint normally restores its learned per-joint std instead.
      "init_std": 0.35,
      "std_type": "log",
      # Prevent exploration noise from destroying the pretrained trot gait.
      "std_range": (0.15, 0.38),
    },
    # Hip corrections are deliberately smaller than thigh/calf corrections;
    # terrain adaptation should primarily change swing height and foot placement.
    "residual_joint_scales": (
      0.35, 0.80, 1.00,
      0.35, 0.80, 1.00,
      0.35, 0.80, 1.00,
      0.35, 0.80, 1.00,
    ),
  }
  critic = {
    **common,
    "prior_group": "critic_privileged",
    "delta_proprio_group": "delta_proprio_critic",
    # The privileged critic has a different input dimension and is learned
    # from scratch; the WTW actor checkpoint is only used by the actor prior.
    "wtw_checkpoint": None,
    "freeze_wtw": False,
  }
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(**actor),
    critic=RslRlModelCfg(**critic),
    obs_groups={
      "actor": ("wtw_proprio", "delta_proprio", "delta_map"),
      "critic": ("critic_privileged", "delta_proprio_critic", "delta_map"),
    },
    algorithm=RslRlPpoAlgorithmCfg(
      learning_rate=2.0e-4,
      schedule="fixed",
      num_learning_epochs=5,
      num_mini_batches=4,
      gamma=0.99,
      lam=0.95,
      clip_param=0.2,
      entropy_coef=0.005,
    ),
    experiment_name="go2_wtw_delta_residual",
    num_steps_per_env=24,
    max_iterations=30_000,
    save_interval=50,
    # The residual is bounded independently; this is a final numerical guard
    # around the complete stochastic action sent to the environment.
    clip_actions=5.0,
  )
