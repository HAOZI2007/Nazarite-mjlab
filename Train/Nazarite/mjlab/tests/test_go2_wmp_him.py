"""Focused tests for the independent Go2 WMP-HIM migration."""

from types import SimpleNamespace

import torch
from nazarite.config.train_config.go2.wmp_him_go2_env_cfg import Nazarite_WMP_HIM_Complex_Terrain_Go2
from nazarite.mdp.wmp.observations import empty_world_model_feature
from nazarite.mdp.wmp.rewards import feet_edge, feet_edge_curriculum_scale
from nazarite.mdp.wmp.terminations import base_contact, fall, forward_course_bounds, reference_time_out, velocity_violation
from nazarite.rl.wmp.actor import WMPHIMActorModel
from nazarite.rl.wmp.replay import WorldModelReplay
from nazarite.rl.wmp.runner import WMPHIMOnPolicyRunner
from nazarite.rl.wmp.world_model import WMPWorldModel, WMPWorldModelCfg
from tensordict import TensorDict

from mjlab.sensor import CameraSensorCfg


def _small_world_model() -> WMPWorldModel:
  return WMPWorldModel(
    WMPWorldModelCfg(
      prop_dim=5,
      action_dim=3,
      cnn_depth=4,
      prop_hidden_dim=16,
      prop_layers=2,
      deter_dim=16,
      stoch_groups=2,
      stoch_classes=4,
      rssm_hidden_dim=16,
      learning_rate=1.0e-3,
      grad_clip=10.0,
    )
  )


def test_wmp_task_has_camera_feature_group_and_expected_reward_weights():
  cfg = Nazarite_WMP_HIM_Complex_Terrain_Go2()
  camera = next(
    sensor for sensor in cfg.scene.sensors if sensor.name == "wmp_depth_camera"
  )
  assert isinstance(camera, CameraSensorCfg)
  assert (camera.width, camera.height, camera.data_types) == (64, 64, ("depth",))
  assert camera.enabled_geom_groups == (0,)
  assert cfg.observations["world_model_feature"].terms["rssm_deter"].params == {
    "feature_dim": 512
  }
  assert cfg.observations["actor"].history_length == 6
  assert cfg.rewards["track_linear_velocity"].weight == 1.5
  assert cfg.rewards["track_angular_velocity"].weight == 0.5
  # The reference's plain single-sided 0.5 s threshold at 0.5 weight: with a
  # 0.6 s gait period it pays for a real leap, not for swing timing.
  assert cfg.rewards["feet_air_time"].weight == 0.5
  assert cfg.rewards["feet_air_time"].func.__name__ == "feet_air_time"
  assert cfg.rewards["feet_air_time"].params["threshold"] == 0.5
  assert cfg.rewards["feet_edge"].weight == -1.0
  assert cfg.rewards["track_linear_velocity"].params["terrain_names"] == (
    "gap",
    "high_wall",
    "single_beam",
  )
  assert cfg.rewards["cheat"].params["terrain_names"] == (
    "low_frame",
    "high_wall",
    "plum_blossom",
    "gap",
    "tire_array",
    "single_beam",
  )
  assert cfg.rewards["feet_edge"].params["curriculum_start_iteration"] == 4000
  assert cfg.rewards["feet_edge"].params["curriculum_end_iteration"] == 10000
  assert "stool_base_height" not in cfg.rewards

  # Heading mode replaces the hard-clamped yaw of the port: the reference
  # samples a heading target and turns the error into a bounded yaw rate.
  twist = cfg.commands["twist"]
  assert twist.heading_command is True
  assert twist.ranges.heading == (-3.14, 3.14)
  assert twist.ranges.ang_vel_z == (-1.0, 1.0)
  assert twist.ranges.lin_vel_x == (-0.6, 0.8)
  assert twist.rel_heading_envs == 0.5

  leg_sensor = next(
    sensor for sensor in cfg.scene.sensors if sensor.name == "leg_ground_contact"
  )
  assert leg_sensor.primary.pattern == (
    r"^(FL|FR|RL|RR)_(thigh|calf)_collision_[0-9]+$"
  )

  play_cfg = Nazarite_WMP_HIM_Complex_Terrain_Go2(play=True)
  play_edge = play_cfg.rewards["feet_edge"]
  assert play_edge.params["curriculum_start_iteration"] is None


def test_wmp_actor_observation_has_no_gait_clock():
  from nazarite.config.train_config.go2.wmp_rl_cfg import go2_wmp_him_runner_cfg

  cfg = Nazarite_WMP_HIM_Complex_Terrain_Go2()
  assert "phase" not in cfg.observations["actor"].terms
  assert "phase" not in cfg.observations["critic"].terms
  assert "phase" not in cfg.observations["estimator"].terms

  runner = go2_wmp_him_runner_cfg()
  # 47 -> 45: two-dimension gait clock removed.
  assert runner.actor.num_one_step_obs == 45
  assert runner.actor.history_term_dims == (3, 3, 3, 12, 12, 12)
  assert runner.actor.action_observation_slice == (33, 45)
  # The estimator group is 48 wide (45 plus base linear velocity), and the
  # targets have to reconstruct exactly the 45-dim one-step observation.
  assert runner.algorithm.estimator_velocity_slice == (45, 48)
  targets = runner.algorithm.estimator_target_slices
  assert sum(stop - start for start, stop in targets) == 45


def test_wmp_task_uses_the_reference_reward_set():
  cfg = Nazarite_WMP_HIM_Complex_Terrain_Go2()

  # Aligned with the reference recipe. The manual shaping terms that used to hold
  # 10-15 % of the budget are gone: measured on 2026-10-19_22-53-44, the four
  # terms paying for a lifted foot were worth 1.5 % against 17.7 % for staying
  # smooth. Style is the discriminator's job now.
  assert set(cfg.rewards) == {
    "track_linear_velocity",
    "track_angular_velocity",
    "joint_torques_l2",
    "joint_acc_l2",
    "action_rate_l2",
    "leg_collision",
    "joint_pos_limits",
    "joint_velocity_limits",
    "lin_vel_z",
    "dof_error",
    "is_terminated",
    "stand_still",
    "feet_air_time",
    "feet_stumble",
    # The reference recipe has no clearance term because its course never asks a
    # foot to go over anything. Ours has risers and walls, and without these two
    # the policy stalls at the first obstacle (measured at level 1: every family
    # covered 0.7-2.1 m of the 3.0 m a clean walk would cover).
    "foot_clearance",
    "base_clearance",
    "feet_edge",
    "cheat",
    "stuck",
  }
  assert cfg.rewards["track_linear_velocity"].weight == 1.5
  assert cfg.rewards["track_angular_velocity"].weight == 0.5
  assert cfg.rewards["joint_torques_l2"].weight == -1.0e-4
  assert cfg.rewards["joint_acc_l2"].weight == -2.5e-7
  assert cfg.rewards["action_rate_l2"].weight == -0.03
  assert cfg.rewards["leg_collision"].weight == -1.0
  assert cfg.rewards["lin_vel_z"].weight == -1.0
  # Replaces the tolerance-banded posture reward: one smooth basin around the
  # default stand, so a rear leg held straight back is charged every step.
  assert cfg.rewards["dof_error"].weight == -0.04
  assert cfg.rewards["dof_error"].func.__name__ == "dof_error"
  # The reference's plain single-sided 0.5 s threshold: with a 0.6 s gait period
  # this pays for a real leap, not for swing timing.
  assert cfg.rewards["feet_air_time"].weight == 0.5
  assert cfg.rewards["feet_air_time"].params["threshold"] == 0.5
  assert cfg.rewards["feet_air_time"].func.__name__ == "feet_air_time"
  assert cfg.rewards["feet_stumble"].weight == -0.5
  assert cfg.rewards["feet_stumble"].params["horizontal_ratio"] == 2.0
  # A foot asked to clear a 14 cm riser needs roughly 0.20 m under the sole; the
  # flat target stays low so plain walking is not taxed.
  clearance = cfg.rewards["foot_clearance"]
  assert clearance.weight == -0.15
  assert clearance.func.__name__ == "foot_clearance_terrain"
  assert clearance.params["target_height"] == 0.20
  assert clearance.params["flat_target_height"] == 0.10
  assert clearance.params["swing_only"] is True
  assert clearance.params["asset_cfg"].site_names == ("FL", "FR", "RL", "RR")
  trunk = cfg.rewards["base_clearance"]
  assert trunk.weight == 0.3
  assert trunk.func.__name__ == "base_clearance_reward"
  assert trunk.params["excluded_terrain_names"] == ("low_frame",)
  assert cfg.rewards["feet_edge"].weight == -1.0
  assert cfg.rewards["is_terminated"].weight == -10.0
  assert cfg.rewards["stand_still"].weight == -0.5

  assert set(cfg.terminations) == {
    "time_out",
    "base_contact",
    "velocity_violation",
    "fall",
    "forward_course_bounds",
    "stall_timeout",
  }
  assert cfg.terminations["stall_timeout"].params["time_limit"] == 2.0
  assert cfg.terminations["stall_timeout"].params["min_terrain_level"] == 2

  # Gait diagnostics, and the accumulators the curriculum gate reads.
  assert {
    "gait_diagonal_match",
    "gait_base_clearance",
    "gait_base_tilt",
    "gait_hind_swing_ratio",
    "gait_front_hind_swing_gap",
  } <= set(cfg.metrics)


def test_wmp_task_uses_reference_termination_and_domain_randomization_cfg():
  cfg = Nazarite_WMP_HIM_Complex_Terrain_Go2()

  assert cfg.terminations["time_out"].time_out
  assert cfg.terminations["base_contact"].params["force_threshold"] == 1.0
  assert cfg.terminations["velocity_violation"].params == {
    "command_name": "twist",
    "terrain_level_threshold": 3,
    "velocity_error_threshold": 1.5,
  }
  assert cfg.terminations["fall"].params["downward_velocity_threshold"] == -3.0
  # stairs and ramp used to sit on the omni-directional sampler, where the expected
  # forward displacement is ~0 m and the 4 m promotion line is unreachable.
  assert cfg.terminations["forward_course_bounds"].params["terrain_names"] == (
    "stairs",
    "low_frame",
    "high_wall",
    "plum_blossom",
    "gap",
    "tire_array",
    "single_beam",
    "ramp",
  )

  assert cfg.events["robot_friction"].params["ranges"] == (0.5, 2.0)
  assert "foot_friction" not in cfg.events
  assert cfg.events["base_added_mass"].params["ranges"] == (0.0, 3.0)
  assert cfg.events["link_mass"].params["ranges"] == (0.8, 1.2)
  assert "pseudo_inertia" not in cfg.events
  assert cfg.events["pd_gains"].params["kp_range"] == (0.8, 1.2)
  assert cfg.events["pd_gains"].params["kd_range"] == (0.8, 1.2)
  assert cfg.events["effort_limits"].params["effort_limit_range"] == (0.8, 1.2)
  assert cfg.events["push_robot"].interval_range_s == (15.0, 15.0)
  assert cfg.events["depth_camera_mount"].mode == "startup"

  # Spawn inside the 1.2 m lane and at the nominal standing height; the source
  # HIM reset distribution is intentionally not reused for WMP.
  pose_range = cfg.events["reset_base"].params["pose_range"]
  assert pose_range["y"] == (-0.2, 0.2)
  assert pose_range["z"] == (0.0, 0.0)
  assert pose_range["x"] == (-0.5, 0.5)

  robot_cfg = cfg.scene.entities["robot"]
  assert robot_cfg.articulation is not None
  for actuator in robot_cfg.articulation.actuators:
    assert actuator.delay_min_lag == 0
    assert actuator.delay_max_lag == 5  # 10 ms at the Go2's 2 ms physics step.
    assert actuator.delay_update_period == cfg.decimation

  base_contact_sensor = next(
    sensor for sensor in cfg.scene.sensors if sensor.name == "nonfoot_ground_touch"
  )
  assert base_contact_sensor.reduce == "netforce"
  assert base_contact_sensor.history_length == cfg.decimation


def test_wmp_reset_distribution_does_not_change_shared_him_config():
  from nazarite.config.train_config.go2.him_complex_env_cfg import Nazarite_HIM_Complex_Terrain_Go2

  Nazarite_WMP_HIM_Complex_Terrain_Go2()
  him_pose_range = (
    Nazarite_HIM_Complex_Terrain_Go2().events["reset_base"].params["pose_range"]
  )
  assert him_pose_range["y"] == (-0.5, 0.5)
  assert him_pose_range["z"] == (0.01, 0.05)


def test_reference_termination_terms_match_source_thresholds():
  batch = 3
  impact_history = torch.zeros(batch, 1, 4, 3)
  impact_history[1, 0, 2, 2] = 1.01
  impact_history[2, 0, 0, 0] = 1.0
  contact_env = SimpleNamespace(
    num_envs=batch,
    scene={
      "nonfoot_ground_touch": SimpleNamespace(
        data=SimpleNamespace(force_history=impact_history, force=None)
      )
    },
  )
  assert base_contact(contact_env).tolist() == [False, True, False]

  robot = SimpleNamespace(
    data=SimpleNamespace(
      root_link_lin_vel_w=torch.tensor(
        [[0.0, 0.0, -3.01], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
      ),
      projected_gravity_b=torch.tensor(
        [[0.0, 0.0, -1.0], [0.0, 0.0, 0.01], [0.0, 0.0, 0.0]]
      ),
    )
  )
  fall_env = SimpleNamespace(scene={"robot": robot})
  assert fall(fall_env).tolist() == [True, True, False]


def test_forward_course_bounds_only_blocks_forward_obstacle_tiles():
  terrain = SimpleNamespace(
    cfg=SimpleNamespace(
      terrain_generator=SimpleNamespace(
        size=(8.0, 8.0),
        sub_terrains={"stairs": object(), "gap": object()},
      )
    ),
    terrain_types=torch.tensor([0, 1, 1]),
  )
  # The track is 7.20 m wide with a 0.40 m void on either side, so the margin of
  # 0.60 m puts the bound at |y| = 3.4 m: 3.2 m is still on deck, 3.9 m is not.
  robot = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=torch.tensor([[0.0, 4.5, 0.0], [0.0, 3.2, 0.0], [0.0, 3.9, 0.0]])
    )
  )

  class Scene(dict):
    pass

  scene = Scene(robot=robot)
  scene.terrain = terrain
  scene.env_origins = torch.zeros(3, 3)

  env = SimpleNamespace(
    num_envs=3,
    device="cpu",
    scene=scene,
  )

  assert forward_course_bounds(env, ("gap",)).tolist() == [False, False, True]

  class _Scene(dict):
    terrain = SimpleNamespace(terrain_levels=torch.tensor([4, 4, 3, 4, 4]))

  velocity_robot = SimpleNamespace(
    data=SimpleNamespace(
      root_link_lin_vel_b=torch.tensor(
        [
          [-2.0, 0.0, 0.0],
          [2.0, 0.0, 0.0],
          [-5.0, 0.0, 0.0],
          [0.4, 0.0, 0.0],
          [-2.0, 0.0, 0.0],
        ]
      )
    )
  )
  commands = torch.tensor(
    [
      [1.0, 0.0, 0.0],
      [-1.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [-1.0, 0.0, 0.0],
      [-1.0, 0.0, 0.0],
    ]
  )
  velocity_env = SimpleNamespace(
    num_envs=5,
    device="cpu",
    scene=_Scene(robot=velocity_robot),
    command_manager=SimpleNamespace(get_command=lambda _name: commands),
  )
  assert velocity_violation(velocity_env).tolist() == [True, True, False, False, False]

  timeout_env = SimpleNamespace(
    episode_length_buf=torch.tensor([99, 100, 101]), max_episode_length=100
  )
  assert reference_time_out(timeout_env).tolist() == [False, False, True]


def test_wmp_base_contact_allows_upright_low_frame_touch_but_keeps_fall_terminal():
  force = torch.tensor([[[2.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]])
  robot = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=torch.tensor([[0.0, 0.0, 0.32], [0.0, 0.0, 0.12]]),
      projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, 0.1]]),
    )
  )
  env = SimpleNamespace(
    num_envs=2,
    scene={
      "nonfoot_ground_touch": SimpleNamespace(
        data=SimpleNamespace(force_history=force, force=None)
      ),
      "robot": robot,
    },
  )

  result = base_contact(
    env,
    require_fall_state=True,
    min_base_height=0.18,
  )

  assert result.tolist() == [False, True]


def test_feet_edge_reshapes_flat_raycast_distances_per_foot():
  batch, num_feet, rays_per_foot = 3, 4, 9
  distances = torch.ones(batch, num_feet * rays_per_foot)
  distances[0, 0] = -1.0  # One unsupported ray under foot 0.
  distances[2, 2 * rays_per_foot : 3 * rays_per_foot] = -1.0
  contact = torch.zeros(batch, num_feet, dtype=torch.bool)
  contact[0, 0] = True
  contact[1, 1] = True  # This environment is not on a sparse terrain.
  contact[2, 2] = True

  scene = {
    "terrain": SimpleNamespace(
      cfg=SimpleNamespace(
        terrain_generator=SimpleNamespace(sub_terrains={"tire_array": None})
      ),
      terrain_types=torch.tensor([0, 1, 0]),
    ),
    "wmp_feet_support": SimpleNamespace(
      num_frames=num_feet,
      num_rays_per_frame=rays_per_foot,
      data=SimpleNamespace(distances=distances),
    ),
    "feet_ground_contact": SimpleNamespace(data=SimpleNamespace(found=contact)),
  }
  env = SimpleNamespace(
    num_envs=batch,
    device="cpu",
    scene=scene,
    command_manager=SimpleNamespace(
      get_command=lambda _name: torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
      )
    ),
  )

  result = feet_edge(env, terrain_names=("tire_array",))

  torch.testing.assert_close(result, torch.tensor([1.0 / 9.0, 0.0, 1.0]))


def test_feet_edge_curriculum_matches_reference_ramp():
  assert feet_edge_curriculum_scale(0) == 0.1
  assert feet_edge_curriculum_scale(4000) == 0.1
  assert feet_edge_curriculum_scale(7000) == 0.55
  assert feet_edge_curriculum_scale(10000) == 1.0
  assert feet_edge_curriculum_scale(12000) == 1.0


def test_wmp_actor_input_is_96_dimensions_and_keeps_him_estimator():
  batch = 3
  obs = TensorDict(
    {
      "actor": torch.zeros(batch, 6, 45),
      "world_model_feature": torch.zeros(batch, 512),
    },
    batch_size=[batch],
  )
  model = WMPHIMActorModel(
    obs,
    {"actor": ["actor"]},
    "actor",
    12,
    hidden_dims=(64, 32),
    # 45, not 47: the gait clock is gone (issue #3). Mirrored from the shipped
    # config so this fixture cannot drift away from what actually trains.
    num_one_step_obs=45,
    history_size=6,
    history_term_dims=(3, 3, 3, 12, 12, 12),
    history_order="frame_major_oldest_first",
    estimator_cfg={
      "enc_hidden_dims": (32, 16),
      "tar_hidden_dims": (32,),
      "num_prototype": 8,
    },
    distribution_cfg={
      "class_name": "rsl_rl.modules.distribution:GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
  )

  latent = model.get_latent(obs)
  actions = model(obs, stochastic_output=True)
  # 45 one-step + 3 estimated base velocity + 16 HIM latent + 32 visual latent.
  assert latent.shape == (batch, 96)
  assert model.estimator.num_latent == 16
  assert actions.shape == (batch, 12)


def test_depth_normalization_rssm_shapes_and_world_model_update():
  depth = torch.tensor([0.0, 1.0, 2.0, float("nan")])
  normalized = WMPWorldModel.normalize_depth(depth)
  torch.testing.assert_close(normalized, torch.tensor([-0.5, 0.0, 0.5, 0.5]))

  model = _small_world_model()
  image = torch.rand(2, 64, 64, 1) - 0.5
  prop = torch.rand(2, 5)
  state = model.observe_step(image, prop, torch.zeros(2, 3), torch.ones(2), state=None)
  assert state["deter"].shape == (2, 16)
  assert model.dynamics.get_deter_feat(state).shape == (2, 16)

  batch = {
    "image": torch.randint(0, 256, (2, 3, 64, 64, 1), dtype=torch.uint8),
    "prop": torch.randn(2, 3, 5),
    "action": torch.randn(2, 3, 3),
    "reward": torch.randn(2, 3),
    "is_first": torch.tensor([[True, False, False], [True, False, False]]),
  }
  metrics = model.train_batch(batch)
  assert torch.isfinite(torch.tensor(metrics["loss"]))
  assert metrics["image_loss"] >= 0.0


def test_runner_camera_depth_is_raw_so_each_consumer_normalizes_once():
  raw_depth = torch.full((2, 64, 64, 1), 1.0)
  runner = object.__new__(WMPHIMOnPolicyRunner)
  runner.env = SimpleNamespace(
    num_envs=2,
    unwrapped=SimpleNamespace(
      scene={"wmp_depth_camera": SimpleNamespace(data=SimpleNamespace(depth=raw_depth))}
    ),
  )
  runner.device = "cpu"

  read_depth = runner._camera_depth()
  torch.testing.assert_close(read_depth, raw_depth)
  torch.testing.assert_close(
    WMPWorldModel.normalize_depth(read_depth), torch.zeros_like(raw_depth)
  )


def test_inference_policy_without_explicit_depth_normalizes_camera_once():
  raw_depth = torch.full((2, 64, 64, 1), 1.0)
  captured = {}

  class FakeWorldModel:
    feature_dim = 4
    dynamics = SimpleNamespace(
      get_deter_feat=lambda _state: torch.zeros(2, 4),
    )

    def eval(self):
      pass

    def observe_step(self, image, prop, action, is_first, state):
      captured["image"] = image.clone()
      return {"deter": torch.zeros(2, 4)}

  class FakePolicy:
    def act_inference(self, _obs):
      return torch.zeros(2, 3)

  runner = object.__new__(WMPHIMOnPolicyRunner)
  runner.env = SimpleNamespace(
    num_envs=2,
    num_actions=3,
    unwrapped=SimpleNamespace(
      scene={"wmp_depth_camera": SimpleNamespace(data=SimpleNamespace(depth=raw_depth))}
    ),
  )
  runner.device = "cpu"
  runner._wm_runtime_cfg = {"action_chunk": 1}
  runner.world_model = FakeWorldModel()
  runner.alg = SimpleNamespace(
    eval_mode=lambda: None,
    get_policy=lambda: FakePolicy(),
  )

  policy = runner.get_inference_policy()
  # 48, not 50: the gait clock is gone, so the estimator group lost two dims.
  obs = TensorDict({"estimator": torch.zeros(2, 48)}, batch_size=[2])
  actions = policy(obs)

  torch.testing.assert_close(captured["image"], torch.zeros_like(raw_depth))
  assert actions.shape == (2, 3)


def test_world_model_and_optimizer_state_round_trip():
  model = _small_world_model()
  batch = {
    "image": torch.randint(0, 256, (1, 2, 64, 64, 1), dtype=torch.uint8),
    "prop": torch.randn(1, 2, 5),
    "action": torch.randn(1, 2, 3),
    "reward": torch.randn(1, 2),
    "is_first": torch.tensor([[True, False]]),
  }
  model.train_batch(batch)
  saved_model = model.state_dict()
  saved_optimizer = model.optimizer.state_dict()

  restored = _small_world_model()
  restored.load_state_dict(saved_model)
  restored.optimizer.load_state_dict(saved_optimizer)

  for name, value in model.state_dict().items():
    torch.testing.assert_close(restored.state_dict()[name], value)
  assert restored.optimizer.state_dict()["state"]


def test_replay_keeps_sampled_sequences_inside_one_episode():
  replay = WorldModelReplay(
    num_envs=2,
    capacity_per_env=8,
    image_shape=(4, 4, 1),
    prop_dim=1,
    action_dim=2,
    device="cpu",
  )
  for step in range(6):
    first = torch.tensor([step in (0, 3), step == 0])
    env0 = step if step < 3 else 100 + step - 3
    env1 = 200 + step
    prop = torch.tensor([[float(env0)], [float(env1)]])
    image = torch.zeros(2, 4, 4, 1, dtype=torch.uint8)
    action = torch.zeros(2, 2)
    reward = torch.zeros(2)
    replay.add(image, prop, action, reward, first)
  assert replay.can_sample(batch_length=2)
  batch = replay.sample(batch_size=12, batch_length=2)
  assert batch["image"].shape == (12, 2, 4, 4, 1)
  assert batch["is_first"][:, 0].all()
  assert not batch["is_first"][:, 1].any()
  torch.testing.assert_close(
    batch["prop"][:, 1, 0] - batch["prop"][:, 0, 0],
    torch.ones(12),
  )
  assert torch.isfinite(batch["prop"]).all()


def test_world_model_observation_placeholder_has_configured_width():
  env = SimpleNamespace(num_envs=4, device="cpu")
  output = empty_world_model_feature(env, feature_dim=512)
  assert output.shape == (4, 512)
  assert torch.count_nonzero(output) == 0
