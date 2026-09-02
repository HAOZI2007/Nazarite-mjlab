from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .config.train_config.env_cfgs.frnet_recovery_env_cfgs import (
  Nazarite_FRNet_Recovery_Go2,
  Nazarite_FRNet_Recovery_Terrain_Go2,
)
from .config.train_config.env_cfgs.go2_env_cfgs import (
  Nazarite_Velocity_Flat_Go2_No_WTW,
  Nazarite_Velocity_Flat_Go2_WTW,
)
from .config.train_config.frnet_rl_cfg import frnet_go2_recovery_runner_cfg
from .config.train_config.rl_cfg import unitree_go2_normal_ppo_runner_cfg

register_mjlab_task(
  task_id="Nazarite-Velocity-Flat-Go2",
  env_cfg=Nazarite_Velocity_Flat_Go2_No_WTW(),
  play_env_cfg=Nazarite_Velocity_Flat_Go2_No_WTW(play=True),
  rl_cfg=unitree_go2_normal_ppo_runner_cfg(experiment_name="go2_flat_baseline"),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Nazarite-Velocity-Flat-Go2-WTW",
  env_cfg=Nazarite_Velocity_Flat_Go2_WTW(),
  play_env_cfg=Nazarite_Velocity_Flat_Go2_WTW(play=True),
  rl_cfg=unitree_go2_normal_ppo_runner_cfg(
    experiment_name="go2_flat_wtw_independent"
  ),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Nazarite-FRNet-Recovery-Go2",
  env_cfg=Nazarite_FRNet_Recovery_Go2(),
  play_env_cfg=Nazarite_FRNet_Recovery_Go2(play=True),
  rl_cfg=frnet_go2_recovery_runner_cfg(experiment_name="go2_frnet_recovery_flat"),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Nazarite-FRNet-Recovery-Terrain-Go2",
  env_cfg=Nazarite_FRNet_Recovery_Terrain_Go2(),
  play_env_cfg=Nazarite_FRNet_Recovery_Terrain_Go2(play=True),
  rl_cfg=frnet_go2_recovery_runner_cfg(experiment_name="go2_frnet_recovery_terrain"),
  runner_cls=VelocityOnPolicyRunner,
)
