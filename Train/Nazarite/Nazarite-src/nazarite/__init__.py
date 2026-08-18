from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .config.train_config.env_cfgs import Nazarite_Velocity_Flat_Go2
from .config.train_config.rl_cfg import unitree_go2_normal_ppo_runner_cfg

register_mjlab_task(
  task_id="Nazarite-Velocity-Flat-Go2",
  env_cfg=Nazarite_Velocity_Flat_Go2(),
  play_env_cfg=Nazarite_Velocity_Flat_Go2(play=True),
  rl_cfg=unitree_go2_normal_ppo_runner_cfg(experiment_name="go2_flat"),
  runner_cls=VelocityOnPolicyRunner,
)
