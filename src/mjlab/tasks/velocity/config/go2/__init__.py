from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.tasks.velocity.rl.runner import HIMVelocityOnPolicyRunner

# 💡 注意：因为你复制了 Go1 的文件，里面的函数名大概率还是叫 unitree_go1_...
# 我们暂时不改函数名，只要把外包装的 task_id 改成 Go2 就可以完美运行！
from .env_cfgs import (
  unitree_go2_flat_env_cfg,
  unitree_go2_him_flat_env_cfg,
  unitree_go2_rough_env_cfg,
  unitree_go2_stairs_env_cfg,
  unitree_go2_stand_env_cfg,
)
from .rl_cfg import (
  unitree_go2_him_ppo_runner_cfg,
  unitree_go2_normal_ppo_runner_cfg,
  unitree_go2_RNN_ppo_runner_cfg,
)

# 注册复杂地形 (Rough Terrain) 任务
register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Unitree-Go2",
  env_cfg=unitree_go2_rough_env_cfg(),
  play_env_cfg=unitree_go2_rough_env_cfg(play=True),
  rl_cfg=unitree_go2_RNN_ppo_runner_cfg(experiment_name="go2_rough"),
  runner_cls=VelocityOnPolicyRunner,
)

# 注册平地 (Flat Terrain) 任务
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-Go2",
  env_cfg=unitree_go2_flat_env_cfg(),
  play_env_cfg=unitree_go2_flat_env_cfg(play=True),
  rl_cfg=unitree_go2_normal_ppo_runner_cfg(experiment_name="go2_flat"),
  runner_cls=VelocityOnPolicyRunner,
)

#注册平地站立任务
register_mjlab_task(
  task_id="Mjlab-Go2-Stand-v0",
  env_cfg=unitree_go2_stand_env_cfg(),
  play_env_cfg=unitree_go2_stand_env_cfg(play=True),
  rl_cfg=unitree_go2_normal_ppo_runner_cfg(experiment_name="go2_stand"),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Stairs-Unitree-Go2",
  env_cfg=unitree_go2_stairs_env_cfg(),
  play_env_cfg=unitree_go2_stairs_env_cfg(play=True),
  rl_cfg=unitree_go2_RNN_ppo_runner_cfg(experiment_name="go2_stairs"),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-Unitree-Go2-HIM",
    env_cfg=unitree_go2_him_flat_env_cfg(),
    play_env_cfg=unitree_go2_him_flat_env_cfg(play=True),
    rl_cfg=unitree_go2_him_ppo_runner_cfg(),
    runner_cls=HIMVelocityOnPolicyRunner, # 指定我们手写的 Runner[cite: 6]
)
