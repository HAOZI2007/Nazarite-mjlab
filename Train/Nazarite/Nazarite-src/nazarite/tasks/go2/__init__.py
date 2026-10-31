"""Register GO2 training tasks."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from nazarite.config.train_config.go2.frnet.frnet_recovery_env_cfgs import (
    Nazarite_FRNet_Recovery_Go2,
    Nazarite_FRNet_Recovery_Terrain_Go2,
)
from nazarite.config.train_config.go2.frnet.frnet_rl_cfg import (
    frnet_go2_recovery_runner_cfg,
)
from nazarite.config.train_config.go2.go2_env_cfgs import (
    Nazarite_Velocity_Flat_Go2_No_WTW,
    Nazarite_Velocity_Flat_Go2_WTW,
)
from nazarite.config.train_config.go2.him_complex_env_cfg import (
    Nazarite_HIM_Complex_Terrain_Go2,
)
from nazarite.config.train_config.go2.him_rl_cfg import unitree_go2_him_runner_cfg
from nazarite.config.train_config.go2.velocity_rl_cfg import (
    unitree_go2_normal_ppo_runner_cfg,
)
from nazarite.config.train_config.go2.wmp_him_go2_env_cfg import (
    Nazarite_WMP_HIM_Complex_Terrain_Go2,
)
from nazarite.config.train_config.go2.wmp_rl_cfg import go2_wmp_him_runner_cfg
from nazarite.rl.him.runner import HIMOnPolicyRunner
from nazarite.rl.wmp.runner import WMPHIMOnPolicyRunner

register_mjlab_task(
    task_id="Nazarite-Velocity-Flat-Go2",
    env_cfg=Nazarite_Velocity_Flat_Go2_No_WTW(),
    play_env_cfg=Nazarite_Velocity_Flat_Go2_No_WTW(play=True),
    rl_cfg=unitree_go2_normal_ppo_runner_cfg(experiment_name="go2_flat_baseline"),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Nazarite-HIM-Complex-Terrain-Go2",
    env_cfg=Nazarite_HIM_Complex_Terrain_Go2(),
    play_env_cfg=Nazarite_HIM_Complex_Terrain_Go2(play=True),
    rl_cfg=unitree_go2_him_runner_cfg(),
    runner_cls=HIMOnPolicyRunner,
)

register_mjlab_task(
    task_id="Nazarite-WMP-HIM-Flat-Go2",
    env_cfg=Nazarite_WMP_HIM_Complex_Terrain_Go2(flat=True),
    play_env_cfg=Nazarite_WMP_HIM_Complex_Terrain_Go2(play=True, flat=True),
    rl_cfg=go2_wmp_him_runner_cfg(experiment_name="go2_wmp_him_flat"),
    runner_cls=WMPHIMOnPolicyRunner,
)

register_mjlab_task(
    task_id="Nazarite-WMP-HIM-Complex-Terrain-Go2",
    env_cfg=Nazarite_WMP_HIM_Complex_Terrain_Go2(),
    play_env_cfg=Nazarite_WMP_HIM_Complex_Terrain_Go2(play=True),
    rl_cfg=go2_wmp_him_runner_cfg(),
    runner_cls=WMPHIMOnPolicyRunner,
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
