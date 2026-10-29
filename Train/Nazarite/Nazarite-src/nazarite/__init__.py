from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from nazarite.rl.him_runner import HIMOnPolicyRunner
from nazarite.rl.wmp.runner import WMPHIMOnPolicyRunner

from .config.train_config.env_cfgs.go2_env_cfgs import (
    Nazarite_Velocity_Flat_Go2_No_WTW,
    Nazarite_Velocity_Flat_Go2_WTW,
)
from .config.train_config.env_cfgs.him_complex_env_cfg import (
    Nazarite_HIM_Complex_Terrain_Go2,
)
from .config.train_config.env_cfgs.wmp_him_go2_env_cfg import (
    Nazarite_WMP_HIM_Complex_Terrain_Go2,
)
from .config.train_config.env_cfgs.wtw_delta_attention_env_cfg import (
    Nazarite_Wtw_Delta_Go2,
)
from .config.train_config.env_cfgs.wtw_delta_direct_env_cfg import (
    Nazarite_Wtw_Delta_Direct_Go2,
)
from .config.train_config.frnet_config.frnet_recovery_env_cfgs import (
    Nazarite_FRNet_Recovery_Go2,
    Nazarite_FRNet_Recovery_Terrain_Go2,
)
from .config.train_config.frnet_config.frnet_rl_cfg import (
    frnet_go2_recovery_runner_cfg,
)
from .config.train_config.him_cfg import (
    unitree_go2_him_runner_cfg,
)
from .config.train_config.rl_cfg import unitree_go2_normal_ppo_runner_cfg
from .config.train_config.smp_config.smp_forward_env_cfgs import (
    Nazarite_SMP_Forward_Go2,
)
from .config.train_config.smp_config.smp_prior_rl_cfg import (
    smp_forward_go2_runner_cfg,
)
from .config.train_config.smp_config.smp_teacher_env_cfgs import (
    Nazarite_SMP_Teacher_Go2,
)
from .config.train_config.smp_config.smp_teacher_rl_cfg import (
    smp_teacher_go2_runner_cfg,
)
from .config.train_config.wmp_him_go2_cfg import go2_wmp_him_runner_cfg
from .config.train_config.wtw_delta_attention_rl_cfg import wtw_delta_go2_runner_cfg
from .config.train_config.wtw_delta_direct_rl_cfg import wtw_delta_direct_go2_runner_cfg
from .config.train_config.XZ1_env_cfgs import (
    Nazarite_HIM_Complex_Terrain_XZ1,
    Nazarite_Velocity_Flat_XZ1_WTW,
)
from .config.train_config.XZ1_env_cfgs.collect_env_cfg import (
    Nazarite_WMP_HIM_Collect_XZ1,
)
from .config.train_config.XZ1_env_cfgs.him_rl_cfg import xz1_him_runner_cfg
from .config.train_config.XZ1_env_cfgs.teacher_env_cfg import (
    Nazarite_WMP_HIM_Eval_XZ1,
    Nazarite_WMP_HIM_Teacher_XZ1,
)
from .config.train_config.XZ1_env_cfgs.teacher_rl_cfg import xz1_teacher_runner_cfg
from .config.train_config.XZ1_env_cfgs.wtw_rl_cfg import xz1_wtw_runner_cfg
from .rl.xz1_teacher import XZ1TerrainTeacherRunner

register_mjlab_task(
    task_id="Nazarite-WMP-HIM-Collect-XZ1",
    env_cfg=Nazarite_WMP_HIM_Collect_XZ1(),
    play_env_cfg=Nazarite_WMP_HIM_Collect_XZ1(play=True),
    rl_cfg=xz1_teacher_runner_cfg(),
    runner_cls=XZ1TerrainTeacherRunner,
)

register_mjlab_task(
    task_id="Nazarite-WMP-HIM-Teacher-XZ1",
    env_cfg=Nazarite_WMP_HIM_Teacher_XZ1(),
    play_env_cfg=Nazarite_WMP_HIM_Teacher_XZ1(play=True),
    rl_cfg=xz1_teacher_runner_cfg(),
    runner_cls=XZ1TerrainTeacherRunner,
)

register_mjlab_task(
    task_id="Nazarite-WMP-HIM-Eval-XZ1",
    env_cfg=Nazarite_WMP_HIM_Eval_XZ1(),
    play_env_cfg=Nazarite_WMP_HIM_Eval_XZ1(play=True),
    rl_cfg=xz1_teacher_runner_cfg(),
    runner_cls=XZ1TerrainTeacherRunner,
)

register_mjlab_task(
    task_id="Nazarite-WTW-Delta-Direct-Go2",
    env_cfg=Nazarite_Wtw_Delta_Direct_Go2(),
    play_env_cfg=Nazarite_Wtw_Delta_Direct_Go2(play=True),
    rl_cfg=wtw_delta_direct_go2_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Nazarite-WTW-Delta-Go2",
    env_cfg=Nazarite_Wtw_Delta_Go2(),
    play_env_cfg=Nazarite_Wtw_Delta_Go2(play=True),
    rl_cfg=wtw_delta_go2_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

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
    task_id="Nazarite-Velocity-Flat-XZ1-WTW",
    env_cfg=Nazarite_Velocity_Flat_XZ1_WTW(),
    play_env_cfg=Nazarite_Velocity_Flat_XZ1_WTW(play=True),
    rl_cfg=xz1_wtw_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Nazarite-HIM-Complex-Terrain-XZ1",
    env_cfg=Nazarite_HIM_Complex_Terrain_XZ1(),
    play_env_cfg=Nazarite_HIM_Complex_Terrain_XZ1(play=True),
    rl_cfg=xz1_him_runner_cfg(),
    runner_cls=HIMOnPolicyRunner,
)

register_mjlab_task(
    task_id="Nazarite-SMP-Teacher-Go2",
    env_cfg=Nazarite_SMP_Teacher_Go2(),
    play_env_cfg=Nazarite_SMP_Teacher_Go2(play=True),
    rl_cfg=smp_teacher_go2_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Nazarite-SMP-Forward-Go2",
    env_cfg=Nazarite_SMP_Forward_Go2(),
    play_env_cfg=Nazarite_SMP_Forward_Go2(play=True),
    rl_cfg=smp_forward_go2_runner_cfg(),
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
