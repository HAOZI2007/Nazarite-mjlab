"""Register XZ1 training tasks."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from nazarite.config.train_config.xz1 import (
    Nazarite_HIM_Complex_Terrain_XZ1,
    Nazarite_Velocity_Flat_XZ1_WTW,
)
from nazarite.config.train_config.xz1.collect_env_cfg import (
    Nazarite_WMP_HIM_Collect_XZ1,
)
from nazarite.config.train_config.xz1.him_rl_cfg import xz1_him_runner_cfg
from nazarite.config.train_config.xz1.teacher_env_cfg import (
    XZ1_TEACHER_SCAN_HISTORY_LENGTH,
    XZ1_TEACHER_SCAN_UPDATE_PERIOD,
    Nazarite_WMP_HIM_Eval_XZ1,
    Nazarite_WMP_HIM_Teacher_XZ1,
)
from nazarite.config.train_config.xz1.teacher_rl_cfg import (
    xz1_height_history_training_cfg,
    xz1_teacher_runner_cfg,
)
from nazarite.config.train_config.xz1.wtw_rl_cfg import xz1_wtw_runner_cfg
from nazarite.rl.him.runner import HIMOnPolicyRunner
from nazarite.rl.xz1.teacher import XZ1TerrainTeacherRunner

register_mjlab_task(
    task_id="Nazarite-WMP-HIM-Collect-XZ1",
    env_cfg=Nazarite_WMP_HIM_Collect_XZ1(),
    play_env_cfg=Nazarite_WMP_HIM_Collect_XZ1(play=True),
    rl_cfg=xz1_teacher_runner_cfg(),
    runner_cls=XZ1TerrainTeacherRunner,
)

register_mjlab_task(
    task_id="Nazarite-WMP-HIM-Teacher-XZ1",
    env_cfg=Nazarite_WMP_HIM_Teacher_XZ1(
        scan_history_length=XZ1_TEACHER_SCAN_HISTORY_LENGTH,
        scan_update_period=XZ1_TEACHER_SCAN_UPDATE_PERIOD,
    ),
    play_env_cfg=Nazarite_WMP_HIM_Teacher_XZ1(
        play=True,
        scan_history_length=XZ1_TEACHER_SCAN_HISTORY_LENGTH,
        scan_update_period=XZ1_TEACHER_SCAN_UPDATE_PERIOD,
    ),
    rl_cfg=xz1_height_history_training_cfg(),
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
