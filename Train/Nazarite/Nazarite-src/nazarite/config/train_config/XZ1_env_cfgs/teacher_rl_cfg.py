"""Privileged XZ1 teacher PPO; HIM optimizer and observation contracts kept."""

from dataclasses import dataclass, fields

from nazarite.config.train_config.him_cfg import (
    RslRlHimActorCfg,
    RslRlHimRunnerCfg,
)

from .him_rl_cfg import xz1_him_runner_cfg


@dataclass
class XZ1TeacherActorCfg(RslRlHimActorCfg):
    class_name: str = "nazarite.rl.xz1_teacher:XZ1TerrainTeacherActor"
    scan_group: str = "teacher_scan"


@dataclass
class XZ1TeacherRunnerCfg(RslRlHimRunnerCfg):
    class_name: str = "nazarite.rl.xz1_teacher:XZ1TerrainTeacherRunner"
    initial_checkpoint: str | None = None


def xz1_teacher_runner_cfg() -> XZ1TeacherRunnerCfg:
    baseline = xz1_him_runner_cfg(experiment_name="xz1_wmp_him_teacher")
    values = {item.name: getattr(baseline, item.name) for item in fields(baseline)}
    actor_values = {
        item.name: getattr(baseline.actor, item.name)
        for item in fields(baseline.actor)
        if item.name != "class_name"
    }
    values["actor"] = XZ1TeacherActorCfg(**actor_values)
    values.pop("class_name")
    return XZ1TeacherRunnerCfg(**values)
