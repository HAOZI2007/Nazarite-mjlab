"""Protect task discovery and historical checkpoint imports during refactors."""

import importlib

import pytest

from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls

EXPECTED_TASKS = {
    "Nazarite-Velocity-Flat-Go2",
    "Nazarite-Velocity-Flat-Go2-WTW",
    "Nazarite-HIM-Complex-Terrain-Go2",
    "Nazarite-WMP-HIM-Flat-Go2",
    "Nazarite-WMP-HIM-Complex-Terrain-Go2",
    "Nazarite-FRNet-Recovery-Go2",
    "Nazarite-FRNet-Recovery-Terrain-Go2",
    "Nazarite-Velocity-Flat-XZ1-WTW",
    "Nazarite-HIM-Complex-Terrain-XZ1",
    "Nazarite-WMP-HIM-Collect-XZ1",
    "Nazarite-WMP-HIM-Teacher-XZ1",
    "Nazarite-WMP-HIM-Eval-XZ1",
}


def test_retained_tasks_and_both_configurations_are_available():
    assert {t for t in list_tasks() if t.startswith("Nazarite-")} == EXPECTED_TASKS
    for task in EXPECTED_TASKS:
        train = load_env_cfg(task)
        play = load_env_cfg(task, play=True)
        assert train is not play
        assert train.actions and play.actions
        assert load_rl_cfg(task).actor is not None
        assert load_runner_cls(task) is not None


def test_historical_root_and_wtw_exports():
    import nazarite
    from nazarite.config.train_config.go2.go2_env_cfgs import (
        Nazarite_Velocity_Flat_Go2_WTW,
    )
    from nazarite.mdp.wtw import WTWBehaviorCommandCfg
    from nazarite.mdp.wtw.commands import WTWBehaviorCommandCfg as canonical

    assert nazarite.Nazarite_Velocity_Flat_Go2_WTW is Nazarite_Velocity_Flat_Go2_WTW
    assert WTWBehaviorCommandCfg is canonical


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("nazarite.rl.xz1_teacher", "nazarite.rl.xz1.teacher"),
        ("nazarite.rl.xz1_student", "nazarite.rl.xz1.student"),
        ("nazarite.rl.him_runner", "nazarite.rl.him.runner"),
        ("nazarite.mdp.curriculums", "nazarite.mdp.shared.curriculums"),
        (
            "nazarite.config.train_config.XZ1_env_cfgs.teacher_env_cfg",
            "nazarite.config.train_config.xz1.teacher_env_cfg",
        ),
        (
            "nazarite.config.train_config.env_cfgs.go2_env_cfgs",
            "nazarite.config.train_config.go2.go2_env_cfgs",
        ),
    ],
)
def test_legacy_imports_share_the_canonical_module(old, new):
    assert importlib.import_module(old) is importlib.import_module(new)
