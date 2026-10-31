"""Compatibility package; implementations live in :mod:`sim2sim.shared.him`."""

from importlib import import_module as _import_module

_target = _import_module("sim2sim.shared.him")


def __getattr__(name):
    return getattr(_target, name)
