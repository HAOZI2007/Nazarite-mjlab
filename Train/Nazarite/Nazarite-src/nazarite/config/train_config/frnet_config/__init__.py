"""Compatibility package; implementations live in :mod:`nazarite.config.train_config.go2.frnet`."""

from importlib import import_module as _import_module

_target = _import_module("nazarite.config.train_config.go2.frnet")


def __getattr__(name):
    return getattr(_target, name)
