"""Nazarite task plugin; task IDs stay stable across package reorganizations."""

from .tasks import go2 as go2
from .tasks import xz1 as xz1


def __getattr__(name):
    """Keep historical task-factory imports available at the plugin root."""
    for module in (go2, xz1):
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
