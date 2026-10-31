"""WTW commands and rewards, with historical command exports."""

from . import commands as _commands


def __getattr__(name):
    return getattr(_commands, name)
