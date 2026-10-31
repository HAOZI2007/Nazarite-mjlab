"""Compatibility import; implementation lives in :mod:`nazarite.rl.xz1.teacher`."""

import importlib as _importlib
import sys as _sys

if __name__ == "__main__":
    _importlib.import_module("nazarite.rl.xz1.teacher").main()
else:
    _sys.modules[__name__] = _importlib.import_module("nazarite.rl.xz1.teacher")
