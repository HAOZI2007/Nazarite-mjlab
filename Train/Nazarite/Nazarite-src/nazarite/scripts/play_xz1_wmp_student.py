"""Compatibility import; implementation lives in :mod:`nazarite.scripts.xz1.play_xz1_wmp_student`."""

import importlib as _importlib
import sys as _sys

if __name__ == "__main__":
    _importlib.import_module("nazarite.scripts.xz1.play_xz1_wmp_student").main()
else:
    _sys.modules[__name__] = _importlib.import_module("nazarite.scripts.xz1.play_xz1_wmp_student")
