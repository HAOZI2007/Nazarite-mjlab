"""Compatibility import; implementation lives in :mod:`nazarite.rl.him.runner`."""

import importlib as _importlib
import sys as _sys

if __name__ == "__main__":
    _importlib.import_module("nazarite.rl.him.runner").main()
else:
    _sys.modules[__name__] = _importlib.import_module("nazarite.rl.him.runner")
