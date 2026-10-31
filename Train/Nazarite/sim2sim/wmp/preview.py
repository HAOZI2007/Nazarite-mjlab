"""Compatibility import; implementation lives in :mod:`sim2sim.xz1.wmp.preview`."""

import importlib as _importlib
import sys as _sys

if __name__ == "__main__":
    _importlib.import_module("sim2sim.xz1.wmp.preview").main()
else:
    _sys.modules[__name__] = _importlib.import_module("sim2sim.xz1.wmp.preview")
