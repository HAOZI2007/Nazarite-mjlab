"""Compatibility import; implementation lives in :mod:`sim2sim.shared.math_utils`."""

import importlib as _importlib
import sys as _sys

if __name__ == "__main__":
    _importlib.import_module("sim2sim.shared.math_utils").main()
else:
    _sys.modules[__name__] = _importlib.import_module("sim2sim.shared.math_utils")
