"""Compatibility import; implementation lives in :mod:`nazarite.scripts.go2.evaluate_frnet_recovery`."""

import importlib as _importlib
import sys as _sys

if __name__ == "__main__":
    _importlib.import_module("nazarite.scripts.go2.evaluate_frnet_recovery").main()
else:
    _sys.modules[__name__] = _importlib.import_module("nazarite.scripts.go2.evaluate_frnet_recovery")
