"""Compatibility import; implementation lives in :mod:`nazarite.mdp.wmp.rewards`."""

import importlib as _importlib
import sys as _sys

if __name__ == "__main__":
    _importlib.import_module("nazarite.mdp.wmp.rewards").main()
else:
    _sys.modules[__name__] = _importlib.import_module("nazarite.mdp.wmp.rewards")
