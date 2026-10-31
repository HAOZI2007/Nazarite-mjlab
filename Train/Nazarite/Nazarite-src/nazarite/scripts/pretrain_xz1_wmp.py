"""Compatibility import; implementation lives in :mod:`nazarite.scripts.xz1.pretrain_xz1_wmp`."""

import importlib as _importlib
import sys as _sys

if __name__ == "__main__":
    _importlib.import_module("nazarite.scripts.xz1.pretrain_xz1_wmp").main()
else:
    _sys.modules[__name__] = _importlib.import_module("nazarite.scripts.xz1.pretrain_xz1_wmp")
