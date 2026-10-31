"""Compatibility import; implementation lives in :mod:`nazarite.config.train_config.go2.go2_env_cfgs`."""

import importlib as _importlib
import sys as _sys

if __name__ == "__main__":
    _importlib.import_module("nazarite.config.train_config.go2.go2_env_cfgs").main()
else:
    _sys.modules[__name__] = _importlib.import_module("nazarite.config.train_config.go2.go2_env_cfgs")
