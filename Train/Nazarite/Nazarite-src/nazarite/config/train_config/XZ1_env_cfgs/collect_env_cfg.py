"""Compatibility import; implementation lives in :mod:`nazarite.config.train_config.xz1.collect_env_cfg`."""

import importlib as _importlib
import sys as _sys

if __name__ == "__main__":
    _importlib.import_module("nazarite.config.train_config.xz1.collect_env_cfg").main()
else:
    _sys.modules[__name__] = _importlib.import_module("nazarite.config.train_config.xz1.collect_env_cfg")
