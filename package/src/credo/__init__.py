"""CREDO compatibility import package.

The implementation lives under the historical ``cape`` package. This alias
lets downstream code use ``import credo`` without breaking existing ``cape``
imports.
"""
from __future__ import annotations

import importlib
import pkgutil
import sys

from cape import __version__

_ALIASED_SUBMODULES = ("config", "data", "eval", "losses", "models", "training")

for _name in _ALIASED_SUBMODULES:
    _target_name = f"cape.{_name}"
    _alias_name = f"{__name__}.{_name}"
    _module = importlib.import_module(_target_name)
    sys.modules[_alias_name] = _module
    setattr(sys.modules[__name__], _name, _module)
    if hasattr(_module, "__path__"):
        for _info in pkgutil.walk_packages(_module.__path__, prefix=f"{_target_name}."):
            _child = importlib.import_module(_info.name)
            _child_alias = f"{__name__}.{_info.name[len('cape.'):]}"
            sys.modules[_child_alias] = _child

__all__ = ["__version__"]
