"""Stable lazy facade over the computational modules.

Each public object has one defining module. Cross-module references resolve here
at call time, avoiding circular imports and permitting explicit dataset adapters.
No source is executed dynamically. Assigning an API symbol replaces it in its
owning module, so tests and adapters observe one consistent object.
"""
from __future__ import annotations

import importlib
import sys
import types

from ._exports import EXPORTS

__all__ = sorted(EXPORTS)


def _owner(name):
    module = EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"RTT computational API has no attribute {name!r}")
    return importlib.import_module(f"{__package__}.{module}")


def __getattr__(name):
    return getattr(_owner(name), name)


def __dir__():
    return sorted(set(globals()) | set(EXPORTS))


class _PublicAPI(types.ModuleType):
    def __setattr__(self, name, value):
        if name in EXPORTS:
            setattr(_owner(name), name, value)
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in EXPORTS:
            delattr(_owner(name), name)
        else:
            super().__delattr__(name)


class _CompatibilityAPI(_PublicAPI):
    def __getattr__(self, name):
        return __getattr__(name)

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(EXPORTS))


def install_compatibility_facade(module_name):
    """Expose the computational API from a historical script import path."""
    module = sys.modules[module_name]
    module.__class__ = _CompatibilityAPI
    module.__all__ = __all__


sys.modules[__name__].__class__ = _PublicAPI
