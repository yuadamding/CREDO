"""Deprecated compact-v3 objective import shim."""

from .recipes.compact_sde_v3 import objective as _implementation
from .recipes.compact_sde_v3.objective import *  # noqa: F403


def __getattr__(name: str):
    return getattr(_implementation, name)
