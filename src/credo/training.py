"""Deprecated compact-v3 trainer import shim."""

from .recipes.compact_sde_v3 import training as _implementation
from .recipes.compact_sde_v3.training import *  # noqa: F403


def __getattr__(name: str):
    return getattr(_implementation, name)
