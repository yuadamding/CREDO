"""Deprecated compact-v3 model import shim."""

from .recipes.compact_sde_v3 import model as _implementation
from .recipes.compact_sde_v3.model import *  # noqa: F403


def __getattr__(name: str):
    return getattr(_implementation, name)
