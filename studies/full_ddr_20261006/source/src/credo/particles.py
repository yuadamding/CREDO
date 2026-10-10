"""Deprecated compact-v3 particle-solver import shim."""

from .recipes.compact_sde_v3 import particles as _implementation
from .recipes.compact_sde_v3.particles import *  # noqa: F403


def __getattr__(name: str):
    return getattr(_implementation, name)
