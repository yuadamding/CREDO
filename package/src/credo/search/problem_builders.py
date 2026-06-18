"""Package-level data-to-problem builder registry for search adapters.

Search should call stable package APIs, not CLI ``argparse.Namespace`` internals.
Project-specific code can register dataset factories by ``data_id`` and
``dataset_kind``; the search runner can then build problems from ``RunConfig``.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, Optional


ProblemKind = Literal["endpoint", "trajectory", "single_time"]
ProblemFactory = Callable[[Any], Any]


class ProblemBuilderRegistry:
    """Registry of Namespace-free problem factories keyed by kind/data_id."""

    def __init__(self) -> None:
        self._factories: dict[tuple[str, str], ProblemFactory] = {}

    def register(self, kind: ProblemKind, data_id: str, factory: ProblemFactory) -> None:
        if kind not in {"endpoint", "trajectory", "single_time"}:
            raise ValueError(f"Unknown problem kind {kind!r}.")
        if not data_id:
            raise ValueError("data_id must not be empty.")
        self._factories[(kind, str(data_id))] = factory

    def build(self, kind: ProblemKind, cfg: Any) -> Any:
        data_id = _data_id(cfg)
        key = (kind, data_id)
        try:
            factory = self._factories[key]
        except KeyError as exc:
            available = sorted(f"{k}:{d}" for k, d in self._factories)
            raise KeyError(
                f"No CREDO problem builder registered for kind={kind!r}, data_id={data_id!r}. "
                f"Available builders: {available}."
            ) from exc
        return factory(cfg)

    def clear(self) -> None:
        self._factories.clear()


DEFAULT_PROBLEM_BUILDERS = ProblemBuilderRegistry()


def register_problem_builder(
    kind: ProblemKind,
    data_id: str,
    factory: ProblemFactory,
    *,
    registry: ProblemBuilderRegistry = DEFAULT_PROBLEM_BUILDERS,
) -> None:
    """Register a project/package data adapter for config-driven search."""
    registry.register(kind, data_id, factory)


def build_endpoint_problem_from_config(
    cfg: Any,
    *,
    registry: ProblemBuilderRegistry = DEFAULT_PROBLEM_BUILDERS,
) -> Any:
    """Build an endpoint problem from a validated ``RunConfig``."""
    return registry.build("endpoint", cfg)


def build_trajectory_problem_from_config(
    cfg: Any,
    *,
    registry: ProblemBuilderRegistry = DEFAULT_PROBLEM_BUILDERS,
) -> Any:
    """Build a trajectory problem from a validated ``RunConfig``."""
    return registry.build("trajectory", cfg)


def build_single_time_problem_from_config(
    cfg: Any,
    *,
    registry: ProblemBuilderRegistry = DEFAULT_PROBLEM_BUILDERS,
) -> Any:
    """Build a single-time problem from a validated ``RunConfig``."""
    return registry.build("single_time", cfg)


def clear_problem_builders(
    *,
    registry: ProblemBuilderRegistry = DEFAULT_PROBLEM_BUILDERS,
) -> None:
    """Clear registered builders, mainly for tests and isolated scripts."""
    registry.clear()


def _data_id(cfg: Any) -> str:
    data_id: Optional[str] = getattr(cfg, "data_id", None)
    if data_id is None and isinstance(cfg, dict):
        data_id = cfg.get("data_id")
    if data_id is None or not str(data_id):
        raise ValueError("RunConfig.data_id is required to build a problem from config.")
    return str(data_id)


__all__ = [
    "DEFAULT_PROBLEM_BUILDERS",
    "ProblemBuilderRegistry",
    "ProblemFactory",
    "ProblemKind",
    "build_endpoint_problem_from_config",
    "build_single_time_problem_from_config",
    "build_trajectory_problem_from_config",
    "clear_problem_builders",
    "register_problem_builder",
]
