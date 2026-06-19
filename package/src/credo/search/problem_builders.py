"""Package-level data-to-problem builder registry for search adapters.

Search should call stable package APIs, not CLI ``argparse.Namespace`` internals.
Project-specific code can register dataset factories by ``data_id`` and
``dataset_kind``; the search runner can then build problems from ``RunConfig``.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Literal, Optional


ProblemKind = Literal["endpoint", "trajectory", "single_time"]
ProblemFactory = Callable[[Any], Any]


@dataclass(frozen=True)
class ProblemBuilderMetadata:
    builder_name: str
    builder_version: str
    data_path_hash: Optional[str] = None
    mass_table_hash: Optional[str] = None
    split_file_hash: Optional[str] = None
    fold_assignment_hash: Optional[str] = None
    latent_source: Optional[str] = None
    latent_key: Optional[str] = None
    gene_panel_hash: Optional[str] = None
    normalization_hash: Optional[str] = None
    hvg_preprocessing_hash: Optional[str] = None
    encoder_checkpoint_hash: Optional[str] = None
    representation_config_sha256: Optional[str] = None
    dataset_organism: Optional[str] = None
    gene_symbol_namespace: Optional[str] = None
    expression_gene_universe_hash: Optional[str] = None
    decoder_gene_panel_hash: Optional[str] = None
    fold_grid_sha256: Optional[str] = None
    seed_grid: Optional[str] = None
    split_manifest_sha256: Optional[str] = None
    homolog_map_name: Optional[str] = None
    homolog_map_version: Optional[str] = None
    homolog_map_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.builder_name:
            raise ValueError("builder_name must not be empty.")
        if not self.builder_version:
            raise ValueError("builder_version must not be empty.")

    def to_record(self) -> dict[str, Optional[str]]:
        return asdict(self)


class ProblemBuilderRegistry:
    """Registry of Namespace-free problem factories keyed by kind/data_id."""

    def __init__(self) -> None:
        self._factories: dict[tuple[str, str], ProblemFactory] = {}
        self._metadata: dict[tuple[str, str], ProblemBuilderMetadata] = {}

    def register(
        self,
        kind: ProblemKind,
        data_id: str,
        factory: ProblemFactory,
        *,
        metadata: ProblemBuilderMetadata | None = None,
    ) -> None:
        if kind not in {"endpoint", "trajectory", "single_time"}:
            raise ValueError(f"Unknown problem kind {kind!r}.")
        if not data_id:
            raise ValueError("data_id must not be empty.")
        key = (kind, str(data_id))
        self._factories[key] = factory
        if metadata is not None:
            self._metadata[key] = metadata

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

    def metadata(self, kind: ProblemKind, cfg_or_data_id: Any) -> ProblemBuilderMetadata:
        data_id = _data_id(cfg_or_data_id) if not isinstance(cfg_or_data_id, str) else cfg_or_data_id
        key = (kind, data_id)
        try:
            return self._metadata[key]
        except KeyError as exc:
            raise KeyError(
                f"No CREDO problem-builder metadata registered for kind={kind!r}, data_id={data_id!r}."
            ) from exc

    def clear(self) -> None:
        self._factories.clear()
        self._metadata.clear()


DEFAULT_PROBLEM_BUILDERS = ProblemBuilderRegistry()


def register_problem_builder(
    kind: ProblemKind,
    data_id: str,
    factory: ProblemFactory,
    *,
    metadata: ProblemBuilderMetadata | None = None,
    registry: ProblemBuilderRegistry = DEFAULT_PROBLEM_BUILDERS,
) -> None:
    """Register a project/package data adapter for config-driven search."""
    registry.register(kind, data_id, factory, metadata=metadata)


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


def problem_builder_metadata(
    kind: ProblemKind,
    cfg_or_data_id: Any,
    *,
    registry: ProblemBuilderRegistry = DEFAULT_PROBLEM_BUILDERS,
) -> ProblemBuilderMetadata:
    """Return registered reproducibility fingerprints for a problem builder."""
    return registry.metadata(kind, cfg_or_data_id)


def _data_id(cfg: Any) -> str:
    data_id: Optional[str] = getattr(cfg, "data_id", None)
    if data_id is None and isinstance(cfg, dict):
        data_id = cfg.get("data_id")
    if data_id is None or not str(data_id):
        raise ValueError("RunConfig.data_id is required to build a problem from config.")
    return str(data_id)


__all__ = [
    "DEFAULT_PROBLEM_BUILDERS",
    "ProblemBuilderMetadata",
    "ProblemBuilderRegistry",
    "ProblemFactory",
    "ProblemKind",
    "build_endpoint_problem_from_config",
    "build_single_time_problem_from_config",
    "build_trajectory_problem_from_config",
    "clear_problem_builders",
    "problem_builder_metadata",
    "register_problem_builder",
]
