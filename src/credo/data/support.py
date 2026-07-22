"""Backend-neutral empirical-law and support-store contracts."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .validation import ValidationIssue, ValidationReport


@dataclass(frozen=True, order=True)
class SupportRef:
    """Stable reference to one empirical law in one store and representation."""

    store_id: str
    representation_id: str
    support_key: str

    def __post_init__(self) -> None:
        for name in ("store_id", "representation_id", "support_key"):
            value = str(getattr(self, name))
            if not value:
                raise ValueError(f"SupportRef.{name} must be nonempty.")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class EmpiricalLaw:
    """Conditional empirical law with probabilities summing to one."""

    coordinates: np.ndarray
    probabilities: np.ndarray

    def __post_init__(self) -> None:
        coordinates = np.array(self.coordinates, dtype=np.float32, copy=True)
        probabilities = np.array(self.probabilities, dtype=np.float64, copy=True).reshape(-1)
        if coordinates.ndim != 2 or coordinates.shape[0] == 0:
            raise ValueError("EmpiricalLaw.coordinates must have shape [n_atoms, dimension].")
        if len(probabilities) != len(coordinates):
            raise ValueError("EmpiricalLaw coordinates and probabilities must have equal length.")
        if not np.isfinite(coordinates).all():
            raise ValueError("EmpiricalLaw coordinates contain non-finite values.")
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
            raise ValueError("EmpiricalLaw probabilities must be finite and nonnegative.")
        if not np.isclose(probabilities.sum(), 1.0, rtol=1e-7, atol=1e-10):
            raise ValueError("EmpiricalLaw probabilities must sum to one.")
        coordinates.setflags(write=False)
        probabilities.setflags(write=False)
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "probabilities", probabilities)

    @property
    def dimension(self) -> int:
        return int(self.coordinates.shape[1])


@dataclass(frozen=True)
class AbundanceValue:
    """One selected abundance observation, independent of support geometry."""

    observation_id: str
    channel_id: str
    value: float | None
    observed: bool
    denominator_id: str | None = None
    transform_id: str | None = None
    source_artifact_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("observation_id", "channel_id"):
            value = str(getattr(self, name))
            if not value:
                raise ValueError(f"AbundanceValue.{name} must be nonempty.")
            object.__setattr__(self, name, value)
        if self.observed:
            if self.value is None or not np.isfinite(float(self.value)) or float(self.value) < 0:
                raise ValueError("Observed abundance must be finite and nonnegative.")
            object.__setattr__(self, "value", float(self.value))
        elif self.value is not None:
            value = float(self.value)
            if not np.isfinite(value) or value < 0:
                raise ValueError("Unobserved abundance values must be finite and nonnegative.")
            object.__setattr__(self, "value", value)
        for name in ("denominator_id", "transform_id", "source_artifact_id"):
            value = getattr(self, name)
            if value is not None:
                normalized = str(value)
                if not normalized:
                    raise ValueError(f"AbundanceValue.{name} must be nonempty when provided.")
                object.__setattr__(self, name, normalized)


@dataclass(frozen=True)
class MeasureSnapshot:
    """Geometry and one abundance channel for a single observation."""

    observation_id: str
    law: EmpiricalLaw | None
    abundance: AbundanceValue | None


@runtime_checkable
class SupportStore(Protocol):
    """Storage-independent access to normalized empirical laws."""

    @property
    def store_id(self) -> str: ...

    def representation_ids(self) -> tuple[str, ...]: ...

    def dimension(self, representation_id: str) -> int: ...

    def contains(self, ref: SupportRef) -> bool: ...

    def read(self, ref: SupportRef) -> EmpiricalLaw: ...

    def read_many(self, refs: Sequence[SupportRef]) -> Mapping[SupportRef, EmpiricalLaw]: ...

    def validate(self, *, full_scan: bool = False) -> ValidationReport: ...

    def close(self) -> None: ...


class SupportStoreRegistry(Mapping[str, SupportStore]):
    """Immutable routing table for one or more support backends."""

    def __init__(self, stores: Mapping[str, SupportStore] | Sequence[SupportStore]) -> None:
        values = tuple(stores.values()) if isinstance(stores, Mapping) else tuple(stores)
        catalog = {store.store_id: store for store in values}
        if not catalog:
            raise ValueError("SupportStoreRegistry requires at least one store.")
        if len(catalog) != len(values):
            raise ValueError("Support store identifiers must be unique.")
        if isinstance(stores, Mapping):
            mismatched = [key for key, store in stores.items() if str(key) != store.store_id]
            if mismatched:
                raise ValueError(
                    f"SupportStoreRegistry keys must equal store_id; invalid={mismatched[:5]}."
                )
        self._stores = MappingProxyType(catalog)

    def __getitem__(self, store_id: str) -> SupportStore:
        return self._stores[str(store_id)]

    def __iter__(self) -> Iterator[str]:
        return iter(self._stores)

    def __len__(self) -> int:
        return len(self._stores)

    @property
    def store_ids(self) -> tuple[str, ...]:
        return tuple(self._stores)

    def contains(self, ref: SupportRef) -> bool:
        store = self._stores.get(ref.store_id)
        return store is not None and store.contains(ref)

    def read(self, ref: SupportRef) -> EmpiricalLaw:
        try:
            store = self._stores[ref.store_id]
        except KeyError as exc:
            raise KeyError(f"Unknown support store {ref.store_id!r}.") from exc
        return store.read(ref)

    def read_many(self, refs: Sequence[SupportRef]) -> Mapping[SupportRef, EmpiricalLaw]:
        return MappingProxyType({ref: self.read(ref) for ref in refs})

    def validate(self, *, full_scan: bool = False) -> ValidationReport:
        report = ValidationReport()
        for store in self._stores.values():
            report = report.merged(store.validate(full_scan=full_scan))
        return report

    def close(self) -> None:
        for store in self._stores.values():
            store.close()


class InMemorySupportStore:
    """Small immutable support backend for tests and simulations."""

    def __init__(self, store_id: str, laws: Mapping[SupportRef, EmpiricalLaw]) -> None:
        self._store_id = str(store_id)
        if not self._store_id:
            raise ValueError("InMemorySupportStore.store_id must be nonempty.")
        normalized = dict(laws)
        if not normalized:
            raise ValueError("InMemorySupportStore requires at least one empirical law.")
        dimensions: dict[str, int] = {}
        for ref, law in normalized.items():
            if not isinstance(ref, SupportRef) or not isinstance(law, EmpiricalLaw):
                raise TypeError("InMemorySupportStore requires SupportRef to EmpiricalLaw entries.")
            existing = dimensions.setdefault(ref.representation_id, law.dimension)
            if existing != law.dimension:
                raise ValueError(
                    f"Representation {ref.representation_id!r} has inconsistent dimensions."
                )
        self._laws = MappingProxyType(normalized)
        self._dimensions = MappingProxyType(dimensions)
        self._closed = False

    @property
    def store_id(self) -> str:
        return self._store_id

    def representation_ids(self) -> tuple[str, ...]:
        return tuple(self._dimensions)

    def dimension(self, representation_id: str) -> int:
        return self._dimensions[str(representation_id)]

    def contains(self, ref: SupportRef) -> bool:
        return not self._closed and ref.store_id == self._store_id and ref in self._laws

    def read(self, ref: SupportRef) -> EmpiricalLaw:
        if self._closed:
            raise RuntimeError("Support store is closed.")
        try:
            return self._laws[ref]
        except KeyError as exc:
            raise KeyError(f"Unknown support reference {ref!r}.") from exc

    def read_many(self, refs: Sequence[SupportRef]) -> Mapping[SupportRef, EmpiricalLaw]:
        return MappingProxyType({ref: self.read(ref) for ref in refs})

    def validate(self, *, full_scan: bool = False) -> ValidationReport:
        del full_scan
        if self._closed:
            return ValidationReport(
                (ValidationIssue("error", "support.closed", "Support store is closed."),)
            )
        return ValidationReport()

    def close(self) -> None:
        self._closed = True


class LegacyFiniteMeasureSupportStore:
    """Compatibility adapter from finite measures to normalized empirical laws."""

    def __init__(
        self,
        *,
        store_id: str,
        representation_id: str,
        latent_dim: int,
        measures: Mapping[str, Mapping[str, Any]],
        support_pairs: Mapping[str, tuple[str, str]],
    ) -> None:
        self._store_id = str(store_id)
        self._representation_id = str(representation_id)
        self._latent_dim = int(latent_dim)
        if not self._store_id or not self._representation_id or self._latent_dim < 1:
            raise ValueError("Legacy support identifiers and latent dimension must be valid.")
        self._measures = measures
        self._support_pairs = MappingProxyType(
            {str(key): (str(pair[0]), str(pair[1])) for key, pair in support_pairs.items()}
        )
        self._closed = False

    @property
    def store_id(self) -> str:
        return self._store_id

    def representation_ids(self) -> tuple[str, ...]:
        return (self._representation_id,)

    def dimension(self, representation_id: str) -> int:
        if str(representation_id) != self._representation_id:
            raise KeyError(f"Unknown representation_id {representation_id!r}.")
        return self._latent_dim

    def contains(self, ref: SupportRef) -> bool:
        return (
            not self._closed
            and ref.store_id == self._store_id
            and ref.representation_id == self._representation_id
            and ref.support_key in self._support_pairs
        )

    def read(self, ref: SupportRef) -> EmpiricalLaw:
        if self._closed:
            raise RuntimeError("Support store is closed.")
        if not self.contains(ref):
            raise KeyError(f"Unknown support reference {ref!r}.")
        measure = self.finite_measure(ref)
        probabilities = np.asarray(measure.weights, dtype=np.float64) / float(measure.total_mass)
        return EmpiricalLaw(measure.support, probabilities)

    def finite_measure(self, ref: SupportRef) -> Any:
        """Return the original compatibility measure without numerical round-tripping."""
        if not self.contains(ref):
            raise KeyError(f"Unknown support reference {ref!r}.")
        checkpoint_id, series_id = self._support_pairs[ref.support_key]
        return self._measures[checkpoint_id][series_id]

    def read_many(self, refs: Sequence[SupportRef]) -> Mapping[SupportRef, EmpiricalLaw]:
        return MappingProxyType({ref: self.read(ref) for ref in refs})

    def validate(self, *, full_scan: bool = False) -> ValidationReport:
        if self._closed:
            return ValidationReport(
                (ValidationIssue("error", "support.closed", "Support store is closed."),)
            )
        issues: list[ValidationIssue] = []
        available = {
            str(checkpoint_id): set(self._measures[checkpoint_id])
            for checkpoint_id in self._measures
        }
        for support_key, (checkpoint_id, series_id) in self._support_pairs.items():
            if checkpoint_id not in available or series_id not in available[checkpoint_id]:
                issues.append(
                    ValidationIssue(
                        "error",
                        "support.missing",
                        f"Support key {support_key!r} has no legacy finite measure.",
                    )
                )
            elif full_scan:
                ref = SupportRef(self._store_id, self._representation_id, support_key)
                try:
                    law = self.read(ref)
                except (KeyError, TypeError, ValueError) as exc:
                    issues.append(
                        ValidationIssue("error", "support.invalid", f"{support_key!r}: {exc}")
                    )
                else:
                    if law.dimension != self._latent_dim:
                        issues.append(
                            ValidationIssue(
                                "error",
                                "support.dimension",
                                f"Support key {support_key!r} has dimension {law.dimension}.",
                            )
                        )
        return ValidationReport(tuple(issues))

    def close(self) -> None:
        close = getattr(self._measures, "close", None)
        if callable(close):
            close()
        self._closed = True


__all__ = [
    "AbundanceValue",
    "EmpiricalLaw",
    "InMemorySupportStore",
    "LegacyFiniteMeasureSupportStore",
    "MeasureSnapshot",
    "SupportRef",
    "SupportStore",
    "SupportStoreRegistry",
]
