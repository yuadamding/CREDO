"""Typed representation and artifact catalogs."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal


def _sha256(value: str, field_name: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{field_name} must be a SHA-256 hex digest.")
    return digest


@dataclass(frozen=True)
class ArtifactRef:
    """Content-addressed reference to one representation artifact."""

    artifact_id: str
    uri: str
    sha256: str
    size_bytes: int | None
    media_type: str
    semantic_hash: str | None = None

    def __post_init__(self) -> None:
        for name in ("artifact_id", "uri", "media_type"):
            value = str(getattr(self, name))
            if not value:
                raise ValueError(f"ArtifactRef.{name} must be nonempty.")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "sha256", _sha256(self.sha256, "ArtifactRef.sha256"))
        if self.semantic_hash is not None:
            object.__setattr__(
                self,
                "semantic_hash",
                _sha256(self.semantic_hash, "ArtifactRef.semantic_hash"),
            )
        if self.size_bytes is not None and int(self.size_bytes) < 0:
            raise ValueError("ArtifactRef.size_bytes must be nonnegative when provided.")
        if self.size_bytes is not None:
            object.__setattr__(self, "size_bytes", int(self.size_bytes))


@dataclass(frozen=True)
class RepresentationSpec:
    """Coordinates, fit scope, and artifacts for one study representation.

    ``included_series`` and ``included_checkpoints`` record the observations used
    to fit the representation. They do not limit which encoded supports may be
    present in the associated store.
    """

    representation_id: str
    backend: str
    space_kind: Literal["latent", "expression", "token", "multimodal"]
    dimension: int
    support_store_id: str
    support_artifact: ArtifactRef | None = None
    feature_artifact: ArtifactRef | None = None
    encoder_artifact: ArtifactRef | None = None
    decoder_artifact: ArtifactRef | None = None
    normalization_artifact: ArtifactRef | None = None
    fit_split_id: str | None = None
    included_series: tuple[str, ...] = ()
    included_checkpoints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("representation_id", "backend", "support_store_id"):
            value = str(getattr(self, name))
            if not value:
                raise ValueError(f"RepresentationSpec.{name} must be nonempty.")
            object.__setattr__(self, name, value)
        if self.space_kind not in {"latent", "expression", "token", "multimodal"}:
            raise ValueError(f"Unsupported representation space_kind {self.space_kind!r}.")
        if int(self.dimension) < 1:
            raise ValueError("RepresentationSpec.dimension must be positive.")
        object.__setattr__(self, "dimension", int(self.dimension))
        if self.fit_split_id is not None:
            fit_split_id = str(self.fit_split_id)
            if not fit_split_id:
                raise ValueError("RepresentationSpec.fit_split_id must be nonempty when set.")
            object.__setattr__(self, "fit_split_id", fit_split_id)
        for name in ("included_series", "included_checkpoints"):
            values = tuple(str(value) for value in getattr(self, name))
            if any(not value for value in values) or len(values) != len(set(values)):
                raise ValueError(f"RepresentationSpec.{name} must contain unique nonempty IDs.")
            object.__setattr__(self, name, values)


class RepresentationCatalog(Mapping[str, RepresentationSpec]):
    """Immutable collection of named representation variants."""

    def __init__(
        self,
        representations: Mapping[str, RepresentationSpec] | tuple[RepresentationSpec, ...],
    ) -> None:
        if isinstance(representations, Mapping):
            values = tuple(representations.values())
            mismatched = [
                key for key, value in representations.items() if str(key) != value.representation_id
            ]
            if mismatched:
                raise ValueError(
                    "RepresentationCatalog mapping keys must equal representation_id; "
                    f"invalid={mismatched[:5]}."
                )
        else:
            values = tuple(representations)
        catalog = {value.representation_id: value for value in values}
        if not catalog:
            raise ValueError("RepresentationCatalog requires at least one representation.")
        if len(catalog) != len(values):
            raise ValueError("Representation identifiers must be unique.")
        self._catalog = MappingProxyType(catalog)

    def __getitem__(self, representation_id: str) -> RepresentationSpec:
        return self._catalog[str(representation_id)]

    def __iter__(self) -> Iterator[str]:
        return iter(self._catalog)

    def __len__(self) -> int:
        return len(self._catalog)

    @property
    def representation_ids(self) -> tuple[str, ...]:
        return tuple(self._catalog)


__all__ = ["ArtifactRef", "RepresentationCatalog", "RepresentationSpec"]
