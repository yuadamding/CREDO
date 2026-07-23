"""Study codec protocol, registry, and storage-neutral resolver."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from importlib import metadata
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from .study import Study, VerificationLevel


@runtime_checkable
class StudyCodec(Protocol):
    codec_id: str
    readable_schema_versions: frozenset[int]
    writable_schema_versions: frozenset[int]

    def probe(self, source: Any) -> bool: ...

    def read(
        self,
        source: Any,
        *,
        verify: VerificationLevel = "semantic",
        **kwargs: Any,
    ) -> Study: ...

    def write(self, study: Study, destination: str | Path) -> Any: ...


class StudyCodecRegistry(Mapping[str, StudyCodec]):
    """Process-local immutable-by-ID codec registry."""

    def __init__(self) -> None:
        self._codecs: dict[str, StudyCodec] = {}

    def __getitem__(self, codec_id: str) -> StudyCodec:
        return self._codecs[str(codec_id)]

    def __iter__(self) -> Iterator[str]:
        return iter(self._codecs)

    def __len__(self) -> int:
        return len(self._codecs)

    def register(self, codec: StudyCodec) -> None:
        if not isinstance(codec, StudyCodec):
            raise TypeError("Study codec registration requires the complete StudyCodec protocol.")
        codec_id = str(codec.codec_id)
        if not codec_id:
            raise ValueError("Study codec_id must be nonempty.")
        existing = self._codecs.get(codec_id)
        if existing is not None and type(existing) is not type(codec):
            raise ValueError(f"Study codec {codec_id!r} is already registered.")
        self._codecs[codec_id] = codec

    @property
    def codecs(self) -> Mapping[str, StudyCodec]:
        return MappingProxyType(self._codecs)

    def resolve(self, source: Any) -> StudyCodec:
        _register_builtins()
        matches = [codec for codec in self._codecs.values() if codec.probe(source)]
        if not matches:
            raise ValueError(f"No registered Study codec recognizes {source!r}.")
        if len(matches) > 1:
            identifiers = sorted(codec.codec_id for codec in matches)
            raise ValueError(f"Study source is ambiguous across codecs: {identifiers}.")
        return matches[0]


study_codecs = StudyCodecRegistry()
_entry_points_discovered = False


def _register_builtins() -> None:
    from .legacy import CurrentFiveFileStudyCodec
    from .native import NativeStudyV3Codec

    if "credo.current_five_file" not in study_codecs:
        study_codecs.register(CurrentFiveFileStudyCodec())
    if "credo.native_study" not in study_codecs:
        study_codecs.register(NativeStudyV3Codec())
    _discover_entry_points()


def _discover_entry_points() -> None:
    global _entry_points_discovered
    if _entry_points_discovered:
        return
    _entry_points_discovered = True
    for entry_point in metadata.entry_points(group="credo.study_codecs"):
        codec = entry_point.load()
        codec = codec() if isinstance(codec, type) else codec
        study_codecs.register(codec)


def register_study_codec(codec: StudyCodec) -> None:
    study_codecs.register(codec)


def available_study_codecs() -> tuple[str, ...]:
    _register_builtins()
    return tuple(sorted(study_codecs))


def open_study(
    source: Any,
    *,
    verify: VerificationLevel = "semantic",
    **kwargs: Any,
) -> Study:
    """Probe registered codecs and open one semantic Study."""
    codec = study_codecs.resolve(source)
    return codec.read(source, verify=verify, **kwargs)


__all__ = [
    "StudyCodec",
    "StudyCodecRegistry",
    "available_study_codecs",
    "open_study",
    "register_study_codec",
    "study_codecs",
]
