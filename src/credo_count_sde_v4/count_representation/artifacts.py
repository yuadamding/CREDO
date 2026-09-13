"""No-clobber count-representation artifacts with separate versioned manifests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..canonical import sha256_file
from ..contracts.models import ArtifactRef
from ..data.prepared_shards import regular_member, verified_member
from ..errors import ContractError
from ..forecast.artifacts import write_json
from ..forecast.contracts import process_environment
from ..persistence.artifacts import publish_directory, verify_directory
from ..runtime_identity import environment_identity, implementation_tree_hash
from .contracts import CountRepresentationSpec, RepresentationManifest, numerical_settings


def runtime_check(spec: CountRepresentationSpec) -> None:
    CountRepresentationSpec.model_validate(spec.model_dump())
    if (
        spec.implementation_sha256 != implementation_tree_hash()
        or spec.environment != environment_identity()
        or spec.process_environment != process_environment()
        or spec.numerical_settings != numerical_settings()
    ):
        raise ContractError("Representation runtime differs from frozen specification.")


def output_check(root: Path, destination: Path) -> None:
    if destination.resolve().is_relative_to(root.resolve()):
        raise ContractError(
            "Representation outputs must remain outside the prepared input package."
        )
    if destination.exists():
        raise FileExistsError(destination)


def verify(root: Path, *, stage: str | None = None) -> RepresentationManifest:
    verify_directory(root)
    record = RepresentationManifest.model_validate_json(
        regular_member(root, "COMPLETE.json").read_text()
    )
    if stage is not None and record.stage != stage:
        raise ContractError("Representation stage mismatch.")
    names = {a.relative_uri for a in record.artifacts} | {
        "COMPLETE.json",
        "artifacts.json",
        "COMMITTED",
    }
    if names != {p.name for p in root.iterdir()}:
        raise ContractError("Unexpected representation payload set.")
    for item in record.artifacts:
        with verified_member(root, item):
            pass
    return record


def publish(
    destination: Path,
    spec: CountRepresentationSpec,
    *,
    stage: str,
    parents: dict[str, str],
    writer: Callable[[Path], dict[str, Any]],
) -> RepresentationManifest:
    def write(path: Path) -> None:
        facts = writer(path)
        write_json(path / "specification.json", spec.model_dump(mode="json"))
        artifacts = tuple(
            ArtifactRef(
                schema_id="credo.count_representation_payload",
                schema_version=1,
                relative_uri=p.name,
                size_bytes=p.stat().st_size,
                sha256=sha256_file(p),
                media_type="application/octet-stream",
            )
            for p in sorted(path.iterdir())
        )
        record = RepresentationManifest.model_validate(
            dict(
                stage=stage,
                specification_sha256=spec.identity(),
                parents=parents,
                artifacts=artifacts,
                facts=facts,
            )
        )
        write_json(path / "COMPLETE.json", record.model_dump(mode="json"))

    publish_directory(destination, write)
    return verify(destination, stage=stage)
