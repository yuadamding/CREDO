"""No-clobber immutable stage publication and full content verification."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..canonical import canonical_json_bytes, sha256_file
from ..contracts.models import ArtifactRef
from ..data.prepared_shards import regular_member, verified_member
from ..errors import ContractError
from ..persistence.artifacts import publish_directory, verify_directory
from .contracts import BundleManifest


def write_json(path: Path, value: Any) -> None:
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(value) + b"\n")


def verify_bundle(
    root: Path, *, stage: str | None = None, specification: str | None = None
) -> BundleManifest:
    verify_directory(root)
    marker = regular_member(root, "COMPLETE.json")
    result = BundleManifest.model_validate_json(marker.read_text())
    if (
        stage is not None
        and result.stage != stage
        or specification is not None
        and result.specification_sha256 != specification
    ):
        raise ContractError("Bundle stage or execution specification mismatch.")
    names = {a.relative_uri for a in result.artifacts} | {
        "COMPLETE.json",
        "artifacts.json",
        "COMMITTED",
    }
    if {p.name for p in root.iterdir()} != names:
        raise ContractError("Bundle contains missing or unlisted files.")
    for artifact in result.artifacts:
        with verified_member(root, artifact):
            pass
    return result


def publish_bundle(
    destination: Path,
    *,
    stage: str,
    specification: str,
    parents: dict[str, str],
    writer: Callable[[Path], dict[str, Any]],
) -> BundleManifest:
    def write(temporary: Path) -> None:
        facts = writer(temporary)
        artifacts = tuple(
            ArtifactRef(
                schema_id="credo.baseline_stage_payload",
                schema_version=1,
                sha256=sha256_file(path),
                size_bytes=path.stat().st_size,
                media_type="application/octet-stream",
                relative_uri=path.name,
            )
            for path in sorted(temporary.iterdir())
        )
        record = BundleManifest.model_validate(
            dict(
                stage=stage,
                specification_sha256=specification,
                parents=parents,
                artifacts=artifacts,
                facts=facts,
            )
        )
        write_json(temporary / "COMPLETE.json", record.model_dump(mode="json"))

    publish_directory(destination, write)
    return verify_bundle(destination, stage=stage, specification=specification)


def read_spec(root: Path) -> dict[str, Any]:
    with regular_member(root, "specification.json").open() as handle:
        return dict(json.load(handle))
