"""Cycle-free semantic inner publication and outer final seal for Dev37."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..canonical import atomic_json, canonical_json_bytes, sha256_file
from ..contracts import (
    ArtifactRef,
    G00CDecisionReceiptV5,
    G00CExecutionBundleV5,
    G00CFinalSealV1,
    G00CInnerPublicationInventoryV4,
    G00CInnerPublicationManifestV4,
)
from ..errors import IntegrityError
from .g00c_v3 import _path

INNER_CONTROL_NAMES = {"artifacts.json", "SHA256SUMS", "COMMITTED", "PUBLICATION_EVENT.json"}
OUTER_NAMES = {
    "EXECUTION_BUNDLE.json",
    "INNER_PUBLICATION_MANIFEST.json",
    "DECISION_RECEIPT.json",
    "artifacts.json",
    "SHA256SUMS",
    "COMMITTED",
    "FINAL_SEAL.json",
}


def _parse_sha256sums(path: Path) -> dict[str, str]:
    answer: dict[str, str] = {}
    for line in path.read_text().splitlines():
        digest, separator, name = line.partition("  ")
        if separator != "  " or len(digest) != 64 or name in answer:
            raise IntegrityError("Dev37 SHA256SUMS is malformed or duplicated.")
        answer[name] = digest
    return answer


def _collect_refs(
    value: Any,
    prefix: str,
    answer: dict[tuple[str, str], tuple[str, ArtifactRef]],
) -> None:
    if isinstance(value, ArtifactRef):
        key = (value.sha256, value.relative_uri)
        answer.setdefault(key, (prefix, value))
        return
    if isinstance(value, BaseModel):
        for field in type(value).model_fields:
            _collect_refs(getattr(value, field), f"{prefix}.{field}", answer)
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _collect_refs(item, f"{prefix}.{index}", answer)
        return
    if isinstance(value, dict):
        for key in sorted(value):
            _collect_refs(value[key], f"{prefix}.{key}", answer)


def semantic_artifact_map_v4(models: Mapping[str, BaseModel]) -> dict[str, ArtifactRef]:
    """Return one deterministic semantic role for every distinct typed ArtifactRef."""

    collected: dict[tuple[str, str], tuple[str, ArtifactRef]] = {}
    for name in sorted(models):
        _collect_refs(models[name], name, collected)
    roles = {role: artifact for role, artifact in collected.values()}
    if len(roles) != len(collected):
        raise IntegrityError("Dev37 semantic artifact roles collide.")
    return dict(sorted(roles.items()))


def verify_g00c_inner_publication_v4(
    publication_root: Path,
    bundle: G00CExecutionBundleV5,
    manifest: G00CInnerPublicationManifestV4,
    *,
    expected_artifacts: Mapping[str, ArtifactRef],
) -> G00CInnerPublicationInventoryV4:
    """Require semantic role = bundle ref = inventory bytes = published payload."""

    if (
        manifest.execution_authority_id != bundle.execution_authority_id
        or manifest.selection_freeze_id != bundle.selection_freeze_id
        or manifest.feature_selection_result_id != bundle.feature_selection_result_id
        or manifest.sample_size_selection_result_id != bundle.sample_size_selection_result_id
        or manifest.terminal_status != bundle.terminal_status
    ):
        raise IntegrityError("Dev37 inner publication is cross-wired.")
    inventory_path = _path(publication_root, manifest.artifact_inventory)
    inventory = G00CInnerPublicationInventoryV4.model_validate_json(inventory_path.read_text())
    if inventory.inventory_id != manifest.artifact_inventory_id:
        raise IntegrityError("Dev37 inner inventory identity differs from its manifest.")
    canonical_inventory = canonical_json_bytes(inventory.model_dump(mode="json")) + b"\n"
    if (
        inventory_path.name != "artifacts.json"
        or inventory_path.read_bytes() != canonical_inventory
    ):
        raise IntegrityError("Dev37 on-disk artifacts.json differs from the bound inventory.")
    observed = {item.role: item for item in inventory.artifacts}
    if set(observed) != set(expected_artifacts):
        raise IntegrityError("Dev37 inner publication omits or adds a semantic artifact role.")
    names: set[str] = set()
    for role, expected in expected_artifacts.items():
        item = observed[role]
        if item.source_artifact != expected or item.filename in names:
            raise IntegrityError("Dev37 semantic role maps to another artifact or filename.")
        names.add(item.filename)
        payload = publication_root / item.filename
        if (
            payload.is_symlink()
            or not payload.is_file()
            or payload.stat().st_size != expected.size_bytes
            or sha256_file(payload) != expected.sha256
        ):
            raise IntegrityError(f"Dev37 published semantic artifact failed: {role}.")
    sums = _parse_sha256sums(_path(publication_root, manifest.sha256sums))
    expected_sums = {item.filename: item.source_artifact.sha256 for item in inventory.artifacts}
    if sums != expected_sums:
        raise IntegrityError("Dev37 inner SHA inventory differs from semantic artifacts.")
    if _path(publication_root, manifest.committed).read_text() != f"{bundle.terminal_status}\n":
        raise IntegrityError("Dev37 inner COMMITTED marker differs from terminal status.")
    event = json.loads(_path(publication_root, manifest.publication_event_receipt).read_text())
    if event != {
        "destination_preexisted": False,
        "directory_fsync_completed": True,
        "manifest_written_last": True,
        "no_clobber": True,
        "publisher_implementation_sha256": manifest.publisher_implementation_sha256,
        "status": "pass",
    }:
        raise IntegrityError("Dev37 inner publication event differs from the commit protocol.")
    actual = {
        path.relative_to(publication_root).as_posix()
        for path in publication_root.rglob("*")
        if path.is_file()
    }
    if actual != names | INNER_CONTROL_NAMES:
        raise IntegrityError("Dev37 inner publication contains missing or extra files.")
    return inventory


def publish_g00c_inner_v4(
    destination: Path,
    *,
    terminal_status: str,
    publisher_implementation_sha256: str,
    writer: Callable[[Path], G00CInnerPublicationManifestV4],
) -> None:
    """Publish the decision-free inner layer through a fresh sibling directory."""

    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        manifest = writer(temporary)
        if manifest.terminal_status != terminal_status:
            raise IntegrityError("Dev37 inner publisher received a cross-wired manifest.")
        atomic_json(
            temporary / "PUBLICATION_EVENT.json",
            {
                "destination_preexisted": False,
                "directory_fsync_completed": True,
                "manifest_written_last": True,
                "no_clobber": True,
                "publisher_implementation_sha256": publisher_implementation_sha256,
                "status": "pass",
            },
        )
        (temporary / "COMMITTED").write_text(f"{terminal_status}\n")
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, destination)
        descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def verify_g00c_final_seal_v1(
    seal_root: Path,
    seal: G00CFinalSealV1,
) -> tuple[G00CExecutionBundleV5, G00CInnerPublicationManifestV4, G00CDecisionReceiptV5]:
    """Verify the outer layer after bundle and decision identities exist."""

    if {
        path.relative_to(seal_root).as_posix() for path in seal_root.rglob("*") if path.is_file()
    } != OUTER_NAMES:
        raise IntegrityError("Dev37 outer seal contains missing or extra files.")
    bundle = G00CExecutionBundleV5.model_validate_json(
        _path(seal_root, seal.execution_bundle).read_text()
    )
    manifest = G00CInnerPublicationManifestV4.model_validate_json(
        _path(seal_root, seal.inner_publication_manifest).read_text()
    )
    decision = G00CDecisionReceiptV5.model_validate_json(
        _path(seal_root, seal.final_decision).read_text()
    )
    if (
        bundle.bundle_id != seal.execution_bundle_id
        or manifest.manifest_id != seal.inner_publication_manifest_id
        or decision.receipt_id != seal.final_decision_id
        or decision.execution_bundle_id != bundle.bundle_id
        or decision.inner_publication_manifest_id != manifest.manifest_id
        or bundle.inner_publication_manifest.sha256 != seal.inner_publication_manifest.sha256
    ):
        raise IntegrityError("Dev37 outer seal identities are cross-wired.")
    inventory_path = _path(seal_root, seal.outer_artifact_inventory)
    expected_inventory = {
        "decision": seal.final_decision.model_dump(mode="json"),
        "execution_bundle": seal.execution_bundle.model_dump(mode="json"),
        "inner_publication_manifest": seal.inner_publication_manifest.model_dump(mode="json"),
    }
    if json.loads(inventory_path.read_text()) != expected_inventory:
        raise IntegrityError("Dev37 outer artifacts.json differs from the sealed references.")
    expected_sums = {
        "DECISION_RECEIPT.json": seal.final_decision.sha256,
        "EXECUTION_BUNDLE.json": seal.execution_bundle.sha256,
        "INNER_PUBLICATION_MANIFEST.json": seal.inner_publication_manifest.sha256,
        "artifacts.json": seal.outer_artifact_inventory.sha256,
    }
    if _parse_sha256sums(_path(seal_root, seal.sha256sums)) != expected_sums:
        raise IntegrityError("Dev37 outer SHA inventory differs from the sealed references.")
    if _path(seal_root, seal.committed).read_text() != "sealed\n":
        raise IntegrityError("Dev37 outer COMMITTED marker differs from the fixed seal state.")
    on_disk = G00CFinalSealV1.model_validate_json((seal_root / "FINAL_SEAL.json").read_text())
    if on_disk != seal:
        raise IntegrityError("Dev37 on-disk final seal differs from the supplied seal.")
    return bundle, manifest, decision
