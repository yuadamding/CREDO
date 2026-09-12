from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel, TypeAdapter
from scipy import sparse

from credo_count_sde_v4.canonical import canonical_json_bytes, contract_id, sha256_file
from credo_count_sde_v4.contracts import (
    ArtifactRef,
    FoldRowRoleRecord,
    G00CCommonSupportPriorV2,
    G00CD1ExecutionAuthorityFreezeV3,
    G00CDecisionReceiptV5,
    G00CDurableRestartReceiptV4,
    G00CExecutionBundleV5,
    G00CFeatureRankingReceiptV4,
    G00CFeatureSelectionResultV3,
    G00CFinalSealV1,
    G00CHierarchyDerivationReceiptV4,
    G00CImplementationAuthorityV3,
    G00CImplementationBindingV3,
    G00CInnerPublicationInventoryV4,
    G00CInnerPublicationManifestV4,
    G00CMaterializationReceiptV3,
    G00CMaterializationReceiptV4,
    G00CMonitorFreezeV1,
    G00CProcessTreeMonitorReceiptV4,
    G00CRefitReplayReceiptV4,
    G00CSamplerEvidenceV4,
    G00CSamplerPlanEntryV4,
    G00CSamplerPlanV4,
    G00CSampleSizeExtensionFreezeV2,
    G00CSemanticPublicationArtifactV4,
    G00CSourceAccessLedgerReceiptV4,
    G00CSourceFileBindingV4,
    G00CSourcePlaneBindingV4,
    G00CSupportAuditContractV2,
    G00CSupportAuditReceiptV3,
    VirtualCanonicalCountStoreManifestV1,
    VirtualCountSourceV1,
)
from credo_count_sde_v4.errors import IntegrityError
from credo_count_sde_v4.store.g00c_evidence_v4 import (
    ACCESS_COLUMNS,
    MONITOR_COLUMNS,
    verify_g00c_monitor_v4,
    verify_g00c_restart_v4,
    verify_g00c_source_access_v4,
)
from credo_count_sde_v4.store.g00c_extension_v4 import (
    verify_g00c_extension_freeze_v2,
)
from credo_count_sde_v4.store.g00c_materialization_v3 import _physical_runs
from credo_count_sde_v4.store.g00c_materialization_v4 import (
    verify_g00c_materialization_v4,
)
from credo_count_sde_v4.store.g00c_publication_v4 import (
    _parse_sha256sums,
    publish_g00c_inner_v4,
    semantic_artifact_map_v4,
    verify_g00c_final_seal_v1,
    verify_g00c_inner_publication_v4,
)
from credo_count_sde_v4.store.g00c_refit_replay_v4 import (
    REPLAY_COLUMNS,
    _checkpoint_codes,
    _thin_and_accumulate,
    derive_checkpoint_statistics_v4,
    recompute_refit_rows_v4,
    verify_g00c_refit_replay_v4,
)
from credo_count_sde_v4.store.g00c_sampler_v3 import (
    derive_support_table_v3,
    replay_sampler_plan_v3,
)
from credo_count_sde_v4.store.g00c_sampler_v4 import (
    _implementation_hash as _sampler_implementation_hash,
)
from credo_count_sde_v4.store.g00c_sampler_v4 import verify_g00c_sampler_v4, verify_g00c_support_v4
from credo_count_sde_v4.store.g00c_selection import (
    checkpoint_multinomial_refit_common_support,
    derive_refit_seed_schedule,
)
from credo_count_sde_v4.store.g00c_source_v4 import (
    RANKING_COLUMNS,
    _canonical_feature_records,
    _locate_source_indices,
    _source_feature_ids,
    derive_hierarchy_from_g00b_v4,
    open_verified_g00b_v4,
    open_verified_source_plane_v4,
    recompute_feature_ranking_v4,
    verify_g00c_feature_ranking_v4,
    verify_g00c_hierarchy_v4,
)
from credo_count_sde_v4.store.g00c_v2 import _expected_physical_order
from credo_count_sde_v4.store.g00c_v3 import REFIT_V3_COLUMNS, _ordered_row_hash
from credo_count_sde_v4.store.g00c_v5 import (
    VerifiedG00CExecutionV5,
    _artifact_refs,
    _expected_access_rows,
    _semantic_models,
    _verify_refit_binding,
    build_g00c_decision_receipt_v5,
)
from credo_count_sde_v4.store.g00c_v5 import (
    _implementation_hash as _v5_implementation_hash,
)
from credo_count_sde_v4.store.virtual import VirtualCanonicalCountStore
from tests.unit.test_virtual_store import (
    _artifact,
    _rewrite_v2_manifest,
    _upgrade_virtual_store_to_v2,
)


def _identified(model: type[Any], payload: dict[str, Any], id_field: str) -> Any:
    normalized = {
        name: TypeAdapter(field.annotation).validate_python(payload[name])
        for name, field in model.model_fields.items()
        if name in payload
    }
    provisional = model.model_construct(**normalized)
    payload[id_field] = provisional.identity(id_field=id_field)
    return model.model_validate(payload)


def _write_h5ad(path: Path, matrix: np.ndarray, feature_ids: list[str]) -> None:
    csr = sparse.csr_matrix(matrix, dtype=np.int32)
    with h5py.File(path, "x") as handle:
        group = handle.create_group("X")
        group.create_dataset("data", data=csr.data)
        group.create_dataset("indices", data=csr.indices.astype(np.int32))
        group.create_dataset("indptr", data=csr.indptr.astype(np.int64))
        var = handle.create_group("var")
        var.create_dataset("_index", data=np.asarray(feature_ids, dtype=h5py.string_dtype()))


def _write_compact_h5(
    path: Path,
    *,
    store: VirtualCanonicalCountStore,
    row_ids: np.ndarray,
    feature_indices: np.ndarray,
    feature_ids: list[str],
) -> None:
    matrix = store.rows(row_ids).matrix[:, feature_indices].tocsr()
    with h5py.File(path, "x", libver="earliest") as handle:
        handle.create_dataset("row_ids", data=np.asarray(row_ids, dtype=np.int64))
        handle.create_dataset(
            "feature_ids", data=np.asarray(feature_ids, dtype=h5py.string_dtype())
        )
        handle.create_dataset("indptr", data=np.asarray(matrix.indptr, dtype=np.int64))
        handle.create_dataset("indices", data=np.asarray(matrix.indices, dtype=np.int32))
        handle.create_dataset("data", data=np.asarray(matrix.data, dtype=np.int32))


def _build_real_g00b(tmp_path: Path) -> tuple[VirtualCanonicalCountStore, Path, Path]:
    """Create a real, fully verified V2 store with all three checkpoints and 4,097 genes."""

    source_root = tmp_path / "sources"
    source_root.mkdir()
    canonical_ids = [f"g{index:04d}" for index in range(4096)] + ["CUSTOM001_PuroR"]
    identity = np.arange(4097, dtype=np.int32)
    rotated = np.r_[np.arange(1, 4097), 0].astype(np.int32)
    reversed_order = np.arange(4096, -1, -1, dtype=np.int32)
    physical_ids = (
        canonical_ids,
        [canonical_ids[-1], *canonical_ids[:-1]],
        list(reversed(canonical_ids)),
    )
    canonical_rows = (
        np.asarray([[60, 1, 2], [60, 3, 4]], dtype=np.int32),
        np.asarray([[60, 5, 6]], dtype=np.int32),
        np.asarray([[60, 7, 8]], dtype=np.int32),
    )
    permutations = (identity, rotated, reversed_order)
    source_matrices: list[np.ndarray] = []
    for rows, permutation in zip(canonical_rows, permutations, strict=True):
        canonical = np.zeros((len(rows), 4097), dtype=np.int32)
        canonical[:, :3] = rows
        inverse = np.argsort(permutation)
        source_matrices.append(canonical[:, inverse])
    source_specs = (
        ("D2_Rest", "D2", "Rest", 0.0, "rest.h5ad"),
        ("D3_Stim8hr", "D3", "Stim8hr", 8.0, "stim8.h5ad"),
        ("D4_Stim48hr", "D4", "Stim48hr", 48.0, "stim48.h5ad"),
    )
    source_paths: list[Path] = []
    for spec, matrix, ids in zip(source_specs, source_matrices, physical_ids, strict=True):
        path = source_root / spec[-1]
        _write_h5ad(path, matrix, ids)
        source_paths.append(path)

    store_root = tmp_path / "virtual"
    store_root.mkdir()
    locator = store_root / "row-locator.h5"
    with h5py.File(locator, "x") as handle:
        handle.create_dataset("row_ids_sorted", data=np.asarray([10, 20, 30, 40], dtype=np.int64))
        handle.create_dataset(
            "source_indices_sorted", data=np.asarray([0, 0, 1, 2], dtype=np.int16)
        )
        handle.create_dataset("source_rows_sorted", data=np.asarray([0, 1, 0, 0], dtype=np.int64))
        handle.create_dataset("guide_codes_sorted", data=np.asarray([0, 1, 0, 1], dtype=np.int32))
        handle.create_dataset("target_codes_sorted", data=np.asarray([0, 1, 0, 1], dtype=np.int32))
        handle.create_dataset(
            "guide_ids", data=np.asarray(["guide-a", "guide-b"], dtype=h5py.string_dtype())
        )
        handle.create_dataset(
            "target_ids", data=np.asarray(["target-a", "target-b"], dtype=h5py.string_dtype())
        )
    permutation_path = store_root / "feature-permutations.npz"
    np.savez(
        permutation_path,
        source_000000=identity,
        source_000001=rotated,
        source_000002=reversed_order,
    )
    sources = tuple(
        VirtualCountSourceV1(
            source_id=spec[0],
            donor_id=spec[1],
            checkpoint=spec[2],
            physical_time_hours=spec[3],
            relative_uri=spec[4],
            source_file_sha256=sha256_file(path),
            rows=matrix.shape[0],
            features=4097,
            nnz=matrix.nonzero()[0].size,
            eligible_rows=matrix.shape[0],
            eligible_nnz=matrix.nonzero()[0].size,
            source_feature_order_hash=hashlib.sha256(canonical_json_bytes(ids)).hexdigest(),
            canonical_permutation_hash=hashlib.sha256(
                permutation.astype("<i4").tobytes()
            ).hexdigest(),
        )
        for spec, path, matrix, ids, permutation in zip(
            source_specs,
            source_paths,
            source_matrices,
            physical_ids,
            permutations,
            strict=True,
        )
    )
    feature_records = [
        {"feature_id": value, "namespace": "test", "namespace_version": "1"}
        for value in canonical_ids
    ]
    feature_hash = hashlib.sha256(canonical_json_bytes(feature_records)).hexdigest()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "virtual_store_id": "pending",
        "source_authority_id": "authority-1",
        "canonical_feature_index_hash": feature_hash,
        "guide_catalog_hash": hashlib.sha256(
            canonical_json_bytes(["guide-a", "guide-b"])
        ).hexdigest(),
        "target_catalog_hash": hashlib.sha256(
            canonical_json_bytes(["target-a", "target-b"])
        ).hexdigest(),
        "eligibility_rule": "guide_group == targeting single sgRNA AND low_quality == false",
        "eligible_rows": 4,
        "features": 4097,
        "row_locator": _artifact(
            locator, root=store_root, media_type="application/x-hdf5"
        ).model_dump(mode="json"),
        "feature_permutations": _artifact(
            permutation_path, root=store_root, media_type="application/x-npz"
        ).model_dump(mode="json"),
        "sources": [item.model_dump(mode="json") for item in sources],
    }
    normalized = VirtualCanonicalCountStoreManifestV1.model_construct(
        **{
            name: TypeAdapter(field.annotation).validate_python(payload[name])
            for name, field in VirtualCanonicalCountStoreManifestV1.model_fields.items()
            if name in payload
        }
    ).model_dump(mode="json")
    payload["virtual_store_id"] = contract_id(normalized, id_field="virtual_store_id")
    manifest = VirtualCanonicalCountStoreManifestV1.model_validate(payload)
    (store_root / "manifest.json").write_text(manifest.model_dump_json() + "\n")
    store = _upgrade_virtual_store_to_v2(
        VirtualCanonicalCountStore(store_root, source_root=source_root)
    )
    manifest_payload = store.manifest.model_dump(mode="json")
    manifest_payload["source_authority_id"] = "a" * 64
    store = _rewrite_v2_manifest(store, manifest_payload)
    store.verify(full=True)
    feature_index = tmp_path / "FEATURE_INDEX.json"
    feature_index.write_text(
        json.dumps(
            {
                "feature_count": len(feature_records),
                "features": feature_records,
                "ordered_hash": feature_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    return store, source_root, feature_index


def _binding(
    authority_root: Path,
    store: VirtualCanonicalCountStore,
    feature_index: Path,
) -> G00CSourcePlaneBindingV4:
    parent = authority_root / "accepted-g00b.json"
    parent.write_bytes((store.path / "manifest.json").read_bytes())
    target = authority_root / "FEATURE_INDEX.json"
    target.write_bytes(feature_index.read_bytes())
    manifest = store.manifest
    payload = {
        "binding_id": "0" * 64,
        "accepted_g00b_parent": _artifact(
            parent, root=authority_root, media_type="application/json"
        ),
        "accepted_g00b_manifest_sha256": sha256_file(parent),
        "virtual_store_id": manifest.virtual_store_id,
        "source_authority_id": manifest.source_authority_id,
        "virtual_store_relative_uri": store.path.name,
        "canonical_feature_index": _artifact(
            target, root=authority_root, media_type="application/json"
        ),
        "canonical_feature_index_hash": manifest.canonical_feature_index_hash,
        "row_locator_sha256": manifest.row_locator.sha256,
        "feature_permutations_sha256": manifest.feature_permutations.sha256,
        "guide_target_crosswalk_sha256": manifest.guide_target_crosswalk.sha256,
        "source_files": tuple(
            G00CSourceFileBindingV4(
                source_id=item.source_id,
                checkpoint=item.checkpoint,
                source_file_sha256=item.source_file_sha256,
            )
            for item in manifest.sources
        ),
        "feature_count": manifest.features,
        "puro_r_canonical_index": 4096,
    }
    return _identified(G00CSourcePlaneBindingV4, payload, "binding_id")


def _build_extension_stop(
    root: Path,
) -> tuple[G00CExecutionBundleV5, G00CDecisionReceiptV5, G00CFinalSealV1, ArtifactRef]:
    """Build one valid outer-sealed extension stop without reading expression values."""

    root.mkdir()
    external = root.parent / "extension-stop-external"
    external.mkdir()
    payload = external / "payload.bin"
    payload.write_bytes(b"extension-stop-evidence")
    dummy = _artifact(payload, root=external, media_type="application/octet-stream")
    inventory_path = external / "inner-artifacts.json"
    inventory_path.write_text("{}\n")
    sums_path = external / "inner-SHA256SUMS"
    sums_path.write_text("")
    inner_committed = external / "inner-COMMITTED"
    inner_committed.write_text("extension_required\n")
    event_path = external / "inner-PUBLICATION_EVENT.json"
    event_path.write_text("{}\n")
    manifest = _identified(
        G00CInnerPublicationManifestV4,
        {
            "manifest_id": "0" * 64,
            "execution_authority_id": "a" * 64,
            "selection_freeze_id": "b" * 64,
            "feature_selection_result_id": "c" * 64,
            "sample_size_selection_result_id": "d" * 64,
            "sampler_evidence_id": "e" * 64,
            "terminal_status": "extension_required",
            "artifact_inventory": _artifact(
                inventory_path, root=external, media_type="application/json"
            ),
            "artifact_inventory_id": "f" * 64,
            "sha256sums": _artifact(sums_path, root=external, media_type="text/plain"),
            "committed": _artifact(inner_committed, root=external, media_type="text/plain"),
            "publication_event_receipt": _artifact(
                event_path, root=external, media_type="application/json"
            ),
            "publisher_implementation_sha256": "1" * 64,
        },
        "manifest_id",
    )
    manifest_path = root / "INNER_PUBLICATION_MANIFEST.json"
    manifest_path.write_text(manifest.model_dump_json() + "\n")
    manifest_ref = _artifact(manifest_path, root=root, media_type="application/json")
    bundle = _identified(
        G00CExecutionBundleV5,
        {
            "bundle_id": "0" * 64,
            "execution_authority": dummy,
            "execution_authority_id": manifest.execution_authority_id,
            "selection_freeze": dummy,
            "selection_freeze_id": manifest.selection_freeze_id,
            "seed_schedule": dummy,
            "seed_schedule_id": "2" * 64,
            "row_roles": dummy,
            "feature_ranking_receipt": dummy,
            "feature_selection_result": dummy,
            "feature_selection_result_id": manifest.feature_selection_result_id,
            "sample_size_selection_result": dummy,
            "sample_size_selection_result_id": manifest.sample_size_selection_result_id,
            "common_support_receipt": dummy,
            "feature_refit_replay_receipt": dummy,
            "sample_refit_replay_receipt": dummy,
            "sampler_evidence": dummy,
            "sampler_restart_receipt": dummy,
            "source_access_ledger_receipt": dummy,
            "monitor_receipt": dummy,
            "base_support_audit_contract": dummy,
            "base_support_audit_receipt": dummy,
            "inner_publication_manifest": manifest_ref,
            "terminal_status": "extension_required",
        },
        "bundle_id",
    )
    bundle_path = root / "EXECUTION_BUNDLE.json"
    bundle_path.write_text(bundle.model_dump_json() + "\n")
    bundle_ref = _artifact(bundle_path, root=root, media_type="application/json")
    decision = _identified(
        G00CDecisionReceiptV5,
        {
            "receipt_id": "0" * 64,
            "execution_bundle_id": bundle.bundle_id,
            "execution_authority_id": bundle.execution_authority_id,
            "selection_freeze_id": bundle.selection_freeze_id,
            "feature_selection_result_id": bundle.feature_selection_result_id,
            "sample_size_selection_result_id": bundle.sample_size_selection_result_id,
            "feature_ranking_receipt_id": "3" * 64,
            "sampler_evidence_id": manifest.sampler_evidence_id,
            "sampler_restart_receipt_id": "4" * 64,
            "feature_refit_replay_receipt_id": "5" * 64,
            "sample_refit_replay_receipt_id": "6" * 64,
            "support_audit_receipt_ids": ("7" * 64,),
            "source_access_ledger_receipt_id": "8" * 64,
            "monitor_receipt_id": "9" * 64,
            "inner_publication_manifest_id": manifest.manifest_id,
            "verified_artifact_sha256s": (dummy.sha256,),
            "terminal_status": "extension_required",
            "may_parent_g00d": False,
        },
        "receipt_id",
    )
    decision_path = root / "DECISION_RECEIPT.json"
    decision_path.write_text(decision.model_dump_json() + "\n")
    decision_ref = _artifact(decision_path, root=root, media_type="application/json")
    inventory = root / "artifacts.json"
    inventory.write_text(
        json.dumps(
            {
                "decision": decision_ref.model_dump(mode="json"),
                "execution_bundle": bundle_ref.model_dump(mode="json"),
                "inner_publication_manifest": manifest_ref.model_dump(mode="json"),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    inventory_ref = _artifact(inventory, root=root, media_type="application/json")
    outer_sums = root / "SHA256SUMS"
    outer_sums.write_text(
        "".join(
            f"{ref.sha256}  {name}\n"
            for name, ref in (
                ("DECISION_RECEIPT.json", decision_ref),
                ("EXECUTION_BUNDLE.json", bundle_ref),
                ("INNER_PUBLICATION_MANIFEST.json", manifest_ref),
                ("artifacts.json", inventory_ref),
            )
        )
    )
    committed = root / "COMMITTED"
    committed.write_text("sealed\n")
    seal = _identified(
        G00CFinalSealV1,
        {
            "seal_id": "0" * 64,
            "execution_bundle": bundle_ref,
            "execution_bundle_id": bundle.bundle_id,
            "inner_publication_manifest": manifest_ref,
            "inner_publication_manifest_id": manifest.manifest_id,
            "final_decision": decision_ref,
            "final_decision_id": decision.receipt_id,
            "outer_artifact_inventory": inventory_ref,
            "sha256sums": _artifact(outer_sums, root=root, media_type="text/plain"),
            "committed": _artifact(committed, root=root, media_type="text/plain"),
        },
        "seal_id",
    )
    seal_path = root / "FINAL_SEAL.json"
    seal_path.write_text(seal.model_dump_json() + "\n")
    return (
        bundle,
        decision,
        seal,
        _artifact(seal_path, root=root.parent, media_type="application/json"),
    )


def test_dev37_internally_verified_real_store_and_source_ranking(tmp_path: Path) -> None:
    store, source_root, feature_index = _build_real_g00b(tmp_path)
    authority_root = tmp_path / "authority"
    authority_root.mkdir()
    binding = _binding(authority_root, store, feature_index)
    opened, feature_ids = open_verified_source_plane_v4(
        authority_root,
        binding,
        source_plane_root=tmp_path,
        source_files_root=source_root,
    )
    authority_for_open = G00CD1ExecutionAuthorityFreezeV3.model_construct(source_plane=binding)
    reopened, reopened_feature_ids = open_verified_g00b_v4(
        authority_root,
        authority_for_open,
        source_plane_root=tmp_path,
        source_files_root=source_root,
    )
    assert reopened.manifest == opened.manifest
    assert reopened_feature_ids == feature_ids
    assert isinstance(opened, VirtualCanonicalCountStore)
    assert feature_ids[-1] == "CUSTOM001_PuroR"
    accesses: list[tuple[str, tuple[int, ...]]] = []
    ranking = recompute_feature_ranking_v4(
        opened,
        feature_ids,
        np.asarray([10, 20, 30, 40], dtype=np.int64),
        puro_r_canonical_index=4096,
        batch_size=2,
        access_callback=lambda role, rows: accesses.append(
            (role, tuple(np.asarray(rows, dtype=np.int64)))
        ),
    )
    assert accesses == [
        ("feature_ranking_training_fit", (10, 20)),
        ("feature_ranking_training_fit", (30, 40)),
    ]
    assert tuple(ranking.columns) == RANKING_COLUMNS
    assert len(ranking) == 4096
    assert "CUSTOM001_PuroR" not in set(ranking["feature_id"])
    ranking_path = authority_root / "ranking.parquet"
    ranking.to_parquet(ranking_path, index=False)
    reference = authority_root / "reference.parquet"
    pd.DataFrame({"rank": [1, 2, 3, 4], "row_id": [10, 20, 30, 40]}).to_parquet(
        reference, index=False
    )
    ordered_ids = tuple(ranking["feature_id"].astype(str))
    receipt = _identified(
        G00CFeatureRankingReceiptV4,
        {
            "receipt_id": "0" * 64,
            "execution_authority_id": "a" * 64,
            "source_binding_id": binding.binding_id,
            "fit_rows_hash": _ordered_row_hash(np.asarray([10, 20, 30, 40])),
            "fit_row_count": 4,
            "complete_ranking": _artifact(
                ranking_path, root=authority_root, media_type="application/vnd.apache.parquet"
            ),
            "ordered_feature_hash": hashlib.sha256(
                canonical_json_bytes(list(ordered_ids))
            ).hexdigest(),
            "selected_prefix_hash": hashlib.sha256(
                canonical_json_bytes(list(ordered_ids))
            ).hexdigest(),
            "canonical_feature_index_hash": binding.canonical_feature_index_hash,
            "puro_r_canonical_index": 4096,
        },
        "receipt_id",
    )
    authority = G00CD1ExecutionAuthorityFreezeV3.model_construct(
        authority_id="a" * 64,
        source_plane=binding,
        feature_reference_rows=_artifact(
            reference, root=authority_root, media_type="application/vnd.apache.parquet"
        ),
        feature_reference_rows_hash=_ordered_row_hash(np.asarray([10, 20, 30, 40])),
    )
    observed = verify_g00c_feature_ranking_v4(
        authority_root, authority, opened, feature_ids, receipt
    )
    assert observed.equals(ranking)
    malformed_reference = authority_root / "malformed-reference.parquet"
    pd.DataFrame({"row_id": [10]}).to_parquet(malformed_reference, index=False)
    with pytest.raises(IntegrityError, match="another schema"):
        verify_g00c_feature_ranking_v4(
            authority_root,
            authority.model_copy(
                update={
                    "feature_reference_rows": _artifact(
                        malformed_reference,
                        root=authority_root,
                        media_type="application/vnd.apache.parquet",
                    )
                }
            ),
            opened,
            feature_ids,
            receipt,
        )
    with pytest.raises(IntegrityError, match="receipt differs"):
        verify_g00c_feature_ranking_v4(
            authority_root,
            authority,
            opened,
            feature_ids,
            receipt.model_copy(update={"execution_authority_id": "b" * 64}),
        )


def test_dev37_rejects_source_substitution_and_feature_mislabel(tmp_path: Path) -> None:
    store, source_root, feature_index = _build_real_g00b(tmp_path)
    authority_root = tmp_path / "authority"
    authority_root.mkdir()
    binding = _binding(authority_root, store, feature_index)
    source_path = source_root / "rest.h5ad"
    with h5py.File(source_path, "r+") as handle:
        handle["var/_index"][0] = "wrong-feature"
    with pytest.raises(IntegrityError):
        open_verified_source_plane_v4(
            authority_root,
            binding,
            source_plane_root=tmp_path,
            source_files_root=source_root,
        )


def test_dev37_restart_requires_distinct_fresh_process_outputs(tmp_path: Path) -> None:
    def write(name: str, payload: bytes) -> ArtifactRef:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return _artifact(path, root=tmp_path, media_type="application/octet-stream")

    uninterrupted = write("attempt-a/output.bin", b"same")
    resumed = write("attempt-b/output.bin", b"same")
    checkpoint = write("attempt-b/checkpoint.json", b"checkpoint")
    interrupted_path = tmp_path / "attempt-b/interrupted.json"
    interrupted_path.write_text(
        json.dumps(
            {
                "attempt_id": "attempt-b",
                "final_payload_published": False,
                "status": "interrupted_checkpoint_committed",
            }
        )
    )
    process_a = tmp_path / "attempt-a/process.json"
    process_b = tmp_path / "attempt-b/process.json"
    process_a.write_text(
        json.dumps({"attempt_id": "attempt-a", "exit_code": 0, "pid": 101, "status": "complete"})
    )
    process_b.write_text(
        json.dumps({"attempt_id": "attempt-b", "exit_code": 0, "pid": 202, "status": "complete"})
    )
    receipt = _identified(
        G00CDurableRestartReceiptV4,
        {
            "receipt_id": "0" * 64,
            "component": "sampler",
            "uninterrupted_attempt_id": "attempt-a",
            "resumed_attempt_id": "attempt-b",
            "durable_checkpoint": checkpoint,
            "interrupted_no_final_publication_receipt": _artifact(
                interrupted_path, root=tmp_path, media_type="application/json"
            ),
            "uninterrupted_outputs": (uninterrupted,),
            "resumed_outputs": (resumed,),
            "uninterrupted_process_receipt": _artifact(
                process_a, root=tmp_path, media_type="application/json"
            ),
            "resumed_process_receipt": _artifact(
                process_b, root=tmp_path, media_type="application/json"
            ),
        },
        "receipt_id",
    )
    verify_g00c_restart_v4(tmp_path, receipt)
    resumed_path = tmp_path / resumed.relative_uri
    resumed_path.write_bytes(b"different")
    with pytest.raises(IntegrityError):
        verify_g00c_restart_v4(tmp_path, receipt)


def test_dev37_refit_statistics_are_checkpoint_indexed_and_source_derived(
    tmp_path: Path,
) -> None:
    store, _, feature_index = _build_real_g00b(tmp_path)
    feature_ids = tuple(
        item["feature_id"] for item in json.loads(feature_index.read_text())["features"]
    )
    assert feature_ids[-1] == "CUSTOM001_PuroR"
    schedule = derive_refit_seed_schedule("2" * 64)
    candidates = (256, 4096)
    entries = tuple(
        G00CSamplerPlanEntryV4.model_construct(
            candidate_kind="feature_count",
            candidate_value=candidate,
            refit_draw_id=draw,
            macro_updates=2,
            expected_trace_rows=8192,
        )
        for candidate in candidates
        for draw in range(59)
    )
    plan = G00CSamplerPlanV4.model_construct(entries=entries)
    trace_rows: list[dict[str, object]] = []
    repeated = np.resize(np.asarray([10, 20, 30, 40], dtype=np.int64), 8192)
    for entry_index, _entry in enumerate(entries):
        trace_rows.extend(
            {
                "entry_index": entry_index,
                "draw_index": index,
                "row_id": int(row_id),
                "inverse_probability_weight": 1.0,
            }
            for index, row_id in enumerate(repeated)
        )
    trace = pd.DataFrame(trace_rows)
    training, validation, fit_hashes, validation_hash, thinning_hash = (
        derive_checkpoint_statistics_v4(
            store,
            plan,
            trace,
            np.asarray([10, 20, 30, 40], dtype=np.int64),
            np.asarray([10, 30, 40], dtype=np.int64),
            np.arange(4096, dtype=np.int64),
            schedule,
            candidate_kind="feature_count",
            batch_size=8192,
        )
    )
    assert training.shape == (59, 2, 3, 4096)
    assert validation.shape == (59, 3, 4096)
    assert np.array_equal(training[:, 0], training[:, 1])
    root = tmp_path / "evidence"
    root.mkdir()
    schedule_path = root / "schedule.json"
    schedule_path.write_text(schedule.model_dump_json() + "\n")
    stats_path = root / "statistics.npz"
    np.savez(
        stats_path,
        candidate_values=np.asarray(candidates, dtype=np.int64),
        training_counts=training,
        validation_counts=validation,
    )
    records: list[dict[str, object]] = []
    candidate_index = {value: index for index, value in enumerate(candidates)}
    for draw in range(59):
        seeds = schedule.records[draw]
        for candidate in candidates:
            fit = checkpoint_multinomial_refit_common_support(
                training[draw, candidate_index[candidate]],
                modeled_features=candidate,
                per_feature_pseudocount=0.5,
            )
            probabilities = fit.expanded_probabilities
            counts = validation[draw]
            total = int(counts.sum())
            nll = float(-np.sum(counts * np.log(probabilities)))
            records.append(
                {
                    "candidate_kind": "feature_count",
                    "draw_id": draw,
                    "candidate_value": candidate,
                    "initialization": seeds.initialization,
                    "training_sampler": seeds.training_sampler,
                    "thinning": seeds.thinning,
                    "validation_evaluation": seeds.validation_evaluation,
                    "stochastic_optimizer_or_augmentation": (
                        seeds.stochastic_optimizer_or_augmentation
                    ),
                    "restart_interruption_point": seeds.restart_interruption_point,
                    "fit_row_hash": fit_hashes[candidate_index[candidate]],
                    "validation_row_hash": validation_hash,
                    "model_config_hash": "3" * 64,
                    "final_state_hash": hashlib.sha256(
                        np.asarray(probabilities, dtype="<f8").tobytes()
                    ).hexdigest(),
                    "validation_total_count": total,
                    "validation_nll_sum": nll,
                    "validation_nll_per_count": nll / total,
                    "fit_status": "pass",
                }
            )
    records_frame = pd.DataFrame(records, columns=REFIT_V3_COLUMNS)
    records_path = root / "records.parquet"
    records_frame.to_parquet(records_path, index=False)
    authority = G00CD1ExecutionAuthorityFreezeV3.model_construct(
        authority_id="4" * 64,
        selection_freeze_id="5" * 64,
        seed_schedule=_artifact(schedule_path, root=root, media_type="application/json"),
        seed_schedule_id=schedule.schedule_id,
        source_plane=G00CSourcePlaneBindingV4.model_construct(binding_id="6" * 64),
    )
    provisional = G00CRefitReplayReceiptV4.model_construct(
        candidate_kind="feature_count",
        selected_candidate=256,
        reference_candidate=4096,
        modeled_feature_count=4096,
        candidate_values=candidates,
        preregistered_audit_draw_ids=(0, 58),
    )
    replayed = recompute_refit_rows_v4(records_frame, provisional, training, validation)
    replayed_path = root / "replayed.parquet"
    replayed.to_parquet(replayed_path, index=False)
    receipt = _identified(
        G00CRefitReplayReceiptV4,
        {
            "receipt_id": "0" * 64,
            "execution_authority_id": authority.authority_id,
            "selection_freeze_id": authority.selection_freeze_id,
            "seed_schedule_id": schedule.schedule_id,
            "source_binding_id": authority.source_plane.binding_id,
            "sampler_evidence_id": "7" * 64,
            "source_access_ledger_receipt_id": "8" * 64,
            "candidate_kind": "feature_count",
            "selected_candidate": 256,
            "reference_candidate": 4096,
            "modeled_feature_count": 4096,
            "candidate_values": candidates,
            "refit_records": _artifact(
                records_path, root=root, media_type="application/vnd.apache.parquet"
            ),
            "source_derived_statistics": _artifact(
                stats_path, root=root, media_type="application/x-npz"
            ),
            "replayed_rows": _artifact(
                replayed_path, root=root, media_type="application/vnd.apache.parquet"
            ),
            "fit_row_hashes": fit_hashes,
            "validation_row_hash": validation_hash,
            "training_count_hash": hashlib.sha256(
                np.asarray(training, dtype="<f8").tobytes()
            ).hexdigest(),
            "validation_count_vector_hash": hashlib.sha256(
                np.asarray(validation, dtype="<f8").tobytes()
            ).hexdigest(),
            "thinning_trace_hash": thinning_hash,
            "training_shape": training.shape,
            "validation_shape": validation.shape,
            "preregistered_audit_draw_ids": (0, 58),
        },
        "receipt_id",
    )
    replay_accesses: list[tuple[str, int]] = []
    observed = verify_g00c_refit_replay_v4(
        root,
        authority,
        receipt,
        store,
        plan,
        trace,
        np.asarray([10, 20, 30, 40], dtype=np.int64),
        np.asarray([10, 30, 40], dtype=np.int64),
        np.arange(4096, dtype=np.int64),
        access_callback=lambda role, rows: replay_accesses.append((role, len(rows))),
    )
    assert tuple(observed.columns) == REPLAY_COLUMNS
    assert {role for role, _ in replay_accesses} == {
        "refit_training_fit",
        "refit_training_validation",
    }
    unequal_trace = trace.copy()
    unequal_trace.loc[
        unequal_trace["entry_index"].astype(int) == 1,
        "inverse_probability_weight",
    ] = 2.0
    with pytest.raises(IntegrityError, match="feature statistics differ"):
        derive_checkpoint_statistics_v4(
            store,
            plan,
            unequal_trace,
            np.asarray([10, 20, 30, 40], dtype=np.int64),
            np.asarray([10, 30, 40], dtype=np.int64),
            np.arange(4096, dtype=np.int64),
            schedule,
            candidate_kind="feature_count",
        )
    with pytest.raises(IntegrityError, match="cross-wired"):
        verify_g00c_refit_replay_v4(
            root,
            authority,
            receipt.model_copy(update={"execution_authority_id": "f" * 64}),
            store,
            plan,
            trace,
            np.asarray([10, 20, 30, 40], dtype=np.int64),
            np.asarray([10, 30, 40], dtype=np.int64),
            np.arange(4096, dtype=np.int64),
        )
    with pytest.raises(IntegrityError, match="sufficient statistics"):
        verify_g00c_refit_replay_v4(
            root,
            authority,
            receipt,
            store,
            plan,
            trace,
            np.asarray([10, 20, 30, 40], dtype=np.int64),
            np.asarray([10, 30, 40], dtype=np.int64),
            np.arange(4096, dtype=np.int64),
            additional_plan_trace=(plan, trace),
        )
    wrong_schema_path = root / "wrong-statistics.npz"
    np.savez(wrong_schema_path, wrong=np.asarray([1]))
    wrong_schema_payload = receipt.model_dump(mode="json")
    wrong_schema_payload.update(
        {
            "receipt_id": "0" * 64,
            "source_derived_statistics": _artifact(
                wrong_schema_path, root=root, media_type="application/x-npz"
            ).model_dump(mode="json"),
        }
    )
    wrong_schema_receipt = _identified(G00CRefitReplayReceiptV4, wrong_schema_payload, "receipt_id")
    with pytest.raises(IntegrityError, match="another schema"):
        verify_g00c_refit_replay_v4(
            root,
            authority,
            wrong_schema_receipt,
            store,
            plan,
            trace,
            np.asarray([10, 20, 30, 40], dtype=np.int64),
            np.asarray([10, 30, 40], dtype=np.int64),
            np.arange(4096, dtype=np.int64),
        )
    unreadable_path = root / "unreadable-statistics.npz"
    unreadable_path.write_text("not-npz")
    unreadable_payload = receipt.model_dump(mode="json")
    unreadable_payload.update(
        {
            "receipt_id": "0" * 64,
            "source_derived_statistics": _artifact(
                unreadable_path, root=root, media_type="application/x-npz"
            ).model_dump(mode="json"),
        }
    )
    unreadable_receipt = _identified(G00CRefitReplayReceiptV4, unreadable_payload, "receipt_id")
    with pytest.raises(IntegrityError, match="cannot be read"):
        verify_g00c_refit_replay_v4(
            root,
            authority,
            unreadable_receipt,
            store,
            plan,
            trace,
            np.asarray([10, 20, 30, 40], dtype=np.int64),
            np.asarray([10, 30, 40], dtype=np.int64),
            np.arange(4096, dtype=np.int64),
        )
    altered = training.copy()
    altered[0, 0, 0, 0] += 1
    np.savez(
        stats_path,
        candidate_values=np.asarray(candidates, dtype=np.int64),
        training_counts=altered,
        validation_counts=validation,
    )
    altered_ref = _artifact(stats_path, root=root, media_type="application/x-npz")
    altered_payload = receipt.model_dump(mode="json")
    altered_payload["receipt_id"] = "0" * 64
    altered_payload["source_derived_statistics"] = altered_ref.model_dump(mode="json")
    altered_payload["training_count_hash"] = hashlib.sha256(
        np.asarray(altered, dtype="<f8").tobytes()
    ).hexdigest()
    altered_receipt = _identified(G00CRefitReplayReceiptV4, altered_payload, "receipt_id")
    with pytest.raises(IntegrityError):
        verify_g00c_refit_replay_v4(
            root,
            authority,
            altered_receipt,
            store,
            plan,
            trace,
            np.asarray([10, 20, 30, 40], dtype=np.int64),
            np.asarray([10, 30, 40], dtype=np.int64),
            np.arange(4096, dtype=np.int64),
        )


def test_dev37_sampler_plan_binds_complete_preaccess_grid() -> None:
    schedule = derive_refit_seed_schedule("0" * 64)
    entries = tuple(
        G00CSamplerPlanEntryV4(
            candidate_kind=kind,
            candidate_value=value,
            refit_draw_id=draw,
            macro_updates=2,
            expected_trace_rows=8192,
        )
        for kind, values in (
            ("feature_count", (256, 512, 1024, 2048, 4096)),
            ("training_cells", (50_000, 100_000, 250_000, 500_000, 1_000_000)),
        )
        for value in values
        for draw in range(59)
    )
    plan = _identified(
        G00CSamplerPlanV4,
        {
            "plan_id": "0" * 64,
            "execution_authority_namespace": "1" * 64,
            "seed_schedule_id": schedule.schedule_id,
            "grid_stage": "base",
            "entries": entries,
            "resume_after_macro_update": 1,
            "expected_total_trace_rows": len(entries) * 8192,
        },
        "plan_id",
    )
    assert plan.expected_total_trace_rows == 4_833_280
    with pytest.raises(ValueError, match="plan_id mismatch"):
        G00CSamplerPlanV4.model_validate({**plan.model_dump(mode="json"), "plan_id": "f" * 64})
    with pytest.raises(ValueError):
        G00CSamplerPlanV4.model_validate(
            plan.model_copy(update={"entries": plan.entries[:-1], "plan_id": "0" * 64}).model_dump(
                mode="json"
            )
        )


def test_dev37_inner_publication_is_semantically_bound_and_decision_free(
    tmp_path: Path,
) -> None:
    publication = tmp_path / "inner"
    publication.mkdir()
    source = tmp_path / "source.bin"
    source.write_bytes(b"typed-source")
    source_ref = _artifact(source, root=tmp_path, media_type="application/octet-stream")
    published = publication / "SOURCE.bin"
    published.write_bytes(source.read_bytes())
    mapping = G00CSemanticPublicationArtifactV4(
        role="feature_ranking.complete_ranking",
        filename="SOURCE.bin",
        source_artifact=source_ref,
        published_sha256=source_ref.sha256,
        published_size_bytes=source_ref.size_bytes,
    )
    inventory = _identified(
        G00CInnerPublicationInventoryV4,
        {
            "inventory_id": "0" * 64,
            "terminal_status": "extension_required",
            "artifacts": (mapping,),
        },
        "inventory_id",
    )
    inventory_path = publication / "artifacts.json"
    inventory_path.write_bytes(canonical_json_bytes(inventory.model_dump(mode="json")) + b"\n")
    sums = publication / "SHA256SUMS"
    sums.write_text(f"{source_ref.sha256}  SOURCE.bin\n")
    committed = publication / "COMMITTED"
    committed.write_text("extension_required\n")
    publisher_hash = "9" * 64
    event = publication / "PUBLICATION_EVENT.json"
    event.write_text(
        json.dumps(
            {
                "destination_preexisted": False,
                "directory_fsync_completed": True,
                "manifest_written_last": True,
                "no_clobber": True,
                "publisher_implementation_sha256": publisher_hash,
                "status": "pass",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    bundle = G00CExecutionBundleV5.model_construct(
        execution_authority_id="a" * 64,
        selection_freeze_id="b" * 64,
        feature_selection_result_id="c" * 64,
        sample_size_selection_result_id="d" * 64,
        terminal_status="extension_required",
    )
    manifest = _identified(
        G00CInnerPublicationManifestV4,
        {
            "manifest_id": "0" * 64,
            "execution_authority_id": bundle.execution_authority_id,
            "selection_freeze_id": bundle.selection_freeze_id,
            "feature_selection_result_id": bundle.feature_selection_result_id,
            "sample_size_selection_result_id": bundle.sample_size_selection_result_id,
            "sampler_evidence_id": "e" * 64,
            "terminal_status": "extension_required",
            "artifact_inventory": _artifact(
                inventory_path, root=publication, media_type="application/json"
            ),
            "artifact_inventory_id": inventory.inventory_id,
            "sha256sums": _artifact(sums, root=publication, media_type="text/plain"),
            "committed": _artifact(committed, root=publication, media_type="text/plain"),
            "publication_event_receipt": _artifact(
                event, root=publication, media_type="application/json"
            ),
            "publisher_implementation_sha256": publisher_hash,
        },
        "manifest_id",
    )
    verify_g00c_inner_publication_v4(
        publication,
        bundle,
        manifest,
        expected_artifacts={"feature_ranking.complete_ranking": source_ref},
    )
    with pytest.raises(ValueError, match="inventory_id mismatch"):
        G00CInnerPublicationInventoryV4.model_validate(
            {**inventory.model_dump(mode="json"), "inventory_id": "f" * 64}
        )
    with pytest.raises(ValueError, match="manifest_id mismatch"):
        G00CInnerPublicationManifestV4.model_validate(
            {**manifest.model_dump(mode="json"), "manifest_id": "f" * 64}
        )
    arbitrary = tmp_path / "arbitrary.bin"
    arbitrary.write_bytes(b"arbitrary-payload")
    with pytest.raises(IntegrityError):
        verify_g00c_inner_publication_v4(
            publication,
            bundle,
            manifest,
            expected_artifacts={
                "feature_ranking.complete_ranking": _artifact(
                    arbitrary, root=tmp_path, media_type="application/octet-stream"
                )
            },
        )


def test_dev37_monitor_and_access_are_derived_from_real_store(tmp_path: Path) -> None:
    store, _, feature_index = _build_real_g00b(tmp_path)
    authority_root = tmp_path / "authority"
    authority_root.mkdir()
    binding = _binding(authority_root, store, feature_index)
    role_table = pd.DataFrame(
        {
            "row_id": [10, 20, 30, 40],
            "role": [
                "training_fit",
                "training_validation",
                "heldout_source_query",
                "protected_heldout_stimulated",
            ],
        }
    )
    role_path = authority_root / "roles.parquet"
    role_table.to_parquet(role_path, index=False)
    role_records = tuple(
        FoldRowRoleRecord(
            role=role,  # type: ignore[arg-type]
            rows=1,
            row_ids_hash=_ordered_row_hash(np.asarray([row_id], dtype=np.int64)),
        )
        for row_id, role in zip(role_table["row_id"], role_table["role"], strict=True)
    )
    authority = G00CD1ExecutionAuthorityFreezeV3.model_construct(
        authority_id="a" * 64,
        row_role_freeze=_artifact(
            role_path, root=authority_root, media_type="application/vnd.apache.parquet"
        ),
        row_roles=role_records,
        source_plane=binding,
        monitored_process_tree_ceiling_bytes=68_719_476_736,
    )
    ledger = pd.DataFrame(
        {
            "sequence": [0, 1, 2],
            "monotonic_ns": [1_100_000_000, 1_200_000_000, 1_300_000_000],
            "role": [
                "feature_ranking_training_fit",
                "refit_training_validation",
                "materialization_heldout_source_query",
            ],
            "row_id": [10, 20, 30],
            "source_index": [0, 0, 1],
            "checkpoint": ["Rest", "Rest", "Stim8hr"],
        },
        columns=ACCESS_COLUMNS,
    )
    ledger_path = authority_root / "access.parquet"
    ledger.to_parquet(ledger_path, index=False)
    role_hashes = {
        str(role): _ordered_row_hash(frame["row_id"].to_numpy(dtype=np.int64))
        for role, frame in ledger.groupby(ledger["role"].astype(str), sort=True)
    }
    access = _identified(
        G00CSourceAccessLedgerReceiptV4,
        {
            "receipt_id": "0" * 64,
            "execution_authority_id": authority.authority_id,
            "source_binding_id": binding.binding_id,
            "ledger": _artifact(
                ledger_path, root=authority_root, media_type="application/vnd.apache.parquet"
            ),
            "access_rows": len(ledger),
            "role_row_hashes": role_hashes,
            "protected_row_ids_sha256": role_records[-1].row_ids_hash,
        },
        "receipt_id",
    )
    assert verify_g00c_source_access_v4(authority_root, authority, store, access).equals(ledger)

    monitor_trace = pd.DataFrame(
        {
            "monotonic_ns": [1_000_000_000, 1_200_000_000, 1_400_000_000],
            "root_pid": [101, 101, 101],
            "process_tree_rss_bytes": [1024, 2048, 1536],
            "descendants": [1, 2, 1],
            "readable": [True, True, True],
        },
        columns=MONITOR_COLUMNS,
    )
    monitor_path = authority_root / "monitor.parquet"
    monitor_trace.to_parquet(monitor_path, index=False)
    freeze = G00CMonitorFreezeV1(
        implementation_sha256="b" * 64,
        polling_interval_milliseconds=200,
        maximum_unreadable_samples=0,
        maximum_unreadable_fraction=0,
        maximum_consecutive_unreadable_samples=0,
        maximum_temporal_gap_milliseconds=500,
        maximum_process_tree_rss_bytes=4096,
    )
    monitor = _identified(
        G00CProcessTreeMonitorReceiptV4,
        {
            "receipt_id": "0" * 64,
            "execution_authority_id": authority.authority_id,
            "attempt_id": "synthetic-monitor-a",
            "monitor_pid": 99,
            "monitored_root_pid": 101,
            "trace": _artifact(
                monitor_path, root=authority_root, media_type="application/vnd.apache.parquet"
            ),
            "samples": 3,
            "maximum_process_tree_rss_bytes": 2048,
            "unreadable_samples": 0,
            "maximum_consecutive_unreadable_samples": 0,
            "maximum_temporal_gap_milliseconds": 200.0,
            "descendants_observed": 2,
            "access_start_monotonic_ns": 1_100_000_000,
            "access_end_monotonic_ns": 1_300_000_000,
            "monitor_start_monotonic_ns": 1_000_000_000,
            "monitor_end_monotonic_ns": 1_400_000_000,
        },
        "receipt_id",
    )
    assert verify_g00c_monitor_v4(authority_root, freeze, authority, monitor).equals(monitor_trace)
    with pytest.raises(IntegrityError, match="violates"):
        verify_g00c_monitor_v4(
            authority_root,
            freeze.model_copy(update={"maximum_process_tree_rss_bytes": 1024}),
            authority,
            monitor,
        )


def test_dev37_inner_publisher_and_outer_seal_are_cycle_free(tmp_path: Path) -> None:
    publisher_hash = "1" * 64
    source = tmp_path / "source.bin"
    source.write_bytes(b"semantic-payload")
    source_ref = _artifact(source, root=tmp_path, media_type="application/octet-stream")

    def writer(root: Path) -> G00CInnerPublicationManifestV4:
        root.mkdir(parents=True, exist_ok=True)
        payload = root / "payload.bin"
        payload.write_bytes(source.read_bytes())
        mapping = G00CSemanticPublicationArtifactV4(
            role="source.payload",
            filename=payload.name,
            source_artifact=source_ref,
            published_sha256=source_ref.sha256,
            published_size_bytes=source_ref.size_bytes,
        )
        inventory = _identified(
            G00CInnerPublicationInventoryV4,
            {
                "inventory_id": "0" * 64,
                "terminal_status": "extension_required",
                "artifacts": (mapping,),
            },
            "inventory_id",
        )
        inventory_path = root / "artifacts.json"
        inventory_path.write_bytes(canonical_json_bytes(inventory.model_dump(mode="json")) + b"\n")
        sums = root / "SHA256SUMS"
        sums.write_text(f"{source_ref.sha256}  payload.bin\n")
        event = root / "PUBLICATION_EVENT.json"
        event.write_text(
            json.dumps(
                {
                    "destination_preexisted": False,
                    "directory_fsync_completed": True,
                    "manifest_written_last": True,
                    "no_clobber": True,
                    "publisher_implementation_sha256": publisher_hash,
                    "status": "pass",
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        committed = root / "COMMITTED"
        committed.write_text("extension_required\n")
        return _identified(
            G00CInnerPublicationManifestV4,
            {
                "manifest_id": "0" * 64,
                "execution_authority_id": "a" * 64,
                "selection_freeze_id": "b" * 64,
                "feature_selection_result_id": "c" * 64,
                "sample_size_selection_result_id": "d" * 64,
                "sampler_evidence_id": "e" * 64,
                "terminal_status": "extension_required",
                "artifact_inventory": _artifact(
                    inventory_path, root=root, media_type="application/json"
                ),
                "artifact_inventory_id": inventory.inventory_id,
                "sha256sums": _artifact(sums, root=root, media_type="text/plain"),
                "committed": _artifact(committed, root=root, media_type="text/plain"),
                "publication_event_receipt": _artifact(
                    event, root=root, media_type="application/json"
                ),
                "publisher_implementation_sha256": publisher_hash,
            },
            "manifest_id",
        )

    inner = tmp_path / "inner"
    publish_g00c_inner_v4(
        inner,
        terminal_status="extension_required",
        publisher_implementation_sha256=publisher_hash,
        writer=writer,
    )
    assert (inner / "COMMITTED").read_text() == "extension_required\n"
    with pytest.raises(FileExistsError):
        publish_g00c_inner_v4(
            inner,
            terminal_status="extension_required",
            publisher_implementation_sha256=publisher_hash,
            writer=writer,
        )

    seal_root = tmp_path / "outer"
    seal_root.mkdir()
    manifest = writer(tmp_path / "manifest-staging")
    manifest_path = seal_root / "INNER_PUBLICATION_MANIFEST.json"
    manifest_path.write_text(manifest.model_dump_json() + "\n")
    manifest_ref = _artifact(manifest_path, root=seal_root, media_type="application/json")
    dummy = _artifact(source, root=tmp_path, media_type="application/octet-stream")
    bundle = _identified(
        G00CExecutionBundleV5,
        {
            "bundle_id": "0" * 64,
            "execution_authority": dummy,
            "execution_authority_id": manifest.execution_authority_id,
            "selection_freeze": dummy,
            "selection_freeze_id": manifest.selection_freeze_id,
            "seed_schedule": dummy,
            "seed_schedule_id": "2" * 64,
            "row_roles": dummy,
            "feature_ranking_receipt": dummy,
            "feature_selection_result": dummy,
            "feature_selection_result_id": manifest.feature_selection_result_id,
            "sample_size_selection_result": dummy,
            "sample_size_selection_result_id": manifest.sample_size_selection_result_id,
            "common_support_receipt": dummy,
            "feature_refit_replay_receipt": dummy,
            "sample_refit_replay_receipt": dummy,
            "sampler_evidence": dummy,
            "sampler_restart_receipt": dummy,
            "source_access_ledger_receipt": dummy,
            "monitor_receipt": dummy,
            "base_support_audit_contract": dummy,
            "base_support_audit_receipt": dummy,
            "inner_publication_manifest": manifest_ref,
            "terminal_status": "extension_required",
        },
        "bundle_id",
    )
    bundle_path = seal_root / "EXECUTION_BUNDLE.json"
    bundle_path.write_text(bundle.model_dump_json() + "\n")
    bundle_ref = _artifact(bundle_path, root=seal_root, media_type="application/json")
    decision = _identified(
        G00CDecisionReceiptV5,
        {
            "receipt_id": "0" * 64,
            "execution_bundle_id": bundle.bundle_id,
            "execution_authority_id": bundle.execution_authority_id,
            "selection_freeze_id": bundle.selection_freeze_id,
            "feature_selection_result_id": bundle.feature_selection_result_id,
            "sample_size_selection_result_id": bundle.sample_size_selection_result_id,
            "feature_ranking_receipt_id": "3" * 64,
            "sampler_evidence_id": manifest.sampler_evidence_id,
            "sampler_restart_receipt_id": "4" * 64,
            "feature_refit_replay_receipt_id": "5" * 64,
            "sample_refit_replay_receipt_id": "6" * 64,
            "support_audit_receipt_ids": ("7" * 64,),
            "source_access_ledger_receipt_id": "8" * 64,
            "monitor_receipt_id": "9" * 64,
            "inner_publication_manifest_id": manifest.manifest_id,
            "verified_artifact_sha256s": (dummy.sha256,),
            "terminal_status": "extension_required",
            "may_parent_g00d": False,
        },
        "receipt_id",
    )
    decision_path = seal_root / "DECISION_RECEIPT.json"
    decision_path.write_text(decision.model_dump_json() + "\n")
    decision_ref = _artifact(decision_path, root=seal_root, media_type="application/json")
    inventory_path = seal_root / "artifacts.json"
    inventory_path.write_text(
        json.dumps(
            {
                "decision": decision_ref.model_dump(mode="json"),
                "execution_bundle": bundle_ref.model_dump(mode="json"),
                "inner_publication_manifest": manifest_ref.model_dump(mode="json"),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    inventory_ref = _artifact(inventory_path, root=seal_root, media_type="application/json")
    sums_path = seal_root / "SHA256SUMS"
    sums_path.write_text(
        "".join(
            f"{artifact.sha256}  {name}\n"
            for name, artifact in (
                ("DECISION_RECEIPT.json", decision_ref),
                ("EXECUTION_BUNDLE.json", bundle_ref),
                ("INNER_PUBLICATION_MANIFEST.json", manifest_ref),
                ("artifacts.json", inventory_ref),
            )
        )
    )
    committed_path = seal_root / "COMMITTED"
    committed_path.write_text("sealed\n")
    seal = _identified(
        G00CFinalSealV1,
        {
            "seal_id": "0" * 64,
            "execution_bundle": bundle_ref,
            "execution_bundle_id": bundle.bundle_id,
            "inner_publication_manifest": manifest_ref,
            "inner_publication_manifest_id": manifest.manifest_id,
            "final_decision": decision_ref,
            "final_decision_id": decision.receipt_id,
            "outer_artifact_inventory": inventory_ref,
            "sha256sums": _artifact(sums_path, root=seal_root, media_type="text/plain"),
            "committed": _artifact(committed_path, root=seal_root, media_type="text/plain"),
        },
        "seal_id",
    )
    (seal_root / "FINAL_SEAL.json").write_text(seal.model_dump_json() + "\n")
    assert verify_g00c_final_seal_v1(seal_root, seal) == (bundle, manifest, decision)
    with pytest.raises(ValueError, match="seal_id mismatch"):
        G00CFinalSealV1.model_validate({**seal.model_dump(mode="json"), "seal_id": "f" * 64})
    committed_path.write_text("wrong\n")
    with pytest.raises(IntegrityError):
        verify_g00c_final_seal_v1(seal_root, seal)


def test_dev37_sampler_and_support_replay_use_preaccess_plan(tmp_path: Path) -> None:
    root = tmp_path / "sampler"
    root.mkdir()
    schedule = derive_refit_seed_schedule("0" * 64)
    schedule_path = root / "schedule.json"
    schedule_path.write_text(schedule.model_dump_json() + "\n")
    hierarchy = pd.DataFrame(
        {
            "row_id": [10, 11, 12, 13],
            "source_index": [0, 0, 1, 1],
            "target_code": [0, 1, 0, 1],
            "guide_code": [0, 1, 0, 1],
            "is_control": [False, False, True, True],
        }
    )
    hierarchy_path = root / "hierarchy.parquet"
    hierarchy.to_parquet(hierarchy_path, index=False)
    order = np.asarray([10, 11, 12, 13], dtype=np.int64)
    order_path = root / "order.parquet"
    pd.DataFrame({"rank": [1, 2, 3, 4], "row_id": order}).to_parquet(order_path, index=False)
    entry = G00CSamplerPlanEntryV4(
        candidate_kind="training_cells",
        candidate_value=4,
        refit_draw_id=0,
        macro_updates=2,
        expected_trace_rows=8192,
    )
    plan = G00CSamplerPlanV4.model_construct(
        plan_id="1" * 64,
        execution_authority_namespace="2" * 64,
        seed_schedule_id=schedule.schedule_id,
        grid_stage="base",
        entries=(entry,),
        resume_after_macro_update=1,
        microbatch_cells=512,
        microbatches_per_macro_update=8,
        macrobatch_cells=4096,
        hierarchy="donor_checkpoint_then_target_then_guide_then_row_equal_v1",
        inverse_probability_weights="exact_raw_inverse_draw_probability",
        thinning="ordered_entry_stream_binomial_half_count_v2",
        expected_total_trace_rows=8192,
    )
    plan_path = root / "plan.json"
    plan_path.write_text(plan.model_dump_json() + "\n")
    trace, states = replay_sampler_plan_v3(
        plan,
        hierarchy,
        order,
        schedule,
        resumed=False,  # type: ignore[arg-type]
    )
    resumed_trace, resumed_states = replay_sampler_plan_v3(
        plan,
        hierarchy,
        order,
        schedule,
        resumed=True,  # type: ignore[arg-type]
    )
    trace_paths = (
        root / "attempt-a/draw.parquet",
        root / "attempt-b/draw.parquet",
        root / "attempt-a/state.parquet",
        root / "attempt-b/state.parquet",
    )
    for path in trace_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    trace.to_parquet(trace_paths[0], index=False)
    resumed_trace.to_parquet(trace_paths[1], index=False)
    states.to_parquet(trace_paths[2], index=False)
    resumed_states.to_parquet(trace_paths[3], index=False)
    checkpoint = root / "attempt-b/checkpoint.json"
    checkpoint.write_text("{}\n")
    interrupted = root / "attempt-b/interrupted.json"
    interrupted.write_text(
        json.dumps(
            {
                "attempt_id": "sampler-b",
                "final_payload_published": False,
                "status": "interrupted_checkpoint_committed",
            }
        )
    )
    process_paths = (root / "attempt-a/process.json", root / "attempt-b/process.json")
    process_paths[0].write_text(
        json.dumps({"attempt_id": "sampler-a", "exit_code": 0, "pid": 11, "status": "complete"})
    )
    process_paths[1].write_text(
        json.dumps({"attempt_id": "sampler-b", "exit_code": 0, "pid": 12, "status": "complete"})
    )
    trace_refs = tuple(
        _artifact(path, root=root, media_type="application/vnd.apache.parquet")
        for path in trace_paths
    )
    restart = _identified(
        G00CDurableRestartReceiptV4,
        {
            "receipt_id": "0" * 64,
            "component": "sampler",
            "uninterrupted_attempt_id": "sampler-a",
            "resumed_attempt_id": "sampler-b",
            "durable_checkpoint": _artifact(checkpoint, root=root, media_type="application/json"),
            "interrupted_no_final_publication_receipt": _artifact(
                interrupted, root=root, media_type="application/json"
            ),
            "uninterrupted_outputs": (trace_refs[0], trace_refs[2]),
            "resumed_outputs": (trace_refs[1], trace_refs[3]),
            "uninterrupted_process_receipt": _artifact(
                process_paths[0], root=root, media_type="application/json"
            ),
            "resumed_process_receipt": _artifact(
                process_paths[1], root=root, media_type="application/json"
            ),
        },
        "receipt_id",
    )
    restart_path = root / "restart.json"
    restart_path.write_text(restart.model_dump_json() + "\n")
    sampler_impl = root / "sampler.py"
    support_impl = root / "support.py"
    sampler_impl.write_text("# sampler\n")
    support_impl.write_text("# support\n")
    sampler_ref = _artifact(sampler_impl, root=root, media_type="text/x-python")
    support_ref = _artifact(support_impl, root=root, media_type="text/x-python")
    authority = G00CD1ExecutionAuthorityFreezeV3.model_construct(
        authority_id="3" * 64,
        selection_freeze_id=plan.execution_authority_namespace,
        base_sampler_plan=_artifact(plan_path, root=root, media_type="application/json"),
        base_sampler_plan_id=plan.plan_id,
        base_sampler_expected_trace_rows=plan.expected_total_trace_rows,
        seed_schedule=_artifact(schedule_path, root=root, media_type="application/json"),
        nested_training_row_order=_artifact(
            order_path, root=root, media_type="application/vnd.apache.parquet"
        ),
        sampler_row_hierarchy=_artifact(
            hierarchy_path, root=root, media_type="application/vnd.apache.parquet"
        ),
        implementation=G00CImplementationAuthorityV3.model_construct(
            implementations=(
                G00CImplementationBindingV3(role="sampler", artifact=sampler_ref),
                G00CImplementationBindingV3(role="support_auditor", artifact=support_ref),
            )
        ),
    )
    evidence = _identified(
        G00CSamplerEvidenceV4,
        {
            "evidence_id": "0" * 64,
            "execution_authority_id": authority.authority_id,
            "sampler_plan": authority.base_sampler_plan,
            "sampler_plan_id": plan.plan_id,
            "row_hierarchy": authority.sampler_row_hierarchy,
            "nested_training_row_order": authority.nested_training_row_order,
            "uninterrupted_draw_trace": trace_refs[0],
            "resumed_draw_trace": trace_refs[1],
            "uninterrupted_state_trace": trace_refs[2],
            "resumed_state_trace": trace_refs[3],
            "restart_receipt": _artifact(restart_path, root=root, media_type="application/json"),
            "restart_receipt_id": restart.receipt_id,
            "implementation_sha256": sampler_ref.sha256,
        },
        "evidence_id",
    )
    # The replay above proves the algorithm on a bounded real trace. The authority
    # verifier must nevertheless reject that deliberately incomplete plan rather
    # than accepting a test-sized grid as promotion-bearing evidence.
    with pytest.raises(IntegrityError, match="model artifact is invalid"):
        verify_g00c_sampler_v4(root, authority, evidence, hierarchy)

    contract = G00CSupportAuditContractV2.model_construct(
        contract_id="4" * 64,
        stage="base",
        dimensions=(
            "donor_checkpoint",
            "target",
            "guide",
            "control_vs_targeting",
            "sampler_stratum",
        ),
        feature_candidate_counts=(),
        cell_candidate_counts=(4,),
    )
    support = derive_support_table_v3(
        contract,
        plan,
        hierarchy,
        order,
        trace,  # type: ignore[arg-type]
    )
    assert set(support["candidate_value"].astype(int)) == {4}


def test_dev37_hierarchy_is_independently_rederived_from_real_g00b(tmp_path: Path) -> None:
    store, _, feature_index = _build_real_g00b(tmp_path)
    root = tmp_path / "hierarchy-authority"
    root.mkdir()
    binding = _binding(root, store, feature_index)
    roles_path = root / "roles.parquet"
    pd.DataFrame(
        {
            "row_id": [10, 20, 30, 40],
            "role": ["training_fit"] * 4,
        }
    ).to_parquet(roles_path, index=False)
    derived = derive_hierarchy_from_g00b_v4(store, np.asarray([10, 20, 30, 40], dtype=np.int64))
    hierarchy_path = root / "hierarchy.parquet"
    derived.to_parquet(hierarchy_path, index=False)

    def digest(column: str, dtype: str) -> str:
        return hashlib.sha256(
            np.asarray(derived[column], dtype=dtype).tobytes(order="C")
        ).hexdigest()

    receipt = _identified(
        G00CHierarchyDerivationReceiptV4,
        {
            "receipt_id": "0" * 64,
            "source_binding_id": binding.binding_id,
            "hierarchy": _artifact(
                hierarchy_path, root=root, media_type="application/vnd.apache.parquet"
            ),
            "row_count": len(derived),
            "ordered_row_ids_sha256": digest("row_id", "<i8"),
            "source_index_sha256": digest("source_index", "<i2"),
            "target_code_sha256": digest("target_code", "<i4"),
            "guide_code_sha256": digest("guide_code", "<i4"),
            "is_control_sha256": digest("is_control", "|b1"),
        },
        "receipt_id",
    )
    receipt_path = root / "hierarchy-receipt.json"
    receipt_path.write_text(receipt.model_dump_json() + "\n")
    authority = G00CD1ExecutionAuthorityFreezeV3.model_construct(
        source_plane=binding,
        row_role_freeze=_artifact(
            roles_path, root=root, media_type="application/vnd.apache.parquet"
        ),
        sampler_row_hierarchy=receipt.hierarchy,
        hierarchy_derivation_receipt=_artifact(
            receipt_path, root=root, media_type="application/json"
        ),
    )
    assert verify_g00c_hierarchy_v4(root, authority, store).equals(derived)
    malformed_roles = root / "malformed-roles.parquet"
    pd.DataFrame({"wrong": [10]}).to_parquet(malformed_roles, index=False)
    with pytest.raises(IntegrityError, match="another schema"):
        verify_g00c_hierarchy_v4(
            root,
            authority.model_copy(
                update={
                    "row_role_freeze": _artifact(
                        malformed_roles,
                        root=root,
                        media_type="application/vnd.apache.parquet",
                    )
                }
            ),
            store,
        )
    altered = derived.copy()
    altered.loc[0, "target_code"] += 1
    altered.to_parquet(hierarchy_path, index=False)
    changed = _artifact(hierarchy_path, root=root, media_type="application/vnd.apache.parquet")
    authority = authority.model_copy(update={"sampler_row_hierarchy": changed})
    with pytest.raises(IntegrityError, match="hierarchy differs"):
        verify_g00c_hierarchy_v4(root, authority, store)
    derived.to_parquet(hierarchy_path, index=False)
    restored_hierarchy = _artifact(
        hierarchy_path, root=root, media_type="application/vnd.apache.parquet"
    )
    wrong_receipt_payload = receipt.model_dump(mode="json")
    wrong_receipt_payload.update({"receipt_id": "0" * 64, "source_binding_id": "f" * 64})
    wrong_receipt = _identified(
        G00CHierarchyDerivationReceiptV4, wrong_receipt_payload, "receipt_id"
    )
    wrong_receipt_path = root / "wrong-hierarchy-receipt.json"
    wrong_receipt_path.write_text(wrong_receipt.model_dump_json() + "\n")
    with pytest.raises(IntegrityError, match="receipt differs"):
        verify_g00c_hierarchy_v4(
            root,
            authority.model_copy(
                update={
                    "sampler_row_hierarchy": restored_hierarchy,
                    "hierarchy_derivation_receipt": _artifact(
                        wrong_receipt_path, root=root, media_type="application/json"
                    ),
                }
            ),
            store,
        )


def test_dev37_decision_is_derived_only_from_verified_execution() -> None:
    verified = VerifiedG00CExecutionV5(
        terminal_status="pass",
        execution_bundle_id="0" * 64,
        execution_authority_id="1" * 64,
        selection_freeze_id="2" * 64,
        feature_selection_result_id="3" * 64,
        sample_size_selection_result_id="4" * 64,
        feature_ranking_receipt_id="5" * 64,
        sampler_evidence_id="6" * 64,
        sampler_restart_receipt_id="7" * 64,
        feature_refit_replay_receipt_id="8" * 64,
        sample_refit_replay_receipt_id="9" * 64,
        support_audit_receipt_ids=("a" * 64,),
        source_access_ledger_receipt_id="b" * 64,
        monitor_receipt_id="c" * 64,
        materialization_receipt_id="d" * 64,
        inner_publication_manifest_id="e" * 64,
        verified_artifact_sha256s=("f" * 64,),
    )
    decision = build_g00c_decision_receipt_v5(verified)
    assert decision.may_parent_g00d
    assert decision.materialization_receipt_id == verified.materialization_receipt_id
    stopped = build_g00c_decision_receipt_v5(
        VerifiedG00CExecutionV5(
            **{
                **verified.__dict__,
                "terminal_status": "extension_required",
                "materialization_receipt_id": None,
            }
        )
    )
    assert not stopped.may_parent_g00d


def test_dev37_materialization_replays_real_source_bytes_and_fresh_writer(
    tmp_path: Path,
) -> None:
    store, _, feature_index = _build_real_g00b(tmp_path)
    root = tmp_path / "materialization"
    root.mkdir()
    binding = _binding(root, store, feature_index)

    roles = pd.DataFrame(
        {
            "row_id": [10, 20, 30, 40],
            "role": [
                "training_fit",
                "training_validation",
                "heldout_source_query",
                "protected_heldout_stimulated",
            ],
        }
    )
    roles_path = root / "roles.parquet"
    roles.to_parquet(roles_path, index=False)
    selected_rows = np.asarray([10], dtype=np.int64)
    compact_order, _ = _expected_physical_order(store, np.asarray([10, 20, 30], dtype=np.int64))
    selected_features = pd.DataFrame(
        {
            "rank": [1, 2],
            "canonical_index": [0, 1],
            "feature_id": ["g0000", "g0001"],
        }
    )
    features_path = root / "selected-features.parquet"
    rows_path = root / "selected-rows.parquet"
    physical_path = root / "physical-runs.parquet"
    selected_features.to_parquet(features_path, index=False)
    pd.DataFrame({"row_id": selected_rows}).to_parquet(rows_path, index=False)
    _physical_runs(store, compact_order).to_parquet(physical_path, index=False)

    attempt_a = root / "attempt-a"
    attempt_b = root / "attempt-b"
    attempt_a.mkdir()
    attempt_b.mkdir()
    primary_a = attempt_a / "counts.h5"
    primary_b = attempt_b / "counts.h5"
    puro_a = attempt_a / "puro-r.h5"
    puro_b = attempt_b / "puro-r.h5"
    for path in (primary_a, primary_b):
        _write_compact_h5(
            path,
            store=store,
            row_ids=compact_order,
            feature_indices=np.asarray([0, 1], dtype=np.int64),
            feature_ids=["g0000", "g0001"],
        )
    for path in (puro_a, puro_b):
        _write_compact_h5(
            path,
            store=store,
            row_ids=compact_order,
            feature_indices=np.asarray([4096], dtype=np.int64),
            feature_ids=["CUSTOM001_PuroR"],
        )
    primary_a_ref = _artifact(primary_a, root=root, media_type="application/x-hdf5")
    primary_b_ref = _artifact(primary_b, root=root, media_type="application/x-hdf5")
    puro_a_ref = _artifact(puro_a, root=root, media_type="application/x-hdf5")
    puro_b_ref = _artifact(puro_b, root=root, media_type="application/x-hdf5")
    assert primary_a_ref.sha256 == primary_b_ref.sha256
    assert puro_a_ref.sha256 == puro_b_ref.sha256

    protected_hash = hashlib.sha256(np.asarray([40], dtype="<i8").tobytes()).hexdigest()
    access_path = root / "protected-access.json"
    access_path.write_text(
        json.dumps(
            {
                "compact_rows_read": 3,
                "protected_expression_reads": 0,
                "protected_row_ids_sha256": protected_hash,
                "schema_version": 1,
                "status": "pass",
            },
            sort_keys=True,
        )
        + "\n"
    )
    reload_path = root / "reload.json"
    reload_path.write_text(
        json.dumps(
            {
                "compact_payload_sha256": primary_a_ref.sha256,
                "primary_features": 2,
                "puro_r_sidecar_sha256": puro_a_ref.sha256,
                "rows": 3,
                "schema_version": 1,
                "sidecar_features": 1,
                "status": "pass",
            },
            sort_keys=True,
        )
        + "\n"
    )
    verifier_path = root / "materialization-verifier.py"
    verifier_path.write_text("# source-backed verifier\n")
    verifier_ref = _artifact(verifier_path, root=root, media_type="text/x-python")
    authority = G00CD1ExecutionAuthorityFreezeV3.model_construct(
        authority_id="a" * 64,
        source_plane=binding,
        row_role_freeze=_artifact(
            roles_path, root=root, media_type="application/vnd.apache.parquet"
        ),
        implementation=G00CImplementationAuthorityV3.model_construct(
            implementations=(
                G00CImplementationBindingV3(role="materialization_verifier", artifact=verifier_ref),
            )
        ),
    )
    base = _identified(
        G00CMaterializationReceiptV3,
        {
            "receipt_id": "0" * 64,
            "execution_authority_id": authority.authority_id,
            "compact_payload": primary_a_ref,
            "puro_r_sidecar": puro_a_ref,
            "selected_features": _artifact(
                features_path, root=root, media_type="application/vnd.apache.parquet"
            ),
            "selected_rows": _artifact(
                rows_path, root=root, media_type="application/vnd.apache.parquet"
            ),
            "physical_runs": _artifact(
                physical_path, root=root, media_type="application/vnd.apache.parquet"
            ),
            "selected_feature_count": 2,
            "compact_row_count": 3,
            "puro_r_canonical_index": 4096,
            "compact_verification_block_rows": 2,
            "compact_data_dtype": "int32",
            "compact_index_dtype": "int32",
            "maximum_observed_count": 60,
            "selected_row_set_sha256": hashlib.sha256(
                np.asarray(selected_rows, dtype="<i8").tobytes()
            ).hexdigest(),
            "compact_ordered_row_ids_sha256": hashlib.sha256(
                np.asarray(compact_order, dtype="<i8").tobytes()
            ).hexdigest(),
            "protected_row_ids_sha256": protected_hash,
            "uninterrupted_compact_payload": primary_a_ref,
            "resumed_compact_payload": primary_b_ref,
            "uninterrupted_puro_r_sidecar": puro_a_ref,
            "resumed_puro_r_sidecar": puro_b_ref,
            "protected_access_receipt": _artifact(
                access_path, root=root, media_type="application/json"
            ),
            "reload_receipt": _artifact(reload_path, root=root, media_type="application/json"),
            "verifier_implementation_sha256": verifier_ref.sha256,
        },
        "receipt_id",
    )
    base_path = root / "base-materialization.json"
    base_path.write_text(base.model_dump_json() + "\n")
    checkpoint = attempt_b / "checkpoint.json"
    checkpoint.write_text('{"status":"committed"}\n')
    interrupted = attempt_b / "interrupted.json"
    interrupted.write_text(
        json.dumps(
            {
                "attempt_id": "writer-b",
                "final_payload_published": False,
                "status": "interrupted_checkpoint_committed",
            }
        )
        + "\n"
    )
    process_a = attempt_a / "process.json"
    process_b = attempt_b / "process.json"
    process_a.write_text(
        json.dumps({"attempt_id": "writer-a", "exit_code": 0, "pid": 101, "status": "complete"})
        + "\n"
    )
    process_b.write_text(
        json.dumps({"attempt_id": "writer-b", "exit_code": 0, "pid": 202, "status": "complete"})
        + "\n"
    )
    restart = _identified(
        G00CDurableRestartReceiptV4,
        {
            "receipt_id": "0" * 64,
            "component": "writer",
            "uninterrupted_attempt_id": "writer-a",
            "resumed_attempt_id": "writer-b",
            "durable_checkpoint": _artifact(checkpoint, root=root, media_type="application/json"),
            "interrupted_no_final_publication_receipt": _artifact(
                interrupted, root=root, media_type="application/json"
            ),
            "uninterrupted_outputs": (primary_a_ref, puro_a_ref),
            "resumed_outputs": (primary_b_ref, puro_b_ref),
            "uninterrupted_process_receipt": _artifact(
                process_a, root=root, media_type="application/json"
            ),
            "resumed_process_receipt": _artifact(
                process_b, root=root, media_type="application/json"
            ),
        },
        "receipt_id",
    )
    restart_path = root / "writer-restart.json"
    restart_path.write_text(restart.model_dump_json() + "\n")
    wrapper = _identified(
        G00CMaterializationReceiptV4,
        {
            "receipt_id": "0" * 64,
            "execution_authority_id": authority.authority_id,
            "source_binding_id": binding.binding_id,
            "base_materialization_receipt": _artifact(
                base_path, root=root, media_type="application/json"
            ),
            "base_materialization_receipt_id": base.receipt_id,
            "source_access_ledger_receipt": base.protected_access_receipt,
            "source_access_ledger_receipt_id": "b" * 64,
            "monitor_receipt": base.reload_receipt,
            "monitor_receipt_id": "c" * 64,
            "writer_restart_receipt": _artifact(
                restart_path, root=root, media_type="application/json"
            ),
            "writer_restart_receipt_id": restart.receipt_id,
        },
        "receipt_id",
    )
    observed = verify_g00c_materialization_v4(
        root,
        store,
        authority,
        wrapper,
        expected_selected_feature_ids=("g0000", "g0001"),
        expected_selected_rows=selected_rows,
    )
    assert observed == base
    with pytest.raises(IntegrityError, match="cross-wired"):
        verify_g00c_materialization_v4(
            root,
            store,
            authority,
            wrapper.model_copy(update={"execution_authority_id": "d" * 64}),
            expected_selected_feature_ids=("g0000", "g0001"),
            expected_selected_rows=selected_rows,
        )
    with pytest.raises(IntegrityError, match="selection differs"):
        verify_g00c_materialization_v4(
            root,
            store,
            authority,
            wrapper,
            expected_selected_feature_ids=("g0001", "g0000"),
            expected_selected_rows=selected_rows,
        )
    malformed_roles_path = root / "malformed-roles.parquet"
    pd.DataFrame({"wrong": [10]}).to_parquet(malformed_roles_path, index=False)
    malformed_authority = authority.model_copy(
        update={
            "row_role_freeze": _artifact(
                malformed_roles_path,
                root=root,
                media_type="application/vnd.apache.parquet",
            )
        }
    )
    with pytest.raises(IntegrityError, match="malformed row-role"):
        verify_g00c_materialization_v4(
            root,
            store,
            malformed_authority,
            wrapper,
            expected_selected_feature_ids=("g0000", "g0001"),
            expected_selected_rows=selected_rows,
        )
    changed_roles_path = root / "changed-roles.parquet"
    pd.DataFrame(
        {
            "row_id": [10, 20, 40, 30],
            "role": [
                "training_fit",
                "training_validation",
                "heldout_source_query",
                "protected_heldout_stimulated",
            ],
        }
    ).to_parquet(changed_roles_path, index=False)
    changed_authority = authority.model_copy(
        update={
            "row_role_freeze": _artifact(
                changed_roles_path,
                root=root,
                media_type="application/vnd.apache.parquet",
            )
        }
    )
    with pytest.raises(IntegrityError, match="row hashes differ"):
        verify_g00c_materialization_v4(
            root,
            store,
            changed_authority,
            wrapper,
            expected_selected_feature_ids=("g0000", "g0001"),
            expected_selected_rows=selected_rows,
        )
    missing_implementation_authority = authority.model_copy(
        update={"implementation": G00CImplementationAuthorityV3.model_construct(implementations=())}
    )
    with pytest.raises(IntegrityError, match="no unique materialization_verifier"):
        verify_g00c_materialization_v4(
            root,
            store,
            missing_implementation_authority,
            wrapper,
            expected_selected_feature_ids=("g0000", "g0001"),
            expected_selected_rows=selected_rows,
        )

    def wrapped_base(
        name: str, changed_base: G00CMaterializationReceiptV3
    ) -> G00CMaterializationReceiptV4:
        path = root / f"base-{name}.json"
        path.write_text(changed_base.model_dump_json() + "\n")
        payload = wrapper.model_dump(mode="json")
        payload.update(
            {
                "receipt_id": "0" * 64,
                "base_materialization_receipt": _artifact(
                    path, root=root, media_type="application/json"
                ).model_dump(mode="json"),
                "base_materialization_receipt_id": changed_base.receipt_id,
            }
        )
        return _identified(G00CMaterializationReceiptV4, payload, "receipt_id")

    wrong_physical_path = root / "wrong-physical.parquet"
    pd.DataFrame({"wrong": [1]}).to_parquet(wrong_physical_path, index=False)
    base_payload = base.model_dump(mode="json")
    base_payload.update(
        {
            "receipt_id": "0" * 64,
            "physical_runs": _artifact(
                wrong_physical_path,
                root=root,
                media_type="application/vnd.apache.parquet",
            ).model_dump(mode="json"),
        }
    )
    wrong_physical = _identified(G00CMaterializationReceiptV3, base_payload, "receipt_id")
    with pytest.raises(IntegrityError, match="physical-run evidence"):
        verify_g00c_materialization_v4(
            root,
            store,
            authority,
            wrapped_base("physical", wrong_physical),
            expected_selected_feature_ids=("g0000", "g0001"),
            expected_selected_rows=selected_rows,
        )

    separated_features_path = root / "separated-features.parquet"
    pd.DataFrame(
        {
            "rank": [1, 2],
            "canonical_index": [4096, 1],
            "feature_id": ["CUSTOM001_PuroR", "g0001"],
        }
    ).to_parquet(separated_features_path, index=False)
    base_payload = base.model_dump(mode="json")
    base_payload.update(
        {
            "receipt_id": "0" * 64,
            "selected_features": _artifact(
                separated_features_path,
                root=root,
                media_type="application/vnd.apache.parquet",
            ).model_dump(mode="json"),
        }
    )
    bad_separation = _identified(G00CMaterializationReceiptV3, base_payload, "receipt_id")
    with pytest.raises(IntegrityError, match="feature separation"):
        verify_g00c_materialization_v4(
            root,
            store,
            authority,
            wrapped_base("separation", bad_separation),
            expected_selected_feature_ids=("CUSTOM001_PuroR", "g0001"),
            expected_selected_rows=selected_rows,
        )

    base_payload = base.model_dump(mode="json")
    base_payload.update(
        {
            "receipt_id": "0" * 64,
            "compact_payload": primary_b_ref.model_dump(mode="json"),
        }
    )
    wrong_output_binding = _identified(G00CMaterializationReceiptV3, base_payload, "receipt_id")
    with pytest.raises(IntegrityError, match="does not bind"):
        verify_g00c_materialization_v4(
            root,
            store,
            authority,
            wrapped_base("binding", wrong_output_binding),
            expected_selected_feature_ids=("g0000", "g0001"),
            expected_selected_rows=selected_rows,
        )

    wrong_reload_path = root / "wrong-reload.json"
    wrong_reload_path.write_text('{"status":"wrong"}\n')
    base_payload = base.model_dump(mode="json")
    base_payload.update(
        {
            "receipt_id": "0" * 64,
            "reload_receipt": _artifact(
                wrong_reload_path, root=root, media_type="application/json"
            ).model_dump(mode="json"),
        }
    )
    wrong_reload = _identified(G00CMaterializationReceiptV3, base_payload, "receipt_id")
    with pytest.raises(IntegrityError, match="reload receipt"):
        verify_g00c_materialization_v4(
            root,
            store,
            authority,
            wrapped_base("reload", wrong_reload),
            expected_selected_feature_ids=("g0000", "g0001"),
            expected_selected_rows=selected_rows,
        )
    with h5py.File(primary_b, "r+") as handle:
        handle["data"][0] += 1
    with pytest.raises(IntegrityError):
        verify_g00c_materialization_v4(
            root,
            store,
            authority,
            wrapper,
            expected_selected_feature_ids=("g0000", "g0001"),
            expected_selected_rows=selected_rows,
        )


def test_dev37_extension_parents_one_sealed_v5_stop_and_exact_two_million_plan(
    tmp_path: Path,
) -> None:
    seal_root = tmp_path / "base-seal"
    bundle, decision, seal, seal_ref = _build_extension_stop(seal_root)
    bundle_ref = _artifact(
        seal_root / "EXECUTION_BUNDLE.json", root=tmp_path, media_type="application/json"
    )
    decision_ref = _artifact(
        seal_root / "DECISION_RECEIPT.json", root=tmp_path, media_type="application/json"
    )
    seed_path = tmp_path / "seed-schedule.json"
    seed_path.write_text("{}\n")
    seed_ref = _artifact(seed_path, root=tmp_path, media_type="application/json")
    feature_path = tmp_path / "selected-features.parquet"
    pd.DataFrame(
        {
            "rank": np.arange(1, 257),
            "canonical_index": np.arange(256),
            "feature_id": [f"g{index:04d}" for index in range(256)],
        }
    ).to_parquet(feature_path, index=False)
    feature_ref = _artifact(
        feature_path, root=tmp_path, media_type="application/vnd.apache.parquet"
    )
    feature = G00CFeatureSelectionResultV3.model_construct(
        result_id=bundle.feature_selection_result_id,
        selected_feature_count=256,
        selected_feature_order_sha256="1" * 64,
        ordered_features=feature_ref,
    )
    authority = G00CD1ExecutionAuthorityFreezeV3.model_construct(
        authority_id=bundle.execution_authority_id,
        selection_freeze_id=bundle.selection_freeze_id,
        seed_schedule_id=bundle.seed_schedule_id,
        seed_schedule=seed_ref,
        source_plane=G00CSourcePlaneBindingV4.model_construct(binding_id="2" * 64),
    )
    entries = tuple(
        G00CSamplerPlanEntryV4(
            candidate_kind="training_cells",
            candidate_value=2_000_000,
            refit_draw_id=draw,
            macro_updates=2,
            expected_trace_rows=8192,
        )
        for draw in range(59)
    )
    plan = _identified(
        G00CSamplerPlanV4,
        {
            "plan_id": "0" * 64,
            "execution_authority_namespace": authority.selection_freeze_id,
            "seed_schedule_id": authority.seed_schedule_id,
            "grid_stage": "extension",
            "entries": entries,
            "resume_after_macro_update": 1,
            "expected_total_trace_rows": 59 * 8192,
        },
        "plan_id",
    )
    plan_path = tmp_path / "extension-plan.json"
    plan_path.write_text(plan.model_dump_json() + "\n")
    plan_ref = _artifact(plan_path, root=tmp_path, media_type="application/json")
    support = _identified(
        G00CSupportAuditContractV2,
        {
            "contract_id": "0" * 64,
            "stage": "extension",
            "feature_candidate_counts": (),
            "cell_candidate_counts": (2_000_000,),
        },
        "contract_id",
    )
    support_path = tmp_path / "extension-support.json"
    support_path.write_text(support.model_dump_json() + "\n")
    dummy_support = tmp_path / "base-support.json"
    dummy_support.write_text("{}\n")
    extension = _identified(
        G00CSampleSizeExtensionFreezeV2,
        {
            "extension_freeze_id": "0" * 64,
            "base_execution_bundle": bundle_ref,
            "base_execution_bundle_id": bundle.bundle_id,
            "base_extension_required_receipt": decision_ref,
            "base_extension_required_receipt_id": decision.receipt_id,
            "base_final_seal": seal_ref,
            "base_final_seal_id": seal.seal_id,
            "execution_authority_id": authority.authority_id,
            "source_binding_id": authority.source_plane.binding_id,
            "feature_selection_result_id": feature.result_id,
            "selected_feature_count": feature.selected_feature_count,
            "selected_feature_order_sha256": feature.selected_feature_order_sha256,
            "selected_feature_surface": feature.ordered_features,
            "seed_schedule_id": authority.seed_schedule_id,
            "seed_schedule_artifact": authority.seed_schedule,
            "base_support_audit_receipt": _artifact(
                dummy_support, root=tmp_path, media_type="application/json"
            ),
            "extension_sampler_plan": plan_ref,
            "extension_sampler_plan_id": plan.plan_id,
            "extension_support_audit_contract": _artifact(
                support_path, root=tmp_path, media_type="application/json"
            ),
            "fresh_attempt_id": "extension-attempt-a",
            "publication_root_uri": "publication/extension-a",
        },
        "extension_freeze_id",
    )
    assert (
        verify_g00c_extension_freeze_v2(
            tmp_path, extension, authority=authority, feature_result=feature
        )
        == plan
    )
    with pytest.raises(ValueError, match="extension_freeze_id mismatch"):
        G00CSampleSizeExtensionFreezeV2.model_validate(
            {**extension.model_dump(mode="json"), "extension_freeze_id": "f" * 64}
        )
    with pytest.raises(ValueError, match="exactly one"):
        extension.model_copy(update={"added_candidate_cells": (1,)}).validate_extension()
    with pytest.raises(IntegrityError, match="changes authority"):
        verify_g00c_extension_freeze_v2(
            tmp_path,
            extension.model_copy(update={"source_binding_id": "f" * 64}),
            authority=authority,
            feature_result=feature,
        )


def test_dev37_v5_helpers_bind_typed_artifacts_refits_and_materialization_access(
    tmp_path: Path,
) -> None:
    class ArtifactContainer(BaseModel):
        direct: ArtifactRef

    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"typed")
    artifact = _artifact(payload, root=tmp_path, media_type="application/octet-stream")
    container = ArtifactContainer(direct=artifact)
    nested = {"direct": artifact, "models": (container,), "ignored": 3}
    refs = _artifact_refs(nested)
    assert refs == [artifact, artifact]

    roles_path = tmp_path / "roles.parquet"
    pd.DataFrame(
        {
            "row_id": [10, 20, 30, 40],
            "role": [
                "training_fit",
                "training_validation",
                "heldout_source_query",
                "protected_heldout_stimulated",
            ],
        }
    ).to_parquet(roles_path, index=False)
    decision_impl = tmp_path / "decision.py"
    decision_impl.write_text("# verifier\n")
    decision_ref = _artifact(decision_impl, root=tmp_path, media_type="text/x-python")
    authority = G00CD1ExecutionAuthorityFreezeV3.model_construct(
        authority_id="2" * 64,
        row_role_freeze=_artifact(
            roles_path, root=tmp_path, media_type="application/vnd.apache.parquet"
        ),
        implementation=G00CImplementationAuthorityV3.model_construct(
            implementations=(
                G00CImplementationBindingV3(role="decision_verifier", artifact=decision_ref),
            )
        ),
    )
    assert _v5_implementation_hash(authority, "decision_verifier") == decision_ref.sha256
    with pytest.raises(IntegrityError, match="no unique sampler"):
        _v5_implementation_hash(authority, "sampler")
    assert _expected_access_rows(
        [("feature_ranking_training_fit", 10)],
        root=tmp_path,
        authority=authority,
        selected_rows=np.asarray([10], dtype=np.int64),
    ) == [
        ("feature_ranking_training_fit", 10),
        ("materialization_training_fit", 10),
        ("materialization_training_validation", 20),
        ("materialization_heldout_source_query", 30),
    ]
    assert _expected_access_rows([], root=tmp_path, authority=authority, selected_rows=None) == []

    feature = G00CFeatureSelectionResultV3.model_construct(
        refit_records=artifact,
        selected_feature_count=256,
    )
    replay = G00CRefitReplayReceiptV4.model_construct(
        candidate_kind="feature_count",
        refit_records=artifact,
        selected_candidate=256,
        reference_candidate=4096,
        modeled_feature_count=4096,
        sampler_evidence_id="3" * 64,
        source_access_ledger_receipt_id="4" * 64,
    )
    _verify_refit_binding(
        replay,
        result=feature,
        kind="feature_count",
        selected=256,
        reference=4096,
        modeled_features=4096,
        sampler_evidence_id="3" * 64,
        access_receipt_id="4" * 64,
    )
    with pytest.raises(IntegrityError, match="another selection"):
        _verify_refit_binding(
            replay,
            result=feature,
            kind="training_cells",
            selected=256,
            reference=4096,
            modeled_features=4096,
            sampler_evidence_id="3" * 64,
            access_receipt_id="4" * 64,
        )
    models = _semantic_models(
        authority=authority,
        freeze=container,
        schedule=container,
        ranking=container,
        feature=feature,
        sample=container,
        common=container,
        sampler=container,
        sampler_restart=container,
        access=container,
        monitor=container,
        support=(container, container),
        feature_replay=replay,
        sample_replay=replay,
        extension=container,
        extension_sampler=None,
        extension_restart=container,
        materialization=None,
    )
    assert "support_0" in models and "extension_freeze" in models
    assert "extension_sampler" not in models and "materialization" not in models


def test_dev37_contracts_fail_closed_on_aliases_gaps_and_terminal_cross_wiring(
    tmp_path: Path,
) -> None:
    store, _, feature_index = _build_real_g00b(tmp_path)
    authority_root = tmp_path / "contract-authority"
    authority_root.mkdir()
    binding = _binding(authority_root, store, feature_index)

    def invalid_binding(**changes: object) -> None:
        payload = binding.model_dump(mode="json")
        payload.update(changes)
        payload["binding_id"] = "0" * 64
        with pytest.raises(ValueError):
            G00CSourcePlaneBindingV4.model_validate(payload)

    invalid_binding(accepted_g00b_manifest_sha256="f" * 64)
    invalid_binding(puro_r_canonical_index=binding.feature_count)
    invalid_binding(source_files=(binding.source_files[0], binding.source_files[0]))
    with pytest.raises(ValueError, match="binding_id mismatch"):
        G00CSourcePlaneBindingV4.model_validate(
            {**binding.model_dump(mode="json"), "binding_id": "f" * 64}
        )

    with pytest.raises(ValueError, match="trace size"):
        G00CSamplerPlanEntryV4(
            candidate_kind="training_cells",
            candidate_value=2_000_000,
            refit_draw_id=0,
            macro_updates=2,
            expected_trace_rows=4096,
        )
    entries = tuple(
        G00CSamplerPlanEntryV4(
            candidate_kind="training_cells",
            candidate_value=2_000_000,
            refit_draw_id=draw,
            macro_updates=2,
            expected_trace_rows=8192,
        )
        for draw in range(59)
    )
    plan = _identified(
        G00CSamplerPlanV4,
        {
            "plan_id": "0" * 64,
            "execution_authority_namespace": "1" * 64,
            "seed_schedule_id": "2" * 64,
            "grid_stage": "extension",
            "entries": entries,
            "resume_after_macro_update": 1,
            "expected_total_trace_rows": 59 * 8192,
        },
        "plan_id",
    )

    def invalid_plan(**changes: object) -> None:
        payload = plan.model_dump(mode="json")
        payload.update(changes)
        payload["plan_id"] = "0" * 64
        with pytest.raises(ValueError):
            G00CSamplerPlanV4.model_validate(payload)

    invalid_plan(entries=())
    invalid_plan(entries=(*entries[:-1], entries[0]))
    invalid_plan(resume_after_macro_update=2)
    invalid_plan(expected_total_trace_rows=1)

    payload_path = tmp_path / "contract-payload.bin"
    payload_path.write_bytes(b"same")
    other_path = tmp_path / "contract-other.bin"
    other_path.write_bytes(b"same")
    changed_path = tmp_path / "contract-changed.bin"
    changed_path.write_bytes(b"different")
    artifact = _artifact(payload_path, root=tmp_path, media_type="application/octet-stream")
    other = _artifact(other_path, root=tmp_path, media_type="application/octet-stream")
    changed = _artifact(changed_path, root=tmp_path, media_type="application/octet-stream")
    roles = (
        "feature_ranking",
        "source_authority",
        "hierarchy_derivation",
        "refit",
        "sampler",
        "support_auditor",
        "monitor",
        "source_access_auditor",
        "materialization_verifier",
        "refit_replay_verifier",
        "publication_verifier",
        "restart_verifier",
        "extension_verifier",
        "execution_verifier",
        "decision_verifier",
    )
    implementation = G00CImplementationAuthorityV3(
        dev37_code_commit="3" * 40,
        wheel=artifact,
        normalized_sdist=artifact,
        implementation_tree_sha256="4" * 64,
        environment_lock=artifact,
        environment_kind="exact_local_lock",
        execution_environment_digest=f"sha256:{'5' * 64}",
        implementations=tuple(
            G00CImplementationBindingV3(role=role, artifact=artifact)  # type: ignore[arg-type]
            for role in roles
        ),
    )
    with pytest.raises(ValueError, match="ordered set"):
        G00CImplementationAuthorityV3.model_validate(
            implementation.model_copy(
                update={"implementations": tuple(reversed(implementation.implementations))}
            ).model_dump(mode="json")
        )

    monitor = _identified(
        G00CProcessTreeMonitorReceiptV4,
        {
            "receipt_id": "0" * 64,
            "execution_authority_id": "6" * 64,
            "attempt_id": "monitor-a",
            "monitor_pid": 10,
            "monitored_root_pid": 20,
            "trace": artifact,
            "samples": 2,
            "maximum_process_tree_rss_bytes": 100,
            "unreadable_samples": 0,
            "maximum_consecutive_unreadable_samples": 0,
            "maximum_temporal_gap_milliseconds": 1,
            "descendants_observed": 1,
            "access_start_monotonic_ns": 2,
            "access_end_monotonic_ns": 3,
            "monitor_start_monotonic_ns": 1,
            "monitor_end_monotonic_ns": 4,
        },
        "receipt_id",
    )
    for changes in (
        {"monitor_pid": 20},
        {"monitor_start_monotonic_ns": 3},
    ):
        payload = monitor.model_dump(mode="json")
        payload.update(changes)
        payload["receipt_id"] = "0" * 64
        with pytest.raises(ValueError):
            G00CProcessTreeMonitorReceiptV4.model_validate(payload)

    restart_payload = {
        "receipt_id": "0" * 64,
        "component": "writer",
        "uninterrupted_attempt_id": "writer-a",
        "resumed_attempt_id": "writer-b",
        "durable_checkpoint": artifact,
        "interrupted_no_final_publication_receipt": artifact,
        "uninterrupted_outputs": (artifact,),
        "resumed_outputs": (other,),
        "uninterrupted_process_receipt": artifact,
        "resumed_process_receipt": other,
    }
    restart = _identified(G00CDurableRestartReceiptV4, restart_payload, "receipt_id")
    for changes in (
        {"resumed_attempt_id": "writer-a"},
        {"uninterrupted_outputs": ()},
        {"resumed_outputs": (artifact,)},
        {"resumed_outputs": (changed,)},
    ):
        payload = restart.model_dump(mode="json")
        payload.update(changes)
        payload["receipt_id"] = "0" * 64
        with pytest.raises(ValueError):
            G00CDurableRestartReceiptV4.model_validate(payload)

    evidence_payload = {
        "evidence_id": "0" * 64,
        "execution_authority_id": "6" * 64,
        "sampler_plan": artifact,
        "sampler_plan_id": plan.plan_id,
        "row_hierarchy": artifact,
        "nested_training_row_order": artifact,
        "uninterrupted_draw_trace": artifact,
        "resumed_draw_trace": other,
        "uninterrupted_state_trace": artifact,
        "resumed_state_trace": other,
        "restart_receipt": artifact,
        "restart_receipt_id": restart.receipt_id,
        "implementation_sha256": artifact.sha256,
    }
    evidence = _identified(G00CSamplerEvidenceV4, evidence_payload, "evidence_id")
    with pytest.raises(ValueError, match="distinct paths"):
        G00CSamplerEvidenceV4.model_validate(
            evidence.model_copy(
                update={"resumed_draw_trace": artifact, "evidence_id": "0" * 64}
            ).model_dump(mode="json")
        )

    mapping = G00CSemanticPublicationArtifactV4(
        role="a",
        filename="a.bin",
        source_artifact=artifact,
        published_sha256=artifact.sha256,
        published_size_bytes=artifact.size_bytes,
    )
    with pytest.raises(ValueError, match="published bytes"):
        G00CSemanticPublicationArtifactV4.model_validate(
            {**mapping.model_dump(mode="json"), "published_sha256": "7" * 64}
        )
    inventory = _identified(
        G00CInnerPublicationInventoryV4,
        {
            "inventory_id": "0" * 64,
            "terminal_status": "extension_required",
            "artifacts": (mapping,),
        },
        "inventory_id",
    )
    for artifacts in (
        (
            mapping.model_copy(update={"role": "b", "filename": "same.bin"}),
            mapping.model_copy(update={"filename": "same.bin"}),
        ),
        (mapping.model_copy(update={"role": "final_decision"}),),
    ):
        payload = inventory.model_dump(mode="json")
        payload["inventory_id"] = "0" * 64
        payload["artifacts"] = [item.model_dump(mode="json") for item in artifacts]
        with pytest.raises(ValueError):
            G00CInnerPublicationInventoryV4.model_validate(payload)

    bundle_payload = {
        "bundle_id": "0" * 64,
        "execution_authority": artifact,
        "execution_authority_id": "8" * 64,
        "selection_freeze": artifact,
        "selection_freeze_id": "9" * 64,
        "seed_schedule": artifact,
        "seed_schedule_id": "a" * 64,
        "row_roles": artifact,
        "feature_ranking_receipt": artifact,
        "feature_selection_result": artifact,
        "feature_selection_result_id": "b" * 64,
        "sample_size_selection_result": artifact,
        "sample_size_selection_result_id": "c" * 64,
        "common_support_receipt": artifact,
        "feature_refit_replay_receipt": artifact,
        "sample_refit_replay_receipt": artifact,
        "sampler_evidence": artifact,
        "sampler_restart_receipt": artifact,
        "source_access_ledger_receipt": artifact,
        "monitor_receipt": artifact,
        "base_support_audit_contract": artifact,
        "base_support_audit_receipt": artifact,
        "inner_publication_manifest": artifact,
        "terminal_status": "extension_required",
    }
    bundle = _identified(G00CExecutionBundleV5, bundle_payload, "bundle_id")
    with pytest.raises(ValueError, match="bundle_id mismatch"):
        G00CExecutionBundleV5.model_validate(
            {**bundle.model_dump(mode="json"), "bundle_id": "f" * 64}
        )
    for changes in (
        {"terminal_status": "pass"},
        {"materialization_receipt": artifact},
        {"terminal_status": "failed_integrity"},
        {"failure_receipt": artifact},
        {"terminal_status": "fail_no_saturation"},
        {"extension_sampler_evidence": artifact},
    ):
        payload = bundle.model_dump(mode="json")
        payload.update(changes)
        payload["bundle_id"] = "0" * 64
        with pytest.raises(ValueError):
            G00CExecutionBundleV5.model_validate(payload)

    verified = VerifiedG00CExecutionV5(
        terminal_status="extension_required",
        execution_bundle_id=bundle.bundle_id,
        execution_authority_id=bundle.execution_authority_id,
        selection_freeze_id=bundle.selection_freeze_id,
        feature_selection_result_id=bundle.feature_selection_result_id,
        sample_size_selection_result_id=bundle.sample_size_selection_result_id,
        feature_ranking_receipt_id="d" * 64,
        sampler_evidence_id="e" * 64,
        sampler_restart_receipt_id="f" * 64,
        feature_refit_replay_receipt_id="0" * 64,
        sample_refit_replay_receipt_id="1" * 64,
        support_audit_receipt_ids=("2" * 64,),
        source_access_ledger_receipt_id="3" * 64,
        monitor_receipt_id="4" * 64,
        materialization_receipt_id=None,
        inner_publication_manifest_id="5" * 64,
        verified_artifact_sha256s=(artifact.sha256,),
    )
    decision = build_g00c_decision_receipt_v5(verified)
    with pytest.raises(ValueError, match="receipt_id mismatch"):
        G00CDecisionReceiptV5.model_validate(
            {**decision.model_dump(mode="json"), "receipt_id": "f" * 64}
        )
    for changes in (
        {"may_parent_g00d": True},
        {"materialization_receipt_id": "6" * 64},
        {"verified_artifact_sha256s": ()},
        {"verified_artifact_sha256s": (artifact.sha256, artifact.sha256)},
    ):
        payload = decision.model_dump(mode="json")
        payload.update(changes)
        payload["receipt_id"] = "0" * 64
        with pytest.raises(ValueError):
            G00CDecisionReceiptV5.model_validate(payload)


def test_dev37_publication_verifiers_reject_every_control_surface_tamper(
    tmp_path: Path,
) -> None:
    malformed = tmp_path / "malformed.sha256"
    malformed.write_text("not-a-sum\n")
    with pytest.raises(IntegrityError, match="malformed"):
        _parse_sha256sums(malformed)
    malformed.write_text(f"{'0' * 64}  same\n{'1' * 64}  same\n")
    with pytest.raises(IntegrityError, match="duplicated"):
        _parse_sha256sums(malformed)

    class NestedArtifacts(BaseModel):
        direct: ArtifactRef
        sequence: tuple[ArtifactRef, ...]
        mapping: dict[str, ArtifactRef]

    source = tmp_path / "source.bin"
    source.write_bytes(b"semantic-source")
    source_ref = _artifact(source, root=tmp_path, media_type="application/octet-stream")
    semantic = semantic_artifact_map_v4(
        {
            "z": NestedArtifacts(
                direct=source_ref,
                sequence=(source_ref,),
                mapping={"same": source_ref},
            )
        }
    )
    assert semantic == {"z.direct": source_ref}

    publication = tmp_path / "publication"
    publication.mkdir()
    published = publication / "payload.bin"
    published.write_bytes(source.read_bytes())
    mapping = G00CSemanticPublicationArtifactV4(
        role="source.payload",
        filename="payload.bin",
        source_artifact=source_ref,
        published_sha256=source_ref.sha256,
        published_size_bytes=source_ref.size_bytes,
    )
    inventory = _identified(
        G00CInnerPublicationInventoryV4,
        {
            "inventory_id": "0" * 64,
            "terminal_status": "extension_required",
            "artifacts": (mapping,),
        },
        "inventory_id",
    )
    inventory_path = publication / "artifacts.json"
    inventory_path.write_bytes(canonical_json_bytes(inventory.model_dump(mode="json")) + b"\n")
    sums = publication / "SHA256SUMS"
    sums.write_text(f"{source_ref.sha256}  payload.bin\n")
    committed = publication / "COMMITTED"
    committed.write_text("extension_required\n")
    publisher_hash = "2" * 64
    event = publication / "PUBLICATION_EVENT.json"
    event_payload = {
        "destination_preexisted": False,
        "directory_fsync_completed": True,
        "manifest_written_last": True,
        "no_clobber": True,
        "publisher_implementation_sha256": publisher_hash,
        "status": "pass",
    }
    event.write_text(json.dumps(event_payload, sort_keys=True, separators=(",", ":")) + "\n")
    manifest = _identified(
        G00CInnerPublicationManifestV4,
        {
            "manifest_id": "0" * 64,
            "execution_authority_id": "3" * 64,
            "selection_freeze_id": "4" * 64,
            "feature_selection_result_id": "5" * 64,
            "sample_size_selection_result_id": "6" * 64,
            "sampler_evidence_id": "7" * 64,
            "terminal_status": "extension_required",
            "artifact_inventory": _artifact(
                inventory_path, root=publication, media_type="application/json"
            ),
            "artifact_inventory_id": inventory.inventory_id,
            "sha256sums": _artifact(sums, root=publication, media_type="text/plain"),
            "committed": _artifact(committed, root=publication, media_type="text/plain"),
            "publication_event_receipt": _artifact(
                event, root=publication, media_type="application/json"
            ),
            "publisher_implementation_sha256": publisher_hash,
        },
        "manifest_id",
    )
    bundle = G00CExecutionBundleV5.model_construct(
        execution_authority_id=manifest.execution_authority_id,
        selection_freeze_id=manifest.selection_freeze_id,
        feature_selection_result_id=manifest.feature_selection_result_id,
        sample_size_selection_result_id=manifest.sample_size_selection_result_id,
        terminal_status="extension_required",
    )
    expected = {mapping.role: source_ref}
    assert verify_g00c_inner_publication_v4(
        publication, bundle, manifest, expected_artifacts=expected
    )
    with pytest.raises(IntegrityError, match="cross-wired"):
        verify_g00c_inner_publication_v4(
            publication,
            bundle.model_copy(update={"execution_authority_id": "8" * 64}),
            manifest,
            expected_artifacts=expected,
        )
    with pytest.raises(IntegrityError, match="identity"):
        verify_g00c_inner_publication_v4(
            publication,
            bundle,
            manifest.model_copy(update={"artifact_inventory_id": "8" * 64}),
            expected_artifacts=expected,
        )
    with pytest.raises(IntegrityError, match="omits or adds"):
        verify_g00c_inner_publication_v4(publication, bundle, manifest, expected_artifacts={})
    extra = publication / "EXTRA"
    extra.write_text("x")
    with pytest.raises(IntegrityError, match="missing or extra"):
        verify_g00c_inner_publication_v4(publication, bundle, manifest, expected_artifacts=expected)
    extra.unlink()

    sums.write_text(f"{'9' * 64}  payload.bin\n")
    altered_manifest = manifest.model_copy(
        update={"sha256sums": _artifact(sums, root=publication, media_type="text/plain")}
    )
    with pytest.raises(IntegrityError, match="SHA inventory"):
        verify_g00c_inner_publication_v4(
            publication, bundle, altered_manifest, expected_artifacts=expected
        )
    sums.write_text(f"{source_ref.sha256}  payload.bin\n")
    committed.write_text("pass\n")
    altered_manifest = manifest.model_copy(
        update={"committed": _artifact(committed, root=publication, media_type="text/plain")}
    )
    with pytest.raises(IntegrityError, match="COMMITTED"):
        verify_g00c_inner_publication_v4(
            publication, bundle, altered_manifest, expected_artifacts=expected
        )
    committed.write_text("extension_required\n")
    event.write_text(json.dumps({**event_payload, "status": "wrong"}) + "\n")
    altered_manifest = manifest.model_copy(
        update={
            "publication_event_receipt": _artifact(
                event, root=publication, media_type="application/json"
            )
        }
    )
    with pytest.raises(IntegrityError, match="event"):
        verify_g00c_inner_publication_v4(
            publication, bundle, altered_manifest, expected_artifacts=expected
        )

    bad_destination = tmp_path / "bad-publication"

    def wrong_writer(root: Path) -> G00CInnerPublicationManifestV4:
        root.mkdir(exist_ok=True)
        return manifest.model_copy(update={"terminal_status": "pass"})

    with pytest.raises(IntegrityError, match="cross-wired manifest"):
        publish_g00c_inner_v4(
            bad_destination,
            terminal_status="extension_required",
            publisher_implementation_sha256=publisher_hash,
            writer=wrong_writer,
        )
    assert not bad_destination.exists()

    outer = tmp_path / "outer-negative"
    bundle2, _, seal, _ = _build_extension_stop(outer)
    assert bundle2.terminal_status == "extension_required"
    verify_g00c_final_seal_v1(outer, seal)
    outer_extra = outer / "EXTRA"
    outer_extra.write_text("x")
    with pytest.raises(IntegrityError, match="missing or extra"):
        verify_g00c_final_seal_v1(outer, seal)
    outer_extra.unlink()
    with pytest.raises(IntegrityError, match="cross-wired"):
        verify_g00c_final_seal_v1(outer, seal.model_copy(update={"execution_bundle_id": "a" * 64}))
    inventory_path = outer / "artifacts.json"
    inventory_path.write_text("{}\n")
    altered_seal = seal.model_copy(
        update={
            "outer_artifact_inventory": _artifact(
                inventory_path, root=outer, media_type="application/json"
            )
        }
    )
    with pytest.raises(IntegrityError, match="outer artifacts"):
        verify_g00c_final_seal_v1(outer, altered_seal)


def test_dev37_source_and_refit_helpers_fail_closed_on_malformed_surfaces(
    tmp_path: Path,
) -> None:
    malformed = tmp_path / "malformed-features.json"
    malformed.write_text("not-json")
    with pytest.raises(IntegrityError, match="malformed"):
        _canonical_feature_records(malformed)
    malformed.write_text(
        json.dumps(
            {
                "feature_count": 1,
                "features": [{"feature_id": "g", "namespace": "test", "namespace_version": "1"}],
                "ordered_hash": "0" * 64,
            }
        )
    )
    with pytest.raises(IntegrityError, match="hash differs"):
        _canonical_feature_records(malformed)
    records = [
        {"feature_id": "g", "namespace": "test", "namespace_version": "1"},
        {"feature_id": "g", "namespace": "test", "namespace_version": "1"},
    ]
    malformed.write_text(
        json.dumps(
            {
                "feature_count": 2,
                "features": records,
                "ordered_hash": hashlib.sha256(canonical_json_bytes(records)).hexdigest(),
            }
        )
    )
    with pytest.raises(IntegrityError, match="not unique"):
        _canonical_feature_records(malformed)
    records[1] = {"feature_id": "h", "namespace": "test", "extra": "1"}
    malformed.write_text(
        json.dumps(
            {
                "feature_count": 2,
                "features": records,
                "ordered_hash": hashlib.sha256(canonical_json_bytes(records)).hexdigest(),
            }
        )
    )
    with pytest.raises(IntegrityError, match="another schema"):
        _canonical_feature_records(malformed)
    no_var = tmp_path / "no-var.h5ad"
    with h5py.File(no_var, "x") as handle:
        handle.create_group("X")
    with pytest.raises(IntegrityError, match="var/_index"):
        _source_feature_ids(no_var)
    corrupt_h5ad = tmp_path / "corrupt.h5ad"
    corrupt_h5ad.write_text("not-hdf5")
    with pytest.raises(IntegrityError, match="cannot be read"):
        _source_feature_ids(corrupt_h5ad)

    real_root = tmp_path / "real"
    real_root.mkdir()
    store, source_root, feature_index = _build_real_g00b(real_root)
    authority_root = tmp_path / "source-authority"
    authority_root.mkdir()
    binding = _binding(authority_root, store, feature_index)
    with pytest.raises(IntegrityError, match="external G00B manifest"):
        open_verified_source_plane_v4(
            authority_root,
            binding.model_copy(update={"accepted_g00b_manifest_sha256": "e" * 64}),
            source_plane_root=tmp_path / "real",
            source_files_root=source_root,
        )
    altered_feature_index = authority_root / "altered-features.json"
    records = json.loads(feature_index.read_text())
    records["features"][0]["namespace_version"] = "changed"
    records["ordered_hash"] = hashlib.sha256(canonical_json_bytes(records["features"])).hexdigest()
    altered_feature_index.write_bytes(canonical_json_bytes(records) + b"\n")
    with pytest.raises(IntegrityError, match="feature-index bytes"):
        open_verified_source_plane_v4(
            authority_root,
            binding.model_copy(
                update={
                    "canonical_feature_index": _artifact(
                        altered_feature_index,
                        root=authority_root,
                        media_type="application/json",
                    )
                }
            ),
            source_plane_root=tmp_path / "real",
            source_files_root=source_root,
        )
    with pytest.raises(IntegrityError, match="G00B identity"):
        open_verified_source_plane_v4(
            authority_root,
            binding.model_copy(update={"virtual_store_id": "f" * 64}),
            source_plane_root=tmp_path / "real",
            source_files_root=source_root,
        )
    with pytest.raises(IntegrityError, match="PuroR mapping"):
        open_verified_source_plane_v4(
            authority_root,
            binding.model_copy(update={"puro_r_canonical_index": 0}),
            source_plane_root=tmp_path / "real",
            source_files_root=source_root,
        )
    with pytest.raises(IntegrityError, match="absent"):
        derive_hierarchy_from_g00b_v4(store, np.asarray([999], dtype=np.int64))
    with pytest.raises(ValueError, match="nonempty rows"):
        recompute_feature_ranking_v4(
            store,
            tuple(f"g{index:04d}" for index in range(4096)) + ("CUSTOM001_PuroR",),
            np.asarray([], dtype=np.int64),
            puro_r_canonical_index=4096,
        )
    with pytest.raises(IntegrityError, match="matrix width"):
        recompute_feature_ranking_v4(
            store,
            ("wrong",),
            np.asarray([10, 20, 30, 40], dtype=np.int64),
            puro_r_canonical_index=0,
        )
    feature_ids = tuple(f"g{index:04d}" for index in range(4096)) + ("CUSTOM001_PuroR",)
    with pytest.raises(IntegrityError, match="lacks one frozen checkpoint"):
        recompute_feature_ranking_v4(
            store,
            feature_ids,
            np.asarray([10, 20], dtype=np.int64),
            puro_r_canonical_index=4096,
        )
    with pytest.raises(IntegrityError, match="ranking row is absent"):
        _locate_source_indices(store, np.asarray([999], dtype=np.int64))
    with pytest.raises(IntegrityError, match="refit row is absent"):
        _checkpoint_codes(store, np.asarray([999], dtype=np.int64))

    v1_store = VirtualCanonicalCountStore.__new__(VirtualCanonicalCountStore)
    v1_store.manifest = VirtualCanonicalCountStoreManifestV1.model_construct()
    with pytest.raises(IntegrityError, match="accepted V2"):
        derive_hierarchy_from_g00b_v4(v1_store, np.asarray([10], dtype=np.int64))

    zero_root = tmp_path / "zero-library"
    zero_root.mkdir()
    zero_store, zero_sources, _ = _build_real_g00b(zero_root)
    with h5py.File(zero_sources / "rest.h5ad", "r+") as handle:
        handle["X/data"][:3] = 0
    with pytest.raises(IntegrityError, match="zero primary-UMI"):
        recompute_feature_ranking_v4(
            zero_store,
            tuple(f"g{index:04d}" for index in range(4096)) + ("CUSTOM001_PuroR",),
            np.asarray([10, 30, 40], dtype=np.int64),
            puro_r_canonical_index=4096,
        )
    crosswalk_path = zero_store.path / zero_store.manifest.guide_target_crosswalk.relative_uri
    crosswalk = pd.read_parquet(crosswalk_path)
    crosswalk.iloc[:-1].to_parquet(crosswalk_path, index=False)
    with pytest.raises(IntegrityError, match="crosswalk and locator"):
        derive_hierarchy_from_g00b_v4(zero_store, np.asarray([10, 20], dtype=np.int64))
    accumulated, digest = _thin_and_accumulate(
        sparse.csr_matrix(np.asarray([[1, 2], [3, 4]], dtype=np.int32)),
        np.asarray([0, 2], dtype=np.int8),
        np.asarray([1.0, 2.0]),
        seed=5,
    )
    assert accumulated.shape == (3, 2) and len(digest) == 64

    plan = G00CSamplerPlanV4.model_construct(entries=())
    schedule = derive_refit_seed_schedule("1" * 64)
    with pytest.raises(IntegrityError, match="lack candidates"):
        derive_checkpoint_statistics_v4(
            store,
            plan,
            pd.DataFrame(),
            np.asarray([10], dtype=np.int64),
            np.asarray([20], dtype=np.int64),
            np.arange(4096, dtype=np.int64),
            schedule,
            candidate_kind="training_cells",
        )
    entry = G00CSamplerPlanEntryV4.model_construct(
        candidate_kind="training_cells",
        candidate_value=1,
        refit_draw_id=0,
        expected_trace_rows=2,
    )
    short_plan = G00CSamplerPlanV4.model_construct(entries=(entry,))
    short_trace = pd.DataFrame(
        {
            "entry_index": [0],
            "draw_index": [0],
            "row_id": [10],
            "inverse_probability_weight": [1.0],
        }
    )
    with pytest.raises(IntegrityError, match="frozen entry size"):
        derive_checkpoint_statistics_v4(
            store,
            short_plan,
            short_trace,
            np.asarray([10], dtype=np.int64),
            np.asarray([20], dtype=np.int64),
            np.arange(4096, dtype=np.int64),
            schedule,
            candidate_kind="training_cells",
        )
    receipt = G00CRefitReplayReceiptV4.model_construct(
        candidate_kind="training_cells",
        candidate_values=(1,),
        selected_candidate=1,
        reference_candidate=1,
        modeled_feature_count=4096,
        preregistered_audit_draw_ids=(0,),
    )
    records = pd.DataFrame({"draw_id": [0], "candidate_value": [2]})
    with pytest.raises(IntegrityError, match="no source-derived"):
        recompute_refit_rows_v4(
            records,
            receipt,
            np.zeros((59, 1, 3, 4096), dtype=np.float64),
            np.ones((59, 3, 4096), dtype=np.float64),
        )
    zero_records = pd.DataFrame({"draw_id": [0], "candidate_value": [1]})
    with pytest.raises(IntegrityError, match="denominator"):
        recompute_refit_rows_v4(
            zero_records,
            receipt,
            np.ones((59, 1, 3, 4096), dtype=np.float64),
            np.zeros((59, 3, 4096), dtype=np.float64),
        )


def test_dev37_monitor_access_and_restart_verifiers_derive_all_failure_gates(
    tmp_path: Path,
) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    store, _, feature_index = _build_real_g00b(real_root)
    root = tmp_path / "evidence"
    root.mkdir()
    binding = _binding(root, store, feature_index)
    roles = pd.DataFrame(
        {
            "row_id": [10, 20, 30, 40],
            "role": [
                "training_fit",
                "training_validation",
                "heldout_source_query",
                "protected_heldout_stimulated",
            ],
        }
    )
    roles_path = root / "roles.parquet"
    roles.to_parquet(roles_path, index=False)
    role_records = tuple(
        FoldRowRoleRecord(
            role=role,  # type: ignore[arg-type]
            rows=1,
            row_ids_hash=_ordered_row_hash(np.asarray([row_id], dtype=np.int64)),
        )
        for row_id, role in zip(roles["row_id"], roles["role"], strict=True)
    )
    authority = G00CD1ExecutionAuthorityFreezeV3.model_construct(
        authority_id="a" * 64,
        source_plane=binding,
        row_role_freeze=_artifact(
            roles_path, root=root, media_type="application/vnd.apache.parquet"
        ),
        row_roles=role_records,
        monitored_process_tree_ceiling_bytes=4096,
    )

    monitor_table = pd.DataFrame(
        {
            "monotonic_ns": [1_000_000_000, 1_200_000_000, 1_400_000_000],
            "root_pid": [20, 20, 20],
            "process_tree_rss_bytes": [100, 200, 150],
            "descendants": [0, 1, 0],
            "readable": [True, True, True],
        },
        columns=MONITOR_COLUMNS,
    )
    monitor_path = root / "monitor.parquet"
    monitor_table.to_parquet(monitor_path, index=False)
    freeze = G00CMonitorFreezeV1(
        implementation_sha256="b" * 64,
        polling_interval_milliseconds=200,
        maximum_unreadable_samples=0,
        maximum_unreadable_fraction=0,
        maximum_consecutive_unreadable_samples=0,
        maximum_temporal_gap_milliseconds=500,
        maximum_process_tree_rss_bytes=4096,
    )
    monitor = _identified(
        G00CProcessTreeMonitorReceiptV4,
        {
            "receipt_id": "0" * 64,
            "execution_authority_id": authority.authority_id,
            "attempt_id": "monitor-a",
            "monitor_pid": 10,
            "monitored_root_pid": 20,
            "trace": _artifact(
                monitor_path, root=root, media_type="application/vnd.apache.parquet"
            ),
            "samples": 3,
            "maximum_process_tree_rss_bytes": 200,
            "unreadable_samples": 0,
            "maximum_consecutive_unreadable_samples": 0,
            "maximum_temporal_gap_milliseconds": 200.0,
            "descendants_observed": 1,
            "access_start_monotonic_ns": 1_100_000_000,
            "access_end_monotonic_ns": 1_300_000_000,
            "monitor_start_monotonic_ns": 1_000_000_000,
            "monitor_end_monotonic_ns": 1_400_000_000,
        },
        "receipt_id",
    )
    with pytest.raises(IntegrityError, match="another authority"):
        verify_g00c_monitor_v4(
            root, freeze, authority, monitor.model_copy(update={"execution_authority_id": "c" * 64})
        )

    def monitor_failure(table: pd.DataFrame, match: str, **receipt_changes: object) -> None:
        table.to_parquet(monitor_path, index=False)
        changed = monitor.model_copy(
            update={
                "trace": _artifact(
                    monitor_path, root=root, media_type="application/vnd.apache.parquet"
                ),
                **receipt_changes,
            }
        )
        with pytest.raises(IntegrityError, match=match):
            verify_g00c_monitor_v4(root, freeze, authority, changed)

    monitor_failure(monitor_table[["monotonic_ns"]], "another schema")
    duplicate_time = monitor_table.copy()
    duplicate_time.loc[1, "monotonic_ns"] = duplicate_time.loc[0, "monotonic_ns"]
    monitor_failure(duplicate_time, "strictly increasing")
    wrong_pid = monitor_table.copy()
    wrong_pid.loc[0, "root_pid"] = 99
    monitor_failure(wrong_pid, "differs from its trace")
    unreadable = monitor_table.copy()
    unreadable.loc[1, "readable"] = False
    monitor_failure(
        unreadable,
        "violates",
        maximum_process_tree_rss_bytes=150,
        unreadable_samples=1,
        maximum_consecutive_unreadable_samples=1,
        descendants_observed=0,
    )
    monitor_table.to_parquet(monitor_path, index=False)

    ledger = pd.DataFrame(
        {
            "sequence": [0, 1, 2],
            "monotonic_ns": [1, 2, 3],
            "role": [
                "feature_ranking_training_fit",
                "refit_training_validation",
                "materialization_heldout_source_query",
            ],
            "row_id": [10, 20, 30],
            "source_index": [0, 0, 1],
            "checkpoint": ["Rest", "Rest", "Stim8hr"],
        },
        columns=ACCESS_COLUMNS,
    )
    ledger_path = root / "ledger.parquet"
    ledger.to_parquet(ledger_path, index=False)
    role_hashes = {
        str(role): _ordered_row_hash(frame["row_id"].to_numpy(dtype=np.int64))
        for role, frame in ledger.groupby(ledger["role"].astype(str), sort=True)
    }
    access = _identified(
        G00CSourceAccessLedgerReceiptV4,
        {
            "receipt_id": "0" * 64,
            "execution_authority_id": authority.authority_id,
            "source_binding_id": binding.binding_id,
            "ledger": _artifact(
                ledger_path, root=root, media_type="application/vnd.apache.parquet"
            ),
            "access_rows": 3,
            "role_row_hashes": role_hashes,
            "protected_row_ids_sha256": role_records[-1].row_ids_hash,
        },
        "receipt_id",
    )
    with pytest.raises(IntegrityError, match="another authority"):
        verify_g00c_source_access_v4(
            root,
            authority,
            store,
            access.model_copy(update={"execution_authority_id": "d" * 64}),
        )

    def access_failure(table: pd.DataFrame, match: str, **receipt_changes: object) -> None:
        table.to_parquet(ledger_path, index=False)
        changed = access.model_copy(
            update={
                "ledger": _artifact(
                    ledger_path, root=root, media_type="application/vnd.apache.parquet"
                ),
                **receipt_changes,
            }
        )
        with pytest.raises(IntegrityError, match=match):
            verify_g00c_source_access_v4(root, authority, store, changed)

    access_failure(ledger.drop(columns="sequence"), "another row surface")
    wrong_sequence = ledger.copy()
    wrong_sequence.loc[1, "sequence"] = 9
    access_failure(wrong_sequence, "another row surface")
    backward = ledger.copy()
    backward.loc[2, "monotonic_ns"] = 0
    access_failure(backward, "not monotonic")
    access_failure(ledger, "another protected", protected_row_ids_sha256="e" * 64)
    wrong_role = ledger.copy()
    wrong_role.loc[0, "role"] = "unknown"
    access_failure(wrong_role, "outside its declared role")
    access_failure(ledger, "role hashes", role_row_hashes={"wrong": "f" * 64})
    protected = ledger.copy()
    protected.loc[2, ["role", "row_id", "source_index", "checkpoint"]] = [
        "materialization_heldout_source_query",
        40,
        2,
        "Stim48hr",
    ]
    permissive_roles = dict(role_hashes)
    permissive_roles["materialization_heldout_source_query"] = _ordered_row_hash(
        np.asarray([40], dtype=np.int64)
    )
    # It first rejects the row-role mismatch, which is the same protected-access gate.
    access_failure(protected, "outside its declared role", role_row_hashes=permissive_roles)
    unknown = ledger.copy()
    unknown.loc[2, "row_id"] = 999
    unknown_hashes = dict(role_hashes)
    unknown_hashes["materialization_heldout_source_query"] = _ordered_row_hash(
        np.asarray([999], dtype=np.int64)
    )
    access_failure(unknown, "outside its declared role", role_row_hashes=unknown_hashes)
    wrong_source = ledger.copy()
    wrong_source.loc[2, "source_index"] = 0
    access_failure(wrong_source, "source indices")
    wrong_checkpoint = ledger.copy()
    wrong_checkpoint.loc[2, "checkpoint"] = "Rest"
    access_failure(wrong_checkpoint, "checkpoints")

    out_a = root / "a.bin"
    out_b = root / "b.bin"
    out_a.write_bytes(b"same")
    out_b.write_bytes(b"same")
    checkpoint = root / "checkpoint.json"
    checkpoint.write_text("{}\n")
    interrupted = root / "interrupted.json"
    interrupted.write_text(
        json.dumps(
            {
                "attempt_id": "attempt-b",
                "final_payload_published": False,
                "status": "interrupted_checkpoint_committed",
            }
        )
    )
    process_a = root / "process-a.json"
    process_b = root / "process-b.json"
    process_a.write_text(
        json.dumps({"attempt_id": "attempt-a", "exit_code": 0, "pid": 10, "status": "complete"})
    )
    process_b.write_text(
        json.dumps({"attempt_id": "attempt-b", "exit_code": 0, "pid": 20, "status": "complete"})
    )
    restart = _identified(
        G00CDurableRestartReceiptV4,
        {
            "receipt_id": "0" * 64,
            "component": "writer",
            "uninterrupted_attempt_id": "attempt-a",
            "resumed_attempt_id": "attempt-b",
            "durable_checkpoint": _artifact(checkpoint, root=root, media_type="application/json"),
            "interrupted_no_final_publication_receipt": _artifact(
                interrupted, root=root, media_type="application/json"
            ),
            "uninterrupted_outputs": (
                _artifact(out_a, root=root, media_type="application/octet-stream"),
            ),
            "resumed_outputs": (
                _artifact(out_b, root=root, media_type="application/octet-stream"),
            ),
            "uninterrupted_process_receipt": _artifact(
                process_a, root=root, media_type="application/json"
            ),
            "resumed_process_receipt": _artifact(
                process_b, root=root, media_type="application/json"
            ),
        },
        "receipt_id",
    )
    interrupted.write_text("{}\n")
    with pytest.raises(IntegrityError, match="publication evidence"):
        verify_g00c_restart_v4(
            root,
            restart.model_copy(
                update={
                    "interrupted_no_final_publication_receipt": _artifact(
                        interrupted, root=root, media_type="application/json"
                    )
                }
            ),
        )
    interrupted.write_text(
        json.dumps(
            {
                "attempt_id": "attempt-b",
                "final_payload_published": False,
                "status": "interrupted_checkpoint_committed",
            }
        )
    )
    process_b.write_text(
        json.dumps({"attempt_id": "attempt-b", "exit_code": 1, "pid": 20, "status": "failed"})
    )
    with pytest.raises(IntegrityError, match="not independent"):
        verify_g00c_restart_v4(
            root,
            restart.model_copy(
                update={
                    "interrupted_no_final_publication_receipt": _artifact(
                        interrupted, root=root, media_type="application/json"
                    ),
                    "resumed_process_receipt": _artifact(
                        process_b, root=root, media_type="application/json"
                    ),
                }
            ),
        )


def test_dev37_complete_authority_and_receipt_contracts_are_content_addressed(
    tmp_path: Path,
) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    store, _, feature_index = _build_real_g00b(real_root)
    root = tmp_path / "authority"
    root.mkdir()
    binding = _binding(root, store, feature_index)
    payload_path = root / "payload.json"
    payload_path.write_text("{}\n")
    artifact = _artifact(payload_path, root=root, media_type="application/json")
    roles = (
        FoldRowRoleRecord(role="training_fit", rows=2, row_ids_hash="1" * 64),
        FoldRowRoleRecord(role="training_validation", rows=1, row_ids_hash="2" * 64),
        FoldRowRoleRecord(role="heldout_source_query", rows=1, row_ids_hash="3" * 64),
        FoldRowRoleRecord(role="protected_heldout_stimulated", rows=1, row_ids_hash="4" * 64),
    )
    implementation_roles = (
        "feature_ranking",
        "source_authority",
        "hierarchy_derivation",
        "refit",
        "sampler",
        "support_auditor",
        "monitor",
        "source_access_auditor",
        "materialization_verifier",
        "refit_replay_verifier",
        "publication_verifier",
        "restart_verifier",
        "extension_verifier",
        "execution_verifier",
        "decision_verifier",
    )
    implementation = G00CImplementationAuthorityV3(
        dev37_code_commit="5" * 40,
        wheel=artifact,
        normalized_sdist=artifact,
        implementation_tree_sha256="6" * 64,
        environment_lock=artifact,
        environment_kind="exact_local_lock",
        execution_environment_digest=f"sha256:{'7' * 64}",
        implementations=tuple(
            G00CImplementationBindingV3(role=role, artifact=artifact)
            for role in implementation_roles
        ),
    )
    authority = _identified(
        G00CD1ExecutionAuthorityFreezeV3,
        {
            "authority_id": "0" * 64,
            "selection_freeze": artifact,
            "selection_freeze_id": "8" * 64,
            "row_role_freeze": artifact,
            "row_roles": roles,
            "nested_training_row_order": artifact,
            "nested_training_row_order_hash": "9" * 64,
            "feature_reference_rows": artifact,
            "feature_reference_rows_hash": "a" * 64,
            "sampler_row_hierarchy": artifact,
            "sampler_hierarchy_rows": 2,
            "hierarchy_derivation_receipt": artifact,
            "source_plane": binding,
            "seed_schedule": artifact,
            "seed_schedule_id": "b" * 64,
            "base_sampler_plan": artifact,
            "base_sampler_plan_id": "c" * 64,
            "base_sampler_expected_trace_rows": 1,
            "common_support_prior": G00CCommonSupportPriorV2(),
            "base_support_audit_contract": artifact,
            "implementation": implementation,
            "fresh_attempt_id": "dev37-test",
            "publication_root_uri": "publication",
        },
        "authority_id",
    )
    assert authority.authority_id == authority.identity(id_field="authority_id")

    def invalid_authority(**changes: object) -> None:
        payload = authority.model_dump(mode="json")
        payload.update(changes)
        payload["authority_id"] = "0" * 64
        with pytest.raises(ValueError):
            G00CD1ExecutionAuthorityFreezeV3.model_validate(payload)

    invalid_authority(row_roles=tuple(reversed(roles)))
    invalid_authority(sampler_hierarchy_rows=3)
    with pytest.raises(ValueError, match="authority_id mismatch"):
        G00CD1ExecutionAuthorityFreezeV3.model_validate(
            {**authority.model_dump(mode="json"), "authority_id": "f" * 64}
        )

    hierarchy = _identified(
        G00CHierarchyDerivationReceiptV4,
        {
            "receipt_id": "0" * 64,
            "source_binding_id": binding.binding_id,
            "hierarchy": artifact,
            "row_count": 2,
            "ordered_row_ids_sha256": "1" * 64,
            "source_index_sha256": "2" * 64,
            "target_code_sha256": "3" * 64,
            "guide_code_sha256": "4" * 64,
            "is_control_sha256": "5" * 64,
        },
        "receipt_id",
    )
    ranking = _identified(
        G00CFeatureRankingReceiptV4,
        {
            "receipt_id": "0" * 64,
            "execution_authority_id": authority.authority_id,
            "source_binding_id": binding.binding_id,
            "fit_rows_hash": "1" * 64,
            "fit_row_count": 2,
            "complete_ranking": artifact,
            "ordered_feature_hash": "2" * 64,
            "selected_prefix_hash": "3" * 64,
            "canonical_feature_index_hash": binding.canonical_feature_index_hash,
            "puro_r_canonical_index": binding.puro_r_canonical_index,
        },
        "receipt_id",
    )
    for model, id_field in ((hierarchy, "receipt_id"), (ranking, "receipt_id")):
        payload = model.model_dump(mode="json")
        payload[id_field] = "f" * 64
        with pytest.raises(ValueError, match="mismatch"):
            type(model).model_validate(payload)

    with pytest.raises(ValueError, match="at least one role"):
        G00CSourceAccessLedgerReceiptV4.model_validate(
            {
                "receipt_id": "0" * 64,
                "execution_authority_id": authority.authority_id,
                "source_binding_id": binding.binding_id,
                "ledger": artifact.model_dump(mode="json"),
                "access_rows": 0,
                "role_row_hashes": {},
                "protected_row_ids_sha256": "4" * 64,
            }
        )
    assert _sampler_implementation_hash(authority, "sampler") == artifact.sha256


def test_dev37_sampler_and_support_verifiers_reject_crosswired_metadata(
    tmp_path: Path,
) -> None:
    root = tmp_path / "sampler-crosswire"
    root.mkdir()
    payload = root / "payload.json"
    payload.write_text("{}\n")
    artifact = _artifact(payload, root=root, media_type="application/json")
    sampler_impl = root / "sampler.py"
    support_impl = root / "support.py"
    sampler_impl.write_text("# sampler\n")
    support_impl.write_text("# support\n")
    sampler_ref = _artifact(sampler_impl, root=root, media_type="text/x-python")
    support_ref = _artifact(support_impl, root=root, media_type="text/x-python")
    authority = G00CD1ExecutionAuthorityFreezeV3.model_construct(
        authority_id="1" * 64,
        selection_freeze_id="2" * 64,
        base_sampler_plan=artifact,
        base_sampler_plan_id="3" * 64,
        base_sampler_expected_trace_rows=1,
        sampler_row_hierarchy=artifact,
        nested_training_row_order=artifact,
        seed_schedule=artifact,
        implementation=G00CImplementationAuthorityV3.model_construct(
            implementations=(
                G00CImplementationBindingV3(role="sampler", artifact=sampler_ref),
                G00CImplementationBindingV3(role="support_auditor", artifact=support_ref),
            )
        ),
    )
    evidence = G00CSamplerEvidenceV4.model_construct(
        execution_authority_id="4" * 64,
        sampler_plan=artifact,
        sampler_plan_id=authority.base_sampler_plan_id,
        row_hierarchy=artifact,
        nested_training_row_order=artifact,
        implementation_sha256=sampler_ref.sha256,
    )
    with pytest.raises(IntegrityError, match="pre-access authority"):
        verify_g00c_sampler_v4(root, authority, evidence, pd.DataFrame())
    with pytest.raises(IntegrityError, match="no unique missing"):
        _sampler_implementation_hash(authority, "missing")

    schedule = derive_refit_seed_schedule("8" * 64)
    entries = tuple(
        G00CSamplerPlanEntryV4(
            candidate_kind=kind,
            candidate_value=value,
            refit_draw_id=draw,
            macro_updates=2,
            expected_trace_rows=8192,
        )
        for kind, values in (
            ("feature_count", (256, 512, 1024, 2048, 4096)),
            ("training_cells", (50_000, 100_000, 250_000, 500_000, 1_000_000)),
        )
        for value in values
        for draw in range(59)
    )
    plan = _identified(
        G00CSamplerPlanV4,
        {
            "plan_id": "0" * 64,
            "execution_authority_namespace": authority.selection_freeze_id,
            "seed_schedule_id": schedule.schedule_id,
            "grid_stage": "base",
            "entries": entries,
            "resume_after_macro_update": 1,
            "expected_total_trace_rows": len(entries) * 8192,
        },
        "plan_id",
    )
    plan_path = root / "plan.json"
    plan_path.write_text(plan.model_dump_json() + "\n")
    plan_ref = _artifact(plan_path, root=root, media_type="application/json")
    deep_authority = authority.model_copy(
        update={
            "base_sampler_plan": plan_ref,
            "base_sampler_plan_id": plan.plan_id,
            "base_sampler_expected_trace_rows": plan.expected_total_trace_rows,
        }
    )
    deep_evidence = evidence.model_copy(
        update={
            "execution_authority_id": deep_authority.authority_id,
            "sampler_plan": plan_ref,
            "sampler_plan_id": plan.plan_id,
        }
    )
    with pytest.raises(IntegrityError, match="model artifact is invalid"):
        verify_g00c_sampler_v4(root, deep_authority, deep_evidence, pd.DataFrame())
    schedule_path = root / "schedule.json"
    schedule_path.write_text(schedule.model_dump_json() + "\n")
    schedule_ref = _artifact(schedule_path, root=root, media_type="application/json")
    wrong_schedule_authority = deep_authority.model_copy(update={"seed_schedule": schedule_ref})
    wrong_schedule_payload = plan.model_dump(mode="json")
    wrong_schedule_payload.update({"plan_id": "0" * 64, "seed_schedule_id": "9" * 64})
    wrong_schedule_plan = _identified(G00CSamplerPlanV4, wrong_schedule_payload, "plan_id")
    wrong_schedule_path = root / "wrong-schedule-plan.json"
    wrong_schedule_path.write_text(wrong_schedule_plan.model_dump_json() + "\n")
    wrong_schedule_ref = _artifact(wrong_schedule_path, root=root, media_type="application/json")
    wrong_schedule_authority = wrong_schedule_authority.model_copy(
        update={
            "base_sampler_plan": wrong_schedule_ref,
            "base_sampler_plan_id": wrong_schedule_plan.plan_id,
        }
    )
    wrong_schedule_evidence = deep_evidence.model_copy(
        update={
            "sampler_plan": wrong_schedule_ref,
            "sampler_plan_id": wrong_schedule_plan.plan_id,
        }
    )
    with pytest.raises(IntegrityError, match="another seed schedule"):
        verify_g00c_sampler_v4(
            root, wrong_schedule_authority, wrong_schedule_evidence, pd.DataFrame()
        )
    bad_order_path = root / "bad-order.parquet"
    pd.DataFrame({"wrong": [1]}).to_parquet(bad_order_path, index=False)
    order_authority = deep_authority.model_copy(
        update={
            "seed_schedule": schedule_ref,
            "nested_training_row_order": _artifact(
                bad_order_path, root=root, media_type="application/vnd.apache.parquet"
            ),
        }
    )
    order_evidence = deep_evidence.model_copy(
        update={"nested_training_row_order": order_authority.nested_training_row_order}
    )
    with pytest.raises(IntegrityError, match="another schema"):
        verify_g00c_sampler_v4(root, order_authority, order_evidence, pd.DataFrame())

    contract = _identified(
        G00CSupportAuditContractV2,
        {
            "contract_id": "0" * 64,
            "stage": "extension",
            "feature_candidate_counts": (),
            "cell_candidate_counts": (2_000_000,),
        },
        "contract_id",
    )
    contract_path = root / "support-contract.json"
    contract_path.write_text(contract.model_dump_json() + "\n")
    contract_ref = _artifact(contract_path, root=root, media_type="application/json")
    receipt = G00CSupportAuditReceiptV3.model_construct(
        execution_authority_id="6" * 64,
        support_contract=contract_ref,
        support_contract_id=contract.contract_id,
        sampler_evidence=artifact,
        sampler_evidence_id="7" * 64,
        implementation_sha256=support_ref.sha256,
    )
    with pytest.raises(IntegrityError, match="cross-wired"):
        verify_g00c_support_v4(
            root,
            authority,
            evidence,
            artifact,
            receipt,
            plan=G00CSamplerPlanV4.model_construct(),
            hierarchy=pd.DataFrame(),
            order=np.asarray([], dtype=np.int64),
            trace=pd.DataFrame(),
        )


def test_dev37_all_terminal_receipt_ids_and_shapes_fail_closed(tmp_path: Path) -> None:
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"same")
    second.write_bytes(b"same")
    artifact = _artifact(first, root=tmp_path, media_type="application/octet-stream")
    other = _artifact(second, root=tmp_path, media_type="application/octet-stream")

    access = _identified(
        G00CSourceAccessLedgerReceiptV4,
        {
            "receipt_id": "0" * 64,
            "execution_authority_id": "1" * 64,
            "source_binding_id": "2" * 64,
            "ledger": artifact,
            "access_rows": 1,
            "role_row_hashes": {"training": "3" * 64},
            "protected_row_ids_sha256": "4" * 64,
        },
        "receipt_id",
    )
    monitor = _identified(
        G00CProcessTreeMonitorReceiptV4,
        {
            "receipt_id": "0" * 64,
            "execution_authority_id": "1" * 64,
            "attempt_id": "attempt",
            "monitor_pid": 1,
            "monitored_root_pid": 2,
            "trace": artifact,
            "samples": 2,
            "maximum_process_tree_rss_bytes": 1,
            "unreadable_samples": 0,
            "maximum_consecutive_unreadable_samples": 0,
            "maximum_temporal_gap_milliseconds": 1,
            "descendants_observed": 0,
            "access_start_monotonic_ns": 2,
            "access_end_monotonic_ns": 3,
            "monitor_start_monotonic_ns": 1,
            "monitor_end_monotonic_ns": 4,
        },
        "receipt_id",
    )
    restart = _identified(
        G00CDurableRestartReceiptV4,
        {
            "receipt_id": "0" * 64,
            "component": "writer",
            "uninterrupted_attempt_id": "a",
            "resumed_attempt_id": "b",
            "durable_checkpoint": artifact,
            "interrupted_no_final_publication_receipt": artifact,
            "uninterrupted_outputs": (artifact,),
            "resumed_outputs": (other,),
            "uninterrupted_process_receipt": artifact,
            "resumed_process_receipt": other,
        },
        "receipt_id",
    )
    evidence = _identified(
        G00CSamplerEvidenceV4,
        {
            "evidence_id": "0" * 64,
            "execution_authority_id": "1" * 64,
            "sampler_plan": artifact,
            "sampler_plan_id": "2" * 64,
            "row_hierarchy": artifact,
            "nested_training_row_order": artifact,
            "uninterrupted_draw_trace": artifact,
            "resumed_draw_trace": other,
            "uninterrupted_state_trace": artifact,
            "resumed_state_trace": other,
            "restart_receipt": artifact,
            "restart_receipt_id": restart.receipt_id,
            "implementation_sha256": "3" * 64,
        },
        "evidence_id",
    )
    refit = _identified(
        G00CRefitReplayReceiptV4,
        {
            "receipt_id": "0" * 64,
            "execution_authority_id": "1" * 64,
            "selection_freeze_id": "2" * 64,
            "seed_schedule_id": "3" * 64,
            "source_binding_id": "4" * 64,
            "sampler_evidence_id": evidence.evidence_id,
            "source_access_ledger_receipt_id": access.receipt_id,
            "candidate_kind": "feature_count",
            "selected_candidate": 256,
            "reference_candidate": 4096,
            "modeled_feature_count": 4096,
            "candidate_values": (256, 4096),
            "refit_records": artifact,
            "source_derived_statistics": artifact,
            "replayed_rows": artifact,
            "fit_row_hashes": ("5" * 64, "6" * 64),
            "validation_row_hash": "7" * 64,
            "training_count_hash": "8" * 64,
            "validation_count_vector_hash": "9" * 64,
            "thinning_trace_hash": "a" * 64,
            "training_shape": (59, 2, 3, 4096),
            "validation_shape": (59, 3, 4096),
        },
        "receipt_id",
    )
    materialization = _identified(
        G00CMaterializationReceiptV4,
        {
            "receipt_id": "0" * 64,
            "execution_authority_id": "1" * 64,
            "source_binding_id": "2" * 64,
            "base_materialization_receipt": artifact,
            "base_materialization_receipt_id": "3" * 64,
            "source_access_ledger_receipt": artifact,
            "source_access_ledger_receipt_id": access.receipt_id,
            "monitor_receipt": artifact,
            "monitor_receipt_id": monitor.receipt_id,
            "writer_restart_receipt": artifact,
            "writer_restart_receipt_id": restart.receipt_id,
        },
        "receipt_id",
    )
    for model, id_field in (
        (access, "receipt_id"),
        (monitor, "receipt_id"),
        (restart, "receipt_id"),
        (evidence, "evidence_id"),
        (refit, "receipt_id"),
        (materialization, "receipt_id"),
    ):
        payload = model.model_dump(mode="json")
        payload[id_field] = "f" * 64
        with pytest.raises(ValueError, match="mismatch"):
            type(model).model_validate(payload)

    for changes in (
        {"candidate_values": ()},
        {"training_shape": (58, 2, 3, 4096)},
        {"validation_shape": (58, 3, 4096)},
        {"fit_row_hashes": ("5" * 64,)},
    ):
        payload = refit.model_dump(mode="json")
        payload.update(changes)
        payload["receipt_id"] = "0" * 64
        with pytest.raises(ValueError):
            G00CRefitReplayReceiptV4.model_validate(payload)
