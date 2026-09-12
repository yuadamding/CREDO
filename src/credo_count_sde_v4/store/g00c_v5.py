"""Source-derived Dev37 G00C execution, decision, and final-seal verification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

from ..contracts import (
    ArtifactRef,
    G00CCommonSupportMetricReceiptV1,
    G00CD1ExecutionAuthorityFreezeV3,
    G00CDecisionReceiptV5,
    G00CDurableRestartReceiptV4,
    G00CExecutionBundleV5,
    G00CFeatureRankingReceiptV4,
    G00CFeatureSelectionResultV3,
    G00CFinalSealV1,
    G00CInnerPublicationManifestV4,
    G00CMaterializationReceiptV4,
    G00CProcessTreeMonitorReceiptV4,
    G00CRefitReplayReceiptV4,
    G00CRefitSeedScheduleV1,
    G00CSamplerEvidenceV4,
    G00CSampleSizeExtensionFreezeV2,
    G00CSampleSizeSelectionResultV3,
    G00CSourceAccessLedgerReceiptV4,
    G00CSupportAuditReceiptV3,
)
from ..errors import IntegrityError
from .g00c_evidence_v4 import verify_g00c_monitor_v4, verify_g00c_source_access_v4
from .g00c_extension_v4 import verify_g00c_extension_freeze_v2
from .g00c_materialization_v3 import _role_rows
from .g00c_materialization_v4 import verify_g00c_materialization_v4
from .g00c_publication_v4 import (
    semantic_artifact_map_v4,
    verify_g00c_final_seal_v1,
    verify_g00c_inner_publication_v4,
)
from .g00c_refit_replay_v4 import verify_g00c_refit_replay_v4
from .g00c_sampler_v4 import verify_g00c_sampler_v4, verify_g00c_support_v4
from .g00c_source_v4 import (
    open_verified_g00b_v4,
    verify_g00c_feature_ranking_v4,
    verify_g00c_hierarchy_v4,
)
from .g00c_v3 import (
    _path,
    _read_model,
    _verify_feature_selection_v3,
    _verify_sample_size_selection_v3,
    verify_g00c_d1_freeze_v1,
)
from .g00c_v4 import _derived_eligibility


@dataclass(frozen=True)
class VerifiedG00CExecutionV5:
    """Only values recomputed through the complete Dev37 source-derived chain."""

    terminal_status: str
    execution_bundle_id: str
    execution_authority_id: str
    selection_freeze_id: str
    feature_selection_result_id: str
    sample_size_selection_result_id: str
    feature_ranking_receipt_id: str
    sampler_evidence_id: str
    sampler_restart_receipt_id: str
    feature_refit_replay_receipt_id: str
    sample_refit_replay_receipt_id: str
    support_audit_receipt_ids: tuple[str, ...]
    source_access_ledger_receipt_id: str
    monitor_receipt_id: str
    materialization_receipt_id: str | None
    inner_publication_manifest_id: str
    verified_artifact_sha256s: tuple[str, ...]


def _implementation_hash(authority: G00CD1ExecutionAuthorityFreezeV3, role: str) -> str:
    matches = [
        binding.artifact.sha256
        for binding in authority.implementation.implementations
        if binding.role == role
    ]
    if len(matches) != 1:
        raise IntegrityError(f"Dev37 authority has no unique {role} implementation.")
    return matches[0]


def _artifact_refs(value: Any) -> list[ArtifactRef]:
    if isinstance(value, ArtifactRef):
        return [value]
    if isinstance(value, BaseModel):
        refs: list[ArtifactRef] = []
        for field in type(value).model_fields:
            refs.extend(_artifact_refs(getattr(value, field)))
        return refs
    if isinstance(value, (tuple, list)):
        refs = []
        for item in value:
            refs.extend(_artifact_refs(item))
        return refs
    if isinstance(value, dict):
        refs = []
        for item in value.values():
            refs.extend(_artifact_refs(item))
        return refs
    return []


def _verify_refit_binding(
    receipt: G00CRefitReplayReceiptV4,
    *,
    result: G00CFeatureSelectionResultV3 | G00CSampleSizeSelectionResultV3,
    kind: str,
    selected: int,
    reference: int,
    modeled_features: int,
    sampler_evidence_id: str,
    access_receipt_id: str,
) -> None:
    if (
        receipt.candidate_kind != kind
        or receipt.refit_records != result.refit_records
        or receipt.selected_candidate != selected
        or receipt.reference_candidate != reference
        or receipt.modeled_feature_count != modeled_features
        or receipt.sampler_evidence_id != sampler_evidence_id
        or receipt.source_access_ledger_receipt_id != access_receipt_id
    ):
        raise IntegrityError("Dev37 refit replay binds another selection/source surface.")


def _expected_access_rows(
    accessed: list[tuple[str, int]],
    *,
    root: Path,
    authority: G00CD1ExecutionAuthorityFreezeV3,
    selected_rows: np.ndarray[Any, Any] | None,
) -> list[tuple[str, int]]:
    expected = list(accessed)
    if selected_rows is not None:
        roles = _role_rows(root, authority)
        for role, values in (
            ("materialization_training_fit", selected_rows),
            ("materialization_training_validation", roles["training_validation"]),
            ("materialization_heldout_source_query", roles["heldout_source_query"]),
        ):
            expected.extend((role, int(value)) for value in values)
    return expected


def _semantic_models(
    *,
    authority: G00CD1ExecutionAuthorityFreezeV3,
    freeze: BaseModel,
    schedule: BaseModel,
    ranking: BaseModel,
    feature: BaseModel,
    sample: BaseModel,
    common: BaseModel,
    sampler: BaseModel,
    sampler_restart: BaseModel,
    access: BaseModel,
    monitor: BaseModel,
    support: tuple[BaseModel, ...],
    feature_replay: BaseModel,
    sample_replay: BaseModel,
    extension: BaseModel | None,
    extension_sampler: BaseModel | None,
    extension_restart: BaseModel | None,
    materialization: BaseModel | None,
) -> dict[str, BaseModel]:
    models: dict[str, BaseModel] = {
        "authority": authority,
        "common_support": common,
        "feature_ranking": ranking,
        "feature_refit_replay": feature_replay,
        "feature_selection": feature,
        "freeze": freeze,
        "monitor": monitor,
        "sample_refit_replay": sample_replay,
        "sample_selection": sample,
        "sampler": sampler,
        "sampler_restart": sampler_restart,
        "schedule": schedule,
        "source_access": access,
    }
    for index, receipt in enumerate(support):
        models[f"support_{index}"] = receipt
    for name, model in (
        ("extension_freeze", extension),
        ("extension_sampler", extension_sampler),
        ("extension_restart", extension_restart),
        ("materialization", materialization),
    ):
        if model is not None:
            models[name] = model
    return models


def verify_g00c_execution_v5(
    root: Path,
    publication_root: Path,
    bundle: G00CExecutionBundleV5,
    *,
    source_plane_root: Path,
    source_files_root: Path,
) -> VerifiedG00CExecutionV5:
    """Recompute every Dev37 claim-bearing relation from authority and source bytes."""

    authority = _read_model(root, bundle.execution_authority, G00CD1ExecutionAuthorityFreezeV3)
    freeze = verify_g00c_d1_freeze_v1(root, authority)
    if (
        bundle.execution_authority_id != authority.authority_id
        or bundle.selection_freeze != authority.selection_freeze
        or bundle.selection_freeze_id != freeze.freeze_id
        or bundle.seed_schedule != authority.seed_schedule
        or bundle.seed_schedule_id != authority.seed_schedule_id
        or bundle.row_roles != authority.row_role_freeze
        or bundle.base_support_audit_contract != authority.base_support_audit_contract
    ):
        raise IntegrityError("Dev37 execution bundle is cross-wired to another authority.")
    store, canonical_feature_ids = open_verified_g00b_v4(
        root,
        authority,
        source_plane_root=source_plane_root,
        source_files_root=source_files_root,
    )
    hierarchy = verify_g00c_hierarchy_v4(root, authority, store)
    schedule = _read_model(root, bundle.seed_schedule, G00CRefitSeedScheduleV1)
    if schedule.schedule_id != bundle.seed_schedule_id:
        raise IntegrityError("Dev37 execution uses another seed schedule.")

    accessed: list[tuple[str, int]] = []

    def record_access(role: str, row_ids: np.ndarray[Any, Any]) -> None:
        accessed.extend((role, int(value)) for value in row_ids)

    ranking_receipt = _read_model(root, bundle.feature_ranking_receipt, G00CFeatureRankingReceiptV4)
    ranking = verify_g00c_feature_ranking_v4(
        root,
        authority,
        store,
        canonical_feature_ids,
        ranking_receipt,
        access_callback=record_access,
    )
    feature = _read_model(root, bundle.feature_selection_result, G00CFeatureSelectionResultV3)
    sample = _read_model(root, bundle.sample_size_selection_result, G00CSampleSizeSelectionResultV3)
    if (
        feature.result_id != bundle.feature_selection_result_id
        or sample.result_id != bundle.sample_size_selection_result_id
    ):
        raise IntegrityError("Dev37 selection result identities differ from their artifacts.")
    ordered_features = pd.read_parquet(_path(root, feature.ordered_features))
    source_prefix = ranking.iloc[:4096][["rank", "canonical_index", "feature_id"]]
    if not ordered_features.equals(source_prefix.reset_index(drop=True)):
        raise IntegrityError("Dev37 selected feature surface is not the source-derived prefix.")

    sampler = _read_model(root, bundle.sampler_evidence, G00CSamplerEvidenceV4)
    plan, order, trace = verify_g00c_sampler_v4(root, authority, sampler, hierarchy)
    sampler_restart = _read_model(root, bundle.sampler_restart_receipt, G00CDurableRestartReceiptV4)
    if sampler.restart_receipt != bundle.sampler_restart_receipt or (
        sampler_restart.receipt_id != sampler.restart_receipt_id
    ):
        raise IntegrityError("Dev37 sampler restart receipt is cross-wired.")
    base_support = _read_model(root, bundle.base_support_audit_receipt, G00CSupportAuditReceiptV3)
    if (
        base_support.support_contract != bundle.base_support_audit_contract
        or base_support.sampler_evidence != bundle.sampler_evidence
    ):
        raise IntegrityError("Dev37 base support evidence is cross-wired.")
    support_tables = [
        verify_g00c_support_v4(
            root,
            authority,
            sampler,
            bundle.sampler_evidence,
            base_support,
            plan=plan,
            hierarchy=hierarchy,
            order=order,
            trace=trace,
        )
    ]
    support_receipts = [base_support]
    extension = None
    extension_sampler = None
    extension_restart = None
    extension_plan_trace = None
    sample_candidates = freeze.base_cell_grid
    if sample.grid_stage == "extension":
        required = (
            bundle.extension_freeze,
            bundle.extension_sampler_evidence,
            bundle.extension_sampler_restart_receipt,
            bundle.extension_support_audit_contract,
            bundle.extension_support_audit_receipt,
        )
        if any(item is None for item in required):
            raise IntegrityError("Dev37 extension execution lacks its complete V4 chain.")
        extension = _read_model(
            root,
            bundle.extension_freeze,  # type: ignore[arg-type]
            G00CSampleSizeExtensionFreezeV2,
        )
        extension_plan = verify_g00c_extension_freeze_v2(
            root, extension, authority=authority, feature_result=feature
        )
        extension_sampler = _read_model(
            root,
            bundle.extension_sampler_evidence,  # type: ignore[arg-type]
            G00CSamplerEvidenceV4,
        )
        checked_plan, extension_order, extension_trace = verify_g00c_sampler_v4(
            root,
            authority,
            extension_sampler,
            hierarchy,
            expected_plan_artifact=extension.extension_sampler_plan,
            expected_plan_id=extension.extension_sampler_plan_id,
        )
        if checked_plan != extension_plan or not np.array_equal(extension_order, order):
            raise IntegrityError("Dev37 extension sampler changes the frozen row order.")
        extension_restart = _read_model(
            root,
            bundle.extension_sampler_restart_receipt,  # type: ignore[arg-type]
            G00CDurableRestartReceiptV4,
        )
        if (
            extension_sampler.restart_receipt != bundle.extension_sampler_restart_receipt
            or extension_restart.receipt_id != extension_sampler.restart_receipt_id
        ):
            raise IntegrityError("Dev37 extension restart receipt is cross-wired.")
        extension_support = _read_model(
            root,
            bundle.extension_support_audit_receipt,  # type: ignore[arg-type]
            G00CSupportAuditReceiptV3,
        )
        if (
            extension_support.support_contract != bundle.extension_support_audit_contract
            or extension_support.sampler_evidence != bundle.extension_sampler_evidence
        ):
            raise IntegrityError("Dev37 extension support receipt is cross-wired.")
        support_tables.append(
            verify_g00c_support_v4(
                root,
                authority,
                extension_sampler,
                bundle.extension_sampler_evidence,  # type: ignore[arg-type]
                extension_support,
                plan=extension_plan,
                hierarchy=hierarchy,
                order=order,
                trace=extension_trace,
            )
        )
        support_receipts.append(extension_support)
        extension_plan_trace = (extension_plan, extension_trace)
        sample_candidates = (*freeze.base_cell_grid, 2_000_000)

    feature_support = _derived_eligibility(
        tuple(support_tables),
        kind="feature_count",
        candidates=freeze.feature_ranking.candidate_feature_counts,
    )
    sample_support = _derived_eligibility(
        tuple(support_tables), kind="training_cells", candidates=sample_candidates
    )
    selected_features, _ = _verify_feature_selection_v3(
        root,
        freeze=freeze,
        bundle=bundle,
        result=feature,
        schedule=schedule,
        support_eligible_override=feature_support,
    )
    selected_rows = _verify_sample_size_selection_v3(
        root,
        freeze=freeze,
        authority=authority,
        bundle=bundle,
        feature_result=feature,
        result=sample,
        schedule=schedule,
        support_eligible_override=sample_support,
    )
    feature_replay = _read_model(
        root, bundle.feature_refit_replay_receipt, G00CRefitReplayReceiptV4
    )
    sample_replay = _read_model(root, bundle.sample_refit_replay_receipt, G00CRefitReplayReceiptV4)
    access_receipt = _read_model(
        root, bundle.source_access_ledger_receipt, G00CSourceAccessLedgerReceiptV4
    )
    _verify_refit_binding(
        feature_replay,
        result=feature,
        kind="feature_count",
        selected=feature.selected_feature_count,
        reference=4096,
        modeled_features=4096,
        sampler_evidence_id=sampler.evidence_id,
        access_receipt_id=access_receipt.receipt_id,
    )
    selected_sample = sample.selected_training_cells or sample_candidates[-1]
    _verify_refit_binding(
        sample_replay,
        result=sample,
        kind="training_cells",
        selected=selected_sample,
        reference=sample_candidates[-1],
        modeled_features=feature.selected_feature_count,
        sampler_evidence_id=sampler.evidence_id,
        access_receipt_id=access_receipt.receipt_id,
    )
    feature_indices = ordered_features["canonical_index"].to_numpy(dtype=np.int64)
    validation_roles = _role_rows(root, authority)
    validation_rows = validation_roles["training_validation"]
    verify_g00c_refit_replay_v4(
        root,
        authority,
        feature_replay,
        store,
        plan,
        trace,
        order,
        validation_rows,
        feature_indices,
        access_callback=record_access,
    )
    verify_g00c_refit_replay_v4(
        root,
        authority,
        sample_replay,
        store,
        plan,
        trace,
        order,
        validation_rows,
        feature_indices,
        access_callback=record_access,
        additional_plan_trace=extension_plan_trace,
    )
    expected_status = "pass" if sample.selection_status == "selected" else sample.selection_status
    if bundle.terminal_status != "failed_integrity" and bundle.terminal_status != expected_status:
        raise IntegrityError("Dev37 terminal status differs from source-derived selection.")

    materialization = None
    materialization_base = None
    if bundle.terminal_status == "pass":
        if selected_rows is None or bundle.materialization_receipt is None:
            raise IntegrityError("Dev37 pass requires selected rows and materialization evidence.")
        materialization = _read_model(
            root, bundle.materialization_receipt, G00CMaterializationReceiptV4
        )
        if (
            materialization.source_access_ledger_receipt != bundle.source_access_ledger_receipt
            or materialization.source_access_ledger_receipt_id != access_receipt.receipt_id
            or materialization.monitor_receipt != bundle.monitor_receipt
        ):
            raise IntegrityError("Dev37 materialization does not bind monitor/access evidence.")
        materialization_base = verify_g00c_materialization_v4(
            root,
            store,
            authority,
            materialization,
            expected_selected_feature_ids=selected_features,
            expected_selected_rows=selected_rows,
        )
    if bundle.terminal_status == "failed_integrity":
        if bundle.failure_receipt is None:
            raise IntegrityError("Dev37 integrity failure lacks a typed receipt.")
        failure = json.loads(_path(root, bundle.failure_receipt).read_text())
        if failure != {
            "biological_claims": False,
            "execution_authority_id": authority.authority_id,
            "schema_version": 1,
            "selection_freeze_id": freeze.freeze_id,
            "status": "failed_integrity",
        }:
            raise IntegrityError("Dev37 integrity failure receipt is not exact.")

    ledger = verify_g00c_source_access_v4(root, authority, store, access_receipt)
    expected_access = _expected_access_rows(
        accessed,
        root=root,
        authority=authority,
        selected_rows=selected_rows if bundle.terminal_status == "pass" else None,
    )
    observed_access = list(
        zip(ledger["role"].astype(str), ledger["row_id"].astype(int), strict=True)
    )
    if observed_access != expected_access:
        raise IntegrityError("Dev37 full source-access ledger omits, adds, or reorders reads.")
    monitor = _read_model(root, bundle.monitor_receipt, G00CProcessTreeMonitorReceiptV4)
    monitor_trace = verify_g00c_monitor_v4(root, freeze.monitor, authority, monitor)
    if _implementation_hash(
        authority, "monitor"
    ) != freeze.monitor.implementation_sha256 or _implementation_hash(
        authority, "source_access_auditor"
    ) not in {item.sha256 for item in _artifact_refs(authority)}:
        raise IntegrityError("Dev37 monitoring/access implementations differ from authority.")
    if len(ledger):
        first_access = int(ledger["monotonic_ns"].iloc[0])
        last_access = int(ledger["monotonic_ns"].iloc[-1])
        if (
            monitor.access_start_monotonic_ns != first_access
            or monitor.access_end_monotonic_ns != last_access
            or first_access < int(monitor_trace["monotonic_ns"].iloc[0])
            or last_access > int(monitor_trace["monotonic_ns"].iloc[-1])
        ):
            raise IntegrityError("Dev37 process-tree monitor does not cover the source ledger.")

    common = _read_model(root, bundle.common_support_receipt, G00CCommonSupportMetricReceiptV1)
    publication = _read_model(
        root, bundle.inner_publication_manifest, G00CInnerPublicationManifestV4
    )
    semantic_models = _semantic_models(
        authority=authority,
        freeze=freeze,
        schedule=schedule,
        ranking=ranking_receipt,
        feature=feature,
        sample=sample,
        common=common,
        sampler=sampler,
        sampler_restart=sampler_restart,
        access=access_receipt,
        monitor=monitor,
        support=tuple(support_receipts),
        feature_replay=feature_replay,
        sample_replay=sample_replay,
        extension=extension,
        extension_sampler=extension_sampler,
        extension_restart=extension_restart,
        materialization=materialization,
    )
    expected_publication = semantic_artifact_map_v4(semantic_models)
    verify_g00c_inner_publication_v4(
        publication_root,
        bundle,
        publication,
        expected_artifacts=expected_publication,
    )
    if (
        publication.sampler_evidence_id != sampler.evidence_id
        or publication.publisher_implementation_sha256
        != _implementation_hash(authority, "publication_verifier")
    ):
        raise IntegrityError("Dev37 inner publication differs from source-derived evidence.")

    models = tuple(semantic_models.values()) + (
        bundle,
        publication,
        *((materialization_base,) if materialization_base is not None else ()),
    )
    verified = {
        freeze.dev33_canary.authority_archive_sha256,
        authority.source_plane.accepted_g00b_manifest_sha256,
        *(item.source_file_sha256 for item in authority.source_plane.source_files),
    }
    for model in models:
        for artifact in _artifact_refs(model):
            _path(root if model is not publication else publication_root, artifact)
            verified.add(artifact.sha256)
    verified.add(bundle.inner_publication_manifest.sha256)
    verified.update(item.sha256 for item in expected_publication.values())
    return VerifiedG00CExecutionV5(
        terminal_status=bundle.terminal_status,
        execution_bundle_id=bundle.bundle_id,
        execution_authority_id=authority.authority_id,
        selection_freeze_id=freeze.freeze_id,
        feature_selection_result_id=feature.result_id,
        sample_size_selection_result_id=sample.result_id,
        feature_ranking_receipt_id=ranking_receipt.receipt_id,
        sampler_evidence_id=sampler.evidence_id,
        sampler_restart_receipt_id=sampler_restart.receipt_id,
        feature_refit_replay_receipt_id=feature_replay.receipt_id,
        sample_refit_replay_receipt_id=sample_replay.receipt_id,
        support_audit_receipt_ids=tuple(item.receipt_id for item in support_receipts),
        source_access_ledger_receipt_id=access_receipt.receipt_id,
        monitor_receipt_id=monitor.receipt_id,
        materialization_receipt_id=(
            materialization.receipt_id if materialization is not None else None
        ),
        inner_publication_manifest_id=publication.manifest_id,
        verified_artifact_sha256s=tuple(sorted(verified)),
    )


def build_g00c_decision_receipt_v5(
    verified: VerifiedG00CExecutionV5,
) -> G00CDecisionReceiptV5:
    """Construct the outer-seal decision from independently verified values only."""

    payload: dict[str, Any] = {
        "receipt_id": "0" * 64,
        "execution_bundle_id": verified.execution_bundle_id,
        "execution_authority_id": verified.execution_authority_id,
        "selection_freeze_id": verified.selection_freeze_id,
        "feature_selection_result_id": verified.feature_selection_result_id,
        "sample_size_selection_result_id": verified.sample_size_selection_result_id,
        "feature_ranking_receipt_id": verified.feature_ranking_receipt_id,
        "sampler_evidence_id": verified.sampler_evidence_id,
        "sampler_restart_receipt_id": verified.sampler_restart_receipt_id,
        "feature_refit_replay_receipt_id": verified.feature_refit_replay_receipt_id,
        "sample_refit_replay_receipt_id": verified.sample_refit_replay_receipt_id,
        "support_audit_receipt_ids": verified.support_audit_receipt_ids,
        "source_access_ledger_receipt_id": verified.source_access_ledger_receipt_id,
        "monitor_receipt_id": verified.monitor_receipt_id,
        "materialization_receipt_id": verified.materialization_receipt_id,
        "inner_publication_manifest_id": verified.inner_publication_manifest_id,
        "verified_artifact_sha256s": verified.verified_artifact_sha256s,
        "terminal_status": verified.terminal_status,
        "may_parent_g00d": verified.terminal_status == "pass",
    }
    provisional = G00CDecisionReceiptV5.model_construct(**payload)
    payload["receipt_id"] = provisional.identity(id_field="receipt_id")
    return G00CDecisionReceiptV5.model_validate(payload)


def verify_g00c_decision_v5(
    root: Path,
    publication_root: Path,
    bundle: G00CExecutionBundleV5,
    receipt: G00CDecisionReceiptV5,
    *,
    source_plane_root: Path,
    source_files_root: Path,
) -> VerifiedG00CExecutionV5:
    """Reject a decision unless the complete Dev37 execution reproduces it."""

    verified = verify_g00c_execution_v5(
        root,
        publication_root,
        bundle,
        source_plane_root=source_plane_root,
        source_files_root=source_files_root,
    )
    authority = _read_model(root, bundle.execution_authority, G00CD1ExecutionAuthorityFreezeV3)
    if (
        _implementation_hash(authority, "decision_verifier")
        not in verified.verified_artifact_sha256s
    ):
        raise IntegrityError("Dev37 decision verifier bytes were not independently verified.")
    if receipt != build_g00c_decision_receipt_v5(verified):
        raise IntegrityError("Dev37 decision differs from source-derived sealed evidence.")
    return verified


def verify_g00c_sealed_decision_v5(
    root: Path,
    publication_root: Path,
    seal_root: Path,
    seal: G00CFinalSealV1,
    *,
    source_plane_root: Path,
    source_files_root: Path,
) -> VerifiedG00CExecutionV5:
    """Verify both layers and return the only Dev37 result eligible to parent G00D."""

    bundle, manifest, decision = verify_g00c_final_seal_v1(seal_root, seal)
    if manifest.manifest_id != decision.inner_publication_manifest_id:
        raise IntegrityError("Dev37 outer seal binds another inner publication identity.")
    verified = verify_g00c_decision_v5(
        root,
        publication_root,
        bundle,
        decision,
        source_plane_root=source_plane_root,
        source_files_root=source_files_root,
    )
    if (
        verified.inner_publication_manifest_id != manifest.manifest_id
        or verified.terminal_status != "pass"
        or not decision.may_parent_g00d
    ):
        raise IntegrityError("Only a sealed Dev37 pass may parent G00D.")
    return verified
