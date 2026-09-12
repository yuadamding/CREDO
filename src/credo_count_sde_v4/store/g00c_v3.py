"""Artifact-derived Dev35 G00C execution and decision verification.

This module never opens cohort expression sources.  It verifies only immutable
selection evidence and metadata artifacts.  A claim-bearing compact payload
requires a separately supplied materialization verifier whose implementation
hash is frozen by the concrete D1 authority.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from ..canonical import canonical_json_bytes, sha256_file
from ..contracts import (
    ArtifactRef,
    G00CCommonSupportMetricReceiptV1,
    G00CD1ExecutionAuthorityFreezeV1,
    G00CD1ExecutionAuthorityFreezeV2,
    G00CD1ExecutionAuthorityFreezeV3,
    G00CDecisionReceiptV3,
    G00CExecutionBundleV3,
    G00CExecutionBundleV4,
    G00CExecutionBundleV5,
    G00CFeatureSelectionResultV3,
    G00CHierarchyDerivationReceiptV4,
    G00CRefitSeedScheduleV1,
    G00CSamplerPlanV4,
    G00CSampleSizeExtensionFreezeV1,
    G00CSampleSizeExtensionFreezeV2,
    G00CSampleSizeSelectionResultV3,
    G00CSelectionFreezeContractV1,
    G00CSupportAuditContractV2,
)
from ..errors import IntegrityError
from .g00c_selection import feature_selection_decision_v3, sample_size_decision_v3

REFIT_V3_COLUMNS = (
    "candidate_kind",
    "draw_id",
    "candidate_value",
    "initialization",
    "training_sampler",
    "thinning",
    "validation_evaluation",
    "stochastic_optimizer_or_augmentation",
    "restart_interruption_point",
    "fit_row_hash",
    "validation_row_hash",
    "model_config_hash",
    "final_state_hash",
    "validation_total_count",
    "validation_nll_sum",
    "validation_nll_per_count",
    "fit_status",
)

SUPPORT_V1_COLUMNS = (
    "candidate_kind",
    "candidate_value",
    "dimension",
    "stratum_id",
    "cells",
    "weighted_effective_sample_size",
    "maximum_to_median_weight_ratio",
    "zero_support",
    "selection_eligible",
)

FEATURE_CURVE_V3_COLUMNS = (
    "feature_count",
    "mean_validation_nll",
    "q95_absolute_paired_nll_difference_to_4096",
    "support_eligible",
)

SAMPLE_CURVE_V3_COLUMNS = (
    "training_cells",
    "mean_validation_nll",
    "q95_absolute_paired_nll_difference_to_reference",
    "support_eligible",
)
MAXIMUM_FROZEN_TRAINING_ROWS = 2_000_000
FEATURE_REFERENCE_ROW_COUNT = 1_000_000


@dataclass(frozen=True)
class VerifiedG00CExecutionV3:
    """Values recomputed from one complete V3 authority chain."""

    terminal_status: str
    execution_bundle_id: str
    execution_authority_id: str
    selection_freeze_id: str
    feature_selection_result_id: str
    sample_size_selection_result_id: str
    seed_schedule_id: str
    common_support_receipt_sha256: str
    support_audit_sha256s: tuple[str, ...]
    grid_stage: str
    verified_artifact_sha256s: tuple[str, ...]


class MaterializedPassVerifier(Protocol):
    """Independent source/count verifier required before a V3 pass can parent G00D."""

    implementation_sha256: str

    def __call__(
        self,
        root: Path,
        bundle: G00CExecutionBundleV3,
        selected_feature_ids: tuple[str, ...],
        selected_training_rows: np.ndarray[Any, Any],
    ) -> None: ...


def _path(root: Path, artifact: ArtifactRef) -> Path:
    path = root / artifact.relative_uri
    if not path.is_file() or sha256_file(path) != artifact.sha256:
        raise IntegrityError(f"G00C V3 artifact failed verification: {artifact.relative_uri}.")
    if path.stat().st_size != artifact.size_bytes:
        raise IntegrityError(f"G00C V3 artifact size differs: {artifact.relative_uri}.")
    return path


def _read_model(root: Path, artifact: ArtifactRef, model: type[Any]) -> Any:
    try:
        return model.model_validate_json(_path(root, artifact).read_text())
    except Exception as exc:
        raise IntegrityError(
            f"G00C V3 model artifact is invalid: {artifact.relative_uri}."
        ) from exc


def _row_set_hash(values: np.ndarray[Any, Any]) -> str:
    ordered = np.sort(np.asarray(values, dtype="<i8"), kind="stable")
    return hashlib.sha256(ordered.tobytes(order="C")).hexdigest()


def _ordered_row_hash(values: np.ndarray[Any, Any]) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes(order="C")).hexdigest()


def _hash_int64(values: np.ndarray[Any, Any]) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes(order="C")).hexdigest()


def _implementation_hash(
    authority: (
        G00CD1ExecutionAuthorityFreezeV1
        | G00CD1ExecutionAuthorityFreezeV2
        | G00CD1ExecutionAuthorityFreezeV3
    ),
    role: str,
) -> str:
    matches = [
        binding.artifact.sha256
        for binding in authority.implementation.implementations
        if binding.role == role
    ]
    if len(matches) != 1:
        raise IntegrityError(f"Dev35 authority has no unique {role} implementation.")
    return matches[0]


def verify_g00c_d1_freeze_v1(
    root: Path,
    authority: (
        G00CD1ExecutionAuthorityFreezeV1
        | G00CD1ExecutionAuthorityFreezeV2
        | G00CD1ExecutionAuthorityFreezeV3
    ),
) -> G00CSelectionFreezeContractV1:
    """Verify byte-level relations in the concrete metadata-only D1 freeze."""

    freeze = _read_model(root, authority.selection_freeze, G00CSelectionFreezeContractV1)
    if freeze.freeze_id != authority.selection_freeze_id:
        raise IntegrityError("Dev35 D1 authority binds a different Dev34 freeze identity.")
    for binding in freeze.parent_bindings:
        _path(root, binding.artifact)
    archive = root / freeze.dev33_canary.authority_archive_uri
    if (
        not archive.is_file()
        or sha256_file(archive) != freeze.dev33_canary.authority_archive_sha256
    ):
        raise IntegrityError("Dev35 cannot verify the immutable Dev33-B prerequisite archive.")

    schedule = _read_model(root, authority.seed_schedule, G00CRefitSeedScheduleV1)
    if (
        schedule != freeze.seed_schedule
        or schedule.schedule_id != authority.seed_schedule_id
        or authority.seed_schedule != freeze.seed_schedule_artifact
        or authority.seed_schedule != freeze.refits.seed_schedule_artifact
    ):
        raise IntegrityError("Dev35 embedded and referenced seed schedules differ.")

    roles = pd.read_parquet(_path(root, authority.row_role_freeze))
    if tuple(roles.columns) != ("row_id", "role") or roles["row_id"].duplicated().any():
        raise IntegrityError("Dev35 row-role freeze has an invalid schema or duplicate rows.")
    row_ids = roles["row_id"].to_numpy(dtype=np.int64)
    role_names = roles["role"].astype(str).to_numpy()
    if (
        len(row_ids) != freeze.parent_eligible_rows
        or authority.row_role_freeze != freeze.row_role_freeze
    ):
        raise IntegrityError("Dev35 row-role bytes do not cover the exact frozen universe.")
    if authority.row_roles != freeze.row_roles:
        raise IntegrityError("Dev35 authority and design freeze row-role summaries differ.")
    for record in authority.row_roles:
        selected = row_ids[role_names == record.role]
        if len(selected) != record.rows or _row_set_hash(selected) != record.row_ids_hash:
            raise IntegrityError(f"Dev35 row-role artifact differs for {record.role}.")

    order = pd.read_parquet(_path(root, authority.nested_training_row_order))
    if tuple(order.columns) != ("rank", "row_id"):
        raise IntegrityError("Dev35 nested training order has an invalid schema.")
    ordered_rows = order["row_id"].to_numpy(dtype=np.int64)
    if (
        len(ordered_rows) < MAXIMUM_FROZEN_TRAINING_ROWS
        or not np.array_equal(order["rank"].to_numpy(dtype=np.int64), np.arange(1, len(order) + 1))
        or len(np.unique(ordered_rows)) != len(ordered_rows)
        or not np.isin(ordered_rows, row_ids[role_names == "training_fit"]).all()
        or _ordered_row_hash(ordered_rows) != authority.nested_training_row_order_hash
        or authority.nested_training_row_order_hash != freeze.training_scale_row_order_hash
    ):
        raise IntegrityError("Dev35 nested training order differs from the frozen semantic hash.")

    reference = pd.read_parquet(_path(root, authority.feature_reference_rows))
    if tuple(reference.columns) != ("rank", "row_id"):
        raise IntegrityError("Dev35 feature-reference rows have an invalid schema.")
    reference_rows = reference["row_id"].to_numpy(dtype=np.int64)
    if (
        len(reference_rows) != FEATURE_REFERENCE_ROW_COUNT
        or not np.array_equal(
            reference["rank"].to_numpy(dtype=np.int64),
            np.arange(1, FEATURE_REFERENCE_ROW_COUNT + 1),
        )
        or not np.array_equal(reference_rows, ordered_rows[:FEATURE_REFERENCE_ROW_COUNT])
        or _ordered_row_hash(reference_rows) != authority.feature_reference_rows_hash
        or authority.feature_reference_rows_hash != freeze.feature_ranking.fit_reference_rows_hash
        or authority.feature_reference_rows_hash
        != freeze.serial_selection.feature_selection_reference_rows_hash
    ):
        raise IntegrityError("Dev35 feature reference is not the first million frozen rows.")

    for artifact in (
        authority.implementation.wheel,
        authority.implementation.normalized_sdist,
        authority.implementation.environment_lock,
        *(binding.artifact for binding in authority.implementation.implementations),
    ):
        _path(root, artifact)
    environment = json.loads(_path(root, authority.implementation.environment_lock).read_text())
    if (
        environment.get("environment_kind") != authority.implementation.environment_kind
        or environment.get("execution_environment_digest")
        != authority.implementation.execution_environment_digest
    ):
        raise IntegrityError("Dev35 tested-environment bytes differ from the authority.")
    prior = json.loads(
        _path(root, freeze.common_support_metric.residual_frequency_artifact).read_text()
    )
    if prior != authority.common_support_prior.model_dump(mode="json"):
        raise IntegrityError("Dev35 common-support prior bytes differ from the authority.")
    base_support = _read_model(
        root, authority.base_support_audit_contract, G00CSupportAuditContractV2
    )
    if base_support.stage != "base":
        raise IntegrityError("Dev35 D1 authority does not bind the base support contract.")
    if isinstance(authority, (G00CD1ExecutionAuthorityFreezeV2, G00CD1ExecutionAuthorityFreezeV3)):
        hierarchy = pd.read_parquet(_path(root, authority.sampler_row_hierarchy))
        expected_columns = (
            "row_id",
            "source_index",
            "target_code",
            "guide_code",
            "is_control",
        )
        training_fit = row_ids[role_names == "training_fit"]
        if (
            tuple(hierarchy.columns) != expected_columns
            or len(hierarchy) != authority.sampler_hierarchy_rows
            or hierarchy["row_id"].duplicated().any()
            or _row_set_hash(hierarchy["row_id"].to_numpy(dtype=np.int64))
            != _row_set_hash(training_fit)
        ):
            raise IntegrityError("Dev36 sampler hierarchy differs from training-fit authority.")
        if isinstance(authority, G00CD1ExecutionAuthorityFreezeV3):
            receipt = _read_model(
                root,
                authority.hierarchy_derivation_receipt,
                G00CHierarchyDerivationReceiptV4,
            )
            column_hashes = (
                _hash_int64(hierarchy["row_id"].to_numpy(dtype=np.int64)),
                hashlib.sha256(
                    np.asarray(hierarchy["source_index"], dtype="<i2").tobytes()
                ).hexdigest(),
                hashlib.sha256(
                    np.asarray(hierarchy["target_code"], dtype="<i4").tobytes()
                ).hexdigest(),
                hashlib.sha256(
                    np.asarray(hierarchy["guide_code"], dtype="<i4").tobytes()
                ).hexdigest(),
                hashlib.sha256(
                    np.asarray(hierarchy["is_control"], dtype="|b1").tobytes()
                ).hexdigest(),
            )
            if (
                receipt.source_binding_id != authority.source_plane.binding_id
                or receipt.hierarchy != authority.sampler_row_hierarchy
                or receipt.row_count != authority.sampler_hierarchy_rows
                or column_hashes
                != (
                    receipt.ordered_row_ids_sha256,
                    receipt.source_index_sha256,
                    receipt.target_code_sha256,
                    receipt.guide_code_sha256,
                    receipt.is_control_sha256,
                )
            ):
                raise IntegrityError("Dev37 hierarchy receipt differs from frozen columns.")
            _path(root, authority.source_plane.accepted_g00b_parent)
            _path(root, authority.source_plane.canonical_feature_index)
            plan = _read_model(root, authority.base_sampler_plan, G00CSamplerPlanV4)
            if (
                plan.plan_id != authority.base_sampler_plan_id
                or plan.execution_authority_namespace != authority.selection_freeze_id
                or plan.seed_schedule_id != authority.seed_schedule_id
                or plan.grid_stage != "base"
                or plan.expected_total_trace_rows != authority.base_sampler_expected_trace_rows
            ):
                raise IntegrityError("Dev37 sampler plan differs from pre-access authority.")
    return freeze


def verify_g00c_extension_freeze_v1(
    root: Path,
    extension: G00CSampleSizeExtensionFreezeV1,
    *,
    base_freeze: G00CSelectionFreezeContractV1,
    base_feature_result: G00CFeatureSelectionResultV3,
    base_feature_artifact: ArtifactRef,
) -> None:
    """Verify that the two-million candidate was authorized before extension access."""

    base_bundle = _read_model(root, extension.base_execution_bundle, G00CExecutionBundleV3)
    receipt = _read_model(root, extension.base_extension_required_receipt, G00CDecisionReceiptV3)
    verify_g00c_decision_v3(root, base_bundle, receipt)
    schedule = _read_model(root, extension.seed_schedule_artifact, G00CRefitSeedScheduleV1)
    feature = _read_model(
        root, extension.base_feature_selection_result, G00CFeatureSelectionResultV3
    )
    selected_surface = pd.read_parquet(_path(root, extension.selected_feature_surface))
    ordered = pd.read_parquet(_path(root, feature.ordered_features))
    _path(root, extension.base_support_audit)
    extension_support = _read_model(
        root, extension.extension_support_audit_contract, G00CSupportAuditContractV2
    )
    if (
        receipt.terminal_status != "extension_required"
        or receipt.may_parent_g00d
        or receipt.selection_freeze_id != base_freeze.freeze_id
        or extension.base_selection_freeze_id != base_freeze.freeze_id
        or extension.base_feature_selection_result != base_feature_artifact
        or feature != base_feature_result
        or extension.selected_feature_count != feature.selected_feature_count
        or extension.selected_feature_order_sha256 != feature.selected_feature_order_sha256
        or extension.seed_schedule_id != base_freeze.seed_schedule.schedule_id
        or schedule != base_freeze.seed_schedule
        or extension_support.stage != "extension"
        or tuple(selected_surface.columns) != ("rank", "canonical_index", "feature_id")
        or not selected_surface.equals(ordered.iloc[: feature.selected_feature_count])
    ):
        raise IntegrityError("Dev35 extension authority differs from the passed base stop.")


def _load_refits_v3(
    root: Path,
    artifact: ArtifactRef,
    *,
    kind: str,
    candidates: tuple[int, ...],
    validation_row_hash: str,
    schedule: G00CRefitSeedScheduleV1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = pd.read_parquet(_path(root, artifact))
    if tuple(records.columns) != REFIT_V3_COLUMNS:
        raise IntegrityError("Dev35 refit provenance has an invalid schema.")
    if records.duplicated(["draw_id", "candidate_value"]).any():
        raise IntegrityError("Dev35 refit provenance contains duplicate draw/candidate pairs.")
    pivot = records.pivot(
        index="draw_id", columns="candidate_value", values="validation_nll_per_count"
    )
    if (
        set(records["candidate_kind"].astype(str)) != {kind}
        or tuple(pivot.columns.astype(int)) != candidates
        or tuple(pivot.index.astype(int)) != tuple(range(59))
        or pivot.isna().any().any()
    ):
        raise IntegrityError("Dev35 refit provenance is incomplete or uses another grid.")
    hashes = ("fit_row_hash", "validation_row_hash", "model_config_hash", "final_state_hash")
    if any(not records[field].astype(str).str.fullmatch(r"[0-9a-f]{64}").all() for field in hashes):
        raise IntegrityError("Dev35 refit provenance contains an invalid content hash.")
    if set(records["validation_row_hash"].astype(str)) != {validation_row_hash}:
        raise IntegrityError("Dev35 refits bind another validation-row set.")
    if (
        (records["fit_status"].astype(str) != "pass").any()
        or (records["validation_total_count"].astype(np.int64) <= 0).any()
        or not np.isfinite(records["validation_nll_sum"].astype(float)).all()
        or not np.isfinite(records["validation_nll_per_count"].astype(float)).all()
    ):
        raise IntegrityError("Dev35 refit provenance contains an invalid fit or NLL.")
    reconstructed = records["validation_nll_sum"].astype(float) / records[
        "validation_total_count"
    ].astype(float)
    if not np.allclose(
        reconstructed, records["validation_nll_per_count"].astype(float), atol=1e-15, rtol=0
    ):
        raise IntegrityError("Dev35 per-count NLL is not derived from its sum and denominator.")
    totals = records.pivot(
        index="draw_id", columns="candidate_value", values="validation_total_count"
    )
    if not np.equal(totals.to_numpy(), totals.to_numpy()[:, :1]).all():
        raise IntegrityError("Dev35 candidates do not share the same count denominator per draw.")
    schedule_by_draw = {record.draw_id: record for record in schedule.records}
    for stream in (
        "initialization",
        "training_sampler",
        "thinning",
        "validation_evaluation",
        "stochastic_optimizer_or_augmentation",
        "restart_interruption_point",
    ):
        expected = records["draw_id"].map(
            {draw: getattr(record, stream) for draw, record in schedule_by_draw.items()}
        )
        if not np.array_equal(
            records[stream].to_numpy(dtype=np.uint64), expected.to_numpy(dtype=np.uint64)
        ):
            raise IntegrityError(f"Dev35 refit {stream} seeds differ from the frozen schedule.")
    return records, pivot


def _support_eligibility(
    root: Path,
    artifacts: tuple[ArtifactRef, ...],
    *,
    kind: str,
    candidates: tuple[int, ...],
) -> tuple[bool, ...]:
    frames = [pd.read_parquet(_path(root, artifact)) for artifact in artifacts]
    if not frames:
        raise IntegrityError("Dev35 support audit artifacts are absent.")
    audit = pd.concat(frames, ignore_index=True)
    if tuple(audit.columns) != SUPPORT_V1_COLUMNS:
        raise IntegrityError("Dev35 support audit has an invalid schema.")
    selected = audit.loc[audit["candidate_kind"].astype(str) == kind].copy()
    required_dimensions = {
        "donor_checkpoint",
        "target",
        "guide",
        "control_vs_targeting",
        "sampler_stratum",
    }
    if set(selected["candidate_value"].astype(int)) != set(candidates):
        raise IntegrityError("Dev35 support audit omits a frozen candidate.")
    eligibility: list[bool] = []
    for candidate in candidates:
        rows = selected.loc[selected["candidate_value"].astype(int) == candidate]
        if (
            set(rows["dimension"].astype(str)) != required_dimensions
            or rows.duplicated(["dimension", "stratum_id"]).any()
            or (rows["cells"].astype(np.int64) < 0).any()
            or not np.isfinite(rows["weighted_effective_sample_size"].astype(float)).all()
            or not np.isfinite(rows["maximum_to_median_weight_ratio"].astype(float)).all()
        ):
            raise IntegrityError("Dev35 support rows are incomplete or nonfinite.")
        eligible = bool(
            rows["selection_eligible"].astype(bool).all()
            and not rows["zero_support"].astype(bool).any()
        )
        eligibility.append(eligible)
    return tuple(eligibility)


def _verify_common_support_receipt(
    root: Path,
    receipt: G00CCommonSupportMetricReceiptV1,
    *,
    freeze: G00CSelectionFreezeContractV1,
    schedule: G00CRefitSeedScheduleV1,
    refit_records: pd.DataFrame,
) -> None:
    if (
        receipt.selection_freeze_id != freeze.freeze_id
        or receipt.seed_schedule_id != schedule.schedule_id
        or receipt.seed_schedule_artifact != freeze.seed_schedule_artifact
        or receipt.prior.reference_feature_count != 4096
        or receipt.prior.selection_threshold != freeze.selection_margins.feature_equivalence_epsilon
    ):
        raise IntegrityError("Dev35 common-support receipt is cross-wired.")
    _path(root, receipt.validation_counts)
    residual_path = _path(root, receipt.residual_frequencies)
    sensitivity = json.loads(_path(root, receipt.prior_sensitivity).read_text())
    if sensitivity.get("status") != "pass" or sensitivity.get("selection_threshold") != 0.0001:
        raise IntegrityError("Dev35 common-support prior sensitivity did not pass.")
    try:
        with np.load(residual_path, allow_pickle=False) as payload:
            if set(payload.files) != {"256", "512", "1024", "2048"}:
                raise IntegrityError("Dev35 residual-frequency artifact has another prefix grid.")
            for key in payload.files:
                frequencies = np.asarray(payload[key], dtype=np.float64)
                if (
                    frequencies.ndim != 2
                    or frequencies.shape[0] != 3
                    or frequencies.shape[1] != 4096 - int(key)
                    or not np.isfinite(frequencies).all()
                    or np.any(frequencies <= 0.0)
                    or not np.allclose(frequencies.sum(axis=1), 1.0, atol=1e-12, rtol=1e-12)
                ):
                    raise IntegrityError("Dev35 residual frequencies are not strictly positive.")
    except IntegrityError:
        raise
    except Exception as exc:
        raise IntegrityError("Dev35 residual-frequency artifact cannot be read.") from exc
    totals = (
        refit_records.loc[refit_records["candidate_value"].astype(int) == 4096]
        .sort_values("draw_id")["validation_total_count"]
        .to_numpy(dtype=np.int64)
    )
    if _hash_int64(totals) != receipt.validation_total_count_hash:
        raise IntegrityError("Dev35 common-support count denominator hash differs.")


def _verify_feature_selection_v3(
    root: Path,
    *,
    freeze: G00CSelectionFreezeContractV1,
    bundle: G00CExecutionBundleV3 | G00CExecutionBundleV4 | G00CExecutionBundleV5,
    result: G00CFeatureSelectionResultV3,
    schedule: G00CRefitSeedScheduleV1,
    support_eligible_override: tuple[bool, ...] | None = None,
) -> tuple[tuple[str, ...], pd.DataFrame]:
    candidates = freeze.feature_ranking.candidate_feature_counts
    records, pivot = _load_refits_v3(
        root,
        result.refit_records,
        kind="feature_count",
        candidates=candidates,
        validation_row_hash=freeze.feature_ranking.validation_rows_hash,
        schedule=schedule,
    )
    if (
        result.fit_rows_hash != freeze.feature_ranking.fit_reference_rows_hash
        or result.validation_rows_hash != freeze.feature_ranking.validation_rows_hash
    ):
        raise IntegrityError("Dev35 feature result binds other fit or validation rows.")
    reference = pivot[4096].to_numpy(dtype=float)
    differences = np.abs(pivot.to_numpy(dtype=float) - reference[:, None])
    q95 = np.quantile(differences, 0.95, axis=0, method="linear")
    means = pivot.mean(axis=0).to_numpy(dtype=float)
    support = support_eligible_override
    if support is None:
        if not isinstance(bundle, G00CExecutionBundleV3):
            raise IntegrityError("Dev36 feature verification requires derived support.")
        support = _support_eligibility(
            root, (bundle.base_support_audit,), kind="feature_count", candidates=candidates
        )
    if len(support) != len(candidates):
        raise IntegrityError("Dev36 feature support does not cover the frozen grid.")
    curve = pd.read_parquet(_path(root, result.curve))
    if (
        tuple(curve.columns) != FEATURE_CURVE_V3_COLUMNS
        or tuple(curve["feature_count"].astype(int)) != candidates
        or not np.allclose(curve["mean_validation_nll"], means, atol=1e-12, rtol=0)
        or not np.allclose(
            curve["q95_absolute_paired_nll_difference_to_4096"], q95, atol=1e-12, rtol=0
        )
        or tuple(curve["support_eligible"].astype(bool)) != support
    ):
        raise IntegrityError("Dev35 feature curve is not derived from full refit evidence.")
    selected = feature_selection_decision_v3(
        candidates=candidates,
        paired_absolute_difference_q95=q95,
        support_eligible=support,
        epsilon=freeze.selection_margins.feature_equivalence_epsilon,
    )
    if result.selected_feature_count != selected:
        raise IntegrityError("Dev35 selected feature width is not the smallest supported prefix.")
    ordered = pd.read_parquet(_path(root, result.ordered_features))
    if tuple(ordered.columns) != ("rank", "canonical_index", "feature_id"):
        raise IntegrityError("Dev35 ordered-feature table has an invalid schema.")
    if (
        len(ordered) != 4096
        or not np.array_equal(ordered["rank"].to_numpy(dtype=int), np.arange(1, 4097))
        or ordered["feature_id"].duplicated().any()
        or ordered["canonical_index"].duplicated().any()
    ):
        raise IntegrityError("Dev35 ordered-feature table is not one unique 4,096-gene ranking.")
    selected_ids = tuple(ordered.iloc[:selected]["feature_id"].astype(str))
    selected_hash = hashlib.sha256(canonical_json_bytes(selected_ids)).hexdigest()
    if selected_hash != result.selected_feature_order_sha256:
        raise IntegrityError("Dev35 selected feature-order hash is not the true ordered prefix.")
    common = _read_model(
        root, result.common_support_metric_receipt, G00CCommonSupportMetricReceiptV1
    )
    if (
        result.common_support_metric_receipt != bundle.common_support_receipt
        or common.refit_records != result.refit_records
    ):
        raise IntegrityError("Dev35 common-support receipt binds other feature refits.")
    _verify_common_support_receipt(
        root, common, freeze=freeze, schedule=schedule, refit_records=records
    )
    return selected_ids, records


def _verify_sample_size_selection_v3(
    root: Path,
    *,
    freeze: G00CSelectionFreezeContractV1,
    authority: (
        G00CD1ExecutionAuthorityFreezeV1
        | G00CD1ExecutionAuthorityFreezeV2
        | G00CD1ExecutionAuthorityFreezeV3
    ),
    bundle: G00CExecutionBundleV3 | G00CExecutionBundleV4 | G00CExecutionBundleV5,
    feature_result: G00CFeatureSelectionResultV3,
    result: G00CSampleSizeSelectionResultV3,
    schedule: G00CRefitSeedScheduleV1,
    support_eligible_override: tuple[bool, ...] | None = None,
) -> np.ndarray[Any, Any] | None:
    if (
        result.parent_feature_selection_result_sha256 != bundle.feature_selection_result.sha256
        or result.selected_feature_count != feature_result.selected_feature_count
        or result.selected_feature_order_sha256 != feature_result.selected_feature_order_sha256
        or result.training_scale_row_order != authority.nested_training_row_order
    ):
        raise IntegrityError("Dev35 sample result does not bind its exact feature/order parents.")
    candidates = (
        freeze.base_cell_grid
        if result.grid_stage == "base"
        else (*freeze.base_cell_grid, *freeze.extension_additional_cell_grid)
    )
    records, pivot = _load_refits_v3(
        root,
        result.refit_records,
        kind="training_cells",
        candidates=candidates,
        validation_row_hash=freeze.feature_ranking.validation_rows_hash,
        schedule=schedule,
    )
    order = pd.read_parquet(_path(root, result.training_scale_row_order))
    if tuple(order.columns) != ("rank", "row_id"):
        raise IntegrityError("Dev35 training-scale order has an invalid schema.")
    ordered_rows = order["row_id"].to_numpy(dtype=np.int64)
    if len(ordered_rows) < candidates[-1] or not np.array_equal(
        order["rank"].to_numpy(dtype=np.int64), np.arange(1, len(order) + 1)
    ):
        raise IntegrityError("Dev35 training-scale order is not consecutively ranked.")
    fit_hashes = records.groupby("candidate_value", sort=True)["fit_row_hash"].agg(set)
    if any(
        fit_hashes.loc[candidate] != {_ordered_row_hash(ordered_rows[:candidate])}
        for candidate in candidates
    ):
        raise IntegrityError("Dev35 sample refits do not bind exact nested row prefixes.")
    reference = pivot[candidates[-1]].to_numpy(dtype=float)
    differences = np.abs(pivot.to_numpy(dtype=float) - reference[:, None])
    q95 = np.quantile(differences, 0.95, axis=0, method="linear")
    means = pivot.mean(axis=0).to_numpy(dtype=float)
    support = support_eligible_override
    if support is None:
        if not isinstance(bundle, G00CExecutionBundleV3):
            raise IntegrityError("Dev36 sample verification requires derived support.")
        support_artifacts: tuple[ArtifactRef, ...] = (bundle.base_support_audit,)
        if result.grid_stage == "extension":
            if bundle.extension_support_audit is None:
                raise IntegrityError("Dev35 extension result lacks two-million support evidence.")
            support_artifacts = (*support_artifacts, bundle.extension_support_audit)
        support = _support_eligibility(
            root, support_artifacts, kind="training_cells", candidates=candidates
        )
    if len(support) != len(candidates):
        raise IntegrityError("Dev36 sample support does not cover the frozen grid.")
    curve = pd.read_parquet(_path(root, result.curve))
    if (
        tuple(curve.columns) != SAMPLE_CURVE_V3_COLUMNS
        or tuple(curve["training_cells"].astype(int)) != candidates
        or not np.allclose(curve["mean_validation_nll"], means, atol=1e-12, rtol=0)
        or not np.allclose(
            curve["q95_absolute_paired_nll_difference_to_reference"], q95, atol=1e-12, rtol=0
        )
        or tuple(curve["support_eligible"].astype(bool)) != support
    ):
        raise IntegrityError("Dev35 sample curve is not derived from full refit evidence.")
    expected_status, expected_cells = sample_size_decision_v3(
        grid_stage=result.grid_stage,
        candidates=candidates,
        paired_absolute_difference_q95=q95,
        support_eligible=support,
        epsilon=freeze.selection_margins.cell_equivalence_epsilon,
    )
    if (
        result.selection_status != expected_status
        or result.selected_training_cells != expected_cells
    ):
        raise IntegrityError("Dev35 sample-size status is not artifact-derived.")
    if result.grid_stage == "extension":
        if bundle.extension_freeze is None:
            raise IntegrityError("Dev35 extension result lacks a pre-access extension authority.")
        if isinstance(bundle, G00CExecutionBundleV5):
            from .g00c_extension_v4 import verify_g00c_extension_freeze_v2

            if not isinstance(authority, G00CD1ExecutionAuthorityFreezeV3):
                raise IntegrityError("A Dev37 execution bundle requires a Dev37 authority.")

            extension_v2 = _read_model(
                root, bundle.extension_freeze, G00CSampleSizeExtensionFreezeV2
            )
            if (
                result.base_grid_extension_required_receipt
                != extension_v2.base_extension_required_receipt
            ):
                raise IntegrityError("Dev37 extension result binds another base stop receipt.")
            verify_g00c_extension_freeze_v2(
                root,
                extension_v2,
                authority=authority,
                feature_result=feature_result,
            )
        else:
            extension = _read_model(root, bundle.extension_freeze, G00CSampleSizeExtensionFreezeV1)
            if (
                result.base_grid_extension_required_receipt
                != extension.base_extension_required_receipt
            ):
                raise IntegrityError("Dev35 extension result binds another base stop receipt.")
            verify_g00c_extension_freeze_v1(
                root,
                extension,
                base_freeze=freeze,
                base_feature_result=feature_result,
                base_feature_artifact=bundle.feature_selection_result,
            )
    if expected_cells is None:
        return None
    if result.selected_training_rows is None:
        raise IntegrityError("Dev35 selected sample result lacks exact selected rows.")
    selected = pd.read_parquet(_path(root, result.selected_training_rows))
    if tuple(selected.columns) != ("row_id",):
        raise IntegrityError("Dev35 selected-row artifact has an invalid schema.")
    selected_rows = selected["row_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(selected_rows, ordered_rows[:expected_cells]):
        raise IntegrityError("Dev35 selected rows are not the true frozen prefix.")
    return selected_rows


def verify_g00c_execution_v3(
    root: Path,
    bundle: G00CExecutionBundleV3,
    *,
    materialized_pass_verifier: MaterializedPassVerifier | None = None,
) -> VerifiedG00CExecutionV3:
    """Recompute the complete Dev35 authority chain and terminal status."""

    authority = _read_model(root, bundle.execution_authority, G00CD1ExecutionAuthorityFreezeV1)
    freeze = verify_g00c_d1_freeze_v1(root, authority)
    if (
        authority.authority_id != bundle.execution_authority_id
        or bundle.selection_freeze != authority.selection_freeze
        or bundle.selection_freeze_id != freeze.freeze_id
        or bundle.row_roles != authority.row_role_freeze
        or bundle.seed_schedule != authority.seed_schedule
        or bundle.base_support_audit_contract != authority.base_support_audit_contract
    ):
        raise IntegrityError("Dev35 execution bundle is cross-wired to another authority.")
    base_support_contract = _read_model(
        root, bundle.base_support_audit_contract, G00CSupportAuditContractV2
    )
    if base_support_contract.stage != "base":
        raise IntegrityError("Dev35 execution bundle binds another base support contract.")
    schedule = _read_model(root, bundle.seed_schedule, G00CRefitSeedScheduleV1)
    if schedule.schedule_id != bundle.seed_schedule_id:
        raise IntegrityError("Dev35 execution bundle seed schedule identity differs.")
    feature = _read_model(root, bundle.feature_selection_result, G00CFeatureSelectionResultV3)
    sample = _read_model(root, bundle.sample_size_selection_result, G00CSampleSizeSelectionResultV3)
    if (
        feature.result_id != bundle.feature_selection_result_id
        or sample.result_id != bundle.sample_size_selection_result_id
    ):
        raise IntegrityError("Dev35 result artifact identities differ from the bundle.")
    selected_features, _ = _verify_feature_selection_v3(
        root, freeze=freeze, bundle=bundle, result=feature, schedule=schedule
    )
    selected_rows = _verify_sample_size_selection_v3(
        root,
        freeze=freeze,
        authority=authority,
        bundle=bundle,
        feature_result=feature,
        result=sample,
        schedule=schedule,
    )
    if bundle.extension_freeze is not None:
        extension = _read_model(root, bundle.extension_freeze, G00CSampleSizeExtensionFreezeV1)
        if bundle.extension_support_audit_contract != extension.extension_support_audit_contract:
            raise IntegrityError("Dev35 execution binds another extension support contract.")
    publication = json.loads(_path(root, bundle.publication_manifest).read_text())
    if publication.get("terminal_status") != bundle.terminal_status:
        raise IntegrityError("Dev35 publication status differs from the execution bundle.")
    expected_status = "pass" if sample.selection_status == "selected" else sample.selection_status
    if bundle.terminal_status != "failed_integrity" and bundle.terminal_status != expected_status:
        raise IntegrityError("Dev35 bundle terminal status differs from recomputed selection.")
    _path(root, bundle.sampler_evidence)
    if bundle.terminal_status == "pass":
        if selected_rows is None or materialized_pass_verifier is None:
            raise IntegrityError(
                "Dev35 pass requires independent compact materialization verification."
            )
        expected_verifier = _implementation_hash(authority, "execution_verifier")
        if materialized_pass_verifier.implementation_sha256 != expected_verifier:
            raise IntegrityError("Dev35 pass verifier differs from the frozen implementation.")
        materialized_pass_verifier(root, bundle, selected_features, selected_rows)
    elif bundle.terminal_status == "failed_integrity":
        failure = json.loads(_path(root, bundle.failure_receipt).read_text())  # type: ignore[arg-type]
        if failure.get("status") != "failed_integrity":
            raise IntegrityError("Dev35 integrity-failure receipt has another status.")
    common = _read_model(root, bundle.common_support_receipt, G00CCommonSupportMetricReceiptV1)
    authority_artifacts = (
        authority.row_role_freeze,
        authority.nested_training_row_order,
        authority.feature_reference_rows,
        authority.base_support_audit_contract,
        authority.implementation.wheel,
        authority.implementation.normalized_sdist,
        authority.implementation.environment_lock,
        *(binding.artifact for binding in authority.implementation.implementations),
        *(binding.artifact for binding in freeze.parent_bindings),
    )
    verified = {
        artifact.sha256
        for artifact in (
            bundle.execution_authority,
            bundle.selection_freeze,
            bundle.seed_schedule,
            bundle.row_roles,
            bundle.feature_selection_result,
            bundle.sample_size_selection_result,
            bundle.common_support_receipt,
            bundle.base_support_audit_contract,
            bundle.base_support_audit,
            bundle.sampler_evidence,
            bundle.publication_manifest,
            feature.curve,
            feature.refit_records,
            feature.ordered_features,
            feature.common_support_metric_receipt,
            sample.curve,
            sample.refit_records,
            sample.training_scale_row_order,
            common.validation_counts,
            common.residual_frequencies,
            common.prior_sensitivity,
            *authority_artifacts,
            *(
                artifact
                for artifact in (
                    sample.base_grid_extension_required_receipt,
                    sample.selected_training_rows,
                    bundle.extension_freeze,
                    bundle.extension_support_audit_contract,
                    bundle.extension_support_audit,
                    bundle.compact_payload,
                    bundle.compact_verification_receipt,
                    bundle.reload_receipt,
                    bundle.failure_receipt,
                )
                if artifact is not None
            ),
        )
    }
    verified.add(freeze.dev33_canary.authority_archive_sha256)
    if bundle.extension_freeze is not None:
        extension = _read_model(root, bundle.extension_freeze, G00CSampleSizeExtensionFreezeV1)
        verified.update(
            artifact.sha256
            for artifact in (
                extension.base_execution_bundle,
                extension.base_extension_required_receipt,
                extension.base_feature_selection_result,
                extension.selected_feature_surface,
                extension.seed_schedule_artifact,
                extension.base_support_audit,
                extension.extension_support_audit_contract,
            )
        )
    return VerifiedG00CExecutionV3(
        terminal_status=bundle.terminal_status,
        execution_bundle_id=bundle.bundle_id,
        execution_authority_id=authority.authority_id,
        selection_freeze_id=freeze.freeze_id,
        feature_selection_result_id=feature.result_id,
        sample_size_selection_result_id=sample.result_id,
        seed_schedule_id=schedule.schedule_id,
        common_support_receipt_sha256=bundle.common_support_receipt.sha256,
        support_audit_sha256s=tuple(
            artifact.sha256
            for artifact in (bundle.base_support_audit, bundle.extension_support_audit)
            if artifact is not None
        ),
        grid_stage=sample.grid_stage,
        verified_artifact_sha256s=tuple(sorted(verified)),
    )


def build_g00c_decision_receipt_v3(
    verified: VerifiedG00CExecutionV3,
    *,
    verifier_implementation_sha256: str,
) -> G00CDecisionReceiptV3:
    """Build a receipt only from values returned by the artifact-reading verifier."""

    payload: dict[str, Any] = {
        "receipt_id": "0" * 64,
        "execution_bundle_id": verified.execution_bundle_id,
        "execution_authority_id": verified.execution_authority_id,
        "selection_freeze_id": verified.selection_freeze_id,
        "feature_selection_result_id": verified.feature_selection_result_id,
        "sample_size_selection_result_id": verified.sample_size_selection_result_id,
        "seed_schedule_id": verified.seed_schedule_id,
        "common_support_receipt_sha256": verified.common_support_receipt_sha256,
        "support_audit_sha256s": verified.support_audit_sha256s,
        "verifier_implementation_sha256": verifier_implementation_sha256,
        "verified_artifact_sha256s": verified.verified_artifact_sha256s,
        "grid_stage": verified.grid_stage,
        "terminal_status": verified.terminal_status,
        "may_parent_g00d": verified.terminal_status == "pass",
    }
    provisional = G00CDecisionReceiptV3.model_construct(**payload)
    payload["receipt_id"] = provisional.identity(id_field="receipt_id")
    return G00CDecisionReceiptV3.model_validate(payload)


def verify_g00c_decision_v3(
    root: Path,
    bundle: G00CExecutionBundleV3,
    receipt: G00CDecisionReceiptV3,
    *,
    materialized_pass_verifier: MaterializedPassVerifier | None = None,
) -> None:
    """Reject any V3 receipt whose terminal decision is not artifact-derived."""

    verified = verify_g00c_execution_v3(
        root, bundle, materialized_pass_verifier=materialized_pass_verifier
    )
    authority = _read_model(root, bundle.execution_authority, G00CD1ExecutionAuthorityFreezeV1)
    expected_implementation = _implementation_hash(authority, "decision_verifier")
    expected = build_g00c_decision_receipt_v3(
        verified, verifier_implementation_sha256=expected_implementation
    )
    if receipt != expected:
        raise IntegrityError("Dev35 decision receipt differs from recomputed V3 evidence.")
