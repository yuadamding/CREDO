"""Claim-support diagnostics for CREDO setting search.

These helpers keep final-selection evidence explicit: convergence across
particle/step fidelities, empirical null calibration, baseline-export manifests,
and biology-axis gates are recorded as structured payloads rather than prose.
"""
from __future__ import annotations

import hashlib
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Mapping, Optional


BASELINE_KINDS: tuple[str, ...] = (
    "credo_endpoint_proxy",
    "credo_weak_form",
    "moscot_time",
    "wot_temporal_ot",
    "wfr_mfm_unbalanced_transport",
    "scdiffeq_sde",
)
BASELINE_STATUSES: tuple[str, ...] = ("planned", "available", "skipped", "failed")
BASELINE_PROVENANCE_FIELDS: tuple[str, ...] = (
    "baseline_version",
    "baseline_commit_sha",
    "export_schema_version",
    "artifact_sha256",
    "input_measure_hash",
    "latent_space_hash",
    "mass_table_hash",
    "split_manifest_sha256",
)
BASELINE_CODE_IDENTITY_KINDS: tuple[str, ...] = (
    "git_commit",
    "package_lock",
    "container_digest",
    "published_release",
)


@dataclass(frozen=True)
class FidelityRecord:
    """One refit/evaluation of the same setting at a particle/step fidelity."""

    n_particles: int
    n_steps: int
    delta_log_mass_by_id: Mapping[str, float] = field(default_factory=dict)
    endpoint_metric_by_id: Mapping[str, float] = field(default_factory=dict)
    biology_axis_score_by_axis: Mapping[str, float] = field(default_factory=dict)
    top_hit_scores: Mapping[str, float] = field(default_factory=dict)
    ess_min: float = math.nan

    def __post_init__(self) -> None:
        if int(self.n_particles) <= 0:
            raise ValueError("n_particles must be positive.")
        if int(self.n_steps) <= 0:
            raise ValueError("n_steps must be positive.")


@dataclass(frozen=True)
class ConvergenceThresholds:
    rank_correlation_min: float = 0.90
    sign_stability_min: float = 0.90
    endpoint_drift_median_max: float = math.inf
    top_hit_jaccard_min: float = 0.90
    ess_floor: float = 0.10
    top_k: int = 20


@dataclass(frozen=True)
class BiologyAxisSpec:
    name: str
    markers: tuple[str, ...]
    expected_direction: Optional[str] = None
    organism: str = "unspecified"
    gene_symbol_namespace: str = "unspecified"
    module_source: Optional[str] = None
    min_coverage: float = 0.0
    min_markers_scored: int = 1
    axis_kind: str = "expression_module"
    null_alternative: Literal["greater", "less", "two_sided"] = "two_sided"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Biology axis name must not be empty.")
        if not self.markers:
            raise ValueError(f"Biology axis {self.name!r} must include at least one marker.")
        if self.expected_direction not in {None, "positive", "negative", "either"}:
            raise ValueError("expected_direction must be positive, negative, either, or None.")
        if self.axis_kind not in {"expression_module", "perturbation_anchor", "metadata_artifact"}:
            raise ValueError("axis_kind must be expression_module, perturbation_anchor, or metadata_artifact.")
        if self.null_alternative not in {"greater", "less", "two_sided"}:
            raise ValueError("null_alternative must be greater, less, or two_sided.")
        if not 0.0 <= float(self.min_coverage) <= 1.0:
            raise ValueError("min_coverage must be in [0, 1].")
        if int(self.min_markers_scored) < 1:
            raise ValueError("min_markers_scored must be positive.")


@dataclass(frozen=True)
class HomologMarkerStatus:
    """Original-marker-level homolog mapping and scoring status."""

    original_marker: str
    mapped_symbols: tuple[str, ...] = ()
    scored_symbols: tuple[str, ...] = ()
    mapping_status: Literal["unique_scored", "unique_unscored", "ambiguous", "unmapped"] = "unmapped"

    def __post_init__(self) -> None:
        if not self.original_marker:
            raise ValueError("original_marker must not be empty.")
        if self.mapping_status not in {"unique_scored", "unique_unscored", "ambiguous", "unmapped"}:
            raise ValueError(f"Unknown homolog mapping_status {self.mapping_status!r}.")
        object.__setattr__(self, "mapped_symbols", _symbol_tuple(self.mapped_symbols))
        object.__setattr__(self, "scored_symbols", _symbol_tuple(self.scored_symbols))


DEFAULT_BIOLOGY_AXES: tuple[BiologyAxisSpec, ...] = (
    BiologyAxisSpec(
        name="renz_expansion_anchor_perturbations",
        markers=("Notch1", "Notch2", "Fat1", "Trp53", "Fgf3"),
        expected_direction="positive",
        axis_kind="perturbation_anchor",
        organism="mouse",
        gene_symbol_namespace="MGI",
        module_source="Renz_2024_P4_P60_field_expansion",
        min_coverage=0.7,
        null_alternative="greater",
    ),
    BiologyAxisSpec(
        name="renz_tnf_ap1_nfkb_expression_module",
        markers=(
            "Jun",
            "Fos",
            "Junb",
            "Jund",
            "Ccn1",
            "Socs3",
            "Nfkbiz",
            "Atf3",
            "Dusp1",
            "Ier2",
        ),
        expected_direction="positive",
        axis_kind="expression_module",
        organism="mouse",
        gene_symbol_namespace="MGI",
        module_source="Renz_2024_TNF_AP1_NFkB_expression_module",
        min_coverage=0.7,
        null_alternative="greater",
    ),
    BiologyAxisSpec(
        name="cis_like_epithelial",
        markers=("TP63", "ATP1B3", "KRT5", "KRT14", "KRT17", "SOX2", "EPCAM"),
        expected_direction="either",
        axis_kind="expression_module",
        organism="human",
        gene_symbol_namespace="HGNC",
        module_source="Choi_2023_CIS_like_LP",
        min_coverage=0.7,
        null_alternative="two_sided",
    ),
    BiologyAxisSpec(
        name="pemt_tsk",
        markers=("LGALS7B", "VIM", "SNAI2", "LAMC2", "ITGA5"),
        expected_direction="either",
        axis_kind="expression_module",
        organism="human",
        gene_symbol_namespace="HGNC",
        module_source="Choi_2023_CC1_Punovuori_2024_pEMT",
        min_coverage=0.7,
        null_alternative="two_sided",
    ),
    BiologyAxisSpec(
        name="caf_ecm",
        markers=("COL1A1", "COL1A2", "FN1", "POSTN", "THBS2"),
        expected_direction="either",
        axis_kind="expression_module",
        organism="human",
        gene_symbol_namespace="HGNC",
        module_source="Punovuori_2024_CAF_ECM",
        min_coverage=0.7,
        null_alternative="two_sided",
    ),
    BiologyAxisSpec(
        name="myeloid",
        markers=("LYZ", "S100A8", "S100A9", "FCGR3A"),
        expected_direction="either",
        axis_kind="expression_module",
        organism="human",
        gene_symbol_namespace="HGNC",
        module_source="HNSCC_inflammatory_myeloid",
        min_coverage=0.7,
        null_alternative="two_sided",
    ),
    BiologyAxisSpec(
        name="guide_artifact",
        markers=("guide_concordance",),
        expected_direction="negative",
        axis_kind="metadata_artifact",
        organism="metadata",
        gene_symbol_namespace="metric",
        module_source="CREDO_guide_artifact_axis",
        null_alternative="less",
    ),
    BiologyAxisSpec(
        name="large_gene_artifact",
        markers=("gene_size",),
        expected_direction="negative",
        axis_kind="metadata_artifact",
        organism="metadata",
        gene_symbol_namespace="metric",
        module_source="CREDO_large_gene_artifact_axis",
        null_alternative="less",
    ),
)


def claim_grade_convergence_thresholds(
    *,
    endpoint_drift_median_max: float,
    rank_correlation_min: float = 0.90,
    sign_stability_min: float = 0.90,
    top_hit_jaccard_min: float = 0.90,
    ess_floor: float = 0.10,
    top_k: int = 20,
) -> ConvergenceThresholds:
    """Build claim-grade convergence gates with a finite endpoint-drift ceiling."""
    if not math.isfinite(float(endpoint_drift_median_max)) or float(endpoint_drift_median_max) < 0:
        raise ValueError(
            "endpoint_drift_median_max must be a finite non-negative threshold for claim-grade convergence."
        )
    return ConvergenceThresholds(
        rank_correlation_min=rank_correlation_min,
        sign_stability_min=sign_stability_min,
        endpoint_drift_median_max=float(endpoint_drift_median_max),
        top_hit_jaccard_min=top_hit_jaccard_min,
        ess_floor=ess_floor,
        top_k=top_k,
    )


def estimate_convergence_thresholds_from_pilot(
    *,
    between_perturbation_endpoint_distances: Iterable[float],
    within_perturbation_fold_endpoint_distances: Iterable[float],
    rank_correlation_min: float = 0.90,
    sign_stability_min: float = 0.90,
    top_hit_jaccard_min: float = 0.90,
    ess_floor: float = 0.10,
    top_k: int = 20,
) -> ConvergenceThresholds:
    """Calibrate endpoint-drift convergence gates from pilot geometry scales."""
    between = _finite_nonnegative_values(between_perturbation_endpoint_distances)
    within = _finite_nonnegative_values(within_perturbation_fold_endpoint_distances)
    if not between or not within:
        raise ValueError(
            "pilot calibration requires finite non-negative between and within distances."
        )
    endpoint_drift_median_max = min(0.10 * statistics.median(between), 0.25 * statistics.median(within))
    return claim_grade_convergence_thresholds(
        endpoint_drift_median_max=endpoint_drift_median_max,
        rank_correlation_min=rank_correlation_min,
        sign_stability_min=sign_stability_min,
        top_hit_jaccard_min=top_hit_jaccard_min,
        ess_floor=ess_floor,
        top_k=top_k,
    )


def particle_step_convergence_diagnostics(
    records: Iterable[FidelityRecord],
    *,
    thresholds: ConvergenceThresholds = ConvergenceThresholds(),
) -> dict[str, object]:
    """Summarize whether a setting is stable across particle/step fidelities."""
    ordered = sorted(records, key=lambda item: (item.n_particles, item.n_steps))
    if len(ordered) < 2:
        return {
            "n_fidelities": len(ordered),
            "passes": False,
            "failed_gates": "insufficient_fidelities",
        }

    reference = ordered[-1]
    comparisons = [(candidate, reference) for candidate in ordered[:-1]]
    rank_corrs = [
        _spearman_mapping(candidate.delta_log_mass_by_id, reference.delta_log_mass_by_id)
        for candidate, reference in comparisons
    ]
    sign_stabilities = [
        _sign_stability(candidate.biology_axis_score_by_axis, reference.biology_axis_score_by_axis)
        for candidate, reference in comparisons
    ]
    endpoint_drifts = [
        _median_abs_diff(candidate.endpoint_metric_by_id, reference.endpoint_metric_by_id)
        for candidate, reference in comparisons
    ]
    top_jaccards = [
        _top_hit_jaccard(candidate.top_hit_scores, reference.top_hit_scores, thresholds.top_k)
        for candidate, reference in comparisons
    ]
    ess_min = min(_finite_or_default(item.ess_min, -math.inf) for item in ordered)

    summary = {
        "n_fidelities": len(ordered),
        "reference_n_particles": reference.n_particles,
        "reference_n_steps": reference.n_steps,
        "rank_correlation_delta_log_mass_min": _min_finite(rank_corrs),
        "sign_stability_biology_axes_min": _min_finite(sign_stabilities),
        "endpoint_drift_median_max": _max_finite(endpoint_drifts),
        "top_hit_jaccard_min": _min_finite(top_jaccards),
        "ess_min": ess_min,
    }
    failed = []
    if _lt(summary["rank_correlation_delta_log_mass_min"], thresholds.rank_correlation_min):
        failed.append("delta_log_mass_rank_correlation")
    if _lt(summary["sign_stability_biology_axes_min"], thresholds.sign_stability_min):
        failed.append("biology_axis_sign_stability")
    if _gt(summary["endpoint_drift_median_max"], thresholds.endpoint_drift_median_max):
        failed.append("endpoint_drift")
    if _lt(summary["top_hit_jaccard_min"], thresholds.top_hit_jaccard_min):
        failed.append("top_hit_jaccard")
    if _lt(summary["ess_min"], thresholds.ess_floor):
        failed.append("ess_floor")
    summary["passes"] = not failed
    summary["failed_gates"] = ",".join(failed)
    return summary


def summarize_null_distribution(
    null_values: Iterable[float],
    observed_gap: float,
    *,
    alternative: str = "greater",
) -> dict[str, float]:
    """Return null mean/p95/z/empirical-p for one observed diagnostic gap."""
    values = [_finite_or_nan(value) for value in null_values]
    values = [value for value in values if math.isfinite(value)]
    observed = float(observed_gap)
    if alternative not in {"greater", "less", "two_sided"}:
        raise ValueError("alternative must be 'greater', 'less', or 'two_sided'.")
    if not values or not math.isfinite(observed):
        return {
            "null_gap_mean": math.nan,
            "null_gap_p05": math.nan,
            "null_gap_p95": math.nan,
            "null_abs_p95": math.nan,
            "null_tail_quantile": math.nan,
            "null_tail_direction": alternative,
            "null_gap_z": math.nan,
            "empirical_p": math.nan,
            "fdr": math.nan,
        }
    mean = sum(values) / len(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    if alternative == "greater":
        extreme = sum(1 for value in values if value >= observed)
    elif alternative == "less":
        extreme = sum(1 for value in values if value <= observed)
    else:
        extreme = sum(1 for value in values if abs(value - mean) >= abs(observed - mean))
    empirical_p = (extreme + 1.0) / (len(values) + 1.0)
    if sd > 0:
        null_gap_z = (observed - mean) / sd
    elif observed == mean:
        null_gap_z = 0.0
    else:
        null_gap_z = math.copysign(math.inf, observed - mean)
    return {
        "null_gap_mean": mean,
        "null_gap_p05": _quantile(values, 0.05),
        "null_gap_p95": _quantile(values, 0.95),
        "null_abs_p95": _quantile([abs(value - mean) for value in values], 0.95),
        "null_tail_quantile": _tail_quantile(values, mean=mean, alternative=alternative),
        "null_tail_direction": alternative,
        "null_gap_z": null_gap_z,
        "empirical_p": empirical_p,
        "fdr": empirical_p,
    }


def summarize_null_suite(
    observed_gaps: Mapping[str, float],
    nulls_by_name: Mapping[str, Iterable[float]],
    *,
    alternative: str = "greater",
    alternatives: Optional[Mapping[str, str]] = None,
    axes: Optional[Iterable[BiologyAxisSpec]] = None,
) -> dict[str, dict[str, float]]:
    """Summarize multiple nulls and apply Benjamini-Hochberg FDR correction."""
    axis_alternatives = {axis.name: axis.null_alternative for axis in axes or ()}
    alternatives = alternatives or {}
    summaries = {
        name: summarize_null_distribution(
            nulls_by_name.get(name, ()),
            observed,
            alternative=alternatives.get(name, axis_alternatives.get(name, alternative)),
        )
        for name, observed in observed_gaps.items()
    }
    p_values = {name: summary["empirical_p"] for name, summary in summaries.items()}
    for name, fdr in _benjamini_hochberg(p_values).items():
        summaries[name]["fdr"] = fdr
    return summaries


def baseline_export_record(
    baseline_kind: str,
    *,
    artifact_path: str | Path | None = None,
    artifact_sha256: str | None = None,
    metrics: Optional[Mapping[str, float]] = None,
    status: str = "planned",
    notes: str | None = None,
    baseline_version: str | None = None,
    baseline_commit_sha: str | None = None,
    baseline_code_identity_kind: str | None = None,
    baseline_code_identity: str | None = None,
    baseline_package_lock_sha256: str | None = None,
    baseline_container_digest: str | None = None,
    export_schema_version: str | None = None,
    input_measure_hash: str | None = None,
    latent_space_hash: str | None = None,
    mass_table_hash: str | None = None,
    split_manifest_sha256: str | None = None,
) -> dict[str, object]:
    """Build one stable baseline-export manifest row."""
    if baseline_kind not in BASELINE_KINDS:
        raise ValueError(f"Unknown baseline_kind {baseline_kind!r}; expected one of {BASELINE_KINDS}.")
    if status not in BASELINE_STATUSES:
        raise ValueError(f"Unknown baseline status {status!r}; expected one of {BASELINE_STATUSES}.")
    if baseline_code_identity_kind is not None and baseline_code_identity_kind not in BASELINE_CODE_IDENTITY_KINDS:
        raise ValueError(
            f"Unknown baseline_code_identity_kind {baseline_code_identity_kind!r}; "
            f"expected one of {BASELINE_CODE_IDENTITY_KINDS}."
        )
    record: dict[str, object] = {
        "baseline_kind": baseline_kind,
        "status": status,
        "artifact_path": None if artifact_path is None else str(artifact_path),
        "artifact_sha256": artifact_sha256,
        "notes": notes,
        "baseline_version": baseline_version,
        "baseline_commit_sha": baseline_commit_sha,
        "baseline_code_identity_kind": baseline_code_identity_kind,
        "baseline_code_identity": baseline_code_identity,
        "baseline_package_lock_sha256": baseline_package_lock_sha256,
        "baseline_container_digest": baseline_container_digest,
        "export_schema_version": export_schema_version,
        "input_measure_hash": input_measure_hash,
        "latent_space_hash": latent_space_hash,
        "mass_table_hash": mass_table_hash,
        "split_manifest_sha256": split_manifest_sha256,
    }
    for key, value in dict(metrics or {}).items():
        record[f"metric.{key}"] = float(value)
    return record


def baseline_export_record_from_file(
    baseline_kind: str,
    artifact_path: str | Path,
    *,
    artifact_sha256: str | None = None,
    metrics: Optional[Mapping[str, float]] = None,
    status: str = "available",
    notes: str | None = None,
    baseline_version: str | None = None,
    baseline_commit_sha: str | None = None,
    baseline_code_identity_kind: str | None = None,
    baseline_code_identity: str | None = None,
    baseline_package_lock_sha256: str | None = None,
    baseline_container_digest: str | None = None,
    export_schema_version: str | None = None,
    input_measure_hash: str | None = None,
    latent_space_hash: str | None = None,
    mass_table_hash: str | None = None,
    split_manifest_sha256: str | None = None,
) -> dict[str, object]:
    """Build a baseline record after hashing a local artifact."""
    path = Path(artifact_path)
    computed_sha256: str | None = None
    if status == "available" or path.exists():
        if not path.is_file():
            raise FileNotFoundError(f"Baseline artifact does not exist or is not a file: {path}")
        computed_sha256 = _sha256_file(path)
        if artifact_sha256 is not None and artifact_sha256 != computed_sha256:
            raise ValueError(
                f"artifact_sha256 mismatch for {path}: expected {artifact_sha256}, got {computed_sha256}."
            )
    return baseline_export_record(
        baseline_kind,
        artifact_path=path,
        artifact_sha256=computed_sha256 or artifact_sha256,
        metrics=metrics,
        status=status,
        notes=notes,
        baseline_version=baseline_version,
        baseline_commit_sha=baseline_commit_sha,
        baseline_code_identity_kind=baseline_code_identity_kind,
        baseline_code_identity=baseline_code_identity,
        baseline_package_lock_sha256=baseline_package_lock_sha256,
        baseline_container_digest=baseline_container_digest,
        export_schema_version=export_schema_version,
        input_measure_hash=input_measure_hash,
        latent_space_hash=latent_space_hash,
        mass_table_hash=mass_table_hash,
        split_manifest_sha256=split_manifest_sha256,
    )


def baseline_export_manifest(
    records: Iterable[Mapping[str, object]],
    *,
    required: Iterable[str] = BASELINE_KINDS,
    require_provenance: bool = True,
) -> dict[str, object]:
    """Summarize which method-baseline exports are available for a claim."""
    rows = [dict(record) for record in records]
    available = {
        str(row.get("baseline_kind"))
        for row in rows
        if row.get("status") == "available" and row.get("artifact_path")
    }
    required_set = set(required)
    missing = sorted(required_set - available)
    provenance_missing = []
    if require_provenance:
        for row in rows:
            baseline = str(row.get("baseline_kind"))
            if baseline not in required_set or row.get("status") != "available":
                continue
            if _blank(row.get("artifact_path")):
                provenance_missing.append(f"{baseline}.artifact_path")
            for field in BASELINE_PROVENANCE_FIELDS:
                if field == "baseline_commit_sha":
                    if not _baseline_code_identity_ready(row):
                        provenance_missing.append(f"{baseline}.baseline_commit_sha")
                    continue
                if _blank(row.get(field)):
                    provenance_missing.append(f"{baseline}.{field}")
    return {
        "records": rows,
        "required_missing": missing,
        "provenance_missing": sorted(provenance_missing),
        "complete": not missing and not provenance_missing,
    }


def evaluate_biology_axis_gates(
    axis_scores: Mapping[str, float],
    *,
    null_summaries: Optional[Mapping[str, Mapping[str, float]]] = None,
    coverage_by_axis: Optional[Mapping[str, float]] = None,
    dataset_organism: str | None = None,
    homolog_mapped_axes: Optional[Iterable[str]] = None,
    homolog_map_name: str | None = None,
    homolog_map_version: str | None = None,
    homolog_map_sha256: str | None = None,
    homolog_marker_counts: Optional[Mapping[str, int]] = None,
    homolog_marker_status_by_axis: Optional[
        Mapping[str, Iterable[HomologMarkerStatus | Mapping[str, object]]]
    ] = None,
    scored_coverage_by_axis: Optional[Mapping[str, float]] = None,
    unmapped_markers_by_axis: Optional[Mapping[str, Iterable[str]]] = None,
    ambiguous_markers_by_axis: Optional[Mapping[str, Iterable[str]]] = None,
    axes: Iterable[BiologyAxisSpec] = DEFAULT_BIOLOGY_AXES,
    score_abs_min: float = 0.0,
    empirical_p_max: float = 0.05,
    fdr_max: float = 0.10,
) -> dict[str, dict[str, object]]:
    """Gate biology support separately for each configured axis."""
    null_summaries = null_summaries or {}
    coverage_by_axis = coverage_by_axis or {}
    scored_coverage_by_axis = scored_coverage_by_axis or {}
    unmapped_markers_by_axis = unmapped_markers_by_axis or {}
    ambiguous_markers_by_axis = ambiguous_markers_by_axis or {}
    homolog_marker_counts = homolog_marker_counts or {}
    homolog_marker_status_by_axis = homolog_marker_status_by_axis or {}
    homolog_mapped = set(homolog_mapped_axes or ())
    out: dict[str, dict[str, object]] = {}
    for axis in axes:
        score = _finite_or_nan(axis_scores.get(axis.name))
        null_summary = null_summaries.get(axis.name, {})
        empirical_p = _finite_or_nan(null_summary.get("empirical_p"))
        fdr = _finite_or_nan(null_summary.get("fdr"))
        coverage = _finite_or_nan(coverage_by_axis.get(axis.name))
        scored_coverage = _finite_or_nan(scored_coverage_by_axis.get(axis.name, coverage))
        unmapped_markers = tuple(unmapped_markers_by_axis.get(axis.name, ()))
        ambiguous_markers = tuple(ambiguous_markers_by_axis.get(axis.name, ()))
        unmapped_set = {str(marker) for marker in unmapped_markers}
        ambiguous_set = {str(marker) for marker in ambiguous_markers}
        raw_statuses = homolog_marker_status_by_axis.get(axis.name)
        if raw_statuses is None:
            marker_counts = _homolog_marker_counts_from_aggregates(
                axis,
                n_target_symbols_after_mapping=int(homolog_marker_counts.get(axis.name, len(axis.markers))),
                unmapped_set=unmapped_set,
                ambiguous_set=ambiguous_set,
            )
        else:
            marker_counts = _homolog_marker_counts_from_statuses(axis, raw_statuses)
        n_original_markers = int(marker_counts["n_original_markers"])
        n_original_markers_after_mapping = int(marker_counts["n_original_markers_after_mapping"])
        n_original_markers_unique_scored = int(marker_counts["n_original_markers_unique_scored"])
        n_original_markers_ambiguous = int(marker_counts["n_original_markers_ambiguous"])
        n_original_markers_unmapped = int(marker_counts["n_original_markers_unmapped"])
        n_original_markers_mapped_unique = int(marker_counts["n_original_markers_mapped_unique"])
        n_original_markers_scored_ambiguous = int(marker_counts["n_original_markers_scored_ambiguous"])
        n_original_markers_scored_including_ambiguous = int(
            marker_counts["n_original_markers_scored_including_ambiguous"]
        )
        n_target_symbols_after_mapping = int(marker_counts["n_target_symbols_after_mapping"])
        n_target_symbols_scored = int(marker_counts["n_target_symbols_scored"])
        coverage_scored_unique_computed = _coverage_fraction(
            n_original_markers_unique_scored,
            n_original_markers,
        )
        coverage_scored_including_ambiguous_computed = _coverage_fraction(
            n_original_markers_scored_including_ambiguous,
            n_original_markers,
        )
        ambiguous_marker_fraction = _coverage_fraction(n_original_markers_ambiguous, n_original_markers)
        if axis.name in homolog_mapped and math.isfinite(scored_coverage):
            scored_coverage_for_gate = min(scored_coverage, coverage_scored_unique_computed)
        elif axis.name in homolog_mapped:
            scored_coverage_for_gate = coverage_scored_unique_computed
        else:
            scored_coverage_for_gate = scored_coverage
        failed = []
        if not math.isfinite(score):
            failed.append("missing_axis_score")
        elif abs(score) < score_abs_min:
            failed.append("axis_score_below_floor")
        if axis.expected_direction == "positive" and math.isfinite(score) and score <= 0:
            failed.append("wrong_direction")
        if axis.expected_direction == "negative" and math.isfinite(score) and score >= 0:
            failed.append("wrong_direction")
        if math.isfinite(empirical_p) and empirical_p > empirical_p_max:
            failed.append("empirical_p")
        if math.isfinite(fdr) and fdr > fdr_max:
            failed.append("fdr")
        if axis.min_coverage > 0:
            if not math.isfinite(scored_coverage_for_gate):
                failed.append("missing_marker_coverage")
            elif scored_coverage_for_gate < axis.min_coverage:
                failed.append("marker_coverage")
        organism_mismatch = _organism_mismatch(axis, dataset_organism)
        if organism_mismatch and axis.name not in homolog_mapped:
            failed.append("organism_mismatch")
        if axis.name in homolog_mapped and any(
            _blank(value) for value in (homolog_map_name, homolog_map_version, homolog_map_sha256)
        ):
            failed.append("missing_homolog_map_provenance")
        if n_original_markers_unique_scored < axis.min_markers_scored:
            failed.append("markers_scored")
        out[axis.name] = {
            "axis": axis.name,
            "axis_kind": axis.axis_kind,
            "markers": ",".join(axis.markers),
            "organism": axis.organism,
            "dataset_organism": dataset_organism,
            "original_organism": axis.organism,
            "scored_organism": dataset_organism or axis.organism,
            "gene_symbol_namespace": axis.gene_symbol_namespace,
            "module_source": axis.module_source,
            "null_alternative": axis.null_alternative,
            "marker_coverage": coverage,
            "min_coverage": axis.min_coverage,
            "min_markers_scored": axis.min_markers_scored,
            "coverage_original": coverage,
            "coverage_scored": scored_coverage,
            "coverage_scored_supplied": scored_coverage,
            "coverage_scored_for_gate": scored_coverage_for_gate,
            "homolog_mapped": bool(axis.name in homolog_mapped),
            "homolog_map_name": homolog_map_name,
            "homolog_map_version": homolog_map_version,
            "homolog_map_sha256": homolog_map_sha256,
            "n_markers_before_mapping": n_original_markers,
            "n_markers_after_mapping": n_original_markers_after_mapping,
            "n_markers_original": n_original_markers,
            "n_markers_mapped": n_original_markers_after_mapping,
            "n_markers_mapped_unique": n_original_markers_mapped_unique,
            "n_markers_mapped_ambiguous": n_original_markers_ambiguous,
            "n_markers_scored": n_original_markers_unique_scored,
            "n_markers_scored_unique": n_original_markers_unique_scored,
            "n_markers_scored_ambiguous": n_original_markers_scored_ambiguous,
            "n_markers_scored_including_ambiguous": n_original_markers_scored_including_ambiguous,
            "coverage_scored_unique": coverage_scored_unique_computed,
            "coverage_scored_unique_computed": coverage_scored_unique_computed,
            "coverage_scored_including_ambiguous": coverage_scored_including_ambiguous_computed,
            "coverage_scored_including_ambiguous_computed": coverage_scored_including_ambiguous_computed,
            "ambiguous_marker_fraction": ambiguous_marker_fraction,
            "n_original_markers": n_original_markers,
            "n_original_markers_after_mapping": n_original_markers_after_mapping,
            "n_original_markers_unique_scored": n_original_markers_unique_scored,
            "n_original_markers_ambiguous": n_original_markers_ambiguous,
            "n_original_markers_unmapped": n_original_markers_unmapped,
            "n_target_symbols_after_mapping": n_target_symbols_after_mapping,
            "n_target_symbols_scored": n_target_symbols_scored,
            "unmapped_markers": ",".join(str(marker) for marker in unmapped_markers),
            "ambiguous_many_to_many_markers": ",".join(str(marker) for marker in ambiguous_markers),
            "diffusion_dependence_label": _diffusion_dependence_label(null_summary),
            "score": score,
            "empirical_p": empirical_p,
            "fdr": fdr,
            "pass": not failed,
            "failed_gates": ",".join(failed),
        }
    return out


def required_baselines_for_claim(
    claim_scope: Literal["biology", "method_superiority"],
) -> tuple[str, ...]:
    """Return baseline exports required for a claim scope."""
    if claim_scope == "biology":
        return ("credo_endpoint_proxy",)
    if claim_scope == "method_superiority":
        return (
            "credo_endpoint_proxy",
            "moscot_time",
            "wot_temporal_ot",
            "wfr_mfm_unbalanced_transport",
            "scdiffeq_sde",
        )
    raise ValueError("claim_scope must be 'biology' or 'method_superiority'.")


def _pairs(records: list[FidelityRecord]) -> Iterable[tuple[FidelityRecord, FidelityRecord]]:
    for i, left in enumerate(records):
        for right in records[i + 1:]:
            yield left, right


def _spearman_mapping(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    keys = [key for key in a if key in b and math.isfinite(_finite_or_nan(a[key])) and math.isfinite(_finite_or_nan(b[key]))]
    if len(keys) < 2:
        return math.nan
    ranks_a = _ranks([float(a[key]) for key in keys])
    ranks_b = _ranks([float(b[key]) for key in keys])
    return _pearson(ranks_a, ranks_b)


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for pos in range(i, j + 1):
            ranks[order[pos]] = rank
        i = j + 1
    return ranks


def _pearson(a: list[float], b: list[float]) -> float:
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    da = [value - mean_a for value in a]
    db = [value - mean_b for value in b]
    denom = math.sqrt(sum(value * value for value in da) * sum(value * value for value in db))
    if denom == 0:
        return math.nan
    return sum(x * y for x, y in zip(da, db)) / denom


def _sign_stability(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    keys = [key for key in a if key in b]
    signs = [(_sign(a[key]), _sign(b[key])) for key in keys]
    signs = [(left, right) for left, right in signs if left != 0 and right != 0]
    if not signs:
        return math.nan
    return sum(1 for left, right in signs if left == right) / len(signs)


def _median_abs_diff(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    values = [
        abs(float(a[key]) - float(b[key]))
        for key in a
        if key in b and math.isfinite(_finite_or_nan(a[key])) and math.isfinite(_finite_or_nan(b[key]))
    ]
    return statistics.median(values) if values else math.nan


def _top_hit_jaccard(a: Mapping[str, float], b: Mapping[str, float], top_k: int) -> float:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    top_a = _top_keys(a, top_k)
    top_b = _top_keys(b, top_k)
    if not top_a and not top_b:
        return math.nan
    return len(top_a & top_b) / len(top_a | top_b)


def _top_keys(values: Mapping[str, float], top_k: int) -> set[str]:
    ranked = sorted(
        ((key, abs(float(value))) for key, value in values.items() if math.isfinite(_finite_or_nan(value))),
        key=lambda item: item[1],
        reverse=True,
    )
    return {key for key, _ in ranked[:top_k]}


def _benjamini_hochberg(p_values: Mapping[str, float]) -> dict[str, float]:
    finite = [(name, float(p)) for name, p in p_values.items() if math.isfinite(_finite_or_nan(p))]
    finite.sort(key=lambda item: item[1])
    m = len(finite)
    adjusted: dict[str, float] = {name: math.nan for name in p_values}
    running = 1.0
    for rank, (name, p_value) in reversed(list(enumerate(finite, start=1))):
        running = min(running, p_value * m / rank)
        adjusted[name] = running
    return adjusted


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[int(pos)]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def _tail_quantile(values: list[float], *, mean: float, alternative: str) -> float:
    if alternative == "less":
        return _quantile(values, 0.05)
    if alternative == "two_sided":
        return _quantile([abs(value - mean) for value in values], 0.95)
    return _quantile(values, 0.95)


def _diffusion_dependence_label(summary: Mapping[str, object]) -> str:
    status = summary.get("particle_step_convergence_status")
    if isinstance(status, str) and status.strip().lower() not in {"", "pass", "passed"}:
        return "diffusion_unstable"
    stability = _finite_or_nan(summary.get("biology_axis_stability_under_diffusion_ablation"))
    if math.isfinite(stability) and stability < 0.5:
        return "diffusion_unstable"
    delta = _finite_or_nan(summary.get("diffusion_ablation_delta"))
    if not math.isfinite(delta):
        return "diffusion_unassessed"
    if abs(delta) >= 0.5:
        return "diffusion_dependent"
    if abs(delta) >= 0.1:
        return "diffusion_sensitive"
    return "diffusion_independent"


def _finite_or_nan(value: object) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _finite_or_default(value: object, default: float) -> float:
    out = _finite_or_nan(value)
    return out if math.isfinite(out) else default


def _finite_nonnegative_values(values: Iterable[float]) -> list[float]:
    out = []
    for value in values:
        finite = _finite_or_nan(value)
        if math.isfinite(finite) and finite >= 0:
            out.append(finite)
    return out


def _symbol_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    try:
        iterator = iter(value)  # type: ignore[arg-type]
    except TypeError:
        return (str(value),)
    return tuple(str(symbol) for symbol in iterator)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _homolog_marker_counts_from_statuses(
    axis: BiologyAxisSpec,
    statuses: Iterable[HomologMarkerStatus | Mapping[str, object]],
) -> dict[str, int]:
    status_by_marker = {
        status.original_marker: status
        for status in (_coerce_homolog_marker_status(status) for status in statuses)
    }
    axis_statuses = [
        status_by_marker.get(str(marker), HomologMarkerStatus(original_marker=str(marker)))
        for marker in axis.markers
    ]
    n_original_markers = len(axis.markers)
    n_original_markers_ambiguous = sum(
        status.mapping_status == "ambiguous" for status in axis_statuses
    )
    n_original_markers_unmapped = sum(
        status.mapping_status == "unmapped" for status in axis_statuses
    )
    n_original_markers_unique_scored = sum(
        status.mapping_status == "unique_scored" for status in axis_statuses
    )
    n_original_markers_scored_ambiguous = sum(
        status.mapping_status == "ambiguous" and bool(status.scored_symbols)
        for status in axis_statuses
    )
    n_original_markers_after_mapping = n_original_markers - n_original_markers_unmapped
    n_original_markers_mapped_unique = sum(
        status.mapping_status in {"unique_scored", "unique_unscored"}
        for status in axis_statuses
    )
    n_original_markers_scored_including_ambiguous = (
        n_original_markers_unique_scored + n_original_markers_scored_ambiguous
    )
    target_symbols_after_mapping = {
        symbol for status in axis_statuses for symbol in status.mapped_symbols
    }
    target_symbols_scored = {
        symbol for status in axis_statuses for symbol in status.scored_symbols
    }
    return {
        "n_original_markers": n_original_markers,
        "n_original_markers_after_mapping": n_original_markers_after_mapping,
        "n_original_markers_unique_scored": n_original_markers_unique_scored,
        "n_original_markers_ambiguous": n_original_markers_ambiguous,
        "n_original_markers_unmapped": n_original_markers_unmapped,
        "n_original_markers_mapped_unique": n_original_markers_mapped_unique,
        "n_original_markers_scored_ambiguous": n_original_markers_scored_ambiguous,
        "n_original_markers_scored_including_ambiguous": n_original_markers_scored_including_ambiguous,
        "n_target_symbols_after_mapping": len(target_symbols_after_mapping),
        "n_target_symbols_scored": len(target_symbols_scored),
    }


def _homolog_marker_counts_from_aggregates(
    axis: BiologyAxisSpec,
    *,
    n_target_symbols_after_mapping: int,
    unmapped_set: set[str],
    ambiguous_set: set[str],
) -> dict[str, int]:
    n_original_markers = len(axis.markers)
    n_target_symbols_after_mapping = max(0, int(n_target_symbols_after_mapping))
    n_original_markers_after_mapping_upper = min(
        n_original_markers,
        n_target_symbols_after_mapping,
    )
    n_original_markers_unmapped = min(
        n_original_markers,
        max(len(unmapped_set), n_original_markers - n_original_markers_after_mapping_upper),
    )
    n_original_markers_after_mapping = n_original_markers - n_original_markers_unmapped
    n_original_markers_ambiguous = min(len(ambiguous_set), n_original_markers_after_mapping)
    n_original_markers_scored_including_ambiguous = n_original_markers_after_mapping
    n_original_markers_scored_ambiguous = min(
        len(ambiguous_set - unmapped_set),
        n_original_markers_scored_including_ambiguous,
    )
    n_original_markers_unique_scored = max(
        0,
        n_original_markers_scored_including_ambiguous - n_original_markers_scored_ambiguous,
    )
    n_original_markers_mapped_unique = max(
        0,
        n_original_markers_after_mapping - n_original_markers_ambiguous,
    )
    return {
        "n_original_markers": n_original_markers,
        "n_original_markers_after_mapping": n_original_markers_after_mapping,
        "n_original_markers_unique_scored": n_original_markers_unique_scored,
        "n_original_markers_ambiguous": n_original_markers_ambiguous,
        "n_original_markers_unmapped": n_original_markers_unmapped,
        "n_original_markers_mapped_unique": n_original_markers_mapped_unique,
        "n_original_markers_scored_ambiguous": n_original_markers_scored_ambiguous,
        "n_original_markers_scored_including_ambiguous": n_original_markers_scored_including_ambiguous,
        "n_target_symbols_after_mapping": n_target_symbols_after_mapping,
        "n_target_symbols_scored": max(0, n_target_symbols_after_mapping - len(unmapped_set)),
    }


def _coerce_homolog_marker_status(status: HomologMarkerStatus | Mapping[str, object]) -> HomologMarkerStatus:
    if isinstance(status, HomologMarkerStatus):
        return status
    if isinstance(status, Mapping):
        if "original_marker" not in status:
            raise ValueError("Homolog marker status mappings require original_marker.")
        return HomologMarkerStatus(
            original_marker=str(status["original_marker"]),
            mapped_symbols=_symbol_tuple(status.get("mapped_symbols")),
            scored_symbols=_symbol_tuple(status.get("scored_symbols")),
            mapping_status=str(status.get("mapping_status", "unmapped")),
        )
    raise TypeError("homolog marker statuses must be HomologMarkerStatus or mapping objects.")


def _coverage_fraction(count: int, total: int) -> float:
    if total <= 0:
        return math.nan
    return min(1.0, max(0.0, float(count) / float(total)))


def _blank(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _baseline_code_identity_ready(record: Mapping[str, object]) -> bool:
    if not _blank(record.get("baseline_commit_sha")):
        return True
    if not _blank(record.get("baseline_package_lock_sha256")):
        return True
    if not _blank(record.get("baseline_container_digest")):
        return True
    identity_kind = record.get("baseline_code_identity_kind")
    return (
        identity_kind in BASELINE_CODE_IDENTITY_KINDS
        and not _blank(record.get("baseline_code_identity"))
    )


def _organism_mismatch(axis: BiologyAxisSpec, dataset_organism: str | None) -> bool:
    if dataset_organism is None:
        return False
    axis_organism = axis.organism.strip().lower()
    data_organism = dataset_organism.strip().lower()
    if axis_organism in {"", "unspecified", "metadata"}:
        return False
    return bool(data_organism) and axis_organism != data_organism


def _min_finite(values: Iterable[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return min(finite) if finite else math.nan


def _max_finite(values: Iterable[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return max(finite) if finite else math.nan


def _lt(value: object, threshold: float) -> bool:
    out = _finite_or_nan(value)
    return not math.isfinite(out) or out < threshold


def _gt(value: object, threshold: float) -> bool:
    out = _finite_or_nan(value)
    return not math.isfinite(out) or out > threshold


def _sign(value: object) -> int:
    out = _finite_or_nan(value)
    if not math.isfinite(out) or out == 0:
        return 0
    return 1 if out > 0 else -1


__all__ = [
    "BASELINE_CODE_IDENTITY_KINDS",
    "BASELINE_KINDS",
    "BASELINE_STATUSES",
    "DEFAULT_BIOLOGY_AXES",
    "BiologyAxisSpec",
    "ConvergenceThresholds",
    "FidelityRecord",
    "HomologMarkerStatus",
    "baseline_export_manifest",
    "baseline_export_record",
    "baseline_export_record_from_file",
    "BASELINE_PROVENANCE_FIELDS",
    "claim_grade_convergence_thresholds",
    "estimate_convergence_thresholds_from_pilot",
    "evaluate_biology_axis_gates",
    "particle_step_convergence_diagnostics",
    "required_baselines_for_claim",
    "summarize_null_distribution",
    "summarize_null_suite",
]
