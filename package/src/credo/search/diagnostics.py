"""Claim-support diagnostics for CREDO setting search.

These helpers keep final-selection evidence explicit: convergence across
particle/step fidelities, empirical null calibration, baseline-export manifests,
and biology-axis gates are recorded as structured payloads rather than prose.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional


BASELINE_KINDS: tuple[str, ...] = (
    "credo_endpoint_proxy",
    "credo_weak_form",
    "moscot_time",
    "wot_temporal_ot",
    "wfr_mfm_unbalanced_transport",
    "scdiffeq_sde",
)
BASELINE_STATUSES: tuple[str, ...] = ("planned", "available", "skipped", "failed")


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

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Biology axis name must not be empty.")
        if not self.markers:
            raise ValueError(f"Biology axis {self.name!r} must include at least one marker.")
        if self.expected_direction not in {None, "positive", "negative", "either"}:
            raise ValueError("expected_direction must be positive, negative, either, or None.")
        if not 0.0 <= float(self.min_coverage) <= 1.0:
            raise ValueError("min_coverage must be in [0, 1].")


DEFAULT_BIOLOGY_AXES: tuple[BiologyAxisSpec, ...] = (
    BiologyAxisSpec(
        name="tnf_expansion",
        markers=("Notch1", "Notch2", "Fat1", "Trp53", "Fgf3"),
        expected_direction="positive",
        organism="mouse",
        gene_symbol_namespace="MGI",
        module_source="Renz_2024_P4_P60_field_expansion",
        min_coverage=0.7,
    ),
    BiologyAxisSpec(
        name="cis_like_epithelial",
        markers=("TP63", "ATP1B3", "KRT5", "KRT14", "KRT17", "SOX2", "EPCAM"),
        expected_direction="either",
        organism="human",
        gene_symbol_namespace="HGNC",
        module_source="Choi_2023_CIS_like_LP",
        min_coverage=0.7,
    ),
    BiologyAxisSpec(
        name="pemt_tsk",
        markers=("LGALS7B", "VIM", "SNAI2", "LAMC2", "ITGA5"),
        expected_direction="either",
        organism="human",
        gene_symbol_namespace="HGNC",
        module_source="Choi_2023_CC1_Punovuori_2024_pEMT",
        min_coverage=0.7,
    ),
    BiologyAxisSpec(
        name="caf_ecm",
        markers=("COL1A1", "COL1A2", "FN1", "POSTN", "THBS2"),
        expected_direction="either",
        organism="human",
        gene_symbol_namespace="HGNC",
        module_source="Punovuori_2024_CAF_ECM",
        min_coverage=0.7,
    ),
    BiologyAxisSpec(
        name="myeloid",
        markers=("LYZ", "S100A8", "S100A9", "FCGR3A"),
        expected_direction="either",
        organism="human",
        gene_symbol_namespace="HGNC",
        module_source="HNSCC_inflammatory_myeloid",
        min_coverage=0.7,
    ),
    BiologyAxisSpec(
        name="guide_artifact",
        markers=("guide_concordance",),
        expected_direction="negative",
        organism="metadata",
        gene_symbol_namespace="metric",
        module_source="CREDO_guide_artifact_axis",
    ),
    BiologyAxisSpec(
        name="large_gene_artifact",
        markers=("gene_size",),
        expected_direction="negative",
        organism="metadata",
        gene_symbol_namespace="metric",
        module_source="CREDO_large_gene_artifact_axis",
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
            "null_gap_p95": math.nan,
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
        extreme = sum(1 for value in values if abs(value) >= abs(observed))
    empirical_p = (extreme + 1.0) / (len(values) + 1.0)
    if sd > 0:
        null_gap_z = (observed - mean) / sd
    elif observed == mean:
        null_gap_z = 0.0
    else:
        null_gap_z = math.copysign(math.inf, observed - mean)
    return {
        "null_gap_mean": mean,
        "null_gap_p95": _quantile(values, 0.95),
        "null_gap_z": null_gap_z,
        "empirical_p": empirical_p,
        "fdr": empirical_p,
    }


def summarize_null_suite(
    observed_gaps: Mapping[str, float],
    nulls_by_name: Mapping[str, Iterable[float]],
    *,
    alternative: str = "greater",
) -> dict[str, dict[str, float]]:
    """Summarize multiple nulls and apply Benjamini-Hochberg FDR correction."""
    summaries = {
        name: summarize_null_distribution(nulls_by_name.get(name, ()), observed, alternative=alternative)
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
    metrics: Optional[Mapping[str, float]] = None,
    status: str = "planned",
    notes: str | None = None,
) -> dict[str, object]:
    """Build one stable baseline-export manifest row."""
    if baseline_kind not in BASELINE_KINDS:
        raise ValueError(f"Unknown baseline_kind {baseline_kind!r}; expected one of {BASELINE_KINDS}.")
    if status not in BASELINE_STATUSES:
        raise ValueError(f"Unknown baseline status {status!r}; expected one of {BASELINE_STATUSES}.")
    record: dict[str, object] = {
        "baseline_kind": baseline_kind,
        "status": status,
        "artifact_path": None if artifact_path is None else str(artifact_path),
        "notes": notes,
    }
    for key, value in dict(metrics or {}).items():
        record[f"metric.{key}"] = float(value)
    return record


def baseline_export_manifest(
    records: Iterable[Mapping[str, object]],
    *,
    required: Iterable[str] = BASELINE_KINDS,
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
    return {"records": rows, "required_missing": missing, "complete": not missing}


def evaluate_biology_axis_gates(
    axis_scores: Mapping[str, float],
    *,
    null_summaries: Optional[Mapping[str, Mapping[str, float]]] = None,
    coverage_by_axis: Optional[Mapping[str, float]] = None,
    axes: Iterable[BiologyAxisSpec] = DEFAULT_BIOLOGY_AXES,
    score_abs_min: float = 0.0,
    empirical_p_max: float = 0.05,
    fdr_max: float = 0.10,
) -> dict[str, dict[str, object]]:
    """Gate biology support separately for each configured axis."""
    null_summaries = null_summaries or {}
    coverage_by_axis = coverage_by_axis or {}
    out: dict[str, dict[str, object]] = {}
    for axis in axes:
        score = _finite_or_nan(axis_scores.get(axis.name))
        null_summary = null_summaries.get(axis.name, {})
        empirical_p = _finite_or_nan(null_summary.get("empirical_p"))
        fdr = _finite_or_nan(null_summary.get("fdr"))
        coverage = _finite_or_nan(coverage_by_axis.get(axis.name))
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
        if math.isfinite(coverage) and coverage < axis.min_coverage:
            failed.append("marker_coverage")
        out[axis.name] = {
            "axis": axis.name,
            "markers": ",".join(axis.markers),
            "organism": axis.organism,
            "gene_symbol_namespace": axis.gene_symbol_namespace,
            "module_source": axis.module_source,
            "marker_coverage": coverage,
            "min_coverage": axis.min_coverage,
            "score": score,
            "empirical_p": empirical_p,
            "fdr": fdr,
            "pass": not failed,
            "failed_gates": ",".join(failed),
        }
    return out


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


def _finite_or_nan(value: object) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _finite_or_default(value: object, default: float) -> float:
    out = _finite_or_nan(value)
    return out if math.isfinite(out) else default


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
    "BASELINE_KINDS",
    "BASELINE_STATUSES",
    "DEFAULT_BIOLOGY_AXES",
    "BiologyAxisSpec",
    "ConvergenceThresholds",
    "FidelityRecord",
    "baseline_export_manifest",
    "baseline_export_record",
    "claim_grade_convergence_thresholds",
    "evaluate_biology_axis_gates",
    "particle_step_convergence_diagnostics",
    "summarize_null_distribution",
    "summarize_null_suite",
]
