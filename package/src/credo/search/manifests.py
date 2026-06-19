"""Trial-manifest database for CREDO setting search.

Stores one flat record per completed trial (spec + metrics + objective vector +
constraints + a stable spec hash) so a search run is fully reproducible and the
Pareto analysis can be redone offline. Records are appended as JSONL and can be
materialized to a flat table (list of dicts) for CSV export or pandas.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
import uuid
from pathlib import Path
from typing import Any, Iterable

# Filesystem-safe trial id (prevents path traversal / surprising directory names).
# Dots are excluded so ".." can never appear as a path component.
_TRIAL_ID_UNSAFE = re.compile(r"[^A-Za-z0-9_=-]+")


def _slug(value: str) -> str:
    cleaned = _TRIAL_ID_UNSAFE.sub("-", str(value)).strip("-")[:120]
    return cleaned or uuid.uuid4().hex

from .metrics import CREDOTrialResult
from .objective import SearchProfile
from .space import CREDOTrialSpec

# Bump when the flattened trial-record schema changes (field names/prefixes).
SEARCH_SCHEMA_VERSION = "credo.search.v2"


def spec_sha256(spec: CREDOTrialSpec) -> str:
    """Deterministic content hash of a trial spec (for dedup / cache keys)."""
    payload = json.dumps(dataclasses.asdict(spec), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def setting_sha256(spec: CREDOTrialSpec, builder_metadata: Any | None = None) -> str:
    """Hash a setting across folds/seeds by excluding stochastic split identity.

    Builder fingerprints are part of setting identity when present: two studies
    with the same hyperparameters but different preprocessing or gene panels are
    not the same final setting.
    """
    payload_dict = dataclasses.asdict(spec)
    payload_dict.pop("seed", None)
    payload_dict.pop("fold_id", None)
    builder_record = _builder_metadata_record(builder_metadata)
    if builder_record:
        payload_dict["builder_metadata"] = builder_record
    payload = json.dumps(payload_dict, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def trial_record(result: CREDOTrialResult) -> dict[str, Any]:
    """Flatten a :class:`CREDOTrialResult` into one JSON-serializable row."""
    spec = result.spec
    builder_record = _builder_metadata_record(result.builder_metadata)
    record: dict[str, Any] = {
        "schema_version": SEARCH_SCHEMA_VERSION,
        "spec_sha256": spec_sha256(spec),
        "setting_sha256": setting_sha256(spec, builder_record),
        "run_dir": result.run_dir,
        "checkpoint_path": result.checkpoint_path,
        "history_path": result.history_path,
        "eval_summary_path": result.eval_summary_path,
        "resolved_config_path": result.resolved_config_path,
        "failure_type": result.failure_type,
        "failure_message": result.failure_message,
        "pruner_score": result.pruner_score,
        "feasible": result.feasible,
    }
    record.update({f"spec.{k}": v for k, v in dataclasses.asdict(spec).items()})
    record.update({f"metric.{k}": v for k, v in dataclasses.asdict(result.metrics).items()})
    record.update({f"objective.{k}": v for k, v in result.objective_vector.items()})
    endpoint_value, endpoint_kind = _canonical_endpoint(result.metrics)
    record["objective.endpoint_metric_kind"] = endpoint_kind
    record["objective.endpoint_geometry_or_proxy"] = endpoint_value
    record["objective.endpoint_mass_penalty_or_nan"] = (
        float(result.metrics.endpoint_mass_penalty)
        if _finite(result.metrics.endpoint_mass_penalty)
        else math.nan
    )
    record.update({f"constraint.{k}": v for k, v in result.constraints.items()})
    record.update({f"constraints.{k}": v for k, v in result.threshold_metadata.items()})
    record.update({f"builder.{k}": v for k, v in builder_record.items()})
    return record


def _builder_metadata_record(metadata: Any | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    if hasattr(metadata, "to_record"):
        metadata = metadata.to_record()
    elif dataclasses.is_dataclass(metadata):
        metadata = dataclasses.asdict(metadata)
    if not isinstance(metadata, dict):
        raise TypeError(
            "builder_metadata must be a mapping, dataclass, or object with to_record()."
        )
    return {str(key): value for key, value in metadata.items()}


def _canonical_endpoint(metrics: Any) -> tuple[float, str]:
    if _finite(getattr(metrics, "endpoint_sinkhorn", None)):
        return float(metrics.endpoint_sinkhorn), "decomposed"
    if _finite(getattr(metrics, "endpoint_geom_mass", None)):
        return float(metrics.endpoint_geom_mass), "combined_proxy"
    return math.nan, "missing"


def _record_with_canonical_endpoint(record: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)
    if "objective.endpoint_metric_kind" not in out:
        if _finite(out.get("objective.endpoint_geometry")):
            out["objective.endpoint_metric_kind"] = "decomposed"
        elif _finite(out.get("objective.endpoint_geom_mass")):
            out["objective.endpoint_metric_kind"] = "combined_proxy"
        else:
            out["objective.endpoint_metric_kind"] = "missing"
    if "objective.endpoint_geometry_or_proxy" not in out:
        if _finite(out.get("objective.endpoint_geometry")):
            out["objective.endpoint_geometry_or_proxy"] = float(out["objective.endpoint_geometry"])
        elif _finite(out.get("objective.endpoint_geom_mass")):
            out["objective.endpoint_geometry_or_proxy"] = float(out["objective.endpoint_geom_mass"])
        else:
            out["objective.endpoint_geometry_or_proxy"] = math.nan
    if "objective.endpoint_mass_penalty_or_nan" not in out:
        out["objective.endpoint_mass_penalty_or_nan"] = (
            float(out["objective.endpoint_mass_penalty"])
            if _finite(out.get("objective.endpoint_mass_penalty"))
            else math.nan
        )
    return out


def append_trial_record(path: str | Path, result: CREDOTrialResult) -> Path:
    """Append one trial record as a JSONL line.

    NOTE: this is a single-writer convenience. Concurrent search workers
    (multiprocess Optuna/Ray) can interleave appends and corrupt the file. For
    parallel sweeps, have each worker call :func:`write_trial_dir` (one atomic
    file per trial) and build the JSONL with :func:`reduce_trial_dirs` afterward,
    or use Optuna RDB storage as the source of truth.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(trial_record(result), sort_keys=True, default=str))
        handle.write("\n")
    return out


def write_trial_dir(
    root: str | Path,
    result: CREDOTrialResult,
    *,
    index: int | None = None,
    trial_id: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Write one trial as its own directory (parallel-safe source of truth).

    Layout: ``<root>/trial_<idx?>_<trial_id?>_<spec_sha8>/result.json``. Each
    worker writes a distinct directory atomically (temp file + rename), so
    concurrent trials do not contend on a shared file. Pass a unique ``index``
    and/or ``trial_id`` (e.g. Optuna ``trial.number`` or a UUID) so that
    re-runs of the same spec hash -- different seed/fold or a retry -- do NOT
    overwrite each other (the spec hash alone is not unique across seeds/folds).
    If ``trial_id`` is omitted a random UUID is generated, so write_trial_dir is
    overwrite-safe by default. Use :func:`reduce_trial_dirs` to materialize a
    combined JSONL cache.
    """
    trial_id = _slug(trial_id) if trial_id is not None else uuid.uuid4().hex
    record = trial_record(result)
    record["trial_id"] = trial_id
    record["trial_index"] = index
    sha8 = str(record["spec_sha256"])[:8]
    parts = ["trial"]
    if index is not None:
        parts.append(f"{index:06d}")
    parts.append(str(trial_id))
    parts.append(sha8)
    name = "_".join(parts)
    trial_dir = Path(root) / name
    if trial_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Trial directory already exists: {trial_dir}. Pass a unique index/trial_id, "
            "or overwrite=True to replace it."
        )
    trial_dir.mkdir(parents=True, exist_ok=overwrite)
    target = trial_dir / "result.json"
    tmp = trial_dir / "result.json.tmp"
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(target)  # atomic on POSIX within a filesystem
    return trial_dir


def reduce_trial_dirs(root: str | Path, out_jsonl: str | Path) -> Path:
    """Collect all per-trial ``result.json`` files under ``root`` into one JSONL.

    The per-trial directories are the source of truth; the JSONL is a rebuildable
    cache, so this is safe to re-run after a partially-completed sweep.
    """
    root_path = Path(root)
    out = Path(out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    records = sorted(
        (json.loads(p.read_text(encoding="utf-8")) for p in root_path.glob("trial_*/result.json")),
        key=lambda r: (
            r["trial_index"] if isinstance(r.get("trial_index"), int) else 10**12,
            str(r.get("trial_id", "")),
            str(r.get("spec_sha256", "")),
        ),
    )
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, default=str))
            handle.write("\n")
    return out


def load_trial_records(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL trial database back into a list of records."""
    out = Path(path)
    if not out.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in out.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def pareto_front(
    records: Iterable[dict[str, Any]],
    objectives: list[str],
    *,
    feasible_only: bool = True,
) -> list[dict[str, Any]]:
    """Return the non-dominated records over the given (minimized) objectives.

    ``objectives`` are record keys (e.g. ``"objective.endpoint_geometry"``).
    Records missing any objective key, or (when ``feasible_only``) flagged
    infeasible, are excluded.
    """
    rows = [r for r in records if all(_finite(r.get(o)) for o in objectives)]
    if feasible_only:
        rows = [r for r in rows if r.get("feasible", False)]
    front: list[dict[str, Any]] = []
    for cand in rows:
        dominated = False
        for other in rows:
            if other is cand:
                continue
            if _dominates(other, cand, objectives):
                dominated = True
                break
        if not dominated:
            front.append(cand)
    return front


def select_final_candidates(
    records: Iterable[dict[str, Any]],
    *,
    profile: SearchProfile | str = SearchProfile.CLAIM_GRADE,
    require_feasible: bool = True,
    require_heldout: bool = True,
    require_guide_concordance: bool | None = None,
    require_builder_metadata: bool | None = None,
    require_finite_thresholds: bool | None = None,
    require_explicit_grid: bool | None = None,
    group_by: tuple[str, ...] = ("setting_sha256",),
    aggregate_by: tuple[str, ...] = ("spec.fold_id", "spec.seed"),
    objectives: list[str] | None = None,
    sort_by: str | None = None,
    min_folds: int | None = None,
    min_seeds: int | None = None,
    required_folds: Iterable[Any] | None = None,
    required_seeds: Iterable[Any] | None = None,
    expected_thresholds_sha256: str | None = None,
    expected_fold_grid_sha256: str | None = None,
    expected_seed_grid: Iterable[Any] | str | None = None,
    expected_split_manifest_sha256: str | None = None,
) -> list[dict[str, Any]]:
    """Aggregate trial records and return constrained Pareto-final candidates.

    This is the selection path for refit / claim-grade settings. Scalar
    ``pruner_score`` is allowed for cheap screening, but claim-grade final
    selection must be based on aggregated objective axes after fold/seed checks.
    """
    profile = SearchProfile(profile)
    if profile is SearchProfile.CLAIM_GRADE and sort_by == "pruner_score":
        raise ValueError("CLAIM_GRADE final selection cannot rank by pruner_score.")
    if profile in (SearchProfile.PARETO_REFIT, SearchProfile.CLAIM_GRADE) and not aggregate_by:
        raise ValueError(f"{profile.value} selection requires fold/seed aggregation.")

    fold_min = _profile_min_folds(profile) if min_folds is None else min_folds
    seed_min = _profile_min_seeds(profile) if min_seeds is None else min_seeds
    builder_required = (
        profile is SearchProfile.CLAIM_GRADE
        if require_builder_metadata is None
        else bool(require_builder_metadata)
    )
    finite_thresholds_required = (
        profile is SearchProfile.CLAIM_GRADE
        if require_finite_thresholds is None
        else bool(require_finite_thresholds)
    )
    explicit_grid_required = (
        profile is SearchProfile.CLAIM_GRADE
        if require_explicit_grid is None
        else bool(require_explicit_grid)
    )
    if explicit_grid_required and (required_folds is None or required_seeds is None):
        raise ValueError(
            f"{profile.value} selection requires explicit required_folds and required_seeds."
        )
    group_keys = tuple(_record_key(name) for name in group_by)
    aggregate_keys = tuple(_record_key(name) for name in aggregate_by)
    required_fold_values = tuple(required_folds) if required_folds is not None else None
    required_seed_values = tuple(required_seeds) if required_seeds is not None else None
    requested_fold_grid_sha256 = expected_fold_grid_sha256 or _fold_grid_sha256(required_fold_values)
    requested_seed_grid = _canonical_seed_grid(
        expected_seed_grid if expected_seed_grid is not None else required_seed_values
    )
    all_rows = [_record_with_canonical_endpoint(row) for row in records]
    all_grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in all_rows:
        key = tuple(row.get(name) for name in group_keys)
        all_grouped.setdefault(key, []).append(row)

    rows = list(all_rows)
    if require_feasible:
        rows = [r for r in rows if bool(r.get("feasible", False))]
    if require_heldout:
        rows = [r for r in rows if r.get("metric.validation_source") == "held_out"]
    if profile is SearchProfile.CLAIM_GRADE:
        rows = [
            r
            for r in rows
            if _record_claim_grade_ready(
                r,
                require_guide_concordance=require_guide_concordance,
                require_builder_metadata=builder_required,
                require_finite_thresholds=finite_thresholds_required,
                expected_fold_grid_sha256=requested_fold_grid_sha256,
                expected_seed_grid=requested_seed_grid,
                expected_split_manifest_sha256=expected_split_manifest_sha256,
            )
        ]
    elif profile is SearchProfile.PARETO_REFIT:
        rows = [r for r in rows if _record_pareto_refit_ready(r)]
    objective_names = objectives or _default_objectives_from_records(rows, profile=profile)
    if not objective_names:
        return []
    objective_keys = [_objective_key(name) for name in objective_names]

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(name) for name in group_keys)
        grouped.setdefault(key, []).append(row)

    candidates: list[dict[str, Any]] = []
    skipped_for_coverage = False
    for key, group in grouped.items():
        coverage = _fold_seed_coverage(
            group,
            aggregate_keys,
            min_folds=fold_min,
            min_seeds=seed_min,
            required_folds=required_fold_values,
            required_seeds=required_seed_values,
        )
        if not coverage["ok"]:
            skipped_for_coverage = True
            continue
        if profile is SearchProfile.CLAIM_GRADE and not _claim_grade_group_uniformity_ready(
            group,
            expected_thresholds_sha256=expected_thresholds_sha256,
        ):
            continue
        total_group = all_grouped.get(key, group)
        aggregate = _aggregate_group(
            key=key,
            group_by=group_by,
            group=group,
            total_group=total_group,
            profile=profile,
            objective_keys=objective_keys,
            aggregate_keys=aggregate_keys,
            fold_count=int(coverage["fold_count"]),
            seed_count=int(coverage["seed_count"]),
            missing_pairs=coverage["missing_pairs"],
        )
        if all(_finite(aggregate.get(f"{key}.mean")) for key in objective_keys):
            candidates.append(aggregate)

    if not candidates and skipped_for_coverage:
        raise ValueError(
            f"{profile.value} selection found no candidates with required "
            f"{fold_min} folds x {seed_min} seeds coverage."
        )
    if sort_by is not None:
        sort_key = _aggregate_sort_key(sort_by)
        candidates.sort(key=lambda row: _sort_value(row.get(sort_key)))
        return candidates
    pareto_objectives = [f"{key}.mean" for key in objective_keys]
    return pareto_front(candidates, pareto_objectives, feasible_only=False)


def _dominates(a: dict[str, Any], b: dict[str, Any], objectives: list[str]) -> bool:
    no_worse = all(float(a[o]) <= float(b[o]) for o in objectives)
    strictly_better = any(float(a[o]) < float(b[o]) for o in objectives)
    return no_worse and strictly_better


def _aggregate_group(
    *,
    key: tuple[Any, ...],
    group_by: tuple[str, ...],
    group: list[dict[str, Any]],
    total_group: list[dict[str, Any]],
    profile: SearchProfile,
    objective_keys: list[str],
    aggregate_keys: tuple[str, ...],
    fold_count: int,
    seed_count: int,
    missing_pairs: object,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "profile": profile.value,
        "n_trials": len(group),
        "n_trials_total": len(total_group),
        "n_trials_feasible": sum(1 for row in total_group if row.get("feasible", False)),
        "fold_count": fold_count,
        "seed_count": seed_count,
        "fold_pass_rate": _axis_pass_rate(group, aggregate_keys[0] if len(aggregate_keys) >= 1 else None),
        "fold_pass_rate_all": _axis_pass_rate(
            total_group, aggregate_keys[0] if len(aggregate_keys) >= 1 else None
        ),
        "fold_pass_rate_after_filter": _axis_pass_rate(
            group, aggregate_keys[0] if len(aggregate_keys) >= 1 else None
        ),
        "seed_pass_rate_all": _axis_pass_rate(
            total_group, aggregate_keys[1] if len(aggregate_keys) >= 2 else None
        ),
        "seed_pass_rate_after_filter": _axis_pass_rate(
            group, aggregate_keys[1] if len(aggregate_keys) >= 2 else None
        ),
        "missing_fold_seed_pairs": missing_pairs,
        "worst_fold_ess": _min_finite(
            row.get("metric.min_ess_frac_over_time", row.get("metric.terminal_ess_frac_min"))
            for row in group
        ),
        "worst_fold_max_weight_frac": _max_finite(row.get("metric.max_weight_frac_mean") for row in group),
        "worst_source_ess_frac": _min_finite(row.get("metric.source_ess_frac") for row in group),
        "worst_factual_min_ess_frac_over_time": _min_finite(
            row.get("metric.factual_min_ess_frac_over_time") for row in group
        ),
        "worst_reference_min_ess_frac_over_time": _min_finite(
            row.get("metric.reference_min_ess_frac_over_time") for row in group
        ),
        "worst_factual_max_weight_frac": _max_finite(
            row.get("metric.factual_max_weight_frac") for row in group
        ),
        "worst_reference_max_weight_frac": _max_finite(
            row.get("metric.reference_max_weight_frac") for row in group
        ),
        "worst_factual_logw_range": _max_finite(row.get("metric.factual_logw_range") for row in group),
        "worst_reference_logw_range": _max_finite(
            row.get("metric.reference_logw_range") for row in group
        ),
        "pruner_score.mean": _mean_finite(row.get("pruner_score") for row in group),
    }
    for name, value in zip(group_by, key):
        out[name] = value
    for objective in objective_keys:
        values = [_to_float(row.get(objective)) for row in group]
        finite = [value for value in values if math.isfinite(value)]
        if finite:
            out[f"{objective}.mean"] = sum(finite) / len(finite)
            out[f"{objective}.se"] = _standard_error(finite)
    return out


def _default_objectives_from_records(
    records: list[dict[str, Any]],
    *,
    profile: SearchProfile,
) -> list[str]:
    if not records:
        return ["endpoint_geom_mass", "mass_error"]
    endpoint_kinds = {
        row.get("objective.endpoint_metric_kind")
        for row in records
        if row.get("objective.endpoint_metric_kind") is not None
    }
    if (
        len(endpoint_kinds) > 1
        and all(_finite(row.get("objective.endpoint_geometry_or_proxy")) for row in records)
    ):
        objectives = ["endpoint_geometry_or_proxy"]
    elif all(_finite(row.get("objective.endpoint_geometry")) for row in records):
        objectives = ["endpoint_geometry"]
    elif all(_finite(row.get("objective.endpoint_geometry_or_proxy")) for row in records):
        objectives = ["endpoint_geometry_or_proxy"]
    elif all(_finite(row.get("objective.endpoint_geom_mass")) for row in records):
        objectives = ["endpoint_geom_mass"]
    else:
        return []
    if all(_finite(row.get("objective.endpoint_mass_penalty")) for row in records):
        objectives.append("endpoint_mass_penalty")
    for name in (
        "mass_error",
        "counterfactual_null_gap",
        "guide_concordance_gap",
    ):
        key = _objective_key(name)
        if all(_finite(row.get(key)) for row in records):
            objectives.append(name)
    if profile is not SearchProfile.CLAIM_GRADE and all(
        _finite(row.get("objective.gpu_seconds")) for row in records
    ):
        objectives.append("gpu_seconds")
    return objectives


def _objective_key(name: str) -> str:
    return name if name.startswith("objective.") else f"objective.{name}"


def _aggregate_sort_key(name: str) -> str:
    if name == "pruner_score":
        return "pruner_score.mean"
    if name.endswith(".mean") or name.endswith(".se"):
        return name
    if name.startswith("objective."):
        return f"{name}.mean"
    return f"objective.{name}.mean"


def _record_key(name: str) -> str:
    if "." in name or name in {"spec_sha256", "setting_sha256", "schema_version", "feasible", "pruner_score"}:
        return name
    return f"spec.{name}"


def _distinct_values(records: list[dict[str, Any]], key: str) -> set[Any]:
    return {record.get(key) for record in records if record.get(key) is not None}


def _fold_seed_coverage(
    records: list[dict[str, Any]],
    aggregate_keys: tuple[str, ...],
    *,
    min_folds: int,
    min_seeds: int,
    required_folds: Iterable[Any] | None = None,
    required_seeds: Iterable[Any] | None = None,
) -> dict[str, object]:
    if len(aggregate_keys) < 2:
        return {
            "ok": (
                min_folds <= 1
                and min_seeds <= 1
                and required_folds is None
                and required_seeds is None
            ),
            "fold_count": 0,
            "seed_count": 0,
            "missing_pairs": [],
        }
    fold_key, seed_key = aggregate_keys[:2]
    observed_folds = sorted({_canonical_fold_id(value) for value in _distinct_values(records, fold_key)})
    observed_seeds = sorted({_canonical_seed_id(value) for value in _distinct_values(records, seed_key)})
    folds = _canonical_fold_values(required_folds) if required_folds is not None else observed_folds
    seeds = _canonical_seed_values(required_seeds) if required_seeds is not None else observed_seeds
    observed_pairs = {
        (_canonical_fold_id(row.get(fold_key)), _canonical_seed_id(row.get(seed_key)))
        for row in records
        if row.get(fold_key) is not None and row.get(seed_key) is not None
    }
    required_pairs = {(fold, seed) for fold in folds for seed in seeds}
    missing = sorted(required_pairs - observed_pairs, key=lambda item: (str(item[0]), str(item[1])))
    ok = (
        len(folds) >= min_folds
        and len(seeds) >= min_seeds
        and len(records) >= min_folds * min_seeds
        and not missing
    )
    return {
        "ok": ok,
        "fold_count": len(observed_folds),
        "seed_count": len(observed_seeds),
        "missing_pairs": [f"{fold}:{seed}" for fold, seed in missing],
    }


def _axis_pass_rate(records: list[dict[str, Any]], key: str | None) -> float:
    if key is None:
        return math.nan
    groups: dict[Any, list[dict[str, Any]]] = {}
    for record in records:
        value = record.get(key)
        if value is not None:
            groups.setdefault(value, []).append(record)
    if not groups:
        return math.nan
    passed = sum(1 for group in groups.values() if any(row.get("feasible", False) for row in group))
    return passed / len(groups)


def _profile_min_folds(profile: SearchProfile) -> int:
    return 4 if profile in (SearchProfile.PARETO_REFIT, SearchProfile.CLAIM_GRADE) else 1


def _profile_min_seeds(profile: SearchProfile) -> int:
    return 3 if profile in (SearchProfile.PARETO_REFIT, SearchProfile.CLAIM_GRADE) else 1


def _record_pareto_refit_ready(record: dict[str, Any]) -> bool:
    return (
        record.get("metric.mass_error_kind") == "abs_log_residual"
        and _finite(record.get("metric.mass_error_value"))
        and _finite(record.get("metric.control_null_gap"))
    )


_GUIDE_CONCORDANCE_CLAIM_TYPES = {
    "same_perturbation_counterfactual",
    "guide_generalization",
    "target_gene_generalization",
}

_REQUIRED_BUILDER_FIELDS = (
    "builder.builder_name",
    "builder.builder_version",
    "builder.data_path_hash",
    "builder.mass_table_hash",
    "builder.split_file_hash",
    "builder.fold_assignment_hash",
    "builder.latent_source",
    "builder.latent_key",
    "builder.gene_panel_hash",
    "builder.normalization_hash",
    "builder.hvg_preprocessing_hash",
    "builder.dataset_organism",
    "builder.gene_symbol_namespace",
    "builder.expression_gene_universe_hash",
    "builder.decoder_gene_panel_hash",
    "builder.fold_grid_sha256",
    "builder.seed_grid",
    "builder.split_manifest_sha256",
)


def _record_claim_grade_ready(
    record: dict[str, Any],
    *,
    require_guide_concordance: bool | None,
    require_builder_metadata: bool,
    require_finite_thresholds: bool,
    expected_fold_grid_sha256: str | None,
    expected_seed_grid: str | None,
    expected_split_manifest_sha256: str | None,
) -> bool:
    branch_ready = _record_pareto_refit_ready(record) and all(
        _finite(record.get(key))
        for key in (
            "metric.source_ess_frac",
            "metric.factual_terminal_ess_frac",
            "metric.reference_terminal_ess_frac",
            "metric.factual_min_ess_frac_over_time",
            "metric.reference_min_ess_frac_over_time",
            "metric.factual_max_weight_frac",
            "metric.reference_max_weight_frac",
            "metric.factual_logw_range",
            "metric.reference_logw_range",
        )
    )
    if not branch_ready:
        return False
    if require_guide_concordance is True or (
        require_guide_concordance is None
        and record.get("spec.claim_type") in _GUIDE_CONCORDANCE_CLAIM_TYPES
    ):
        if not _finite(record.get("metric.guide_concordance_gap")):
            return False
        if require_finite_thresholds and not _guide_threshold_ready(record):
            return False
    if require_builder_metadata:
        if not _claim_grade_builder_metadata_ready(record):
            return False
        if not _claim_grade_builder_grid_ready(
            record,
            expected_fold_grid_sha256=expected_fold_grid_sha256,
            expected_seed_grid=expected_seed_grid,
            expected_split_manifest_sha256=expected_split_manifest_sha256,
        ):
            return False
    if require_finite_thresholds and not _claim_grade_thresholds_ready(record):
        return False
    return True


def _claim_grade_builder_metadata_ready(record: dict[str, Any]) -> bool:
    if any(_blank(record.get(key)) for key in _REQUIRED_BUILDER_FIELDS):
        return False
    return not (
        _blank(record.get("builder.encoder_checkpoint_hash"))
        and _blank(record.get("builder.representation_config_sha256"))
    )


def _claim_grade_thresholds_ready(record: dict[str, Any]) -> bool:
    mass_error_kind = record.get(
        "constraints.mass_error_kind",
        record.get("constraints.required_mass_error_kind"),
    )
    mass_error_max = record.get(
        "constraints.mass_error_max",
        record.get("constraints.log_mass_error_max"),
    )
    return (
        record.get("constraints.threshold_profile") == "claim_grade_finite"
        and _finite(record.get("constraints.control_null_max"))
        and mass_error_kind == "abs_log_residual"
        and _finite(mass_error_max)
        and not _blank(record.get("constraints.thresholds_sha256"))
    )


def _claim_grade_builder_grid_ready(
    record: dict[str, Any],
    *,
    expected_fold_grid_sha256: str | None,
    expected_seed_grid: str | None,
    expected_split_manifest_sha256: str | None,
) -> bool:
    if expected_fold_grid_sha256 is not None and record.get("builder.fold_grid_sha256") != expected_fold_grid_sha256:
        return False
    if expected_seed_grid is not None and _canonical_seed_grid(record.get("builder.seed_grid")) != expected_seed_grid:
        return False
    return not (
        expected_split_manifest_sha256 is not None
        and record.get("builder.split_manifest_sha256") != expected_split_manifest_sha256
    )


def _claim_grade_group_uniformity_ready(
    group: list[dict[str, Any]],
    *,
    expected_thresholds_sha256: str | None,
) -> bool:
    threshold_hashes = _unique_nonblank(row.get("constraints.thresholds_sha256") for row in group)
    if len(threshold_hashes) != 1:
        return False
    return expected_thresholds_sha256 is None or next(iter(threshold_hashes)) == expected_thresholds_sha256


def _guide_threshold_ready(record: dict[str, Any]) -> bool:
    return (
        bool(record.get("constraints.require_guide_concordance"))
        and _finite(record.get("constraints.guide_concordance_max"))
    )


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _unique_nonblank(values: Iterable[Any]) -> set[str]:
    return {str(value) for value in values if not _blank(value)}


def _canonical_fold_id(value: Any) -> str:
    return str(value).strip()


def _canonical_fold_values(values: Iterable[Any]) -> list[str]:
    return sorted({_canonical_fold_id(value) for value in values})


def _canonical_seed_id(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("seed values must be integer-like, not booleans.")
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"seed values must be integer-like, got {value!r}.")
        return int(value)
    text = str(value).strip()
    if not text:
        raise ValueError("seed values must not be blank.")
    return int(text)


def _canonical_seed_values(values: Iterable[Any]) -> list[int]:
    return sorted({_canonical_seed_id(value) for value in values})


def _canonical_seed_grid(values: Iterable[Any] | str | None) -> str | None:
    if values is None:
        return None
    if isinstance(values, str):
        values = [value.strip() for value in values.split(",") if value.strip()]
    return ",".join(str(seed) for seed in _canonical_seed_values(values))


def _fold_grid_sha256(folds: Iterable[Any] | None) -> str | None:
    if folds is None:
        return None
    payload = json.dumps({"folds": _canonical_fold_values(folds)}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _to_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _mean_finite(values: Iterable[Any]) -> float:
    finite = [_to_float(value) for value in values]
    finite = [value for value in finite if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else math.nan


def _min_finite(values: Iterable[Any]) -> float:
    finite = [_to_float(value) for value in values]
    finite = [value for value in finite if math.isfinite(value)]
    return min(finite) if finite else math.nan


def _max_finite(values: Iterable[Any]) -> float:
    finite = [_to_float(value) for value in values]
    finite = [value for value in finite if math.isfinite(value)]
    return max(finite) if finite else math.nan


def _standard_error(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def _sort_value(value: Any) -> float:
    out = _to_float(value)
    return out if math.isfinite(out) else math.inf


def _finite(value: Any) -> bool:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return f == f and f not in (float("inf"), float("-inf"))


__all__ = [
    "append_trial_record",
    "load_trial_records",
    "pareto_front",
    "reduce_trial_dirs",
    "select_final_candidates",
    "setting_sha256",
    "spec_sha256",
    "trial_record",
    "write_trial_dir",
]
