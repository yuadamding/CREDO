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


def setting_sha256(spec: CREDOTrialSpec) -> str:
    """Hash a setting across folds/seeds by excluding stochastic split identity."""
    payload_dict = dataclasses.asdict(spec)
    payload_dict.pop("seed", None)
    payload_dict.pop("fold_id", None)
    payload = json.dumps(payload_dict, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def trial_record(result: CREDOTrialResult) -> dict[str, Any]:
    """Flatten a :class:`CREDOTrialResult` into one JSON-serializable row."""
    spec = result.spec
    record: dict[str, Any] = {
        "schema_version": SEARCH_SCHEMA_VERSION,
        "spec_sha256": spec_sha256(spec),
        "setting_sha256": setting_sha256(spec),
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
    record.update({f"constraint.{k}": v for k, v in result.constraints.items()})
    return record


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
    group_by: tuple[str, ...] = ("setting_sha256",),
    aggregate_by: tuple[str, ...] = ("spec.fold_id", "spec.seed"),
    objectives: list[str] | None = None,
    sort_by: str | None = None,
    min_folds: int | None = None,
    min_seeds: int | None = None,
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
    group_keys = tuple(_record_key(name) for name in group_by)
    aggregate_keys = tuple(_record_key(name) for name in aggregate_by)
    objective_names = objectives or ["endpoint_geom_mass", "mass_error", "gpu_seconds"]
    objective_keys = [_objective_key(name) for name in objective_names]

    rows = list(records)
    if require_feasible:
        rows = [r for r in rows if bool(r.get("feasible", False))]
    if require_heldout:
        rows = [r for r in rows if r.get("metric.validation_source") == "held_out"]
    if profile is SearchProfile.CLAIM_GRADE:
        rows = [r for r in rows if _record_claim_grade_ready(r)]
    elif profile is SearchProfile.PARETO_REFIT:
        rows = [r for r in rows if _record_pareto_refit_ready(r)]

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(name) for name in group_keys)
        grouped.setdefault(key, []).append(row)

    candidates: list[dict[str, Any]] = []
    skipped_for_coverage = False
    for key, group in grouped.items():
        folds = _distinct_values(group, aggregate_keys[0]) if len(aggregate_keys) >= 1 else set()
        seeds = _distinct_values(group, aggregate_keys[1]) if len(aggregate_keys) >= 2 else set()
        if len(folds) < fold_min or len(seeds) < seed_min:
            skipped_for_coverage = True
            continue
        aggregate = _aggregate_group(
            key=key,
            group_by=group_by,
            group=group,
            profile=profile,
            objective_keys=objective_keys,
            fold_count=len(folds),
            seed_count=len(seeds),
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
    profile: SearchProfile,
    objective_keys: list[str],
    fold_count: int,
    seed_count: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "profile": profile.value,
        "n_trials": len(group),
        "fold_count": fold_count,
        "seed_count": seed_count,
        "fold_pass_rate": sum(1.0 for row in group if row.get("feasible", False)) / max(len(group), 1),
        "worst_fold_ess": _min_finite(
            row.get("metric.min_ess_frac_over_time", row.get("metric.terminal_ess_frac_min"))
            for row in group
        ),
        "worst_fold_max_weight_frac": _max_finite(row.get("metric.max_weight_frac_mean") for row in group),
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


def _record_claim_grade_ready(record: dict[str, Any]) -> bool:
    return _record_pareto_refit_ready(record) and all(
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
