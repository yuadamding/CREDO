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
from .space import CREDOTrialSpec

# Bump when the flattened trial-record schema changes (field names/prefixes).
SEARCH_SCHEMA_VERSION = "credo.search.v1"


def spec_sha256(spec: CREDOTrialSpec) -> str:
    """Deterministic content hash of a trial spec (for dedup / cache keys)."""
    payload = json.dumps(dataclasses.asdict(spec), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def trial_record(result: CREDOTrialResult) -> dict[str, Any]:
    """Flatten a :class:`CREDOTrialResult` into one JSON-serializable row."""
    spec = result.spec
    record: dict[str, Any] = {
        "schema_version": SEARCH_SCHEMA_VERSION,
        "spec_sha256": spec_sha256(spec),
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
    trial_dir.mkdir(parents=True, exist_ok=True)
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


def _dominates(a: dict[str, Any], b: dict[str, Any], objectives: list[str]) -> bool:
    no_worse = all(float(a[o]) <= float(b[o]) for o in objectives)
    strictly_better = any(float(a[o]) < float(b[o]) for o in objectives)
    return no_worse and strictly_better


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
    "spec_sha256",
    "trial_record",
    "write_trial_dir",
]
