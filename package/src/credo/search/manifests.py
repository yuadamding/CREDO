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
from pathlib import Path
from typing import Any, Iterable

from .metrics import CREDOTrialResult
from .space import CREDOTrialSpec


def spec_sha256(spec: CREDOTrialSpec) -> str:
    """Deterministic content hash of a trial spec (for dedup / cache keys)."""
    payload = json.dumps(dataclasses.asdict(spec), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def trial_record(result: CREDOTrialResult) -> dict[str, Any]:
    """Flatten a :class:`CREDOTrialResult` into one JSON-serializable row."""
    spec = result.spec
    record: dict[str, Any] = {
        "spec_sha256": spec_sha256(spec),
        "run_dir": result.run_dir,
        "checkpoint_path": result.checkpoint_path,
        "pruner_score": result.pruner_score,
        "feasible": result.feasible,
    }
    record.update({f"spec.{k}": v for k, v in dataclasses.asdict(spec).items()})
    record.update({f"metric.{k}": v for k, v in dataclasses.asdict(result.metrics).items()})
    record.update({f"objective.{k}": v for k, v in result.objective_vector.items()})
    record.update({f"constraint.{k}": v for k, v in result.constraints.items()})
    return record


def append_trial_record(path: str | Path, result: CREDOTrialResult) -> Path:
    """Append one trial record as a JSONL line."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(trial_record(result), sort_keys=True, default=str))
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
    "spec_sha256",
    "trial_record",
]
