"""One recipe-neutral checkpoint evaluation facade."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd

COMMON_METRIC_COLUMNS = (
    "recipe_id",
    "recipe_version",
    "representation_id",
    "split_id",
    "measure_id",
    "time_label",
    "geometry",
    "log_mass_error",
    "predicted_log_mass",
    "observed_log_mass",
    "ess_fraction",
    "max_weight_fraction",
    "evaluation_particles",
    "integration_steps",
    "evaluation_seed",
)


def compact_split_id(run: Any) -> str:
    split_payload = {
        "strategy": run.validation_strategy,
        "source": run.validation_source,
        "train_measure_ids": list(run.train_measure_ids),
        "validation_measure_ids": list(run.validation_measure_ids),
        "train_time_labels": list(run.train_time_labels),
        "validation_time_labels": list(run.validation_time_labels),
    }
    split_hash = hashlib.sha256(
        json.dumps(split_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"compact-v3:{run.validation_strategy}:{split_hash[:12]}"


def _validate_common_metrics(metrics: Any) -> pd.DataFrame:
    if not isinstance(metrics, pd.DataFrame):
        raise TypeError("A CREDO evaluator must return a pandas DataFrame.")
    missing = set(COMMON_METRIC_COLUMNS) - set(metrics.columns)
    if missing:
        raise ValueError(f"Evaluator omitted common metric columns: {sorted(missing)}")
    if metrics.duplicated(["measure_id", "time_label"]).any():
        raise ValueError("Evaluation metrics must have one row per measure/checkpoint.")
    if (metrics["evaluation_particles"] < 2).any():
        raise ValueError("Evaluation metrics contain an invalid particle count.")
    if (metrics["integration_steps"] < 1).any() or (metrics["evaluation_seed"] < 0).any():
        raise ValueError("Evaluation metrics contain invalid integration or seed provenance.")
    return metrics


def standardize_compact_metrics(
    run: Any,
    metrics: pd.DataFrame,
    *,
    particles: int,
    seed: int,
) -> pd.DataFrame:
    frame = metrics.copy()
    frame.insert(0, "recipe_id", "credo.compact_sde_v3")
    frame.insert(1, "recipe_version", "3.0")
    frame.insert(2, "representation_id", run.data.representation.representation_id)
    frame.insert(3, "split_id", compact_split_id(run))
    frame["evaluation_particles"] = int(particles)
    frame["integration_steps"] = int(len(run.grid) - 1)
    frame["evaluation_seed"] = int(seed)
    return frame


def evaluate(
    run: Any,
    *,
    study: Any = None,
    split: Any = None,
    particles: int | None = None,
    seed: int | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Evaluate any first-class recipe through its runtime adapter."""
    require = getattr(run, "require", None)
    if callable(require):
        require("evaluate")
    method = getattr(run, "evaluate_runtime", None)
    if not callable(method):
        raise TypeError("Run does not implement the CREDO evaluation runtime contract.")
    if split is not None:
        current = getattr(run, "split", None)
        if current is not None and split != current:
            raise ValueError("Requested split disagrees with the run's immutable split contract.")
    metrics = method(study=study, particles=particles, seed=seed, **kwargs)
    return _validate_common_metrics(metrics)


__all__ = ["COMMON_METRIC_COLUMNS", "evaluate", "standardize_compact_metrics"]
