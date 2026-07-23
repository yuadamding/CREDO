"""One recipe-neutral checkpoint evaluation facade."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

COMMON_PREDICTION_COLUMNS = (
    "run_id",
    "recipe_id",
    "recipe_version",
    "representation_id",
    "series_id",
    "observation_id",
    "checkpoint_id",
    "predicted_log_abundance",
    "observed_log_abundance",
)
COMMON_METRIC_COLUMNS = (
    "run_id",
    "recipe_id",
    "recipe_version",
    "series_id",
    "observation_id",
    "checkpoint_id",
    "metric_name",
    "value",
    "unit",
)
COMMON_DIAGNOSTIC_COLUMNS = (
    "run_id",
    "recipe_id",
    "recipe_version",
    "series_id",
    "observation_id",
    "checkpoint_id",
    "diagnostic_name",
    "value",
    "unit",
)


@dataclass(frozen=True)
class EvaluationTables:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    diagnostics: pd.DataFrame


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
    required = {"recipe_id", "recipe_version", "representation_id", "split_id"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Evaluator omitted common metric columns: {sorted(missing)}")
    identifier_columns = (
        ["series_id", "checkpoint_id"]
        if {"series_id", "checkpoint_id"} <= set(metrics.columns)
        else ["measure_id", "time_label"]
    )
    if metrics.duplicated(identifier_columns).any():
        raise ValueError("Evaluation metrics must have one row per measure/checkpoint.")
    if "evaluation_particles" in metrics and (metrics["evaluation_particles"] < 2).any():
        raise ValueError("Evaluation metrics contain an invalid particle count.")
    if "integration_steps" in metrics and (metrics["integration_steps"] < 1).any():
        raise ValueError("Evaluation metrics contain invalid integration or seed provenance.")
    if "evaluation_seed" in metrics and (metrics["evaluation_seed"] < 0).any():
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
    frame["series_id"] = frame["measure_id"].astype(str)
    frame["checkpoint_id"] = frame["time_label"].astype(str)
    observation_map = run.data.metadata.get("observation_id_by_series_checkpoint", {})
    frame["observation_id"] = [
        observation_map.get(f"{series_id}\0{checkpoint_id}", f"{series_id}@{checkpoint_id}")
        for series_id, checkpoint_id in zip(frame["series_id"], frame["checkpoint_id"], strict=True)
    ]
    frame["evaluation_particles"] = int(particles)
    frame["integration_steps"] = int(len(run.grid) - 1)
    frame["evaluation_seed"] = int(seed)
    return frame


def evaluation_tables(
    run: Any,
    metrics: pd.DataFrame,
    *,
    run_id: str,
) -> EvaluationTables:
    """Split one recipe evaluation into stable predictions, metrics, and diagnostics."""
    frame = metrics.copy()
    if "series_id" not in frame:
        frame["series_id"] = frame["measure_id"].astype(str)
    if "checkpoint_id" not in frame:
        frame["checkpoint_id"] = frame["time_label"].astype(str)
    if "observation_id" not in frame:
        observation_map = getattr(run, "data", None)
        observation_map = (
            {}
            if observation_map is None
            else observation_map.metadata.get("observation_id_by_series_checkpoint", {})
        )
        frame["observation_id"] = [
            observation_map.get(f"{series_id}\0{checkpoint_id}", f"{series_id}@{checkpoint_id}")
            for series_id, checkpoint_id in zip(
                frame["series_id"], frame["checkpoint_id"], strict=True
            )
        ]
    identity = pd.DataFrame(
        {
            "run_id": run_id,
            "recipe_id": frame["recipe_id"],
            "recipe_version": frame["recipe_version"],
            "series_id": frame["series_id"],
            "observation_id": frame["observation_id"],
            "checkpoint_id": frame["checkpoint_id"],
        }
    )
    predictions = identity.copy()
    predictions.insert(3, "representation_id", frame["representation_id"].to_numpy())
    predictions["predicted_log_abundance"] = frame.get("predicted_log_mass", np.nan)
    predictions["observed_log_abundance"] = frame.get("observed_log_mass", np.nan)
    predictions = predictions.loc[:, COMMON_PREDICTION_COLUMNS]

    metric_specs = {
        "geometry": ("sinkhorn_divergence", "latent_squared_distance"),
        "log_mass_error": ("log_abundance_squared_error", "squared_log_abundance"),
    }
    metric_rows = []
    for source, (name, unit) in metric_specs.items():
        if source not in frame:
            continue
        values = identity.copy()
        values["metric_name"] = name
        values["value"] = pd.to_numeric(frame[source], errors="coerce")
        values["unit"] = unit
        metric_rows.append(values)
    common_metrics = (
        pd.concat(metric_rows, ignore_index=True)
        if metric_rows
        else pd.DataFrame(columns=COMMON_METRIC_COLUMNS)
    )
    common_metrics = common_metrics.loc[:, COMMON_METRIC_COLUMNS]

    diagnostic_specs = {
        "ess_fraction": ("particle_ess_fraction", "fraction"),
        "max_weight_fraction": ("particle_max_weight_fraction", "fraction"),
        "evaluation_particles": ("particle_count", "particles"),
        "integration_steps": ("integration_steps", "steps"),
        "evaluation_seed": ("evaluation_seed", "seed"),
    }
    diagnostic_rows = []
    for source, (name, unit) in diagnostic_specs.items():
        if source not in frame:
            continue
        values = identity.copy()
        values["diagnostic_name"] = name
        values["value"] = pd.to_numeric(frame[source], errors="coerce")
        values["unit"] = unit
        diagnostic_rows.append(values)
    diagnostics = (
        pd.concat(diagnostic_rows, ignore_index=True)
        if diagnostic_rows
        else pd.DataFrame(columns=COMMON_DIAGNOSTIC_COLUMNS)
    )
    diagnostics = diagnostics.loc[:, COMMON_DIAGNOSTIC_COLUMNS]
    return EvaluationTables(predictions, common_metrics, diagnostics)


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


__all__ = [
    "COMMON_DIAGNOSTIC_COLUMNS",
    "COMMON_METRIC_COLUMNS",
    "COMMON_PREDICTION_COLUMNS",
    "EvaluationTables",
    "evaluate",
    "evaluation_tables",
    "standardize_compact_metrics",
]
