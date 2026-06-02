"""Lightweight AnnData schema validation for CREDO inputs."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

import anndata as ad
import numpy as np
import pandas as pd


DEFAULT_OBS_COLUMNS = ("perturbation_id", "time_label")
SCHEMA_OBS_COLUMNS = {
    "custom": (),
    "minimal": DEFAULT_OBS_COLUMNS,
    "endpoint": DEFAULT_OBS_COLUMNS,
    "trajectory": ("perturbation_id", "time_label", "sample_id"),
}
STRICT_SCHEMA_OBS_COLUMNS = {
    "custom": (),
    "minimal": DEFAULT_OBS_COLUMNS,
    "endpoint": ("perturbation_id", "time_label", "sample_id"),
    "trajectory": ("perturbation_id", "time_label", "sample_id", "physical_time"),
}


SchemaName = Literal["custom", "minimal", "endpoint", "trajectory"]


def _dedupe_columns(columns: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for column in columns:
        if column not in seen:
            out.append(column)
            seen.add(column)
    return out


def _column_null_count(series: pd.Series) -> int:
    return int(series.isna().sum())


def _column_empty_count(series: pd.Series) -> int:
    non_null = series.dropna()
    if len(non_null) == 0:
        return 0
    return int(non_null.astype(str).str.strip().eq("").sum())


def _normalise_columns(columns: Iterable[str] | str | None) -> list[str]:
    if columns is None:
        return []
    if isinstance(columns, str):
        return [columns]
    return list(columns)


def validate_anndata_schema(
    data_path: str | Path,
    *,
    schema: SchemaName = "minimal",
    latent_key: str = "X_pca",
    obs_columns: Iterable[str] | str | None = None,
    strict: bool = False,
    max_latent_values_to_check: int = 10_000_000,
) -> dict[str, object]:
    """Validate the package-level AnnData contract used by CREDO loaders."""
    if schema not in SCHEMA_OBS_COLUMNS:
        raise ValueError(f"schema must be one of {sorted(SCHEMA_OBS_COLUMNS)}.")
    base_columns = STRICT_SCHEMA_OBS_COLUMNS[schema] if strict else SCHEMA_OBS_COLUMNS[schema]
    required_obs_columns = _dedupe_columns([*base_columns, *_normalise_columns(obs_columns)])
    path = Path(data_path)
    report: dict[str, object] = {
        "ok": True,
        "path": str(path),
        "schema": schema,
        "schema_version": 1,
        "strict": bool(strict),
        "shape": None,
        "n_cells": None,
        "n_genes": None,
        "n_perturbations": None,
        "n_time_labels": None,
        "time_label_counts": {},
        "perturbation_counts": {},
        "latent_key": latent_key,
        "latent_shape": None,
        "latent_dim": None,
        "latent_row_count_matches_obs": None,
        "latent_values_checked": False,
        "latent_values_checked_mode": "skipped",
        "latent_values_checked_count": 0,
        "latent_values_finite": None,
        "obs_index_unique": None,
        "obs_columns_required": required_obs_columns,
        "obs_columns_missing": [],
        "obs_columns_null_counts": {},
        "obs_columns_empty_counts": {},
        "required_columns_non_null": None,
        "required_columns_non_empty": None,
        "obsm_keys": [],
        "errors": [],
    }
    errors: list[str] = []
    if not path.exists():
        errors.append(f"data_path does not exist: {path}")
        report["ok"] = False
        report["errors"] = errors
        return report

    try:
        data = ad.read_h5ad(path, backed="r")
    except Exception as exc:  # pragma: no cover - exact backend errors vary
        errors.append(f"failed to read AnnData: {exc}")
        report["ok"] = False
        report["errors"] = errors
        return report

    try:
        report["shape"] = [int(data.n_obs), int(data.n_vars)]
        report["n_cells"] = int(data.n_obs)
        report["n_genes"] = int(data.n_vars)
        if data.n_obs == 0 or data.n_vars == 0:
            errors.append("AnnData must have non-zero observations and variables.")
        report["obs_index_unique"] = bool(data.obs_names.is_unique)
        if not data.obs_names.is_unique:
            errors.append("AnnData observation index must be unique.")
        missing = [column for column in required_obs_columns if column not in data.obs]
        report["obs_columns_missing"] = missing
        if missing:
            errors.append("missing obs columns: " + ", ".join(missing))
        present_required = [column for column in required_obs_columns if column in data.obs]
        null_counts = {column: _column_null_count(data.obs[column]) for column in present_required}
        empty_counts = {column: _column_empty_count(data.obs[column]) for column in present_required}
        report["obs_columns_null_counts"] = null_counts
        report["obs_columns_empty_counts"] = empty_counts
        non_null = all(count == 0 for count in null_counts.values())
        non_empty = all(count == 0 for count in empty_counts.values())
        report["required_columns_non_null"] = bool(non_null) if present_required else None
        report["required_columns_non_empty"] = bool(non_empty) if present_required else None
        for column, count in null_counts.items():
            if count:
                errors.append(f"obs column {column} contains {count} missing values")
        for column, count in empty_counts.items():
            if count and strict:
                errors.append(f"obs column {column} contains {count} empty values")
        if "perturbation_id" in data.obs:
            perturbation_counts = data.obs["perturbation_id"].astype(str).value_counts(dropna=False)
            report["n_perturbations"] = int(len(perturbation_counts))
            report["perturbation_counts"] = {str(key): int(value) for key, value in perturbation_counts.items()}
        if "time_label" in data.obs:
            time_counts = data.obs["time_label"].astype(str).value_counts(dropna=False)
            report["n_time_labels"] = int(len(time_counts))
            report["time_label_counts"] = {str(key): int(value) for key, value in time_counts.items()}
        obsm_keys = list(data.obsm.keys())
        report["obsm_keys"] = obsm_keys
        if latent_key and latent_key not in data.obsm:
            errors.append(f"missing latent embedding in obsm: {latent_key}")
        elif latent_key:
            latent = data.obsm[latent_key]
            latent_shape = [int(dim) for dim in latent.shape]
            report["latent_shape"] = latent_shape
            report["latent_dim"] = int(latent_shape[1]) if len(latent_shape) > 1 else None
            if len(latent_shape) != 2 or latent_shape[1] <= 0:
                errors.append(f"latent embedding {latent_key} must be a 2D array with positive width")
            row_count_matches = latent_shape[0] == int(data.n_obs)
            report["latent_row_count_matches_obs"] = bool(row_count_matches)
            if not row_count_matches:
                errors.append(
                    f"latent embedding {latent_key} has {latent_shape[0]} rows, "
                    f"expected {int(data.n_obs)}"
                )
            n_values = int(np.prod(latent_shape)) if latent_shape else 0
            if n_values <= int(max_latent_values_to_check):
                latent_array = np.asarray(latent)
                report["latent_values_checked"] = True
                report["latent_values_checked_mode"] = "full"
                report["latent_values_checked_count"] = n_values
                finite = bool(np.isfinite(latent_array).all())
                report["latent_values_finite"] = finite
                if not finite:
                    errors.append(f"latent embedding {latent_key} contains non-finite values")
    finally:
        data.file.close()

    report["ok"] = not errors
    report["errors"] = errors
    return report


__all__ = [
    "DEFAULT_OBS_COLUMNS",
    "SCHEMA_OBS_COLUMNS",
    "STRICT_SCHEMA_OBS_COLUMNS",
    "validate_anndata_schema",
]
