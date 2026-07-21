"""Lightweight AnnData schema validation for CREDO inputs."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal, Mapping

import anndata as ad
import numpy as np
import pandas as pd


DEFAULT_OBS_COLUMNS = ("perturbation_id", "time_label")
SCHEMA_OBS_COLUMNS = {
    "custom": (),
    "minimal": DEFAULT_OBS_COLUMNS,
    "endpoint": DEFAULT_OBS_COLUMNS,
    "single_time": ("cell_id", "is_control"),
    "trajectory": ("perturbation_id", "time_label", "sample_id"),
}
STRICT_SCHEMA_OBS_COLUMNS = {
    "custom": (),
    "minimal": DEFAULT_OBS_COLUMNS,
    "endpoint": ("perturbation_id", "time_label", "sample_id"),
    "single_time": ("cell_id", "is_control"),
    "trajectory": ("perturbation_id", "time_label", "sample_id", "physical_time"),
}


SchemaName = Literal["custom", "minimal", "endpoint", "single_time", "trajectory"]


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
    column_map: Mapping[str, str | None] | None = None,
    strict: bool = False,
    max_latent_values_to_check: int = 10_000_000,
) -> dict[str, object]:
    """Validate the package-level AnnData contract used by CREDO loaders."""
    if schema not in SCHEMA_OBS_COLUMNS:
        raise ValueError(f"schema must be one of {sorted(SCHEMA_OBS_COLUMNS)}.")
    column_map = dict(column_map or {})
    if schema == "single_time":
        cell_col = column_map.get("cell_id") or "cell_id"
        control_col = column_map.get("control") or "is_control"
        base_columns = (cell_col, control_col)
    else:
        base_columns = STRICT_SCHEMA_OBS_COLUMNS[schema] if strict else SCHEMA_OBS_COLUMNS[schema]
    required_obs_columns = _dedupe_columns([*base_columns, *_normalise_columns(obs_columns)])
    path = Path(data_path)
    report: dict[str, object] = {
        "ok": True,
        "path": str(path),
        "schema": schema,
        "schema_version": 1,
        "strict": bool(strict),
        "column_map": column_map,
        "shape": None,
        "n_cells": None,
        "n_genes": None,
        "n_perturbations": None,
        "n_time_labels": None,
        "time_label_counts": {},
        "perturbation_counts": {},
        "n_controls": None,
        "n_non_controls": None,
        "control_column_valid": None,
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
        if data.n_obs == 0:
            errors.append("AnnData must have non-zero observations.")
        if data.n_vars == 0 and not latent_key:
            errors.append(
                "AnnData with zero variables requires a latent embedding key."
            )
        report["obs_index_unique"] = bool(data.obs_names.is_unique)
        if not data.obs_names.is_unique:
            errors.append("AnnData observation index must be unique.")
        missing = [column for column in required_obs_columns if column not in data.obs]
        report["obs_columns_missing"] = missing
        if missing:
            errors.append("missing obs columns: " + ", ".join(missing))
        if schema == "single_time":
            perturbation_col = column_map.get("perturbation") or "perturbation_id"
            guide_col = column_map.get("guide") or "guide_id"
            sample_col = column_map.get("sample") or "sample_id"
            batch_col = column_map.get("batch") or "batch_id"
            single_time_selected_columns = []
            if perturbation_col not in data.obs and guide_col not in data.obs:
                errors.append(
                    "single_time schema requires a perturbation or guide column "
                    f"({perturbation_col!r} or {guide_col!r})"
                )
            else:
                single_time_selected_columns.append(
                    perturbation_col if perturbation_col in data.obs else guide_col,
                )
            if sample_col not in data.obs and batch_col not in data.obs:
                errors.append(
                    "single_time schema requires a sample or batch column "
                    f"({sample_col!r} or {batch_col!r})"
                )
            else:
                single_time_selected_columns.append(
                    sample_col if sample_col in data.obs else batch_col,
                )
        else:
            perturbation_col = "perturbation_id"
            guide_col = "guide_id"
            sample_col = "sample_id"
            batch_col = "batch_id"
            control_col = "is_control"
            single_time_selected_columns = []
        effective_required_obs_columns = _dedupe_columns([*required_obs_columns, *single_time_selected_columns])
        report["obs_columns_required"] = effective_required_obs_columns
        present_required = _dedupe_columns(
            [
                *[column for column in effective_required_obs_columns if column in data.obs],
                *single_time_selected_columns,
            ],
        )
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
        if perturbation_col in data.obs:
            perturbation_counts = data.obs[perturbation_col].astype(str).value_counts(dropna=False)
            report["n_perturbations"] = int(len(perturbation_counts))
            report["perturbation_counts"] = {str(key): int(value) for key, value in perturbation_counts.items()}
        elif schema == "single_time" and guide_col in data.obs:
            guide_counts = data.obs[guide_col].astype(str).value_counts(dropna=False)
            report["n_perturbations"] = int(len(guide_counts))
            report["perturbation_counts"] = {str(key): int(value) for key, value in guide_counts.items()}
        if schema == "single_time" and control_col in data.obs:
            control_values = data.obs[control_col]
            if pd.api.types.is_bool_dtype(control_values):
                control_mask = control_values.to_numpy(dtype=bool)
                report["control_column_valid"] = True
            else:
                normalized = control_values.astype(str).str.strip().str.lower()
                valid = normalized.isin({"true", "false", "1", "0", "yes", "no"})
                report["control_column_valid"] = bool(valid.all())
                if not bool(valid.all()):
                    errors.append(f"single_time obs[{control_col!r}] must be boolean-like")
                control_mask = normalized.isin({"true", "1", "yes"}).to_numpy()
            n_controls = int(control_mask.sum())
            report["n_controls"] = n_controls
            report["n_non_controls"] = int(len(control_mask) - n_controls)
            if n_controls == 0:
                errors.append("single_time schema requires at least one control cell")
            if len(control_mask) - n_controls == 0:
                errors.append("single_time schema requires at least one non-control cell")
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
