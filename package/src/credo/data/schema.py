"""Lightweight AnnData schema validation for CREDO inputs."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

import anndata as ad
import numpy as np


DEFAULT_OBS_COLUMNS = ("perturbation_id", "time_label")
SCHEMA_OBS_COLUMNS = {
    "minimal": DEFAULT_OBS_COLUMNS,
    "endpoint": DEFAULT_OBS_COLUMNS,
    "trajectory": ("perturbation_id", "time_label", "sample_id"),
}


def validate_anndata_schema(
    data_path: str | Path,
    *,
    schema: Literal["minimal", "endpoint", "trajectory"] = "minimal",
    latent_key: str = "X_pca",
    obs_columns: Iterable[str] | None = None,
    max_latent_values_to_check: int = 10_000_000,
) -> dict[str, object]:
    """Validate the package-level AnnData contract used by CREDO loaders."""
    if schema not in SCHEMA_OBS_COLUMNS:
        raise ValueError(f"schema must be one of {sorted(SCHEMA_OBS_COLUMNS)}.")
    required_obs_columns = list(obs_columns) if obs_columns is not None else list(SCHEMA_OBS_COLUMNS[schema])
    path = Path(data_path)
    report: dict[str, object] = {
        "ok": True,
        "path": str(path),
        "schema": schema,
        "schema_version": 1,
        "shape": None,
        "latent_key": latent_key,
        "latent_shape": None,
        "latent_row_count_matches_obs": None,
        "latent_values_checked": False,
        "latent_values_finite": None,
        "obs_index_unique": None,
        "obs_columns_required": required_obs_columns,
        "obs_columns_missing": [],
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
        if data.n_obs == 0 or data.n_vars == 0:
            errors.append("AnnData must have non-zero observations and variables.")
        report["obs_index_unique"] = bool(data.obs_names.is_unique)
        if not data.obs_names.is_unique:
            errors.append("AnnData observation index must be unique.")
        missing = [column for column in required_obs_columns if column not in data.obs]
        report["obs_columns_missing"] = missing
        if missing:
            errors.append("missing obs columns: " + ", ".join(missing))
        obsm_keys = list(data.obsm.keys())
        report["obsm_keys"] = obsm_keys
        if latent_key and latent_key not in data.obsm:
            errors.append(f"missing latent embedding in obsm: {latent_key}")
        elif latent_key:
            latent = data.obsm[latent_key]
            latent_shape = [int(dim) for dim in latent.shape]
            report["latent_shape"] = latent_shape
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
                finite = bool(np.isfinite(latent_array).all())
                report["latent_values_finite"] = finite
                if not finite:
                    errors.append(f"latent embedding {latent_key} contains non-finite values")
    finally:
        data.file.close()

    report["ok"] = not errors
    report["errors"] = errors
    return report


__all__ = ["DEFAULT_OBS_COLUMNS", "SCHEMA_OBS_COLUMNS", "validate_anndata_schema"]
