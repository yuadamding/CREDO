"""Lightweight AnnData schema validation for CREDO inputs."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import anndata as ad


DEFAULT_OBS_COLUMNS = ("perturbation_id", "time_label")


def validate_anndata_schema(
    data_path: str | Path,
    *,
    latent_key: str = "X_pca",
    obs_columns: Iterable[str] = DEFAULT_OBS_COLUMNS,
) -> dict[str, object]:
    """Validate the package-level AnnData contract used by CREDO loaders."""
    path = Path(data_path)
    report: dict[str, object] = {
        "ok": True,
        "path": str(path),
        "shape": None,
        "latent_key": latent_key,
        "obs_columns_required": list(obs_columns),
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
        missing = [column for column in obs_columns if column not in data.obs]
        report["obs_columns_missing"] = missing
        if missing:
            errors.append("missing obs columns: " + ", ".join(missing))
        obsm_keys = list(data.obsm.keys())
        report["obsm_keys"] = obsm_keys
        if latent_key and latent_key not in data.obsm:
            errors.append(f"missing latent embedding in obsm: {latent_key}")
    finally:
        data.file.close()

    report["ok"] = not errors
    report["errors"] = errors
    return report


__all__ = ["DEFAULT_OBS_COLUMNS", "validate_anndata_schema"]
