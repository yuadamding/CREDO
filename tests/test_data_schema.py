from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from credo.cli.validate_data import main as validate_data_main
from credo.data.schema import validate_anndata_schema


pytestmark = pytest.mark.unit


def _write_adata(
    path,
    *,
    include_latent: bool = True,
    include_time: bool = True,
    include_sample: bool = False,
    latent_values: np.ndarray | None = None,
) -> None:
    obs = pd.DataFrame(
        {
            "perturbation_id": ["ctrl", "gene_a"],
            **({"time_label": ["t0", "t1"]} if include_time else {}),
            **({"sample_id": ["s0", "s1"]} if include_sample else {}),
        },
        index=["cell_0", "cell_1"],
    )
    data = ad.AnnData(X=np.ones((2, 3), dtype=np.float32), obs=obs)
    if include_latent:
        data.obsm["X_pca"] = (
            latent_values
            if latent_values is not None
            else np.ones((2, 2), dtype=np.float32)
        )
    data.write_h5ad(path)


def test_validate_anndata_schema_accepts_minimal_credo_input(tmp_path) -> None:
    path = tmp_path / "tiny.h5ad"
    _write_adata(path)

    report = validate_anndata_schema(path)

    assert report["ok"] is True
    assert report["schema"] == "minimal"
    assert report["shape"] == [2, 3]
    assert report["latent_shape"] == [2, 2]
    assert report["latent_row_count_matches_obs"] is True
    assert report["latent_values_checked"] is True
    assert report["latent_values_finite"] is True
    assert report["obs_index_unique"] is True
    assert report["obs_columns_missing"] == []


def test_validate_anndata_schema_reports_missing_contract_fields(tmp_path) -> None:
    path = tmp_path / "tiny_missing.h5ad"
    _write_adata(path, include_latent=False, include_time=False)

    report = validate_anndata_schema(path)

    assert report["ok"] is False
    assert report["obs_columns_missing"] == ["time_label"]
    assert any("X_pca" in error for error in report["errors"])


def test_validate_anndata_schema_trajectory_schema_requires_sample_id(tmp_path) -> None:
    path = tmp_path / "tiny_endpoint.h5ad"
    _write_adata(path)

    report = validate_anndata_schema(path, schema="trajectory")

    assert report["ok"] is False
    assert report["schema"] == "trajectory"
    assert report["obs_columns_missing"] == ["sample_id"]


def test_validate_anndata_schema_trajectory_schema_accepts_sample_id(tmp_path) -> None:
    path = tmp_path / "tiny_trajectory.h5ad"
    _write_adata(path, include_sample=True)

    report = validate_anndata_schema(path, schema="trajectory")

    assert report["ok"] is True
    assert report["obs_columns_missing"] == []


def test_validate_anndata_schema_rejects_nonfinite_latent_values(tmp_path) -> None:
    path = tmp_path / "tiny_bad_latent.h5ad"
    latent = np.asarray([[1.0, np.nan], [2.0, 3.0]], dtype=np.float32)
    _write_adata(path, latent_values=latent)

    report = validate_anndata_schema(path)

    assert report["ok"] is False
    assert report["latent_values_finite"] is False
    assert any("non-finite" in error for error in report["errors"])


def test_validate_data_cli_emits_json_report(tmp_path, capsys) -> None:
    path = tmp_path / "tiny.h5ad"
    _write_adata(path, include_sample=True)

    exit_code = validate_data_main(["--data-path", str(path), "--schema", "trajectory", "--json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    report = json.loads(captured.out)
    assert report["ok"] is True
    assert report["schema"] == "trajectory"
