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
    include_physical_time: bool = False,
    perturbation_values: list[str | None] | None = None,
    latent_values: np.ndarray | None = None,
) -> None:
    obs = pd.DataFrame(
        {
            "perturbation_id": perturbation_values or ["ctrl", "gene_a"],
            **({"time_label": ["t0", "t1"]} if include_time else {}),
            **({"sample_id": ["s0", "s1"]} if include_sample else {}),
            **({"physical_time": [0.0, 1.0]} if include_physical_time else {}),
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
    assert report["n_cells"] == 2
    assert report["n_genes"] == 3
    assert report["n_perturbations"] == 2
    assert report["n_time_labels"] == 2
    assert report["latent_dim"] == 2
    assert report["latent_values_checked_mode"] == "full"


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


def test_validate_anndata_schema_strict_trajectory_requires_physical_time(tmp_path) -> None:
    path = tmp_path / "tiny_strict_trajectory.h5ad"
    _write_adata(path, include_sample=True)

    report = validate_anndata_schema(path, schema="trajectory", strict=True)

    assert report["ok"] is False
    assert report["strict"] is True
    assert report["obs_columns_missing"] == ["physical_time"]


def test_validate_anndata_schema_flags_missing_and_empty_required_values(tmp_path) -> None:
    path = tmp_path / "tiny_empty_required.h5ad"
    _write_adata(path, perturbation_values=["ctrl", ""])

    report = validate_anndata_schema(path)
    strict_report = validate_anndata_schema(path, strict=True)

    assert report["ok"] is True
    assert report["required_columns_non_empty"] is False
    assert report["obs_columns_empty_counts"]["perturbation_id"] == 1
    assert strict_report["ok"] is False
    assert any("empty values" in error for error in strict_report["errors"])


def test_validate_anndata_schema_custom_schema_uses_supplied_columns_only(tmp_path) -> None:
    path = tmp_path / "tiny_custom.h5ad"
    obs = pd.DataFrame({"target_gene": ["ctrl", "gene_a"]}, index=["cell_0", "cell_1"])
    data = ad.AnnData(X=np.ones((2, 3), dtype=np.float32), obs=obs)
    data.obsm["X_pca"] = np.ones((2, 2), dtype=np.float32)
    data.write_h5ad(path)

    report = validate_anndata_schema(path, schema="custom", obs_columns=["target_gene"])

    assert report["ok"] is True
    assert report["obs_columns_required"] == ["target_gene"]


def test_validate_anndata_schema_accepts_single_obs_column_string(tmp_path) -> None:
    path = tmp_path / "tiny_custom_string.h5ad"
    obs = pd.DataFrame({"target_gene": ["ctrl", "gene_a"]}, index=["cell_0", "cell_1"])
    data = ad.AnnData(X=np.ones((2, 3), dtype=np.float32), obs=obs)
    data.obsm["X_pca"] = np.ones((2, 2), dtype=np.float32)
    data.write_h5ad(path)

    report = validate_anndata_schema(path, schema="custom", obs_columns="target_gene")

    assert report["ok"] is True
    assert report["obs_columns_required"] == ["target_gene"]


def test_validate_anndata_schema_adds_custom_columns_to_schema(tmp_path) -> None:
    path = tmp_path / "tiny_additive.h5ad"
    _write_adata(path)

    report = validate_anndata_schema(path, schema="minimal", obs_columns=["target_gene"])

    assert report["ok"] is False
    assert report["obs_columns_required"] == ["perturbation_id", "time_label", "target_gene"]
    assert report["obs_columns_missing"] == ["target_gene"]


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
    _write_adata(path, include_sample=True, include_physical_time=True)

    exit_code = validate_data_main([
        "--data-path",
        str(path),
        "--schema",
        "trajectory",
        "--strict",
        "--json",
    ])
    captured = capsys.readouterr()

    assert exit_code == 0
    report = json.loads(captured.out)
    assert report["ok"] is True
    assert report["schema"] == "trajectory"
