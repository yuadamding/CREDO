from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from credo.cli.validate_data import main as validate_data_main
from credo.data.schema import validate_anndata_schema


pytestmark = pytest.mark.unit


def _write_adata(path, *, include_latent: bool = True, include_time: bool = True) -> None:
    obs = pd.DataFrame(
        {
            "perturbation_id": ["ctrl", "gene_a"],
            **({"time_label": ["t0", "t1"]} if include_time else {}),
        },
        index=["cell_0", "cell_1"],
    )
    data = ad.AnnData(X=np.ones((2, 3), dtype=np.float32), obs=obs)
    if include_latent:
        data.obsm["X_pca"] = np.ones((2, 2), dtype=np.float32)
    data.write_h5ad(path)


def test_validate_anndata_schema_accepts_minimal_credo_input(tmp_path) -> None:
    path = tmp_path / "tiny.h5ad"
    _write_adata(path)

    report = validate_anndata_schema(path)

    assert report["ok"] is True
    assert report["shape"] == [2, 3]
    assert report["obs_columns_missing"] == []


def test_validate_anndata_schema_reports_missing_contract_fields(tmp_path) -> None:
    path = tmp_path / "tiny_missing.h5ad"
    _write_adata(path, include_latent=False, include_time=False)

    report = validate_anndata_schema(path)

    assert report["ok"] is False
    assert report["obs_columns_missing"] == ["time_label"]
    assert any("X_pca" in error for error in report["errors"])


def test_validate_data_cli_emits_json_report(tmp_path, capsys) -> None:
    path = tmp_path / "tiny.h5ad"
    _write_adata(path)

    exit_code = validate_data_main(["--data-path", str(path), "--json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    report = json.loads(captured.out)
    assert report["ok"] is True
