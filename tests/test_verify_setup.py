from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.runner


def _env() -> dict[str, str]:
    env = os.environ.copy()
    path = str(ROOT / "package" / "src")
    env["PYTHONPATH"] = path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def test_verify_setup_without_data_succeeds() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verify_setup.py"), "--json"],
        cwd=ROOT,
        env=_env(),
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(result.stdout)
    assert report["ok"] is True
    assert report["data"]["checked"] is False
    assert report["environment"]["required_imports"]["credo"]["ok"] is True
    assert report["environment"]["required_imports"]["credo"]["version"] == "2.0.1"


def test_verify_setup_check_data_requires_existing_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing.h5ad"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "verify_setup.py"),
            "--json",
            "--check-data",
            "--data-path",
            str(missing),
        ],
        cwd=ROOT,
        env=_env(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["ok"] is False
    assert report["data"]["checked"] is True
    assert "Missing data file" in report["data"]["error"]


def test_verify_setup_check_data_uses_credo_schema_validator(tmp_path: Path) -> None:
    path = tmp_path / "tiny_trajectory.h5ad"
    obs = pd.DataFrame(
        {
            "perturbation_id": ["ctrl", "gene_a"],
            "time_label": ["t0", "t1"],
            "sample_id": ["s0", "s1"],
            "physical_time": [0.0, 1.0],
        },
        index=["cell_0", "cell_1"],
    )
    data = ad.AnnData(X=np.ones((2, 3), dtype=np.float32), obs=obs)
    data.obsm["X_pca"] = np.ones((2, 2), dtype=np.float32)
    data.write_h5ad(path)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "verify_setup.py"),
            "--json",
            "--check-data",
            "--data-path",
            str(path),
            "--data-schema",
            "trajectory",
            "--strict-data-schema",
        ],
        cwd=ROOT,
        env=_env(),
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(result.stdout)
    assert report["ok"] is True
    assert report["data"]["checked"] is True
    assert report["data"]["schema"] == "trajectory"
    assert report["data"]["strict"] is True
    assert report["data"]["latent_row_count_matches_obs"] is True


def test_verify_setup_accepts_single_time_schema(tmp_path: Path) -> None:
    path = tmp_path / "tiny_single_time.h5ad"
    obs = pd.DataFrame(
        {
            "cell_id": ["c0", "c1", "c2", "c3"],
            "guide_id": ["ctrl_g1", "ctrl_g1", "ga_g1", "ga_g1"],
            "target_gene": ["ctrl", "ctrl", "gene_a", "gene_a"],
            "is_control": [True, True, False, False],
            "sample_id": ["s1", "s1", "s1", "s1"],
        },
        index=["cell_0", "cell_1", "cell_2", "cell_3"],
    )
    data = ad.AnnData(X=np.ones((4, 3), dtype=np.float32), obs=obs)
    data.obsm["X_pca"] = np.ones((4, 2), dtype=np.float32)
    data.write_h5ad(path)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "verify_setup.py"),
            "--json",
            "--check-data",
            "--data-path",
            str(path),
            "--data-schema",
            "single_time",
            "--strict-data-schema",
        ],
        cwd=ROOT,
        env=_env(),
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(result.stdout)
    assert report["ok"] is True
    assert report["data"]["schema"] == "single_time"
    assert report["data"]["n_controls"] == 2
    assert report["data"]["n_non_controls"] == 2


def test_verify_setup_single_time_schema_accepts_custom_columns(tmp_path: Path) -> None:
    path = tmp_path / "tiny_single_time_custom.h5ad"
    obs = pd.DataFrame(
        {
            "cell_id": ["c0", "c1", "c2", "c3"],
            "sgrna": ["ctrl_g1", "ctrl_g1", "ga_g1", "ga_g1"],
            "gene": ["ctrl", "ctrl", "gene_a", "gene_a"],
            "nontargeting_flag": [True, True, False, False],
            "donor": ["s1", "s1", "s1", "s1"],
        },
        index=["cell_0", "cell_1", "cell_2", "cell_3"],
    )
    data = ad.AnnData(X=np.ones((4, 3), dtype=np.float32), obs=obs)
    data.obsm["X_pca"] = np.ones((4, 2), dtype=np.float32)
    data.write_h5ad(path)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "verify_setup.py"),
            "--json",
            "--check-data",
            "--data-path",
            str(path),
            "--data-schema",
            "single_time",
            "--strict-data-schema",
            "--guide-col",
            "sgrna",
            "--target-gene-col",
            "gene",
            "--control-col",
            "nontargeting_flag",
            "--sample-col",
            "donor",
        ],
        cwd=ROOT,
        env=_env(),
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(result.stdout)
    assert report["ok"] is True
    assert report["data"]["column_map"]["control"] == "nontargeting_flag"
    assert "donor" in report["data"]["obs_columns_required"]
    assert report["data"]["obs_columns_empty_counts"]["donor"] == 0
    assert report["data"]["n_controls"] == 2
    assert report["data"]["n_non_controls"] == 2
