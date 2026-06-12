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


def _write_single_time_input(path: Path) -> None:
    obs = pd.DataFrame(
        {
            "cell_id": [f"c{i}" for i in range(8)],
            "perturbation_id": ["ctrl", "ctrl", "gene_a", "gene_a", "ctrl", "ctrl", "gene_a", "gene_a"],
            "guide_id": ["ctrl_g1", "ctrl_g1", "ga_g1", "ga_g1", "ctrl_g1", "ctrl_g1", "ga_g2", "ga_g2"],
            "target_gene": ["ctrl", "ctrl", "gene_a", "gene_a", "ctrl", "ctrl", "gene_a", "gene_a"],
            "is_control": [True, True, False, False, True, True, False, False],
            "sample_id": ["s1", "s1", "s1", "s1", "s2", "s2", "s2", "s2"],
        },
        index=[f"cell_{i}" for i in range(8)],
    )
    data = ad.AnnData(X=np.ones((8, 3), dtype=np.float32), obs=obs)
    data.obsm["X_pca"] = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [1.0, 0.0],
            [1.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
            [0.0, 1.0],
            [0.0, 1.1],
        ],
        dtype=np.float32,
    )
    data.write_h5ad(path)


def test_single_time_runner_default_ecology_writes_effect_outputs(tmp_path: Path) -> None:
    data_path = tmp_path / "single_time.h5ad"
    out_dir = tmp_path / "out"
    _write_single_time_input(data_path)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "runners" / "run_credo_single_time.py"),
            "--data-path",
            str(data_path),
            "--output-dir",
            str(out_dir),
            "--latent-key",
            "X_pca",
            "--perturbation-col",
            "perturbation_id",
            "--guide-col",
            "guide_id",
            "--target-gene-col",
            "target_gene",
            "--control-col",
            "is_control",
            "--sample-col",
            "sample_id",
            "--embedding-level",
            "target_gene",
            "--view-level",
            "view",
            "--strict-data-schema",
            "--effect-vector-components",
            "delta_log_mass,latent_mean_shift,latent_variance_shift",
            "--epochs",
            "1",
            "--n-particles",
            "4",
            "--n-steps",
            "1",
            "--hidden-dim",
            "8",
            "--depth",
            "1",
            "--n-programs",
            "2",
            "--mediator-dim",
            "1",
            "--embedding-dim",
            "2",
            "--lambda-weak",
            "0",
            "--lambda-reg-net",
            "0",
            "--lambda-reg-diffusion",
            "0",
            "--lambda-reg-embed",
            "0",
        ],
        cwd=ROOT,
        env=_env(),
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    claim = json.loads((out_dir / "single_time_claim_report.json").read_text())
    summary = json.loads((out_dir / "single_time_problem_summary.json").read_text())
    effects = pd.read_csv(out_dir / "single_time_effects.csv")
    endpoints = pd.read_csv(out_dir / "single_time_endpoint_metrics.csv")
    guide = pd.read_csv(out_dir / "single_time_guide_concordance.csv")
    controls = pd.read_csv(out_dir / "single_time_control_null.csv")

    assert claim["view_key_level"] == "sample_guide"
    assert claim["effect_vector_components"] == [
        "delta_log_mass",
        "latent_mean_shift",
        "latent_variance_shift",
    ]
    assert summary["view_key_level"] == "sample_guide"
    assert summary["effect_vector_components"] == [
        "delta_log_mass",
        "latent_mean_shift",
        "latent_variance_shift",
    ]
    assert set(effects["view_id"]) == {"s1::ctrl_g1", "s2::ctrl_g1", "s1::ga_g1", "s2::ga_g2"}
    assert {"delta_log_mass", "latent_mean_shift_norm", "latent_variance_shift_norm"} <= set(effects.columns)
    assert {"endpoint_sinkhorn", "mass_error", "endpoint_geom_mass"} <= set(endpoints.columns)
    assert set(guide["target_gene"]) == {"gene_a"}
    assert set(controls["is_control"]) == {True}
