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
import torch

from runners.run_credo_single_time import _weight_diagnostics


ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.runner


def test_weight_diagnostic_ess_fraction_is_bounded() -> None:
    for n_particles in range(1, 65):
        diagnostics = _weight_diagnostics(
            torch.zeros(n_particles, dtype=torch.float32),
            prefix="uniform",
        )
        assert 0.0 <= diagnostics["uniform_ess_frac"] <= 1.0


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


def _write_custom_single_time_input(path: Path) -> None:
    obs = pd.DataFrame(
        {
            "cell_id": [f"c{i}" for i in range(8)],
            "condition": ["ctrl", "ctrl", "gene_a", "gene_a", "ctrl", "ctrl", "gene_a", "gene_a"],
            "sgrna": ["ctrl_g1", "ctrl_g1", "ga_g1", "ga_g1", "ctrl_g1", "ctrl_g1", "ga_g2", "ga_g2"],
            "gene": ["ctrl", "ctrl", "gene_a", "gene_a", "ctrl", "ctrl", "gene_a", "gene_a"],
            "nontargeting_flag": [True, True, False, False, True, True, False, False],
            "donor": ["s1", "s1", "s1", "s1", "s2", "s2", "s2", "s2"],
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
            "--context-sampling",
            "epoch_resample",
            "--context-gradient-mode",
            "detached_cache",
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
    control_summary = pd.read_csv(out_dir / "single_time_control_null_summary.csv")
    mean_shift = pd.read_csv(out_dir / "single_time_latent_mean_shift_by_dim.csv")
    variance_shift = pd.read_csv(out_dir / "single_time_latent_variance_shift_by_dim.csv")
    resolved = json.loads((out_dir / "single_time_resolved_config.json").read_text())

    assert claim["view_key_level"] == "sample_guide"
    assert claim["context_sampling"] == "epoch_resample"
    assert claim["context_gradient_mode"] == "detached_cache"
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
    assert resolved["single_time"]["context_sampling"] == "epoch_resample"
    assert resolved["output_dir"] == str(out_dir)
    assert resolved["model"]["embedding_dim"] == 2
    assert resolved["model"]["n_programs"] == 2
    assert resolved["model"]["mediator_dim"] == 1
    assert resolved["model"]["hidden_dim"] == 8
    assert resolved["model"]["depth"] == 1
    assert (out_dir / "single_time_command.txt").read_text().startswith("python runners/run_credo_single_time.py")
    assert (out_dir / "single_time_git_sha.txt").read_text().strip()
    assert set(effects["view_id"]) == {"s1::ctrl_g1", "s2::ctrl_g1", "s1::ga_g1", "s2::ga_g2"}
    assert {
        "diagnostic_delta_log_mass",
        "abundance_delta_log_mass_claimable",
        "training_view_level",
        "report_view_level",
        "report_is_posthoc_view_level",
        "training_context_sampling",
        "report_context_sampling",
        "terminal_ess_frac",
        "source_ess_frac",
        "factual_terminal_ess_frac",
        "reference_terminal_ess_frac",
        "weight_diagnostic_status",
        "max_weight_frac",
        "logw_range",
        "factual_max_weight_frac",
        "reference_max_weight_frac",
        "factual_logw_range",
        "reference_logw_range",
        "delta_log_mass_semantics",
        "delta_mass_semantics",
        "control_null_z_diagnostic_delta_log_mass",
        "control_null_abs_p95_exceeded_diagnostic_delta_log_mass",
        "latent_mean_shift_norm",
        "latent_variance_shift_norm",
    } <= set(effects.columns)
    assert set(effects["training_view_level"]) == {"view"}
    assert set(effects["report_view_level"]) == {"view"}
    assert set(effects["report_is_posthoc_view_level"]) == {False}
    assert set(effects["training_context_sampling"]) == {"epoch_resample"}
    assert set(effects["report_context_sampling"]) == {"epoch_resample"}
    assert set(effects["delta_log_mass_semantics"]) == {
        "deprecated_alias_for_diagnostic_finite_measure_weight_effect",
    }
    assert effects["abundance_delta_log_mass_claimable"].isna().all()
    assert effects["terminal_ess_frac"].between(0.0, 1.0).all()
    assert effects["source_ess_frac"].between(0.0, 1.0).all()
    assert effects["factual_terminal_ess_frac"].between(0.0, 1.0).all()
    assert effects["reference_terminal_ess_frac"].between(0.0, 1.0).all()
    assert set(effects["weight_diagnostic_status"]) <= {"ok", "warn", "fail"}
    assert {
        "endpoint_sinkhorn",
        "mass_error",
        "log_mass_residual",
        "endpoint_geom_mass",
        "factual_endpoint_sinkhorn",
        "reference_endpoint_sinkhorn",
        "delta_endpoint_sinkhorn_ref_minus_fact",
        "factual_mass_error",
        "factual_log_mass_residual",
        "reference_mass_error",
        "reference_log_mass_residual",
        "delta_mass_error_ref_minus_fact",
        "delta_log_mass_residual_ref_minus_fact",
        "factual_target_mean_shift_norm",
        "reference_target_mean_shift_norm",
    } <= set(endpoints.columns)
    assert np.allclose(endpoints["endpoint_sinkhorn"], endpoints["factual_endpoint_sinkhorn"])
    assert np.allclose(endpoints["mass_error"], endpoints["factual_mass_error"])
    assert (endpoints["mass_error"] >= 0).all()
    assert (endpoints["factual_mass_error"] >= 0).all()
    assert (endpoints["reference_mass_error"] >= 0).all()
    assert np.allclose(endpoints["factual_mass_error"], endpoints["factual_log_mass_residual"].abs())
    assert np.allclose(endpoints["reference_mass_error"], endpoints["reference_log_mass_residual"].abs())
    assert np.allclose(
        endpoints["delta_mass_error_ref_minus_fact"],
        endpoints["reference_mass_error"] - endpoints["factual_mass_error"],
    )
    assert np.allclose(
        endpoints["delta_log_mass_residual_ref_minus_fact"],
        endpoints["reference_log_mass_residual"] - endpoints["factual_log_mass_residual"],
    )
    assert set(guide["target_gene"]) == {"gene_a"}
    assert set(guide["guide_concordance_evaluable"]) == {True}
    assert set(guide["guide_concordance_claimable"]) == {True}
    assert set(guide["guide_concordance_is_posthoc"]) == {False}
    assert {"n_views", "n_guides", "n_samples", "training_view_level", "report_view_level"} <= set(guide.columns)
    assert set(controls["is_control"]) == {True}
    assert set(control_summary["metric"]) == {
        "diagnostic_delta_log_mass",
        "latent_mean_shift_norm",
        "latent_variance_shift_norm",
    }
    assert set(mean_shift["view_id"]) == set(effects["view_id"])
    assert set(variance_shift["view_id"]) == set(effects["view_id"])


def test_single_time_runner_strict_schema_uses_custom_column_map(tmp_path: Path) -> None:
    data_path = tmp_path / "single_time_custom.h5ad"
    out_dir = tmp_path / "out_custom"
    _write_custom_single_time_input(data_path)

    subprocess.run(
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
            "condition",
            "--guide-col",
            "sgrna",
            "--target-gene-col",
            "gene",
            "--control-col",
            "nontargeting_flag",
            "--sample-col",
            "donor",
            "--embedding-level",
            "target_gene",
            "--view-level",
            "embedding",
            "--strict-data-schema",
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

    effects = pd.read_csv(out_dir / "single_time_effects.csv")
    guide = pd.read_csv(out_dir / "single_time_guide_concordance.csv")
    assert set(effects["training_view_level"]) == {"embedding"}
    assert set(effects["report_view_level"]) == {"view"}
    assert set(effects["report_is_posthoc_view_level"]) == {True}
    assert set(effects["guide_id"]) == {"ctrl_g1", "ga_g1", "ga_g2"}
    assert set(guide["guide_concordance_claimable"]) == {False}
    assert set(guide["guide_concordance_is_posthoc"]) == {True}
