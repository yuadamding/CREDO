from __future__ import annotations

import json

import numpy as np
import pandas as pd
import anndata as ad
import pytest

from runners.run_credo_trajectory import build_study_from_anndata, main as trajectory_main, parse_args


def test_lps_trajectory_runner_smoke(tmp_path) -> None:
    rows = []
    latent = []
    rng = np.random.default_rng(9)
    for sample in ["D1"]:
        for pid, is_control in [("ctrl", True), ("LPS__mono", False)]:
            for time_i, (label, physical) in enumerate([("90m", 1.5), ("6h", 6.0), ("10h", 10.0)]):
                for cell_i in range(3):
                    rows.append(
                        {
                            "cell_id": f"{sample}_{pid}_{label}_{cell_i}",
                            "time_label": label,
                            "physical_time": physical,
                            "sample_id": sample,
                            "perturbation_id": pid,
                            "is_control": is_control,
                            "mass_value": 1.0,
                        }
                    )
                    latent.append(rng.normal(loc=float(time_i) if not is_control else 0.0, scale=0.03, size=2))
    obs = pd.DataFrame(rows).set_index("cell_id", drop=False)
    adata = ad.AnnData(X=np.ones((len(obs), 4), dtype=np.float32), obs=obs)
    adata.obsm["X_pca"] = np.asarray(latent, dtype=np.float32)
    data_path = tmp_path / "credo_lps_90m_6h_10h_celltype.h5ad"
    adata.write_h5ad(data_path)

    output_dir = tmp_path / "run"
    trajectory_main(
        [
            "--data-path", str(data_path),
            "--output-dir", str(output_dir),
            "--source-label", "90m",
            "--target-labels", "6h,10h",
            "--physical-times", "90m:1.5,6h:6.0,10h:10.0",
            "--mass-mode", "group_total",
            "--epochs", "1",
            "--n-particles", "3",
            "--steps-per-interval", "1",
            "--embedding-dim", "2",
            "--n-programs", "2",
            "--mediator-dim", "2",
            "--hidden-dim", "16",
            "--depth", "1",
            "--lambda-weak", "0",
            "--lambda-count", "0",
            "--lambda-reg-net", "0",
            "--lambda-reg-diffusion", "0",
            "--sinkhorn-max-iter", "5",
            "--ecology-off",
        ]
    )

    assert (output_dir / "checkpoint_last.pt").exists()
    assert (output_dir / "measure_key_manifest.csv").exists()
    assert (output_dir / "predicted_metrics_by_key_time.csv").exists()
    assert (output_dir / "mass_table.csv").exists()
    assert (output_dir / "cell_count_table.csv").exists()
    assert (output_dir / "mass_summary_by_time_sample.csv").exists()
    assert (output_dir / "run_manifest.json").exists()
    mass_table = pd.read_csv(output_dir / "mass_table.csv")
    assert set(mass_table["mass"].round(6)) == {1.0}
    pred = pd.read_csv(output_dir / "predicted_metrics_by_key_time.csv")
    assert {"physical_time", "normalized_tau", "interval_physical_duration"}.issubset(pred.columns)
    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    assert manifest["package_version"] == "2.0.9"


def test_lps_trajectory_runner_vae_source_only_with_extra_timepoint(tmp_path) -> None:
    rows = []
    rng = np.random.default_rng(19)
    for sample in ["D1"]:
        for pid, is_control in [("ctrl", True), ("LPS__mono", False)]:
            for time_i, (label, physical) in enumerate(
                [("0h", 0.0), ("90m", 1.5), ("6h", 6.0), ("10h", 10.0)]
            ):
                for cell_i in range(3):
                    rows.append(
                        {
                            "cell_id": f"{sample}_{pid}_{label}_{cell_i}",
                            "time_label": label,
                            "physical_time": physical,
                            "sample_id": sample,
                            "perturbation_id": pid,
                            "is_control": is_control,
                            "mass_value": 1.0,
                        }
                    )
    obs = pd.DataFrame(rows).set_index("cell_id", drop=False)
    counts = rng.poisson(lam=3.0, size=(len(obs), 6)).astype(np.float32)
    counts[counts < 0] = 0
    adata = ad.AnnData(X=counts, obs=obs)
    adata.layers["counts"] = counts.copy()
    data_path = tmp_path / "credo_lps_extra_timepoint.h5ad"
    adata.write_h5ad(data_path)

    output_dir = tmp_path / "vae_run"
    trajectory_main(
        [
            "--data-path", str(data_path),
            "--output-dir", str(output_dir),
            "--source-label", "90m",
            "--target-labels", "6h,10h",
            "--physical-times", "90m:1.5,6h:6.0,10h:10.0",
            "--mass-mode", "group_total",
            "--latent-source", "vae",
            "--vae-layer", "counts",
            "--vae-latent-dim", "2",
            "--vae-hidden-dim", "4",
            "--vae-depth", "1",
            "--vae-epochs", "1",
            "--vae-batch-size", "4",
            "--vae-val-frac", "0.25",
            "--no-vae-use-amp",
            "--expression-top-genes", "4",
            "--epochs", "1",
            "--n-particles", "3",
            "--eval-particles", "5",
            "--steps-per-interval", "1",
            "--embedding-dim", "2",
            "--n-programs", "2",
            "--mediator-dim", "2",
            "--hidden-dim", "16",
            "--depth", "1",
            "--lambda-weak", "0",
            "--lambda-count", "0",
            "--lambda-reg-net", "0",
            "--lambda-reg-diffusion", "0",
            "--sinkhorn-max-iter", "5",
            "--ecology-off",
        ]
    )

    assert (output_dir / "checkpoint_last.pt").exists()
    assert (output_dir / "vae_artifact" / "vae_metadata.json").exists()
    assert (output_dir / "vae_artifact" / "vae_state_dict.pt").exists()
    assert (output_dir / "vae_artifact" / "vae_history.csv").exists()
    assert (output_dir / "vae_artifact" / "vae_gene_mask.npy").exists()
    assert (output_dir / "vae_artifact" / "vae_gene_names.txt").exists()


def test_lps_trajectory_runner_ambiguous_constant_mass_requires_mode(tmp_path) -> None:
    rows = []
    latent = []
    rng = np.random.default_rng(23)
    for pid, is_control in [("ctrl", True), ("LPS__mono", False)]:
        for time_i, (label, physical) in enumerate([("90m", 1.5), ("6h", 6.0), ("10h", 10.0)]):
            for cell_i in range(2):
                rows.append(
                    {
                        "cell_id": f"{pid}_{label}_{cell_i}",
                        "time_label": label,
                        "physical_time": physical,
                        "sample_id": "D1",
                        "perturbation_id": pid,
                        "is_control": is_control,
                        "mass_value": 1.0,
                    }
                )
                latent.append(rng.normal(size=2))
    obs = pd.DataFrame(rows).set_index("cell_id", drop=False)
    adata = ad.AnnData(X=np.ones((len(obs), 4), dtype=np.float32), obs=obs)
    adata.obsm["X_pca"] = np.asarray(latent, dtype=np.float32)
    data_path = tmp_path / "ambiguous_mass.h5ad"
    adata.write_h5ad(data_path)

    with pytest.raises(ValueError, match="Specify --mass-mode"):
        trajectory_main(
            [
                "--data-path", str(data_path),
                "--output-dir", str(tmp_path / "run"),
                "--source-label", "90m",
                "--target-labels", "6h,10h",
                "--physical-times", "90m:1.5,6h:6.0,10h:10.0",
                "--epochs", "1",
                "--n-particles", "2",
                "--steps-per-interval", "1",
                "--embedding-dim", "2",
                "--n-programs", "2",
                "--mediator-dim", "2",
                "--hidden-dim", "16",
                "--depth", "1",
                "--lambda-weak", "0",
                "--lambda-count", "0",
                "--lambda-reg-net", "0",
                "--lambda-reg-diffusion", "0",
                "--sinkhorn-max-iter", "5",
                "--ecology-off",
            ]
        )


def test_per_cell_mass_mode_sums_contributions(tmp_path) -> None:
    rows = []
    latent = []
    rng = np.random.default_rng(29)
    for label, physical in [("90m", 1.5), ("6h", 6.0), ("10h", 10.0)]:
        for cell_i in range(2):
            rows.append(
                {
                    "cell_id": f"ctrl_{label}_{cell_i}",
                    "time_label": label,
                    "physical_time": physical,
                    "sample_id": "D1",
                    "perturbation_id": "ctrl",
                    "is_control": True,
                    "mass_value": 0.5,
                }
            )
            latent.append(rng.normal(size=2))
    obs = pd.DataFrame(rows).set_index("cell_id", drop=False)
    adata = ad.AnnData(X=np.ones((len(obs), 4), dtype=np.float32), obs=obs)
    adata.obsm["X_pca"] = np.asarray(latent, dtype=np.float32)
    data_path = tmp_path / "per_cell_mass.h5ad"
    adata.write_h5ad(data_path)

    args = parse_args(
        [
            "--data-path", str(data_path),
            "--output-dir", str(tmp_path / "run"),
            "--source-label", "90m",
            "--target-labels", "6h,10h",
            "--physical-times", "90m:1.5,6h:6.0,10h:10.0",
            "--mass-mode", "per_cell_contribution",
        ]
    )
    study = build_study_from_anndata(args)

    assert study.mass_table.get("ctrl", "90m", "D1") == 1.0


def test_vae_gene_mask_uses_rank_not_var_order() -> None:
    from runners.run_credo_trajectory import _column_mask_for_vae

    adata = ad.AnnData(X=np.ones((3, 5), dtype=np.float32))
    adata.var_names = [f"g{i}" for i in range(5)]
    adata.var["hv_gene"] = [True, True, True, True, True]
    adata.var["hv_rank"] = [5, 1, 4, 2, 3]
    args = parse_args(
        [
            "--data-path", "dummy.h5ad",
            "--output-dir", "dummy",
            "--expression-gene-mask-col", "hv_gene",
            "--expression-gene-rank-col", "hv_rank",
            "--expression-top-genes", "2",
        ]
    )

    mask = _column_mask_for_vae(adata, args)

    assert adata.var_names[mask].tolist() == ["g1", "g3"]
