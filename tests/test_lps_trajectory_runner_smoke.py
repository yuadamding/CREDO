from __future__ import annotations

import numpy as np
import pandas as pd
import anndata as ad

from runners.run_credo_trajectory import main as trajectory_main


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
