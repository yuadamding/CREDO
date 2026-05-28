from __future__ import annotations

import json

import numpy as np
import pandas as pd
import anndata as ad
import pytest

from runners.run_credo_trajectory import build_study_from_anndata, main as trajectory_main, parse_args


def _write_tiny_runner_adata(
    tmp_path,
    *,
    mass_values: list[float] | None = None,
    include_mass_col: bool = True,
) -> str:
    rows = []
    latent = []
    rng = np.random.default_rng(41)
    i = 0
    for pid, is_control in [("ctrl", True), ("LPS__mono", False)]:
        for time_i, (label, physical) in enumerate([("90m", 1.5), ("6h", 6.0), ("10h", 10.0)]):
            for cell_i in range(2):
                row = {
                    "cell_id": f"{pid}_{label}_{cell_i}",
                    "time_label": label,
                    "physical_time": physical,
                    "sample_id": "D1",
                    "perturbation_id": pid,
                    "is_control": is_control,
                }
                if include_mass_col:
                    row["mass_value"] = float(mass_values[i] if mass_values is not None else 1.0)
                rows.append(row)
                latent.append(rng.normal(loc=float(time_i), scale=0.01, size=2))
                i += 1
    obs = pd.DataFrame(rows).set_index("cell_id", drop=False)
    adata = ad.AnnData(X=np.ones((len(obs), 4), dtype=np.float32), obs=obs)
    adata.obsm["X_pca"] = np.asarray(latent, dtype=np.float32)
    data_path = tmp_path / "tiny_runner_input.h5ad"
    adata.write_h5ad(data_path)
    return str(data_path)


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
    assert (output_dir / "input_manifest.json").exists()
    assert (output_dir / "final_manifest.json").exists()
    mass_table = pd.read_csv(output_dir / "mass_table.csv")
    assert set(mass_table["mass"].round(6)) == {1.0}
    pred = pd.read_csv(output_dir / "predicted_metrics_by_key_time.csv")
    assert {"physical_time", "normalized_tau", "interval_physical_duration"}.issubset(pred.columns)
    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    assert manifest["package_version"] == "2.0"
    assert manifest["requested_mass_mode"] == "group_total"
    assert manifest["resolved_mass_mode"] == "group_total"
    input_manifest = json.loads((output_dir / "input_manifest.json").read_text())
    final_manifest = json.loads((output_dir / "final_manifest.json").read_text())
    assert input_manifest["requested_mass_mode"] == "group_total"
    assert input_manifest["resolved_mass_mode"] == "group_total"
    assert len(input_manifest["mass_table_sha256"]) == 64
    assert final_manifest["package_version"] == "2.0"
    assert final_manifest["requested_mass_mode"] == "group_total"
    assert final_manifest["resolved_mass_mode"] == "group_total"
    assert "mass_table.csv" in final_manifest["outputs"]


def test_lps_trajectory_runner_transformer_context_smoke(tmp_path) -> None:
    data_path = _write_tiny_runner_adata(tmp_path)
    output_dir = tmp_path / "transformer_run"

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
            "--eval-particles", "4",
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
            "--context-kind", "transformer",
            "--transformer-growth-only",
            "--transformer-token-dim", "8",
            "--transformer-heads", "2",
            "--transformer-within-layers", "1",
            "--transformer-cross-layers", "1",
            "--transformer-inducing", "2",
            "--transformer-dropout", "0",
            "--ecology-on",
        ]
    )

    assert (output_dir / "checkpoint_last.pt").exists()
    assert (output_dir / "predicted_metrics_by_key_time.csv").exists()
    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    assert manifest["args"]["context_kind"] == "transformer"
    assert manifest["args"]["transformer_growth_only"] is True
    config = json.loads((output_dir / "run_config.json").read_text())
    assert config["model"]["context_kind"] == "transformer"
    assert config["model"]["transformer_growth_only"] is True


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
    metadata = json.loads((output_dir / "vae_artifact" / "vae_metadata.json").read_text())
    hyperparams = metadata["vae_hyperparams"]
    assert hyperparams["gene_selection_scope"] == "source_only"
    assert len(hyperparams["requested_row_mask_sha256"]) == 64
    assert len(hyperparams["vae_fit_mask_sha256"]) == 64
    assert len(hyperparams["gene_selection_mask_sha256"]) == 64
    assert hyperparams["requested_row_mask_sha256"] != hyperparams["vae_fit_mask_sha256"]


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


def test_explicit_mass_mode_missing_mass_column_raises(tmp_path) -> None:
    data_path = _write_tiny_runner_adata(tmp_path, include_mass_col=False)

    for mode in ["group_total", "per_cell_contribution"]:
        args = parse_args(
            [
                "--data-path", data_path,
                "--output-dir", str(tmp_path / f"run_{mode}"),
                "--source-label", "90m",
                "--target-labels", "6h,10h",
                "--physical-times", "90m:1.5,6h:6.0,10h:10.0",
                "--mass-mode", mode,
                "--mass-col", "missing_mass",
            ]
        )
        with pytest.raises(KeyError, match="--mass-col"):
            build_study_from_anndata(args)


def test_auto_mass_mode_rejects_any_constant_multicell_group(tmp_path) -> None:
    values = []
    for i in range(12):
        values.append(1.0 if i < 2 else 1.0 + 0.01 * i)
    data_path = _write_tiny_runner_adata(tmp_path, mass_values=values)
    args = parse_args(
        [
            "--data-path", data_path,
            "--output-dir", str(tmp_path / "run_auto"),
            "--source-label", "90m",
            "--target-labels", "6h,10h",
            "--physical-times", "90m:1.5,6h:6.0,10h:10.0",
            "--mass-mode", "auto",
        ]
    )

    with pytest.raises(ValueError, match="constant within at least one multi-cell group"):
        build_study_from_anndata(args)


def test_group_total_rejects_nonconstant_group_values(tmp_path) -> None:
    values = [1.0 + 0.01 * i for i in range(12)]
    data_path = _write_tiny_runner_adata(tmp_path, mass_values=values)
    args = parse_args(
        [
            "--data-path", data_path,
            "--output-dir", str(tmp_path / "run_group_total"),
            "--source-label", "90m",
            "--target-labels", "6h,10h",
            "--physical-times", "90m:1.5,6h:6.0,10h:10.0",
            "--mass-mode", "group_total",
        ]
    )

    with pytest.raises(ValueError, match="group_total requires"):
        build_study_from_anndata(args)


def test_count_mass_mode_ignores_mass_column_by_design(tmp_path) -> None:
    data_path = _write_tiny_runner_adata(tmp_path, mass_values=[100.0] * 12)
    args = parse_args(
        [
            "--data-path", data_path,
            "--output-dir", str(tmp_path / "run_count"),
            "--source-label", "90m",
            "--target-labels", "6h,10h",
            "--physical-times", "90m:1.5,6h:6.0,10h:10.0",
            "--mass-mode", "count",
        ]
    )

    study = build_study_from_anndata(args)

    assert study.mass_table.get("ctrl", "90m", "D1") == 2.0
    assert args.resolved_mass_mode == "count"


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


def test_vae_fallback_gene_selection_uses_fit_mask_not_targets() -> None:
    from runners.run_credo_trajectory import _column_mask_for_vae

    counts = np.asarray(
        [
            [1.0, 5.0, 1.0],
            [50.0, 5.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 100.0, 1.0],
        ],
        dtype=np.float32,
    )
    obs = pd.DataFrame({"time_label": ["90m", "90m", "10h", "10h"]})
    adata = ad.AnnData(X=counts, obs=obs)
    adata.var_names = ["source_var", "target_var", "flat"]
    adata.var["hv_gene"] = [True, True, True]
    args = parse_args(
        [
            "--data-path", "dummy.h5ad",
            "--output-dir", "dummy",
            "--expression-gene-mask-col", "hv_gene",
            "--expression-top-genes", "1",
        ]
    )
    source_mask = obs["time_label"].eq("90m").to_numpy()

    mask = _column_mask_for_vae(adata, args, selection_mask=source_mask)

    assert adata.var_names[mask].tolist() == ["source_var"]
