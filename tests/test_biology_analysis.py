from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch


pytestmark = pytest.mark.biology


ROOT = Path(__file__).resolve().parents[1]


def _biology_module():
    analysis_dir = str(ROOT / "analysis")
    if analysis_dir not in sys.path:
        sys.path.insert(0, analysis_dir)
    import extract_biology_effects

    return extract_biology_effects


def _counterfactual_biology_module():
    analysis_dir = str(ROOT / "analysis")
    if analysis_dir not in sys.path:
        sys.path.insert(0, analysis_dir)
    import run_counterfactual_biology

    return run_counterfactual_biology


def test_counterfactual_biology_build_model_honors_transformer_config() -> None:
    mod = _counterfactual_biology_module()
    config = {
        "supported_perturbations": ["gene_a", "ctrl"],
        "control_ids": ["ctrl"],
        "shared_guide_embedding": False,
        "resolved_n_programs": 3,
        "program_assignment_scale": 1.0,
        "config": {
            "model": {
                "embedding_dim": 6,
                "n_programs": 3,
                "mediator_dim": 5,
                "hidden_dim": 12,
                "depth": 1,
                "time_frequencies": 2,
                "sigma_min": 1e-3,
                "r_max": 2.0,
                "n_payoff_ranks": 2,
                "ecological_growth": True,
                "use_growth_intercept": True,
                "control_mode": "soft_ref",
                "control_ref_penalty": 5e-4,
                "context_kind": "transformer",
                "transformer_token_dim": 16,
                "transformer_heads": 2,
                "transformer_within_layers": 1,
                "transformer_cross_layers": 1,
                "transformer_inducing": 4,
                "transformer_dropout": 0.0,
                "mass_attention_temperature": 0.75,
                "transformer_growth_only": True,
            }
        },
    }

    model = mod._build_model(config, latent_dim=4, program_centroids=None, device="cpu")

    assert model.context_kind == "transformer"
    assert model.transformer_growth_only is True
    assert model.context_agg.token_in[0].out_features == 16
    assert model.meanfield_context_agg is not None


def test_counterfactual_biology_program_fractions_supports_causal_attention() -> None:
    mod = _counterfactual_biology_module()
    from credo.models.weighted_sde import ParticleRollout

    config = {
        "supported_perturbations": ["gene_a", "ctrl"],
        "control_ids": ["ctrl"],
        "shared_guide_embedding": False,
        "resolved_n_programs": 3,
        "program_assignment_scale": 1.0,
        "config": {
            "model": {
                "embedding_dim": 4,
                "n_programs": 3,
                "mediator_dim": 2,
                "hidden_dim": 8,
                "depth": 1,
                "ecological_growth": True,
                "use_growth_intercept": True,
                "control_mode": "soft_ref",
                "context_kind": "causal_attention",
                "causal_token_dim": 8,
                "causal_heads": 1,
                "causal_n_mediators": 2,
                "causal_dropout": 0.0,
                "causal_growth_only": True,
            }
        },
    }
    model = mod._build_model(config, latent_dim=4, program_centroids=None, device="cpu")
    rollout = ParticleRollout(
        z_steps=torch.randn(2, 1, 5, 4),
        logw_steps=torch.zeros(2, 1, 5),
        tau_steps=torch.linspace(0.0, 1.0, 2),
        log_m0=torch.zeros(1),
    )

    fractions = mod._program_fractions(model, rollout)

    assert fractions.shape == (3,)
    assert torch.isfinite(fractions).all()
    assert torch.allclose(fractions.sum(), torch.tensor(1.0), atol=1e-5)


def test_extract_biology_effects_with_shared_and_signatures(tmp_path: Path) -> None:
    with_root = tmp_path / "with"
    shared_root = tmp_path / "shared"
    for root, shared in [(with_root, False), (shared_root, True)]:
        run_dir = root / "setting_a" / "fold_0"
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text(
            json.dumps(
                {
                    "split": {"split_strategy": "random_kfold", "fold_index": 0},
                    "shared_guide_embedding": shared,
                    "use_state_centroids": True,
                }
            )
        )
        (run_dir / "results_summary.json").write_text(json.dumps({"shared_guide_embedding": shared}))
        pd.DataFrame(
            [
                {
                    "perturbation_id": "Notch1_sg1",
                    "uot": 1.0,
                    "mass_pred": 20.0 if not shared else 12.0,
                    "mass_true": 18.0,
                    "mass_rel_error": 0.1 if not shared else 0.4,
                    "is_control": False,
                    "n_init_atoms": 10,
                    "n_term_atoms": 18,
                },
                {
                    "perturbation_id": "NTC",
                    "uot": 0.5,
                    "mass_pred": 10.0,
                    "mass_true": 10.0,
                    "mass_rel_error": 0.2,
                    "is_control": True,
                    "n_init_atoms": 10,
                    "n_term_atoms": 10,
                },
            ]
        ).to_csv(run_dir / "test_endpoint_metrics.csv", index=False)
        pd.DataFrame(
            [
                {
                    "perturbation_id": "Notch1_sg1",
                    "state_tv": 0.2,
                    "dominant_state_pred": "TSK",
                    "dominant_state_true": "TSK",
                    "dominant_state_match": not shared,
                    "pred_expansion_ratio": 2.0 if not shared else 1.1,
                    "true_expansion_ratio": 1.8,
                    "expansion_ratio_gap": 0.2 if not shared else -0.7,
                }
            ]
        ).to_csv(run_dir / "test_state_metrics.csv", index=False)

    sig = pd.DataFrame(
        [
            {"perturbation_id": "Notch1_sg1", "time_label": "P4", "tnf_expansion": 0.0, "autocrine_tnf_tsk": 0.0},
            {"perturbation_id": "Notch1_sg1", "time_label": "P60", "tnf_expansion": 1.5, "autocrine_tnf_tsk": 2.0},
        ]
    )
    sig_path = tmp_path / "signature_group_scores.csv"
    sig.to_csv(sig_path, index=False)
    cf = pd.DataFrame(
        [
            {
                "perturbation_id": "Notch1_sg1",
                "target_gene": "NOTCH1",
                "fold_id": "fold_0",
                "delta_log_mass_fact_vs_ref": 0.7,
                "geom_shift_fact_vs_ref": 1.2,
                "energy_distance_fact_vs_ref": 1.1,
                "growth_action_fact": 0.3,
                "drift_action_fact": 0.4,
                "diffusion_action_fact": 0.5,
                "context_dependence_geom": 0.6,
                "context_dependence_mass": 0.2,
            },
            {
                "perturbation_id": "Notch1_sg1",
                "target_gene": "NOTCH1",
                "fold_id": "fold_1",
                "delta_log_mass_fact_vs_ref": 0.9,
                "geom_shift_fact_vs_ref": 1.4,
                "energy_distance_fact_vs_ref": 1.3,
                "growth_action_fact": 0.5,
                "drift_action_fact": 0.6,
                "diffusion_action_fact": 0.7,
                "context_dependence_geom": 0.8,
                "context_dependence_mass": 0.4,
            },
        ]
    )
    cf_path = tmp_path / "counterfactual_biology_effects.csv"
    cf.to_csv(cf_path, index=False)
    out_dir = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "analysis" / "extract_biology_effects.py"),
            "--cv-root",
            str(with_root),
            "--shared-cv-root",
            str(shared_root),
            "--signature-scores",
            str(sig_path),
            "--counterfactual-effects",
            str(cf_path),
            "--output-dir",
            str(out_dir),
        ],
        check=True,
    )
    out = pd.read_csv(out_dir / "biological_effects_per_perturbation.csv")
    notch = out.loc[out["perturbation_id"].eq("Notch1_sg1")].iloc[0]
    assert notch["target_gene"] == "NOTCH1"
    assert notch["delta_tnf_expansion_score"] == 1.5
    assert notch["delta_autocrine_tnf_tsk_score"] == 2.0
    assert np.isclose(notch["delta_log_mass_fact_vs_ref"], 0.8)
    assert np.isclose(notch["delta_log_mass"], 0.8)
    assert np.isclose(notch["geom_shift_fact_vs_ref"], 1.3)
    assert np.isclose(notch["energy_distance_fact_vs_ref"], 1.2)
    assert np.isclose(notch["legacy_geom_shift_fact_vs_ref"], 1.3)
    assert np.isclose(notch["diffusion_action"], 0.6)
    assert np.isclose(notch["context_dependence_geom"], 0.7)
    assert notch["counterfactual_n_folds"] == 2
    assert notch["delta_log_mass_fact_vs_ref_std"] > 0
    assert "priority_class_v2" in out.columns
    assert "biological_interpretation_gate" in out.columns
    assert "fold_stability_pass" in out.columns
    assert "guide_concordance_pass" in out.columns
    assert "negative_control_gap_pass" in out.columns
    assert notch["biological_interpretation_gate"] == "needs-guide-concordance"
    assert notch["guide_concordance_status"] == "not_assessable"
    assert notch["shared_guide_null_gap"] > 0


def test_biological_gates_detect_guide_discordance(tmp_path: Path) -> None:
    cv_root = tmp_path / "with"
    for fold in range(2):
        run_dir = cv_root / "setting_a" / f"fold_{fold}"
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text(
            json.dumps({"split": {"split_strategy": "random_kfold", "fold_index": fold}})
        )
        (run_dir / "results_summary.json").write_text(json.dumps({"shared_guide_embedding": False}))
        pd.DataFrame(
            [
                {
                    "perturbation_id": "GeneA_sg1",
                    "mass_pred": 20.0,
                    "mass_true": 10.0,
                    "mass_rel_error": 0.1,
                    "is_control": False,
                    "n_init_atoms": 10,
                    "n_term_atoms": 20,
                },
                {
                    "perturbation_id": "GeneA_sg2",
                    "mass_pred": 5.0,
                    "mass_true": 10.0,
                    "mass_rel_error": 0.1,
                    "is_control": False,
                    "n_init_atoms": 10,
                    "n_term_atoms": 5,
                },
                {
                    "perturbation_id": "NTC",
                    "mass_pred": 10.0,
                    "mass_true": 10.0,
                    "mass_rel_error": 0.0,
                    "is_control": True,
                    "n_init_atoms": 10,
                    "n_term_atoms": 10,
                },
            ]
        ).to_csv(run_dir / "test_endpoint_metrics.csv", index=False)
        pd.DataFrame(
            [
                {"perturbation_id": "GeneA_sg1", "pred_expansion_ratio": 2.0, "true_expansion_ratio": 2.0},
                {"perturbation_id": "GeneA_sg2", "pred_expansion_ratio": 0.5, "true_expansion_ratio": 0.5},
            ]
        ).to_csv(run_dir / "test_state_metrics.csv", index=False)

    cf = pd.DataFrame(
        [
            {"perturbation_id": "GeneA_sg1", "delta_log_mass_fact_vs_ref": 0.8, "diffusion_action_fact": 0.2},
            {"perturbation_id": "GeneA_sg1", "delta_log_mass_fact_vs_ref": 0.9, "diffusion_action_fact": 0.3},
            {"perturbation_id": "GeneA_sg2", "delta_log_mass_fact_vs_ref": -0.7, "diffusion_action_fact": 0.2},
            {"perturbation_id": "GeneA_sg2", "delta_log_mass_fact_vs_ref": -0.8, "diffusion_action_fact": 0.3},
        ]
    )
    cf_path = tmp_path / "counterfactual.csv"
    cf.to_csv(cf_path, index=False)
    out_dir = tmp_path / "out"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "analysis" / "extract_biology_effects.py"),
            "--cv-root",
            str(cv_root),
            "--counterfactual-effects",
            str(cf_path),
            "--output-dir",
            str(out_dir),
        ],
        check=True,
    )

    out = pd.read_csv(out_dir / "biological_effects_per_perturbation.csv")
    sg1 = out.loc[out["perturbation_id"].eq("GeneA_sg1")].iloc[0]
    sg2 = out.loc[out["perturbation_id"].eq("GeneA_sg2")].iloc[0]
    assert sg1["same_gene_n_guides"] == 2
    assert sg2["same_gene_n_guides"] == 2
    assert sg1["same_gene_sgrna_concordance"] == 0.5
    assert sg1["guide_concordance_pass"] == np.False_
    assert sg1["biological_interpretation_gate"] == "needs-guide-concordance"


def test_guide_concordance_uses_sgrna_id_when_present() -> None:
    mod = _biology_module()
    df = pd.DataFrame(
        [
            {
                "perturbation_id": "GeneA_collapsed",
                "sgRNA_id": "GeneA_sg1",
                "target_gene": "GENEA",
                "delta_log_mass_fact_vs_ref": 0.8,
            },
            {
                "perturbation_id": "GeneA_collapsed",
                "sgRNA_id": "GeneA_sg2",
                "target_gene": "GENEA",
                "delta_log_mass_fact_vs_ref": 0.9,
            },
        ]
    )

    out = mod._add_guide_concordance(df)

    assert out["same_gene_n_guides"].iloc[0] == 2
    assert out["same_gene_sgrna_concordance"].iloc[0] == 1.0


def test_biological_gate_missing_fold_stability_does_not_pass() -> None:
    mod = _biology_module()
    df = pd.DataFrame(
        [
            {
                "perturbation_id": "ctrl",
                "target_gene": "control",
                "is_control": True,
                "priority_class": "watch",
                "delta_log_mass": 0.05,
            },
            {
                "perturbation_id": "GeneA_sg1",
                "target_gene": "GENEA",
                "is_control": False,
                "priority_class": "growth-high",
                "delta_log_mass": 1.0,
                "delta_log_mass_fact_vs_ref": 1.0,
                "counterfactual_n_folds": 2,
                "same_gene_n_guides": 2,
                "same_gene_sgrna_concordance": 1.0,
                "requested_mass_mode": "count",
            },
        ]
    )

    out = mod._add_biological_gates(df)
    row = out.loc[out["perturbation_id"].eq("GeneA_sg1")].iloc[0]

    assert row["fold_stability_pass"] == np.False_
    assert row["biological_interpretation_gate"] == "needs-fold-stability"
    assert row["claim_ready"] == np.False_


def test_ecology_gate_requires_counterfactual_replicates() -> None:
    mod = _biology_module()
    df = pd.DataFrame(
        [
            {
                "perturbation_id": "ctrl",
                "target_gene": "control",
                "is_control": True,
                "priority_class": "watch",
                "delta_log_mass": 0.05,
                "context_dependence_geom": 0.05,
            },
            {
                "perturbation_id": "GeneA_sg1",
                "target_gene": "GENEA",
                "is_control": False,
                "priority_class": "ecology-dependent",
                "delta_log_mass": 1.0,
                "delta_log_mass_fact_vs_ref": 1.0,
                "delta_log_mass_fact_vs_ref_sign_consistency": 1.0,
                "counterfactual_n_folds": 1,
                "same_gene_n_guides": 2,
                "same_gene_sgrna_concordance": 1.0,
                "context_dependence_geom": 1.0,
                "context_dependence_geom_sign_consistency": 1.0,
                "requested_mass_mode": "count",
            },
        ]
    )

    out = mod._add_biological_gates(df)
    row = out.loc[out["perturbation_id"].eq("GeneA_sg1")].iloc[0]

    assert row["counterfactual_replicate_pass"] == np.False_
    assert row["ecology_ablation_pass"] == np.False_
    assert row["biological_interpretation_gate"] == "needs-counterfactual-replicates"


def test_context_sign_missing_does_not_pass_ecology() -> None:
    mod = _biology_module()
    df = pd.DataFrame(
        [
            {
                "perturbation_id": "ctrl",
                "target_gene": "control",
                "is_control": True,
                "priority_class": "watch",
                "delta_log_mass": 0.05,
                "context_dependence_geom": 0.05,
            },
            {
                "perturbation_id": "GeneA_sg1",
                "target_gene": "GENEA",
                "is_control": False,
                "priority_class": "ecology-dependent",
                "delta_log_mass": 1.0,
                "delta_log_mass_fact_vs_ref": 1.0,
                "delta_log_mass_fact_vs_ref_sign_consistency": 1.0,
                "counterfactual_n_folds": 2,
                "same_gene_n_guides": 2,
                "same_gene_sgrna_concordance": 1.0,
                "context_dependence_geom": 1.0,
                "requested_mass_mode": "count",
            },
        ]
    )

    out = mod._add_biological_gates(df)
    row = out.loc[out["perturbation_id"].eq("GeneA_sg1")].iloc[0]

    assert row["ecology_ablation_pass"] == np.False_
    assert row["biological_interpretation_gate"] == "needs-context-ablation"


def test_missing_negative_control_null_blocks_claim_ready() -> None:
    mod = _biology_module()
    df = pd.DataFrame(
        [
            {
                "perturbation_id": "GeneA_sg1",
                "target_gene": "GENEA",
                "is_control": False,
                "priority_class": "growth-high",
                "delta_log_mass": 1.0,
                "delta_log_mass_fact_vs_ref": 1.0,
                "delta_log_mass_fact_vs_ref_sign_consistency": 1.0,
                "counterfactual_n_folds": 2,
                "same_gene_n_guides": 2,
                "same_gene_sgrna_concordance": 1.0,
                "requested_mass_mode": "count",
            },
        ]
    )

    out = mod._add_biological_gates(df)
    row = out.iloc[0]

    assert pd.isna(row["mass_null_gap_pass"])
    assert row["biological_interpretation_gate"] == "missing-mass-null"
    assert row["claim_ready"] == np.False_


def test_single_guide_is_not_claim_ready_strict() -> None:
    mod = _biology_module()
    df = pd.DataFrame(
        [
            {
                "perturbation_id": "ctrl",
                "target_gene": "control",
                "is_control": True,
                "priority_class": "watch",
                "delta_log_mass": 0.05,
            },
            {
                "perturbation_id": "GeneA_sg1",
                "target_gene": "GENEA",
                "is_control": False,
                "priority_class": "growth-high",
                "delta_log_mass": 1.0,
                "delta_log_mass_fact_vs_ref": 1.0,
                "delta_log_mass_fact_vs_ref_sign_consistency": 1.0,
                "counterfactual_n_folds": 2,
                "same_gene_n_guides": 1,
                "requested_mass_mode": "count",
            },
        ]
    )

    out = mod._add_biological_gates(df)
    row = out.loc[out["perturbation_id"].eq("GeneA_sg1")].iloc[0]

    assert row["guide_concordance_status"] == "not_assessable"
    assert row["guide_concordance_pass"] == np.False_
    assert row["biological_interpretation_gate"] == "needs-guide-concordance"
    assert row["claim_ready_strict"] == np.False_
    assert row["claim_ready_screening"] == np.True_


def test_metric_specific_context_null_blocks_ecology_claim() -> None:
    mod = _biology_module()
    df = pd.DataFrame(
        [
            {
                "perturbation_id": "ctrl",
                "target_gene": "control",
                "is_control": True,
                "priority_class": "watch",
                "delta_log_mass": 0.05,
                "context_dependence_geom": 2.0,
            },
            {
                "perturbation_id": "GeneA_sg1",
                "target_gene": "GENEA",
                "is_control": False,
                "priority_class": "ecology-dependent",
                "delta_log_mass": 1.0,
                "delta_log_mass_fact_vs_ref": 1.0,
                "delta_log_mass_fact_vs_ref_sign_consistency": 1.0,
                "counterfactual_n_folds": 2,
                "same_gene_n_guides": 2,
                "same_gene_sgrna_concordance": 1.0,
                "context_dependence_geom": 1.0,
                "context_dependence_geom_sign_consistency": 1.0,
                "requested_mass_mode": "count",
            },
        ]
    )

    out = mod._add_biological_gates(df)
    row = out.loc[out["perturbation_id"].eq("GeneA_sg1")].iloc[0]

    assert row["context_dependence_null_gap_pass"] == np.False_
    assert row["biological_interpretation_gate"] == "below-context-null-gap"


def test_counterfactual_n_folds_counts_unique_fold_ids(tmp_path: Path) -> None:
    mod = _biology_module()
    raw = pd.DataFrame(
        [
            {
                "perturbation_id": "GeneA_sg1",
                "fold_id": "fold_0",
                "delta_log_mass_fact_vs_ref": 0.7,
            },
            {
                "perturbation_id": "GeneA_sg1",
                "fold_id": "fold_1",
                "delta_log_mass_fact_vs_ref": 0.9,
            },
        ]
    )
    path = tmp_path / "counterfactual_unique_folds.csv"
    raw.to_csv(path, index=False)

    out = mod._load_counterfactual_effects(path)

    assert out.loc[0, "counterfactual_n_folds"] == 2


def test_duplicate_counterfactual_fold_rows_raise(tmp_path: Path) -> None:
    mod = _biology_module()
    path = tmp_path / "dupe_cf.csv"
    pd.DataFrame(
        [
            {"perturbation_id": "GeneA_sg1", "fold_id": "fold_0", "delta_log_mass_fact_vs_ref": 0.7},
            {"perturbation_id": "GeneA_sg1", "fold_id": "fold_0", "delta_log_mass_fact_vs_ref": 0.8},
        ]
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Duplicate counterfactual rows"):
        mod._load_counterfactual_effects(path)


def test_plasticity_claim_requires_distributional_metric_null() -> None:
    mod = _biology_module()
    df = pd.DataFrame(
        [
            {
                "perturbation_id": "ctrl",
                "target_gene": "control",
                "is_control": True,
                "priority_class": "watch",
                "delta_log_mass": 0.05,
                "weighted_mean_shift_l2_fact_vs_ref": 0.05,
                "diffusion_action": 0.05,
            },
            {
                "perturbation_id": "GeneA_sg1",
                "target_gene": "GENEA",
                "is_control": False,
                "priority_class": "plasticity/state-shift",
                "delta_log_mass": 1.0,
                "delta_log_mass_fact_vs_ref": 1.0,
                "delta_log_mass_fact_vs_ref_sign_consistency": 1.0,
                "diffusion_action": 1.0,
                "diffusion_action_fact_sign_consistency": 1.0,
                "counterfactual_n_folds": 2,
                "same_gene_n_guides": 2,
                "same_gene_sgrna_concordance": 1.0,
                "weighted_mean_shift_l2_fact_vs_ref": 1.0,
                "requested_mass_mode": "count",
            },
        ]
    )

    out = mod._add_biological_gates(df)
    row = out.loc[out["perturbation_id"].eq("GeneA_sg1")].iloc[0]

    assert pd.isna(row["distribution_shift_null_gap_pass"])
    assert row["biological_interpretation_gate"] == "missing-distribution-shift-null"
    assert row["plasticity_claim_ready"] == np.False_


def test_plasticity_claim_requires_distributional_shift_stability() -> None:
    mod = _biology_module()
    df = pd.DataFrame(
        [
            {
                "perturbation_id": "ctrl",
                "target_gene": "control",
                "is_control": True,
                "priority_class": "watch",
                "delta_log_mass": 0.0,
                "energy_distance_fact_vs_ref": 0.0,
                "diffusion_action": 0.0,
            },
            {
                "perturbation_id": "GeneA_sg1",
                "target_gene": "GENEA",
                "is_control": False,
                "priority_class": "plasticity/state-shift",
                "delta_log_mass": 1.0,
                "delta_log_mass_fact_vs_ref": 1.0,
                "delta_log_mass_fact_vs_ref_sign_consistency": 1.0,
                "diffusion_action": 1.0,
                "diffusion_action_fact_sign_consistency": 1.0,
                "counterfactual_n_folds": 2,
                "same_gene_n_guides": 2,
                "same_gene_sgrna_concordance": 1.0,
                "energy_distance_fact_vs_ref": 1.0,
                "energy_distance_fact_vs_ref_abs_cv": np.nan,
                "requested_mass_mode": "count",
            },
        ]
    )

    out = mod._add_biological_gates(df)
    row = out.loc[out["perturbation_id"].eq("GeneA_sg1")].iloc[0]

    assert row["distribution_shift_null_gap_pass"] == np.True_
    assert row["distribution_shift_stability_pass"] == np.False_
    assert row["biological_interpretation_gate"] == "needs-distribution-shift-stability"
    assert row["plasticity_claim_ready"] == np.False_


def test_distribution_shift_cv_alone_does_not_pass_tiny_effect() -> None:
    mod = _biology_module()
    df = pd.DataFrame(
        [
            {
                "perturbation_id": "ctrl",
                "target_gene": "control",
                "is_control": True,
                "priority_class": "watch",
                "delta_log_mass": 0.0,
                "energy_distance_fact_vs_ref": 0.0,
                "diffusion_action": 0.0,
            },
            {
                "perturbation_id": "GeneA_sg1",
                "target_gene": "GENEA",
                "is_control": False,
                "priority_class": "plasticity/state-shift",
                "delta_log_mass": 1.0,
                "delta_log_mass_fact_vs_ref": 1.0,
                "delta_log_mass_fact_vs_ref_sign_consistency": 1.0,
                "diffusion_action": 1.0,
                "diffusion_action_fact_sign_consistency": 1.0,
                "counterfactual_n_folds": 2,
                "same_gene_n_guides": 2,
                "same_gene_sgrna_concordance": 1.0,
                "energy_distance_fact_vs_ref": 0.01,
                "energy_distance_fact_vs_ref_abs_cv": 0.1,
                "requested_mass_mode": "count",
            },
        ]
    )

    out = mod._add_biological_gates(df)
    row = out.loc[out["perturbation_id"].eq("GeneA_sg1")].iloc[0]

    assert row["distribution_shift_null_gap_pass"] == np.True_
    assert row["distribution_shift_stability_pass"] == np.False_
    assert row["biological_interpretation_gate"] == "needs-distribution-shift-stability"


def test_counterfactual_loader_computes_distribution_fold_support_from_controls(tmp_path: Path) -> None:
    mod = _biology_module()
    path = tmp_path / "cf.csv"
    pd.DataFrame(
        [
            {
                "perturbation_id": "ctrl",
                "fold_id": "fold_0",
                "is_control": "True",
                "energy_distance_fact_vs_ref": 0.0,
            },
            {
                "perturbation_id": "ctrl",
                "fold_id": "fold_1",
                "is_control": "True",
                "energy_distance_fact_vs_ref": 0.0,
            },
            {
                "perturbation_id": "GeneA_sg1",
                "fold_id": "fold_0",
                "is_control": "False",
                "energy_distance_fact_vs_ref": 0.5,
            },
            {
                "perturbation_id": "GeneA_sg1",
                "fold_id": "fold_1",
                "is_control": "False",
                "energy_distance_fact_vs_ref": 0.6,
            },
        ]
    ).to_csv(path, index=False)

    out = mod._load_counterfactual_effects(path)
    row = out.loc[out["perturbation_id"].eq("GeneA_sg1")].iloc[0]

    assert row["counterfactual_n_folds"] == 2
    assert row["energy_distance_fact_vs_ref_fold_support"] == 1.0


def test_distribution_fold_support_uses_unique_run_dir_not_rows(tmp_path: Path) -> None:
    mod = _biology_module()
    path = tmp_path / "cf_run_dir.csv"
    pd.DataFrame(
        [
            {
                "perturbation_id": "ctrl",
                "run_dir": "run_0",
                "is_control": True,
                "energy_distance_fact_vs_ref": 0.0,
            },
            {
                "perturbation_id": "ctrl",
                "run_dir": "run_1",
                "is_control": True,
                "energy_distance_fact_vs_ref": 0.0,
            },
            {
                "perturbation_id": "GeneA_sg1",
                "run_dir": "run_0",
                "is_control": False,
                "energy_distance_fact_vs_ref": 0.5,
            },
            {
                "perturbation_id": "GeneA_sg1",
                "run_dir": "run_0",
                "is_control": False,
                "energy_distance_fact_vs_ref": 0.7,
            },
            {
                "perturbation_id": "GeneA_sg1",
                "run_dir": "run_1",
                "is_control": False,
                "energy_distance_fact_vs_ref": 0.0,
            },
        ]
    ).to_csv(path, index=False)

    out = mod._load_counterfactual_effects(path)
    row = out.loc[out["perturbation_id"].eq("GeneA_sg1")].iloc[0]

    assert row["counterfactual_n_folds"] == 2
    assert row["energy_distance_fact_vs_ref_fold_support"] == 0.5


def test_counterfactual_loader_computes_sample_support_when_sample_ids_present(tmp_path: Path) -> None:
    mod = _biology_module()
    path = tmp_path / "cf_sample.csv"
    pd.DataFrame(
        [
            {
                "perturbation_id": "ctrl",
                "run_dir": "run_0",
                "sample_id": "D1",
                "is_control": True,
                "energy_distance_fact_vs_ref": 0.0,
            },
            {
                "perturbation_id": "ctrl",
                "run_dir": "run_1",
                "sample_id": "D2",
                "is_control": True,
                "energy_distance_fact_vs_ref": 0.0,
            },
            {
                "perturbation_id": "GeneA_sg1",
                "run_dir": "run_0",
                "sample_id": "D1",
                "is_control": False,
                "energy_distance_fact_vs_ref": 0.5,
            },
            {
                "perturbation_id": "GeneA_sg1",
                "run_dir": "run_1",
                "sample_id": "D1",
                "is_control": False,
                "energy_distance_fact_vs_ref": 0.7,
            },
            {
                "perturbation_id": "GeneA_sg1",
                "run_dir": "run_2",
                "sample_id": "D2",
                "is_control": False,
                "energy_distance_fact_vs_ref": 0.0,
            },
        ]
    ).to_csv(path, index=False)

    out = mod._load_counterfactual_effects(path)
    row = out.loc[out["perturbation_id"].eq("GeneA_sg1")].iloc[0]

    assert row["energy_distance_fact_vs_ref_sample_key"] == "sample_id"
    assert row["energy_distance_fact_vs_ref_sample_n"] == 2
    assert row["energy_distance_fact_vs_ref_sample_positive_fraction"] == 0.5
    assert np.isclose(row["energy_distance_fact_vs_ref_sample_effect_median"], 0.3)


def test_program_occupancy_tv_can_supply_plasticity_state_shift_support() -> None:
    mod = _biology_module()
    df = pd.DataFrame(
        [
            {
                "perturbation_id": "ctrl",
                "target_gene": "control",
                "is_control": True,
                "priority_class": "plasticity/state-shift",
                "delta_log_mass": 0.0,
                "program_occupancy_tv_fact_vs_ref": 0.0,
                "diffusion_action": 0.0,
            },
            {
                "perturbation_id": "GeneA_sg1",
                "target_gene": "GENEA",
                "is_control": False,
                "priority_class": "plasticity/state-shift",
                "delta_log_mass": 0.0,
                "delta_log_mass_fact_vs_ref": 0.0,
                "diffusion_action": 1.0,
                "diffusion_action_fact_sign_consistency": 1.0,
                "counterfactual_n_folds": 2,
                "same_gene_n_guides": 2,
                "same_gene_sgrna_concordance": 1.0,
                "program_occupancy_tv_fact_vs_ref": 0.6,
                "program_occupancy_tv_fact_vs_ref_fold_support": 1.0,
                "requested_mass_mode": "count",
            },
        ]
    )

    out = mod._add_biological_gates(df)
    row = out.loc[out["perturbation_id"].eq("GeneA_sg1")].iloc[0]

    assert row["program_occupancy_tv_null_gap_pass"] == np.True_
    assert row["program_occupancy_stability_pass"] == np.True_
    assert row["plasticity_claim_ready"] == np.True_


def test_sample_support_blocks_strict_mass_claim_when_single_sample_drives_effect() -> None:
    mod = _biology_module()
    df = pd.DataFrame(
        [
            {
                "perturbation_id": "ctrl",
                "target_gene": "control",
                "is_control": True,
                "priority_class": "growth-high",
                "delta_log_mass": 0.0,
            },
            {
                "perturbation_id": "GeneA_sg1",
                "target_gene": "GENEA",
                "is_control": False,
                "priority_class": "growth-high",
                "delta_log_mass": 1.0,
                "delta_log_mass_fact_vs_ref": 1.0,
                "delta_log_mass_fact_vs_ref_sign_consistency": 1.0,
                "counterfactual_n_folds": 2,
                "same_gene_n_guides": 2,
                "same_gene_sgrna_concordance": 1.0,
                "requested_mass_mode": "count",
                "mass_counterfactual_effect_sample_n": 1,
                "mass_counterfactual_effect_sample_positive_fraction": 1.0,
            },
        ]
    )

    out = mod._add_biological_gates(df)
    row = out.loc[out["perturbation_id"].eq("GeneA_sg1")].iloc[0]

    assert row["mass_sample_support_pass"] == np.False_
    assert row["expansion_claim_ready"] == np.False_
    assert row["biological_interpretation_gate"] == "needs-sample-support"


def test_tsk_pemt_claim_does_not_require_expansion_ready() -> None:
    mod = _biology_module()
    df = pd.DataFrame(
        [
            {
                "perturbation_id": "ctrl",
                "target_gene": "control",
                "is_control": True,
                "priority_class": "watch",
                "delta_log_mass": 0.05,
                "delta_autocrine_tnf_tsk_score": 0.0,
                "delta_pemt_score": 0.0,
            },
            {
                "perturbation_id": "GeneA_sg1",
                "target_gene": "GENEA",
                "is_control": False,
                "priority_class": "growth-high",
                "delta_log_mass": 0.01,
                "delta_log_mass_fact_vs_ref": 0.01,
                "delta_log_mass_fact_vs_ref_sign_consistency": 1.0,
                "counterfactual_n_folds": 2,
                "same_gene_n_guides": 2,
                "same_gene_sgrna_concordance": 1.0,
                "delta_autocrine_tnf_tsk_score": 2.0,
                "delta_pemt_score": 0.0,
                "z_delta_autocrine_tnf_tsk_score": 1.0,
                "requested_mass_mode": "count",
                "tsk_pemt_program_effect_pos_fold_support": 1.0,
            },
        ]
    )

    out = mod._add_biological_gates(df)
    row = out.loc[out["perturbation_id"].eq("GeneA_sg1")].iloc[0]

    assert row["expansion_claim_ready"] == np.False_
    assert row["tsk_pemt_claim_ready"] == np.True_


def test_negative_mass_delta_not_expansion_claim_ready_and_can_deplete() -> None:
    mod = _biology_module()
    df = pd.DataFrame(
        [
            {
                "perturbation_id": "ctrl",
                "target_gene": "control",
                "is_control": True,
                "priority_class": "growth-high",
                "delta_log_mass": 0.0,
            },
            {
                "perturbation_id": "GeneA_sg1",
                "target_gene": "GENEA",
                "is_control": False,
                "priority_class": "growth-high",
                "delta_log_mass": -1.0,
                "delta_log_mass_fact_vs_ref": -1.0,
                "delta_log_mass_fact_vs_ref_sign_consistency": 1.0,
                "counterfactual_n_folds": 2,
                "same_gene_n_guides": 2,
                "same_gene_sgrna_concordance": 1.0,
                "requested_mass_mode": "count",
            },
        ]
    )

    out = mod._add_biological_gates(df)
    row = out.loc[out["perturbation_id"].eq("GeneA_sg1")].iloc[0]

    assert row["mass_null_gap_pass"] == np.True_
    assert row["expansion_claim_ready"] == np.False_
    assert row["depletion_claim_ready"] == np.True_


def test_negative_tsk_pemt_program_effect_does_not_pass_tsk_claim() -> None:
    mod = _biology_module()
    df = pd.DataFrame(
        [
            {
                "perturbation_id": "ctrl",
                "target_gene": "control",
                "is_control": True,
                "priority_class": "watch",
                "delta_log_mass": 0.05,
                "delta_autocrine_tnf_tsk_score": 0.0,
                "delta_pemt_score": 0.0,
            },
            {
                "perturbation_id": "GeneA_sg1",
                "target_gene": "GENEA",
                "is_control": False,
                "priority_class": "growth-high",
                "delta_log_mass": 0.01,
                "delta_log_mass_fact_vs_ref": 0.01,
                "delta_log_mass_fact_vs_ref_sign_consistency": 1.0,
                "counterfactual_n_folds": 2,
                "same_gene_n_guides": 2,
                "same_gene_sgrna_concordance": 1.0,
                "delta_autocrine_tnf_tsk_score": -2.0,
                "delta_pemt_score": 0.0,
                "z_delta_autocrine_tnf_tsk_score": 1.0,
                "requested_mass_mode": "count",
            },
        ]
    )

    out = mod._add_biological_gates(df)
    row = out.loc[out["perturbation_id"].eq("GeneA_sg1")].iloc[0]

    assert row["tsk_pemt_program_effect_abs"] == 2.0
    assert row["tsk_pemt_program_effect_pos"] == 0.0
    assert row["tsk_pemt_claim_ready"] == np.False_


def test_negative_tnf_program_effect_does_not_pass_tnf_claim() -> None:
    mod = _biology_module()
    df = pd.DataFrame(
        [
            {
                "perturbation_id": "ctrl",
                "target_gene": "control",
                "is_control": True,
                "priority_class": "watch",
                "delta_log_mass": 0.0,
                "delta_tnf_expansion_score": 0.0,
            },
            {
                "perturbation_id": "GeneA_sg1",
                "target_gene": "GENEA",
                "is_control": False,
                "priority_class": "growth-high",
                "delta_log_mass": 1.0,
                "delta_log_mass_fact_vs_ref": 1.0,
                "delta_log_mass_fact_vs_ref_sign_consistency": 1.0,
                "counterfactual_n_folds": 2,
                "same_gene_n_guides": 2,
                "same_gene_sgrna_concordance": 1.0,
                "delta_tnf_expansion_score": -2.0,
                "requested_mass_mode": "count",
            },
        ]
    )

    out = mod._add_biological_gates(df)
    row = out.loc[out["perturbation_id"].eq("GeneA_sg1")].iloc[0]

    assert row["tnf_expansion_program_effect_pos"] == 0.0
    assert row["tnf_expansion_claim_ready"] == np.False_


def test_tnf_expansion_claim_requires_program_fold_support() -> None:
    mod = _biology_module()
    df = pd.DataFrame(
        [
            {
                "perturbation_id": "ctrl",
                "target_gene": "control",
                "is_control": True,
                "priority_class": "growth-high",
                "delta_log_mass": 0.0,
                "delta_tnf_expansion_score": 0.0,
            },
            {
                "perturbation_id": "GeneA_sg1",
                "target_gene": "GENEA",
                "is_control": False,
                "priority_class": "growth-high",
                "delta_log_mass": 1.0,
                "delta_log_mass_fact_vs_ref": 1.0,
                "delta_log_mass_fact_vs_ref_sign_consistency": 1.0,
                "counterfactual_n_folds": 2,
                "same_gene_n_guides": 2,
                "same_gene_sgrna_concordance": 1.0,
                "delta_tnf_expansion_score": 2.0,
                "requested_mass_mode": "count",
            },
        ]
    )

    out = mod._add_biological_gates(df)
    row = out.loc[out["perturbation_id"].eq("GeneA_sg1")].iloc[0]

    assert row["tnf_expansion_program_null_gap_pass"] == np.True_
    assert row["tnf_expansion_program_stability_pass"] == np.False_
    assert row["tnf_expansion_claim_ready"] == np.False_


def test_cis_like_program_null_required_for_cis_claim() -> None:
    mod = _biology_module()
    df = pd.DataFrame(
        [
            {
                "perturbation_id": "ctrl",
                "target_gene": "control",
                "is_control": True,
                "priority_class": "watch",
                "delta_log_mass": 0.0,
                "delta_cis_like_score": 3.0,
            },
            {
                "perturbation_id": "GeneA_sg1",
                "target_gene": "GENEA",
                "is_control": False,
                "priority_class": "growth-high",
                "delta_log_mass": 1.0,
                "delta_log_mass_fact_vs_ref": 1.0,
                "delta_log_mass_fact_vs_ref_sign_consistency": 1.0,
                "counterfactual_n_folds": 2,
                "same_gene_n_guides": 2,
                "same_gene_sgrna_concordance": 1.0,
                "delta_cis_like_score": 2.0,
                "requested_mass_mode": "count",
            },
        ]
    )

    out = mod._add_biological_gates(df)
    row = out.loc[out["perturbation_id"].eq("GeneA_sg1")].iloc[0]

    assert row["cis_like_program_null_gap_pass"] == np.False_
    assert row["cis_like_claim_ready"] == np.False_


def test_tsk_pemt_program_null_required_for_tsk_pemt_claim() -> None:
    mod = _biology_module()
    df = pd.DataFrame(
        [
            {
                "perturbation_id": "ctrl",
                "target_gene": "control",
                "is_control": True,
                "priority_class": "watch",
                "delta_log_mass": 0.0,
                "delta_autocrine_tnf_tsk_score": 3.0,
                "delta_pemt_score": 0.0,
            },
            {
                "perturbation_id": "GeneA_sg1",
                "target_gene": "GENEA",
                "is_control": False,
                "priority_class": "growth-high",
                "delta_log_mass": 0.01,
                "delta_log_mass_fact_vs_ref": 0.01,
                "delta_log_mass_fact_vs_ref_sign_consistency": 1.0,
                "counterfactual_n_folds": 2,
                "same_gene_n_guides": 2,
                "same_gene_sgrna_concordance": 1.0,
                "delta_autocrine_tnf_tsk_score": 2.0,
                "delta_pemt_score": 0.0,
                "z_delta_autocrine_tnf_tsk_score": 1.0,
                "requested_mass_mode": "count",
            },
        ]
    )

    out = mod._add_biological_gates(df)
    row = out.loc[out["perturbation_id"].eq("GeneA_sg1")].iloc[0]

    assert row["tsk_pemt_program_null_gap_pass"] == np.False_
    assert row["tsk_pemt_claim_ready"] == np.False_


def test_auto_mass_mode_metadata_blocks_claim_ready_when_present() -> None:
    mod = _biology_module()
    df = pd.DataFrame(
        [
            {
                "perturbation_id": "ctrl",
                "target_gene": "control",
                "is_control": True,
                "priority_class": "watch",
                "delta_log_mass": 0.05,
            },
            {
                "perturbation_id": "GeneA_sg1",
                "target_gene": "GENEA",
                "is_control": False,
                "priority_class": "growth-high",
                "delta_log_mass": 1.0,
                "delta_log_mass_fact_vs_ref": 1.0,
                "delta_log_mass_fact_vs_ref_sign_consistency": 1.0,
                "counterfactual_n_folds": 2,
                "same_gene_n_guides": 2,
                "same_gene_sgrna_concordance": 1.0,
                "resolved_mass_mode": "auto",
            },
        ]
    )

    out = mod._add_biological_gates(df)
    row = out.loc[out["perturbation_id"].eq("GeneA_sg1")].iloc[0]

    assert row["explicit_mass_mode_pass"] == np.False_
    assert row["biological_interpretation_gate"] == "needs-explicit-mass-mode"
    assert row["claim_ready"] == np.False_


@pytest.mark.parametrize(
    "resolved",
    [
        "subset_only:count:auto_no_mass_value_col",
        "subset_only:count:auto_missing_mass_value_col:mass_value",
        "subset_only:mass_value:auto_per_cell_contribution",
    ],
)
def test_auto_resolved_mass_mode_strings_block_claim_ready(resolved: str) -> None:
    mod = _biology_module()
    df = pd.DataFrame(
        [
            {
                "perturbation_id": "ctrl",
                "target_gene": "control",
                "is_control": True,
                "priority_class": "watch",
                "delta_log_mass": 0.0,
            },
            {
                "perturbation_id": "GeneA_sg1",
                "target_gene": "GENEA",
                "is_control": False,
                "priority_class": "growth-high",
                "delta_log_mass": 1.0,
                "delta_log_mass_fact_vs_ref": 1.0,
                "delta_log_mass_fact_vs_ref_sign_consistency": 1.0,
                "counterfactual_n_folds": 2,
                "same_gene_n_guides": 2,
                "same_gene_sgrna_concordance": 1.0,
                "resolved_mass_mode": resolved,
            },
        ]
    )

    out = mod._add_biological_gates(df)
    row = out.loc[out["perturbation_id"].eq("GeneA_sg1")].iloc[0]

    assert row["explicit_mass_mode_pass"] == np.False_
    assert row["biological_interpretation_gate"] == "needs-explicit-mass-mode"


def test_degenerate_control_null_uses_positive_practical_floor() -> None:
    mod = _biology_module()
    df = pd.DataFrame(
        [
            {
                "perturbation_id": "ctrl",
                "target_gene": "control",
                "is_control": True,
                "priority_class": "watch",
                "delta_log_mass": 0.0,
            },
            {
                "perturbation_id": "GeneA_sg1",
                "target_gene": "GENEA",
                "is_control": False,
                "priority_class": "growth-high",
                "delta_log_mass": 1e-8,
                "delta_log_mass_fact_vs_ref": 1e-8,
                "delta_log_mass_fact_vs_ref_sign_consistency": 1.0,
                "counterfactual_n_folds": 2,
                "same_gene_n_guides": 2,
                "same_gene_sgrna_concordance": 1.0,
                "requested_mass_mode": "count",
            },
        ]
    )

    out = mod._add_biological_gates(df)
    row = out.loc[out["perturbation_id"].eq("GeneA_sg1")].iloc[0]

    assert row["mass_null_abs_q95"] == 0.0
    assert row["mass_null_threshold"] > 0.0
    assert row["mass_null_gap_pass"] == np.False_
    assert row["biological_interpretation_gate"] == "below-mass-null-gap"


def test_practical_null_floor_override_blocks_tiny_metric_effect() -> None:
    mod = _biology_module()
    original = mod.PRACTICAL_NULL_FLOORS.copy()
    try:
        mod._apply_practical_null_floor_overrides({"mass": 0.5})
        df = pd.DataFrame(
            [
                {
                    "perturbation_id": "ctrl",
                    "target_gene": "control",
                    "is_control": True,
                    "priority_class": "growth-high",
                    "delta_log_mass": 0.0,
                },
                {
                    "perturbation_id": "GeneA_sg1",
                    "target_gene": "GENEA",
                    "is_control": False,
                    "priority_class": "growth-high",
                    "delta_log_mass": 0.25,
                    "delta_log_mass_fact_vs_ref": 0.25,
                    "delta_log_mass_fact_vs_ref_sign_consistency": 1.0,
                    "counterfactual_n_folds": 2,
                    "same_gene_n_guides": 2,
                    "same_gene_sgrna_concordance": 1.0,
                    "requested_mass_mode": "count",
                },
            ]
        )

        out = mod._add_biological_gates(df)
        row = out.loc[out["perturbation_id"].eq("GeneA_sg1")].iloc[0]

        assert row["mass_null_practical_floor"] == 0.5
        assert row["mass_null_gap_pass"] == np.False_
        assert row["expansion_claim_ready"] == np.False_
    finally:
        mod.PRACTICAL_NULL_FLOORS.clear()
        mod.PRACTICAL_NULL_FLOORS.update(original)


def test_practical_null_floor_override_validation(tmp_path: Path) -> None:
    mod = _biology_module()
    floors_path = tmp_path / "floors.json"
    floors_path.write_text(json.dumps({"mass": 0.25, "program_occupancy_tv": 0.1}))

    assert mod._load_practical_null_floor_overrides(str(floors_path)) == {
        "mass": 0.25,
        "program_occupancy_tv": 0.1,
    }
    with pytest.raises(ValueError, match="Unknown practical null floor keys"):
        mod._load_practical_null_floor_overrides('{"unknown": 0.1}')
    with pytest.raises(ValueError, match="finite and nonnegative"):
        mod._load_practical_null_floor_overrides('{"mass": -1.0}')


def test_claim_grade_requires_explicit_floor_profile_or_overrides() -> None:
    mod = _biology_module()
    original = mod.PRACTICAL_NULL_FLOORS.copy()
    try:
        with pytest.raises(ValueError, match="--claim-grade requires"):
            mod._configure_practical_null_floors(claim_grade=True)

        metadata = mod._configure_practical_null_floors(
            profile="hnscc_claim_grade",
            claim_grade=True,
        )
        assert metadata["claim_grade"] is True
        assert metadata["practical_null_floor_profile"] == "hnscc_claim_grade"
        assert mod.PRACTICAL_NULL_FLOORS["mass"] == 0.05
        assert mod.PRACTICAL_NULL_FLOORS["program_occupancy_tv"] == 0.03
    finally:
        mod.PRACTICAL_NULL_FLOORS.clear()
        mod.PRACTICAL_NULL_FLOORS.update(original)


def test_practical_null_floor_cli_json_object_is_applied(tmp_path: Path) -> None:
    run_dir = tmp_path / "cv" / "setting" / "fold_0"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps({"split": {"split_strategy": "random_kfold", "fold_index": 0}})
    )
    (run_dir / "results_summary.json").write_text(json.dumps({"requested_mass_mode": "count"}))
    pd.DataFrame(
        [
            {
                "perturbation_id": "ctrl",
                "mass_pred": 1.0,
                "mass_true": 1.0,
                "mass_rel_error": 0.0,
                "is_control": True,
                "n_init_atoms": 2,
                "n_term_atoms": 2,
            },
            {
                "perturbation_id": "GeneA_sg1",
                "mass_pred": 1.0,
                "mass_true": 1.0,
                "mass_rel_error": 0.0,
                "is_control": False,
                "n_init_atoms": 2,
                "n_term_atoms": 2,
            },
        ]
    ).to_csv(run_dir / "test_endpoint_metrics.csv", index=False)
    cf_path = tmp_path / "cf.csv"
    pd.DataFrame(
        [
            {
                "perturbation_id": "ctrl",
                "fold_id": "fold_0",
                "is_control": True,
                "delta_log_mass_fact_vs_ref": 0.0,
            },
            {
                "perturbation_id": "ctrl",
                "fold_id": "fold_1",
                "is_control": True,
                "delta_log_mass_fact_vs_ref": 0.0,
            },
            {
                "perturbation_id": "GeneA_sg1",
                "fold_id": "fold_0",
                "is_control": False,
                "delta_log_mass_fact_vs_ref": 0.25,
            },
            {
                "perturbation_id": "GeneA_sg1",
                "fold_id": "fold_1",
                "is_control": False,
                "delta_log_mass_fact_vs_ref": 0.25,
            },
        ]
    ).to_csv(cf_path, index=False)
    out_dir = tmp_path / "out"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "analysis" / "extract_biology_effects.py"),
            "--cv-root",
            str(tmp_path / "cv"),
            "--counterfactual-effects",
            str(cf_path),
            "--practical-null-floors-json",
            '{"mass": 0.5}',
            "--output-dir",
            str(out_dir),
        ],
        check=True,
    )
    out = pd.read_csv(out_dir / "biological_effects_per_perturbation.csv")
    row = out.loc[out["perturbation_id"].eq("GeneA_sg1")].iloc[0]
    floor_metadata = json.loads((out_dir / "practical_null_floors_used.json").read_text())

    assert row["mass_null_practical_floor"] == 0.5
    assert row["mass_null_gap_pass"] == np.False_
    assert row["mass_counterfactual_effect"] == 0.25
    assert floor_metadata["claim_grade"] is False
    assert floor_metadata["effective_practical_null_floors"]["mass"] == 0.5
    assert floor_metadata["practical_null_floor_overrides"] == {"mass": 0.5}


def test_mass_null_uses_counterfactual_mass_effect_when_available() -> None:
    mod = _biology_module()
    df = pd.DataFrame(
        [
            {
                "perturbation_id": "ctrl",
                "target_gene": "control",
                "is_control": True,
                "priority_class": "growth-high",
                "delta_log_mass": 10.0,
                "delta_log_mass_fact_vs_ref": 0.0,
            },
            {
                "perturbation_id": "GeneA_sg1",
                "target_gene": "GENEA",
                "is_control": False,
                "priority_class": "growth-high",
                "delta_log_mass": 10.0,
                "delta_log_mass_fact_vs_ref": 0.25,
                "delta_log_mass_fact_vs_ref_sign_consistency": 1.0,
                "counterfactual_n_folds": 2,
                "same_gene_n_guides": 2,
                "same_gene_sgrna_concordance": 1.0,
                "requested_mass_mode": "count",
            },
        ]
    )

    out = mod._add_biological_gates(df)
    row = out.loc[out["perturbation_id"].eq("GeneA_sg1")].iloc[0]

    assert row["mass_counterfactual_effect"] == 0.25
    assert row["mass_null_abs_q95"] == 0.0
    assert row["mass_null_gap_pass"] == np.True_
    assert row["expansion_claim_ready"] == np.True_


def test_any_axis_claim_ready_is_separate_from_priority_class_gate() -> None:
    mod = _biology_module()
    df = pd.DataFrame(
        [
            {
                "perturbation_id": "ctrl",
                "target_gene": "control",
                "is_control": True,
                "priority_class": "growth-high",
                "delta_log_mass": 0.05,
                "delta_autocrine_tnf_tsk_score": 0.0,
                "delta_pemt_score": 0.0,
            },
            {
                "perturbation_id": "GeneA_sg1",
                "target_gene": "GENEA",
                "is_control": False,
                "priority_class": "growth-high",
                "delta_log_mass": 0.01,
                "delta_log_mass_fact_vs_ref": 0.01,
                "delta_log_mass_fact_vs_ref_sign_consistency": 1.0,
                "counterfactual_n_folds": 2,
                "same_gene_n_guides": 2,
                "same_gene_sgrna_concordance": 1.0,
                "delta_autocrine_tnf_tsk_score": 2.0,
                "delta_pemt_score": 0.0,
                "z_delta_autocrine_tnf_tsk_score": 1.0,
                "requested_mass_mode": "count",
                "tsk_pemt_program_effect_pos_fold_support": 1.0,
            },
        ]
    )

    out = mod._add_biological_gates(df)
    row = out.loc[out["perturbation_id"].eq("GeneA_sg1")].iloc[0]

    assert row["priority_class_claim_ready_strict"] == row["claim_ready"]
    assert row["claim_ready_strict"] == row["priority_class_claim_ready_strict"]
    assert row["tsk_pemt_claim_ready"] == np.True_
    assert row["any_axis_claim_ready_strict"] == np.True_


def test_biology_ignores_ambiguous_generic_mass_mode_from_run_config(tmp_path: Path) -> None:
    mod = _biology_module()
    run_dir = tmp_path / "setting" / "fold_0"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "data": {"mass_mode": "auto"},
                "split": {"split_strategy": "random_kfold", "fold_index": 0},
            }
        )
    )
    (run_dir / "results_summary.json").write_text("{}")
    pd.DataFrame(
        [
            {
                "perturbation_id": "GeneA_sg1",
                "mass_pred": 2.0,
                "mass_true": 1.0,
                "mass_rel_error": 0.1,
                "is_control": False,
                "n_init_atoms": 2,
                "n_term_atoms": 3,
            }
        ]
    ).to_csv(run_dir / "test_endpoint_metrics.csv", index=False)

    collected = mod._collect_single_root(tmp_path, split="test")
    aggregated = mod._aggregate(collected)

    assert "requested_mass_mode" not in collected.columns
    assert "resolved_mass_mode" not in collected.columns
    assert "requested_mass_mode" not in aggregated.columns
    assert "resolved_mass_mode" not in aggregated.columns


def test_biology_collects_split_resolved_mass_mode_without_generic_requested_auto(tmp_path: Path) -> None:
    mod = _biology_module()
    run_dir = tmp_path / "setting" / "fold_0"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "mass_mode": "auto",
                "test_mass_mode": "subset_only:count:auto_no_mass_value_col",
                "split": {"split_strategy": "random_kfold", "fold_index": 0},
            }
        )
    )
    (run_dir / "results_summary.json").write_text("{}")
    pd.DataFrame(
        [
            {
                "perturbation_id": "GeneA_sg1",
                "mass_pred": 2.0,
                "mass_true": 1.0,
                "mass_rel_error": 0.1,
                "is_control": False,
                "n_init_atoms": 2,
                "n_term_atoms": 3,
            }
        ]
    ).to_csv(run_dir / "test_endpoint_metrics.csv", index=False)

    collected = mod._collect_single_root(tmp_path, split="test")

    assert "requested_mass_mode" not in collected.columns
    assert collected["resolved_mass_mode"].iloc[0] == "subset_only:count:auto_no_mass_value_col"


def test_results_requested_mass_mode_has_priority_over_generic_config_mass_mode(tmp_path: Path) -> None:
    mod = _biology_module()
    run_dir = tmp_path / "setting" / "fold_0"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "mass_mode": "auto",
                "split": {"split_strategy": "random_kfold", "fold_index": 0},
            }
        )
    )
    (run_dir / "results_summary.json").write_text(json.dumps({"requested_mass_mode": "group_total"}))
    pd.DataFrame(
        [
            {
                "perturbation_id": "GeneA_sg1",
                "mass_pred": 2.0,
                "mass_true": 1.0,
                "mass_rel_error": 0.1,
                "is_control": False,
                "n_init_atoms": 2,
                "n_term_atoms": 3,
            }
        ]
    ).to_csv(run_dir / "test_endpoint_metrics.csv", index=False)

    collected = mod._collect_single_root(tmp_path, split="test")

    assert collected["requested_mass_mode"].iloc[0] == "group_total"
    assert collected["resolved_mass_mode"].iloc[0] == "group_total"


def test_results_requested_mass_mode_overrides_stale_config_requested_mode(tmp_path: Path) -> None:
    mod = _biology_module()
    run_dir = tmp_path / "setting" / "fold_0"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "requested_mass_mode": "auto",
                "split": {"split_strategy": "random_kfold", "fold_index": 0},
            }
        )
    )
    (run_dir / "results_summary.json").write_text(json.dumps({"requested_mass_mode": "count"}))
    pd.DataFrame(
        [
            {
                "perturbation_id": "GeneA_sg1",
                "mass_pred": 2.0,
                "mass_true": 1.0,
                "mass_rel_error": 0.1,
                "is_control": False,
                "n_init_atoms": 2,
                "n_term_atoms": 3,
            }
        ]
    ).to_csv(run_dir / "test_endpoint_metrics.csv", index=False)

    collected = mod._collect_single_root(tmp_path, split="test")

    assert collected["requested_mass_mode"].iloc[0] == "count"


def test_generic_mass_mode_column_does_not_make_claim_ready_explicit() -> None:
    mod = _biology_module()
    df = pd.DataFrame(
        [
            {
                "perturbation_id": "ctrl",
                "target_gene": "control",
                "is_control": True,
                "priority_class": "watch",
                "delta_log_mass": 0.0,
            },
            {
                "perturbation_id": "GeneA_sg1",
                "target_gene": "GENEA",
                "is_control": False,
                "priority_class": "growth-high",
                "delta_log_mass": 1.0,
                "delta_log_mass_fact_vs_ref": 1.0,
                "delta_log_mass_fact_vs_ref_sign_consistency": 1.0,
                "counterfactual_n_folds": 2,
                "same_gene_n_guides": 2,
                "same_gene_sgrna_concordance": 1.0,
                "mass_mode": "count",
            },
        ]
    )

    out = mod._add_biological_gates(df)
    row = out.loc[out["perturbation_id"].eq("GeneA_sg1")].iloc[0]

    assert row["explicit_mass_mode_pass"] == np.False_
    assert row["biological_interpretation_gate"] == "needs-explicit-mass-mode"


def test_string_false_control_flag_is_not_control_for_nulls() -> None:
    mod = _biology_module()
    df = pd.DataFrame(
        [
            {
                "perturbation_id": "ctrl",
                "target_gene": "control",
                "is_control": "True",
                "priority_class": "watch",
                "delta_log_mass": 0.05,
            },
            {
                "perturbation_id": "GeneA_sg1",
                "target_gene": "GENEA",
                "is_control": "False",
                "priority_class": "growth-high",
                "delta_log_mass": 1.0,
                "delta_log_mass_fact_vs_ref": 1.0,
                "delta_log_mass_fact_vs_ref_sign_consistency": 1.0,
                "counterfactual_n_folds": 2,
                "same_gene_n_guides": 2,
                "same_gene_sgrna_concordance": 1.0,
                "requested_mass_mode": "count",
            },
        ]
    )

    out = mod._add_biological_gates(df)
    row = out.loc[out["perturbation_id"].eq("GeneA_sg1")].iloc[0]

    assert out.loc[out["perturbation_id"].eq("GeneA_sg1"), "is_control"].iloc[0] == np.False_
    assert row["mass_null_abs_q95"] == 0.05
    assert row["mass_null_abs_q95"] < row["delta_log_mass"]


def test_invalid_control_flag_string_raises() -> None:
    mod = _biology_module()
    df = pd.DataFrame(
        [
            {
                "perturbation_id": "ctrl",
                "target_gene": "control",
                "is_control": "maybe",
                "priority_class": "watch",
                "delta_log_mass": 0.0,
            }
        ]
    )

    with pytest.raises(ValueError, match="Cannot parse boolean"):
        mod._add_biological_gates(df)


def test_counterfactual_tensor_hashes_reflect_equality() -> None:
    analysis_dir = str(ROOT / "analysis")
    if analysis_dir not in sys.path:
        sys.path.insert(0, analysis_dir)
    from run_counterfactual_biology import _tensor_sha256
    import torch

    x = torch.tensor([[1.0, 2.0]])
    y = x.clone()
    z = torch.tensor([[1.0, 3.0]])

    assert _tensor_sha256(x) == _tensor_sha256(y)
    assert _tensor_sha256(x) != _tensor_sha256(z)


def test_energy_distance_detects_same_mean_distribution_shift() -> None:
    analysis_dir = str(ROOT / "analysis")
    if analysis_dir not in sys.path:
        sys.path.insert(0, analysis_dir)
    from run_counterfactual_biology import _weighted_energy_distance
    import torch

    factual = torch.tensor([[-1.0], [1.0]])
    reference = torch.tensor([[0.0], [0.0]])
    logw = torch.zeros(2)
    mean_f = (torch.softmax(logw, dim=0)[:, None] * factual).sum(dim=0)
    mean_r = (torch.softmax(logw, dim=0)[:, None] * reference).sum(dim=0)

    assert torch.linalg.norm(mean_f - mean_r).item() == 0.0
    assert _weighted_energy_distance(factual, logw, reference, logw) > 0.1


def test_energy_distance_chunked_matches_full_pairwise() -> None:
    analysis_dir = str(ROOT / "analysis")
    if analysis_dir not in sys.path:
        sys.path.insert(0, analysis_dir)
    from run_counterfactual_biology import _weighted_energy_distance
    import torch

    z_a = torch.linspace(-1.0, 1.0, 9).reshape(9, 1)
    z_b = torch.linspace(-0.5, 1.5, 7).reshape(7, 1)
    logw_a = torch.linspace(-0.2, 0.2, 9)
    logw_b = torch.linspace(0.3, -0.1, 7)

    full = _weighted_energy_distance(z_a, logw_a, z_b, logw_b, chunk_size=100)
    chunked = _weighted_energy_distance(z_a, logw_a, z_b, logw_b, chunk_size=3)

    assert np.isclose(full, chunked)


def test_program_occupancy_tv_detects_program_fraction_shift() -> None:
    analysis_dir = str(ROOT / "analysis")
    if analysis_dir not in sys.path:
        sys.path.insert(0, analysis_dir)
    import torch

    factual = torch.tensor([0.7, 0.2, 0.1])
    reference = torch.tensor([0.2, 0.2, 0.6])
    tv = 0.5 * torch.abs(factual - reference).sum().item()

    assert np.isclose(tv, 0.5)


def test_counterfactual_pid_selection_can_include_controls_for_nulls() -> None:
    analysis_dir = str(ROOT / "analysis")
    if analysis_dir not in sys.path:
        sys.path.insert(0, analysis_dir)
    from run_counterfactual_biology import _select_counterfactual_pids

    selected = _select_counterfactual_pids(
        ["ctrl", "GeneA_sg1", "GeneB_sg1"],
        {"ctrl"},
        [],
        include_controls_for_null=True,
        max_perturbations=1,
    )

    assert selected == ["GeneA_sg1", "ctrl"]


def test_counterfactual_mass_mode_requires_explicit_requested_mode() -> None:
    analysis_dir = str(ROOT / "analysis")
    if analysis_dir not in sys.path:
        sys.path.insert(0, analysis_dir)
    from run_counterfactual_biology import _requested_mass_mode_for_counterfactual

    with pytest.raises(ValueError, match="requires explicit requested_mass_mode"):
        _requested_mass_mode_for_counterfactual(
            {"mass_mode": "auto", "test_mass_mode": "subset_only:count:auto_no_mass_value_col"},
            {},
            "test",
        )


def test_counterfactual_mass_mode_uses_results_requested_mode() -> None:
    analysis_dir = str(ROOT / "analysis")
    if analysis_dir not in sys.path:
        sys.path.insert(0, analysis_dir)
    from run_counterfactual_biology import _requested_mass_mode_for_counterfactual

    mode = _requested_mass_mode_for_counterfactual(
        {"mass_mode": "auto"},
        {"requested_mass_mode": "group_total"},
        "test",
    )

    assert mode == "group_total"


def test_counterfactual_test_split_does_not_use_train_mass_mode_as_requested() -> None:
    analysis_dir = str(ROOT / "analysis")
    if analysis_dir not in sys.path:
        sys.path.insert(0, analysis_dir)
    from run_counterfactual_biology import _requested_mass_mode_for_counterfactual

    with pytest.raises(ValueError, match="requires explicit requested_mass_mode"):
        _requested_mass_mode_for_counterfactual(
            {"train_mass_mode": "subset_only:mass_value:group_total"},
            {},
            "test",
        )


def test_score_hnscc_signatures_smoke(tmp_path: Path) -> None:
    x = np.array(
        [
            [5.0, 4.0, 0.0, 0.0, 3.0, 3.0],
            [4.0, 5.0, 0.0, 0.0, 3.0, 3.0],
            [0.0, 0.0, 5.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 5.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    obs = pd.DataFrame(
        {
            "target_gene": ["GeneA", "GeneA", "GeneA", "GeneA"],
            "perturbation_gene": ["GeneA", "GeneA", "GeneA", "GeneA"],
            "is_control": [False, False, False, False],
            "Time point": [4, 4, 60, 60],
            "Library": ["L1", "L1", "L1", "L1"],
            "cell_id": ["c1", "c2", "c3", "c4"],
            "Cell type annotation": ["basal", "basal", "TSK", "TSK"],
        }
    )
    var = pd.DataFrame(index=["JUN", "FOS", "TNF", "MMP9", "Trp63", "Atp1b3"])
    data_path = tmp_path / "mini.h5ad"
    ad.AnnData(X=x, obs=obs, var=var).write_h5ad(data_path)
    out_dir = tmp_path / "sig"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "analysis" / "score_hnscc_signatures.py"),
            "--data-path",
            str(data_path),
            "--output-dir",
            str(out_dir),
        ],
        check=True,
    )
    scores = pd.read_csv(out_dir / "signature_group_scores.csv")
    assert {"tnf_expansion", "autocrine_tnf_tsk", "n_cells"} <= set(scores.columns)
    coverage = pd.read_csv(out_dir / "signature_gene_coverage.csv")
    cis = coverage.loc[coverage["signature"].eq("cis_like")].iloc[0]
    assert "Trp63" in cis["matched_genes"]
    assert "TP63" not in str(cis["missing_genes"])
    p4 = scores.loc[scores["time_label"].eq("P4"), "tnf_expansion"].mean()
    p60 = scores.loc[scores["time_label"].eq("P60"), "autocrine_tnf_tsk"].mean()
    assert p4 > 0
    assert p60 > 0
