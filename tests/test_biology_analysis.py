from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


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
    assert np.isclose(notch["diffusion_action"], 0.6)
    assert np.isclose(notch["context_dependence_geom"], 0.7)
    assert notch["counterfactual_n_folds"] == 2
    assert notch["delta_log_mass_fact_vs_ref_std"] > 0
    assert "priority_class_v2" in out.columns
    assert "biological_interpretation_gate" in out.columns
    assert "fold_stability_pass" in out.columns
    assert "guide_concordance_pass" in out.columns
    assert "negative_control_gap_pass" in out.columns
    assert notch["biological_interpretation_gate"] == "needs-fold-stability"
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
