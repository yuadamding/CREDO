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
                    "mass_rel_error": 0.0,
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
    assert notch["shared_guide_null_gap"] > 0


def test_score_hnscc_signatures_smoke(tmp_path: Path) -> None:
    x = np.array(
        [
            [5.0, 4.0, 0.0, 0.0],
            [4.0, 5.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 4.0],
            [0.0, 0.0, 4.0, 5.0],
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
    var = pd.DataFrame(index=["JUN", "FOS", "TNF", "MMP9"])
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
    p4 = scores.loc[scores["time_label"].eq("P4"), "tnf_expansion"].mean()
    p60 = scores.loc[scores["time_label"].eq("P60"), "autocrine_tnf_tsk"].mean()
    assert p4 > 0
    assert p60 > 0
