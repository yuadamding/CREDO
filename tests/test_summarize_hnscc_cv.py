from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from runners.summarize_hnscc_cv import build_group_summary, collect_run_rows, write_markdown


pytestmark = pytest.mark.runner


def test_summarizer_handles_missing_state_metrics(tmp_path: Path) -> None:
    run_dir = tmp_path / "setting_a" / "fold_0"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps({"split": {"split_strategy": "random_kfold", "fold_index": 0}, "config": {}})
    )
    (run_dir / "results_summary.json").write_text(
        json.dumps(
            {
                "control_mode": "soft_ref",
                "test_summary": {"mean_uot": 1.0, "mean_mass_rel_error": 0.1},
                "train_summary": {"mean_uot": 0.9, "mean_mass_rel_error": 0.08},
                "test_state_summary": None,
                "train_state_summary": None,
                "train_time_s": 12.0,
            }
        )
    )

    rows = collect_run_rows(tmp_path)
    per_fold = pd.DataFrame(rows)
    summary = build_group_summary(per_fold, group_by="setting", ranking_mode="balanced")
    out = tmp_path / "cv_summary.md"
    write_markdown(summary, per_fold, out, group_by="setting", ranking_mode="balanced")

    text = out.read_text()
    assert "setting_a" in text
    assert "n/a" in text


def test_summarizer_ranks_and_labels_test_acc(tmp_path: Path) -> None:
    for setting, acc, uot in [("setting_low", 0.25, 0.7), ("setting_high", 0.75, 1.1)]:
        run_dir = tmp_path / setting / "fold_0"
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text(
            json.dumps({"split": {"split_strategy": "random_kfold", "fold_index": 0}, "config": {}})
        )
        (run_dir / "results_summary.json").write_text(
            json.dumps(
                {
                    "control_mode": "soft_ref",
                    "test_summary": {"mean_uot": uot, "mean_mass_rel_error": 0.1},
                    "train_summary": {"mean_uot": 0.5, "mean_mass_rel_error": 0.08},
                    "test_state_summary": {
                        "mean_state_tv": 0.2,
                        "dominant_state_accuracy": acc,
                        "mean_abs_expansion_ratio_gap": 0.3,
                    },
                    "train_state_summary": {
                        "mean_state_tv": 0.1,
                        "dominant_state_accuracy": acc,
                        "mean_abs_expansion_ratio_gap": 0.2,
                    },
                    "train_time_s": 12.0,
                }
            )
        )

    rows = collect_run_rows(tmp_path)
    per_fold = pd.DataFrame(rows)
    summary = build_group_summary(per_fold, group_by="setting", ranking_mode="test_acc")
    out = tmp_path / "cv_summary.md"
    write_markdown(summary, per_fold, out, group_by="setting", ranking_mode="test_acc")

    text = out.read_text()
    assert "| mean test acc | std test acc |" in text
    assert "| test acc |" in text
    assert summary.loc[0, "setting_name"] == "setting_high"
    assert summary.loc[0, "mean_test_acc"] == 0.75


def test_summarizer_ranks_test_uot(tmp_path: Path) -> None:
    for setting, acc, uot in [("setting_low_uot", 0.25, 0.7), ("setting_high_acc", 0.75, 1.1)]:
        run_dir = tmp_path / setting / "fold_0"
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text(
            json.dumps({"split": {"split_strategy": "random_kfold", "fold_index": 0}, "config": {}})
        )
        (run_dir / "results_summary.json").write_text(
            json.dumps(
                {
                    "control_mode": "soft_ref",
                    "test_summary": {"mean_uot": uot, "mean_mass_rel_error": 0.1},
                    "train_summary": {"mean_uot": 0.5, "mean_mass_rel_error": 0.08},
                    "test_state_summary": {
                        "mean_state_tv": 0.2,
                        "dominant_state_accuracy": acc,
                        "mean_abs_expansion_ratio_gap": 0.3,
                    },
                    "train_state_summary": {
                        "mean_state_tv": 0.1,
                        "dominant_state_accuracy": acc,
                        "mean_abs_expansion_ratio_gap": 0.2,
                    },
                    "train_time_s": 12.0,
                }
            )
        )

    rows = collect_run_rows(tmp_path)
    per_fold = pd.DataFrame(rows)
    summary = build_group_summary(per_fold, group_by="setting", ranking_mode="test_uot")
    out = tmp_path / "cv_summary.md"
    write_markdown(summary, per_fold, out, group_by="setting", ranking_mode="test_uot")

    text = out.read_text()
    assert "Ranking mode: `test_uot`" in text
    assert summary.loc[0, "setting_name"] == "setting_low_uot"
    assert summary.loc[0, "mean_test_uot"] == 0.7
