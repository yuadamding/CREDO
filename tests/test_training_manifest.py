from __future__ import annotations

import json

import pytest

from credo.training.manifest import append_run_manifest_record, build_run_manifest, write_run_manifest


pytestmark = pytest.mark.unit


def test_build_run_manifest_records_core_reproducibility_fields(tmp_path) -> None:
    config = {
        "model": {"context_kind": "transformer"},
        "training": {
            "global_context_batching": "full_context_cache",
            "ess_warn_frac": 0.2,
            "ess_fail_frac": 0.05,
            "ess_claim_grade_min_frac": 0.1,
            "ess_max_weight_frac_fail": 0.5,
        },
    }

    manifest = build_run_manifest(
        config=config,
        supported_pids=["ctrl", "gene_a"],
        active_pids=["ctrl"],
        stage="C",
        n_epochs=3,
        output_dir=tmp_path,
    )
    path = write_run_manifest(tmp_path / "run_manifest.json", manifest)
    loaded = json.loads(path.read_text())

    assert loaded["context_kind"] == "transformer"
    assert loaded["global_context_batching"] == "full_context_cache"
    assert loaded["manifest_schema_version"] == 2
    assert loaded["output_dir"] == str(tmp_path)
    assert len(loaded["config_sha256"]) == 64
    assert loaded["command"]
    assert loaded["cwd"]
    assert isinstance(loaded["git_available"], bool)
    assert loaded["git_dirty"] in {True, False, None}
    assert loaded["supported_perturbation_count"] == 2
    assert loaded["active_perturbation_ids"] == ["ctrl"]
    assert loaded["ess_thresholds"]["ess_claim_grade_min_frac"] == 0.1
    assert loaded["checkpoint_schema_version"] == 1


def test_append_run_manifest_record_preserves_stage_sequence(tmp_path) -> None:
    path = tmp_path / "run_manifest_stages.jsonl"
    append_run_manifest_record(path, {"stage": "C", "n_epochs": 1})
    append_run_manifest_record(path, {"stage": "D", "n_epochs": 2})

    rows = [json.loads(line) for line in path.read_text().splitlines()]

    assert rows == [{"n_epochs": 1, "stage": "C"}, {"n_epochs": 2, "stage": "D"}]
