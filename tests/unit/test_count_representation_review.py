from __future__ import annotations

import json
import runpy
from pathlib import Path

import numpy as np
import pytest
import torch

from credo_count_sde_v4.canonical import sha256_file
from credo_count_sde_v4.count_representation.artifacts import runtime_check
from credo_count_sde_v4.count_representation.contracts import CountRepresentationSpec
from credo_count_sde_v4.count_representation.stream import (
    halves,
    iter_counts,
    partition_audit,
    schedule_receipt,
    training_schedule,
)
from credo_count_sde_v4.count_representation.workflow import (
    calibrate_representation,
    fit_priors,
    refit_representation,
)
from credo_count_sde_v4.errors import ContractError

fixture = runpy.run_path(str(Path(__file__).with_name("test_fold_count_representation.py")))[
    "fixture"
]


def test_interleaving_replay_epoch_rotation_unequal_sources_and_exact_rows(tmp_path):
    root, spec = fixture(tmp_path / "inputs", epochs=(1,))
    payload = spec.model_dump()
    payload["fitting"]["shards"] = [
        s
        for s in payload["fitting"]["shards"]
        if not (s["source_id"] == "fit-a-source" and s["shard"] == 1)
    ]
    payload["fitting"]["shards"][0]["allowed_rows"] = [0, 2, 4]
    spec = CountRepresentationSpec.model_validate(payload)
    seen_schedules, last_sources = [], []
    expected = {
        (r.source_id, r.shard, row)
        for r in spec.fitting.shards
        for row in (range(r.rows) if r.allowed_rows is None else r.allowed_rows)
    }
    for epoch in range(8):
        schedule = training_schedule(spec, epoch)
        assert schedule == training_schedule(spec, epoch)
        assert (
            len(schedule)
            == len(spec.fitting.shards)
            == len(set((r.source_id, r.shard) for r in schedule))
        )
        receipt = schedule_receipt(spec, epoch)
        seen_schedules.append(receipt["ordered_shards_sha256"])
        last_sources.append(receipt["last_source"])
        addresses = [
            (row.source_id, row.shard, row.row_in_shard)
            for _, cells in iter_counts(root, spec, training=True, epoch=epoch)
            for row in cells.itertuples(index=False)
        ]
        assert len(addresses) == len(set(addresses)) and set(addresses) == expected
        assert addresses[-1][0] == receipt["last_source"]
    assert len(set(seen_schedules)) >= 4
    assert set(last_sources[:4]) == set(spec.fitting.source_ids)
    assert set(last_sources[4:]) == set(spec.fitting.source_ids)
    changed = spec.model_dump()
    changed["fitting"]["package_completion_sha256"] = "e" * 64
    changed["query"]["package_completion_sha256"] = "e" * 64
    for shard in changed["query"]["shards"]:
        shard["counts"]["sha256"] = "0" * 64
    assert training_schedule(
        CountRepresentationSpec.model_validate(changed), 1
    ) == training_schedule(spec, 1)
    with pytest.raises(ContractError, match="epoch"):
        training_schedule(spec, -1)
    with pytest.raises(ContractError, match="cannot consume"):
        next(iter_counts(root, spec, training=True, split="query"))


def test_objective_matched_constant_uses_only_scored_training_halves(tmp_path, monkeypatch):
    root, spec = fixture(tmp_path / "inputs", depth_imbalance=True, zero=False)
    partition, _ = partition_audit(root, spec)
    originals = list(iter_counts(root, spec, partition=partition, split="train"))
    rows = {"source": [], "destination": []}
    for counts, cells in originals:
        for target in halves(counts, cells, spec.rules.seed):
            depth = np.asarray(target.sum(axis=1)).ravel()
            for condition in rows:
                selected = cells.condition_role.eq(condition).to_numpy() & (depth > 0)
                rows[condition].extend(target[selected].toarray() / depth[selected, None])

    def training_only(*args, **kwargs):
        assert kwargs["split"] == "train" and kwargs["partition"] == partition
        yield from originals

    monkeypatch.setattr(
        "credo_count_sde_v4.count_representation.workflow.iter_counts", training_only
    )
    priors, report = fit_priors(root, spec, partition)
    for i, condition in enumerate(rows):
        reference = np.asarray(rows[condition]).mean(0)
        matched, pooled = priors["condition_cell_direction"][i], priors["condition_composition"][i]
        assert np.allclose(matched, reference, atol=1e-14)
        assert (matched > 0).all() and matched.sum() == pytest.approx(1)
        assert report["scored_training_cell_directions"][condition] == len(rows[condition])
        matched_ce = -float(reference @ np.log(matched))
        pooled_ce = -float(reference @ np.log(pooled))
        assert matched_ce < pooled_ce - 0.25
    assert report["audit_cells_used"] == report["query_cells_used"] == 0


def test_distinct_source_distributions_report_separate_audit_errors(tmp_path):
    root, spec = fixture(tmp_path / "inputs", source_shift=True, epochs=(1, 2))
    calibrate_representation(root, spec, tmp_path / "cal")
    report = json.loads((tmp_path / "cal/calibration.json").read_text())
    scores = report["candidates"][-1]["source_scores"]
    for family in report["candidates"][-1]["models"]:
        selected = [r for r in scores if r["family"] == family]
        assert {r["source_id"] for r in selected} == set(spec.fitting.source_ids)
        assert all(np.isfinite(r["mean_cell_direction_CE"]) for r in selected)
        assert (
            sum(r["scored_cell_directions"] for r in selected)
            == report["candidates"][-1]["models"][family]["scored_cell_directions"]
        )
    constant = [
        r["mean_cell_direction_CE"] for r in scores if r["family"] == "condition_cell_direction"
    ]
    assert np.ptp(constant) > 0.01
    assert report["primary_constant_comparator"] == "condition_cell_direction"


def test_failed_gate_stops_before_refit_unless_explicit_diagnostic(tmp_path, monkeypatch):
    root, spec = fixture(tmp_path / "inputs", null=True, epochs=(1,), zero=False)
    record = calibrate_representation(root, spec, tmp_path / "cal")
    assert not record.facts["heldout_count_gate"]
    with monkeypatch.context() as context:
        context.setattr(
            "credo_count_sde_v4.count_representation.workflow.initialize",
            lambda *a, **k: pytest.fail("Stopped refit must not initialize or read counts"),
        )
        with pytest.raises(ContractError, match="gate failed"):
            refit_representation(root, tmp_path / "cal", tmp_path / "blocked")
    assert not (tmp_path / "blocked").exists()
    with pytest.raises(ContractError, match="nonempty"):
        refit_representation(root, tmp_path / "cal", tmp_path / "empty", diagnostic_reason=" ")
    fitted = refit_representation(
        root,
        tmp_path / "cal",
        tmp_path / "diagnostic",
        diagnostic_reason="Synthetic failed-gate lifecycle test only",
    )
    assert fitted.facts["diagnostic_refit"] and not fitted.facts["heldout_count_gate"]
    assert not fitted.facts["representation_qualified"] and not fitted.facts["scientific_promotion"]


def test_numerical_settings_are_frozen_and_v1_is_not_reinterpreted(tmp_path):
    _, spec = fixture(tmp_path / "inputs")
    before = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(not before)
        with pytest.raises(ContractError, match="runtime"):
            runtime_check(spec)
    finally:
        torch.use_deterministic_algorithms(before, warn_only=warn)
    payload = spec.model_dump()
    payload["schema_version"] = 1
    with pytest.raises(ValueError):
        CountRepresentationSpec.model_validate(payload)


def test_original_v1_representation_schemas_are_immutable():
    schemas = Path(__file__).resolve().parents[2] / "schemas"
    assert sha256_file(schemas / "fold-count-representation-spec.v1.json") == (
        "9033b305dd5909d8eb8a052cf1eb8ee0d29637428c461835e7f4bc3fd573c0b7"
    )
    assert sha256_file(schemas / "fold-count-representation-bundle.v1.json") == (
        "bb47c6abd7d51949f69df4be8b01803e2b7cd79d8f84a47d6ddcccf39eacb8ad"
    )
