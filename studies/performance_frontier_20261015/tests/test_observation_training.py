"""Eight toy CPU optimizer calls per full file; no real counts or scientific fits."""

from copy import deepcopy
import random

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
import torch

from frontier.checkpoints import TrajectoryLease, load_checkpoint
from frontier.native_inputs import NativeBatch
from frontier.observation import ObservationFold, technical_holdout
from frontier.observation_training import (LogicalRowSampler, ObservationFitter, ObservationRows,
                                           ObservationTrainingSpec)


def source():
    fold = ObservationFold("D3-D4", ("D3", "D4"), "f" * 64)
    # Fixed fixture IDs deliberately supply both hash-defined technical roles.
    ids = [f'D3_Stim8hr:cell-{index}' for index in range(64)]
    held = technical_holdout(ids, fold=fold)
    assert 0 < held.sum() < len(ids)
    matrix = sparse.csr_matrix(np.asarray([[3 + index % 3, 0, 2, 1, 0, 4, 1]
                                          for index in range(64)], dtype=np.uint32))
    # One zero-RNA cell tests distinct eligibility; preserve at least one holdout.
    zero = next(index for index in range(64) if not held[index])
    matrix.data[matrix.indptr[zero]:matrix.indptr[zero + 1]] = 0
    matrix.eliminate_zeros()
    batches = []
    for start in range(0, 64, 4):
        counts = matrix[start:start + 4]
        depth = np.asarray(counts.sum(1)).ravel()
        cells = pd.DataFrame({"source_id": ["D3_Stim8hr"] * 4, "donor_id": ["D3"] * 4,
                              "condition": ["Stim8hr"] * 4})
        batches.append(NativeBatch(counts, cells, tuple(ids[start:start + 4]), depth, depth > 0))
    return ObservationRows.fabricated(batches, fold)


def binding(rows, spec, trajectory):
    return {"trajectory_id": trajectory, "fold_id": rows.fold.fold_id, "seed": 0,
            "input_sha256": rows.identity, "configuration_sha256": spec.identity, "source_sha256": "e" * 64}


def equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal_tree(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            equal_tree(a, b)
    else:
        assert left == right


def test_exact_four_update_uninterrupted_vs_two_plus_two_resume_and_final_scale(tmp_path):
    # 4 + 2 + 2 = exactly eight disposable optimizer calls.
    rows, spec = source(), ObservationTrainingSpec.fixture()
    with TrajectoryLease(tmp_path / "uninterrupted", binding(rows, spec, "reference"), "cpu-reference") as lease:
        full = ObservationFitter(lease, rows, spec, device="cpu")
        report = full.run_until(4)
        expected_model = {key: value.clone() for key, value in full.model.state_dict().items()}
        expected_optimizer = deepcopy(full.optimizer.state_dict())
        expected_sampler = deepcopy(full.sampler.state)
        selected, publication = full.finalize_selected()
        nonheld = list(rows.partition(holdout=False, batch_size=spec.logical_batch))
        expected_cells = sum(len(batch.row_ids) for batch in nonheld)
        assert publication["scaling"]["positive_RNA_cells"] == expected_cells
        assert publication["completed_updates"] == 4 and publication["optimizer_resume"] is False
        assert publication["selected_updates"] == min(full.audit_history, key=lambda item: (item["risk"], item["accepted_updates"]))["accepted_updates"]
        assert bool(selected.scaling_ready) and not any(p.requires_grad for p in selected.parameters())
        assert not bool(full.model.scaling_ready)  # selection/scaling never mutate terminal optimizer model
        # Audit candidate cannot be loaded as a resumable checkpoint.
        with pytest.raises(ValueError, match="latest complete"):
            load_checkpoint(lease, lease.root, expected_sha256=publication["publication"]["sha256"],
                            model=full.model, optimizer=full.optimizer, scheduler=None, execution_scope="synthetic_cpu_only")
        with pytest.raises(RuntimeError, match="healthy completed"):
            full.finalize_selected()
    assert report["fitting_row_exposures"] == 32 and report["scientific_updates"] == 0
    assert report["optimizer_calls"] == report["disposable_optimizer_calls"] == 4
    assert [row["thinning_seed"] for row in report["training_history"]] == [0, 1, 2, 3]
    with TrajectoryLease(tmp_path / "resumed", binding(rows, spec, "resume"), "cpu-first") as lease:
        first = ObservationFitter(lease, rows, spec, device="cpu")
        first.run_until(2)
        checkpoint = first.last_checkpoint
    random.seed(930)
    np.random.seed(930)
    torch.manual_seed(930)
    with TrajectoryLease(tmp_path / "resumed", binding(rows, spec, "resume"), "cpu-second") as lease:
        with pytest.raises(ValueError, match="explicit resume"):
            ObservationFitter(lease, rows, spec, device="cpu")
        second = ObservationFitter.resume(lease, rows, spec, device="cpu", directory=checkpoint["directory"],
                                         expected_sha256=checkpoint["checkpoint_sha256"])
        resumed = second.run_until(4)
        equal_tree(expected_model, second.model.state_dict())
        equal_tree(expected_optimizer, second.optimizer.state_dict())
        equal_tree(expected_sampler, second.sampler.state)
        assert [r["risk"] for r in resumed["audit_history"]] == [r["risk"] for r in report["audit_history"]]
        assert resumed["fitting_row_exposures"] == 32
        assert resumed["training_history"] == report["training_history"]
        assert resumed["accepted_updates"] == resumed["attempted_updates"] == 4


def test_sampler_exact_remainder_excludes_holdout_and_zero_and_crosses_epochs():
    rows = source()
    sampler = LogicalRowSampler(rows, 8)
    first = sampler.next()
    assert first.positive_rna.all() and not technical_holdout(first.row_ids, fold=rows.fold).any()
    saved = deepcopy(sampler.state)
    expected = sampler.next()
    restored = LogicalRowSampler(rows, 8, saved)
    actual = restored.next()
    assert actual.row_ids == expected.row_ids and (actual.rna_counts != expected.rna_counts).nnz == 0
    assert restored.state == sampler.state
    for _ in range(8):
        batch = sampler.next()
        assert len(batch.row_ids) == 8 and not technical_holdout(batch.row_ids, fold=rows.fold).any()
    assert sampler.state["epoch"] >= 1 and sampler.state["position"] == 80


def test_audit_is_independent_full_holdout_fixed_seed_and_does_not_move_optimizer_or_rng(tmp_path):
    rows, spec = source(), ObservationTrainingSpec.fixture()
    with TrajectoryLease(tmp_path / "audit", binding(rows, spec, "audit"), "cpu-audit") as lease:
        owner = ObservationFitter(lease, rows, spec, device="cpu")
        sampler, optimizer = deepcopy(owner.sampler.state), deepcopy(owner.optimizer.state_dict())
        random.seed(81)
        np.random.seed(81)
        torch.manual_seed(81)
        py, npstate, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        first, second = owner._audit(), owner._audit()
        assert first == second and first["positive_RNA_cells"] == sum(len(b.row_ids) for b in rows.partition(holdout=True, batch_size=8))
        assert first["risk"] == first["sum_normalized_cross_entropy"] / first["positive_cell_directions"]
        assert owner.sampler.state == sampler and owner.accepted_updates == owner.attempted_updates == 0
        equal_tree(owner.optimizer.state_dict(), optimizer)
        assert random.getstate() == py and np.array_equal(np.random.get_state()[1], npstate[1])
        assert torch.equal(torch.get_rng_state(), cpu)


def test_failure_stops_owner_without_checkpoint_or_automatic_retry(tmp_path, monkeypatch):
    # Failure occurs before optimizer.step: zero disposable optimizer calls.
    rows, spec = source(), ObservationTrainingSpec.fixture()
    with TrajectoryLease(tmp_path / "failure", binding(rows, spec, "failure"), "cpu-failure") as lease:
        owner = ObservationFitter(lease, rows, spec, device="cpu")
        monkeypatch.setattr(owner.model, "symmetric_count_loss", lambda *a, **k: {"loss": torch.tensor(float("nan"))})
        with pytest.raises(ValueError, match="finite"):
            owner.run_until(2)
        assert owner.failed and owner.attempted_updates == 1 and owner.accepted_updates == 0
        assert owner.progress()["optimizer_calls"] == owner.progress()["disposable_optimizer_calls"] == 0
        assert not list((lease.root / "checkpoints").iterdir())
        with pytest.raises(RuntimeError, match="retry"):
            owner.run_until(2)
        assert (lease.root / "OBSERVATION_FAILED.json").exists()
        with pytest.raises(RuntimeError, match="retained failed"):
            ObservationFitter(lease, rows, spec, device="cpu")


def test_scientific_defaults_cannot_accidentally_run_as_cpu_fixture(tmp_path):
    rows, small = source(), ObservationTrainingSpec.fixture()
    scientific = ObservationTrainingSpec()
    assert scientific.maximum_updates == 20000 and scientific.logical_batch == 2048
    assert scientific.maximum_dense_bytes == 4 * 1024 ** 3
    with pytest.raises(ValueError, match="changed"):
        ObservationTrainingSpec(maximum_updates=4, audit_updates=(2, 4))
    with pytest.raises(ValueError, match="budget"):
        ObservationTrainingSpec.fixture(maximum_updates=9, audit_updates=(9,))
    with TrajectoryLease(tmp_path / "scope", binding(rows, small, "scope"), "cpu-scope") as lease:
        with pytest.raises(ValueError, match="bindings"):
            ObservationFitter(lease, rows, scientific, device="cpu")
        with pytest.raises(ValueError, match="fixtures require CPU"):
            ObservationFitter(lease, rows, small, device="cuda")
        owner = ObservationFitter(lease, rows, small, device="cpu")
        with pytest.raises(ValueError, match="boundary"):
            owner.run_until(3)
        with pytest.raises(RuntimeError, match="healthy completed"):
            owner.finalize_selected()
