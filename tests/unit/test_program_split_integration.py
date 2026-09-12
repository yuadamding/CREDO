from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from credo_count_sde_v4.contracts import ProgramSplitKind
from credo_count_sde_v4.errors import ContractError
from credo_count_sde_v4.programs import ProgramBatch, ProgramDataset, ProgramFitConfig
from credo_count_sde_v4.programs import qualification as q
from credo_count_sde_v4.programs.head import ProgramFitResult, iter_program_predictions
from credo_count_sde_v4.programs.keyed_metrics import GeneSignAccumulator, gene_sign_report


def _dataset():
    labels = np.array(
        [(d, c, g) for d in range(2) for c in range(2) for g in range(6) for _ in range(4)]
    )
    donor, condition, guide = labels.T
    mapping = np.array([0, 0, 1, 1, 2, 2])  # Control target is not zero.
    frequencies = np.array([[80, 20], [20, 80], [80, 20], [20, 80], [50, 50], [50, 50]])
    return ProgramDataset(
        counts=frequencies[guide],
        state=np.arange(len(labels), dtype=float)[:, None],
        donor_index=donor,
        sample_index=donor * 2 + condition,
        checkpoint_index=condition,
        target_index=mapping[guide],
        guide_index=guide,
        guide_to_target=mapping,
        control_guide_indices=(4, 5),
        checkpoint_times=(8.0, 48.0),
        target_descriptors=np.eye(3),
    )


def _config(batch=7):
    return q.ProgramQualificationConfig(
        programs=1,
        fit=ProgramFitConfig(max_epochs=1, patience=1, minibatch_size=batch),
        sparse_factor_rank=1,
    )


def _split(dataset, kind, config):
    return next(
        row
        for row in q._split_masks(
            dataset,
            evaluation_control_fraction=config.evaluation_control_fraction,
            seed=config.seed,
        )
        if row[0] == kind
    )


def _evaluate(dataset, split, config):
    kind, training, targets, references, reason = split
    return q._evaluate_split(
        dataset,
        kind=kind,
        training_indices=training,
        evaluation_indices=targets,
        evaluation_reference_indices=references,
        ineligibility_reason=reason,
        config=config,
        device="cpu",
    )


def _oracle_fit(monkeypatch, *, swap=False):
    fitted_rows, prediction_rows = [], []

    def fit(model, *, training, validation, config, device):
        fitted_rows.extend(training.state[:, 0].tolist() + validation.state[:, 0].tolist())

        def mean(batch):
            prediction_rows.extend(batch.state[:, 0].tolist())
            values = torch.tensor(
                [[0.8, 0.2], [0.2, 0.8], [0.8, 0.2], [0.2, 0.8], [0.5, 0.5], [0.5, 0.5]],
                device=batch.counts.device,
            )
            if swap:
                values = values.flip(1)
            return values[batch.guide_index] * batch.library_size[:, None]

        monkeypatch.setattr(model, "mean", mean)
        return ProgramFitResult(model, (), (), 0, False)

    monkeypatch.setattr(q, "fit_program_head", fit)
    return fitted_rows, prediction_rows


@pytest.mark.parametrize(
    "kind", [ProgramSplitKind.HELDOUT_GUIDE_TARGET_SHARED, ProgramSplitKind.HELDOUT_TARGET]
)
def test_split_to_evaluator_reserved_controls_make_sign_evaluable_and_swaps_fail(monkeypatch, kind):
    dataset, config = _dataset(), _config()
    split = _split(dataset, kind, config)
    fitted, predicted = _oracle_fit(monkeypatch)
    metric, _ = _evaluate(dataset, split, config)
    assert metric.gene_sign_accuracy == 1.0
    assert metric.gene_sign_coverage["supported_units"] == metric.gene_sign_coverage["total_units"]
    assert set(fitted) == set(split[1])
    assert set(predicted) == set(split[2])
    assert not set(split[3]) & (set(fitted) | set(predicted))
    assert metric.evaluation_reference["evaluation_reference_rows"] == len(split[3]) > 0
    _oracle_fit(monkeypatch, swap=True)
    wrong, _ = _evaluate(dataset, split, config)
    assert wrong.gene_sign_accuracy == 0.0


def test_reservation_is_outcome_blind_reproducible_and_disjoint():
    dataset = _dataset()
    changed = replace(dataset, counts=dataset.counts[:, ::-1].copy())
    for left, right in zip(q._split_masks(dataset), q._split_masks(changed), strict=True):
        for a, b in zip(left[1:4], right[1:4], strict=True):
            assert np.array_equal(a, b)
        training, targets, references = left[1:4]
        assert len(set(training) | set(targets) | set(references)) == len(dataset.counts)
        assert not set(training) & set(targets)
        assert not set(training) & set(references)
        assert not set(targets) & set(references)
        assert np.isin(dataset.guide_index[references], (4, 5)).all()
    with pytest.raises(ContractError, match="fraction"):
        q._split_masks(dataset, evaluation_control_fraction=1.0)


def test_reserved_outcome_mutation_cannot_change_actual_fit_or_predictions():
    dataset, config = _dataset(), _config()
    split = _split(dataset, ProgramSplitKind.HELDOUT_TARGET, config)
    first, fit_a = _evaluate(dataset, split, config)
    counts = dataset.counts.copy()
    counts[split[3]] = [90, 10]
    second, fit_b = _evaluate(replace(dataset, counts=counts), split, config)
    assert all(
        torch.equal(a, fit_b.model.state_dict()[key]) for key, a in fit_a.model.state_dict().items()
    )
    batch = q._batch(dataset, split[2])
    with torch.no_grad():
        assert torch.equal(fit_a.model.mean(batch), fit_b.model.mean(batch))
        assert torch.equal(fit_a.model.reference_mean(batch), fit_b.model.reference_mean(batch))
    assert first.predictive_nb_log_likelihood == second.predictive_nb_log_likelihood
    assert first.baselines == second.baselines
    assert first.evaluation_reference == second.evaluation_reference


def test_outer_transfer_sizes_and_chunked_metrics_match_dense_reference(monkeypatch):
    dataset = _dataset()
    _oracle_fit(monkeypatch)
    sizes = []
    original = ProgramBatch.to

    def transfer(self, device):
        sizes.append(len(self.counts))
        return original(self, device)

    monkeypatch.setattr(ProgramBatch, "to", transfer)
    config = _config(batch=3)
    split = _split(dataset, ProgramSplitKind.HELDOUT_TARGET, config)
    metric, fit = _evaluate(dataset, split, config)
    assert max(sizes) <= 3
    assert sum(sizes) == len(split[2])
    fitting, _ = q._inner_split(
        split[1],
        fraction=config.inner_validation_fraction,
        seed=config.seed + list(ProgramSplitKind).index(split[0]),
    )
    batch = q._batch(dataset, split[2])
    with torch.no_grad():
        mean = fit.model.mean(batch).numpy()
        reference = fit.model.reference_mean(batch).numpy()
    expected_ll = q._likelihood_metrics(
        dataset.counts[split[2]], mean, q._dispersion(dataset.counts[fitting])
    )
    assert metric.model_mean_log_likelihood_per_count == pytest.approx(expected_ll[0], abs=1e-12)
    assert metric.model_mean_negative_log_likelihood_per_cell == pytest.approx(
        expected_ll[1], abs=1e-10
    )
    observed = np.concatenate((dataset.counts[split[2]], dataset.counts[split[3]]))
    combined = np.concatenate((split[2], split[3]))
    report = gene_sign_report(
        counts=observed,
        predicted_mean=np.concatenate((mean, dataset.counts[split[3]])),
        predicted_reference_mean=np.concatenate((reference, dataset.counts[split[3]])),
        donor=dataset.donor_index[combined],
        condition=dataset.checkpoint_index[combined],
        guide=dataset.guide_index[combined],
        guide_to_target=dataset.guide_to_target,
        control_guides=(4, 5),
    )
    assert metric.gene_sign_coverage["units"] == report.units
    tables = q.fit_frozen_baselines(
        training_counts=dataset.counts[fitting],
        training_library_size=dataset.counts[fitting].sum(1),
        training_checkpoint=dataset.checkpoint_index[fitting],
        training_target=dataset.target_index[fitting],
        training_guide=dataset.guide_index[fitting],
        control_guide_indices=(4, 5),
        evaluation_library_size=dataset.counts[split[2]].sum(1),
        evaluation_checkpoint=dataset.checkpoint_index[split[2]],
        evaluation_target=dataset.target_index[split[2]],
        sparse_factor_rank=1,
    )
    for actual, table in zip(metric.baselines, tables, strict=True):
        expected = q._likelihood_metrics(
            dataset.counts[split[2]], table.mean, q._dispersion(dataset.counts[fitting])
        )
        assert actual.mean_log_likelihood_per_count == pytest.approx(expected[0], abs=1e-12)
    sizes.clear()
    other, _ = _evaluate(dataset, split, _config(batch=11))
    assert max(sizes) <= 11
    assert other.gene_sign_coverage == metric.gene_sign_coverage
    assert other.predictive_nb_log_likelihood == pytest.approx(
        metric.predictive_nb_log_likelihood, abs=1e-12
    )


@pytest.mark.parametrize(
    "change, message",
    [
        ("overlap", "disjoint"),
        ("noncontrol", "roles"),
        ("duplicate", "unique"),
        ("outofbounds", "valid row"),
    ],
)
def test_invalid_reference_partition_fails_before_fit(monkeypatch, change, message):
    dataset, config = _dataset(), _config()
    split = list(_split(dataset, ProgramSplitKind.HELDOUT_TARGET, config))
    if change == "overlap":
        split[3] = split[1][:1]
    elif change == "noncontrol":
        split[2], split[3] = split[3], split[2]
    elif change == "duplicate":
        split[3] = np.repeat(split[3][:1], 2)
    else:
        split[3] = np.array([len(dataset.counts)])
    monkeypatch.setattr(q, "fit_program_head", lambda *a, **k: pytest.fail("fitting was reached"))
    with pytest.raises(ContractError, match=message):
        _evaluate(dataset, split, config)


@pytest.mark.parametrize("limit", [0, 1])
def test_outer_budget_rejects_before_fitting_or_transfer(monkeypatch, limit):
    dataset, config = _dataset(), replace(_config(), maximum_evaluation_output_bytes=limit)
    split = _split(dataset, ProgramSplitKind.HELDOUT_TARGET, config)
    monkeypatch.setattr(q, "fit_program_head", lambda *a, **k: pytest.fail("fitting was reached"))
    with pytest.raises(ContractError, match="positive|evaluation-output"):
        _evaluate(dataset, split, config)


def test_no_reserved_controls_remains_undefined_not_borrowed(monkeypatch):
    dataset, config = _dataset(), _config()
    split = list(_split(dataset, ProgramSplitKind.HELDOUT_TARGET, config))
    split[1] = np.sort(np.concatenate((split[1], split[3])))
    split[3] = np.empty(0, dtype=int)
    _oracle_fit(monkeypatch)
    metric, _ = _evaluate(dataset, split, config)
    assert metric.gene_sign_accuracy is None
    assert not metric.passed
    assert all(
        row["status"] == "missing_donor_condition_controls"
        for row in metric.gene_sign_coverage["units"]
    )


def test_shared_prediction_and_summary_budgets_fail_closed():
    dataset = _dataset()
    model = q._new_model(dataset, 1, seed=0)
    batch = q._batch(dataset, np.array([0, 1, 2]))
    for rows, budget in ((0, 100), (2, 100), (3, 1)):
        with pytest.raises(ContractError, match="positive|evaluation-output"):
            list(
                iter_program_predictions(
                    model,
                    [batch],
                    device="cpu",
                    maximum_batch_rows=rows,
                    maximum_output_bytes=budget,
                )
            )
    accumulator = GeneSignAccumulator(
        genes=2,
        guide_to_target=dataset.guide_to_target,
        control_guides=(4, 5),
        maximum_output_bytes=1,
    )
    with pytest.raises(ContractError, match="evaluation-output"):
        accumulator.add_targets(
            counts=np.array([[80, 20]]),
            predicted_mean=np.array([[80, 20]]),
            predicted_reference_mean=np.array([[50, 50]]),
            donor=np.array([0]),
            condition=np.array([0]),
            guide=np.array([0]),
        )
