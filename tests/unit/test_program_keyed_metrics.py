from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from credo_count_sde_v4.errors import ContractError
from credo_count_sde_v4.programs import (
    CountLinkedProgramHead,
    ProgramBatch,
    ProgramFitConfig,
    fit_program_head,
)
from credo_count_sde_v4.programs.keyed_metrics import gene_sign_report, sister_guide_reports
from credo_count_sde_v4.programs.qualification import _likelihood_metrics


def _keys():
    # The control target is deliberately NOT zero.
    return dict(
        donor=np.array([0, 0, 0]),
        condition=np.array([1, 1, 1]),
        guide=np.array([0, 1, 2]),
        guide_to_target=np.array([0, 1, 2]),
        control_guides=(2,),
    )


def test_swapped_perturbation_predictions_fail_despite_identical_pooled_mean():
    observed = np.array([[80, 20], [20, 80], [50, 50]])
    reference = np.full((3, 2), 50)
    correct = gene_sign_report(
        counts=observed, predicted_mean=observed, predicted_reference_mean=reference, **_keys()
    )
    swapped = observed[[1, 0, 2]]
    assert np.array_equal(swapped.sum(axis=0), observed.sum(axis=0))
    wrong = gene_sign_report(
        counts=observed, predicted_mean=swapped, predicted_reference_mean=reference, **_keys()
    )
    assert correct.accuracy == 1.0
    assert wrong.accuracy == 0.0
    assert correct.informative_gene_effects == correct.candidate_gene_effects == 4
    assert correct.supported_units == correct.total_units == 2
    assert {row["target"] for row in correct.units} == {0, 1}


def test_missing_matched_controls_and_no_informative_effects_are_undefined():
    observed = np.array([[80, 20], [20, 80], [50, 50]])
    keys = {**_keys(), "condition": np.array([1, 1, 0])}
    report = gene_sign_report(
        counts=observed,
        predicted_mean=observed,
        predicted_reference_mean=np.full((3, 2), 50),
        **keys,
    )
    assert report.accuracy is None
    assert report.supported_units == 0
    assert report.total_units == 2
    assert all(row["status"] == "missing_donor_condition_controls" for row in report.units)
    report = gene_sign_report(
        counts=np.full((3, 2), 50),
        predicted_mean=observed,
        predicted_reference_mean=np.full((3, 2), 50),
        **_keys(),
    )
    assert report.accuracy is None
    assert report.informative_gene_effects == 0
    assert all(row["status"] == "no_informative_effects" for row in report.units)


def test_prediction_reference_is_separate_and_row_order_is_irrelevant():
    observed = np.array([[80, 20], [20, 80], [50, 50]])
    report = gene_sign_report(
        counts=observed, predicted_mean=observed, predicted_reference_mean=observed, **_keys()
    )
    assert report.accuracy == 0.0  # Predicted zero contrast is not the true effect.
    order = np.array([2, 1, 0])
    keys = _keys()
    for key in ("donor", "condition", "guide"):
        keys[key] = keys[key][order]
    permuted = gene_sign_report(
        counts=observed[order],
        predicted_mean=observed[order],
        predicted_reference_mean=np.full((3, 2), 50),
        **keys,
    )
    assert permuted.accuracy == 1.0


def test_sister_guides_join_condition_keys_not_equal_vector_lengths():
    # g0: c0,c1 and g1: c1,c2. Comparing concatenations incorrectly pairs c0/c1.
    counts = np.array([[80, 20], [20, 80], [80, 20], [20, 80], [50, 50], [50, 50], [50, 50]])
    kwargs = dict(
        counts=counts,
        donor=np.zeros(7, dtype=int),
        condition=np.array([0, 1, 1, 2, 0, 1, 2]),
        guide=np.array([0, 0, 1, 1, 2, 2, 2]),
        guide_to_target=np.array([0, 0, 1]),
        control_guides=(2,),
    )
    reports = sister_guide_reports(**kwargs)
    assert reports[0]["target_index"] == 0
    pair = reports[0]["pairs"][0]
    assert pair["shared_donor_conditions"] == [(0, 1)]
    assert pair["compared_gene_keys"] == 2
    assert pair["correlation"] == pytest.approx(-1.0)
    assert pair["support"] == [
        dict(donor=0, condition=1, left_cells=1, right_cells=1, control_cells=1)
    ]
    kwargs["condition"] = np.array([0, 0, 2, 2, 0, 1, 2])
    pair = sister_guide_reports(**kwargs)[0]["pairs"][0]
    assert pair["correlation"] is None
    assert pair["compared_gene_keys"] == 0
    kwargs["condition"] = np.array([0, 0, 2, 2, 1, 1, 1])
    assert sister_guide_reports(**kwargs)[0]["pairs"][0]["correlation"] is None


@pytest.mark.parametrize(
    "bad", [np.array([1, 2]), np.zeros((3, 2)), np.full((3, 2), np.nan), -np.ones((3, 2))]
)
def test_invalid_effect_payload_fails_closed(bad):
    with pytest.raises(ContractError):
        gene_sign_report(
            counts=bad,
            predicted_mean=np.ones((3, 2)),
            predicted_reference_mean=np.ones((3, 2)),
            **_keys(),
        )


@pytest.mark.parametrize(
    "change",
    [
        dict(control_guides=()),
        dict(control_guides=(3,)),
        dict(guide=np.array([0, 1, 3])),
        dict(donor=np.array([0, 0])),
        dict(guide_to_target=np.array([-1, 0, 1])),
    ],
)
def test_invalid_catalog_or_keys_fail_closed(change):
    with pytest.raises(ContractError):
        sister_guide_reports(counts=np.ones((3, 2)), **{**_keys(), **change})


def _head_batch():
    model = CountLinkedProgramHead(
        genes=2,
        programs=1,
        state_dimension=0,
        samples=1,
        checkpoints=2,
        targets=2,
        guides=2,
        guide_to_target=torch.tensor([0, 1]),
        control_guide_indices=(1,),
        checkpoint_times=torch.tensor([8.0, 48.0]),
    )
    counts = torch.tensor([[8.0, 2.0], [5.0, 5.0]]).repeat(4, 1)
    batch = ProgramBatch(
        counts=counts,
        library_size=counts.sum(1),
        state=torch.empty((8, 0)),
        sample_index=torch.zeros(8, dtype=torch.long),
        checkpoint_index=torch.ones(8, dtype=torch.long),
        target_index=torch.tensor([0, 1]).repeat(4),
        guide_index=torch.tensor([0, 1]).repeat(4),
    )
    return model, batch


def test_latent_scale_nonidentifiability_and_control_mask_are_explicit():
    model, batch = _head_batch()
    with torch.no_grad():
        model.guide_deviation.fill_(0.2)
        original = model.mean(batch).clone()
        model.guide_efficiency_logit.fill_(float(torch.logit(torch.tensor(0.25))))
        model.guide_deviation.mul_(2.0)
        assert torch.allclose(model.mean(batch), original)
        assert torch.equal(model.reference_mean(batch)[1::2], original[1::2])
        assert torch.all(model.latent_guide_scale == 0.25)


def test_fitted_dispersion_changes_predictive_score_not_shared_comparison():
    model, batch = _head_batch()
    with torch.no_grad():
        mean = model.mean(batch).numpy()
        counts = batch.counts.numpy()
        common = np.ones(2)
        comparison_before = _likelihood_metrics(counts, mean, common)
        predictive_before = _likelihood_metrics(counts, mean, model.dispersion.numpy())
        model.dispersion_raw.add_(3.0)
        predictive_after = _likelihood_metrics(counts, mean, model.dispersion.numpy())
        assert predictive_after != predictive_before
        assert _likelihood_metrics(counts, mean, common) == comparison_before


def test_fit_only_transfers_minibatches_and_applies_diagnostic_caps(monkeypatch):
    model, batch = _head_batch()
    transfers = []
    original = ProgramBatch.to

    def record(self, device):
        transfers.append(len(self.counts))
        return original(self, device)

    monkeypatch.setattr(ProgramBatch, "to", record)
    config = ProgramFitConfig(max_epochs=1, patience=1, minibatch_size=3)
    fit_program_head(model, training=batch, validation=batch, config=config)
    assert transfers == [3, 3, 2, 3, 3, 2]
    for bad_config in (
        replace(config, maximum_panel_genes=1),
        replace(config, maximum_host_payload_bytes=1),
    ):
        with pytest.raises(ContractError, match="small-panel"):
            fit_program_head(model, training=batch, validation=batch, config=bad_config)
    with pytest.raises(ContractError, match="positive"):
        fit_program_head(
            model,
            training=batch,
            validation=batch,
            config=replace(config, maximum_host_payload_bytes=0),
        )
