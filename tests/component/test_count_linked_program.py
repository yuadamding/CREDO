from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from credo_count_sde_v4.contracts import BaselineName, ProgramSplitKind, ScientificScope
from credo_count_sde_v4.errors import ContractError, IntegrityError
from credo_count_sde_v4.programs import (
    CountLinkedProgramHead,
    ProgramBatch,
    ProgramDataset,
    ProgramFitConfig,
    ProgramQualificationConfig,
    fit_frozen_baselines,
    fit_program_head,
    qualify_and_publish_program_model,
    qualify_program_model,
    simulate_program_counts,
    verify_program_qualification,
)


def _dataset(*, target_descriptors: bool = True, null: bool = False) -> ProgramDataset:
    simulated = simulate_program_counts(
        cells=420,
        genes=30,
        programs=3,
        donors=3,
        checkpoints=3,
        perturbation_targets=3,
        guides_per_target=2,
        controls=2,
        state_dimension=2,
        null=null,
        seed=9,
    )
    return ProgramDataset(
        counts=simulated.counts,
        state=simulated.state,
        donor_index=simulated.donor_index,
        sample_index=simulated.sample_index,
        checkpoint_index=simulated.checkpoint_index,
        target_index=simulated.target_index,
        guide_index=simulated.guide_index,
        guide_to_target=simulated.guide_to_target,
        control_guide_indices=simulated.control_guide_indices,
        checkpoint_times=(0.0, 8.0, 24.0),
        target_descriptors=(simulated.target_descriptors if target_descriptors else None),
        protected_expression_access_contract_id="protected-expression-test",
    )


def _batch(dataset: ProgramDataset, indices: np.ndarray) -> ProgramBatch:
    counts = torch.as_tensor(dataset.counts[indices], dtype=torch.float32)
    return ProgramBatch(
        counts=counts,
        library_size=counts.sum(dim=1),
        state=torch.as_tensor(dataset.state[indices], dtype=torch.float32),
        sample_index=torch.as_tensor(dataset.sample_index[indices]),
        checkpoint_index=torch.as_tensor(dataset.checkpoint_index[indices]),
        target_index=torch.as_tensor(dataset.target_index[indices]),
        guide_index=torch.as_tensor(dataset.guide_index[indices]),
    )


def _model(dataset: ProgramDataset) -> CountLinkedProgramHead:
    return CountLinkedProgramHead(
        genes=dataset.counts.shape[1],
        programs=3,
        state_dimension=dataset.state.shape[1],
        samples=int(dataset.sample_index.max()) + 1,
        checkpoints=3,
        targets=int(dataset.target_index.max()) + 1,
        guides=len(dataset.guide_to_target),
        guide_to_target=torch.as_tensor(dataset.guide_to_target),
        control_guide_indices=dataset.control_guide_indices,
        checkpoint_times=torch.tensor(dataset.checkpoint_times),
        target_descriptors=(
            None
            if dataset.target_descriptors is None
            else torch.as_tensor(dataset.target_descriptors)
        ),
    )


def test_nb_head_uses_exact_library_offset_and_control_reference() -> None:
    dataset = _dataset()
    control_indices = np.where(np.isin(dataset.guide_index, dataset.control_guide_indices))[0][:20]
    batch = _batch(dataset, control_indices)
    model = _model(dataset)
    before = model.log_mean(batch).detach().clone()
    with torch.no_grad():
        assert model.target_descriptor_weight is not None
        model.target_descriptor_weight.fill_(100.0)
        model.guide_deviation.fill_(100.0)
    after = model.log_mean(batch).detach()
    assert torch.equal(before, after)
    assert torch.isfinite(model.negative_log_likelihood(batch))


def test_adversarial_noninteger_and_crosswired_hierarchy_fail() -> None:
    dataset = _dataset()
    batch = _batch(dataset, np.arange(20))
    model = _model(dataset)
    bad_counts = batch.counts.clone()
    bad_counts[0, 0] += 0.5
    with pytest.raises(ContractError, match="integer"):
        model.mean(
            ProgramBatch(
                counts=bad_counts,
                library_size=bad_counts.sum(dim=1),
                state=batch.state,
                sample_index=batch.sample_index,
                checkpoint_index=batch.checkpoint_index,
                target_index=batch.target_index,
                guide_index=batch.guide_index,
            )
        )
    wrong_target = batch.target_index.clone()
    wrong_target[0] = (wrong_target[0] + 1) % model.targets
    with pytest.raises(ContractError, match="hierarchy"):
        model.mean(
            ProgramBatch(
                counts=batch.counts,
                library_size=batch.library_size,
                state=batch.state,
                sample_index=batch.sample_index,
                checkpoint_index=batch.checkpoint_index,
                target_index=wrong_target,
                guide_index=batch.guide_index,
            )
        )


def test_fit_reduces_inner_validation_count_nll() -> None:
    dataset = _dataset()
    model = _model(dataset)
    training = _batch(dataset, np.arange(300))
    validation = _batch(dataset, np.arange(300, 420))
    with torch.no_grad():
        frequency = dataset.counts[:300].sum(axis=0) + 0.5
        frequency = frequency / frequency.sum()
        model.gene_intercept.copy_(torch.as_tensor(np.log(frequency), dtype=torch.float32))
        initial = float(model.negative_log_likelihood(validation))
    fitted = fit_program_head(
        model,
        training=training,
        validation=validation,
        config=ProgramFitConfig(max_epochs=20, patience=6, minibatch_size=128, seed=4),
    )
    assert min(fitted.validation_loss) < initial
    assert fitted.best_epoch >= 0


def test_all_six_baselines_are_training_only_and_frozen_order() -> None:
    dataset = _dataset()
    training = np.arange(300)
    evaluation = np.arange(300, 340)
    kwargs = {
        "training_counts": dataset.counts[training],
        "training_library_size": dataset.counts[training].sum(axis=1),
        "training_checkpoint": dataset.checkpoint_index[training],
        "training_target": dataset.target_index[training],
        "training_guide": dataset.guide_index[training],
        "control_guide_indices": dataset.control_guide_indices,
        "evaluation_library_size": dataset.counts[evaluation].sum(axis=1),
        "evaluation_checkpoint": dataset.checkpoint_index[evaluation],
        "evaluation_target": dataset.target_index[evaluation],
        "sparse_factor_rank": 3,
    }
    first = fit_frozen_baselines(**kwargs)
    altered_evaluation_counts = dataset.counts[evaluation].copy()
    altered_evaluation_counts[:, 0] += 10000
    second = fit_frozen_baselines(**kwargs)
    assert tuple(item.name for item in first) == tuple(BaselineName)
    assert all(
        np.array_equal(left.mean, right.mean) for left, right in zip(first, second, strict=True)
    )
    assert altered_evaluation_counts.sum() > dataset.counts[evaluation].sum()


def test_complete_qualification_reports_noninterchangeable_splits_and_nulls() -> None:
    dataset = _dataset()
    config = ProgramQualificationConfig(
        programs=3,
        fit=ProgramFitConfig(max_epochs=5, patience=2, minibatch_size=128, seed=2),
        stability_seeds=(1, 2, 3),
        null_replicates=3,
        null_fit_epochs=2,
        sparse_factor_rank=3,
        minimum_log_likelihood_improvement=-100.0,
        minimum_gene_sign_accuracy=0.0,
        minimum_seed_loading_stability=0.0,
        minimum_donor_loading_stability=0.0,
        minimum_sister_guide_correlation=-1.0,
        maximum_null_inclusion_rate=1.0,
        effect_inclusion_threshold=10.0,
    )
    result = qualify_program_model(dataset, config=config)
    assert tuple(item.kind for item in result.splits) == tuple(ProgramSplitKind)
    assert all(item.eligible for item in result.splits)
    assert all(
        tuple(metric.name for metric in item.baselines) == tuple(BaselineName)
        for item in result.splits
    )
    assert result.null_replicate_families == (
        "control_label_permutation",
        "target_within_checkpoint_permutation",
        "negative_binomial_no_program",
    )
    assert result.per_target_guide_metrics
    assert 0.0 <= result.target_variance_fraction < 1.0
    assert 0.0 <= result.inconsistent_target_fraction <= 1.0
    reference = result.seed_loadings[0]
    for aligned in result.seed_loadings[1:]:
        correlation = np.corrcoef(reference.T, aligned.T)[:3, 3:]
        assert np.all(np.diag(correlation) >= 0)
    assert not result.scientific_pass
    assert not result.null_inclusion_calibrated
    assert (
        result.null_calibration_semantics
        == "legacy_coefficient_exceedance_not_discovery_calibration"
    )
    for metric in result.splits:
        assert metric.predictive_nb_log_likelihood is not None
        assert (
            metric.common_dispersion_mean_prediction_score
            == metric.model_mean_log_likelihood_per_count
        )
        assert metric.gene_sign_coverage is not None


def test_identifier_only_target_and_missing_time_firewall_remain_ineligible() -> None:
    dataset = _dataset(target_descriptors=False)
    dataset = ProgramDataset(
        **{
            **vars(dataset),
            "protected_expression_access_contract_id": None,
        }
    )
    config = ProgramQualificationConfig(
        programs=3,
        fit=ProgramFitConfig(max_epochs=2, patience=1, minibatch_size=128, seed=2),
        stability_seeds=(1, 2, 3),
        null_replicates=1,
        null_fit_epochs=1,
        sparse_factor_rank=3,
        minimum_log_likelihood_improvement=-100.0,
        minimum_gene_sign_accuracy=0.0,
        minimum_seed_loading_stability=0.0,
        minimum_donor_loading_stability=0.0,
        minimum_sister_guide_correlation=-1.0,
        maximum_null_inclusion_rate=1.0,
        effect_inclusion_threshold=10.0,
    )
    result = qualify_program_model(dataset, config=config)
    target = next(item for item in result.splits if item.kind == ProgramSplitKind.HELDOUT_TARGET)
    time = next(item for item in result.splits if item.kind == ProgramSplitKind.HELDOUT_TIME)
    assert not target.eligible and "identifier-only" in (target.reason or "")
    assert not time.eligible and "protected-expression" in (time.reason or "")
    assert not result.scientific_pass


def test_donor_identity_and_three_checkpoint_requirements_fail_closed() -> None:
    dataset = _dataset()
    donor_unavailable = ProgramDataset(
        **{
            **vars(dataset),
            "checkpoint_index": np.minimum(dataset.checkpoint_index, 1),
            "checkpoint_times": (0.0, 8.0),
            "heldout_donor_eligible": False,
            "heldout_donor_ineligibility_reason": (
                "batch labels are technical libraries, not donor identities"
            ),
        }
    )
    config = ProgramQualificationConfig(
        programs=3,
        fit=ProgramFitConfig(max_epochs=2, patience=1, minibatch_size=128, seed=2),
        stability_seeds=(1, 2, 3),
        null_replicates=1,
        null_fit_epochs=1,
        sparse_factor_rank=3,
        minimum_log_likelihood_improvement=-100.0,
        minimum_gene_sign_accuracy=0.0,
        minimum_seed_loading_stability=0.0,
        minimum_donor_loading_stability=0.0,
        minimum_sister_guide_correlation=-1.0,
        maximum_null_inclusion_rate=1.0,
        effect_inclusion_threshold=10.0,
    )
    result = qualify_program_model(donor_unavailable, config=config)
    donor = next(item for item in result.splits if item.kind == ProgramSplitKind.HELDOUT_DONOR)
    time = next(item for item in result.splits if item.kind == ProgramSplitKind.HELDOUT_TIME)
    assert not donor.eligible and "technical libraries" in (donor.reason or "")
    assert not time.eligible and "at least two fit checkpoints" in (time.reason or "")
    assert not result.scientific_pass


def test_null_simulation_removes_true_program_and_target_effects() -> None:
    simulated = simulate_program_counts(
        cells=200,
        genes=20,
        programs=2,
        donors=3,
        checkpoints=2,
        perturbation_targets=3,
        guides_per_target=2,
        controls=2,
        state_dimension=1,
        null=True,
        seed=10,
    )
    assert np.count_nonzero(simulated.true_loadings) == 0
    assert np.count_nonzero(simulated.true_target_activity) == 0


def test_program_bundle_publication_is_content_verified(tmp_path) -> None:
    dataset = _dataset()
    config = ProgramQualificationConfig(
        programs=3,
        fit=ProgramFitConfig(max_epochs=2, patience=1, minibatch_size=128, seed=2),
        stability_seeds=(1, 2, 3),
        null_replicates=20,
        null_fit_epochs=1,
        sparse_factor_rank=3,
        minimum_log_likelihood_improvement=-100.0,
        minimum_gene_sign_accuracy=0.0,
        minimum_seed_loading_stability=0.0,
        minimum_donor_loading_stability=0.0,
        minimum_sister_guide_correlation=-1.0,
        maximum_null_inclusion_rate=0.999,
        effect_inclusion_threshold=10.0,
        loading_inclusion_threshold=0.01,
    )
    destination = tmp_path / "program-bundle"
    scope = ScientificScope(
        subject="target_3",
        entity_scope=("target_3",),
        sample_scope=("all_donors",),
        time_scope=("0h", "24h"),
        population="simulated cells",
        comparison="shared control guides",
    )
    qualify_and_publish_program_model(
        destination,
        dataset=dataset,
        config=config,
        study_id="simulation",
        capability_assessment_id="capability-test",
        scope=scope,
        perturbation_id="target_3_ko",
        target_id="target_3",
        target_index=3,
        guide_ids=("guide_6", "guide_7"),
        guide_indices=(6, 7),
        feature_order_sha256="a" * 64,
        guide_target_map_sha256="b" * 64,
        fit_row_ids_sha256="c" * 64,
    )
    bundle = verify_program_qualification(destination)
    assert bundle.qualification.status == "fail_qualification"
    assert not bundle.qualification.null_inclusion_calibrated
    detailed = json.loads((destination / "artifacts/qualification_metrics.json").read_text())
    assert detailed["metric_revision"] == 2
    assert not detailed["biological_efficiency_identified"]
    assert detailed["execution_limits"]["maximum_panel_genes"] == 2048
    assert any(item["gene_sign_accuracy"] is None for item in detailed["splits"])
    assert len(bundle.program_definitions) == 3
    assert bundle.qualification.qualification_protocol_id == (
        bundle.qualification_protocol.qualification_protocol_id
    )
    assert bundle.qualification_protocol.max_epochs == 2
    consistency_path = (
        destination / bundle.guide_target_consistency.per_target_artifact.relative_uri
    )
    consistency = json.loads(consistency_path.read_text())
    assert consistency["per_target"]
    assert consistency["target_variance_fraction"] == pytest.approx(
        bundle.guide_target_consistency.target_variance_fraction
    )
    standard_error = np.load(
        destination / bundle.gene_effects[0].standard_error_artifact.relative_uri,
        allow_pickle=False,
    )
    assert np.isfinite(standard_error).all()
    assert np.all(standard_error >= 0)
    state_path = destination / bundle.model_state_artifact.relative_uri
    state_path.write_bytes(state_path.read_bytes() + b"corruption")
    with pytest.raises(IntegrityError, match="size mismatch"):
        verify_program_qualification(destination)
