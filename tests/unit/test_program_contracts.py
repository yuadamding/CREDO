from __future__ import annotations

from typing import Any, TypeVar

import pytest
from pydantic import TypeAdapter, ValidationError

from credo_count_sde_v4.contracts import (
    ArtifactRef,
    BaselineMetric,
    BaselineName,
    CountLinkedProgramContract,
    HeldoutTargetMode,
    LibrarySizeSemantics,
    NegativeBinomialSemantics,
    ProgramNullContract,
    ProgramQualificationProtocol,
    ProgramSplitEvaluation,
    ProgramSplitKind,
    ScientificScope,
    SparseLoadingPrior,
    StrictModel,
)

ModelT = TypeVar("ModelT", bound=StrictModel)


def _identified(model: type[ModelT], payload: dict[str, Any], id_field: str) -> ModelT:
    payload[id_field] = "pending"
    normalized = {
        name: TypeAdapter(field.annotation).validate_python(payload[name])
        for name, field in model.model_fields.items()
        if name in payload
    }
    provisional = model.model_construct(**normalized)
    payload[id_field] = provisional.identity(id_field=id_field)
    return model.model_validate(payload)


def _artifact() -> ArtifactRef:
    return ArtifactRef(
        schema_id="test.target_descriptors",
        schema_version=1,
        sha256="a" * 64,
        size_bytes=10,
        media_type="application/x-npy",
        relative_uri="artifacts/target_descriptors.npy",
    )


def _contract() -> CountLinkedProgramContract:
    return _identified(
        CountLinkedProgramContract,
        {
            "schema_id": "credo.count_linked_program_contract",
            "schema_version": 1,
            "count_likelihood": NegativeBinomialSemantics.NB2_GENE_DISPERSION,
            "library_size_semantics": LibrarySizeSemantics.OBSERVED_TOTAL_COUNT_OFFSET,
            "sparse_loading_prior": SparseLoadingPrior.L1_WITH_UNIT_NORM_COLUMNS,
            "state_dependence_enabled": True,
            "checkpoint_dependence_enabled": True,
            "donor_or_sample_effects_enabled": True,
            "heldout_target_mode": HeldoutTargetMode.PREDECLARED_TARGET_DESCRIPTORS,
            "target_descriptor_artifact": _artifact(),
            "feature_order_sha256": "b" * 64,
            "guide_target_map_sha256": "c" * 64,
            "fit_row_ids_sha256": "d" * 64,
            "genes": 100,
            "programs": 5,
            "state_dimension": 4,
            "samples": 4,
            "checkpoints": 3,
            "checkpoint_time_values": (0.0, 8.0, 24.0),
            "targets": 5,
            "guides": 12,
            "control_guide_indices": (0, 1),
            "loading_l1_weight": 0.01,
            "guide_deviation_l2_weight": 0.01,
            "target_effect_l2_weight": 0.001,
        },
        "program_contract_id",
    )


def test_dev40a_contract_freezes_nb_offset_hierarchy_and_no_dynamics() -> None:
    contract = _contract()
    assert contract.raw_nonnegative_integer_counts_required
    assert contract.gene_specific_dispersion
    assert contract.control_reference_centered
    assert contract.target_guide_hierarchy_enabled
    assert not contract.dynamics_coupled
    assert not contract.checkpoint_schema_modified
    assert contract.checkpoint_time_semantics == "linear_continuous_physical_time"


def test_identifier_only_and_descriptor_modes_cannot_be_crosswired() -> None:
    payload = _contract().model_dump(mode="python")
    payload.pop("program_contract_id")
    payload["heldout_target_mode"] = HeldoutTargetMode.IDENTIFIER_ONLY_NOT_ELIGIBLE
    with pytest.raises(ValidationError, match="descriptor artifact"):
        _identified(CountLinkedProgramContract, payload, "program_contract_id")


def test_eligible_split_requires_every_frozen_baseline() -> None:
    baselines = tuple(
        BaselineMetric(
            baseline=name,
            mean_log_likelihood_per_count=-0.1,
            mean_negative_log_likelihood_per_cell=10.0,
        )
        for name in BaselineName
    )
    split = ProgramSplitEvaluation(
        split_id="heldout-target",
        kind=ProgramSplitKind.HELDOUT_TARGET,
        eligible=True,
        passed=True,
        fit_units=("target_1", "target_2"),
        evaluation_units=("target_3",),
        model_mean_log_likelihood_per_count=-0.09,
        model_mean_negative_log_likelihood_per_cell=9.0,
        baselines=baselines,
        improvement_over_best_baseline=0.01,
        gene_sign_accuracy=0.7,
    )
    assert tuple(item.baseline for item in split.baselines) == tuple(BaselineName)
    with pytest.raises(ValidationError, match="every baseline"):
        ProgramSplitEvaluation(**{**split.model_dump(mode="python"), "baselines": baselines[:-1]})


def test_heldout_time_requires_protected_access_contract() -> None:
    baselines = tuple(
        BaselineMetric(
            baseline=name,
            mean_log_likelihood_per_count=-0.1,
            mean_negative_log_likelihood_per_cell=10.0,
        )
        for name in BaselineName
    )
    with pytest.raises(ValidationError, match="protected-expression"):
        ProgramSplitEvaluation(
            split_id="heldout-time",
            kind=ProgramSplitKind.HELDOUT_TIME,
            eligible=True,
            passed=True,
            fit_units=("0h", "4h"),
            evaluation_units=("8h",),
            model_mean_log_likelihood_per_count=-0.09,
            model_mean_negative_log_likelihood_per_cell=9.0,
            baselines=baselines,
            improvement_over_best_baseline=0.01,
            gene_sign_accuracy=0.7,
        )


def test_program_null_contract_requires_all_frozen_families() -> None:
    valid = _identified(
        ProgramNullContract,
        {
            "schema_id": "credo.program_null_contract",
            "schema_version": 1,
            "null_families": (
                "control_label_permutation",
                "target_within_checkpoint_permutation",
                "negative_binomial_no_program",
            ),
            "replicate_count": 20,
            "maximum_program_inclusion_rate": 0.1,
        },
        "null_contract_id",
    )
    assert valid.selection_threshold_frozen_before_nulls
    payload = valid.model_dump(mode="python")
    payload.pop("null_contract_id")
    payload["null_families"] = payload["null_families"][:-1]
    with pytest.raises(ValidationError):
        _identified(ProgramNullContract, payload, "null_contract_id")


def test_qualification_protocol_binds_thresholds_and_training_budget() -> None:
    protocol = _identified(
        ProgramQualificationProtocol,
        {
            "split_kinds": tuple(ProgramSplitKind),
            "baselines": tuple(BaselineName),
            "stability_seeds": (11, 29, 47),
            "sparse_factor_rank": 6,
            "null_replicates": 20,
            "null_fit_epochs": 100,
            "minimum_log_likelihood_improvement": 0.0,
            "minimum_gene_sign_accuracy": 0.55,
            "minimum_seed_loading_stability": 0.6,
            "minimum_donor_loading_stability": 0.6,
            "minimum_sister_guide_correlation": 0.3,
            "maximum_null_inclusion_rate": 0.1,
            "effect_inclusion_threshold": 0.15,
            "loading_inclusion_threshold": 0.05,
            "inner_validation_fraction": 0.15,
            "seed": 20260821,
            "learning_rate": 0.003,
            "weight_decay": 1e-5,
            "max_epochs": 400,
            "patience": 50,
            "minimum_delta": 1e-4,
            "gradient_clip_norm": 5.0,
            "minibatch_size": 1024,
            "loading_l1_weight": 0.01,
            "guide_deviation_l2_weight": 0.01,
            "target_effect_l2_weight": 0.001,
        },
        "qualification_protocol_id",
    )
    assert protocol.max_epochs == 400
    assert protocol.evaluation_dispersion_semantics.startswith("training_only")
    payload = protocol.model_dump(mode="python")
    payload.pop("qualification_protocol_id")
    payload["baselines"] = payload["baselines"][:-1]
    with pytest.raises(ValidationError, match="baselines"):
        _identified(ProgramQualificationProtocol, payload, "qualification_protocol_id")


def test_scope_contract_rejects_unsorted_units() -> None:
    with pytest.raises(ValidationError, match="sample_scope"):
        ScientificScope(
            subject="target",
            entity_scope=("target",),
            sample_scope=("b", "a"),
            time_scope=("0h", "8h"),
            population="cells",
            comparison="control",
        )
