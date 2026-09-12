from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

import pytest
from pydantic import TypeAdapter, ValidationError

from credo_count_sde_v4 import validate_contract
from credo_count_sde_v4.canonical import contract_id
from credo_count_sde_v4.claims import validate_g14_contracts
from credo_count_sde_v4.contracts import (
    ClaimAdjudication,
    ClaimDecision,
    ClaimRecord,
    ClaimRegistry,
    EvidenceChannel,
    EvidenceTier,
    G14MultiplicityContract,
    G14MultiplicityFamily,
    G14RobustnessPlan,
    G14SealContract,
    RobustnessAxis,
    StrictModel,
)
from credo_count_sde_v4.errors import IntegrityError

OUTPUTS = (
    "ROBUSTNESS_MATRIX.parquet",
    "SIMULTANEOUS_INTERVALS.parquet",
    "MULTIPLICITY_DECISION.json",
    "CLAIM_LEDGER.parquet",
    "FINAL_EVIDENCE_GRAPH.json",
    "FINAL_CLAIM_SEAL_RECEIPT.json",
    "SHA256SUMS",
)
ModelT = TypeVar("ModelT", bound=StrictModel)


def _identified(model: type[ModelT], payload: dict[str, Any], id_field: str) -> ModelT:
    payload[id_field] = "pending"
    normalized_fields = {
        name: TypeAdapter(field.annotation).validate_python(payload[name])
        for name, field in model.model_fields.items()
        if name in payload
    }
    normalized = model.model_construct(**normalized_fields).model_dump(mode="json")
    payload[id_field] = contract_id(normalized, id_field=id_field)
    return model.model_validate(payload)


def _contracts() -> tuple[
    ClaimRegistry,
    G14RobustnessPlan,
    G14MultiplicityContract,
    G14SealContract,
]:
    claim = ClaimRecord(
        claim_id="STATE_E8",
        component_id="G07",
        endpoint="heldout_donor_8h_state",
        candidate="qualified_state_model",
        comparator="strongest_training_only_baseline",
        direction="lower",
        primary_or_secondary="primary",
        claim_family="P",
        resampling_unit="target_within_donor",
        multiplicity_method="paired_max_statistic",
        required_robustness_axis_ids=("algorithmic_seed",),
        eligible_wording="donor-inductive prediction in this four-donor cohort",
        forbidden_wording=("general donor-population prediction", "causal state effect"),
    )
    registry = _identified(
        ClaimRegistry,
        {
            "schema_version": 2,
            "frozen_before_first_relevant_outer_evaluation": True,
            "records": [claim.model_dump(mode="json")],
        },
        "registry_id",
    )
    robustness = _identified(
        G14RobustnessPlan,
        {
            "schema_version": 2,
            "axes": [
                RobustnessAxis(
                    axis_id="algorithmic_seed",
                    values=("base", "sensitivity"),
                    claim_ids=("STATE_E8",),
                ).model_dump(mode="json")
            ],
            "base_contract_hashes": ["a" * 64],
            "model_fitting": False,
            "model_selection": False,
            "threshold_adjustment": False,
            "target_discovery": False,
            "sensitivities_select_reported_model": False,
        },
        "plan_id",
    )
    multiplicity = _identified(
        G14MultiplicityContract,
        {
            "schema_version": 2,
            "claim_registry_id": registry.registry_id,
            "families": [
                G14MultiplicityFamily(
                    family_id="P",
                    claim_ids=("STATE_E8",),
                    method="paired_max_statistic",
                    conditional_on_observed_donors=True,
                ).model_dump(mode="json")
            ],
            "general_population_donor_claim_allowed": False,
        },
        "multiplicity_contract_id",
    )
    seal = _identified(
        G14SealContract,
        {
            "schema_version": 2,
            "claim_registry_id": registry.registry_id,
            "robustness_plan_id": robustness.plan_id,
            "multiplicity_contract_id": multiplicity.multiplicity_contract_id,
            "purpose": "robustness_multiplicity_and_claim_sealing",
            "model_fitting": False,
            "model_selection": False,
            "threshold_adjustment": False,
            "target_discovery": False,
            "claim_decisions": {"STATE_E8": "not_promoted"},
            "component_receipt_hashes": {"G07": "b" * 64},
            "required_outputs": OUTPUTS,
        },
        "g14_contract_id",
    )
    return registry, robustness, multiplicity, seal


def test_g14_contract_graph_is_complete_and_evidence_only() -> None:
    registry, robustness, multiplicity, seal = _contracts()
    validate_g14_contracts(registry, robustness, multiplicity, seal)
    assert not robustness.model_fitting
    assert not robustness.model_selection
    assert not robustness.threshold_adjustment
    assert not robustness.target_discovery
    assert not seal.model_fitting
    assert set(seal.required_outputs) == set(OUTPUTS)


def test_g14_cannot_promote_a_blocked_evidence_adjudication() -> None:
    registry, robustness, multiplicity, seal = _contracts()
    seal_payload = seal.model_dump(mode="python")
    seal_payload["claim_decisions"] = {"STATE_E8": "promoted"}
    seal_payload.pop("g14_contract_id")
    promoted = _identified(G14SealContract, seal_payload, "g14_contract_id")
    blocked = _identified(
        ClaimAdjudication,
        {
            "schema_id": "credo.claim_adjudication",
            "schema_version": 1,
            "claim_id": "STATE_E8",
            "claim_request_id": "dev39-request",
            "capability_assessment_id": "dev39-capability",
            "decision": ClaimDecision.BLOCKED,
            "evidence_tiers": (EvidenceTier.E1_IN_SAMPLE,),
            "evidence_channels": (EvidenceChannel.MODEL_FIT,),
            "permitted_wording": None,
            "blocking_reasons": ("scientific capability failed",),
        },
        "adjudication_id",
    )
    with pytest.raises(IntegrityError, match="blocked evidence adjudication"):
        validate_g14_contracts(
            registry,
            robustness,
            multiplicity,
            promoted,
            {"STATE_E8": blocked},
        )


def test_g14_registry_rejects_unknown_parent_claim() -> None:
    claim = ClaimRecord(
        claim_id="MECHANISM",
        component_id="G09",
        parent_claim_ids=("MISSING",),
        endpoint="joint_model",
        candidate="joint",
        comparator="separate",
        direction="lower",
        primary_or_secondary="primary",
        claim_family="M",
        resampling_unit="target_within_donor",
        multiplicity_method="joint_max_statistic",
        required_robustness_axis_ids=("algorithmic_seed",),
        eligible_wording="internal model mechanism",
        forbidden_wording=("biological causality",),
    )
    payload = {
        "schema_version": 2,
        "registry_id": "pending",
        "frozen_before_first_relevant_outer_evaluation": True,
        "records": [claim.model_dump(mode="json")],
    }
    payload["registry_id"] = contract_id(payload, id_field="registry_id")
    with pytest.raises(ValidationError, match="unknown parent"):
        ClaimRegistry.model_validate(payload)


def test_g14_cross_validation_rejects_uncovered_claim() -> None:
    registry, robustness, multiplicity, seal = _contracts()
    altered_payload = multiplicity.model_dump(mode="json")
    altered_payload["families"][0]["claim_ids"] = ["UNKNOWN"]
    altered_payload["multiplicity_contract_id"] = "pending"
    altered_payload["multiplicity_contract_id"] = contract_id(
        altered_payload, id_field="multiplicity_contract_id"
    )
    altered = G14MultiplicityContract.model_validate(altered_payload)
    altered_seal_payload = seal.model_dump(mode="json")
    altered_seal_payload["multiplicity_contract_id"] = altered.multiplicity_contract_id
    altered_seal_payload["g14_contract_id"] = "pending"
    altered_seal_payload["g14_contract_id"] = contract_id(
        altered_seal_payload, id_field="g14_contract_id"
    )
    altered_seal = G14SealContract.model_validate(altered_seal_payload)
    with pytest.raises(IntegrityError, match="unknown claim"):
        validate_g14_contracts(registry, robustness, altered, altered_seal)


def test_g14_cross_validation_rejects_each_identity_and_coverage_mismatch() -> None:
    registry, robustness, multiplicity, seal = _contracts()
    with pytest.raises(IntegrityError, match="different claim registry"):
        validate_g14_contracts(
            registry,
            robustness,
            multiplicity.model_copy(update={"claim_registry_id": "wrong"}),
            seal,
        )
    with pytest.raises(IntegrityError, match="incompatible contract graph"):
        validate_g14_contracts(
            registry,
            robustness,
            multiplicity,
            seal.model_copy(update={"robustness_plan_id": "wrong"}),
        )

    record = registry.records[0].model_copy(update={"claim_family": "WRONG"})
    wrong_family_registry = registry.model_copy(update={"records": (record,)})
    with pytest.raises(IntegrityError, match="wrong family"):
        validate_g14_contracts(wrong_family_registry, robustness, multiplicity, seal)

    repeated_family = multiplicity.families[0].model_copy(
        update={"claim_ids": ("STATE_E8", "STATE_E8")}
    )
    repeated = multiplicity.model_copy(update={"families": (repeated_family,)})
    repeated_seal = seal.model_copy(
        update={"multiplicity_contract_id": repeated.multiplicity_contract_id}
    )
    with pytest.raises(IntegrityError, match="exactly once"):
        validate_g14_contracts(registry, robustness, repeated, repeated_seal)

    no_axes = robustness.model_copy(update={"axes": ()})
    no_axes_seal = seal.model_copy(update={"robustness_plan_id": no_axes.plan_id})
    with pytest.raises(IntegrityError, match="lacks required robustness"):
        validate_g14_contracts(registry, no_axes, multiplicity, no_axes_seal)

    with pytest.raises(IntegrityError, match="exactly one decision"):
        validate_g14_contracts(
            registry,
            robustness,
            multiplicity,
            seal.model_copy(update={"claim_decisions": {}}),
        )


def test_g14_contracts_dispatch_through_public_validator(tmp_path: Path) -> None:
    registry, robustness, multiplicity, seal = _contracts()
    expected = (
        ("claim-registry.json", registry, "ClaimRegistry"),
        ("robustness.json", robustness, "G14RobustnessPlan"),
        ("multiplicity.json", multiplicity, "G14MultiplicityContract"),
        ("seal.json", seal, "G14SealContract"),
    )
    for name, contract, contract_type in expected:
        path = tmp_path / name
        path.write_text(contract.model_dump_json() + "\n")
        assert validate_contract(path)["contract_type"] == contract_type


def test_g14_dev28_v1_contract_graph_remains_readable(tmp_path: Path) -> None:
    record = {
        "claim_id": "STATE_E8",
        "component_id": "G07",
        "parent_claim_ids": [],
        "endpoint": "heldout_donor_8h_state",
        "candidate": "qualified_state_model",
        "comparator": "baseline",
        "direction": "lower",
        "primary_or_secondary": "primary",
        "claim_family": "P",
        "resampling_unit": "target_within_donor",
        "multiplicity_method": "paired_max_statistic",
        "external_independence_class": "not_external",
        "eligible_wording": "development prediction",
        "forbidden_wording": ["general population claim"],
    }
    registry = {
        "schema_version": 1,
        "registry_id": "pending",
        "frozen_before_first_relevant_outer_evaluation": True,
        "records": [record],
    }
    registry["registry_id"] = contract_id(registry, id_field="registry_id")
    robustness = {
        "schema_version": 1,
        "plan_id": "pending",
        "axes": [
            {
                "axis_id": "seed",
                "values": ["base", "sensitivity"],
                "frozen_before_corresponding_outer_outcomes": True,
            }
        ],
        "base_contract_hashes": ["a" * 64],
        "model_fitting": False,
        "model_selection": False,
        "threshold_adjustment": False,
        "target_discovery": False,
        "sensitivities_select_reported_model": False,
    }
    robustness["plan_id"] = contract_id(robustness, id_field="plan_id")
    multiplicity = {
        "schema_version": 1,
        "multiplicity_contract_id": "pending",
        "claim_registry_id": registry["registry_id"],
        "families": [
            {
                "family_id": "P",
                "claim_ids": ["STATE_E8"],
                "method": "paired_max_statistic",
                "error_rate": 0.05,
                "conditional_on_observed_donors": True,
            }
        ],
        "general_population_donor_claim_allowed": False,
    }
    multiplicity["multiplicity_contract_id"] = contract_id(
        multiplicity, id_field="multiplicity_contract_id"
    )
    seal = {
        "schema_version": 1,
        "g14_contract_id": "pending",
        "claim_registry_id": registry["registry_id"],
        "robustness_plan_id": robustness["plan_id"],
        "multiplicity_contract_id": multiplicity["multiplicity_contract_id"],
        "purpose": "robustness_multiplicity_and_claim_sealing",
        "model_fitting": False,
        "model_selection": False,
        "threshold_adjustment": False,
        "target_discovery": False,
        "required_outputs": list(OUTPUTS),
    }
    seal["g14_contract_id"] = contract_id(seal, id_field="g14_contract_id")
    for name, payload, expected in (
        ("registry-v1.json", registry, "ClaimRegistryV1"),
        ("robustness-v1.json", robustness, "G14RobustnessPlanV1"),
        ("multiplicity-v1.json", multiplicity, "G14MultiplicityContractV1"),
        ("seal-v1.json", seal, "G14SealContractV1"),
    ):
        path = tmp_path / name
        path.write_text(json.dumps(payload) + "\n")
        assert validate_contract(path)["contract_type"] == expected


def test_g14_registry_rejects_parent_cycles() -> None:
    first = ClaimRecord(
        claim_id="A",
        component_id="G07",
        parent_claim_ids=("B",),
        endpoint="a",
        candidate="a",
        comparator="null",
        direction="lower",
        primary_or_secondary="primary",
        claim_family="P",
        resampling_unit="target_within_donor",
        multiplicity_method="paired_max_statistic",
        required_robustness_axis_ids=("seed",),
        eligible_wording="a",
        forbidden_wording=("causal",),
    )
    second = first.model_copy(update={"claim_id": "B", "parent_claim_ids": ("A",)})
    payload = {
        "schema_version": 2,
        "registry_id": "pending",
        "frozen_before_first_relevant_outer_evaluation": True,
        "records": [first.model_dump(mode="json"), second.model_dump(mode="json")],
    }
    payload["registry_id"] = contract_id(payload, id_field="registry_id")
    with pytest.raises(ValidationError, match="acyclic"):
        ClaimRegistry.model_validate(payload)


def test_g14_cross_validation_rejects_method_axis_and_evidence_mismatch() -> None:
    registry, robustness, multiplicity, seal = _contracts()
    family_payload = multiplicity.model_dump(mode="json")
    family_payload["families"][0]["method"] = "bh_fdr_locked_family"
    family_payload["multiplicity_contract_id"] = "pending"
    family_payload["multiplicity_contract_id"] = contract_id(
        family_payload, id_field="multiplicity_contract_id"
    )
    incompatible = G14MultiplicityContract.model_validate(family_payload)
    seal_payload = seal.model_dump(mode="json")
    seal_payload["multiplicity_contract_id"] = incompatible.multiplicity_contract_id
    seal_payload["g14_contract_id"] = "pending"
    seal_payload["g14_contract_id"] = contract_id(seal_payload, id_field="g14_contract_id")
    incompatible_seal = G14SealContract.model_validate(seal_payload)
    with pytest.raises(IntegrityError, match="multiplicity method incompatible"):
        validate_g14_contracts(registry, robustness, incompatible, incompatible_seal)

    axis_payload = robustness.model_dump(mode="json")
    axis_payload["axes"][0]["claim_ids"] = ["UNKNOWN"]
    axis_payload["plan_id"] = "pending"
    axis_payload["plan_id"] = contract_id(axis_payload, id_field="plan_id")
    altered_axis = G14RobustnessPlan.model_validate(axis_payload)
    altered_seal_payload = seal.model_dump(mode="json")
    altered_seal_payload["robustness_plan_id"] = altered_axis.plan_id
    altered_seal_payload["g14_contract_id"] = "pending"
    altered_seal_payload["g14_contract_id"] = contract_id(
        altered_seal_payload, id_field="g14_contract_id"
    )
    with pytest.raises(IntegrityError, match="unknown claims"):
        validate_g14_contracts(
            registry,
            altered_axis,
            multiplicity,
            G14SealContract.model_validate(altered_seal_payload),
        )

    evidence_payload = seal.model_dump(mode="json")
    evidence_payload["component_receipt_hashes"] = {"G08": "c" * 64}
    evidence_payload["g14_contract_id"] = "pending"
    evidence_payload["g14_contract_id"] = contract_id(evidence_payload, id_field="g14_contract_id")
    with pytest.raises(IntegrityError, match="every and only"):
        validate_g14_contracts(
            registry,
            robustness,
            multiplicity,
            G14SealContract.model_validate(evidence_payload),
        )


def test_g14_blocks_mechanism_without_parent_and_unresolved_external_promotion() -> None:
    predictive = ClaimRecord(
        claim_id="STATE",
        component_id="G07",
        endpoint="state",
        candidate="model",
        comparator="baseline",
        direction="lower",
        primary_or_secondary="primary",
        claim_family="P",
        resampling_unit="target_within_donor",
        multiplicity_method="paired_max_statistic",
        required_robustness_axis_ids=("seed",),
        eligible_wording="predictive",
        forbidden_wording=("causal",),
    )
    mechanism = ClaimRecord(
        claim_id="MECHANISM",
        component_id="G09",
        parent_claim_ids=("STATE",),
        endpoint="mechanism",
        candidate="joint",
        comparator="separate",
        direction="lower",
        primary_or_secondary="secondary",
        claim_family="M",
        resampling_unit="target_within_donor",
        multiplicity_method="joint_max_statistic",
        required_robustness_axis_ids=("seed",),
        eligible_wording="internal mechanism",
        forbidden_wording=("causal",),
    )
    external = ClaimRecord(
        claim_id="EXTERNAL",
        component_id="G12",
        endpoint="arrayed_assay",
        candidate="model",
        comparator="null",
        direction="higher",
        primary_or_secondary="secondary",
        claim_family="E",
        resampling_unit="donor",
        multiplicity_method="holm_fwer",
        external_independence_class="unresolved_blocked",
        required_robustness_axis_ids=("seed",),
        eligible_wording="blocked external association",
        forbidden_wording=("independent validation",),
    )
    registry = _identified(
        ClaimRegistry,
        {
            "schema_version": 2,
            "frozen_before_first_relevant_outer_evaluation": True,
            "records": [
                predictive.model_dump(mode="json"),
                mechanism.model_dump(mode="json"),
                external.model_dump(mode="json"),
            ],
        },
        "registry_id",
    )
    robustness = _identified(
        G14RobustnessPlan,
        {
            "schema_version": 2,
            "axes": [
                RobustnessAxis(
                    axis_id="seed",
                    values=("base", "sensitivity"),
                    claim_ids=("STATE", "MECHANISM", "EXTERNAL"),
                ).model_dump(mode="json")
            ],
            "base_contract_hashes": ["a" * 64],
        },
        "plan_id",
    )
    multiplicity = _identified(
        G14MultiplicityContract,
        {
            "schema_version": 2,
            "claim_registry_id": registry.registry_id,
            "families": [
                G14MultiplicityFamily(
                    family_id="P",
                    claim_ids=("STATE",),
                    method="paired_max_statistic",
                    conditional_on_observed_donors=True,
                ).model_dump(mode="json"),
                G14MultiplicityFamily(
                    family_id="M",
                    claim_ids=("MECHANISM",),
                    method="joint_max_or_holm",
                ).model_dump(mode="json"),
                G14MultiplicityFamily(
                    family_id="E",
                    claim_ids=("EXTERNAL",),
                    method="holm_or_westfall_young",
                ).model_dump(mode="json"),
            ],
            "general_population_donor_claim_allowed": False,
        },
        "multiplicity_contract_id",
    )

    def seal(decisions: dict[str, str]) -> G14SealContract:
        return _identified(
            G14SealContract,
            {
                "schema_version": 2,
                "claim_registry_id": registry.registry_id,
                "robustness_plan_id": robustness.plan_id,
                "multiplicity_contract_id": multiplicity.multiplicity_contract_id,
                "claim_decisions": decisions,
                "component_receipt_hashes": {
                    "G07": "b" * 64,
                    "G09": "c" * 64,
                    "G12": "d" * 64,
                },
                "required_outputs": OUTPUTS,
            },
            "g14_contract_id",
        )

    with pytest.raises(IntegrityError, match="predictive parents"):
        validate_g14_contracts(
            registry,
            robustness,
            multiplicity,
            seal({"STATE": "not_promoted", "MECHANISM": "promoted", "EXTERNAL": "blocked"}),
        )
    with pytest.raises(IntegrityError, match="Unresolved external"):
        validate_g14_contracts(
            registry,
            robustness,
            multiplicity,
            seal({"STATE": "promoted", "MECHANISM": "promoted", "EXTERNAL": "promoted"}),
        )
