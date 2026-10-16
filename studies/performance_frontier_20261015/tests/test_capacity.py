"""Fabricated scalar metadata; no model, tensor, optimizer or payload access."""

from dataclasses import replace
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest

from frontier.capacity import (CapacityCalibration, CapacityRunEvidence, compare_capacity_edge,
    report_capacity_selector, objective_scientific_identity, actual_draw_receipt_identity)
from frontier.configuration import DONORS, FAMILIES
from frontier.planner import family_plan, screen_plan
from frontier.selection import CHANNELS
from frontier.selector_contract import ChannelCoverage, CostEvidence, EvidenceContext, Recipe, ScoreEvidence


def digest(value):
    return sha256(str(value).encode()).hexdigest()


def fixed(alpha=1., spread=1.):
    return CapacityCalibration(alpha, spread, digest(("prequery-calibration", alpha, spread)), False)


def run_for(task, *, stage="screen", values=None, alpha=1., spread=1.):
    pair, donor = task.key.training_donors, task.validation_donor
    variant = task.key.configuration_id.split("-")[1]
    context = EvidenceContext(stage, digest("runtime"), digest(("representation", pair)),
        digest(("fitting", pair)), digest(("query-source", donor)), digest(("truth", donor)),
        digest("scorer"), 4096 if stage == "finalist" else 512, pair)
    coverage = {c: ChannelCoverage(digest(("coverage", donor, c)), 1 if "control_endpoint" in c else 20) for c in CHANNELS}
    score = ScoreEvidence(task, Recipe(task.key.configuration_id, alpha, spread), context, coverage,
        values or {c: 1. if variant == "V1" else .8 for c in CHANNELS})
    cost = CostEvidence(task.key.configuration_id, context.runtime_identity,
        digest(("measured-profile", task.identifier)), 2. if variant == "V1" else 3.5,
        50000 if variant == "V1" else 140000)
    return CapacityRunEvidence(score, cost, digest(("checkpoint", task.key.identifier, task.updates)),
        digest("model-source"), digest("objective"), digest(("scales", pair)), digest(("source-noise", pair, task.updates)),
        digest(("parameter-inventory", task.key.identifier)), digest("same-profile-workload"), "seconds_per_bound_forecast")


def evidence(plan=None, selector="D1", family="F0", **kwargs):
    plan = plan or screen_plan()
    return [run_for(task, stage=plan.name, **kwargs) for task in plan.scores_for(selector)
            if task.key.configuration_id in {f"{family}-V1", f"{family}-V2"}]


def first_pair(rows):
    small = next(row for row in rows if row.score.recipe.configuration_id.endswith("V1"))
    large = next(row for row in rows if row.score.recipe.configuration_id.endswith("V2")
                 and row.score.task.key.training_donors == small.score.task.key.training_donors)
    return small, large


def test_exact_width_comparison_keeps_all_channels_and_actual_cost():
    plan = screen_plan()
    small, large = first_pair(evidence(plan))
    result = compare_capacity_edge(small, large, plan=plan, selector="D1", calibration=fixed())
    assert set(result["paired_differences"]) == set(CHANNELS)
    assert all(value == pytest.approx(-.2) for value in result["paired_differences"].values())
    assert result["relationship"] == "larger_dominates"
    assert result["nondominated_widths"] == ["width256"]
    assert result["active_parameter_difference"] == 90000
    assert result["measured_inference_cost_ratio"] == 1.75
    assert result["interpretation"] == "finite_budget_screening_only"
    assert result["longer_budget_inference"] is False and result["additional_fits"] == 0


def test_tradeoffs_and_equal_scores_do_not_create_a_single_winner():
    plan = screen_plan()
    small, large = first_pair(evidence(plan))
    risks = dict(large.score.risks)
    risks["spread_absolute_error"] = 1.2
    large = replace(large, score=replace(large.score, risks=risks))
    result = compare_capacity_edge(small, large, plan=plan, selector="D1", calibration=fixed())
    assert result["relationship"] == "trade_off" and result["nondominated_widths"] == ["width128", "width256"]
    assert result["direction"]["spread_absolute_error"] == "larger_higher_risk"
    large = replace(large, score=replace(large.score, risks=dict(small.score.risks)))
    assert compare_capacity_edge(small, large, plan=plan, selector="D1", calibration=fixed())["relationship"] == "equal"


def test_selector_uses_only_its_three_edges_without_latent_raw_pooling():
    plan = screen_plan()
    result = report_capacity_selector(plan, "D1", "F0", evidence(plan), calibration=fixed())
    assert result["expected_edges"] == result["available_edges"] == 3
    assert result["complete"] is True and result["global_capacity_winner"] is None
    assert result["pooled_raw_latent_risks"] is None and result["selection_rule_changed"] is False
    assert result["direction_counts"]["state_energy"] == {"larger_lower_risk": 3, "equal": 0, "larger_higher_risk": 0}
    assert all("D1" not in row["training_donors"] and row["validation_donor"] != "D1" for row in result["edges"])
    with pytest.raises(ValueError, match="authorized"):
        report_capacity_selector(plan, "D1", "F0", evidence(plan, "D2"), calibration=fixed())


def test_all_six_families_twelve_distinct_directed_edges_each_without_global_summary():
    plan = screen_plan()
    for family in FAMILIES:
        directed = set()
        for selector in DONORS:
            report = report_capacity_selector(plan, selector, family, evidence(plan, selector, family), calibration=fixed())
            directed.update((tuple(row["training_donors"]), row["validation_donor"]) for row in report["edges"])
        assert len(directed) == 12


def test_missing_saved_score_is_unavailable_not_a_favorable_subset():
    plan = screen_plan()
    rows = evidence(plan)
    result = report_capacity_selector(plan, "D1", "F0", rows[:-1], calibration=fixed())
    assert result["complete"] is False and result["available_edges"] == 2
    missing = next(row for row in result["edges"] if not row["available"])
    assert missing["paired_differences"] is None
    assert "missing_saved_score_evidence" in missing["unavailable"].values()
    with pytest.raises(ValueError, match="Duplicate"):
        report_capacity_selector(plan, "D1", "F0", rows+rows[:1], calibration=fixed())


def test_later_pruning_does_not_recover_other_selector_variant_or_request_fits():
    choices = {d: tuple(f"{f}-V1" for f in FAMILIES) for d in DONORS}
    choices["D2"] = tuple(f"{f}-V2" for f in FAMILIES)
    plan = family_plan(choices)
    rows = evidence(plan)
    result = report_capacity_selector(plan, "D1", "F0", rows, calibration=fixed())
    assert result["available_edges"] == 0 and result["additional_updates"] == 0
    assert all(row["unavailable"]["F0-V2"] == "variant_not_authorized_for_this_selector_stage" for row in result["edges"])
    with pytest.raises(ValueError, match="authorized"):
        report_capacity_selector(plan, "D1", "F0", rows+evidence(plan, "D2"), calibration=fixed())


@pytest.mark.parametrize("field", ["model_source_identity", "objective_scientific_identity", "normalization_identity", "actual_draw_identity", "inference_workload_identity", "inference_cost_unit"])
def test_unmatched_method_sampling_and_cost_scope_rejected(field):
    plan = screen_plan()
    small, large = first_pair(evidence(plan))
    large = replace(large, **{field: "other_unit" if field == "inference_cost_unit" else digest("different")})
    with pytest.raises(ValueError, match="Unmatched"):
        compare_capacity_edge(small, large, plan=plan, selector="D1", calibration=fixed())


def test_context_support_updates_and_parameter_counts_are_binding():
    plan = screen_plan()
    small, large = first_pair(evidence(plan))
    changed = replace(large, score=replace(large.score, context=replace(large.score.context, representation_identity=digest("different"))))
    with pytest.raises(ValueError, match="representation"):
        compare_capacity_edge(small, changed, plan=plan, selector="D1", calibration=fixed())
    coverage = dict(large.score.coverage)
    coverage[CHANNELS[0]] = ChannelCoverage(digest("different-support"), 19)
    changed = replace(large, score=replace(large.score, coverage=coverage))
    with pytest.raises(ValueError, match="support"):
        compare_capacity_edge(small, changed, plan=plan, selector="D1", calibration=fixed())
    changed = replace(large, score=replace(large.score, task=replace(large.score.task, updates=3072)))
    with pytest.raises(ValueError, match="authorized"):
        compare_capacity_edge(small, changed, plan=plan, selector="D1", calibration=fixed())
    changed = replace(large, cost=replace(large.cost, active_parameters=small.cost.active_parameters))
    with pytest.raises(ValueError, match="not larger"):
        compare_capacity_edge(small, changed, plan=plan, selector="D1", calibration=fixed())


def test_calibration_must_be_identical_and_fixed_without_query_outcomes():
    plan = screen_plan()
    small, large = first_pair(evidence(plan, alpha=.25, spread=1.3))
    assert compare_capacity_edge(small, large, plan=plan, selector="D1", calibration=fixed(.25, 1.3))["calibration"]["alpha"] == .25
    with pytest.raises(ValueError, match="calibration"):
        compare_capacity_edge(small, large, plan=plan, selector="D1", calibration=fixed())
    with pytest.raises(ValueError, match="without"):
        replace(fixed(), used_query_outcomes=True)


def test_measured_metadata_required_and_no_optional_time_defaults():
    small, _ = first_pair(evidence())
    assert small.measured_fit_seconds is None and small.measured_scoring_seconds is None
    with pytest.raises(ValueError, match="positive measured"):
        replace(small, cost=replace(small.cost, inference_cost=0.))
    with pytest.raises(ValueError, match="configuration and runtime"):
        replace(small, cost=replace(small.cost, runtime_identity=digest("different")))
    with pytest.raises(ValueError, match="seconds"):
        replace(small, measured_fit_seconds=float("nan"))
    with pytest.raises(ValueError, match="SHA-256"):
        replace(small, parameter_inventory_identity="inferred default")


def test_real_registry_ids_differ_but_scientific_objective_matches_only_with_same_rules_and_scales():
    from frontier.objective_registry import (FittingScaleBinding, ObjectiveRules, RULES_SHA256,
        SCALE_NAMES, build_objective_protocol)
    rules = ObjectiveRules((Path(__file__).resolve().parents[1]/"MAIN_OBJECTIVE_RULES.json").read_bytes(), RULES_SHA256)
    scales = FittingScaleBinding(("D3", "D4"), RULES_SHA256, "1"*64, "2"*64, "3"*64,
        "4"*64, "5"*64, "6"*64, dict(zip(SCALE_NAMES, (1., .2, .3, .4, .5))))
    def protocol(config, scale=scales):
        return build_objective_protocol(config, rules=rules, scales=scale, fitting_donors=scale.fitting_donors,
            observation_artifact_sha256=scale.observation_artifact_sha256, fitting_input_sha256=scale.fitting_input_sha256)
    for family in FAMILIES:
        p1, p2 = protocol(f"{family}-V1"), protocol(f"{family}-V2")
        assert p1.protocol_id != p2.protocol_id
        first = objective_scientific_identity(p1, configuration_id=f"{family}-V1", rules=rules, scales=scales)
        second = objective_scientific_identity(p2, configuration_id=f"{family}-V2", rules=rules, scales=scales)
        assert first == second
        with pytest.raises(ValueError, match="exact registry"):
            objective_scientific_identity(replace(p2, state_coefficient=.6), configuration_id=f"{family}-V2", rules=rules, scales=scales)
    changed = replace(scales, values={**dict(scales.values), "state": 2.})
    assert objective_scientific_identity(protocol("F0-V1"), configuration_id="F0-V1", rules=rules, scales=scales) != objective_scientific_identity(
        protocol("F0-V2", changed), configuration_id="F0-V2", rules=rules, scales=changed)
    with pytest.raises(ValueError, match="exact registry"):
        objective_scientific_identity(protocol("F0-V1"), configuration_id="F0-V2", rules=rules, scales=scales)


def test_actual_sampler_receipts_preserve_equal_draws_while_cfg_receipt_ids_differ():
    import numpy as np
    import torch
    from frontier.sampling import LawIdentity, SourceLaw, LawRequest, SamplingContext, SamplerSpecification, sample_batch
    from frontier.selector_contract import _digest
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        source = SourceLaw(LawIdentity("D3", "Stim8hr", "guide1", "source1", "replicate1"),
            np.arange(3*48, dtype=np.float32).reshape(3, 48), np.array([.2, .3, .5], dtype=np.float64))
        spec = SamplerSpecification(maximum_particles=8, maximum_laws=1)
        contexts = [SamplingContext("fabricated", ("D3", "D4"), f"F0-V{variant}", 3, 0, 5, "training") for variant in (1, 2)]
        batches = [sample_batch([LawRequest(source)], context, particles=4, specification=spec) for context in contexts]
        assert _digest(batches[0].receipt) != _digest(batches[1].receipt)
        torch.testing.assert_close(batches[0].states, batches[1].states, rtol=0, atol=0)
        torch.testing.assert_close(batches[0].noise, batches[1].noise, rtol=0, atol=0)
        output_digest = lambda batch: sha256(batch.states.numpy().tobytes()+batch.noise.numpy().tobytes()+batch.source_indices.numpy().tobytes()).hexdigest()
        kwargs = dict(sampler_source_identity="a"*64,
            source_law_payload_identity=sha256(source.states.tobytes()+source.weights.tobytes()).hexdigest(),
            sampled_output_identity=output_digest(batches[0]))
        first, second = (actual_draw_receipt_identity(batch.receipt, **kwargs) for batch in batches)
        assert first == second
        changed = sample_batch([LawRequest(source)], replace(contexts[1], accepted_update=6), particles=4, specification=spec)
        assert actual_draw_receipt_identity(changed.receipt, **{**kwargs, "sampled_output_identity": output_digest(changed)}) != first
        assert actual_draw_receipt_identity(batches[1].receipt, **{**kwargs, "source_law_payload_identity": "c"*64}) != first
        assert actual_draw_receipt_identity(batches[1].receipt, **{**kwargs, "sampled_output_identity": "e"*64}) != first
        corrupted = deepcopy(batches[1].receipt)
        corrupted["streams"][0]["noise_stream_sha256"] = "d"*64
        with pytest.raises(ValueError, match="stream differs"):
            actual_draw_receipt_identity(corrupted, **kwargs)
        corrupted = deepcopy(batches[1].receipt)
        corrupted["standard_normal_draws"] += 1
        with pytest.raises(ValueError, match="counts"):
            actual_draw_receipt_identity(corrupted, **kwargs)
    finally:
        torch.set_num_threads(previous)
