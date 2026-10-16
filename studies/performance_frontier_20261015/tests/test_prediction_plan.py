"""Pure registered metadata: no observation/model/optimizer/remote operations."""

from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from pathlib import Path

import pytest

from frontier import prediction_plan
from frontier.configuration import DONORS, FAMILIES
from frontier.planner import (StagePlan, TrainingTask, family_plan, finalist_plan,
                              outer_plan, screen_plan)
from frontier.prediction_plan import (PredictionPlan, prediction_budget,
    required_before_continuation, require_continuation_forecasts)
from frontier.selector_contract import OuterRecipe, Recipe


def stages(unique=False):
    second = family_plan({d: tuple(f"{f}-V{i if unique else 0}" for f in FAMILIES)
                          for i, d in enumerate(DONORS)})
    third = finalist_plan(second, {d: tuple(f"{f}-V{i if unique else 0}" for f in ("F0", "F1", "F4"))
                                   for i, d in enumerate(DONORS)})
    return screen_plan(), second, third, outer_plan(third)


def frozen(stage):
    return {d: tuple(OuterRecipe(d, tuple(x for x in DONORS if x != d), config, .25, 1.3,
                     str(i+1)*64, str(i+5)*64, .8) for config in stage.eligible_for(d))
            for i, d in enumerate(DONORS)}


def plans(unique=False):
    graph = stages(unique)
    return tuple(PredictionPlan.from_stage(stage, outer_recipes=frozen(stage) if stage.name == "outer" else None)
                 for stage in graph)


@pytest.mark.parametrize("unique,updates", [(False, 663552), (True, 847872)])
def test_complete_graph_counts_training_sharing_without_query_sharing(unique, updates):
    values = plans(unique)
    result = prediction_budget(values)
    assert result["main_training"]["planned_main_updates"] == updates
    assert result["main_training"]["main_update_cap"] == 847872
    assert [len(p.forecasts) for p in values] == [288, 72, 36, 36]
    assert [len(p.groups) for p in values] == [12, 12, 12, 4]
    assert result["raw_bank_operations"] == result["directed_score_edges"] == 432
    assert result["inner_recipe_edge_assessments"] == 396*24 == 9504
    assert result["outer_single_seed_prefix_assessments"] == 72
    assert result["ensemble_operations"] == 12 and result["ensemble_total_assessments"] == 24
    assert result["scientific_launch_authority"] is False
    assert result["automatic_budget_expansion"] is False
    assert result["implementation"]["sha256"] == sha256(Path(prediction_plan.__file__).read_bytes()).hexdigest()


def test_inner_forecast_groups_preserve_query_roles_and_complete_calibration_grid():
    first, second, third, _ = plans(True)
    for plan, per_group, prefix in ((first, 24, 512), (second, 6, 512), (third, 3, 4096)):
        record = plan.record()
        assert record["deletion_authority"] is record["scientific_launch_authority"] is False
        assert record["actual_publication_verification_required"] is True
        assert not plan.ensembles
        for group in plan.groups:
            assert len(group.forecast_ids) == len(group.edge_ids) == per_group
            assert group.query_donor != group.permitted_selector
            assert group.query_donor not in group.fitting_donors
            assert group.permitted_selector not in group.fitting_donors
        for operation in plan.forecasts:
            assert operation.single_seed_prefixes == operation.effective_prefixes == (prefix,)
            assert len(operation.calibration_recipes) == len(set(operation.calibration_recipes)) == 24
            assert Recipe(operation.edge.key.configuration_id, 1, 1) in operation.calibration_recipes
            assert not operation.ensemble_component_totals
            assert operation.edge.key.seed == 0
    pair = [o for o in first.forecasts if o.edge.key.training_donors == ("D3", "D4")
            and o.edge.key.configuration_id == "F0-V0"]
    assert len(pair) == 2 and pair[0].edge.key == pair[1].edge.key
    assert {(o.edge.selector, o.edge.validation_donor) for o in pair} == {("D1", "D2"), ("D2", "D1")}
    assert pair[0].identifier != pair[1].identifier


def test_outer_recipe_triplets_bind_all_banks_corrected_prefixes_and_fixed_total_mixtures():
    outer = plans()[-1]
    bank_map = {o.identifier: o for o in outer.forecasts}
    for group in outer.groups:
        assert group.query_donor == group.permitted_selector and len(group.forecast_ids) == 9
    for ensemble in outer.ensembles:
        r = ensemble.frozen_recipe
        components = [bank_map[key] for key in ensemble.parent_forecast_ids]
        assert [o.edge.key.seed for o in components] == [0, 1, 2]
        assert ensemble.total_particles == (4096, 8192)
        record = ensemble.record()
        assert record["component_counts"] == {"4096": [1366, 1365, 1365], "8192": [2731, 2731, 2730]}
        assert record["newly_generated_particles"] == 0
        for operation in components:
            assert operation.calibration_recipes == (Recipe(r.configuration_id, r.alpha, r.spread_scale),)
            assert operation.single_seed_prefixes == (4096, 8192)
            assert max(operation.effective_prefixes) == 8192
            assert operation.record()["raw_recipe"] == {"configuration_id": r.configuration_id, "alpha": 1., "spread_scale": 1.}


@pytest.mark.parametrize("change", ["absent", "missing_selector", "extra_selector", "missing_recipe", "duplicate_recipe",
    "wrong_config", "wrong_donor", "wrong_fold", "seed", "updates", "calibration", "evidence", "quality", "mixed_decision"])
def test_outer_inventory_never_infers_missing_or_foreign_frozen_recipes(change):
    stage = stages()[-1]
    supplied = frozen(stage)
    if change == "absent": supplied = None
    elif change == "missing_selector": supplied.pop("D4")
    elif change == "extra_selector": supplied["D5"] = supplied["D1"]
    elif change == "missing_recipe": supplied["D1"] = supplied["D1"][:2]
    elif change == "duplicate_recipe": supplied["D1"] = (supplied["D1"][0],)*3
    else:
        edits = {"wrong_config": dict(configuration_id="F0-V1"), "wrong_donor": dict(assessment_donor="D2"),
            "wrong_fold": dict(fitting_donors=("D1", "D3", "D4")), "seed": dict(seeds=(0, 1)),
            "updates": dict(updates=3072), "calibration": dict(alpha=.9), "evidence": dict(selection_evidence_identity="bad"),
            "quality": dict(inner_quality=float("nan")), "mixed_decision": dict(selection_contract_identity="f"*64)}
        supplied["D1"] = (replace(supplied["D1"][0], **edits[change]), *supplied["D1"][1:])
    with pytest.raises((TypeError, ValueError)):
        PredictionPlan.from_stage(stage, outer_recipes=supplied)


def test_missing_stage_edges_or_extra_outer_recipe_argument_rejected():
    first = screen_plan()
    with pytest.raises(ValueError):
        PredictionPlan.from_stage(first, outer_recipes={})
    with pytest.raises(TypeError): PredictionPlan(first)
    # Even a bypassed frozen-dataclass assignment is revalidated at admission.
    object.__setattr__(first, "scores", first.scores[:-1])
    with pytest.raises(ValueError): PredictionPlan.from_stage(first)


def test_plan_records_are_detached_and_metadata_is_immutable():
    plan = plans()[-1]
    original = plan.identity
    record = plan.record(); record["forecasts"].clear()
    assert len(plan.record()["forecasts"]) == 36 and plan.identity == original
    with pytest.raises(FrozenInstanceError): plan.forecasts = ()
    with pytest.raises(FrozenInstanceError): plan.forecasts[0].single_seed_prefixes = (1,)
    with pytest.raises(ValueError): prediction_budget((plan,))


@pytest.mark.parametrize("previous_name", ["screen", "family"])
def test_continuation_requires_other_selectors_previous_forecast_even_without_its_promotion(previous_name):
    first, second, third, _ = stages()
    previous, following = (first, second) if previous_name == "screen" else (second, third)
    task = next(t for t in following.training if t.key.training_donors == ("D3", "D4")
                and t.key.configuration_id == "F0-V0")
    promoted_only_d1 = replace(task, requesting_selectors=("D1",))
    required = required_before_continuation(previous, promoted_only_d1)
    edges = {s.identifier: s for s in previous.scores}
    assert len(required) == 2
    assert {edges[key].selector for key in required} == {"D1", "D2"}
    only_promoter = tuple(key for key in required if edges[key].selector == "D1")
    with pytest.raises(ValueError, match="still requires"):
        require_continuation_forecasts(previous, promoted_only_d1, only_promoter)
    assert require_continuation_forecasts(previous, promoted_only_d1, required) == required
    assert require_continuation_forecasts(previous, promoted_only_d1,
            tuple(s.identifier for s in previous.scores)) == required


@pytest.mark.parametrize("kind", ["missing", "duplicate", "foreign", "next_stage", "list"])
def test_continuation_completion_ids_are_exact_metadata_not_arbitrary_success_flags(kind):
    first, second, _, _ = stages()
    task = second.training[0]
    required = required_before_continuation(first, task)
    supplied = {"missing": required[:1], "duplicate": required+required[:1], "foreign": required+("success",),
        "next_stage": required+(second.scores[0].identifier,), "list": list(required)}[kind]
    with pytest.raises(ValueError): require_continuation_forecasts(first, task, supplied)


def test_invalid_or_unsupported_continuation_transition_rejected():
    first, second, third, fourth = stages(True)
    with pytest.raises(ValueError): required_before_continuation(first, third.training[0])
    with pytest.raises(ValueError): required_before_continuation(third, fourth.training[0])
    task = next(t for t in third.training if t.key.training_donors == ("D3", "D4")
                and t.key.configuration_id == "F0-V0")
    absent_config = replace(task.key, configuration_id="F0-V3")
    # This pair is requested by D1/D2 only; their distinct choices are V0/V1.
    with pytest.raises(ValueError): required_before_continuation(second, replace(task, key=absent_config))
    with pytest.raises(TypeError): required_before_continuation(first, True)
