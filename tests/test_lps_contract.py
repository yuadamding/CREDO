from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

import credo
from credo.contracts import SplitSpec
from credo.data import (
    ContextTable,
    InterventionEventTable,
    LPSCompositionTable,
    PerturbationReferenceBindingTable,
    PopulationPoolTable,
    PopulationSeriesTable,
    SnapshotObservationTable,
    SupportIndexTable,
)
from credo.data.splits import validate_split_plan
from credo.problems import FiniteMeasureDynamicsProblem
from credo.registry import get_recipe


def test_public_study_is_the_domain_specific_lps_contract(tiny_config) -> None:
    study = credo.open_study(tiny_config)
    try:
        assert credo.Study is credo.PerturbSeqStudy
        assert isinstance(study, credo.PerturbSeqStudy)
        assert study.manifest.schema_version == 4
        assert set(study.series.to_pandas()["continuity_kind"]) == {"unknown"}
        assert set(study.intervention_events.to_pandas()["start_relation"]) == {"unknown"}
        assert {
            "start_coordinate",
            "end_coordinate",
            "dose",
            "dose_unit",
        } <= set(study.intervention_events.columns)
    finally:
        study.close()


def test_target_selection_does_not_conflate_guide_target_and_effect(tiny_config) -> None:
    study = credo.open_study(tiny_config)
    try:
        components = study.perturbation_components.to_pandas()
        row = components.loc[components["target_id"].eq("GENE1")].iloc[0]
        assert row["construct_id"] != row["target_id"]
        view = study.view(credo.SelectionSpec(target_ids=("GENE1",)))
        selected = set(view.perturbation_ids)
        expected = set(components.loc[components["target_id"].eq("GENE1"), "perturbation_id"])
        assert selected == expected
        binding = view.effect_binding()
        assert set(binding["perturbation_id"]) == selected
        assert "effect_id" in binding
    finally:
        study.close()


def test_observation_qc_selection_closes_over_series_and_perturbations(tiny_config) -> None:
    study = credo.open_study(tiny_config)
    try:
        selected_series = study.series.series_ids[0]
        observations = study.observations.to_pandas()
        observations["assignment_qc"] = observations["series_id"].map(
            lambda value: "gold" if value == selected_series else "silver"
        )
        native = replace(
            study,
            observations=SnapshotObservationTable(observations),
            provenance={"fixture": "native-v4"},
        )
        view = native.view(credo.SelectionSpec(qc_tiers=("gold",)))
        expected_perturbation = (
            study.series.to_pandas().set_index("series_id").loc[selected_series, "perturbation_id"]
        )
        assert view.series_ids == (selected_series,)
        assert set(view.observations()["series_id"]) == {selected_series}
        assert view.perturbation_ids == (expected_perturbation,)
        assert set(view.perturbations()["perturbation_id"]) == {expected_perturbation}
    finally:
        study.close()


def test_subject_split_compiles_outcome_separated_finite_measures(tiny_config) -> None:
    study = credo.open_study(tiny_config)
    try:
        recipe = get_recipe(tiny_config.recipe)
        view = study.view()
        plan = recipe.plan_split(
            view,
            tiny_config.recipe_config,
            SplitSpec(strategy="subject", validation_values=("D2",)),
        )
        validate_split_plan(view, plan)
        assert plan.task_kind == "subject_generalization"
        assert plan.held_out_subject_ids == ("D2",)
        problem = recipe.compile(view, plan, tiny_config.recipe_config)
        assert isinstance(problem, FiniteMeasureDynamicsProblem)
        assert set(problem.training.measure_meta["sample_id"]) == {"D1"}
        assert set(problem.validation.measure_meta["sample_id"]) == {"D2"}
        assert not (
            set(problem.partition.training_targets.observation_ids)
            & set(problem.partition.validation_targets.observation_ids)
        )
    finally:
        study.close()


def test_compact_compiles_checkpoint_specific_destructive_samples(tiny_config) -> None:
    study = credo.open_study(tiny_config)
    try:
        series = study.series.to_pandas()
        series["context_trajectory_id"] = (
            series["subject_id"].astype(str) + "::stimulation_trajectory"
        )
        subject_by_series = series.set_index("series_id")["subject_id"].astype(str)
        observations = study.observations.to_pandas()
        subjects = observations["series_id"].map(subject_by_series)
        observations["sample_id"] = (
            subjects + "::sample::" + observations["checkpoint_id"].astype(str)
        )
        observations["context_id"] = observations["sample_id"]
        contexts = ContextTable(
            observations[["context_id"]]
            .drop_duplicates()
            .assign(context_kind="destructive_checkpoint_sample")
        )
        compositions = study.compositions.to_pandas()
        context_by_observation = observations.set_index("observation_id")["context_id"]
        compositions["context_id"] = compositions["observation_id"].map(context_by_observation)
        native = replace(
            study,
            contexts=contexts,
            series=PopulationSeriesTable(series),
            observations=SnapshotObservationTable(observations),
            compositions=LPSCompositionTable(compositions),
            provenance={"fixture": "native-v4"},
        )
        settings = tiny_config.recipe_config
        settings = settings.model_copy(
            update={"model": settings.model.model_copy(update={"context": "none"})}
        )
        recipe = get_recipe(tiny_config.recipe)
        view = native.view()
        plan = recipe.plan_split(
            view,
            settings,
            SplitSpec(strategy="none"),
        )
        validate_split_plan(view, plan)
        view.validate_for(recipe.requirements(settings), plan).raise_for_errors()
        problem = recipe.compile(view, plan, settings)
        expected_contexts = set(series["context_trajectory_id"])
        compiled_contexts = set(problem.training.measure_meta["context_group_id"]) | set(
            problem.validation.measure_meta["context_group_id"]
        )
        assert compiled_contexts == expected_contexts
        block_contexts = {
            block.context_group_id
            for data in (problem.training, problem.validation)
            for block in data.count_blocks
        }
        assert block_contexts <= expected_contexts
    finally:
        study.close()


def test_subject_holdout_excludes_source_only_series_from_training(tiny_config) -> None:
    study = credo.open_study(tiny_config)
    try:
        series = study.series.to_pandas()
        held_subject = str(series["subject_id"].iloc[-1])
        source_only_series = str(
            series.loc[series["subject_id"].eq(held_subject), "series_id"].iloc[0]
        )
        observations = study.observations.to_pandas()
        downstream = observations["checkpoint_id"].ne(study.design.source_checkpoint_id)
        hidden = downstream & observations["series_id"].eq(source_only_series)
        hidden_ids = set(observations.loc[hidden, "observation_id"])
        observations.loc[hidden, "geometry_observed"] = False
        support = study.support_index.to_pandas()
        hidden_support = support["observation_id"].isin(hidden_ids)
        support.loc[hidden_support, ["store_id", "support_key"]] = None
        support.loc[hidden_support, "available"] = False
        native = replace(
            study,
            observations=SnapshotObservationTable(observations),
            support_index=SupportIndexTable(support),
        )
        recipe = get_recipe(tiny_config.recipe)
        view = native.view()
        plan = recipe.plan_split(
            view,
            tiny_config.recipe_config,
            SplitSpec(strategy="subject", validation_values=(held_subject,)),
        )
        validate_split_plan(view, plan)
        train_subjects = set(
            series.loc[series["series_id"].isin(plan.train_series_ids), "subject_id"]
        )
        assert held_subject not in train_subjects
        assert source_only_series in plan.validation_series_ids
        assert plan.held_out_subject_ids == (held_subject,)
    finally:
        study.close()


def test_target_split_is_content_addressed_before_recipe_capability_check(tiny_config) -> None:
    study = credo.open_study(tiny_config)
    try:
        recipe = get_recipe(tiny_config.recipe)
        view = study.view()
        targets = study.perturbation_components.to_pandas()["target_id"]
        held_out_target = next(value for value in targets.unique() if value != "__control__")
        plan = recipe.plan_split(
            view,
            tiny_config.recipe_config,
            SplitSpec(strategy="target", validation_values=(held_out_target,)),
        )
        validate_split_plan(view, plan)
        assert plan.task_kind == "target_generalization"
        assert plan.held_out_target_ids == (held_out_target,)
        assert plan.held_out_perturbation_ids
        assert plan.split_id
    finally:
        study.close()


def test_sequencing_pool_is_not_population_ecology(tiny_config) -> None:
    study = credo.open_study(tiny_config)
    try:
        observations = study.observations.to_pandas()
        series = study.series.to_pandas().set_index("series_id")
        observations["population_pool_id"] = (
            observations["sample_id"] + "@" + observations["checkpoint_id"]
        )
        pools = observations[["population_pool_id", "checkpoint_id", "series_id"]].copy()
        pools["experimental_unit_id"] = pools["series_id"].map(series["experimental_unit_id"])
        pools = pools.drop_duplicates("population_pool_id")
        pools["pool_kind"] = "sequencing_library"
        pools["evidence_level"] = "declared"
        pools["description"] = "Shared sequencing denominator only"
        native = replace(
            study,
            observations=SnapshotObservationTable(observations),
            population_pools=PopulationPoolTable(pools.drop(columns="series_id")),
            provenance={"fixture": "native-v4"},
        )
        recipe = get_recipe(tiny_config.recipe)
        report = native.view().validate_for(
            recipe.requirements(tiny_config.recipe_config),
            SplitSpec(strategy="none"),
        )
        assert native.view().ecological_pools().empty
        assert any(issue.code == "recipe.population_ecology" for issue in report.errors)
    finally:
        study.close()


def test_v4_rejects_undeclared_contexts_and_opaque_reference_matching(tiny_config) -> None:
    study = credo.open_study(tiny_config)
    try:
        report = replace(study, contexts=None).validate("semantic")
        assert any(issue.code == "observations.context_fk" for issue in report.errors)
    finally:
        study.close()

    with pytest.raises(ValueError, match="JSON string array"):
        PerturbationReferenceBindingTable(
            pd.DataFrame(
                {
                    "binding_id": ["reference"],
                    "perturbation_id": ["guide-1"],
                    "reference_pool_id": ["ntc"],
                    "scope_kind": ["subject"],
                    "match_keys": ["subject_id"],
                    "counterfactual_effect_id": ["reference"],
                }
            )
        )
    with pytest.raises(ValueError, match="missing columns"):
        InterventionEventTable(
            pd.DataFrame(
                {
                    "event_id": ["event"],
                    "series_id": ["series"],
                    "agent_id": ["guide-1"],
                    "event_kind": ["crispr_i"],
                    "modeled_role": ["primary_perturbation"],
                    "start_relation": ["before_source"],
                    "persistent": [True],
                }
            )
        )
