from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

import credo.data.native_v4 as native_v4
from credo import PerturbSeqStudy, open_study, write_study
from credo.contracts import SplitSpec
from credo.data import (
    AbundanceChannelSpec,
    AbundanceSemantics,
    AbundanceTable,
    ArtifactRef,
    AxisSpec,
    Checkpoint,
    CompositionTable,
    ConditionTable,
    CurrentFiveFileStudyCodec,
    EffectBindingTable,
    EmpiricalLaw,
    InMemorySupportStore,
    ObservationTable,
    PerturbationEffectBindingTable,
    PerturbationReferenceBindingTable,
    PerturbationTable,
    PopulationSeriesTable,
    ReferenceBindingTable,
    ReplicatePolicy,
    RepresentationCatalog,
    RepresentationSpec,
    SelectionSpec,
    SeriesTable,
    StudyDesign,
    StudyManifest,
    SupportIndexTable,
    SupportRef,
    SupportStoreRegistry,
    Transition,
    available_study_codecs,
)
from credo.data import (
    Study as SchemaV3Study,
)
from credo.data.splits import validate_representation_scope, validate_split_plan
from credo.io import RunConfig, load_data, validate_inputs
from credo.registry import get_recipe


def _general_study() -> SchemaV3Study:
    design = StudyDesign(
        axes=(AxisSpec("time", "physical_time", "hour"),),
        checkpoints=(
            Checkpoint("rest", {"time": 0.0}, "source"),
            Checkpoint("stim", {"time": 8.0}, "target"),
        ),
        transitions=(Transition("rest_to_stim", "rest", "stim"),),
        topology="chain",
    )
    conditions = ConditionTable(
        pd.DataFrame(
            {
                "condition_id": ["ifng"],
                "condition_kind": ["cytokine_stimulation"],
                "embedding_id": ["IFNG"],
                "reference_group_id": ["media_control"],
                "is_reference": [False],
            }
        )
    )
    series = SeriesTable(
        pd.DataFrame(
            {
                "series_id": ["donor-a::ifng"],
                "condition_id": ["ifng"],
                "subject_id": ["donor-a"],
                "embedding_id": ["IFNG"],
                "reference_role": ["intervention"],
            }
        )
    )
    observations = ObservationTable(
        pd.DataFrame(
            {
                "observation_id": ["donor-a::ifng@rest", "donor-a::ifng@stim"],
                "series_id": ["donor-a::ifng", "donor-a::ifng"],
                "checkpoint_id": ["rest", "stim"],
                "sample_id": ["donor-a", "donor-a"],
                "geometry_observed": [True, True],
                "context_id": ["well-1", "well-2"],
                "composition_block_id": [None, "well-2@stim"],
            }
        )
    )
    channel = AbundanceChannelSpec(
        channel_id="raw_cell_count",
        semantics=AbundanceSemantics.ABSOLUTE,
        unit="cells",
        denominator_scope="none",
        zero_policy="allowed",
    )
    abundance = AbundanceTable(
        pd.DataFrame(
            {
                "observation_id": ["donor-a::ifng@rest", "donor-a::ifng@stim"],
                "channel_id": ["raw_cell_count", "raw_cell_count"],
                "value": [20.0, 0.0],
                "observed": [True, True],
            }
        ),
        (channel,),
    )
    compositions = CompositionTable(
        pd.DataFrame(
            {
                "composition_block_id": ["well-2@stim"],
                "checkpoint_id": ["stim"],
                "context_id": ["well-2"],
                "series_id": ["donor-a::ifng"],
                "observation_id": ["donor-a::ifng@stim"],
                "exposure": [1.0],
                "count": [0],
                "denominator_id": ["well-2@stim"],
            }
        )
    )
    law = EmpiricalLaw(
        np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
        np.asarray([0.25, 0.75]),
    )
    store = InMemorySupportStore(
        "memory",
        {
            SupportRef("memory", "latent-all", "source-law"): law,
            SupportRef("memory", "latent-all", "target-law"): law,
            SupportRef("memory", "latent-strict", "source-law"): law,
        },
    )
    representations = RepresentationCatalog(
        (
            RepresentationSpec("latent-all", "fixture", "latent", 2, "memory"),
            RepresentationSpec(
                "latent-strict",
                "fixture",
                "latent",
                2,
                "memory",
                fit_split_id="strict-stim-holdout",
                included_series=("donor-a::ifng",),
                included_checkpoints=("rest",),
            ),
        )
    )
    support_index = SupportIndexTable(
        pd.DataFrame(
            {
                "observation_id": [
                    "donor-a::ifng@rest",
                    "donor-a::ifng@stim",
                    "donor-a::ifng@rest",
                    "donor-a::ifng@stim",
                ],
                "representation_id": [
                    "latent-all",
                    "latent-all",
                    "latent-strict",
                    "latent-strict",
                ],
                "store_id": ["memory", "memory", "memory", None],
                "support_key": ["source-law", "target-law", "source-law", None],
                "available": [True, True, True, False],
            }
        )
    )
    return SchemaV3Study(
        manifest=StudyManifest(
            schema_version=3,
            study_id="general-fixture",
            source_schema="native_test",
            primary_representation="latent-all",
            primary_abundance_channel="raw_cell_count",
            primary_effect_binding="fixture_effect",
            primary_reference_binding="fixture_reference",
        ),
        design=design,
        conditions=conditions,
        series=series,
        observations=observations,
        support_index=support_index,
        abundance=abundance,
        compositions=compositions,
        representations=representations,
        supports=store,
        effect_bindings=EffectBindingTable(
            pd.DataFrame(
                {
                    "binding_id": ["fixture_effect"],
                    "condition_id": ["ifng"],
                    "effect_id": ["IFNG"],
                    "parameterization_kind": ["condition_specific"],
                }
            )
        ),
        reference_bindings=ReferenceBindingTable(
            pd.DataFrame(
                {
                    "binding_id": ["fixture_reference"],
                    "condition_id": ["ifng"],
                    "reference_pool_id": ["media_control"],
                    "scope_kind": ["global_condition_pool"],
                }
            )
        ),
    )


def _compile_compact(recipe, view, config=None, requested=None):
    if config is None or not hasattr(config, "validation"):
        config = recipe.config_schema().model_validate(config or {})
    plan = recipe.plan_split(view, config, requested or SplitSpec(strategy="none"))
    validate_split_plan(view, plan)
    return recipe.compile(view, plan, config)


def test_five_file_codec_makes_series_and_observations_explicit(tiny_config) -> None:
    assert available_study_codecs() == (
        "credo.current_five_file",
        "credo.native_perturb_seq_study",
        "credo.native_study",
    )
    study = open_study(tiny_config)
    try:
        assert isinstance(study, PerturbSeqStudy)
        assert study.manifest.source_schema == "five_file_v2"
        assert study.design.topology == "chain"
        assert study.design.checkpoint_ids == ("Rest", "Stim8hr", "Stim48hr")
        assert len(study.perturbations) == 6
        assert len(study.intervention_events) == 12
        assert len(study.series) == 12
        assert len(study.observations) == 36
        assert len(study.abundance) == 35
        observations = study.observations.to_pandas()
        missing = observations.loc[~observations["geometry_observed"]]
        assert missing["observation_id"].tolist() == ["D2::GENE2-2@Stim48hr"]
        assert missing["context_id"].tolist() == ["D2"]
        assert missing["composition_block_id"].tolist() == ["D2@Stim48hr"]
        assert study.compositions is not None
        composition = study.compositions.to_pandas()
        assert "measure_indices" not in composition
        assert "series_id" in composition
        assert "observation_id" in composition
        assert "D2::GENE2-2@Stim48hr" in set(composition["observation_id"])
    finally:
        study.close()


def test_five_file_conversion_does_not_materialize_support_for_abundance(
    tiny_config,
) -> None:
    data = load_data(tiny_config)
    assert len(data.measures._cache) == 0
    study = CurrentFiveFileStudyCodec().from_trajectory(data)
    try:
        assert len(data.measures._cache) == 0
        snapshot = study.snapshot("D1::GENE1-1@Rest")
        assert snapshot.law is not None
        assert snapshot.law.coordinates.shape == (8, 4)
        assert snapshot.law.probabilities.sum() == pytest.approx(1.0)
        assert snapshot.abundance is not None
        assert snapshot.abundance.value > 0
        assert len(data.measures._cache) == 1
    finally:
        study.close()


def test_study_supports_guide_free_conditions_zero_abundance_and_missing_geometry() -> None:
    study = _general_study()
    try:
        assert "guide_id" not in study.conditions.columns
        target = study.snapshot("donor-a::ifng@stim")
        assert target.law is not None
        assert target.abundance is not None
        assert target.abundance.value == 0.0
        assert target.abundance.observed is True
        source = study.snapshot("donor-a::ifng@rest", representation_id="latent-strict")
        assert source.law is not None
        target_strict = study.snapshot("donor-a::ifng@stim", representation_id="latent-strict")
        assert target_strict.law is None
    finally:
        study.close()


def test_abundance_transform_records_input_and_parameters() -> None:
    raw = AbundanceChannelSpec(
        channel_id="raw_count",
        semantics=AbundanceSemantics.ABSOLUTE,
        unit="cells",
        zero_policy="allowed",
    )
    modeled = AbundanceChannelSpec(
        channel_id="modeled_frequency",
        semantics=AbundanceSemantics.RELATIVE,
        denominator_scope="context_checkpoint",
        zero_policy="forbidden",
        transform_id="jeffreys_alpha_0.5",
        input_channel_id="raw_count",
        transform_parameters={"alpha": 0.5},
    )
    table = AbundanceTable(
        pd.DataFrame(
            {
                "observation_id": ["series@rest", "series@rest"],
                "channel_id": ["raw_count", "modeled_frequency"],
                "value": [0.0, 1.0],
                "observed": [True, True],
                "denominator_id": [None, "rest-library"],
                "transform_id": [None, "jeffreys_alpha_0.5"],
            }
        ),
        (raw, modeled),
    )
    assert table.channels["modeled_frequency"].input_channel_id == "raw_count"
    assert table.channels["modeled_frequency"].transform_parameters == {"alpha": 0.5}
    with pytest.raises(TypeError):
        table.channels["modeled_frequency"].transform_parameters["alpha"] = 1.0


def test_multiple_representations_and_views_share_one_support_store() -> None:
    study = _general_study()
    try:
        assert study.representations.representation_ids == ("latent-all", "latent-strict")
        view = study.view(
            SelectionSpec(
                checkpoint_ids=("rest",),
                condition_filter={"condition_kind": "cytokine_stimulation"},
                observation_filter={"context_id": "well-1"},
            ),
            representation_id="latent-strict",
        )
        assert view.study.supports is study.supports
        assert view.series_ids == ("donor-a::ifng",)
        assert view.observation_ids == ("donor-a::ifng@rest",)
        assert view.representation.fit_split_id == "strict-stim-holdout"
    finally:
        study.close()


def test_representations_can_use_different_support_stores() -> None:
    study = _general_study()
    law = study.snapshot("donor-a::ifng@rest").law
    assert law is not None
    all_store = InMemorySupportStore(
        "all-store",
        {
            SupportRef("all-store", "latent-all", "source-law"): law,
            SupportRef("all-store", "latent-all", "target-law"): law,
        },
    )
    strict_store = InMemorySupportStore(
        "strict-store",
        {SupportRef("strict-store", "latent-strict", "source-law"): law},
    )
    representations = RepresentationCatalog(
        (
            replace(study.representations["latent-all"], support_store_id="all-store"),
            replace(study.representations["latent-strict"], support_store_id="strict-store"),
        )
    )
    index = study.support_index.to_pandas()
    index.loc[index["representation_id"].eq("latent-all") & index["available"], "store_id"] = (
        "all-store"
    )
    index.loc[index["representation_id"].eq("latent-strict") & index["available"], "store_id"] = (
        "strict-store"
    )
    multi_store = replace(
        study,
        representations=representations,
        support_index=SupportIndexTable(index),
        supports=SupportStoreRegistry((all_store, strict_store)),
    )
    try:
        assert multi_store.validate().valid
        assert multi_store.snapshot("donor-a::ifng@stim").law is not None
        assert (
            multi_store.snapshot("donor-a::ifng@stim", representation_id="latent-strict").law
            is None
        )
    finally:
        multi_store.close()
        study.close()


def test_five_file_codec_preserves_representation_fit_scope(tiny_config) -> None:
    data = load_data(tiny_config)
    scoped = replace(
        data,
        representation=replace(
            data.representation,
            fit_scope="training_split",
            gene_names_hash="a" * 64,
            gene_mask_hash="b" * 64,
            included_samples=("D1",),
            included_time_labels=("Rest",),
        ),
    )
    study = CurrentFiveFileStudyCodec().from_trajectory(scoped)
    try:
        representation = study.representations[study.manifest.primary_representation]
        assert representation.fit_split_id is not None
        assert representation.included_series == tuple(
            data.measure_meta.loc[data.measure_meta["sample_id"].eq("D1"), "measure_id"].tolist()
        )
        assert representation.included_checkpoints == ("Rest",)
        assert representation.feature_artifact is not None
        assert representation.feature_artifact.sha256 == "a" * 64
        assert representation.feature_artifact.semantic_hash == "b" * 64
        assert study.snapshot("D1::GENE1-1@Stim8hr").law is not None
    finally:
        study.close()


def test_study_rejects_foreign_key_and_representation_dimension_errors() -> None:
    study = _general_study()
    bad_series = study.series.to_pandas()
    bad_series.loc[0, "condition_id"] = "missing"
    malformed = replace(study, series=SeriesTable(bad_series))
    with pytest.raises(ValueError, match="series.condition_fk"):
        malformed.validate().raise_for_errors()

    bad_representation = RepresentationCatalog(
        (RepresentationSpec("latent-all", "fixture", "latent", 3, "memory"),)
    )
    bad_manifest = replace(study.manifest, primary_representation="latent-all")
    malformed = replace(
        study,
        manifest=bad_manifest,
        representations=bad_representation,
    )
    with pytest.raises(ValueError, match="representation.dimension"):
        malformed.validate().raise_for_errors()

    bad_composition = study.compositions.to_pandas()
    bad_composition.loc[0, "context_id"] = "wrong-well"
    malformed = replace(study, compositions=CompositionTable(bad_composition))
    with pytest.raises(ValueError, match="composition.context_alignment"):
        malformed.validate().raise_for_errors()
    study.close()


def test_support_index_rejects_duplicate_qualified_support_keys() -> None:
    support_index = pd.DataFrame(
        {
            "observation_id": ["series-a@rest", "series-b@rest"],
            "representation_id": ["latent", "latent"],
            "store_id": ["memory", "memory"],
            "support_key": ["shared", "shared"],
            "available": [True, True],
        }
    )
    with pytest.raises(ValueError, match="duplicate qualified support key"):
        SupportIndexTable(support_index)


def test_public_table_access_is_immutable() -> None:
    study = _general_study()
    try:
        conditions = study.conditions.to_pandas()
        conditions.loc[0, "condition_id"] = "corrupted"
        assert study.conditions.to_pandas().loc[0, "condition_id"] == "ifng"
        with pytest.raises(TypeError):
            study.conditions.to_pandas(copy=False)
        with pytest.raises(FrozenInstanceError):
            study.manifest = replace(study.manifest, study_id="changed")
    finally:
        study.close()


def test_observations_allow_replicates_with_distinct_ids() -> None:
    observations = ObservationTable(
        pd.DataFrame(
            {
                "observation_id": ["series-a@rest:r1", "series-a@rest:r2"],
                "series_id": ["series-a", "series-a"],
                "checkpoint_id": ["rest", "rest"],
                "sample_id": ["sample-a", "sample-a"],
                "replicate_id": ["r1", "r2"],
                "geometry_observed": [True, True],
            }
        )
    )
    assert len(observations) == 2


def _copy_five_file_config(tiny_config, destination: Path):
    destination.mkdir()
    updates = {}
    for field in ("support", "measure_meta", "masses", "counts", "dataset"):
        source = getattr(tiny_config.data, field)
        if source is None:
            continue
        target = destination / source.name
        shutil.copy2(source, target)
        updates[field] = target
    return tiny_config.model_copy(update={"data": tiny_config.data.model_copy(update=updates)})


def test_verification_levels_defer_semantics_and_support_scans(tiny_config, tmp_path) -> None:
    config = _copy_five_file_config(tiny_config, tmp_path / "copied")
    manifest = json.loads(config.data.dataset.read_text(encoding="utf-8"))
    manifest["representation"]["latent_cache_hash"] = hashlib.sha256(
        config.data.support.read_bytes()
    ).hexdigest()
    manifest["representation"]["producer"]["latent_cache_hash_kind"] = "support_h5ad_file_sha256"
    config.data.dataset.write_text(json.dumps(manifest), encoding="utf-8")
    with h5py.File(config.data.support, "r+") as handle:
        handle[f"obsm/{config.data.latent_key}"][0, 0] = np.nan

    schema_only = open_study(config, verify="schema")
    schema_only.close()
    with pytest.raises(ValueError, match="latent-cache hash disagrees"):
        open_study(config, verify="manifest")

    digest = hashlib.sha256(config.data.support.read_bytes()).hexdigest()
    manifest["representation"]["latent_cache_hash"] = digest
    manifest["representation"]["producer"]["latent_cache_hash_kind"] = "support_h5ad_file_sha256"
    config.data.dataset.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="contains non-finite values"):
        open_study(config, verify="full")


def test_open_study_preserves_configured_support_loading_policy(tiny_config) -> None:
    eager_config = tiny_config.model_copy(
        update={
            "data": tiny_config.data.model_copy(
                update={"lazy_support": False, "support_cache_size": 17}
            )
        }
    )
    eager = open_study(eager_config, verify="semantic")
    try:
        representation = eager.representations[eager.manifest.primary_representation]
        store = eager.supports[representation.support_store_id]
        assert not bool(getattr(store._measures, "is_lazy", False))
    finally:
        eager.close()

    lazy = open_study(eager_config, verify="semantic", lazy_support=True)
    try:
        representation = lazy.representations[lazy.manifest.primary_representation]
        store = lazy.supports[representation.support_store_id]
        assert bool(getattr(store._measures, "is_lazy", False))
        assert store._measures.cache_size == 17
        assert store._measures._atom_weight is None
        assert len(store._measures._packed_positions) > 0
        assert store._measures._packed_indptr[-1] == len(store._measures._packed_positions)
    finally:
        lazy.close()


def test_verify_none_constructs_a_repairable_semantically_invalid_study(
    tiny_config, tmp_path
) -> None:
    counts = pd.read_parquet(tiny_config.data.counts)
    counts.loc[0, "measure_id"] = "unknown-series"
    path = tmp_path / "broken-counts.parquet"
    counts.to_parquet(path, index=False)
    config = tiny_config.model_copy(
        update={"data": tiny_config.data.model_copy(update={"counts": path})}
    )
    study = open_study(config, verify="none")
    try:
        report = study.validate("semantic")
        assert not report.valid
        assert any(issue.code == "compositions.observation_fk" for issue in report.errors)
        assert report.errors[0].location
    finally:
        study.close()
    with pytest.raises(ValueError, match="unknown measures"):
        open_study(config, verify="semantic")


def test_design_validates_reachability_stars_and_ordered_direction() -> None:
    axis = (AxisSpec("time", "physical_time", "hour"),)
    checkpoints = (
        Checkpoint("rest", {"time": 0.0}, "source"),
        Checkpoint("early", {"time": 2.0}, "target"),
        Checkpoint("late", {"time": 8.0}, "target"),
    )
    star = StudyDesign(
        axes=axis,
        checkpoints=checkpoints,
        transitions=(
            Transition("rest_to_early", "rest", "early"),
            Transition("rest_to_late", "rest", "late"),
        ),
        topology="star",
    )
    assert star.source_checkpoint_id == "rest"

    with pytest.raises(ValueError, match="moves backward"):
        StudyDesign(
            axes=axis,
            checkpoints=(
                Checkpoint("rest", {"time": 0.0}, "source"),
                Checkpoint("late", {"time": 8.0}, "intermediate"),
                Checkpoint("early", {"time": 2.0}, "target"),
            ),
            transitions=(
                Transition("rest_to_late", "rest", "late"),
                Transition("late_to_early", "late", "early"),
            ),
            topology="chain",
        )

    with pytest.raises(ValueError, match="reachable"):
        StudyDesign(
            axes=axis,
            checkpoints=checkpoints,
            transitions=(Transition("rest_to_early", "rest", "early"),),
            topology="dag",
        )


def test_compact_recipe_compiles_from_bindings_without_legacy_parameter_columns(
    tiny_config,
) -> None:
    study = open_study(tiny_config)
    try:
        perturbation_frame = study.perturbations.to_pandas().drop(
            columns=["guide_id", "target_gene", "embedding_id", "reference_group_id"],
            errors="ignore",
        )
        series_frame = study.series.to_pandas().drop(
            columns=["guide_id", "target_gene", "embedding_id", "reference_role"],
            errors="ignore",
        )
        guide_free = replace(
            study,
            perturbations=PerturbationTable(perturbation_frame),
            series=PopulationSeriesTable(series_frame),
        )
        view = guide_free.view()
        recipe = get_recipe(tiny_config.recipe)
        compiled = _compile_compact(recipe, view, tiny_config.recipe_config)
        assert "guide_id" in compiled.measure_meta
        assert "target_gene" in compiled.measure_meta
        assert set(compiled.training.measure_ids) | set(compiled.validation.measure_ids) == set(
            guide_free.series.series_ids
        )
        assert set(compiled.embedding_ids) == set(view.effect_binding()["effect_id"])
        assert compiled.control_embedding_ids
    finally:
        study.close()


def test_compact_compiler_requires_an_explicit_positive_modeling_channel() -> None:
    study = _general_study()
    try:
        recipe = get_recipe("credo.compact_sde_v3@3.0")
        with pytest.raises(ValueError, match="positive modeling abundance"):
            _compile_compact(recipe, study.view(), {})
    finally:
        study.close()


def test_empirical_law_owns_immutable_normalized_arrays() -> None:
    coordinates = np.asarray([[1.0], [2.0]])
    probabilities = np.asarray([0.4, 0.6])
    law = EmpiricalLaw(coordinates, probabilities)
    coordinates[0, 0] = 99.0
    probabilities[0] = 0.0
    assert law.coordinates[0, 0] == 1.0
    assert law.probabilities.tolist() == pytest.approx([0.4, 0.6])
    with pytest.raises(ValueError):
        law.probabilities[0] = 0.2


def test_explicit_no_abundance_uses_unit_mass() -> None:
    study = _general_study()
    try:
        assert study.view().abundance_channel == "raw_cell_count"
        assert study.view(abundance_channel=None).abundance_channel is None
        recipe = get_recipe("credo.compact_sde_v3@3.0")
        report = study.view().validate_for(
            recipe.requirements(recipe.config_schema()()), SplitSpec(strategy="none")
        )
        assert not any(issue.code == "recipe.context_scope" for issue in report.errors)

        observations = study.observations.to_pandas()
        observations["context_id"] = "well-static"
        static = replace(study, observations=ObservationTable(observations), compositions=None)
        compiled = _compile_compact(
            recipe,
            static.view(
                SelectionSpec(composition_policy="drop"),
                abundance_channel=None,
            ),
            recipe.config_schema()(),
        )
        assert compiled.mass_semantics.value == "unit"
        assert all(
            measure.total_mass == 1.0
            for checkpoint in compiled.measures.values()
            for measure in checkpoint.values()
        )
    finally:
        study.close()


def test_study_content_hash_tracks_values_and_run_level_effect_bindings(tiny_config) -> None:
    study = open_study(tiny_config)
    try:
        original_hash = study.content_hash()
        reordered_observations = (
            study.observations.to_pandas().sample(frac=1.0, random_state=7).reset_index(drop=True)
        )
        reordered = replace(
            study,
            observations=type(study.observations)(reordered_observations),
        )
        assert reordered.content_hash() == original_hash

        abundance = study.abundance.to_pandas()
        abundance.loc[0, "value"] = float(abundance.loc[0, "value"]) + 1.0
        changed = replace(
            study,
            abundance=AbundanceTable(abundance, study.abundance.channels),
        )
        assert changed.content_hash() != original_hash

        perturbations = study.perturbations.to_pandas()
        target_by_perturbation = (
            study.perturbation_components.to_pandas()
            .drop_duplicates("perturbation_id")
            .set_index("perturbation_id")["target_id"]
            .to_dict()
        )
        guide_effect = pd.DataFrame(
            {
                "binding_id": "condition_specific",
                "perturbation_id": perturbations["perturbation_id"],
                "effect_id": perturbations["perturbation_id"],
                "parameterization_kind": "condition_specific",
            }
        )
        target_effect = pd.DataFrame(
            {
                "binding_id": "target_gene_shared",
                "perturbation_id": perturbations["perturbation_id"],
                "effect_id": perturbations["perturbation_id"].map(target_by_perturbation),
                "parameterization_kind": "target_gene_shared",
            }
        )
        bindings = PerturbationEffectBindingTable(
            pd.concat(
                (study.effect_bindings.to_pandas(), guide_effect, target_effect),
                ignore_index=True,
            )
        )
        rebound = replace(study, effect_bindings=bindings)
        recipe = get_recipe(tiny_config.recipe)
        guide_level = _compile_compact(
            recipe,
            rebound.view(effect_binding_id="condition_specific"),
            tiny_config.recipe_config,
        )
        target_level = _compile_compact(
            recipe,
            rebound.view(effect_binding_id="target_gene_shared"),
            tiny_config.recipe_config,
        )
        assert len(target_level.embedding_ids) < len(guide_level.embedding_ids)
        assert set(target_level.control_embedding_ids) <= set(target_level.embedding_ids)
    finally:
        study.close()


def test_compact_recipe_rejects_multiple_selected_reference_pools(tiny_config) -> None:
    study = open_study(tiny_config)
    try:
        perturbations = study.perturbations.to_pandas()
        references = perturbations.loc[perturbations["is_control"], "perturbation_id"].tolist()
        assert len(references) >= 2
        primary_effects = study.effect_bindings.to_pandas()
        primary_effects = primary_effects.loc[
            primary_effects["binding_id"].eq(study.manifest.primary_effect_binding)
        ].set_index("perturbation_id")["effect_id"]
        alternate = pd.DataFrame(
            {
                "binding_id": "two_reference_pools",
                "perturbation_id": perturbations["perturbation_id"],
                "reference_pool_id": [
                    "pool-a" if index % 2 == 0 else "pool-b" for index in range(len(perturbations))
                ],
                "scope_kind": "global",
                "match_keys": "[]",
            }
        )
        alternate.loc[alternate["perturbation_id"].eq(references[0]), "reference_pool_id"] = (
            "pool-a"
        )
        alternate.loc[alternate["perturbation_id"].eq(references[1]), "reference_pool_id"] = (
            "pool-b"
        )
        reference_effects = {
            "pool-a": primary_effects.loc[references[0]],
            "pool-b": primary_effects.loc[references[1]],
        }
        alternate["counterfactual_effect_id"] = alternate["reference_pool_id"].map(
            reference_effects
        )
        rebound = replace(
            study,
            reference_bindings=PerturbationReferenceBindingTable(
                pd.concat(
                    (study.reference_bindings.to_pandas(), alternate),
                    ignore_index=True,
                )
            ),
        )
        recipe = get_recipe(tiny_config.recipe)
        view = rebound.view(reference_binding_id="two_reference_pools")
        report = view.validate_for(
            recipe.requirements(tiny_config.recipe_config),
            SplitSpec(strategy="none"),
        )
        assert any(issue.code == "recipe.reference_multiplicity" for issue in report.errors)
        with pytest.raises(ValueError, match="multiple reference pools"):
            _compile_compact(recipe, view, tiny_config.recipe_config)
    finally:
        study.close()


def test_study_content_hash_ignores_artifact_location() -> None:
    study = _general_study()
    try:
        artifact = ArtifactRef(
            artifact_id="latent-support",
            uri="/first/location/support.h5",
            sha256="a" * 64,
            size_bytes=123,
            media_type="application/x-hdf5",
        )
        first_spec = replace(study.representations["latent-all"], support_artifact=artifact)
        second_spec = replace(
            first_spec,
            support_artifact=replace(artifact, uri="/relocated/support.h5"),
        )
        strict = study.representations["latent-strict"]
        first = replace(
            study,
            representations=RepresentationCatalog((first_spec, strict)),
        )
        second = replace(
            study,
            representations=RepresentationCatalog((second_spec, strict)),
        )
        assert first.content_hash() == second.content_hash()
    finally:
        study.close()


def test_semantic_split_is_content_bound_and_rejects_representation_leakage(
    tiny_config,
) -> None:
    study = open_study(tiny_config)
    try:
        recipe = get_recipe(tiny_config.recipe)
        shared = recipe.plan_split(study.view(), tiny_config.recipe_config)
        validate_split_plan(study.view(), shared)
        assert shared.representation_evaluation == "transductive"
        with pytest.raises(ValueError, match="content-bound"):
            validate_split_plan(study.view(), replace(shared, split_id=f"sha256:{'0' * 64}"))

        settings = tiny_config.recipe_config
        validation = settings.validation.model_copy(
            update={
                "strategy": "context_group",
                "values": ("D1",),
                "fraction": 0.0,
                "representation_scope": "nested",
            }
        )
        nested_config = settings.model_copy(update={"validation": validation})
        primary = study.manifest.primary_representation
        placeholder = replace(
            study.representations[primary],
            scope_mode="nested_by_subject",
            fit_split_id="pending",
            fit_subject_ids=tuple(study.series.to_pandas()["subject_id"].unique()),
            fit_checkpoint_ids=study.design.checkpoint_ids,
            included_series=study.series.series_ids,
        )
        placeholder_study = replace(
            study,
            representations=RepresentationCatalog((placeholder,)),
        )
        plan = recipe.plan_split(placeholder_study.view(), nested_config)
        contaminated = replace(placeholder, fit_split_id=plan.split_id)
        nested_study = replace(
            study,
            representations=RepresentationCatalog((contaminated,)),
        )
        view = nested_study.view()
        assert recipe.plan_split(view, nested_config).split_id == plan.split_id
        validate_split_plan(view, plan)
        assert plan.representation_evaluation == "inductive"
        with pytest.raises(ValueError, match="held-out identities"):
            validate_representation_scope(view, plan)

        clean = replace(
            contaminated,
            fit_subject_ids=tuple(
                value
                for value in contaminated.fit_subject_ids
                if value not in plan.held_out_subject_ids
            ),
            included_series=tuple(
                value for value in study.series.series_ids if value not in plan.held_out_series
            ),
        )
        clean_study = replace(
            nested_study,
            representations=RepresentationCatalog((clean,)),
        )
        validate_representation_scope(clean_study.view(), plan)

        checkpoint_validation = settings.validation.model_copy(
            update={
                "strategy": "checkpoint",
                "values": ("Stim48hr",),
                "fraction": 0.0,
                "representation_scope": "nested",
            }
        )
        checkpoint_config = settings.model_copy(update={"validation": checkpoint_validation})
        checkpoint_placeholder = replace(
            study.representations[primary],
            scope_mode="nested_by_checkpoint",
            fit_split_id="pending",
            fit_checkpoint_ids=study.design.checkpoint_ids,
            included_checkpoints=study.design.checkpoint_ids,
        )
        checkpoint_placeholder_study = replace(
            study,
            representations=RepresentationCatalog((checkpoint_placeholder,)),
        )
        checkpoint_plan = recipe.plan_split(checkpoint_placeholder_study.view(), checkpoint_config)
        checkpoint_contaminated = replace(
            checkpoint_placeholder,
            fit_split_id=checkpoint_plan.split_id,
        )
        checkpoint_study = replace(
            study,
            representations=RepresentationCatalog((checkpoint_contaminated,)),
        )
        checkpoint_view = checkpoint_study.view()
        assert (
            recipe.plan_split(checkpoint_view, checkpoint_config).split_id
            == checkpoint_plan.split_id
        )
        validate_split_plan(checkpoint_view, checkpoint_plan)
        assert checkpoint_plan.held_out_checkpoints == ("Stim48hr",)
        with pytest.raises(ValueError, match="held-out representation checkpoints"):
            validate_representation_scope(checkpoint_view, checkpoint_plan)
    finally:
        study.close()


def test_composition_selection_mints_denominators_and_preserves_background(tiny_config) -> None:
    study = open_study(tiny_config)
    try:
        compositions = study.compositions.to_pandas()
        positive = compositions.groupby("series_id", observed=True)["count"].min()
        series_id = str(positive[positive.gt(0)].index[0])
        recipe = get_recipe(tiny_config.recipe)
        conditioned_view = study.view(
            SelectionSpec(
                series_ids=(series_id,),
                composition_policy="condition_on_selection",
            )
        )
        conditioned = _compile_compact(
            recipe,
            conditioned_view,
            tiny_config.recipe_config,
        )
        assert conditioned.count_blocks
        compiled_selection_hash = conditioned.training.metadata["selection_hash"]
        assert all(
            block.modeled_denominator_id == f"{block.source_denominator_id}|conditioned:"
            f"{compiled_selection_hash[:16]}"
            for block in conditioned.count_blocks
        )

        preserved = _compile_compact(
            recipe,
            study.view(
                SelectionSpec(
                    series_ids=(series_id,),
                    composition_policy="preserve_background",
                )
            ),
            tiny_config.recipe_config,
        )
        assert all(block.background_series_ids for block in preserved.count_blocks)
        assert all(
            len(block.background_series_ids) == len(block.background_fitness)
            for block in preserved.count_blocks
        )
    finally:
        study.close()


def test_compact_replicate_pooling_concatenates_geometry_and_sums_abundance() -> None:
    original = _general_study()
    observations = original.observations.to_pandas()
    observations["context_id"] = "well-static"
    observations["replicate_id"] = "r1"
    duplicate = observations.copy()
    duplicate["observation_id"] = duplicate["observation_id"] + "::r2"
    duplicate["replicate_id"] = "r2"
    all_observations = ObservationTable(pd.concat((observations, duplicate), ignore_index=True))

    support_rows = original.support_index.to_pandas()
    laws = {}
    for row in support_rows.loc[support_rows["available"]].itertuples(index=False):
        ref = SupportRef(str(row.store_id), str(row.representation_id), str(row.support_key))
        laws[ref] = original.supports.read(ref)
    duplicate_support = []
    for row in support_rows.itertuples(index=False):
        support_key = None if not row.available else f"{row.support_key}::r2"
        duplicate_support.append(
            {
                "observation_id": f"{row.observation_id}::r2",
                "representation_id": row.representation_id,
                "store_id": row.store_id,
                "support_key": support_key,
                "available": row.available,
            }
        )
        if row.available:
            source = SupportRef(str(row.store_id), str(row.representation_id), str(row.support_key))
            target = SupportRef(str(row.store_id), str(row.representation_id), str(support_key))
            laws[target] = laws[source]
    support_index = SupportIndexTable(
        pd.concat((support_rows, pd.DataFrame(duplicate_support)), ignore_index=True)
    )
    abundance = original.abundance.to_pandas()
    abundance.loc[abundance["value"].eq(0), "value"] = 10.0
    duplicate_abundance = abundance.copy()
    duplicate_abundance["observation_id"] += "::r2"
    duplicate_abundance["value"] = [5.0, 3.0]
    pooled_study = replace(
        original,
        observations=all_observations,
        support_index=support_index,
        abundance=AbundanceTable(
            pd.concat((abundance, duplicate_abundance), ignore_index=True),
            original.abundance.channels,
        ),
        compositions=None,
        supports=InMemorySupportStore("memory", laws),
    )
    try:
        policy = ReplicatePolicy(
            mode="pool",
            geometry_pooling="concatenate",
            abundance_pooling="sum",
        )
        recipe = get_recipe("credo.compact_sde_v3@3.0")
        compiled = _compile_compact(
            recipe,
            pooled_study.view(SelectionSpec(replicate_policy=policy)),
            recipe.config_schema()(),
        )
        source = compiled.measures[compiled.axis.source][compiled.measure_ids[0]]
        assert source.total_mass == pytest.approx(25.0)
        assert len(source.support) == 4
        assert next(iter(compiled.metadata["replicate_transform"]["pooled_observations"]))

        selected = _compile_compact(
            recipe,
            pooled_study.view(
                SelectionSpec(replicate_policy=ReplicatePolicy(mode="select", selection_key="r2"))
            ),
            recipe.config_schema()(),
        )
        selected_source = selected.measures[selected.axis.source][selected.measure_ids[0]]
        assert selected_source.total_mass == pytest.approx(5.0)
        assert len(selected_source.support) == 2
    finally:
        pooled_study.close()
        original.close()


def test_native_study_roundtrip_and_manifest_corruption_detection(
    tiny_config,
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(native_v4, "_PACKED_WRITE_ATOMS", 2)
    monkeypatch.setattr(native_v4, "_PACKED_WRITE_LAWS", 2)
    opened = open_study(tiny_config)
    feature_path = tmp_path / "feature-catalog.txt"
    feature_path.write_text("GENE1\nGENE2\n", encoding="utf-8")
    feature_artifact = ArtifactRef(
        artifact_id="feature-catalog",
        uri=str(feature_path),
        sha256=hashlib.sha256(feature_path.read_bytes()).hexdigest(),
        size_bytes=feature_path.stat().st_size,
        media_type="text/plain",
    )
    primary = opened.manifest.primary_representation
    source = replace(
        opened,
        representations=RepresentationCatalog(
            (replace(opened.representations[primary], feature_artifact=feature_artifact),)
        ),
    )
    native = tmp_path / "native-study"
    try:
        expected_hash = source.content_hash()
        manifest = write_study(source, native)
    finally:
        opened.close()
    feature_path.unlink()
    reopened = open_study(manifest, verify="full")
    try:
        assert reopened.content_hash() == expected_hash
        assert reopened.effect_bindings is not None
        assert reopened.reference_bindings is not None
        embedded_feature = reopened.representations[primary].feature_artifact
        assert embedded_feature is not None
        assert Path(embedded_feature.uri).is_file()
        assert Path(embedded_feature.uri).is_relative_to(native)
    finally:
        reopened.close()
    existing = open_study(manifest)
    try:
        with pytest.raises(FileExistsError, match="already exists"):
            write_study(existing, native)
    finally:
        existing.close()

    damaged = tmp_path / "native-damaged-support"
    shutil.copytree(native, damaged)
    damaged_manifest = damaged / "study.json"
    payload = json.loads(damaged_manifest.read_text(encoding="utf-8"))
    support_relative = next(value for value in payload["artifacts"] if value.endswith(".h5"))
    support_path = damaged / support_relative
    with h5py.File(support_path, "r+") as handle:
        coordinates = next(iter(handle["representations"].values()))["coordinates"]
        coordinates[0, 0] = float(coordinates[0, 0]) + 1.0
    payload["artifacts"][support_relative] = {
        "sha256": hashlib.sha256(support_path.read_bytes()).hexdigest(),
        "size_bytes": support_path.stat().st_size,
    }
    damaged_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="support.semantic_hash"):
        open_study(damaged_manifest, verify="full")

    perturbations = native / "perturbations.parquet"
    perturbations.write_bytes(perturbations.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="artifact size mismatch"):
        open_study(manifest, verify="manifest")


def test_failed_native_write_removes_its_transaction_directory(tiny_config, tmp_path) -> None:
    study = open_study(tiny_config)
    artifact_path = tmp_path / "mismatched.txt"
    artifact_path.write_text("content", encoding="utf-8")
    primary = study.manifest.primary_representation
    malformed = replace(
        study,
        representations=RepresentationCatalog(
            (
                replace(
                    study.representations[primary],
                    feature_artifact=ArtifactRef(
                        artifact_id="mismatched",
                        uri=str(artifact_path),
                        sha256="0" * 64,
                        size_bytes=artifact_path.stat().st_size,
                        media_type="text/plain",
                    ),
                ),
            )
        ),
    )
    target = tmp_path / "failed-native-study"
    try:
        with pytest.raises(ValueError, match="hash mismatch"):
            write_study(malformed, target)
        assert not target.exists()
        assert not list(tmp_path.glob(f".{target.name}.tmp-*"))
    finally:
        study.close()


def test_native_study_run_config_is_the_single_input_contract(tiny_config, tmp_path) -> None:
    source = open_study(tiny_config)
    try:
        manifest = write_study(source, tmp_path / "native-config-study")
    finally:
        source.close()
    raw = tiny_config.model_dump(mode="json")
    raw.pop("data")
    raw.pop("axis")
    raw["study"] = str(manifest)
    raw["output"] = str(tmp_path / "native-run")
    config = RunConfig.model_validate(raw)
    summary = validate_inputs(config)
    assert summary["measure_count"] == 12
    assert summary["axis_labels"] == ["Rest", "Stim8hr", "Stim48hr"]
