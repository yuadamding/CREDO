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

from credo import SplitSpec, Study, open_study
from credo.data import (
    AbundanceChannelSpec,
    AbundanceSemantics,
    AbundanceTable,
    AxisSpec,
    Checkpoint,
    CompositionTable,
    ConditionTable,
    CurrentFiveFileStudyCodec,
    EmpiricalLaw,
    InMemorySupportStore,
    ObservationTable,
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
from credo.io import load_data
from credo.registry import get_recipe


def _general_study() -> Study:
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
    return Study(
        manifest=StudyManifest(
            schema_version=3,
            study_id="general-fixture",
            source_schema="native_test",
            primary_representation="latent-all",
            primary_abundance_channel="raw_cell_count",
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
    )


def test_five_file_codec_makes_series_and_observations_explicit(tiny_config) -> None:
    assert available_study_codecs() == ("credo.current_five_file",)
    study = open_study(tiny_config)
    try:
        assert isinstance(study, Study)
        assert study.manifest.source_schema == "five_file_v2"
        assert study.design.topology == "chain"
        assert study.design.checkpoint_ids == ("Rest", "Stim8hr", "Stim48hr")
        assert len(study.conditions) == 6
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
            gene_names_hash="a" * 64,
            gene_mask_hash="b" * 64,
            included_samples=("D1",),
            included_time_labels=("Rest",),
        ),
    )
    study = CurrentFiveFileStudyCodec().from_trajectory(scoped)
    try:
        representation = study.representations[study.manifest.primary_representation]
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
        assert any(issue.code == "composition.foreign_key" for issue in report.errors)
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


def test_compact_recipe_compiles_a_guide_free_study_view(tiny_config) -> None:
    study = open_study(tiny_config)
    try:
        condition_frame = study.conditions.to_pandas().drop(
            columns=["guide_id", "target_gene"], errors="ignore"
        )
        series_frame = study.series.to_pandas().drop(
            columns=["guide_id", "target_gene"], errors="ignore"
        )
        guide_free = replace(
            study,
            conditions=ConditionTable(condition_frame),
            series=SeriesTable(series_frame),
        )
        view = guide_free.view()
        recipe = get_recipe(tiny_config.recipe)
        split = SplitSpec(strategy="none")
        view.validate_for(recipe.requirements(tiny_config.recipe_config), split).raise_for_errors()
        compiled = recipe.compile_study(view, split, tiny_config.recipe_config)
        assert "guide_id" not in compiled.measure_meta
        assert "target_gene" not in compiled.measure_meta
        assert compiled.measure_ids == guide_free.series.series_ids
    finally:
        study.close()


def test_compact_compiler_requires_an_explicit_positive_modeling_channel() -> None:
    study = _general_study()
    try:
        recipe = get_recipe("credo.compact_sde_v3@3.0")
        with pytest.raises(ValueError, match="positive modeling abundance"):
            recipe.compile_study(study.view(), SplitSpec(strategy="none"), {})
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
