from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from credo import Study, open_study
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
    SupportRef,
    Transition,
)
from credo.io import load_data


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
                "geometry_observed": [True, False],
                "context_id": ["well-1", "well-2"],
                "composition_block_id": [None, "well-2@stim"],
                "support_key": ["source-law", None],
            }
        )
    )
    channel = AbundanceChannelSpec(
        channel_id="raw_cell_count",
        semantics=AbundanceSemantics.ABSOLUTE,
        unit="cells",
        denominator_required=False,
        permits_absolute_claim=True,
        permits_relative_claim=True,
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
            SupportRef("latent-all", "source-law"): law,
            SupportRef("latent-strict", "source-law"): law,
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
        abundance=abundance,
        compositions=compositions,
        representations=representations,
        supports=store,
    )


def test_five_file_codec_makes_series_and_observations_explicit(tiny_config) -> None:
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
        assert target.law is None
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
    with pytest.raises(ValueError, match="series.condition_fk"):
        replace(study, series=SeriesTable(bad_series))

    bad_representation = RepresentationCatalog(
        (RepresentationSpec("latent-all", "fixture", "latent", 3, "memory"),)
    )
    bad_manifest = replace(study.manifest, primary_representation="latent-all")
    with pytest.raises(ValueError, match="representation.dimension"):
        replace(
            study,
            manifest=bad_manifest,
            representations=bad_representation,
        )

    bad_composition = study.compositions.to_pandas()
    bad_composition.loc[0, "context_id"] = "wrong-well"
    with pytest.raises(ValueError, match="composition.context_alignment"):
        replace(study, compositions=CompositionTable(bad_composition))
    study.close()


def test_observations_reject_duplicate_support_keys() -> None:
    observations = pd.DataFrame(
        {
            "observation_id": ["series-a@rest", "series-b@rest"],
            "series_id": ["series-a", "series-b"],
            "checkpoint_id": ["rest", "rest"],
            "sample_id": ["sample-a", "sample-b"],
            "geometry_observed": [True, True],
            "support_key": ["shared", "shared"],
        }
    )
    with pytest.raises(ValueError, match="duplicate support_key"):
        ObservationTable(observations)


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
