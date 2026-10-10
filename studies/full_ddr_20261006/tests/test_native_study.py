"""Synthetic native-v4 adapter/real-compiler tests; no cohort fixture access."""
# ruff: noqa: E402

from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "source" / "src")]

import numpy as np
import pytest

from credo.data import (
    ArtifactRef,
    EmpiricalLaw,
    InMemorySupportStore,
    RepresentationSpec,
    SupportRef,
    open_study,
    write_study,
)
from credo.recipes.compact_sde_v3.recipe import CompactV3Config
from native_study import (
    CountObservation,
    Guide,
    build_native_study,
    compile_development,
    count_vector_sha256,
)


def fixture():
    guides = (
        Guide("NTC-1", "NTC", True),
        Guide("a-1", "A", False),
        Guide("a-2", "A", False),
        Guide("b-1", "B", False),
        Guide("new-1", "NEW", False),
    )
    observations, records, laws = [], {}, {}
    for donor in ("D2", "D3", "D4"):
        for checkpoint in ("Stim8hr", "Stim48hr"):
            counts = (
                (2, 2, 0, 1, int(donor == "D2"))
                if checkpoint == "Stim8hr"
                else (1, 3, 0, 0, int(donor == "D2"))
            )
            sid = f"{donor}_{checkpoint}"
            observations.append(
                CountObservation(
                    sid,
                    donor,
                    checkpoint,
                    counts,
                    count_vector_sha256(counts),
                    "a" * 64,
                )
            )
            for index, count in enumerate(counts):
                n = count if index in (0, 1, 4) else 0
                if donor == "D3" and checkpoint == "Stim48hr" and index == 1:
                    n = 0
                key = f"{sid}::{index}"
                records[(sid, index)] = dict(
                    source_id=sid,
                    guide_index=index,
                    captured_cells=count,
                    geometry_cells=n,
                    geometry_absent_cells=count - n,
                    available=n > 0,
                    support_key=key,
                )
                if n:
                    laws[SupportRef("frozen", "r48-test", key)] = EmpiricalLaw(
                        np.arange(n * 2).reshape(n, 2), np.ones(n) / n
                    )
    store = InMemorySupportStore("frozen", laws)
    artifact = ArtifactRef(
        "synthetic-bound-artifact",
        "memory:synthetic",
        "d" * 64,
        None,
        "application/octet-stream",
    )
    representation = RepresentationSpec(
        "r48-test",
        "frozen-r48",
        "latent",
        2,
        "frozen",
        encoder_artifact=artifact,
        normalization_artifact=artifact,
        feature_artifact=artifact,
        support_artifact=artifact,
        fit_subject_ids=("D3", "D4"),
        fit_checkpoint_ids=("Stim48hr",),
        fit_observation_scope="custom",
    )
    return dict(
        study_id="synthetic-all4",
        guides=guides,
        observations=observations,
        support_records=records,
        supports=store,
        representation=representation,
        authority_sha256="b" * 64,
        representation_provenance={
            "fit_conditions": ["Rest", "Stim48hr"],
            "fit_subjects": ["D3", "D4"],
            "frozen_checkpoint": "c" * 64,
        },
    )


def config():
    return CompactV3Config.model_validate(
        {
            "model": {"context": "none"},
            "training": {"epochs": {"state": 1, "mass": 1, "context": 0}},
            "loss": {
                "mass": 0,
                "count_likelihood": "categorical_ce",
                "mass_increment": "particle_log_mass",
            },
        }
    )


def test_build_and_compile_real_native_populations_complete_denominators():
    bundle = build_native_study(**fixture())
    problem, split = compile_development(bundle, config())
    assert bundle.study.manifest.schema_version == 4
    assert set(problem.training.measure_meta.sample_id) == {"D3", "D4"}
    assert set(problem.validation.measure_meta.sample_id) == {"D2"}
    assert split.held_out_subject_ids == ("D2",)
    assert not any(
        "D1" in sid
        for sid in (*problem.training.measure_ids, *problem.validation.measure_ids)
    )
    assert problem.training.axis.values == (8.0, 48.0)
    assert problem.training.axis.normalized_values == (0.0, 1.0)
    for data in (problem.training, problem.validation):
        for block in data.count_blocks:
            assert block.time_label == "Stim48hr"
            assert len(block.counts) + len(block.background_counts) == 5
            assert block.n_total > 0
            assert block.background_fitness.eq(0).all()
            assert block.conditioning_policy == "preserve_background"
            for index, mass in zip(
                block.measure_indices.tolist(), block.exposure.tolist(), strict=True
            ):
                assert data.measures["Stim8hr"][
                    data.measure_ids[index]
                ].total_mass == pytest.approx(mass)
        for sid in data.measure_ids:
            law = data.measures["Stim8hr"][sid]
            assert law.support.ndim == 2
            assert law.weights.sum() == pytest.approx(law.total_mass)
    # D2's previously unseen NEW target has real source geometry, but no fitted
    # target parameters: it remains in the complete denominator as persistence.
    assert "D2::guide-index-4" not in problem.validation.measure_ids
    assert (
        "D2::guide-index-4" in problem.validation.count_blocks[0].background_series_ids
    )


def test_raw_zero_missing_geometry_and_withheld_observation_stay_distinct():
    bundle = build_native_study(**fixture())
    raw = bundle.study.abundance.to_pandas().query("channel_id == 'captured_count'")
    zero = raw[raw.observation_id == "D3::guide-index-2::Stim8hr"].iloc[0]
    withheld = raw[raw.observation_id == "D1::guide-index-2::Stim48hr"].iloc[0]
    assert zero.observed and zero.value == 0
    assert not withheld.observed and np.isnan(withheld.value)
    problem, _ = compile_development(bundle, config())
    assert "D3::guide-index-1" in problem.training.measures["Stim8hr"]
    assert "D3::guide-index-1" not in problem.training.measures["Stim48hr"]
    assert not any(
        "D1" in str(ref)
        for ref in bundle.study.support_index.to_pandas().support_key.dropna()
    )


def test_native_writer_roundtrip_retains_provenance_and_compiles(tmp_path):
    bundle = build_native_study(**fixture())
    manifest = write_study(bundle.study, tmp_path / "native")
    reopened = open_study(manifest, verify="full")
    try:
        rebound = replace(bundle, study=reopened)
        problem, _ = compile_development(rebound, config())
        assert len(problem.training.count_blocks) == 2
        assert reopened.provenance["representation_original_fit_scope"][
            "fit_conditions"
        ] == ["Rest", "Stim48hr"]
        assert reopened.population_pools is None
    finally:
        reopened.close()


def test_source_count_hash_mismatch_rejected():
    values = fixture()
    values["observations"][0] = replace(
        values["observations"][0], vector_sha256="0" * 64
    )
    with pytest.raises(ValueError, match="Count vector content"):
        build_native_study(**values)


def test_count_support_population_mismatch_rejected():
    values = fixture()
    values["support_records"][("D3_Stim8hr", 0)]["captured_cells"] = 99
    with pytest.raises(ValueError, match="counts disagree"):
        build_native_study(**values)


def test_incomplete_catalog_support_metadata_rejected():
    values = fixture()
    del values["support_records"][("D3_Stim8hr", 0)]
    with pytest.raises(ValueError, match="exactly the admitted"):
        build_native_study(**values)


def test_source_donor_relabel_rejected_before_support_lookup():
    values = fixture()
    values["observations"][0] = replace(values["observations"][0], source_id="D3_Stim8hr")
    with pytest.raises(ValueError, match="Canonical source identity"):
        build_native_study(**values)


def test_source_checkpoint_relabel_rejected():
    values = fixture()
    values["observations"][0] = replace(values["observations"][0], source_id="D2_Stim48hr")
    with pytest.raises(ValueError, match="Canonical source identity"):
        build_native_study(**values)


def test_existing_other_guide_law_cannot_be_substituted():
    values = fixture()
    record = values["support_records"][("D2_Stim8hr", 0)]
    # Both guide 0 and 1 have two captured/two geometry cells; counts cannot
    # reveal this substitution, and the wrong reference really exists.
    record["support_key"] = "D2_Stim8hr::1"
    assert values["supports"].contains(SupportRef("frozen", "r48-test", record["support_key"]))
    with pytest.raises(ValueError, match="exact canonical source/guide"):
        build_native_study(**values)


def test_missing_geometry_count_must_reconcile():
    values = fixture()
    values["support_records"][("D3_Stim8hr", 0)]["geometry_absent_cells"] = 1
    with pytest.raises(ValueError, match="Unavailable geometry count"):
        build_native_study(**values)


def test_catalog_denominator_computed_once_per_source(monkeypatch):
    import builtins
    import native_study

    values = fixture()
    calls = []

    def counting_sum(items, start=0):
        calls.append(items)
        return builtins.sum(items, start)

    monkeypatch.setattr(native_study, "sum", counting_sum, raising=False)
    bundle = build_native_study(**values)
    assert len(calls) == len(values["observations"])
    assert {id(items) for items in calls} == {id(obs.counts) for obs in values["observations"]}
    values_by_source = {obs.source_id: obs for obs in values["observations"]}
    abundance = bundle.study.abundance.to_pandas()
    for source, obs in values_by_source.items():
        actual = abundance[abundance.channel_id.eq("regularized_captured_frequency")
                           & abundance.observation_id.str.startswith(f"{obs.donor}::")
                           & abundance.observation_id.str.endswith(f"::{obs.checkpoint}")]
        expected = [(count + 0.5) / (builtins.sum(obs.counts) + 0.5*len(obs.counts)) for count in obs.counts]
        np.testing.assert_allclose(actual.value.to_numpy(), expected, rtol=0, atol=0)


def test_absolute_mass_or_ecology_config_rejected():
    bundle = build_native_study(**fixture())
    candidate = config()
    with pytest.raises(ValueError, match="zero absolute mass"):
        compile_development(
            bundle,
            candidate.model_copy(
                update={"loss": candidate.loss.model_copy(update={"mass": 1})}
            ),
        )
    with pytest.raises(ValueError, match="ecology"):
        compile_development(
            bundle,
            candidate.model_copy(
                update={
                    "model": candidate.model.model_copy(
                        update={"context": "catalog_bank"}
                    )
                }
            ),
        )


def test_missing_frozen_representation_identity_rejected():
    values = fixture()
    values["representation"] = replace(values["representation"], encoder_artifact=None)
    with pytest.raises(ValueError, match="artifact identities"):
        build_native_study(**values)


def test_representation_fit_provenance_cannot_be_silently_reassigned():
    values = fixture()
    values["representation_provenance"]["fit_subjects"] = ["D1", "D2"]
    with pytest.raises(ValueError, match="D3/D4"):
        build_native_study(**values)


def test_residual_native_training_save_reload_preserves_new_contract(tmp_path):
    import torch

    from credo.io import RunConfig
    from credo.recipes.compact_sde_v3.training import Trainer
    from credo.registry import get_recipe
    from credo.runtime import TrainingEngine

    bundle = build_native_study(**fixture())
    manifest = write_study(bundle.study, tmp_path / "native")
    settings = CompactV3Config.model_validate(
        {
            "model": {
                "context": "none",
                "trunk_kind": "residual_mlp",
                "trunk_depth": 3,
                "hidden_dim": 16,
                "embedding_dim": 4,
                "n_programs": 2,
            },
            "training": {
                "epochs": {"state": 0, "mass": 1, "context": 0},
                "particles": 4,
                "measures_per_batch": 16,
                "steps_per_interval": 2,
                "checkpoint_selection": "last",
            },
            "evaluation": {"particles": 4, "measures_per_batch": 16},
            "loss": {
                "mass": 0,
                "count_likelihood": "categorical_ce",
                "mass_increment": "particle_log_mass",
            },
        }
    )
    problem, split = compile_development(bundle, settings)
    run_config = RunConfig.model_validate(
        {
            "study": manifest,
            "recipe_config": settings,
            "output": tmp_path / "run",
            "selection": {
                "series_ids": split.train_series_ids + split.validation_series_ids,
                "composition_policy": "preserve_background",
            },
            "split": {
                "strategy": "subject",
                "train_values": ("D3", "D4"),
                "validation_values": ("D2",),
                "representation_scope": "shared",
            },
        }
    )
    trainer = TrainingEngine().fit(
        get_recipe(run_config.recipe), problem, run_config, device="cpu"
    )
    output = trainer.save()
    loaded = Trainer.load(
        output / "state" / "checkpoint.pt", problem, run_config, device="cpu"
    )
    assert loaded.model.architecture()["trunk_kind"] == "residual_mlp"
    assert loaded.model.architecture()["trunk_depth"] == 3
    assert loaded.settings.loss.count_likelihood == "categorical_ce"
    assert loaded.settings.loss.mass_increment == "particle_log_mass"
    assert loaded.settings.loss.mass == 0
    assert not loaded.log_count_concentration.requires_grad
    assert (
        loaded.bank.mass_increment
        == loaded.validation_bank.mass_increment
        == "particle_log_mass"
    )
    for name, value in trainer.model.state_dict().items():
        torch.testing.assert_close(
            value, loaded.model.state_dict()[name], rtol=0, atol=0
        )
    assert np.isfinite(trainer.metrics.predicted_log_mass).all()
