"""Actual tiny file/loader/compiler/factory joins; zero optimizer steps.

Native arrays and selected observation receipts are fabricated. Normalizer
per-law scores are deliberately fabricated scalar fixtures: this tests their
publication binding/reduction, not a second numerical-normalization experiment.
"""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import pytest
import torch

from frontier import normalization, sampling
from frontier.cache_compiler import (CompilerLimits, NativeShard, SourceCompileSpec, compile_population,
    native_source_input_identity, ordered_native_chunks)
from frontier.cache_storage import load_fold_cache, publish_fold_cache
from frontier.checkpoints import TrajectoryLease, file_sha256
from frontier.fitting_factory import NormalizerArtifact, bind_normalizers, prepare_fitting_components
from frontier.fitting_provider import CatalogGuide, FoldCacheManifest, FoldFittingCache, ProviderLimits, _hash
from frontier.geometry import GeometryOptions, RaggedSinkhornGeometry, geomloss_backend
from frontier.native_inputs import FoldInputs
from frontier.objective_registry import ObjectiveRules, RULES_SHA256
from frontier.observation import ObservationFold
from frontier.observation_loader import load_observation
from frontier.observation_training import ObservationTrainingSpec
from frontier.sampling import SamplingContext, SamplerSpecification
from frontier.training_objective import ObjectiveExecution
from frontier.training_owner import OwnerSchedule
from test_native_inputs import fixture as native_fixture
from test_observation_loader import fixture as observation_fixture


torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def package(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.optim.Adam, "step", lambda *_a, **_k: pytest.fail("No optimizer step authorized"))
    root, _, _, index = native_fixture.__wrapped__(tmp_path / "native")
    fold = FoldInputs("inner-D3-D4", ("D3", "D4"))
    spec = replace(ObservationTrainingSpec.fixture(), features=3, latent_dim=48)
    admission, _ = observation_fixture(tmp_path / "observation", specification=spec,
        observation_fold=ObservationFold(fold.fold_id, fold.training_donors, index.rna_axis_sha256),
        input_sha256=index.access(fold, "representation_fit").identity())
    decoder, binding = load_observation(admission)
    catalog = tuple(CatalogGuide(row.guide_id, row.target_id, bool(row.is_control))
                    for row in index.catalog().itertuples(index=False))
    populations = []
    for donor in fold.training_donors:
        for condition, role in (("Stim8hr", "dynamics_source"), ("Stim48hr", "dynamics_supervision")):
            handle = index.open(root, fold, role, max_batch_rows=3)
            source_id = f"{donor}_{condition}"
            support = handle.support_counts(source_id)
            compile_spec = SourceCompileSpec(fold, catalog, 3, index.rna_axis_sha256, binding,
                donor, condition, source_id, "fabricated-culture", role, handle.access.identity(),
                native_source_input_identity(index, source_id),
                tuple(NativeShard(s.shard, s.rows) for s in index.shards if s.source_id == source_id),
                tuple(map(int, support.captured_cells)), tuple(map(int, support.positive_RNA_cells)), True)
            output = compile_population(compile_spec, ordered_native_chunks(handle, compile_spec, batch_rows=3),
                decoder, limits=CompilerLimits(8*1024**2, 3, 64*1024, 2, 64*1024))
            populations.append(output.population)
    manifest = FoldCacheManifest(fold, catalog, index.rna_axis_sha256, 3, binding,
        index.access(fold, "dynamics_source").identity(), index.access(fold, "dynamics_supervision").identity(),
        tuple(p.reference for p in populations), synthetic_fixture=True)
    limits = ProviderLimits(8*1024**2, 8*1024**2, 64)
    cache = FoldFittingCache(manifest, populations, expected_manifest_sha256=manifest.identity, limits=limits)
    publication = publish_fold_cache(cache, tmp_path / "cache", expected_manifest_sha256=manifest.identity,
        producer_sha256=file_sha256(ROOT / "frontier/cache_compiler.py"), output_cap_bytes=1024**2)
    with load_fold_cache(tmp_path / "cache", expected_manifest_sha256=manifest.identity,
            expected_complete_sha256=publication["complete_sha256"], output_cap_bytes=1024**2, limits=limits) as mapped:
        geometry = RaggedSinkhornGeometry(options=GeometryOptions(100000, 4, geomloss_backend().identity["version"]))
        sampler_binding = dict(context=asdict(SamplingContext("fabricated-factory", fold.training_donors,
            "F0-V0", 731, 0, 0, "scales")), specification=asdict(SamplerSpecification(maximum_laws=1)),
            source_sha256=file_sha256(sampling.__file__), particles=8, pairwise_block=4,
            geometry=geometry.backend_identity, unused_noise_omitted=True, unused_full_source_diagnostics_omitted=True)
        laws = []
        for i in mapped.paired:
            donor, guide = mapped.laws[i]
            entry = catalog[guide]
            row = dict(donor=donor, guide_index=guide, guide_id=entry.guide_id, target_id=entry.target_id,
                is_control=entry.is_control, state_energy=1., spread_absolute_error=2., shape_mmd2=3.,
                joint_mse=4., sinkhorn=5.)
            if not entry.is_control:
                row.update(target_endpoint_mse=6., pooled_effect_mse=7.)
            laws.append(row)
        donors = [dict(donor=d, catalog_categories=3, count_ce_minus_entropy=.2,
                       count_quadratic=1.5, pooled_control_endpoint_mse=.8) for d in fold.training_donors]
        record = normalization.reduce_normalizers(mapped, laws, donors, sampler_binding=sampler_binding,
            producer_sha256=file_sha256(normalization.__file__))
        raw = json.dumps(record, sort_keys=True, allow_nan=False).encode()
        artifact = NormalizerArtifact(raw, hashlib.sha256(raw).hexdigest(), 1024**2)
        args = dict(observation_admission=admission, cache=mapped, native_index=index, normalizers=artifact,
            rules=ObjectiveRules((ROOT / "MAIN_OBJECTIVE_RULES.json").read_bytes(), RULES_SHA256),
            configuration_id="F0-V1", model_seed=0, study_id="fabricated-factory", study_seed=731,
            trajectory_id="fabricated-main", source_sha256="a"*64,
            schedule=OwnerSchedule("synthetic_cpu_only", (2, 4), 1, 2, 32, 2, 2, 8, 4),
            execution=ObjectiveExecution(1, 1, 3, 4, 4, 100000, 3e-6, 3e-6), geometry=geometry)
        yield args, record


def changed_artifact(record, change):
    record = json.loads(json.dumps(record))
    change(record)
    record["content_sha256"] = _hash({k: v for k, v in record.items() if k != "content_sha256"})
    raw = json.dumps(record, sort_keys=True, allow_nan=False).encode()
    return NormalizerArtifact(raw, hashlib.sha256(raw).hexdigest(), 1024**2)


@pytest.mark.parametrize("family", ["F0", "F1", "F2", "F3", "F4", "F5"])
def test_actual_selected_cache_factory_join_including_nonmolecular(package, tmp_path, family):
    args, record = package
    before = torch.get_rng_state().clone()
    prepared = prepare_fitting_components(**{**args, "configuration_id": family + "-V1"})
    assert torch.equal(before, torch.get_rng_state())
    assert prepared.arguments["decoder"] is not None
    assert prepared.identity["fold_id"] == "inner-D3-D4" and prepared.identity["seed"] == 0
    assert prepared.identity["training_owner"]["observation_binding"] == asdict(args["cache"].manifest.observation_binding)
    assert prepared.scale_binding.values["state"] == 5.
    assert prepared.scale_binding.scale_artifact_sha256 == args["normalizers"].file_sha256
    assert prepared.identity["normalizer_content_sha256"] == record["content_sha256"]
    equation_sources = prepared.identity["forecast_model_sources"]
    assert equation_sources["schema"] == "credo.frontier.forecast-model-sources.v1"
    assert equation_sources["producer_source_sha256"] == args["source_sha256"]
    assert set(equation_sources["files"]) == {"frontier/models.py", "frontier/configuration.py"}
    for name, entry in equation_sources["files"].items():
        payload = (ROOT / name).read_bytes()
        assert entry == dict(sha256=hashlib.sha256(payload).hexdigest(), size_bytes=len(payload))
    assert prepared.scale_binding.scale_artifact_sha256 != record["content_sha256"]
    assert prepared.support_factors.target_ids == ("T0",)
    assert prepared.support_factors.support_scores == (2,)
    assert prepared.arguments["model"].target_ids == (("T0",) if family in ("F3", "F5") else ())
    with TrajectoryLease(tmp_path / "owner", prepared.identity, "fixture") as lease:
        owner = prepared.open_owner(lease)
        assert owner.accepted_updates == owner.optimizer_calls == 0
        assert owner.decoder is prepared.arguments["decoder"]
        assert all(not p.requires_grad for p in owner.decoder.parameters())
        assert not owner.history.exists()
        with pytest.raises(ValueError, match="once"):
            prepared.open_owner(lease)


@pytest.mark.parametrize("field,value", [("seed", 1), ("fold_id", "wrong-fold")])
def test_owner_lease_cannot_relabel_fold_or_seed(package, tmp_path, field, value):
    args, _ = package
    prepared = prepare_fitting_components(**args)
    with TrajectoryLease(tmp_path / "owner", {**prepared.identity, field: value}, "fixture") as lease:
        with pytest.raises(ValueError, match="fold/seed"):
            prepared.open_owner(lease)
        assert not (lease.root / "TRAINING_OWNER_STATE.json").exists()


@pytest.mark.parametrize("change", [
    lambda r: r["training_values"].update(state=17.),
    lambda r: r["per_law"].pop(),
    lambda r: r["sampler_binding"]["context"].update(study_seed=999),
    lambda r: r.update(observation_binding={"artifact_sha256": "e"*64, "state_sha256": "f"*64}),
    lambda r: r.update(query_outcomes_used=True),
])
def test_changed_saved_scales_or_provenance_fail_before_model_construction(package, monkeypatch, change):
    args, record = package
    monkeypatch.setattr("frontier.fitting_factory.build_model", lambda *_a, **_k: pytest.fail("must fail before model"))
    with pytest.raises(ValueError):
        prepare_fitting_components(**{**args, "normalizers": changed_artifact(record, change)})


@pytest.mark.parametrize("kind", ["model", "decoder", "count"])
def test_changed_prepared_components_rejected_before_owner(package, tmp_path, kind):
    args, _ = package
    prepared = prepare_fitting_components(**args)
    with torch.no_grad():
        if kind == "model": next(prepared.arguments["model"].parameters()).add_(.1)
        elif kind == "decoder": prepared.arguments["decoder"].center.add_(.1)
        else: prepared.arguments["blocks"][0].endpoint_counts[0] += 1
    with TrajectoryLease(tmp_path / "owner", prepared.identity, "fixture") as lease:
        with pytest.raises(ValueError):
            prepared.open_owner(lease)
        assert not (lease.root / "TRAINING_OWNER_STATE.json").exists()


@pytest.mark.parametrize("field,value", [("model_seed", 1), ("study_seed", 999),
                                        ("configuration_id", "F0-V2")])
def test_provider_context_mutation_rejected_before_owner(package, tmp_path, field, value):
    args, _ = package
    prepared = prepare_fitting_components(**args)
    setattr(prepared.arguments["provider"], field, value)
    with TrajectoryLease(tmp_path / "owner", prepared.identity, "fixture") as lease:
        with pytest.raises(ValueError, match="[Cc]ontext|[Bb]inding|changed"):
            prepared.open_owner(lease)
        assert not (lease.root / "TRAINING_OWNER_STATE.json").exists()


def test_provider_mutation_after_owner_open_rejects_before_sampling(package, tmp_path):
    args, _ = package
    prepared = prepare_fitting_components(**args)
    with TrajectoryLease(tmp_path / "owner", prepared.identity, "fixture") as lease:
        owner = prepared.open_owner(lease)
        provider = prepared.arguments["provider"]
        provider.study_seed += 1
        with pytest.raises(ValueError, match="[Cc]ontext|[Bb]inding|changed"):
            provider.sample(accepted_updates=0, phase="state")
        assert owner.accepted_updates == owner.optimizer_calls == 0
        assert not owner.history.exists()


def test_changed_model_equation_files_rejected_before_owner(package, tmp_path, monkeypatch):
    args, _ = package
    prepared = prepare_fitting_components(**args)
    changed = json.loads(json.dumps(prepared.identity["forecast_model_sources"]))
    changed["files"]["frontier/models.py"]["sha256"] = "0" * 64
    monkeypatch.setattr("frontier.fitting_factory.forecast_model_sources", lambda _: changed)
    with TrajectoryLease(tmp_path / "owner", prepared.identity, "fixture") as lease:
        with pytest.raises(ValueError, match="equation source"):
            prepared.open_owner(lease)
        assert not (lease.root / "TRAINING_OWNER_STATE.json").exists()


def test_native_role_mismatch_rejected_before_observation_loading(package, monkeypatch):
    args, _ = package
    index = args["native_index"]
    actual = index.access
    monkeypatch.setattr(index, "access", lambda fold, role: actual(fold,
        "dynamics_supervision" if role == "dynamics_source" else role))
    monkeypatch.setattr("frontier.fitting_factory.load_observation", lambda *_: pytest.fail("must reject before load"))
    with pytest.raises(ValueError, match="capabilities"):
        prepare_fitting_components(**args)


def test_nonmolecular_wrong_actual_observation_is_rejected(package, monkeypatch):
    args, _ = package
    actual = load_observation
    def changed(admission):
        model, binding = actual(admission)
        with torch.no_grad(): model.scale.add_(.2)
        return model, binding
    monkeypatch.setattr("frontier.fitting_factory.load_observation", changed)
    with pytest.raises(ValueError, match="observation module"):
        prepare_fitting_components(**args)


def test_fixture48_preserves_scientific_spec_and_other_caps():
    tiny = ObservationTrainingSpec.fixture()
    assert tiny.latent_dim == 3
    assert replace(tiny, latent_dim=48).synthetic_fixture
    with pytest.raises(ValueError, match="Fabricated"):
        replace(tiny, latent_dim=49)
    with pytest.raises(ValueError, match="Scientific"):
        replace(ObservationTrainingSpec(), latent_dim=47)


def test_normalizer_file_bytes_bounded_and_hash_distinct():
    with pytest.raises(ValueError, match="bytes"):
        NormalizerArtifact(b"{}", "a"*64, 100)
    with pytest.raises(ValueError, match="bytes"):
        NormalizerArtifact(b"{}", hashlib.sha256(b"{}").hexdigest(), 1)
