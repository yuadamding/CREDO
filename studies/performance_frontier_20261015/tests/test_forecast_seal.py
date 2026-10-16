"""Fabricated packed files only: no models, optimizer, truth or real payloads."""

from contextlib import ExitStack
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from frontier.cache_compiler import native_source_input_identity
from frontier.fitting_provider import CatalogGuide, PopulationArrays, PopulationReference, _hash
from frontier.forecast import ConditionalForecast, ForecastModelBinding, ForecastSource
from frontier.forecast_seal import (ForecastSealEntry, ForecastSealPlan, freeze_calibration_derivation,
                                    freeze_ensemble_derivation, publish_forecast_seal)
from frontier.forecast_storage import ForecastReader, ForecastStorageLimits, ForecastStorageSpec, ForecastWriter
from frontier.measure_ops import calibrate_abundance, calibrate_dispersion, catalog_log_frequencies, source_log_frequencies
from frontier.native_inputs import FoldInputs, VerifiedForecastRelease
from frontier.sampling import SamplingContext
from frontier.selector_contract import Recipe
from frontier.training_owner import ObservationBinding
from test_native_inputs import fixture as native_fixture


@pytest.fixture(autouse=True)
def bounded_threads():
    old = torch.get_num_threads(); torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def ro(values, dtype):
    result = np.asarray(values, dtype=dtype).copy(order="C")
    result.setflags(write=False)
    return result


def source(index, *, fold=None):
    if fold is None:
        fold = FoldInputs("subset-D3-D4", ("D3", "D4"))
    donor, source_id = "D1", "D1_Stim8hr"
    catalog = tuple(CatalogGuide(r.guide_id, r.target_id, bool(r.is_control))
                    for r in index.catalog().itertuples(index=False))
    counts, positive = ro([2, 1, 0], np.int64), ro([1, 1, 0], np.int64)
    offsets = ro([0, 1, 2, 2], np.int64)
    states = ro(np.arange(96).reshape(2, 48)/100, np.float32)
    observation = ObservationBinding("1"*64, "2"*64)
    reference = PopulationReference(donor, "Stim8hr", source_id, "fabricated-series",
        index.access(fold, "query", query_donor=donor).identity(), native_source_input_identity(index, source_id),
        "3"*64, "4"*64, "0"*64)
    population = PopulationArrays(reference, counts, positive, offsets, states, None)
    reference = replace(reference, latent_sha256=population.latent_identity(
        catalog_sha256=_hash([asdict(g) for g in catalog]), observation_binding=observation))
    return ForecastSource(fold, catalog, observation, index.rna_axis_sha256, reference,
                          counts, positive, offsets, states, True)


def fabricated_bank(path, early, seed, *, updates=1024, prefixes=(4, 7), ensemble_totals=()):
    """Use real publication format, but explicitly fabricated particles/receipt."""
    binding = ForecastModelBinding(early.fold, early.observation_binding, "F0-V0",
        str(seed+5)*64, str(seed+7)*64, "9"*64, "a"*64, seed, updates)
    context = SamplingContext("fabricated-seal", early.fold.training_donors, "F0-V0", 17, seed, updates, "forecast")
    spec = ForecastStorageSpec.from_source(early, binding, context, Recipe("F0-V0"), prefixes=prefixes,
                                           ensemble_component_totals=ensemble_totals)
    doc = spec.document
    receipt = {key: doc[key] for key in ("source_identity", "model_binding", "context", "recipe", "prefixes",
        "single_seed_prefixes", "ensemble_component_totals", "ensemble_component_prefixes",
        "maximum_generated_particles", "synthetic_fixture", "supported_guide_indices",
        "unavailable_source_guide_indices", "pooled_control_identity")}
    receipt.update(schema="credo.frontier.in-memory-forecast.v1", complement_mass_multiplier=1,
        pooled_control_is_catalog_entry=False, source_only_support=True, endpoint_inputs_used=False,
        optimizer_calls=0, input_artifact_custody_verified_here=False, checkpoint_loading_verified_here=False,
        publication_claimed=False, scientific_launch_authority=False, control_ids=["NTC"], query_donor="D1",
        sampler_receipts=[], complete_prediction_checks_passed=True, law_storage="in_memory", sink_spec_sha256=None)
    limits = ForecastStorageLimits(128*1024**2, 128*1024**2, 2)
    states = torch.arange(max(prefixes)*48, dtype=torch.float32).reshape(max(prefixes), 48)/100
    with ForecastWriter(path, spec=spec, producer_sha256="b"*64, limits=limits) as writer:
        for prefix in spec.effective_prefixes:
            logs = torch.full((prefix,), -np.log(prefix), dtype=torch.float64)
            law = ConditionalForecast(states[:prefix], calibrate_dispersion(states[:prefix], logs, 1.), logs, 0.)
            for guide in doc["supported_guide_indices"]: writer.emit(prefix, guide, law)
            writer.emit(prefix, "pooled_control", law)
            base = source_log_frequencies(torch.from_numpy(early.captured_counts.copy()))
            raw = catalog_log_frequencies(base, torch.tensor(doc["supported_guide_indices"]),
                                          torch.zeros(2, dtype=torch.float64))
            writer.write_catalog(prefix, log_mass_increments=torch.zeros(3, dtype=torch.float64),
                log_source_frequencies=base, uncalibrated_log_frequencies=raw,
                log_frequencies=calibrate_abundance(base, raw, 1.))
        publication = writer.finish(receipt)
    entry = ForecastSealEntry("seed-"+str(seed)+"/all-prefixes", path.name, spec,
                             publication["complete_sha256"], publication["manifest_sha256"])
    return entry


@pytest.fixture
def package(native_fixture, tmp_path):
    index = native_fixture[3]
    early = source(index)
    root = tmp_path / "banks"; root.mkdir()
    entries = tuple(fabricated_bank(root / ("bank-"+str(seed)), early, seed) for seed in (0, 1))
    plan = ForecastSealPlan(early.fold, "D1", early.reference.source_access_sha256, entries)
    with ExitStack() as stack:
        readers = tuple(stack.enter_context(ForecastReader(root/e.bank_relative_root, spec=e.spec,
            expected_complete_sha256=e.expected_complete_sha256, expected_manifest_sha256=e.expected_manifest_sha256,
            maximum_output_bytes=128*1024**2, maximum_working_bytes=128*1024**2)) for e in entries)
        yield root, index, plan, readers


def publish(package, **changes):
    root, index, plan, readers = package
    args = dict(plan=plan, expectation=plan.expectation, readers=readers, native_index=index, output_cap_bytes=32*1024)
    args.update(changes)
    return publish_forecast_seal(root, "FORECAST_SEAL.json", **args)


def test_exact_banks_seal_and_release_only_this_query_without_truth_reads(package, monkeypatch):
    root, index, plan, readers = package
    from frontier import models
    monkeypatch.setattr(index, "open", lambda *_a, **_k: pytest.fail("No raw handles"))
    monkeypatch.setattr(models, "build_model", lambda *_a, **_k: pytest.fail("No model construction"))
    before = {str(p.relative_to(root)): (p.stat().st_ino, p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest())
              for e in plan.entries for p in (root/e.bank_relative_root).iterdir()}
    release = publish(package)
    assert isinstance(release, VerifiedForecastRelease)
    assert release.expectation == plan.expectation
    assert release.manifest_sha256 == hashlib.sha256((root/"FORECAST_SEAL.json").read_bytes()).hexdigest()
    record = json.loads((root/"FORECAST_SEAL.json").read_text())
    assert tuple(r["forecast_id"] for r in record["forecasts"]) == plan.expectation.forecast_ids
    assert [len(r["artifacts"]) for r in record["forecasts"]] == [7, 6]
    assert sum(len(r["artifacts"]) for r in record["forecasts"]) == len({a["relative_uri"] for r in record["forecasts"] for a in r["artifacts"]})
    assert not (root/".FORECAST_SEAL.json.pending").exists()
    plan_record = json.loads((root/"FORECAST_SEAL.json.plan.json").read_text())
    assert plan_record["expectation"] == plan.expectation.record()
    assert _hash(plan_record["configuration_inventory"]) == plan.configuration_sha256
    for name, identity in before.items():
        p = root/name
        assert (p.stat().st_ino, p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest()) == identity
    truth = index.access(plan.fold, "evaluation_truth", query_donor="D1", release=release)
    assert truth.source_ids == ("D1_Stim48hr",)  # Metadata only, no endpoint payload access.
    with pytest.raises(ValueError, match="different fold/query"):
        index.access(plan.fold, "evaluation_truth", query_donor="D2", release=release)
    with pytest.raises(FileExistsError): publish(package)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "reordered", "closed", "expectation", "cap"])
def test_incomplete_foreign_or_unbound_inventory_never_publishes(package, kind):
    root, index, plan, readers = package
    changes = {}
    if kind == "missing": changes["readers"] = readers[:1]
    elif kind == "duplicate": changes["readers"] = (readers[0], readers[0])
    elif kind == "reordered": changes["readers"] = tuple(reversed(readers))
    elif kind == "closed": readers[0].close()
    elif kind == "expectation": changes["expectation"] = replace(plan.expectation, configuration_sha256="0"*64)
    elif kind == "cap": changes["output_cap_bytes"] = 1
    with pytest.raises((ValueError, RuntimeError)): publish(package, **changes)
    assert not (root/"FORECAST_SEAL.json").exists()
    assert not (root/".FORECAST_SEAL.json.pending").exists()


@pytest.mark.parametrize("kind", ["id", "root", "spec", "path_traversal", "query", "source"])
def test_frozen_plan_rejects_duplicate_or_foreign_entries(package, kind):
    _, _, plan, _ = package
    first, second = plan.entries
    with pytest.raises(ValueError):
        if kind == "id": replace(plan, entries=(first, replace(second, forecast_id=first.forecast_id)))
        elif kind == "root": replace(plan, entries=(first, replace(second, bank_relative_root=first.bank_relative_root)))
        elif kind == "spec": replace(plan, entries=(first, replace(second, spec=first.spec)))
        elif kind == "path_traversal": replace(first, bank_relative_root="../bank-0")
        elif kind == "query": replace(plan, query_donor="D2")
        elif kind == "source": replace(plan, query_access_sha256="0"*64)


def test_changed_native_metadata_identity_cannot_release(package, monkeypatch):
    root, index, _, _ = package
    monkeypatch.setattr(index, "rna_axis_sha256", "0"*64)
    with pytest.raises(ValueError, match="native early query"):
        publish(package)
    assert not (root/"FORECAST_SEAL.json").exists()


@pytest.mark.parametrize("kind", ["root_symlink", "seal_traversal", "unfinished_attempt"])
def test_exclusive_path_and_failed_attempt_guards(package, tmp_path, kind):
    root, index, plan, readers = package
    if kind == "root_symlink":
        link = tmp_path/"alias"; link.symlink_to(root, target_is_directory=True)
        with pytest.raises(ValueError, match="nonsymlink"):
            publish_forecast_seal(link, "FORECAST_SEAL.json", plan=plan, expectation=plan.expectation,
                readers=readers, native_index=index, output_cap_bytes=32*1024)
    elif kind == "seal_traversal":
        with pytest.raises(ValueError):
            publish_forecast_seal(root, "../outside.json", plan=plan, expectation=plan.expectation,
                readers=readers, native_index=index, output_cap_bytes=32*1024)
    else:
        pending = root/".FORECAST_SEAL.json.pending"; pending.write_bytes(b"retained failed attempt")
        with pytest.raises(FileExistsError): publish(package)
        assert pending.read_bytes() == b"retained failed attempt"
    assert not (root/"FORECAST_SEAL.json").exists()


def test_verifier_failure_retains_incomplete_attempt_not_final_seal(package, monkeypatch):
    root, _, _, _ = package
    monkeypatch.setattr(VerifiedForecastRelease, "verify", classmethod(
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("fabricated verification failure"))))
    with pytest.raises(ValueError, match="fabricated verification failure"): publish(package)
    assert (root/".FORECAST_SEAL.json.pending").is_file()
    assert not (root/"FORECAST_SEAL.json").exists()


@pytest.mark.parametrize("moment", ["after_native_verification", "after_final_link"])
@pytest.mark.parametrize("change", ["delete", "replace_identical", "rewrite"])
def test_new_plan_lease_survives_native_verification_and_final_link(package, monkeypatch, moment, change):
    root, _, _, _ = package
    plan_path = root/"FORECAST_SEAL.json.plan.json"
    def alter():
        original = plan_path.read_bytes()
        if change == "delete": plan_path.unlink()
        elif change == "replace_identical":
            replacement = root/"fabricated-replacement-plan.json"
            replacement.write_bytes(original)
            os.replace(replacement, plan_path)
        else: plan_path.write_bytes(original + b" ")
    if moment == "after_native_verification":
        original_verify = VerifiedForecastRelease.verify
        def changed_verify(_cls, *args, **kwargs):
            result = original_verify(*args, **kwargs)
            alter()
            return result
        monkeypatch.setattr(VerifiedForecastRelease, "verify", classmethod(changed_verify))
    else:
        from frontier import forecast_seal
        original_link = os.link
        def changed_link(*args, **kwargs):
            result = original_link(*args, **kwargs)
            alter()
            return result
        monkeypatch.setattr(forecast_seal.os, "link", changed_link)
    with pytest.raises(ValueError, match="Published seal plan (changed|disappeared)"):
        publish(package)
    if moment == "after_native_verification":
        assert not (root/"FORECAST_SEAL.json").exists()
        assert (root/".FORECAST_SEAL.json.pending").exists()


def test_inventory_configuration_digest_binds_recipe_prefix_and_context(package):
    _, _, plan, _ = package
    assert plan.expectation.representation_sha256 == "1"*64
    assert plan.expectation.source_sha256 == "a"*64
    assert plan.expectation.configuration_sha256 == plan.configuration_sha256
    # Even seed/order differences are explicit; this is not a single-config hash.
    reordered = replace(plan, entries=tuple(reversed(plan.entries)))
    assert reordered.configuration_sha256 != plan.configuration_sha256


def with_derivation(package):
    from frontier import calibration_view
    from frontier.calibration_view import CalibrationView
    from frontier.planner import screen_plan
    root, index, plan, readers = package
    stage = screen_plan()
    task = next(t for t in stage.scores_for("D2") if t.key.training_donors == ("D3", "D4")
                and t.key.configuration_id == "F0-V0")
    source_sha = hashlib.sha256(Path(calibration_view.__file__).read_bytes()).hexdigest()
    view = CalibrationView.for_inner(readers[0], plan=stage, selector="D2", task=task,
                                    adapter_source_sha256=source_sha)
    entry = replace(plan.entries[0], derivation_manifest=freeze_calibration_derivation(view))
    frozen = replace(plan, entries=(entry,))
    return (root, index, frozen, readers[:1]), view


def test_full_calibration_grid_is_sealed_from_one_parent_without_deriving_arrays(package, monkeypatch):
    from frontier.calibration_view import CalibrationView
    derived, view = with_derivation(package)
    root, _, plan, _ = derived
    monkeypatch.setattr(CalibrationView, "law", lambda *_a, **_k: pytest.fail("No numerical calibration"))
    monkeypatch.setattr(CalibrationView, "catalog", lambda *_a, **_k: pytest.fail("No numerical calibration"))
    release = publish(derived, derivations=(view,))
    record = json.loads((root/"FORECAST_SEAL.json.plan.json").read_text())
    manifest = record["configuration_inventory"]["entries"][0]["derivation_manifest"]
    assert len(manifest["recipes"]) == 24
    assert manifest["role"]["selector"] == "D2"
    assert manifest["role"]["task"]["validation_donor"] == "D1"
    assert len(release.expectation.forecast_ids) == 1
    assert release.expectation.configuration_sha256 == plan.configuration_sha256
    sealed = json.loads((root/"FORECAST_SEAL.json").read_text())
    assert len(sealed["forecasts"][0]["artifacts"]) == 7


@pytest.mark.parametrize("kind", ["missing_view", "changed_inventory", "foreign_parent", "unplanned"])
def test_derivation_custody_is_required_not_inferred_from_raw_bank(package, kind):
    derived, view = with_derivation(package)
    root, _, plan, _ = derived
    with pytest.raises(ValueError):
        if kind == "missing_view": publish(derived)
        elif kind == "unplanned": publish(package, derivations=(view, None))
        else:
            record = json.loads(plan.entries[0].derivation_manifest)
            if kind == "foreign_parent": record["parent_publication"]["complete_sha256"] = "0"*64
            else: record["recipes"] = record["recipes"][:1]
            raw = (json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()
            changed = replace(plan, entries=(replace(plan.entries[0], derivation_manifest=raw),))
            publish(derived, plan=changed, expectation=changed.expectation, derivations=(view,))
    assert not (root/"FORECAST_SEAL.json").exists()


@pytest.fixture
def ensemble_package(native_fixture, tmp_path):
    from frontier import calibration_view, ensemble_view
    from frontier.calibration_view import CalibrationView
    from frontier.ensemble_view import EnsembleView
    from frontier.planner import ScoreTask, TrajectoryKey
    from frontier.selector_contract import OuterRecipe
    index = native_fixture[3]
    early = source(index, fold=FoldInputs("subset-D2-D3-D4", ("D2", "D3", "D4")))
    root = tmp_path/"outer-banks"; root.mkdir()
    entries = tuple(fabricated_bank(root/f"bank-{seed}", early, seed, updates=9216,
                    prefixes=(16, 23), ensemble_totals=(16, 23)) for seed in (0, 1, 2))
    frozen = OuterRecipe("D1", early.fold.training_donors, "F0-V0", .25, 1.3, "c"*64, "d"*64, .8)
    with ExitStack() as stack:
        readers = tuple(stack.enter_context(ForecastReader(root/e.bank_relative_root, spec=e.spec,
            expected_complete_sha256=e.expected_complete_sha256, expected_manifest_sha256=e.expected_manifest_sha256,
            maximum_output_bytes=128*1024**2, maximum_working_bytes=128*1024**2)) for e in entries)
        views = tuple(CalibrationView.for_outer(reader, frozen=frozen,
            task=ScoreTask(TrajectoryKey(early.fold.training_donors, "F0-V0", seed), "D1", "D1", 9216, "outer"),
            adapter_source_sha256=hashlib.sha256(Path(calibration_view.__file__).read_bytes()).hexdigest())
            for seed, reader in enumerate(readers))
        ensemble = EnsembleView.bind(views, totals=(16, 23), maximum_working_bytes=128*1024**2,
            adapter_source_sha256=hashlib.sha256(Path(ensemble_view.__file__).read_bytes()).hexdigest())
        entries = tuple(replace(e, derivation_manifest=freeze_calibration_derivation(v))
                        for e, v in zip(entries, views))
        plan = ForecastSealPlan(early.fold, "D1", early.reference.source_access_sha256, entries,
                                (freeze_ensemble_derivation(ensemble),))
        yield (root, index, plan, readers), views, ensemble, frozen


def publish_ensemble(package, **changes):
    base, views, ensemble, _ = package
    args = dict(derivations=views, ensembles=(ensemble,), output_cap_bytes=128*1024)
    args.update(changes)
    return publish(base, **args)


def test_empty_ensemble_inventory_preserves_original_configuration_shape_and_hash(package):
    _, _, plan, _ = package
    before = plan.configuration_inventory()
    assert set(before) == {"schema", "entries"}
    explicit = replace(plan, ensemble_derivations=())
    assert explicit.record() == plan.record()
    assert explicit.configuration_sha256 == _hash(before)
    assert explicit.expectation == plan.expectation


def test_exact_ensemble_manifest_is_sealed_without_duplicate_bank_or_particle_writes(ensemble_package, monkeypatch):
    from frontier.calibration_view import CalibrationView
    from frontier.ensemble_view import EnsembleView
    base, views, ensemble, _ = ensemble_package
    root, index, plan, readers = base
    before = {str(p.relative_to(root)): (p.stat().st_ino, hashlib.sha256(p.read_bytes()).hexdigest())
              for r in readers for p in r.root.iterdir()}
    for cls in (CalibrationView, EnsembleView):
        for name in ("law", "catalog"):
            monkeypatch.setattr(cls, name, lambda *_a, **_k: pytest.fail("No numerical derivation during seal"))
    monkeypatch.setattr(index, "open", lambda *_a, **_k: pytest.fail("No expression/outcome handles"))
    release = publish_ensemble(ensemble_package)
    record = json.loads((root/"FORECAST_SEAL.json.plan.json").read_bytes())
    inventory = record["configuration_inventory"]
    assert inventory["ensemble_derivations"] == [ensemble.derivation_manifest]
    assert _hash(inventory) == release.expectation.configuration_sha256 == plan.configuration_sha256
    assert plan.configuration_sha256 != replace(plan, ensemble_derivations=()).configuration_sha256
    sealed = json.loads((root/"FORECAST_SEAL.json").read_bytes())
    assert [len(row["artifacts"]) for row in sealed["forecasts"]] == [7, 6, 6]
    artifacts = [a["relative_uri"] for row in sealed["forecasts"] for a in row["artifacts"]]
    assert len(artifacts) == len(set(artifacts)) == 19
    assert len(release.expectation.model_sha256s) == len(release.expectation.forecast_ids) == 3
    assert {p.name for p in root.iterdir()} == {"bank-0", "bank-1", "bank-2", "FORECAST_SEAL.json", "FORECAST_SEAL.json.plan.json"}
    for name, identity in before.items():
        path = root/name
        assert (path.stat().st_ino, hashlib.sha256(path.read_bytes()).hexdigest()) == identity
    assert index.access(plan.fold, "evaluation_truth", query_donor="D1", release=release).source_ids == ("D1_Stim48hr",)


@pytest.mark.parametrize("kind", ["missing", "unplanned", "duplicate_live", "list", "foreign_type", "changed_manifest"])
def test_live_ensemble_inventory_must_exactly_match_prebound_plan(ensemble_package, kind):
    base, views, ensemble, _ = ensemble_package
    root, _, plan, _ = base
    changes = {}
    if kind == "missing": changes["ensembles"] = ()
    elif kind == "unplanned":
        changed = replace(plan, ensemble_derivations=())
        changes.update(plan=changed, expectation=changed.expectation)
    elif kind == "duplicate_live": changes["ensembles"] = (ensemble, ensemble)
    elif kind == "list": changes["ensembles"] = [ensemble]
    elif kind == "foreign_type": changes["ensembles"] = (views[0],)
    else:
        from frontier import ensemble_view
        from frontier.ensemble_view import EnsembleView
        # Same scientific recipe but different bound host allowance is not the
        # pre-frozen derivation, even though both allowances are admissible.
        changed = EnsembleView.bind(views, totals=(16, 23), maximum_working_bytes=127*1024**2,
            adapter_source_sha256=hashlib.sha256(Path(ensemble_view.__file__).read_bytes()).hexdigest())
        changes["ensembles"] = (changed,)
    with pytest.raises((TypeError, ValueError)):
        publish_ensemble(ensemble_package, **changes)
    assert not (root/"FORECAST_SEAL.json").exists()
    assert not (root/".FORECAST_SEAL.json.pending").exists()


@pytest.mark.parametrize("kind", ["parent_absent", "parent_unbound", "parent_hash", "seed_order", "frozen_recipe",
    "recipe", "model", "total", "counts", "source", "inner", "duplicate", "noncanonical", "list"])
def test_ensemble_plan_rejects_incomplete_or_foreign_metadata(ensemble_package, kind):
    from frontier.cache_storage import _json_bytes
    base, _, _, _ = ensemble_package
    _, _, plan, _ = base
    record = json.loads(plan.ensemble_derivations[0])
    with pytest.raises(ValueError):
        if kind == "parent_absent": replace(plan, entries=plan.entries[:2])
        elif kind == "parent_unbound": replace(plan, entries=(replace(plan.entries[0], derivation_manifest=None), *plan.entries[1:]))
        elif kind == "duplicate": replace(plan, ensemble_derivations=plan.ensemble_derivations*2)
        elif kind == "noncanonical": replace(plan, ensemble_derivations=(plan.ensemble_derivations[0]+b" ",))
        elif kind == "list": replace(plan, ensemble_derivations=list(plan.ensemble_derivations))
        else:
            if kind == "parent_hash": record["component_derivation_sha256"][0] = "0"*64
            elif kind == "seed_order": record["component_derivation_sha256"].reverse()
            elif kind == "frozen_recipe": record["frozen_outer_recipe"]["selection_evidence_identity"] = "e"*64
            elif kind == "recipe": record["recipe"]["alpha"] = .5
            elif kind == "model": record["model_bindings"][0]["model_seed"] = 2
            elif kind == "total": record["totals"] = [17, 23]
            elif kind == "counts": record["component_counts"]["16"] = [5, 5, 6]
            elif kind == "source": record["source_identity"] = "0"*64
            else: record["status"] = "inner"
            replace(plan, ensemble_derivations=(_json_bytes(record),))


@pytest.mark.parametrize("replace_reader", [False, True])
def test_same_manifest_is_not_same_live_ensemble_parent_custody(ensemble_package, replace_reader):
    from frontier import calibration_view
    from frontier.calibration_view import CalibrationView
    from frontier.planner import ScoreTask, TrajectoryKey
    base, views, _, frozen = ensemble_package
    root, _, plan, readers = base
    with ExitStack() as stack:
        admitted = readers[0]
        if replace_reader:
            entry = plan.entries[0]
            admitted = stack.enter_context(ForecastReader(readers[0].root, spec=entry.spec,
                expected_complete_sha256=entry.expected_complete_sha256, expected_manifest_sha256=entry.expected_manifest_sha256,
                maximum_output_bytes=128*1024**2, maximum_working_bytes=128*1024**2))
        other = CalibrationView.for_outer(admitted, frozen=frozen,
            task=ScoreTask(TrajectoryKey(frozen.fitting_donors, "F0-V0", 0), "D1", "D1", 9216, "outer"),
            adapter_source_sha256=hashlib.sha256(Path(calibration_view.__file__).read_bytes()).hexdigest())
        assert freeze_calibration_derivation(other) == plan.entries[0].derivation_manifest
        with pytest.raises(ValueError, match="same live calibration"):
            publish_ensemble(ensemble_package, readers=(admitted, *readers[1:]), derivations=(other, *views[1:]))
    assert not (root/"FORECAST_SEAL.json").exists()


def test_ensemble_parent_closure_during_native_verification_never_releases(ensemble_package, monkeypatch):
    base, _, _, _ = ensemble_package
    root, _, _, readers = base
    original = VerifiedForecastRelease.verify
    def after_verify(_cls, *args, **kwargs):
        result = original(*args, **kwargs)
        readers[0].close()
        return result
    monkeypatch.setattr(VerifiedForecastRelease, "verify", classmethod(after_verify))
    with pytest.raises((ValueError, RuntimeError)):
        publish_ensemble(ensemble_package)
    assert not (root/"FORECAST_SEAL.json").exists()
    assert (root/".FORECAST_SEAL.json.pending").exists()
