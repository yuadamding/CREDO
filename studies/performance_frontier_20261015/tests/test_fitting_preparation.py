"""Tiny fabricated native files and selected exports; zero optimizer steps.

No real input, learned historical model, GPU, remote owner or scheduler access.
The native/observation helpers create fabricated fixtures, not actual fit claims.
"""

from dataclasses import replace
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from frontier import fitting_preparation as module
from frontier.cache_compiler import CompilerLimits
from frontier.fitting_preparation import (FittingPopulation, PreparationLimits,
    PreparedFittingCache, open_prepared_cache, prepare_fitting_cache)
from frontier.fitting_provider import ProviderLimits
from frontier.native_inputs import FoldInputs
from frontier.observation import ObservationFold
from frontier.observation_training import ObservationTrainingSpec
from frontier.training_owner import observation_state_digest
from test_native_inputs import fixture as native_fixture
from test_observation_loader import fixture as observation_fixture


MIB = 1024**2
OUTPUT_CAP = 96*MIB


@pytest.fixture
def package(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    for cls in (torch.optim.Adam, torch.optim.AdamW):
        monkeypatch.setattr(cls, "step", lambda *_a, **_k: pytest.fail("Optimizer step forbidden"))
    root, _, _, index = native_fixture.__wrapped__(tmp_path/"native")
    fold = FoldInputs("subset-D3-D4", ("D3", "D4"))
    specification = replace(ObservationTrainingSpec.fixture(), features=3, latent_dim=48)
    admission, _ = observation_fixture(tmp_path/"observation", specification=specification,
        observation_fold=ObservationFold(fold.fold_id, fold.training_donors, index.rna_axis_sha256),
        input_sha256=index.access(fold, "representation_fit").identity())
    def handle(role, **kw):
        return index.open(root, fold, role, max_batch_rows=3, max_uncompressed_bytes=MIB,
                          max_cached_bytes=MIB, max_output_bytes=64*1024, **kw)
    handles = tuple(handle(role) for role in module.ROLES)
    args = dict(observation_admission=admission,
        populations=tuple(FittingPopulation(d, c, d+"_"+c, "aggregate-culture-unresolved")
                          for d in fold.training_donors for c in ("Stim8hr", "Stim48hr")),
        limits=PreparationLimits(CompilerLimits(8*MIB, 3, 64*1024, 2, 64*1024),
            ProviderLimits(8*MIB, 8*MIB, 64), 64*MIB, 256*1024, 3))
    return root, index, handles, args, handle


def prepare(package, **change):
    _, _, handles, args, _ = package
    return prepare_fitting_cache(handles, **(args | change))


def publish(package, tmp_path):
    owner = prepare(package)
    result = owner.publish(tmp_path/"prepared", output_cap_bytes=OUTPUT_CAP)
    return owner, result


def opened(result, owner):
    return open_prepared_cache(result["root"], expected_complete_sha256=result["complete_sha256"],
        output_cap_bytes=OUTPUT_CAP, limits=owner.cache.limits)


def test_complete_fit_only_cache_reads_each_native_row_once_and_keeps_all_support(package, monkeypatch):
    _, _, handles, args, _ = package
    seen = []
    for handle in handles:
        original = handle.read_rows
        def capture(addresses, original=original):
            seen.extend(addresses)
            return original(addresses)
        monkeypatch.setattr(handle, "read_rows", capture)
    before = torch.get_rng_state().clone()
    owner = prepare(package)
    assert torch.equal(before, torch.get_rng_state())
    assert seen == [(p.source_id, 0, r) for p in args["populations"] for r in range(3)]
    assert len(seen) == 12 and not any("Rest" in a[0] or a[0].startswith(("D1", "D2")) for a in seen)
    assert owner.cache.manifest.fold.training_donors == ("D3", "D4")
    for population in owner.cache.populations.values():
        assert population.captured_counts.tolist() == [2, 1, 0]
        assert population.positive_counts.tolist() == [1, 1, 0]
        assert population.states.shape == (2, 48)
        np.testing.assert_allclose(population.mean_compositions[0], [1/3, 0, 2/3])
        assert np.all(population.mean_compositions[2] == 0)
        assert not population.states.flags.writeable
    receipt = owner.receipt
    assert receipt["optimizer_calls"] == receipt["scientific_updates"] == 0
    assert receipt["held_out_donor_payloads_opened"] is False
    assert receipt["publication_claimed"] is False and len(receipt["compilations"]) == 4
    assert all(e["status"] == "unchanged" for e in receipt["input_exit"])
    assert receipt["estimate"]["conservative_host_estimate_bytes"] <= args["limits"].maximum_host_bytes
    assert receipt["actual_cache_retained_bytes"] == owner.cache.host_bytes
    assert observation_state_digest(owner.observation) == owner.cache.manifest.observation_binding.state_sha256


def test_cache_parent_completion_has_input_exits_and_full_readback(package, tmp_path):
    owner, result = publish(package, tmp_path)
    root = Path(result["root"])
    assert set(p.name for p in root.iterdir()) == {"cache", "PREPARATION.json", "COMPLETE.json"}
    record = json.loads((root/"PREPARATION.json").read_text())
    assert record["cache_verification"]["exit_full_hash_verified"] is True
    assert record["loss_normalization_performed"] is record["main_fitting_performed"] is False
    assert record["cache_publication"]["array_files"] == 20
    assert result["total_bytes"] == sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    with opened(result, owner) as mapped:
        assert mapped.manifest == owner.cache.manifest
        for key, original in owner.cache.populations.items():
            np.testing.assert_array_equal(mapped.populations[key].states, original.states)
        mapped.verify_decoder(owner.observation, owner.cache.manifest.observation_binding)
    assert mapped.verification["exit_full_hash_verified"] is True
    with pytest.raises(ValueError, match="One preparation"):
        owner.publish(tmp_path/"second", output_cap_bytes=OUTPUT_CAP)


@pytest.mark.parametrize("kind", ["missing", "reordered", "duplicate", "foreign_source"])
def test_population_inventory_rejected_before_selected_model_or_payload(package, monkeypatch, kind):
    _, _, _, args, _ = package
    rows = args["populations"]
    rows = {"missing": rows[:-1], "reordered": rows[::-1], "duplicate": rows[:-1]+rows[:1],
            "foreign_source": (replace(rows[0], source_id="D1_Stim8hr"), *rows[1:])}[kind]
    monkeypatch.setattr(module, "load_observation", lambda *_: pytest.fail("No model load for wrong inventory"))
    with pytest.raises(ValueError): prepare(package, populations=rows)


@pytest.mark.parametrize("kind", ["query", "swapped", "wrong_observation_role"])
def test_foreign_handles_and_observation_input_rejected_before_payload(package, monkeypatch, kind):
    _, _, handles, args, make_handle = package
    if kind == "query": handles = (make_handle("query", query_donor="D1"), handles[1])
    elif kind == "swapped": handles = handles[::-1]
    else:
        identity = dict(args["observation_admission"].trajectory_identity, input_sha256="e"*64)
        args = dict(args, observation_admission=replace(args["observation_admission"], trajectory_identity=identity))
    monkeypatch.setattr(module, "load_observation", lambda *_: pytest.fail("No model load for wrong role"))
    with pytest.raises(ValueError): prepare_fitting_cache(handles, **args)


def test_aggregate_budget_checked_before_any_model_or_row_load(package, monkeypatch):
    _, _, _, args, _ = package
    limit = replace(args["limits"], maximum_host_bytes=8*MIB)
    monkeypatch.setattr(module, "load_observation", lambda *_: pytest.fail("No model before memory admission"))
    with pytest.raises(ValueError, match="Aggregate"):
        prepare(package, limits=limit)


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "meta"])
def test_fabricated_work_cannot_run_on_gpu_or_implicit_device(package, monkeypatch, device):
    monkeypatch.setattr(module, "load_observation", lambda *_: pytest.fail("No model load for forbidden device"))
    with pytest.raises(ValueError, match="destination"):
        prepare(package, device=device)


def test_input_lease_metadata_budget_precedes_count_payload(package, monkeypatch):
    _, _, handles, args, _ = package
    for handle in handles:
        monkeypatch.setattr(handle, "read_rows", lambda *_: pytest.fail("No count access before metadata admission"))
    with pytest.raises(ValueError, match="lease evidence"):
        prepare(package, limits=replace(args["limits"], maximum_metadata_bytes=128))


def test_all_fitting_input_leases_cover_the_whole_compile(package, monkeypatch):
    root, _, handles, _, _ = package
    original = handles[0].read_rows
    def changed(addresses):
        batch = original(addresses)
        (root/"source/D4_Stim48hr/population_accounting.parquet").touch()
        return batch
    monkeypatch.setattr(handles[0], "read_rows", changed)
    with pytest.raises(ValueError, match="publication changed"):
        prepare(package)


@pytest.mark.parametrize("kind", ["input", "model", "reader"])
def test_mutated_prepared_dependency_rejects_before_output_creation(package, tmp_path, kind):
    root, _, handles, _, _ = package
    owner = prepare(package)
    if kind == "input": (root/"source/D3_Stim8hr/population_accounting.parquet").touch()
    elif kind == "model":
        with torch.no_grad(): owner.observation.scale.add_(1)
    else: handles[0]._reader.max_batch_rows += 1
    with pytest.raises(ValueError): owner.publish(tmp_path/"prepared", output_cap_bytes=OUTPUT_CAP)
    assert not (tmp_path/"prepared").exists()


def test_selected_artifact_change_cannot_be_hidden_by_retained_encoder(package, tmp_path):
    _, _, _, args, _ = package
    owner = prepare(package)
    path = Path(args["observation_admission"].artifact.path)
    data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        owner.publish(tmp_path/"prepared", output_cap_bytes=OUTPUT_CAP)
    assert (tmp_path/"prepared/FAILED.json").exists()
    assert not (tmp_path/"prepared/COMPLETE.json").exists()
    with pytest.raises(ValueError, match="One preparation"):
        owner.publish(tmp_path/"retry", output_cap_bytes=OUTPUT_CAP)


@pytest.mark.parametrize("late", [False, True])
def test_failed_input_exit_cannot_leave_an_accepted_cache(package, tmp_path, monkeypatch, late):
    raw, _, _, _, _ = package
    owner = prepare(package)
    target = raw/"source/D3_Stim8hr/population_accounting.parquet"
    if late:
        original = module._write_once
        def changed(fd, name, *a, **kw):
            result = original(fd, name, *a, **kw)
            if name == ".COMPLETE.pending": target.touch()
            return result
        monkeypatch.setattr(module, "_write_once", changed)
    else:
        original = module.publish_fold_cache
        def changed(*a, **kw):
            result = original(*a, **kw); target.touch(); return result
        monkeypatch.setattr(module, "publish_fold_cache", changed)
    with pytest.raises(ValueError, match="publication changed"):
        owner.publish(tmp_path/"prepared", output_cap_bytes=OUTPUT_CAP)
    root = tmp_path/"prepared"
    assert (root/"cache/COMPLETE.json").exists()
    assert (root/"FAILED.json").exists()
    assert (root/"COMPLETE.json").exists() is late
    with pytest.raises(ValueError, match="inventory"):
        with open_prepared_cache(root, expected_complete_sha256="a"*64,
                output_cap_bytes=OUTPUT_CAP, limits=owner.cache.limits):
            pytest.fail("Failed owner must not be admitted")


def test_output_cap_and_existing_root_fail_without_clobber(package, tmp_path):
    owner = prepare(package)
    with pytest.raises(ValueError, match="output cap"):
        owner.publish(tmp_path/"prepared", output_cap_bytes=MIB)
    assert not (tmp_path/"prepared").exists()
    root = tmp_path/"existing"; root.mkdir(); (root/"user-file").write_text("keep")
    with pytest.raises(FileExistsError): owner.publish(root, output_cap_bytes=OUTPUT_CAP)
    assert (root/"user-file").read_text() == "keep"


def test_anonymous_cache_cannot_be_called_an_accepted_preparation():
    with pytest.raises(TypeError, match="prepare_fitting_cache"):
        PreparedFittingCache(cache=None, observation=None, handles=None, snapshots=None,
                             admission=None, receipt=None, limits=None)


def test_corrupt_cache_array_rejected_before_consumer(package, tmp_path):
    owner, result = publish(package, tmp_path)
    path = Path(result["root"])/"cache/p000.states.npy"
    data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
    with pytest.raises(ValueError, match="SHA256"):
        with opened(result, owner): pytest.fail("Corrupt arrays must not be yielded")


@pytest.mark.parametrize("raise_in_body", [False, True])
def test_metadata_descriptor_lease_checked_even_when_consumer_raises(package, tmp_path, raise_in_body):
    owner, result = publish(package, tmp_path)
    with pytest.raises(ValueError, match="metadata changed"):
        with opened(result, owner):
            path = Path(result["root"])/"PREPARATION.json"
            replacement = path.with_name("replacement.json")
            replacement.write_bytes(path.read_bytes()); os.replace(replacement, path)
            if raise_in_body: raise RuntimeError("consumer failure")
