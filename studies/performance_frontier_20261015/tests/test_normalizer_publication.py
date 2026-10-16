"""Native toy data -> actual selected export -> cache -> numerical scales.

Only tiny fabricated CPU inputs. No optimizer step, real data or GPU access.
Failure injection changes temporary test artifacts only.
"""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest
import torch

from frontier import normalizer_publication as module
from frontier.cache_compiler import CompilerLimits
from frontier.fitting_factory import bind_normalizers
from frontier.fitting_preparation import (FittingPopulation, PreparationLimits,
    open_prepared_cache, prepare_fitting_cache)
from frontier.fitting_provider import ProviderLimits
from frontier.native_inputs import FoldInputs
from frontier.normalizer_publication import (NormalizationLimits,
    load_normalizer_artifact, publish_normalizers)
from frontier.observation import ObservationFold
from frontier.observation_training import ObservationTrainingSpec
from test_native_inputs import native_package
from test_observation_loader import fixture as observation_fixture
from test_normalization import geometry, PROTOCOL


MIB = 1024**2
CACHE_CAP = 96*MIB


@pytest.fixture
def package(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    for cls in (torch.optim.Adam, torch.optim.AdamW):
        monkeypatch.setattr(cls, "step", lambda *_a, **_k: pytest.fail("Optimizer step forbidden"))
    raw, _, _, index = native_package(tmp_path/"native", temporal=True)
    fold = FoldInputs("subset-D3-D4", ("D3", "D4"))
    admission, _ = observation_fixture(tmp_path/"observation",
        specification=replace(ObservationTrainingSpec.fixture(), features=3, latent_dim=48),
        observation_fold=ObservationFold(fold.fold_id, fold.training_donors, index.rna_axis_sha256),
        input_sha256=index.access(fold, "representation_fit").identity())
    handles = tuple(index.open(raw, fold, role, max_batch_rows=3,
        max_uncompressed_bytes=MIB, max_cached_bytes=MIB, max_output_bytes=64*1024)
        for role in ("dynamics_source", "dynamics_supervision"))
    provider = ProviderLimits(8*MIB, 8*MIB, 64)
    owner = prepare_fitting_cache(handles, observation_admission=admission,
        populations=tuple(FittingPopulation(d, c, d+"_"+c, "fabricated-unresolved-culture")
            for d in fold.training_donors for c in ("Stim8hr", "Stim48hr")),
        limits=PreparationLimits(CompilerLimits(8*MIB, 3, 64*1024, 2, 64*1024),
                                 provider, 64*MIB, 256*1024, 3))
    prepared = owner.publish(tmp_path/"prepared", output_cap_bytes=CACHE_CAP)
    args = dict(preparation_complete_sha256=prepared["complete_sha256"],
        preparation_output_cap_bytes=CACHE_CAP, limits=NormalizationLimits(provider, 4, MIB, 4*MIB),
        geometry=geometry(), study_id="fabricated-normalizer-publication", study_seed=731,
        protocol_payload=PROTOCOL, source_particles=8, block=4)
    return Path(prepared["root"]), args


def publish(package, destination, **changes):
    root, args = package
    return publish_normalizers(root, destination, **(args | changes))


def load(result, **changes):
    return load_normalizer_artifact(result["root"], **(dict(
        expected_complete_sha256=result["complete_sha256"],
        expected_cache_manifest_sha256=result["cache_manifest_sha256"],
        maximum_artifact_bytes=MIB, output_cap_bytes=4*MIB) | changes))


def test_actual_preparation_normalizers_and_factory_binding(package, tmp_path, monkeypatch):
    calls = []
    original = module.normalization.compute_normalizers
    def counted(*a, **kw):
        calls.append(1)
        return original(*a, **kw)
    monkeypatch.setattr(module.normalization, "compute_normalizers", counted)
    rng = torch.get_rng_state().clone()
    result = publish(package, tmp_path/"normalizers")
    assert calls == [1] and torch.equal(rng, torch.get_rng_state())
    artifact = load(result)
    root, args = package
    with open_prepared_cache(root, expected_complete_sha256=args["preparation_complete_sha256"],
            output_cap_bytes=CACHE_CAP, limits=args["limits"].provider) as mapped:
        scales, selection, content = bind_normalizers(mapped, artifact, study_id=args["study_id"],
            study_seed=args["study_seed"], geometry=args["geometry"])
        assert content == result["normalizer_content_sha256"]
        assert len(selection) == 9 and all(v > 0 for v in selection.values())
    rows = json.loads(artifact.payload)
    assert len(rows["training_values"]) == 5 and all(v > 0 for v in rows["training_values"].values())
    assert len(rows["per_law"]) == 4 and rows["fitting_donors"] == ["D3", "D4"]
    assert [r["catalog_categories"] for r in rows["per_donor"]] == [3, 3]
    assert {r["guide_id"] for r in rows["per_law"]} == {"g0", "g1"}
    directory = Path(result["root"])
    evidence = json.loads((directory/"PUBLICATION.json").read_text())
    assert evidence["cache_verification"]["exit_full_hash_verified"] is True
    assert evidence["scalar_reconciliation_passed"] is True
    assert evidence["operations"]["observed_endpoint_atoms"] == 6
    assert evidence["operations"]["nominal_four_cost_matrix_elements"] == 362
    assert evidence["optimizer_steps"] == evidence["scientific_updates"] == 0
    assert evidence["query_outcomes_opened"] is False and evidence["synthetic_fixture"] is True
    assert result["total_bytes"] == sum(p.stat().st_size for p in directory.iterdir())
    assert calls == [1]  # Loading/reconciliation is not a numerical replay.


@pytest.mark.parametrize("change", ["protocol", "laws", "particles", "device", "geometry"])
def test_admission_failure_precedes_computation_and_output(package, tmp_path, monkeypatch, change):
    _, args = package
    changes = {
        "protocol": dict(protocol_payload=b"{}"),
        "laws": dict(limits=replace(args["limits"], maximum_paired_laws=3)),
        "particles": dict(source_particles=129),
        "device": dict(device="cuda:0"),
        "geometry": dict(geometry=type(args["geometry"])(options=replace(
            args["geometry"].options, maximum_geometry_elements=99))),
    }[change]
    monkeypatch.setattr(module.normalization, "compute_normalizers", lambda *_a, **_k: pytest.fail("No arithmetic"))
    with pytest.raises(ValueError): publish(package, tmp_path/"normalizers", **changes)
    assert not (tmp_path/"normalizers").exists()


@pytest.mark.parametrize("kind", ["cache", "metadata", "producer", "post_complete"])
def test_exit_failure_prevents_accepted_publication(package, tmp_path, monkeypatch, kind):
    root, _ = package
    injected = []
    if kind == "post_complete":
        original = module._put
        def changed(fd, name, raw):
            result = original(fd, name, raw)
            if name == ".COMPLETE.pending":
                injected.append(kind)
                (root/"PREPARATION.json").touch()
            return result
        monkeypatch.setattr(module, "_put", changed)
    else:
        original = module.normalization.compute_normalizers
        def changed(*a, **kw):
            result = original(*a, **kw)
            injected.append(kind)
            if kind == "cache": (root/"cache/p000.states.npy").touch()
            elif kind == "metadata": (root/"PREPARATION.json").touch()
            else: monkeypatch.setattr(module, "_sources", lambda: {"changed": "a"*64})
            return result
        monkeypatch.setattr(module.normalization, "compute_normalizers", changed)
    out = tmp_path/"normalizers"
    with pytest.raises(ValueError): publish(package, out)
    assert injected == [kind]
    assert (out/"FAILED.json").exists()
    assert (out/"COMPLETE.json").exists() is (kind == "post_complete")
    with pytest.raises(FileExistsError): publish(package, out)
    with pytest.raises(ValueError, match="inventory"):
        load_normalizer_artifact(out, expected_complete_sha256="a"*64,
            expected_cache_manifest_sha256="b"*64, maximum_artifact_bytes=MIB, output_cap_bytes=4*MIB)


def test_artifact_cap_preserves_failure_without_deleting_evidence_rows(package, tmp_path):
    _, args = package
    with pytest.raises(ValueError, match="artifact cap"):
        publish(package, tmp_path/"normalizers", limits=replace(args["limits"], maximum_artifact_bytes=1))
    assert (tmp_path/"normalizers/FAILED.json").exists()
    assert not (tmp_path/"normalizers/NORMALIZERS.json").exists()


@pytest.mark.parametrize("kind", ["data", "extra", "cache_identity", "cap", "receipt"])
def test_loader_rejects_corrupt_wrong_fold_or_unaccepted_outputs(package, tmp_path, kind):
    result = publish(package, tmp_path/"normalizers")
    out = Path(result["root"])
    changes = {}
    if kind == "data":
        path = out/"NORMALIZERS.json"; data = bytearray(path.read_bytes()); data[3] ^= 1; path.write_bytes(data)
    elif kind == "extra": (out/"FAILED.json").write_text("{}")
    elif kind == "cache_identity": changes["expected_cache_manifest_sha256"] = "b"*64
    elif kind == "cap": changes["maximum_artifact_bytes"] = 1
    else: (out/"PUBLICATION.json").write_text("{}")
    with pytest.raises(ValueError): load(result, **changes)


def test_no_overwrite_or_output_inside_input(package, tmp_path):
    root, _ = package
    with pytest.raises(ValueError, match="separate"):
        publish(package, root/"normalizers")
    out = tmp_path/"existing"; out.mkdir(); (out/"keep").write_text("user content")
    with pytest.raises(FileExistsError): publish(package, out)
    assert (out/"keep").read_text() == "user content"


def test_limits_preserve_publication_and_failure_reserve(package):
    _, args = package
    with pytest.raises(ValueError, match="allowances"):
        replace(args["limits"], output_cap_bytes=MIB)
    with pytest.raises(ValueError, match="allowances"):
        replace(args["limits"], maximum_artifact_bytes=257*MIB, output_cap_bytes=300*MIB)
