"""Synthetic-only custody and budget tests; no historical array reads."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import d2_forecast_reuse as reuse


def descriptor(path):
    raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(reuse.canonical(value))
    return descriptor(path)


def evidence(value):
    raw = reuse.canonical(value)
    return dict(raw_utf8=raw.decode(), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    old = tmp_path / "old"
    old.mkdir()
    monkeypatch.setattr(reuse, "PRIOR_ROOT", old)
    metadata = json.loads((ROOT / "d2_forecast_v3/REUSE_METADATA_20261010.json").read_text())
    rec = json.loads((ROOT / "d2_forecast_v3/RECOVERY_RECONCILIATION_20261010.json").read_text())
    manifest = reuse.build_reuse_manifest(rec, metadata)
    manifest["output_root"] = str(old / "output")
    for name in reuse.PRODUCER_HASHES:
        path = old / "bundle" / name
        path.parent.mkdir(exist_ok=True)
        shutil.copyfile(ROOT / "d2_forecast_v3/bundle" / name, path)
        manifest["producer_files"][name] = descriptor(path)
    output = old / "output"
    output.mkdir()
    startup = json.loads(manifest["evidence"]["startup"]["raw_utf8"])
    startup["writer_uid"] = os.getuid()
    manifest["evidence"]["startup"] = evidence(startup)
    release = json.loads(manifest["evidence"]["startup_release"]["raw_utf8"])
    release["startup_sha256"] = manifest["evidence"]["startup"]["sha256"]
    manifest["evidence"]["startup_release"] = evidence(release)
    for name, key in (("FAILURE.json", "failure"), ("STARTUP.json", "startup")):
        (output / name).write_text(manifest["evidence"][key]["raw_utf8"])
    source = {key: rec["banks"][0]["metadata"]["bank"]["value"][key] for key in reuse.SOURCE_KEYS}
    ids, indices, checkpoints, banks = ["D2::0", "D2::1"], [0, 1], [], []
    for index, record in enumerate(rec["banks"]):
        label = "ddr_mlp128_e8--seed-" + str(index)
        directory = output / "banks" / label
        directory.mkdir(parents=True)
        checkpoint = tmp_path / (label + ".pt")
        checkpoint.write_bytes(("synthetic frozen checkpoint " + str(index)).encode())
        binding = dict(descriptor(checkpoint), checkpoint_path=str(checkpoint),
                       candidate_id="ddr_mlp128_e8", seed=index, parameter_count=257427)
        checkpoints.append(binding)
        array = directory / "forecast.h5"
        # The real producer performed shape checks. Reuse streams byte identities;
        # this stand-in intentionally requires no HDF5 or torch dependency.
        array.write_bytes(("synthetic full immutable forecast " + str(index)).encode())
        array_id = descriptor(array)
        bank = copy.deepcopy(record["metadata"]["bank"]["value"])
        bank.update(modeled_measure_ids=ids, modeled_guide_indices=indices,
                    checkpoint_sha256=binding["sha256"],
                    artifacts={"forecast.h5": dict(sha256=array_id["sha256"], bytes=array_id["size_bytes"])})
        bank_id = write_json(directory / "BANK.json", bank)
        loader = copy.deepcopy(record["metadata"]["loader"]["value"])
        loader["checkpoint_sha256"] = binding["sha256"]
        loader_id = write_json(output / (label + "--loader.json"), loader)
        qid = write_json(output / (label + "--qualification.json"), record["metadata"]["qualification"]["value"])
        banks.append(dict(candidate_id="ddr_mlp128_e8", seed=index, bank=bank_id,
                          forecast=array_id, loader=loader_id, qualification=qid))
    manifest["banks"] = banks
    sizes = [dict(path=p.relative_to(output).as_posix(), size_bytes=p.stat().st_size)
             for p in sorted(output.rglob("*")) if p.is_file()]
    manifest.update(output_inventory_sizes=sizes, retained_output_bytes=sum(r["size_bytes"] for r in sizes))
    args = dict(source_identity=source, measure_ids=ids, guide_indices=indices,
                checkpoints=checkpoints, owner=os.getuid())
    return manifest, args, tmp_path


def test_readonly_requalification_and_new_seals(fixture):
    manifest, args, tmp_path = fixture
    before = {p: descriptor(p) for p in reuse.PRIOR_ROOT.rglob("*") if p.is_file()}
    result = reuse.validate_reuse(manifest, **args)
    assert result["retained_bytes"] == manifest["retained_output_bytes"]
    assert result["old_run_status"] == "failed"
    assert len(result["banks"]) == 3
    fresh = tmp_path / "new"
    fresh.mkdir()
    rows = reuse.publish_reuse_seals(fresh, manifest, result, owner=os.getuid(), fresh_lease_exit_verified=True)
    assert all(row["storage"] == "external_reused" and row["seal"] == row["reuse_seal"] for row in rows)
    for row in rows:
        seal = json.loads((fresh / row["reuse_seal_path"]).read_text())
        assert seal == reuse.expected_reuse_seal(manifest, result, row)
        assert not seal["historical_positive_lease_receipt_present"]
        assert not seal["historical_run_relabelled_successful"]
        assert "inference" in seal["historical_lease_exit_basis"]
    assert before == {p: descriptor(p) for p in reuse.PRIOR_ROOT.rglob("*") if p.is_file()}


@pytest.mark.parametrize("field", ["prior_job_uid", "prior_plan_sha256", "prior_implementation_sha256", "prior_run_id"])
def test_wrong_producer(fixture, field):
    manifest, args, _ = fixture
    manifest[field] = "wrong"
    with pytest.raises(ValueError, match="producer"):
        reuse.validate_reuse(manifest, **args)


@pytest.mark.parametrize("kind", ["bank", "forecast", "loader", "qualification"])
def test_mutated_original(fixture, kind):
    manifest, args, _ = fixture
    path = Path(manifest["banks"][0][kind]["path"])
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="identity"):
        reuse.validate_reuse(manifest, **args)


@pytest.mark.parametrize("kind", ["source", "membership", "checkpoint", "qualification", "bank", "failure", "cleanup"])
def test_wrong_scientific_or_lifecycle_binding(fixture, kind):
    manifest, args, _ = fixture
    if kind == "source":
        args["source_identity"]["source_counts_sha256"] = "a" * 64
    elif kind == "membership":
        args["measure_ids"] = ["D2::1", "D2::0"]
    elif kind == "checkpoint":
        Path(args["checkpoints"][0]["checkpoint_path"]).write_bytes(b"changed")
    elif kind in {"qualification", "bank"}:
        desc = manifest["banks"][0][kind]
        value = json.loads(Path(desc["path"]).read_text())
        value["parameter_updates"] = 1
        manifest["banks"][0][kind] = write_json(Path(desc["path"]), value)
    else:
        value = json.loads(manifest["evidence"][kind]["raw_utf8"])
        value["status"] = "passed"
        manifest["evidence"][kind] = evidence(value)
    with pytest.raises(ValueError):
        reuse.validate_reuse(manifest, **args)


def test_cap_includes_all_retained_output(fixture):
    manifest, args, _ = fixture
    with pytest.raises(ValueError, match="360-GiB"):
        reuse.validate_reuse(manifest, maximum_new_bytes=360 * reuse.GIB, **args)
    result = reuse.validate_reuse(manifest, maximum_new_bytes=360 * reuse.GIB - manifest["retained_output_bytes"], **args)
    assert result["retained_bytes"] > sum(r["forecast"]["size_bytes"] for r in manifest["banks"])


def test_unexpected_retained_output(fixture):
    manifest, args, _ = fixture
    (reuse.PRIOR_ROOT / "output" / "unregistered").write_bytes(b"new")
    with pytest.raises(ValueError, match="inventory changed"):
        reuse.validate_reuse(manifest, **args)


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "owner"])
def test_unsafe_input(fixture, kind):
    manifest, args, tmp_path = fixture
    path = Path(manifest["banks"][0]["forecast"]["path"])
    if kind == "owner":
        args["owner"] += 1
    elif kind == "hardlink":
        os.link(str(path), str(tmp_path / "alias"))
    else:
        original = tmp_path / "original"
        path.rename(original)
        path.symlink_to(original)
    with pytest.raises(ValueError):
        reuse.validate_reuse(manifest, **args)


def test_no_seal_without_fresh_lease_or_after_input_change(fixture):
    manifest, args, tmp_path = fixture
    result = reuse.validate_reuse(manifest, **args)
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    with pytest.raises(ValueError, match="fresh requalification"):
        reuse.publish_reuse_seals(fresh, manifest, result, owner=os.getuid())
    path = Path(manifest["banks"][0]["forecast"]["path"])
    path.write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="changed before"):
        reuse.publish_reuse_seals(fresh, manifest, result, owner=os.getuid(), fresh_lease_exit_verified=True)
    assert not (fresh / "reuse-seals").exists()


def test_no_old_write_or_overwrite(fixture):
    manifest, args, tmp_path = fixture
    result = reuse.validate_reuse(manifest, **args)
    with pytest.raises(ValueError, match="historical"):
        reuse.publish_reuse_seals(reuse.PRIOR_ROOT / "output", manifest, result,
                                 owner=os.getuid(), fresh_lease_exit_verified=True)
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    reuse.publish_reuse_seals(fresh, manifest, result, owner=os.getuid(), fresh_lease_exit_verified=True)
    with pytest.raises(FileExistsError):
        reuse.publish_reuse_seals(fresh, manifest, result, owner=os.getuid(), fresh_lease_exit_verified=True)


def test_readonly_real_metadata_factory_and_python36_syntax():
    rec = json.loads((ROOT / "d2_forecast_v3/RECOVERY_RECONCILIATION_20261010.json").read_text())
    meta = json.loads((ROOT / "d2_forecast_v3/REUSE_METADATA_20261010.json").read_text())
    manifest = reuse.build_reuse_manifest(rec, meta)
    assert manifest["retained_output_bytes"] == 115243519144
    assert manifest["retained_output_bytes"] + 240 * reuse.GIB < 360 * reuse.GIB
    assert len(manifest["banks"]) == 3
    ast.parse((ROOT / "d2_forecast_reuse.py").read_text(), feature_version=(3, 6))
