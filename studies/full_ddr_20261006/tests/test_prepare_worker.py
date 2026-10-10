"""Synthetic-only lifecycle regressions; never open cohort arrays or use CUDA."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_worker as worker
import prepare_launch as launch


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(worker.canonical(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_bundle(root):
    plan = {
        "run_id": "synthetic",
        "worker_sha256": hashlib.sha256(b"pass\n").hexdigest(),
        "legacy_source_root": "/bound/code",
        "legacy_source_files": {"x.py": "a" * 64},
    }
    (root / "worker.py").write_bytes(b"pass\n")
    plan["implementation_sha256"] = put(
        root / "IMPLEMENTATION.json", {"worker.py": plan["worker_sha256"]}
    )
    plan["legacy_source_binding_sha256"] = put(
        root / "LEGACY_SOURCE_BINDING.json",
        {"root": plan["legacy_source_root"], "files": plan["legacy_source_files"]},
    )
    digest = put(root / "PLAN.json", plan)
    put(
        root / "INVENTORY.json",
        {p.name: worker.file_identity(p) for p in root.iterdir()},
    )
    return plan, digest


def startup_fixture(output):
    startup = {"plan_sha256": "a" * 64, "pod_uid": "pod-123"}
    put(output / "STARTUP.json", startup)
    release = {
        "status": "startup_released",
        "plan_sha256": startup["plan_sha256"],
        "pod_uid": startup["pod_uid"],
        "startup_sha256": worker.file_identity(output / "STARTUP.json")["sha256"],
        "uid": "job-123",
        "actual_pod_audit_passed": True,
        "tests_permitted": True,
        "scientific_promotion": False,
        "actual_pod_sha256": "b" * 64,
    }
    return startup, release


def test_no_argument_cli_uses_bound_environment_and_sibling_plan(tmp_path, monkeypatch):
    plan, digest = synthetic_bundle(tmp_path)
    monkeypatch.setenv("CREDO_RUN_ID", plan["run_id"])
    monkeypatch.setenv("CREDO_PLAN_SHA256", digest)
    assert worker.load_plan([], root=tmp_path) == (tmp_path, plan, digest)
    with pytest.raises(RuntimeError, match="environment"):
        worker.load_plan(["--plan-sha256", "b" * 64], root=tmp_path)
    with pytest.raises(RuntimeError, match="sibling"):
        worker.load_plan(["--plan", str(tmp_path / "other.json")], root=tmp_path)


@pytest.mark.parametrize("mutation", ["extra", "bytes", "implementation", "symlink"])
def test_exact_bundle_membership_and_implementation(tmp_path, mutation):
    plan, _ = synthetic_bundle(tmp_path)
    worker.verify_bundle(tmp_path, plan)
    if mutation == "extra":
        (tmp_path / "extra").write_bytes(b"not bound")
    elif mutation == "bytes":
        (tmp_path / "worker.py").write_bytes(b"changed\n")
    elif mutation == "implementation":
        plan["implementation_sha256"] = "c" * 64
    else:
        (tmp_path / "link").symlink_to(tmp_path / "worker.py")
    with pytest.raises(RuntimeError):
        worker.verify_bundle(tmp_path, plan)


def test_release_polls_without_deadline_and_validates_exact_identity(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(worker, "WRITER", (os.getuid(), os.getgid()))
    startup, release = startup_fixture(tmp_path)
    polls = []

    def sleep(seconds):
        polls.append(seconds)
        if len(polls) == 4:
            put(tmp_path / "STARTUP_RELEASE.json", release)

    assert worker.wait_for_release(tmp_path, startup, sleep=sleep) == release
    assert polls == [2, 2, 2, 2]


@pytest.mark.parametrize(
    "field,value",
    [
        ("plan_sha256", "b" * 64),
        ("pod_uid", "wrong"),
        ("startup_sha256", "c" * 64),
        ("actual_pod_audit_passed", False),
        ("tests_permitted", False),
        ("scientific_promotion", True),
        ("uid", ""),
    ],
)
def test_wrong_release_cannot_start_work(tmp_path, monkeypatch, field, value):
    monkeypatch.setattr(worker, "WRITER", (os.getuid(), os.getgid()))
    startup, release = startup_fixture(tmp_path)
    release[field] = value
    put(tmp_path / "STARTUP_RELEASE.json", release)
    with pytest.raises(RuntimeError, match="startup release"):
        worker.wait_for_release(
            tmp_path, startup, sleep=lambda _: pytest.fail("Should not poll")
        )


def test_release_symlink_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "WRITER", (os.getuid(), os.getgid()))
    startup, release = startup_fixture(tmp_path)
    put(tmp_path / "another.json", release)
    (tmp_path / "STARTUP_RELEASE.json").symlink_to(tmp_path / "another.json")
    with pytest.raises(RuntimeError, match="regular"):
        worker.wait_for_release(tmp_path, startup)


def test_startup_matches_exact_controller_schema(monkeypatch):
    monkeypatch.setenv("CREDO_POD_NAME", "pod")
    monkeypatch.setenv("CREDO_POD_UID", "pod-123")
    plan = {"run_id": "run", "image": "pinned", "implementation_sha256": "c" * 64}
    record = worker.startup_record(
        plan,
        "a" * 64,
        SimpleNamespace(name="NVIDIA H100", total_memory=80_000_000_000),
        torch,
    )
    assert record["gpu"] == {
        "name": "NVIDIA H100",
        "count": 1,
        "total_memory_bytes": 80_000_000_000,
        "compute_capability": "9.0",
    }
    assert record["writer_uid"] == os.getuid()
    assert record["writer_gid"] == os.getgid()
    for key in (
        "bundle_inventory_verified",
        "writer_canary_removed",
        "required_imports_passed",
    ):
        assert record[key] is True
    assert record["tests_started"] is record["real_data_opened"] is False
    assert record["implementation_sha256"] == plan["implementation_sha256"]


def test_completion_full_inventory_budget_and_no_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "WRITER", (os.getuid(), os.getgid()))
    startup, _ = startup_fixture(tmp_path)
    put(tmp_path / "one" / "source.json", {"data": "synthetic"})
    results = [
        dict(source_id=s, rows=2, geometry_cells=1, completion_sha256="a" * 64)
        for s in worker.DEVELOPMENT_SOURCES
    ]
    plan = dict(
        run_id="run",
        expected_eligible_rows=8,
        maximum_output_bytes=16 * 1024**3,
        implementation_sha256="b" * 64,
    )
    result = worker.publish_completion(
        tmp_path,
        plan,
        "a" * 64,
        startup,
        results,
        elapsed_seconds=10000000.0,
        peak_bytes=0,
    )
    assert result["status"] == "completed_full_ddr_development_preparation"
    assert result["eligible_rows"] == 8 and result["parameter_updates"] == 0
    assert result["preparation_only"] and not result["D1_endpoint_access"]
    assert result["output_bytes"] == sum(row["size_bytes"] for row in result["files"])
    assert {row["path"] for row in result["files"]} == {
        "STARTUP.json",
        "one/source.json",
    }
    with pytest.raises(RuntimeError, match="overwritten"):
        worker.publish_completion(
            tmp_path,
            plan,
            "a" * 64,
            startup,
            results,
            elapsed_seconds=10000001.0,
            peak_bytes=0,
        )


def test_completion_cap_includes_complete_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "WRITER", (os.getuid(), os.getgid()))
    startup, _ = startup_fixture(tmp_path)
    results = [dict(source_id=s, rows=1) for s in worker.DEVELOPMENT_SOURCES]
    plan = dict(
        run_id="run",
        expected_eligible_rows=4,
        maximum_output_bytes=(tmp_path / "STARTUP.json").stat().st_size,
        implementation_sha256="b" * 64,
    )
    with pytest.raises(RuntimeError, match="cap"):
        worker.publish_completion(
            tmp_path, plan, "a" * 64, startup, results, elapsed_seconds=0, peak_bytes=0
        )
    assert not (tmp_path / "COMPLETE.json").exists()


def test_pinned_legacy_import_closure_and_dense_budget_are_actual_apis():
    # Separate interpreter avoids contamination by a moving installed v4 package.
    root = launch.R48_BUNDLE / "repo/src"
    closure = launch.legacy_source_closure()
    script = """
import json, sys
import prepare_worker
plan = json.loads(sys.argv[1])
network, contracts, reader, artifacts, canonical = prepare_worker.legacy_imports(plan)
rules = contracts.RepresentationRules(device='cpu', maximum_dense_working_bytes=2*1024**3)
network.dense_budget(rules, 18129, 2048)
try:
    network.dense_budget(rules, 18129, 1000000)
except RuntimeError:
    pass
else:
    raise AssertionError('Dense admission must reject over-budget batch')
assert all(hasattr(network, x) for x in ('initialize', 'dense_budget'))
assert hasattr(network.SparseCountAutoencoder, 'encode')
assert all(hasattr(reader.PreparedShardReader, x) for x in ('guide_catalog', 'read_rows'))
assert hasattr(artifacts, 'load_tensor_file')
from pathlib import Path
metadata_path = Path(prepare_worker.__file__).with_name('METADATA_PREPARATION.json')
metadata = json.loads(metadata_path.read_bytes())
plan.update(metadata_sha256=prepare_worker.sha_file(metadata_path),
            authority_sha256=prepare_worker.sha_file(metadata_path.with_name('AUTHORITY.json')))
accesses = [prepare_worker.make_access(plan, metadata, source, reader, canonical)
            for source in prepare_worker.DEVELOPMENT_SOURCES]
assert tuple(a.source_ids[0] for a in accesses) == prepare_worker.DEVELOPMENT_SOURCES
assert sum(s.rows for a in accesses for s in a.shards) == 7590115
assert all(s.source_id in prepare_worker.DEVELOPMENT_SOURCES and s.allowed_rows is None
           for a in accesses for s in a.shards)
assert all(a.role == 'dynamics_supervision' and len(a.identity()) == 64 for a in accesses)
print('pinned APIs and actual import closure passed')
"""
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            script,
            json.dumps(
                {"legacy_source_root": str(root), "legacy_source_files": closure}
            ),
        ],
        cwd=Path(worker.__file__).parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "actual import closure passed" in result.stdout


@pytest.mark.parametrize("failure", [None, "dense_budget", "release"])
def test_main_synthetic_forward_work_waits_for_release_and_publishes(
    tmp_path, monkeypatch, failure
):
    """Real writer with toy CPU encoder/reader; CUDA properties are mocked only."""
    events = []
    output, fitted = tmp_path / "output", tmp_path / "fitted"
    fitted.mkdir()
    features = [{"feature_id": "gene", "is_RNA": True}]
    metadata = {
        "canonical_features": features,
        "shards": [{"source_id": s, "rows": 2} for s in worker.DEVELOPMENT_SOURCES],
    }
    plan = dict(
        run_id="synthetic",
        bundle_root=str(tmp_path),
        output=str(output),
        metadata_path=str(tmp_path / "metadata.json"),
        metadata_sha256=put(tmp_path / "metadata.json", metadata),
        authority_path=str(tmp_path / "authority.json"),
        authority_sha256=put(tmp_path / "authority.json", {"synthetic": True}),
        fitted_root=str(fitted),
        native_root=str(tmp_path / "native"),
        batch_rows=2048,
        expected_eligible_rows=8,
        maximum_output_bytes=16 * 1024**3,
        implementation_sha256="a" * 64,
        image="synthetic",
    )
    monkeypatch.setattr(
        worker,
        "FITTED_SPEC_SHA",
        put(fitted / "specification.json", {"features": features, "rules": {}}),
    )
    monkeypatch.setattr(
        worker,
        "FITTED_COMPLETE_SHA",
        put(fitted / "COMPLETE.json", {"synthetic": True}),
    )
    (fitted / "model.safetensors").write_bytes(
        b"synthetic CPU fixture, not real tensors"
    )
    monkeypatch.setattr(
        worker, "MODEL_SHA", worker.sha_file(fitted / "model.safetensors")
    )
    monkeypatch.setattr(worker, "WRITER", (os.getuid(), os.getgid()))
    monkeypatch.setattr(worker, "load_plan", lambda _: (tmp_path, plan, "f" * 64))
    monkeypatch.setattr(worker, "admit", lambda _: None)
    monkeypatch.setattr(
        worker, "verify_bundle", lambda *args: events.append("verified")
    )
    monkeypatch.setenv("CREDO_POD_NAME", "synthetic-pod")
    monkeypatch.setenv("CREDO_POD_UID", "synthetic-pod-uid")
    for name, fn in {
        "is_available": lambda: True,
        "device_count": lambda: 1,
        "get_device_properties": lambda _: SimpleNamespace(
            name="H100 synthetic", total_memory=80_000_000_000
        ),
        "get_device_capability": lambda _: (9, 0),
        "max_memory_allocated": lambda: 0,
        "max_memory_reserved": lambda: 0,
    }.items():
        monkeypatch.setattr(torch.cuda, name, fn)
    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _: None)
    rules = SimpleNamespace(device="cuda", batch_rows=2048, latent_dim=1)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("scale", torch.ones(1))

        def encode(self, counts):
            events.append("encode")
            return torch.as_tensor(counts.toarray(), dtype=torch.float32)

    def initialize(*args):
        events.append("initialize")
        return Model()

    def dense_budget(*args):
        events.append("dense_budget")
        if failure == "dense_budget":
            raise RuntimeError("synthetic dense budget failure")

    modules = (
        SimpleNamespace(initialize=initialize, dense_budget=dense_budget),
        SimpleNamespace(
            RepresentationRules=SimpleNamespace(model_validate=lambda _: rules)
        ),
        None,
        SimpleNamespace(
            load_tensor_file=lambda *args, **kwargs: {"scale": torch.ones(1)}
        ),
        SimpleNamespace(contract_id=lambda _: worker.REPRESENTATION_ID),
    )
    guides = pd.DataFrame(
        {
            "guide_index": [0],
            "guide_id": ["g"],
            "target_id": ["t"],
            "is_control": [False],
        }
    )

    class Reader:
        def __init__(self, root, access, **kwargs):
            events.append("reader")
            self.source = access.source

        def guide_catalog(self):
            return guides

        def read_rows(self, addresses):
            return SimpleNamespace(
                matrix=sparse.csr_matrix(np.array([[3], [0]], dtype=np.uint32)),
                cells=pd.DataFrame(
                    {
                        "source_id": [self.source] * 2,
                        "row_in_shard": [0, 1],
                        "cell_id": [self.source + "-a", self.source + "-b"],
                        "guide_index": [0, 0],
                    }
                ),
            )

    reader = SimpleNamespace(
        PreparedShardReader=Reader, RowAddress=lambda **kwargs: kwargs
    )
    modules = (*modules[:2], reader, *modules[3:])
    monkeypatch.setattr(worker, "legacy_imports", lambda _: modules)
    monkeypatch.setattr(
        worker,
        "make_access",
        lambda plan, metadata, source, *args: SimpleNamespace(
            source=source,
            shards=[SimpleNamespace(rows=2, shard=0)],
            identity=lambda: "a" * 64,
            feature_order_sha256="b" * 64,
        ),
    )
    original_wait = worker.wait_for_release

    def release(out, startup):
        assert events == ["verified", "dense_budget"]
        value = dict(
            status="startup_released",
            plan_sha256=startup["plan_sha256"],
            pod_uid=startup["pod_uid"],
            startup_sha256=worker.sha_file(out / "STARTUP.json"),
            uid="synthetic-job",
            actual_pod_audit_passed=True,
            actual_pod_sha256="b" * 64,
            tests_permitted=True,
            scientific_promotion=False,
        )
        if failure == "release":
            value["pod_uid"] = "other-pod"
        put(out / "STARTUP_RELEASE.json", value)
        events.append("release")
        return original_wait(out, startup)

    monkeypatch.setattr(worker, "wait_for_release", release)
    if failure:
        with pytest.raises(RuntimeError, match="dense budget|startup release"):
            worker.main([])
        assert (
            "initialize" not in events
            and "reader" not in events
            and "encode" not in events
        )
        assert (output / "FAILED.json").is_file()
        assert not (output / "COMPLETE.json").exists()
        assert (output / "STARTUP.json").exists() == (failure == "release")
        return
    worker.main([])
    result = json.loads((output / "COMPLETE.json").read_bytes())
    assert result["eligible_rows"] == 8 and len(result["sources"]) == 4
    assert all(row["geometry_cells"] == 1 for row in result["sources"])
    assert (
        events.index("release")
        < events.index("initialize")
        < events.index("reader")
        < events.index("encode")
    )
    assert not (output / "FAILED.json").exists()
    assert all(
        worker.file_identity(output / row["path"])
        == {"sha256": row["sha256"], "size_bytes": row["size_bytes"]}
        for row in result["files"]
    )
