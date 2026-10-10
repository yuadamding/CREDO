"""Read-only requalification of three exact v3 MLP forecasts.

No arrays are loaded, no model is run, and no old publication is modified.
The old run remains failed.  This module is also usable by the Python 3.6
controller; it deliberately has no numerical/runtime dependencies.
"""
import hashlib
import json
import os
from pathlib import Path
import stat

GIB = 1024 ** 3
PRIOR_RUN = "credo-full-ddr-d2-forecast-20261010-v3"
PRIOR_UID = "59c65683-4390-4c7f-9929-7f109755b082"
PRIOR_PLAN = "2cfbea9a18025757362342fb046ec127155507dd2f73c89f998cb24e2044d40a"
PRIOR_IMPLEMENTATION = "ee18368f38bbc7200a1528a7995503bdb99dc0512c054c0bf57bd485c8cdbc95"
PRIOR_ROOT = Path("/rsrch8/home/bcb/yding4/perturbseq/credo_full_ddr_gse314342_20261006_v1/d2_forecast_v3")
PRODUCER_HASHES = {
    "PLAN.json": PRIOR_PLAN,
    "worker.py": "32035a8273898b12214cfd6cf139e6efccc94c7c1d7adb73a78a63b03b7ab1ec",
    "d2_forecast.py": "d69fa9597a5ea757d16508fd3810ffac80b88864e113b603dc83bcd1b1163ef5",
    "latent_support.py": "5e89b104c9282726ba95e6af0e0f8ff81f2ee333b87bcb734aa32a3e6869189e",
    "full_forecast.py": "ea4e3436209f06d364791b5e414234af7c10e6379fa673d90c385e4b06492188",
}
SOURCE_KEYS = ("donor", "source", "catalog_size", "catalog_sha256", "source_counts_sha256",
               "source_artifact_sha256", "representation_sha256", "support_sha256", "source_pseudocount")
LEASE_INFERENCE = (
    "The pinned v3 cached_snapshot finally block verified source-path identity on "
    "exception unwind before the outer handler wrote the original agreement error. "
    "This is an inference from the frozen producer and retained failure, not a "
    "historical positive source_lease_exit_verified receipt or overall v3 success."
)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def _same(record, expected, message):
    if any(record.get(k) != v or type(record.get(k)) is not type(v) for k, v in expected.items()):
        raise ValueError(message)


def _signature(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns,
            info.st_uid, info.st_nlink)


def _path(path, owner, directory=False):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Exact absolute nontraversing path required")
    for parent in path.parents:
        if not stat.S_ISDIR(parent.lstat().st_mode):
            raise ValueError("Unsafe ancestor")
    info = path.lstat()
    predicate = stat.S_ISDIR if directory else stat.S_ISREG
    if not predicate(info.st_mode) or info.st_uid != owner:
        raise ValueError("Wrong owner or file type")
    if not directory and info.st_nlink != 1:
        raise ValueError("Aliased immutable input")
    return info


def file_identity(path, owner=307469, metadata=False):
    """Hash by bounded streaming; reject path/FD changes and metadata oversize."""
    path = Path(path)
    before = _path(path, owner)
    if metadata and before.st_size > 16 * 1024 ** 2:
        raise ValueError("Oversized metadata")
    digest, chunks = hashlib.sha256(), []
    with os.fdopen(os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        if _signature(os.fstat(stream.fileno())) != _signature(before):
            raise ValueError("Input changed before read")
        while True:
            chunk = stream.read(4 * 1024 ** 2)
            if not chunk:
                break
            digest.update(chunk)
            if metadata:
                chunks.append(chunk)
        if (_signature(os.fstat(stream.fileno())) != _signature(before)
                or _signature(_path(path, owner)) != _signature(before)):
            raise ValueError("Input changed while hashing")
    return dict(sha256=digest.hexdigest(), size_bytes=before.st_size), b"".join(chunks)


def _file(binding, expected_path, owner, metadata=False):
    if binding.get("path") != str(expected_path):
        raise ValueError("External dependency path differs")
    identity, raw = file_identity(expected_path, owner, metadata)
    if identity != {k: binding.get(k) for k in ("sha256", "size_bytes")}:
        raise ValueError("External dependency identity differs")
    return identity, _json(raw) if metadata else None


def _evidence(record):
    raw = record["raw_utf8"].encode("utf-8")
    if len(raw) > 16 * 1024 ** 2 or len(raw) != record.get("size_bytes"):
        raise ValueError("Evidence byte budget or size differs")
    if hashlib.sha256(raw).hexdigest() != record.get("sha256"):
        raise ValueError("Evidence identity differs")
    return _json(raw)


def _producer(manifest, owner):
    _same(manifest, dict(schema="credo.full_ddr.d2_reuse_binding.v1", prior_run_id=PRIOR_RUN,
        prior_job_uid=PRIOR_UID, prior_plan_sha256=PRIOR_PLAN,
        prior_implementation_sha256=PRIOR_IMPLEMENTATION, output_root=str(PRIOR_ROOT / "output"),
        maximum_retained_plus_new_bytes=360 * GIB, combined_D2_phase_output_ceiling_bytes=400 * GIB),
        "Exact failed v3 producer and unchanged caps required")
    files = manifest.get("producer_files", {})
    if set(files) != set(PRODUCER_HASHES):
        raise ValueError("Frozen exception-unwind producer files missing")
    for name, digest in PRODUCER_HASHES.items():
        if files[name].get("sha256") != digest:
            raise ValueError("Historical producer code changed")
        _file(files[name], PRIOR_ROOT / "bundle" / name, owner)
    records = manifest.get("evidence", {})
    if set(records) != {"failure", "terminal", "cleanup", "startup", "startup_release"}:
        raise ValueError("Exact failed lifecycle evidence required")
    evidence = {key: _evidence(record) for key, record in records.items()}
    _same(evidence["failure"], dict(status="failed_no_automatic_retry",
        error="RuntimeError('Fixed scalar/batched forecast agreement failed: state')",
        model=dict(candidate_id="ddr_res256_e8", seed=0), written_banks=3,
        parameter_updates=0, D2_scores_computed=False, D1_access=False, LARRY_access=False),
        "Failure does not establish the exact preserved three-bank exception path")
    _same(evidence["terminal"], dict(status="failed", application_pass=False,
        kubernetes_complete=False, run_id=PRIOR_RUN, job=PRIOR_RUN, uid=PRIOR_UID,
        plan_sha256=PRIOR_PLAN, result_complete_sha256=None), "Prior run is not exact failed terminal")
    _same(evidence["cleanup"], dict(status="complete", job=PRIOR_RUN, uid=PRIOR_UID,
        reads=[dict(job_absent=True, owned_pods=0)] * 2), "Prior cleanup is not reconciled")
    _same(evidence["startup"], dict(status="startup_guard_passed", run_id=PRIOR_RUN,
        plan_sha256=PRIOR_PLAN, implementation_sha256=PRIOR_IMPLEMENTATION,
        writer_uid=owner, parameter_updates=0), "Historical worker startup differs")
    _same(evidence["startup_release"], dict(status="startup_released", uid=PRIOR_UID,
        plan_sha256=PRIOR_PLAN, actual_pod_audit_passed=True,
        pod_uid=evidence["startup"].get("pod_uid"), startup_sha256=records["startup"]["sha256"]),
        "Historical startup release differs")
    output = PRIOR_ROOT / "output"
    _path(output, owner, directory=True)
    # The immutable home copies must also be the exact embedded evidence.
    for name, key in (("FAILURE.json", "failure"), ("STARTUP.json", "startup")):
        actual, _ = file_identity(output / name, owner, True)
        if actual != {k: records[key][k] for k in ("sha256", "size_bytes")}:
            raise ValueError("Historical output evidence changed")
    if (output / "COMPLETE.json").exists() or (output / "PREDICTION_SEAL.json").exists():
        raise ValueError("Cannot relabel old completion as this failed-bank recovery")
    return evidence


def build_reuse_manifest(reconciliation, metadata):
    """Construct a pinned binding from already captured read-only receipts."""
    _same(reconciliation, dict(status="failed_v3_reconciled_three_unsealed_banks_no_scores",
        job=PRIOR_RUN, uid=PRIOR_UID, plan_sha256=PRIOR_PLAN, remote_writes=False,
        completion_absent=True, seal_absent=True,
        reads=[dict(job_absent=True, owned_pods=0)] * 2), "Exact prior reconciliation required")
    if reconciliation.get("process", {}).get("status") not in {
            "absent", "exited_zombie", "bound_process_absent_pid_reused"}:
        raise ValueError("Prior controller absence must be reconciled")
    _same(metadata, dict(remote_writes=False), "Reuse metadata must be read-only")
    banks = []
    for record in reconciliation["banks"]:
        if record.get("completion_absent") is not True:
            raise ValueError("Only unsealed old banks can enter this recovery")
        descriptors = record["metadata"]
        array = descriptors["bank"]["value"]["artifacts"]["forecast.h5"]
        if record["array_stat"].get("size_bytes") != array["bytes"]:
            raise ValueError("Retained array size differs from its producer binding")
        row = dict(candidate_id=record["candidate_id"], seed=record["seed"],
            forecast=dict(path=str(Path(record["root"]) / "forecast.h5"),
                          sha256=array["sha256"], size_bytes=array["bytes"]))
        row.update({key: {field: descriptors[key][field] for field in ("path", "sha256", "size_bytes")}
                    for key in ("bank", "loader", "qualification")})
        banks.append(row)
    result = dict(schema="credo.full_ddr.d2_reuse_binding.v1", prior_run_id=PRIOR_RUN,
        prior_job_uid=PRIOR_UID, prior_plan_sha256=PRIOR_PLAN,
        prior_implementation_sha256=PRIOR_IMPLEMENTATION, output_root=str(PRIOR_ROOT / "output"),
        producer_files=metadata["producer_files"], evidence=metadata["evidence"], banks=banks,
        maximum_retained_plus_new_bytes=360 * GIB, combined_D2_phase_output_ceiling_bytes=400 * GIB,
        retained_output_bytes=metadata["retained_output_bytes"],
        output_inventory_sizes=metadata["output_inventory_sizes"])
    if [(r["candidate_id"], r["seed"]) for r in banks] != [("ddr_mlp128_e8", s) for s in (0, 1, 2)]:
        raise ValueError("Exact MLP seed triplet required")
    if sum(r["size_bytes"] for r in result["output_inventory_sizes"]) != result["retained_output_bytes"]:
        raise ValueError("Retained output inventory does not reconcile")
    return result


def validate_reuse(manifest, *, source_identity, measure_ids, guide_indices, checkpoints,
                   owner=307469, maximum_new_bytes=240 * GIB):
    """Verify the three old banks under a CURRENT source lease and checkpoint binding.

    Caller must supply the freshly verified source identity/membership, preserve
    that lease through prediction, and recheck it before publishing new seals.
    """
    _producer(manifest, owner)
    if (set(source_identity) != set(SOURCE_KEYS) or source_identity["donor"] != "D2"
            or source_identity["source"] != "D2_Stim8hr"
            or len(measure_ids) != len(guide_indices) or not measure_ids
            or len(set(measure_ids)) != len(measure_ids) or len(set(guide_indices)) != len(guide_indices)):
        raise ValueError("Fresh complete source/membership identity required")
    expected_pairs = [("ddr_mlp128_e8", seed) for seed in (0, 1, 2)]
    rows = manifest.get("banks", [])
    if [(r.get("candidate_id"), r.get("seed")) for r in rows] != expected_pairs:
        raise ValueError("Only the exact three MLP banks are reusable")
    checkpoint_lookup = {(c["candidate_id"], c["seed"]): c for c in checkpoints}
    if len(checkpoint_lookup) != len(checkpoints):
        raise ValueError("Duplicate checkpoint identities")
    output, result = PRIOR_ROOT / "output", []
    for row, pair in zip(rows, expected_pairs):
        checkpoint = checkpoint_lookup[pair]
        paths = [row[key]["path"] for key in ("bank", "forecast", "qualification", "loader")]
        paths.append(checkpoint["checkpoint_path"])
        signatures = {path: list(_signature(_path(path, owner))) for path in paths}
        actual, _ = file_identity(checkpoint["checkpoint_path"], owner)
        if actual != {k: checkpoint[k] for k in ("sha256", "size_bytes")}:
            raise ValueError("Frozen checkpoint changed after original prediction")
        label = pair[0] + "--seed-" + str(pair[1])
        directory = output / "banks" / label
        _path(directory, owner, directory=True)
        if set(p.name for p in directory.iterdir()) != {"BANK.json", "forecast.h5"}:
            raise ValueError("Old bank must remain the exact unsealed two-file publication")
        bank_id, bank = _file(row["bank"], directory / "BANK.json", owner, True)
        array_id, _ = _file(row["forecast"], directory / "forecast.h5", owner)
        _, qualification = _file(row["qualification"], output / (label + "--qualification.json"), owner, True)
        _, loader = _file(row["loader"], output / (label + "--loader.json"), owner, True)
        _same(bank, dict(schema="credo.all4_full_ddr.forecast_bank.v1", status="arrays_written_not_sealed",
            seed=pair[1], prefixes=[4096, 8192], latent_dim=48, steps_per_interval=8,
            physical_interval_hours=40, checkpoint_sha256=checkpoint["sha256"],
            implementation_sha256=PRIOR_IMPLEMENTATION, parameter_updates=0,
            endpoint_observations_read=False, modeled_measure_ids=list(measure_ids),
            modeled_guide_indices=list(guide_indices), random_stream_identity="full-ddr-forecast-v1",
            source_sampling="weighted_iid_with_replacement_at_maximum_prefix",
            complement_policy="fixed_unnormalized_regularized_source_persistence",
            maximum_output_bytes=40 * GIB), "Old bank forecast contract differs")
        _same(bank, source_identity, "Old bank and fresh source identity differ")
        if bank.get("artifacts") != {"forecast.h5": dict(sha256=array_id["sha256"], bytes=array_id["size_bytes"])}:
            raise ValueError("Old bank array binding differs")
        if array_id["size_bytes"] + row["bank"]["size_bytes"] > 40 * GIB:
            raise ValueError("Old bank exceeds its cap")
        _same(qualification, dict(status="passed", scope="source_only_scalar_vs_batched_not_endpoint_precision",
            guides=16, particles=8192, steps=8, seed=pair[1], atol=2e-5, rtol=1e-5,
            parameter_updates=0, endpoint_observations_read=False), "Old numerical qualification differs")
        maxima = qualification.get("maximum_absolute_difference", {})
        if (set(maxima) != {"state", "log_weight", "mass_increment", "conditional_features"}
                or any(type(v) not in (int, float) or not 0 <= v < float("inf") for v in maxima.values())):
            raise ValueError("Incomplete old precision evidence")
        _same(loader, dict(checkpoint_sha256=checkpoint["sha256"], parameter_count=checkpoint["parameter_count"],
            candidate_id=pair[0], seed=pair[1], parameter_updates=0, optimizer_constructed=False,
            catalog_refresh_performed=False, endpoint_observations_read=False, evaluation_performed=False),
            "Old frozen loader evidence differs")
        if any(list(_signature(_path(path, owner))) != signature for path, signature in signatures.items()):
            raise ValueError("Old dependency changed during requalification")
        result.append(dict(candidate_id=pair[0], seed=pair[1], path=str(directory), storage="external_reused",
            checkpoint_sha256=checkpoint["sha256"], bank_identity=bank_id, array_identity=array_id,
            qualification=qualification, loader=loader, bank=bank, original_signatures=signatures))
    # Count all retained failed-attempt output, not just the reusable arrays.
    retained_bytes, names, sizes = 0, set(), []
    for path in output.rglob("*"):
        info = path.lstat()
        _path(path, owner, directory=stat.S_ISDIR(info.st_mode))
        if stat.S_ISREG(info.st_mode):
            retained_bytes += info.st_size
            sizes.append(dict(path=path.relative_to(output).as_posix(), size_bytes=info.st_size))
            if path.name == "forecast.h5":
                names.add(str(path.parent))
    if names != {r["path"] for r in result}:
        raise ValueError("Unexpected additional retained forecast arrays")
    if (retained_bytes != manifest.get("retained_output_bytes")
            or sorted(sizes, key=lambda row: row["path"]) != manifest.get("output_inventory_sizes")):
        raise ValueError("Retained failed-attempt inventory changed")
    if (type(maximum_new_bytes) is not int or maximum_new_bytes < 0
            or retained_bytes + maximum_new_bytes > 360 * GIB):
        raise ValueError("Retained plus new forecast output exceeds unchanged 360-GiB cap")
    return dict(status="three_failed_run_banks_requalified_not_old_run_success", banks=result,
        retained_bytes=retained_bytes, maximum_new_bytes=maximum_new_bytes,
        binding_sha256=hashlib.sha256(canonical(manifest)).hexdigest(),
        historical_lease_exit_basis=LEASE_INFERENCE, old_run_status="failed",
        parameter_updates=0, endpoint_observations_read=False)


def expected_reuse_seal(manifest, validated, row):
    """Pure exact seal definition shared by the worker and independent controller."""
    if validated["binding_sha256"] != hashlib.sha256(canonical(manifest)).hexdigest():
        raise ValueError("Requalification binding changed")
    return dict(schema="credo.full_ddr.d2_reused_bank_seal.v1", status="requalified_external_forecast_bank",
        prior_run_id=PRIOR_RUN, prior_job_uid=PRIOR_UID, prior_plan_sha256=PRIOR_PLAN,
        prior_run_status="failed", historical_positive_lease_receipt_present=False,
        historical_lease_exit_basis=LEASE_INFERENCE, binding_sha256=validated["binding_sha256"],
        candidate_id=row["candidate_id"], seed=row["seed"], original_bank_path=row["path"],
        bank_identity=row["bank_identity"], array_identity=row["array_identity"],
        checkpoint_sha256=row["checkpoint_sha256"],
        fresh_source_identity_verified=True, fresh_checkpoint_identity_verified=True,
        fresh_source_lease_exit_verified=True,
        original_artifacts_modified=False, historical_run_relabelled_successful=False,
        parameter_updates=0, endpoint_observations_read=False)


def publish_reuse_seals(output, manifest, validated, owner=307469, fresh_lease_exit_verified=False):
    """Write NEW recovery seals only after caller closes its fresh checked lease."""
    output = Path(output)
    _path(output, owner, directory=True)
    if output == PRIOR_ROOT or PRIOR_ROOT in output.parents or output in PRIOR_ROOT.parents:
        raise ValueError("Recovery seals cannot alter the historical attempt")
    if (fresh_lease_exit_verified is not True
            or validated.get("status") != "three_failed_run_banks_requalified_not_old_run_success"
            or validated.get("binding_sha256") != hashlib.sha256(canonical(manifest)).hexdigest()):
        raise ValueError("A matching fresh requalification is required")
    # Whole dependency signatures are checked before any new seal is written.
    for row in validated["banks"]:
        for path, signature in row["original_signatures"].items():
            if list(_signature(_path(path, owner))) != signature:
                raise ValueError("Requalified original changed before recovery seal")
    directory = output / "reuse-seals"
    directory.mkdir(mode=0o700)
    result = []
    for row in validated["banks"]:
        path = directory / (row["candidate_id"] + "--seed-" + str(row["seed"]) + ".json")
        seal = expected_reuse_seal(manifest, validated, row)
        raw = canonical(seal)
        with os.fdopen(os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        identity, _ = file_identity(path, owner, True)
        result.append(dict(row, reuse_seal_path=path.relative_to(output).as_posix(), reuse_seal=identity,
                           seal=identity))
    return result
