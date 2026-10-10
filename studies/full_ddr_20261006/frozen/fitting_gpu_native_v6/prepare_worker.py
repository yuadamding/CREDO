"""H100 worker for the four missing development latent sources, no fitting.

Uses the pinned original R48 representation's network/reader code. New data
authorization is explicit; the historical representation specification is not
modified or falsely recertified for the widened study. No automatic retry.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import stat
import sys
import time

from encode_sources import authorize_sources, export_source, publish_json, sha_file
from prepare_manifest import canonical


MODEL_SHA = "200d38d0e84b89ba7c5b84ca2d28b8c854c2e4ca33cfc705e12d6e84f98955c2"
FITTED_SPEC_SHA = "0d32e5b72017dd747fb8d12d5745fc6c84460545ab46868571aaf4250f2007e3"
FITTED_COMPLETE_SHA = "aa331bc947305a483507b62f7c42ff285801ad3bbe60ce38fb4af57b6a47040c"
REPRESENTATION_ID = "c5d095c646a047fe1d0d9e4dd179c0d736fd51f91065a1147e87e6b7f20d2c54"
DEVELOPMENT_SOURCES = ("D3_Stim8hr", "D4_Stim8hr", "D2_Stim8hr", "D2_Stim48hr")
WRITER = (307469, 600815)


def file_identity(path: Path, *, owner: int | None = None) -> dict:
    """Hash a stable regular member without following a symlink ancestor."""
    path = Path(path).absolute()
    if any(not stat.S_ISDIR(p.lstat().st_mode) for p in path.parents):
        raise RuntimeError("Non-directory/symlink artifact ancestor.")
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or (
        owner is not None and before.st_uid != owner
    ):
        raise RuntimeError("Expected owned regular artifact.")

    def signature(s):
        return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)

    digest = hashlib.sha256()
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as handle:
        if signature(os.fstat(handle.fileno())) != signature(before):
            raise RuntimeError("Artifact changed at open.")
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
        if signature(os.fstat(handle.fileno())) != signature(before):
            raise RuntimeError("Artifact changed during read.")
    if signature(path.lstat()) != signature(before):
        raise RuntimeError("Artifact path changed during read.")
    return {"sha256": digest.hexdigest(), "size_bytes": before.st_size}


def checked_json(path: Path, expected: str) -> dict:
    first = file_identity(path)
    if first["sha256"] != expected or first["size_bytes"] > 64 * 1024**2:
        raise RuntimeError("Pinned JSON identity mismatch: " + path.name)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected or file_identity(path) != first:
        raise RuntimeError("Pinned JSON changed during read.")
    return json.loads(raw)


def load_plan(argv=None, *, root: Path | None = None) -> tuple[Path, dict, str]:
    """The institutional command takes no arguments; optional CLI values must agree."""
    root = Path(__file__).absolute().parent if root is None else Path(root).absolute()
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=root / "PLAN.json")
    parser.add_argument("--plan-sha256", default=os.environ.get("CREDO_PLAN_SHA256"))
    args = parser.parse_args(argv)
    if args.plan.absolute() != root / "PLAN.json":
        raise RuntimeError("Only the sibling immutable PLAN is admitted.")
    if not args.plan_sha256 or args.plan_sha256 != os.environ.get("CREDO_PLAN_SHA256"):
        raise RuntimeError("Plan/environment identity differs.")
    plan = checked_json(args.plan, args.plan_sha256)
    if plan.get("run_id") != os.environ.get("CREDO_RUN_ID"):
        raise RuntimeError("Worker/run identity differs.")
    return root, plan, args.plan_sha256


def verify_bundle(root: Path, plan: dict) -> None:
    """Require exact publication membership and the implementation/source identities."""
    inventory_path = root / "INVENTORY.json"
    reference = file_identity(inventory_path)
    if reference["size_bytes"] > 64 * 1024**2:
        raise RuntimeError("Oversized bundle inventory.")
    inventory = checked_json(inventory_path, reference["sha256"])
    actual = {}
    for path in root.rglob("*"):
        mode = path.lstat().st_mode
        if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            raise RuntimeError("Special/symlink bundle entry.")
        if stat.S_ISREG(mode) and path != inventory_path:
            actual[path.relative_to(root).as_posix()] = file_identity(path)
    if actual != inventory:
        raise RuntimeError("Immutable bundle inventory differs.")
    implementation = checked_json(
        root / "IMPLEMENTATION.json", plan["implementation_sha256"]
    )
    expected = {
        name: identity["sha256"]
        for name, identity in actual.items()
        if name.endswith(".py") and name not in {"control.py", "publication.py"}
    }
    if (
        implementation != expected
        or actual["worker.py"]["sha256"] != plan["worker_sha256"]
    ):
        raise RuntimeError("Implementation identity differs.")
    legacy = checked_json(
        root / "LEGACY_SOURCE_BINDING.json", plan["legacy_source_binding_sha256"]
    )
    if (
        legacy["root"] != plan["legacy_source_root"]
        or legacy["files"] != plan["legacy_source_files"]
    ):
        raise RuntimeError("Frozen source binding differs.")


def wait_for_release(output: Path, startup: dict, *, sleep=None) -> dict:
    """No whole-job timeout: poll only for the exact controller-owned release."""
    sleep = time.sleep if sleep is None else sleep
    path = output / "STARTUP_RELEASE.json"
    while not path.exists() and not path.is_symlink():
        sleep(2)
    identity = file_identity(path, owner=WRITER[0])
    release = checked_json(path, identity["sha256"])
    expected = {
        "status": "startup_released",
        "plan_sha256": startup["plan_sha256"],
        "startup_sha256": file_identity(output / "STARTUP.json")["sha256"],
        "pod_uid": startup["pod_uid"],
        "actual_pod_audit_passed": True,
        "tests_permitted": True,
        "scientific_promotion": False,
    }
    if (
        any(release.get(key) != value for key, value in expected.items())
        or not isinstance(release.get("uid"), str)
        or not release["uid"]
        or not isinstance(release.get("actual_pod_sha256"), str)
        or len(release["actual_pod_sha256"]) != 64
        or set(release["actual_pod_sha256"]) - set("0123456789abcdef")
    ):
        raise RuntimeError("Invalid actual-Pod startup release.")
    return release


def startup_record(plan: dict, plan_sha: str, prop, torch) -> dict:
    if not os.environ.get("CREDO_POD_UID") or not os.environ.get("CREDO_POD_NAME"):
        raise RuntimeError("Exact worker Pod identity required.")
    return dict(
        status="startup_guard_passed",
        run_id=plan["run_id"],
        plan_sha256=plan_sha,
        pod_name=os.environ["CREDO_POD_NAME"],
        pod_uid=os.environ["CREDO_POD_UID"],
        image=plan["image"],
        implementation_sha256=plan["implementation_sha256"],
        gpu=dict(
            name=prop.name,
            count=1,
            total_memory_bytes=prop.total_memory,
            compute_capability="9.0",
        ),
        python=platform.python_version(),
        torch=torch.__version__,
        writer_uid=os.getuid(),
        writer_gid=os.getgid(),
        writer_canary_removed=True,
        bundle_inventory_verified=True,
        required_imports_passed=True,
        tests_started=False,
        real_data_opened=False,
        scientific_promotion=False,
        model_sha256=MODEL_SHA,
        parameter_updates=0,
        started_utc=datetime.now(timezone.utc).isoformat(),
    )


def publish_completion(
    output: Path,
    plan: dict,
    plan_sha: str,
    startup: dict,
    results: list[dict],
    *,
    elapsed_seconds: float,
    peak_bytes: int,
) -> dict:
    if tuple(r["source_id"] for r in results) != DEVELOPMENT_SOURCES:
        raise RuntimeError("Completion sources/order differ.")
    rows = sum(r["rows"] for r in results)
    if rows != plan["expected_eligible_rows"]:
        raise RuntimeError("Completion source row inventory differs.")
    files = []
    for path in sorted(output.rglob("*")):
        info = path.lstat()
        if info.st_uid != WRITER[0] or not (
            stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)
        ):
            raise RuntimeError("Unexpected result owner or special member.")
        if stat.S_ISREG(info.st_mode):
            name = path.relative_to(output).as_posix()
            if name in {"COMPLETE.json", "FAILED.json"}:
                raise RuntimeError("Existing completion/failure cannot be overwritten.")
            files.append(dict(path=name, **file_identity(path, owner=WRITER[0])))
    total = sum(item["size_bytes"] for item in files)
    result = dict(
        status="completed_full_ddr_development_preparation",
        run_id=plan["run_id"],
        plan_sha256=plan_sha,
        pod_uid=startup["pod_uid"],
        implementation_sha256=plan["implementation_sha256"],
        sources=results,
        eligible_rows=rows,
        parameter_updates=0,
        LARRY_access=False,
        D1_endpoint_access=False,
        preparation_only=True,
        scientific_training_complete=False,
        scientific_promotion=False,
        elapsed_seconds=elapsed_seconds,
        output_bytes=total,
        files=files,
        peak_allocated_bytes=peak_bytes,
        completed_utc=datetime.now(timezone.utc).isoformat(),
    )
    if total + len(canonical(result)) > plan["maximum_output_bytes"]:
        raise RuntimeError("Preparation output cap exceeded.")
    publish_json(output / "COMPLETE.json", result)
    return result


def admit(plan: dict) -> None:
    if plan.get("schema") != "credo.all4_ddr.encoding_execution.v1":
        raise RuntimeError("Unknown encoding execution schema.")
    if tuple(plan.get("sources", ())) != DEVELOPMENT_SOURCES:
        raise RuntimeError("Exact four-source preparation required.")
    authorize_sources(DEVELOPMENT_SOURCES, "development")
    if (
        plan.get("scientific_optimizer_updates") != 0
        or plan.get("automatic_retries") != 0
    ):
        raise RuntimeError("Preparation has no fitting or retry authority.")
    if plan.get("elapsed_time_deadline_seconds") is not None:
        raise RuntimeError("No elapsed-time ceiling is permitted.")
    if plan.get("batch_rows") != 2048 or plan.get("threads") != 8:
        raise RuntimeError(
            "Retained forward batch and bound CPU thread policy required."
        )
    if (
        plan.get("image")
        != "hpcharbor.mdanderson.edu/yding41/credo_env@sha256:138d7191792ce1139708751f109817a976c5b96ba3a00be3dff88173b62eb21a"
    ):
        raise RuntimeError("Only the approved pinned CREDO image is admitted.")
    if plan.get("maximum_output_bytes") != 16 * 1024**3:
        raise RuntimeError("Preparation output cap differs.")
    if (
        plan.get("schema_id") != "credo.all4_full_ddr.development_preparation.v1"
        or plan.get("worker_output") != plan.get("output")
        or plan.get("maximum_new_output_bytes") != plan["maximum_output_bytes"]
        or plan.get("parameter_updates") != 0
        or plan.get("preparation_only") is not True
        or plan.get("D1_endpoint_access") is not False
        or plan.get("LARRY_access") is not False
    ):
        raise RuntimeError("Lifecycle and encoding authority differ.")
    for key, value in (
        ("model_sha256", MODEL_SHA),
        ("fitted_specification_sha256", FITTED_SPEC_SHA),
        ("fitted_complete_sha256", FITTED_COMPLETE_SHA),
        ("representation_identity", REPRESENTATION_ID),
    ):
        if plan.get(key) != value:
            raise RuntimeError("Frozen model authority differs.")
    for field in ("output", "native_root", "fitted_root", "legacy_source_root"):
        path = Path(plan[field])
        if not path.is_absolute() or ".." in path.parts:
            raise RuntimeError("Absolute canonical project paths required.")
        if not path.is_relative_to("/rsrch8/home/bcb/yding4/perturbseq"):
            raise RuntimeError("Path outside approved project.")
    if Path(plan["output"]).is_relative_to(plan["native_root"]):
        raise RuntimeError("Outputs cannot modify inputs.")


def legacy_imports(plan: dict):
    """Bind only required historical Python source, not a moving installed package."""
    root = Path(plan["legacy_source_root"])
    for name, expected in plan["legacy_source_files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".py":
            raise RuntimeError("Unsafe legacy source binding.")
        if sha_file(root / relative) != expected:
            raise RuntimeError("Legacy representation/reader source changed.")
    sys.path.insert(0, str(root))
    modules = [
        importlib.import_module(name)
        for name in (
            "credo_count_sde_v4.count_representation.network",
            "credo_count_sde_v4.count_representation.contracts",
            "credo_count_sde_v4.data.prepared_shards",
            "credo_count_sde_v4.persistence.artifacts",
            "credo_count_sde_v4.canonical",
        )
    ]
    for name, module in list(sys.modules.items()):
        if name == "credo_count_sde_v4" or name.startswith("credo_count_sde_v4."):
            origin = Path(module.__file__).resolve()
            relative = str(origin.relative_to(root.resolve()))
            if relative not in plan["legacy_source_files"]:
                raise RuntimeError("Imported unbound legacy dependency: " + relative)
    return modules


def make_access(plan, metadata, source_id, reader_module, canonical_module):
    from credo_count_sde_v4.contracts.models import ArtifactRef

    def artifact(name, sha, size):
        return ArtifactRef(
            schema_id="prepared-package-member",
            schema_version=1,
            relative_uri=name,
            sha256=sha,
            size_bytes=size,
            media_type="application/octet-stream",
        )

    guides = metadata["canonical_guides"]
    ordered = [
        dict(
            guide_index=i,
            guide_id=g["sgRNA"],
            target_id=g["primary_target_id"],
            is_control=g["is_design_ntc"],
        )
        for i, g in enumerate(guides)
    ]
    guide_file = metadata["input_binding"]["metadata_tables"]["source/guides.parquet"]
    binding = reader_module.GuideCatalogBinding(
        artifact=artifact(
            "source/guides.parquet", guide_file["sha256"], guide_file["size_bytes"]
        ),
        ordered_catalog_sha256=canonical_module.contract_id(ordered),
        guide_id_column="sgRNA",
        target_id_column="primary_target_id",
        control_column="is_design_ntc",
        index_column=None,
    )
    shards = tuple(
        reader_module.PreparedShard(
            source_id=source_id,
            shard=r["shard"],
            rows=r["rows"],
            nnz=r["nnz"],
            counts=artifact(r["path"], r["sha256"], r["bytes"]),
            cells=artifact(r["cells_path"], r["cells_sha256"], r["cell_table_bytes"]),
        )
        for r in metadata["shards"]
        if r["source_id"] == source_id
    )
    features = [
        {"feature_id": f["feature_id"], "is_RNA": f["is_RNA"]}
        for f in metadata["canonical_features"]
    ]
    return reader_module.PreparedAccess(
        package_completion_sha256=metadata["input_binding"]["native_completion_sha256"],
        package_inventory_sha256=metadata["input_binding"]["native_inventory_sha256"],
        parent_view_sha256=plan["metadata_sha256"],
        amendment_sha256=plan["authority_sha256"],
        feature_order_sha256=canonical_module.contract_id(features),
        guide_catalog=binding,
        n_features=len(features),
        task_id="gse314342_all4_full_ddr_20261006_v1",
        source_ids=(source_id,),
        shards=shards,
        role="dynamics_supervision",
    )


def main(argv=None) -> None:
    root, plan, plan_sha = load_plan(argv)
    admit(plan)
    if root != Path(plan["bundle_root"]):
        raise RuntimeError("Worker bundle location differs.")
    verify_bundle(root, plan)
    output = Path(plan["output"])
    if output.exists() or any(p.is_symlink() for p in (output, *output.parents)):
        raise RuntimeError("New non-symlink output root required; never retry into it.")
    if (os.getuid(), os.getgid()) != WRITER:
        raise RuntimeError("Unexpected institutional writer identity.")
    output.mkdir(mode=0o700)
    started = time.monotonic()
    try:
        import torch
        import numpy as np
        import pyarrow as pa

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one CUDA H100 required, no CPU fallback.")
        prop = torch.cuda.get_device_properties(0)
        if (
            "H100" not in prop.name
            or "MIG" in prop.name
            or prop.total_memory < 75_000_000_000
        ):
            raise RuntimeError("Full H100 80GB-class device required.")
        if torch.cuda.get_device_capability(0) != (9, 0):
            raise RuntimeError("H100 capability mismatch.")
        torch.set_num_threads(8)
        torch.set_num_interop_threads(2)
        pa.set_cpu_count(8)
        pa.set_io_thread_count(2)
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        network, contracts, reader_module, tensor_io, canonical_module = legacy_imports(
            plan
        )
        metadata = checked_json(Path(plan["metadata_path"]), plan["metadata_sha256"])
        checked_json(Path(plan["authority_path"]), plan["authority_sha256"])
        fitted = Path(plan["fitted_root"])
        spec = checked_json(fitted / "specification.json", FITTED_SPEC_SHA)
        fitted_record = checked_json(fitted / "COMPLETE.json", FITTED_COMPLETE_SHA)
        if canonical_module.contract_id(fitted_record) != REPRESENTATION_ID:
            raise RuntimeError("Frozen fitted representation identity differs.")
        features = [
            {"feature_id": f["feature_id"], "is_RNA": f["is_RNA"]}
            for f in metadata["canonical_features"]
        ]
        if spec["features"] != features:
            raise RuntimeError("Frozen model and count axes disagree.")
        rules = contracts.RepresentationRules.model_validate(spec["rules"])
        if rules.device != "cuda" or rules.batch_rows != plan["batch_rows"]:
            raise RuntimeError("Retained encoder numerical batch/device changed.")
        rna_positions = [i for i, feature in enumerate(features) if feature["is_RNA"]]
        network.dense_budget(rules, len(rna_positions), rules.batch_rows)
        expected_rows = sum(
            r["rows"]
            for r in metadata["shards"]
            if r["source_id"] in DEVELOPMENT_SOURCES
        )
        if expected_rows != plan["expected_eligible_rows"]:
            raise RuntimeError("Authorized source row inventory differs.")
        canary = output / "writer-canary"
        publish_json(canary, {"owner": os.getuid()})
        if canary.stat().st_uid != os.getuid():
            raise RuntimeError("Output ownership mismatch.")
        canary.unlink()
        startup = startup_record(plan, plan_sha, prop, torch)
        publish_json(output / "STARTUP.json", startup)
        print(json.dumps(startup, sort_keys=True), flush=True)
        wait_for_release(output, startup)
        # Tensor loading and all prepared count access occur only after release.
        if sha_file(fitted / "model.safetensors") != MODEL_SHA:
            raise RuntimeError("Frozen checkpoint hash differs.")
        model = network.initialize(rules, len(rna_positions))
        state = tensor_io.load_tensor_file(fitted / "model.safetensors", device="cuda")
        model.load_state_dict(state, strict=True)
        if any(not bool(torch.isfinite(v).all()) for v in state.values()) or not bool(
            (model.scale > 0).all()
        ):
            raise RuntimeError("Invalid frozen encoder parameters/scaler.")
        model.eval().requires_grad_(False)
        del state
        progress_path = output / "PROGRESS.jsonl"

        def progress(row):
            row.update(
                utc=datetime.now(timezone.utc).isoformat(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            )
            with progress_path.open("ab") as handle:
                handle.write(canonical(row))
            print(json.dumps(row, sort_keys=True), flush=True)

        results = []
        for source in DEVELOPMENT_SOURCES:
            access = make_access(
                plan, metadata, source, reader_module, canonical_module
            )
            reader = reader_module.PreparedShardReader(
                Path(plan["native_root"]), access, max_batch_rows=rules.batch_rows
            )
            guides = reader.guide_catalog()

            def blocks(reader=reader, access=access, source=source):
                for record in access.shards:
                    for start in range(0, record.rows, rules.batch_rows):
                        addresses = [
                            reader_module.RowAddress(
                                source_id=source, shard=record.shard, row=r
                            )
                            for r in range(
                                start, min(record.rows, start + rules.batch_rows)
                            )
                        ]
                        block = reader.read_rows(addresses)
                        counts = block.matrix[:, rna_positions].tocsr()
                        cells = block.cells.copy()
                        depth = np.asarray(counts.sum(axis=1), dtype=np.int64).ravel()
                        if "RNA_UMIs" in cells and not np.array_equal(
                            cells.RNA_UMIs, depth
                        ):
                            raise RuntimeError("Canonical RNA library depth differs.")
                        cells["RNA_UMIs"] = depth
                        cells["shard"] = record.shard
                        yield counts, cells

            with torch.inference_mode():
                result = export_source(
                    output / source,
                    source_id=source,
                    phase="development",
                    expected_rows=sum(r.rows for r in access.shards),
                    guides=guides,
                    latent_dim=rules.latent_dim,
                    rna_features=len(rna_positions),
                    blocks=blocks(),
                    encode=lambda counts: model.encode(counts).cpu().numpy(),
                    lineage=dict(
                        input_access_sha256=access.identity(),
                        representation_sha256=REPRESENTATION_ID,
                        model_sha256=MODEL_SHA,
                        scaler_sha256=MODEL_SHA,
                        gene_axis_sha256=access.feature_order_sha256,
                        implementation_sha256=plan["implementation_sha256"],
                    ),
                    maximum_output_bytes=4 * 1024**3,
                    progress=progress,
                )
            results.append(
                dict(
                    source_id=source,
                    completion_sha256=sha_file(output / source / "COMPLETE.json"),
                    rows=result["expected_rows"],
                    geometry_cells=result["geometry_cells"],
                )
            )
            del reader
        if sha_file(fitted / "model.safetensors") != MODEL_SHA:
            raise RuntimeError("Frozen checkpoint changed during encoding.")
        verify_bundle(root, plan)
        publish_completion(
            output,
            plan,
            plan_sha,
            startup,
            results,
            elapsed_seconds=time.monotonic() - started,
            peak_bytes=torch.cuda.max_memory_allocated(),
        )
    except Exception as error:
        publish_json(
            output / "FAILED.json",
            dict(
                status="failed_no_retry",
                type=type(error).__name__,
                message=str(error),
                plan_sha256=plan_sha,
                elapsed_seconds=time.monotonic() - started,
            ),
        )
        raise


if __name__ == "__main__":
    main()
