"""Build an immutable, no-submit H100 preparation bundle for the new DDR study.

Only explicitly listed new-study inputs and hash-bound historical lifecycle
helpers are packaged. This builder makes no remote calls or scheduler mutations.
"""

from __future__ import annotations

import argparse
import ast
import copy
import gzip
import hashlib
import io
import importlib.util
import json
import os
from pathlib import Path
import stat
import tarfile

BASE = Path(__file__).resolve().parent
HISTORICAL = BASE.parent / (
    "credo_full_cohort_20260912/R48_downstream_v1/main_stim8_to48_v1/"
    "preparation_v1/worker_v1/bundle"
)
IMAGE = "hpcharbor.mdanderson.edu/yding41/credo_env@sha256:138d7191792ce1139708751f109817a976c5b96ba3a00be3dff88173b62eb21a"
RUN_ID = "credo-full-ddr-prepare-20261006-v1"
SUFFIX = "/perturbseq/credo_full_ddr_gse314342_20261006_v1/preparation_v1"
HOME_RUN = "/rsrch8/home/bcb/yding4" + SUFFIX
CONTROL_ROOT = "/rsrch8/scratch/bcb/yding4" + SUFFIX
POLICY = "no_runtime_deadline_user_20260919"
R48_BUNDLE = (
    BASE.parent / "credo_full_cohort_20260912/r48_full_pipeline_20260914_a2/bundle"
)
R48_REMOTE = "/rsrch8/home/bcb/yding4/perturbseq/r48_full_pipeline_20260914_a2"
MAXIMUM_BUNDLE_BYTES = 100_000_000
PINNED = {
    "STATIC_REUSE.json": "21eb4a54a1f3c1e905463c6f162d782773e7e18c0fe81976434d26cafea7ceda",
    "gpu_v2_static.py": "0d4d597ba46815b3e146735e62d4383c004c558a057377563c70e9cd20d8c061",
    "lifecycle_core.py": "3da1a0a5b0d0d26019918ee48e2ce24cd2e616272de7854f4c82ba2e37e7ebd2",
    "runtime_policy.py": "bbe70b8caf28f398fd939a887822df2462f5077b9cdc70ebc5a443a719485fa5",
    "deployment.py": "1c88af69851bb00028488d714bb4fdef014cf3f3cfd20c081cd470c9d7df0ae2",
    "gpu-job.json": "3c61a00b7ab390aeebfbff0259e51597e4f5850aed9107a08a6567b2effe6550",
}
STUDY_FILES = (
    "AUTHORITY.json",
    "STUDY_PLAN.json",
    "METADATA_PREPARATION.json",
    "SELECTION_CONTRACT.json",
    "prepare_manifest.py",
    "encode_sources.py",
    "latent_support.py",
    "native_study.py",
    "selection.py",
)


def canonical(value):
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_regular(path):
    path = Path(path).absolute()
    for parent in path.parents:
        if not stat.S_ISDIR(parent.lstat().st_mode):
            raise ValueError("Symlink/non-directory input ancestor")
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > MAXIMUM_BUNDLE_BYTES:
        raise ValueError("Bounded regular source file required")

    def signature(value):
        return (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as handle:
        if signature(os.fstat(handle.fileno())) != signature(before):
            raise ValueError("Source changed at open")
        raw = handle.read()
        if signature(os.fstat(handle.fileno())) != signature(before):
            raise ValueError("Source changed during read")
    if signature(path.lstat()) != signature(before):
        raise ValueError("Source path changed during read")
    return raw


def inherited_payload(historical=HISTORICAL):
    """Copy exact helper/proof bytes, not old scientific validation semantics."""
    historical = Path(historical)
    pinned = {name: read_regular(historical / name) for name in PINNED}
    if any(sha(raw) != PINNED[name] for name, raw in pinned.items()):
        raise ValueError("Historical lifecycle/template source changed")
    authority_raw = pinned["STATIC_REUSE.json"]
    authority = json.loads(authority_raw)
    payload = {"STATIC_REUSE.json": authority_raw}
    for name, expected in authority["files"].items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != name:
            raise ValueError("Unsafe static authority member")
        raw = read_regular(historical / path)
        if sha(raw) != expected:
            raise ValueError("Historical static evidence/source hash changed")
        payload[name] = raw
    payload["gpu_v2_static.py"] = pinned["gpu_v2_static.py"]
    payload["deployment.py"] = pinned["deployment.py"]
    payload["lifecycle_core.py"] = pinned["lifecycle_core.py"]
    payload["runtime_policy.py"] = pinned["runtime_policy.py"]
    return payload, json.loads(pinned["gpu-job.json"])


def legacy_source_closure(bundle=R48_BUNDLE):
    """Bind the static transitive local-package imports of the frozen encoder.

    This parses source only. It does not import old models or open fitted tensors.
    The worker additionally rejects any actual imported package member not bound.
    """
    bundle = Path(bundle)
    root = bundle / "repo/src"
    inventory_raw = read_regular(bundle / "INVENTORY.json")
    if (
        sha(inventory_raw)
        != "5fb67d78f6a87c2adfad135fa9ee06a9b95ad66a63962d70e48c444ade9edce7"
    ):
        raise ValueError("Retained R48 source inventory changed")
    inventory = json.loads(inventory_raw)
    pending = [
        "credo_count_sde_v4." + suffix
        for suffix in (
            "count_representation.network",
            "count_representation.contracts",
            "data.prepared_shards",
            "persistence.artifacts",
            "canonical",
            "contracts.models",
        )
    ]
    seen, result = set(), {}

    def resolve(name):
        path = root.joinpath(*name.split("."))
        if path.with_suffix(".py").is_file():
            return path.with_suffix(".py"), False
        if (path / "__init__.py").is_file():
            return path / "__init__.py", True
        return None, False

    while pending:
        name = pending.pop()
        if name in seen or not name.startswith("credo_count_sde_v4"):
            continue
        path, package = resolve(name)
        if path is None:
            continue  # A from-import may name a class, not another module.
        seen.add(name)
        raw = read_regular(path)
        key = path.relative_to(bundle).as_posix()
        if inventory.get(key) != {"bytes": len(raw), "sha256": sha(raw)}:
            raise ValueError(
                "Frozen R48 source differs from its publication inventory: " + key
            )
        result[path.relative_to(root).as_posix()] = sha(raw)
        package_name = name if package else name.rpartition(".")[0]
        if "." in name:
            pending.append(name.rpartition(".")[0])
        for node in ast.walk(ast.parse(raw)):
            if isinstance(node, ast.Import):
                pending.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imported = (
                    importlib.util.resolve_name("." * node.level + module, package_name)
                    if node.level
                    else module
                )
                pending.append(imported)
                pending.extend(
                    imported + "." + alias.name
                    for alias in node.names
                    if alias.name != "*"
                )
    if not result:
        raise ValueError("Frozen encoder dependency closure is empty")
    return dict(sorted(result.items()))


def publication_wrapper():
    """Reuse tested atomic publication; replace only old study's exact root guard."""
    return (
        '''"""Stdlib-only exact new-study scope around the unchanged publisher."""
import hashlib
import json
from pathlib import Path
import deployment

EXPECTED = %r
ROOTS = %r

def validated_roots(plan):
    if (plan.get("control_root") != ROOTS[0]
            or plan.get("bundle_root") != ROOTS[1]+"/bundle"
            or plan.get("worker_output") != ROOTS[1]+"/output"
            or plan.get("run_id") != %r):
        raise ValueError("New study preparation roots/identity differ")
    return tuple(Path(value) for value in ROOTS)

def publish_archive(raw, build, **kwargs):
    path = Path(__file__).with_name("deployment.py")
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != EXPECTED:
        raise ValueError("Reused publication helper changed")
    deployment.validated_roots = validated_roots
    return deployment.publish_archive(raw, build, **kwargs)
'''
        % (PINNED["deployment.py"], (CONTROL_ROOT, HOME_RUN), RUN_ID)
    ).encode()


def render_manifest(template, plan):
    manifest = copy.deepcopy(template)
    manifest["metadata"]["name"] = RUN_ID
    manifest["metadata"]["annotations"] = {"seadragon_run_id": RUN_ID}
    spec = manifest["spec"]
    spec.update(backoffLimit=0, completions=1, parallelism=1, suspend=True)
    if (
        "activeDeadlineSeconds" in spec
        or "activeDeadlineSeconds" in spec["template"]["spec"]
    ):
        raise ValueError("Runtime deadlines are forbidden")
    pod = spec["template"]["spec"]
    worker = pod["containers"][0]
    worker["command"] = ["/opt/credo/bin/python", HOME_RUN + "/bundle/worker.py"]
    worker["image"] = IMAGE
    values = {entry["name"]: entry for entry in worker["env"]}
    values["CREDO_RUN_ID"]["value"] = RUN_ID
    values["CREDO_PLAN_SHA256"]["value"] = sha(canonical(plan))
    values["PYTHONPATH"] = {
        "name": "PYTHONPATH",
        "value": HOME_RUN + "/bundle/source/src",
    }
    worker["env"] = [values[key] for key in sorted(values)]
    for kind in ("requests", "limits"):
        worker["resources"][kind].update(
            cpu="24", memory="200Gi", **{"nvidia.com/gpu": "1"}
        )
    return manifest


def assemble(study_root=BASE, historical=HISTORICAL):
    study_root = Path(study_root)
    payload, template = inherited_payload(historical)
    for name in STUDY_FILES:
        payload[name] = read_regular(study_root / name)
    payload["worker.py"] = read_regular(study_root / "prepare_worker.py")
    payload["control.py"] = read_regular(study_root / "preparation_control.py")
    for name in ("prepare_launch.py", "prepare_operate.py"):
        payload["controller/" + name] = read_regular(study_root / name)
    source = study_root / "source/src/credo"
    if source.is_symlink() or not source.is_dir():
        raise ValueError("The isolated main-CREDO source package is required")
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(study_root)
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            raise ValueError("Source-package symlinks cannot be staged")
        if path.is_file():
            payload[relative.as_posix()] = read_regular(path)
    for name in ("source/LICENSE", "source/pyproject.toml"):
        payload[name] = read_regular(study_root / name)
    payload["publication.py"] = publication_wrapper()
    implementation = {
        name: sha(raw)
        for name, raw in sorted(payload.items())
        if name.endswith(".py") and name not in {"control.py", "publication.py"}
    }
    payload["IMPLEMENTATION.json"] = canonical(implementation)
    legacy_files = legacy_source_closure()
    payload["LEGACY_SOURCE_BINDING.json"] = canonical(
        {
            "root": R48_REMOTE + "/bundle/repo/src",
            "files": legacy_files,
            "validation": "exact_old_inventory_and_static_import_closure_no_numerical_import",
        }
    )
    plan = {
        "schema": "credo.all4_ddr.encoding_execution.v1",
        "schema_id": "credo.all4_full_ddr.development_preparation.v1",
        "run_id": RUN_ID,
        "bundle_root": HOME_RUN + "/bundle",
        "control_root": CONTROL_ROOT,
        "worker_output": HOME_RUN + "/output",
        "image": IMAGE,
        "runtime_deadline_policy": POLICY,
        "automatic_retries": 0,
        "gpu_ceiling": 1,
        "authorized_CPUs": 24,
        "authorized_memory_GiB": 200,
        "torch_threads": 8,
        "threads": 8,
        "batch_rows": 2048,
        "maximum_new_output_bytes": 16 * 1024**3,
        "maximum_output_bytes": 16 * 1024**3,
        "output": HOME_RUN + "/output",
        "native_root": "/rsrch8/home/bcb/yding4/perturbseq/inputs/GSE314342_count_native_v1",
        "fitted_root": R48_REMOTE + "/output/fitted",
        "legacy_source_root": R48_REMOTE + "/bundle/repo/src",
        "legacy_source_files": legacy_files,
        "legacy_source_binding_sha256": sha(payload["LEGACY_SOURCE_BINDING.json"]),
        "model_sha256": "200d38d0e84b89ba7c5b84ca2d28b8c854c2e4ca33cfc705e12d6e84f98955c2",
        "fitted_specification_sha256": "0d32e5b72017dd747fb8d12d5745fc6c84460545ab46868571aaf4250f2007e3",
        "fitted_complete_sha256": "aa331bc947305a483507b62f7c42ff285801ad3bbe60ce38fb4af57b6a47040c",
        "representation_identity": "c5d095c646a047fe1d0d9e4dd179c0d736fd51f91065a1147e87e6b7f20d2c54",
        "metadata_path": HOME_RUN + "/bundle/METADATA_PREPARATION.json",
        "metadata_sha256": sha(payload["METADATA_PREPARATION.json"]),
        "authority_path": HOME_RUN + "/bundle/AUTHORITY.json",
        "implementation_sha256": sha(payload["IMPLEMENTATION.json"]),
        "control_sha256": sha(payload["control.py"]),
        "worker_sha256": sha(payload["worker.py"]),
        "lifecycle_core_sha256": PINNED["lifecycle_core.py"],
        "authority_sha256": sha(payload["AUTHORITY.json"]),
        "study_plan_sha256": sha(payload["STUDY_PLAN.json"]),
        "metadata_preparation_sha256": sha(payload["METADATA_PREPARATION.json"]),
        "static_reuse_sha256": sha(payload["STATIC_REUSE.json"]),
        "static_reuse_validator_sha256": sha(payload["gpu_v2_static.py"]),
        "startup_gate": "inline_h100_controller_release_v1",
        "startup_gate_authority": "New all-four-donor full-DDR H100 study; frozen encoding preparation only",
        "preparation_only": True,
        "parameter_updates": 0,
        "D1_endpoint_access": False,
        "scientific_optimizer_updates": 0,
        "LARRY_access": False,
        "scientific_promotion": False,
        "sources": ["D3_Stim8hr", "D4_Stim8hr", "D2_Stim8hr", "D2_Stim48hr"],
        "expected_eligible_rows": 7590115,
        "authorized_new_source_exports": [
            "D2_Stim8hr",
            "D2_Stim48hr",
            "D3_Stim8hr",
            "D4_Stim8hr",
        ],
        "output_completion_status": "completed_full_ddr_development_preparation",
    }
    payload["PLAN.json"] = canonical(plan)
    payload["gpu-job.json"] = canonical(render_manifest(template, plan))
    payload["REUSED_LIFECYCLE_SOURCES.json"] = canonical(
        {
            "origin": str(Path(historical).absolute()),
            "unchanged_helpers": {
                name: sha(payload[name])
                for name in ("lifecycle_core.py", "runtime_policy.py", "deployment.py")
            },
            "historical_scientific_control_executed": False,
            "new_control": "control.py",
            "no_remote_actions_by_builder": True,
        }
    )
    if sum(map(len, payload.values())) > MAXIMUM_BUNDLE_BYTES:
        raise ValueError("Exact allowlisted bundle exceeds 100MB bound")
    return payload


def pack(payload):
    inventory = {
        name: {"sha256": sha(raw), "size_bytes": len(raw)}
        for name, raw in sorted(payload.items())
    }
    if "INVENTORY.json" in payload:
        raise ValueError("Inventory is generated, not supplied")
    members = dict(payload, **{"INVENTORY.json": canonical(inventory)})
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0, filename="") as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for name, raw in sorted(members.items()):
                path = Path(name)
                if path.is_absolute() or ".." in path.parts or path.as_posix() != name:
                    raise ValueError("Noncanonical bundle member")
                info = tarfile.TarInfo("bundle/" + name)
                info.size, info.mode, info.mtime = len(raw), 0o400, 0
                info.uid = info.gid = 0
                archive.addfile(info, io.BytesIO(raw))
    raw_archive = buffer.getvalue()
    build = {
        "schema": "credo.all4_full_ddr.prepare_bundle.v1",
        "plan_sha256": sha(payload["PLAN.json"]),
        "archive_sha256": sha(raw_archive),
        "inventory_sha256": sha(members["INVENTORY.json"]),
        "files": inventory,
        "run_id": RUN_ID,
        "remote_submitted": False,
    }
    return members, raw_archive, build


def build(destination, *, study_root=BASE, historical=HISTORICAL):
    members, archive, receipt = pack(assemble(study_root, historical))
    destination = Path(destination)
    destination.mkdir(mode=0o700)
    bundle = destination / "bundle"
    bundle.mkdir(mode=0o700)
    for name, raw in sorted(members.items()):
        path = bundle / name
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with os.fdopen(
            os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400),
            "wb",
        ) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if read_regular(path) != raw:
            raise ValueError("Local publication readback differs")
    for name, raw in (("bundle.tar.gz", archive), ("BUILD.json", canonical(receipt))):
        with os.fdopen(
            os.open(
                destination / name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o400,
            ),
            "wb",
        ) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    for path in sorted((p for p in bundle.rglob("*") if p.is_dir()), reverse=True):
        path.chmod(0o500)
    bundle.chmod(0o500)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.output)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "files"},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
