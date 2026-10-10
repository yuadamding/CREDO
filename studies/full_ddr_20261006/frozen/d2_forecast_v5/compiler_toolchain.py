"""Bound, image-independent host compiler for Triton's CUDA launcher stubs.

No package installation, image mutation, training, or input-data access. The
upstream archive is verified before extraction into the worker's own emptyDir.
"""
from __future__ import annotations

import ctypes
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import stat
import subprocess
import tarfile

ARCHIVE_SHA = "d45312e61ebcc48032b77bc4cf7fd6915c11fa16e4aad116b66c9468211230ea"
ARCHIVE_BYTES = 47082308
ROOT_NAME = "zig-linux-x86_64-0.13.0"
WORKSPACE = Path("/tmp/credo-compiler")
MAX_UNPACKED = 512 * 1024**2
MAX_MEMBERS = 20000
TARGET = "x86_64-linux-gnu.2.17"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def checked_file(path, expected=None, *, maximum=100_000_000):
    path = Path(path).absolute()
    for parent in path.parents:
        if not stat.S_ISDIR(parent.lstat().st_mode):
            raise RuntimeError("Unsafe compiler-source ancestor")
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
        raise RuntimeError("Bounded regular compiler source required")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        opened = os.fstat(stream.fileno())
        raw = stream.read(maximum + 1)
        closed = os.fstat(stream.fileno())
    keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    def signature(value):
        return tuple(getattr(value, key) for key in keys)
    if not (signature(before) == signature(opened) == signature(closed) == signature(path.lstat())):
        raise RuntimeError("Compiler source changed during read")
    if len(raw) > maximum or (expected is not None and digest(raw) != expected):
        raise RuntimeError("Compiler-source hash differs")
    return raw


def archive_inventory(raw, *, root_name=ROOT_NAME, maximum=MAX_UNPACKED):
    """Validate and hash every regular member without extracting any bytes."""
    entries, seen, total = {}, set(), 0
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:xz") as archive:
        for item in archive:
            name = item.name
            path = PurePosixPath(name)
            if (not name or path.is_absolute() or ".." in path.parts or path.as_posix() != name
                    or path.parts[0] != root_name or name in seen
                    or not (item.isdir() or item.isfile()) or len(seen) >= MAX_MEMBERS):
                raise RuntimeError("Unsafe standalone compiler archive member")
            seen.add(name)
            if item.isdir():
                continue
            total += item.size
            if item.size < 0 or total > maximum:
                raise RuntimeError("Standalone compiler unpacking cap exceeded")
            source = archive.extractfile(item)
            if source is None:
                raise RuntimeError("Compiler archive member unavailable")
            h, size = hashlib.sha256(), 0
            while True:
                chunk = source.read(1024**2)
                if not chunk:
                    break
                size += len(chunk)
                h.update(chunk)
            if size != item.size:
                raise RuntimeError("Truncated standalone compiler member")
            entries[name] = dict(sha256=h.hexdigest(), size_bytes=size)
    if not entries or root_name + "/zig" not in entries or root_name + "/LICENSE" not in entries:
        raise RuntimeError("Incomplete standalone compiler archive")
    return entries


def unpack(raw, inventory, destination):
    """Exclusive extraction; no symlinks, mutable shared tools, or overwrite."""
    destination = Path(destination).absolute()
    for parent in destination.parents:
        if not stat.S_ISDIR(parent.lstat().st_mode):
            raise RuntimeError("Unsafe compiler workspace ancestor")
    if os.path.lexists(destination):
        raise RuntimeError("Compiler workspace exists; no overwrite or retry")
    if archive_inventory(raw) != inventory:
        raise RuntimeError("Standalone compiler member inventory differs")
    destination.mkdir(mode=0o700)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:xz") as archive:
        for item in archive:
            if not item.isfile():
                continue
            path = destination / item.name
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            record = inventory[item.name]
            h, size = hashlib.sha256(), 0
            with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb") as stream:
                source = archive.extractfile(item)
                while True:
                    chunk = source.read(1024**2)
                    if not chunk:
                        break
                    size += len(chunk)
                    h.update(chunk)
                    stream.write(chunk)
                stream.flush()
                os.fchmod(stream.fileno(), 0o500 if item.name == ROOT_NAME + "/zig" else 0o400)
            if record != dict(sha256=h.hexdigest(), size_bytes=size):
                raise RuntimeError("Compiler extraction hash differs")
    return destination / ROOT_NAME / "zig"


def wrapper_source(binary, mode, cache, *, python_executable="/opt/credo/bin/python"):
    if mode not in {"cc", "c++"} or not Path(binary).is_absolute() or not Path(cache).is_absolute():
        raise RuntimeError("Invalid bound compiler wrapper")
    # Absolute exec plus inherited argv: no shell expansion or mutable PATH.
    # Restrict compiler child work to the same eight CPU cores for this worker.
    return ("#!" + python_executable + "\nimport os,sys\n"
        + "os.environ['ZIG_GLOBAL_CACHE_DIR']=" + repr(str(cache)) + "\n"
        + "os.environ['ZIG_LOCAL_CACHE_DIR']=" + repr(str(cache)) + "\n"
        + "if hasattr(os,'sched_getaffinity'):\n"
        + " os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:8])\n"
        + "os.execv(" + repr(str(binary)) + ",[" + repr(str(binary)) + ","
        + repr(mode) + ",'-target'," + repr(TARGET) + "]+sys.argv[1:])\n").encode()


def bind_wrappers(binary, workspace, *, python_executable="/opt/credo/bin/python"):
    workspace = Path(workspace)
    cache = workspace / "cache"
    cache.mkdir(mode=0o700)
    refs = {}
    for filename, mode in (("cc", "cc"), ("cxx", "c++")):
        path = workspace / filename
        raw = wrapper_source(binary, mode, cache, python_executable=python_executable)
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o500), "wb") as stream:
            stream.write(raw)
            stream.flush()
        refs[filename] = dict(path=str(path), sha256=digest(raw))
    os.environ.update(CC=str(workspace / "cc"), CXX=str(workspace / "cxx"),
        ZIG_GLOBAL_CACHE_DIR=str(cache), ZIG_LOCAL_CACHE_DIR=str(cache), TORCHINDUCTOR_COMPILE_THREADS="8")
    return refs


def native_probes(bundle, workspace):
    """Exactly two native builds: Triton's real host builder and C++ ABI."""
    import triton
    from triton.runtime import build
    if triton.__version__ != "3.2.0":
        raise RuntimeError("Pinned-image Triton 3.2.0 host builder required")
    probe = Path(workspace) / "probe"
    probe.mkdir(mode=0o700)
    extension = build._build("credo_compiler_probe", str(bundle / "compiler_probe.c"), str(probe), [], [], [])
    spec = importlib.util.spec_from_file_location("credo_compiler_probe", extension)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.value() != 307469:
        raise RuntimeError("Triton-built Python extension probe differs")
    cpp = probe / "cxx-probe.so"
    result = subprocess.run([os.environ["CXX"], str(bundle / "compiler_probe.cpp"), "-std=c++17", "-O2",
        "-shared", "-fPIC", "-o", str(cpp)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError("Bound C++ compiler probe failed: " + result.stderr.decode(errors="replace")[-3000:])
    library = ctypes.CDLL(str(cpp))
    library.credo_compiler_value.restype = ctypes.c_int
    if library.credo_compiler_value() != 307469:
        raise RuntimeError("C++ native library probe differs")
    return dict(native_builds=2, triton_version=triton.__version__,
        triton_build_function_module=build._build.__module__, python_extension_loaded=True, cpp_library_loaded=True)


def prepare_compiler(bundle, plan):
    bundle = Path(bundle)
    raw = checked_file(bundle / "COMPILER_SPEC.json", plan["compiler_spec_sha256"])
    source = json.loads(raw)
    if (source.get("sha256") != ARCHIVE_SHA or source.get("size_bytes") != ARCHIVE_BYTES
            or source.get("root_name") != ROOT_NAME or source.get("workspace") != str(WORKSPACE)
            or source.get("version") != "0.13.0" or source.get("target") != TARGET
            or source.get("maximum_unpacked_bytes") != MAX_UNPACKED or source.get("threads") != 8
            or source.get("maximum_members") != MAX_MEMBERS or platform.machine() != "x86_64"):
        raise RuntimeError("Unqualified standalone compiler contract")
    archive = checked_file(bundle / "toolchain.tar.xz", ARCHIVE_SHA)
    if len(archive) != ARCHIVE_BYTES:
        raise RuntimeError("Standalone compiler archive length differs")
    inventory_raw = checked_file(bundle / "COMPILER_INVENTORY.json", source["inventory_sha256"])
    inventory = json.loads(inventory_raw)
    for name in ("compiler_toolchain.py", "compiler_probe.c", "compiler_probe.cpp", "compiler_preflight_worker.py"):
        checked_file(bundle / name, source["probe_source_sha256"][name])
    candidates = {name: shutil.which(name) is not None for name in ("gcc", "clang", "cc", "c++")}
    binary = unpack(archive, inventory, WORKSPACE)
    wrappers = bind_wrappers(binary, WORKSPACE)
    version = subprocess.run([str(binary), "version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if version.returncode or version.stdout.strip() != b"0.13.0":
        raise RuntimeError("Standalone compiler version differs")
    probes = native_probes(bundle, WORKSPACE)
    size = sum(p.stat().st_size for p in WORKSPACE.rglob("*") if p.is_file())
    if size > source["maximum_cache_bytes"]:
        raise RuntimeError("Compiler/cache budget exceeded")
    return dict(status="compiler_dependency_surface_verified", version="0.13.0", archive_sha256=ARCHIVE_SHA,
        inventory_sha256=source["inventory_sha256"], compiler_spec_sha256=digest(raw), target=TARGET,
        python_version=platform.python_version(), torch_package_version=importlib.metadata.version("torch"),
        unmodified_image_compiler_candidates_present=candidates, wrappers=wrappers, ephemeral_bytes=size,
        cohort_access=False, scientific_optimizer_updates=0, **probes)


def verified_compiler_gate(plan, *, current_plan_sha256=None):
    """Read only the exact, completed same-image compiler predecessor."""
    path = Path(plan["compiler_preflight_gate_path"])
    raw = checked_file(path, maximum=16*1024**2)
    value = json.loads(raw)
    qualifier_sha = current_plan_sha256 if plan.get("phase") == "qualification" else plan["compiler_preflight_qualification_plan_sha256"]
    if not qualifier_sha:
        raise RuntimeError("Compiler predecessor plan is not bound")
    required = dict(status="passed_same_image_compiler_prerequisite", image=plan["image"],
        qualification_plan_sha256=qualifier_sha,
        compiler_spec_sha256=plan["compiler_spec_sha256"], compiler_archive_sha256=ARCHIVE_SHA,
        job=plan["compiler_preflight_job"], native_builds=2, data_free=True,
        GPU_requests=0, visibility_mask="none", scientific_optimizer_updates=0)
    if (any(value.get(k) != v for k, v in required.items()) or not value.get("uid")
            or value.get("cleanup_reads", [])[-2:] != [dict(job_absent=True, owned_pods=0)]*2):
        raise RuntimeError("Same-image compiler predecessor unresolved")
    complete = checked_file(path.parent / "COMPLETE.json", value["compiler_complete_sha256"], maximum=16*1024**2)
    record = json.loads(complete)
    if (record.get("status") != "compiler_dependency_surface_verified" or record.get("compiler_spec_sha256") != plan["compiler_spec_sha256"]
            or record.get("archive_sha256") != ARCHIVE_SHA or record.get("native_builds") != 2
            or record.get("python_extension_loaded") is not True or record.get("cpp_library_loaded") is not True
            or record.get("cohort_access") is not False or record.get("scientific_optimizer_updates") != 0
            or record.get("python_version") != "3.12.13" or record.get("torch_package_version") != "2.6.0+cu124"
            or record.get("triton_version") != "3.2.0"):
        raise RuntimeError("Compiler predecessor native/ABI coverage differs")
    return dict(path=str(path), sha256=digest(raw), size_bytes=len(raw))
