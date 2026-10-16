"""Exclusive fold-cache publication and descriptor-bound read-only mmap leases.

Numeric .npy v2 arrays plus strict JSON only; no pickle or path references to a
shared store. This first layout copies raw summaries into EACH fold, so reuse
of computation does not imply storage sharing or admission within 384 GiB. The
caller must reserve actual live output bytes before any scientific invocation.

The complete FoldCacheManifest identity and COMPLETE content identity are
provided by the caller. Array file bytes, canonical headers and array-content
hashes are checked before/while constructing the existing FoldFittingCache.
Loaded cache.assert_unchanged() extends the provider's consumption boundary
with live file-descriptor/path identity and exact directory inventory checks.
Context exit rehashes every file by default; verification I/O is reported, not
confused with training work. These leases detect changes, not hostile OS writes.

No decoder is loaded or qualified here. The caller must separately load the
selected artifact and call cache.verify_decoder for EVERY family. Neither
publication nor a syntactically valid hash grants data or execution authority.
Failed publications are left incomplete; there is no deletion/retry/recovery.
Only the writer's own temporary COMPLETE name is unlinked after its atomic,
exclusive final link. Existing destination directories/files are never reused.
"""

from dataclasses import asdict, dataclass, fields
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
from types import MappingProxyType

import numpy as np

from .fitting_provider import (CatalogGuide, FoldCacheManifest, FoldFittingCache,
    PopulationArrays, PopulationReference, ProviderLimits, _array_hash, _sha)
from .native_inputs import FoldInputs
from .training_owner import ObservationBinding


STORAGE_SCHEMA = "credo.frontier.fold-cache-storage.v1"
COMPLETE_SCHEMA = "credo.frontier.fold-cache-storage-complete.v1"
ARRAY_FIELDS = ("captured_counts", "positive_counts", "offsets", "states", "mean_compositions")
MAXIMUM_METADATA_BYTES = 32 * 1024**2
IO_BLOCK = 8 * 1024**2


def _path(value):
    path = Path(value)
    if (not path.is_absolute() or ".." in path.parts or not path.name
            or any(p.is_symlink() for p in (path, *path.parents))):
        raise ValueError("Explicit absolute nonsymlink cache path without traversal required")
    return path


def _stamp(info):
    return tuple(getattr(info, k) for k in
        ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode", "st_nlink"))


def _json_bytes(value):
    result = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()
    if len(result) > MAXIMUM_METADATA_BYTES:
        raise ValueError("Cache metadata exceeds its 32 MiB bound")
    return result


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate cache metadata key")
        result[key] = value
    return result


def _parse(raw):
    value = json.loads(raw, object_pairs_hook=_unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite cache metadata")))
    if not isinstance(value, dict):
        raise ValueError("Cache metadata must be an object")
    return value


def _digest_stream(stream):
    stream.seek(0)
    digest = hashlib.sha256()
    while chunk := stream.read(IO_BLOCK):
        digest.update(chunk)
    stream.seek(0)
    return digest.hexdigest()


def _header(dtype, shape):
    buffer = io.BytesIO()
    np.lib.format.write_array_header_2_0(buffer,
        dict(descr=np.dtype(dtype).str, fortran_order=False, shape=tuple(shape)))
    return buffer.getvalue()


def _blocks(array):
    if array.size:
        view = memoryview(array).cast("B")
        for start in range(0, len(view), IO_BLOCK):
            yield view[start:start+IO_BLOCK]


def _array_record(index, field, array):
    header = _header(array.dtype, array.shape)
    digest = hashlib.sha256(header)
    for chunk in _blocks(array):
        digest.update(chunk)
    return dict(population_index=index, field=field, relative_path=f"p{index:03d}.{field}.npy",
        dtype=array.dtype.str, shape=list(array.shape), fortran_order=False, data_offset=len(header),
        nbytes=array.nbytes, file_bytes=len(header)+array.nbytes,
        header_sha256=hashlib.sha256(header).hexdigest(), array_sha256=_array_hash(array),
        file_sha256=digest.hexdigest())


def _write_once(root_fd, name, chunks, *, expected_sha256, expected_bytes):
    fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=root_fd)
    with os.fdopen(fd, "w+b") as stream:
        for chunk in chunks:
            stream.write(chunk)
        stream.flush()
        os.fsync(stream.fileno())
        before = os.fstat(stream.fileno())
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or before.st_size != expected_bytes or _digest_stream(stream) != expected_sha256
                or _stamp(os.fstat(stream.fileno())) != _stamp(before)
                or _stamp(os.stat(name, dir_fd=root_fd, follow_symlinks=False)) != _stamp(before)):
            raise ValueError("Published file readback differs: " + name)
    os.fsync(root_fd)
    return _stamp(before)


def _completion(manifest_sha256, producer_sha256, metadata, array_bytes, array_count, output_cap_bytes):
    value = dict(schema=COMPLETE_SCHEMA, status="complete", manifest_sha256=manifest_sha256,
        producer_sha256=producer_sha256, manifest_file=dict(relative_path="MANIFEST.json",
            sha256=hashlib.sha256(metadata).hexdigest(), size_bytes=len(metadata)),
        array_files=array_count, output_cap_bytes=output_cap_bytes, total_bytes=0)
    while True:
        encoded = _json_bytes(value)
        total = array_bytes+len(metadata)+len(encoded)
        if value["total_bytes"] == total:
            return value, encoded
        value["total_bytes"] = total


def publish_fold_cache(cache, destination, *, expected_manifest_sha256, producer_sha256, output_cap_bytes):
    """Publish a validated admitted cache into an exclusive NEW directory.

    The byte cap includes every .npy header and both JSON files. The transient
    COMPLETE name is a hard link to the same inode, not another content copy.
    Preflight rejects an insufficient cap before creating the destination.
    Failed outputs are preserved and never treated as restartable publications.
    """
    if not isinstance(cache, FoldFittingCache):
        raise TypeError("An admitted FoldFittingCache is required")
    _sha(expected_manifest_sha256); _sha(producer_sha256)
    if cache.manifest.identity != expected_manifest_sha256:
        raise ValueError("Expected full cache manifest identity differs")
    if type(output_cap_bytes) is not int or output_cap_bytes < 1:
        raise ValueError("Explicit positive output-byte cap required")
    cache.assert_unchanged()
    root = _path(destination)
    if os.path.lexists(root):
        raise FileExistsError("Publication destination already exists; no overwrite or resume")
    populations = tuple(cache.populations[r.donor, r.condition] for r in cache.manifest.populations)
    records = []
    for index, population in enumerate(populations):
        if (population.raw_identity(catalog_sha256=cache.manifest.catalog_sha256,
                feature_axis_sha256=cache.manifest.feature_axis_sha256) != population.reference.raw_summary_sha256
                or population.latent_identity(catalog_sha256=cache.manifest.catalog_sha256,
                    observation_binding=cache.manifest.observation_binding) != population.reference.latent_sha256):
            raise ValueError("Source array content changed before publication")
        records.extend(_array_record(index, field, getattr(population, field)) for field in ARRAY_FIELDS)
    metadata = _json_bytes(dict(schema=STORAGE_SCHEMA, manifest=asdict(cache.manifest),
        manifest_sha256=expected_manifest_sha256, producer_sha256=producer_sha256, arrays=records,
        layout="per_fold_copies_no_shared_raw_store"))
    completion, complete_bytes = _completion(expected_manifest_sha256, producer_sha256, metadata,
        sum(r["file_bytes"] for r in records), len(records), output_cap_bytes)
    if completion["total_bytes"] > output_cap_bytes:
        raise ValueError("Complete publication including headers/metadata exceeds output cap")
    parent = _path(root.parent)
    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    root_fd = None
    try:
        if _stamp(os.fstat(parent_fd)) != _stamp(parent.lstat()):
            raise ValueError("Publication parent changed while opening")
        os.mkdir(root.name, 0o700, dir_fd=parent_fd)
        root_fd = os.open(root.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
        os.fsync(parent_fd)
        published_stamps = {}
        for record in records:
            array = getattr(populations[record["population_index"]], record["field"])
            def chunks():
                yield _header(array.dtype, array.shape)
                yield from _blocks(array)
            published_stamps[record["relative_path"]] = _write_once(root_fd, record["relative_path"], chunks(),
                expected_sha256=record["file_sha256"], expected_bytes=record["file_bytes"])
        published_stamps["MANIFEST.json"] = _write_once(root_fd, "MANIFEST.json", (metadata,),
            expected_sha256=hashlib.sha256(metadata).hexdigest(), expected_bytes=len(metadata))
        cache.assert_unchanged()
        expected_names = {r["relative_path"] for r in records} | {"MANIFEST.json"}
        if set(os.listdir(root_fd)) != expected_names:
            raise ValueError("Unexpected file before cache completion")
        for name, stamp in published_stamps.items():
            if _stamp(os.stat(name, dir_fd=root_fd, follow_symlinks=False)) != stamp:
                raise ValueError("Published file changed before completion: " + name)
        complete_sha = hashlib.sha256(complete_bytes).hexdigest()
        _write_once(root_fd, ".COMPLETE.pending", (complete_bytes,), expected_sha256=complete_sha, expected_bytes=len(complete_bytes))
        # Atomic no-clobber publication; unlike rename, link fails if another
        # COMPLETE name exists. Remove only this writer's own temporary alias.
        os.link(".COMPLETE.pending", "COMPLETE.json", src_dir_fd=root_fd, dst_dir_fd=root_fd, follow_symlinks=False)
        os.unlink(".COMPLETE.pending", dir_fd=root_fd)
        os.fsync(root_fd)
        if set(os.listdir(root_fd)) != expected_names | {"COMPLETE.json"}:
            raise ValueError("Unexpected file at cache completion")
        for name, stamp in published_stamps.items():
            if _stamp(os.stat(name, dir_fd=root_fd, follow_symlinks=False)) != stamp:
                raise ValueError("Published file changed at completion: " + name)
        total = sum(os.stat(name, dir_fd=root_fd, follow_symlinks=False).st_size for name in os.listdir(root_fd))
        if total != completion["total_bytes"] or _stamp(os.fstat(root_fd)) != _stamp(_path(root).lstat()):
            raise ValueError("Final publication size/path changed")
        return dict(root=str(root), manifest_sha256=expected_manifest_sha256, producer_sha256=producer_sha256,
            complete_sha256=complete_sha, complete_bytes=len(complete_bytes), total_bytes=total,
            array_files=len(records), output_cap_bytes=output_cap_bytes,
            readback_verified=True, scientific_launch_authority=False)
    finally:
        if root_fd is not None:
            os.close(root_fd)
        os.close(parent_fd)


@dataclass(frozen=True)
class _FileLease:
    name: str
    stream: object
    stamp: tuple
    sha256: str
    size: int


def _open_verified(root_fd, name, expected_sha256, *, size=None, maximum):
    _sha(expected_sha256)
    before = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size < 1
            or before.st_size > maximum or (size is not None and before.st_size != size)):
        raise ValueError("Expected regular single-link file size required: " + name)
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=root_fd)
    stream = os.fdopen(fd, "rb")
    try:
        stamp = _stamp(before)
        if _stamp(os.fstat(stream.fileno())) != stamp or _digest_stream(stream) != expected_sha256:
            raise ValueError("Cache file changed or SHA256 differs: " + name)
        if _stamp(os.fstat(stream.fileno())) != stamp or _stamp(os.stat(name, dir_fd=root_fd, follow_symlinks=False)) != stamp:
            raise ValueError("Cache file changed during verification: " + name)
        return _FileLease(name, stream, stamp, expected_sha256, before.st_size)
    except BaseException:
        stream.close()
        raise


def _decode_manifest(value):
    if not isinstance(value, dict) or set(value) != {field.name for field in fields(FoldCacheManifest)}:
        raise ValueError("Exact FoldCacheManifest fields required")
    v = dict(value)
    fold = dict(v["fold"])
    fold["training_donors"] = tuple(fold["training_donors"])
    v["fold"] = FoldInputs(**fold)
    v["catalog"] = tuple(CatalogGuide(**row) for row in v["catalog"])
    v["observation_binding"] = ObservationBinding(**v["observation_binding"])
    v["populations"] = tuple(PopulationReference(**row) for row in v["populations"])
    return FoldCacheManifest(**v)


def _validate_array_record(record, index, field, manifest, output_cap):
    keys = {"population_index", "field", "relative_path", "dtype", "shape", "fortran_order",
            "data_offset", "nbytes", "file_bytes", "header_sha256", "array_sha256", "file_sha256"}
    if (not isinstance(record, dict) or set(record) != keys or type(record["population_index"]) is not int
            or record["population_index"] != index or record["field"] != field
            or record["relative_path"] != f"p{index:03d}.{field}.npy" or record["fortran_order"] is not False):
        raise ValueError("Fixed ordered array inventory required; no traversal or unexpected fields")
    for key in ("header_sha256", "array_sha256", "file_sha256"):
        _sha(record[key])
    dtype = np.dtype(np.float32 if field == "states" else np.float64 if field == "mean_compositions" else np.int64)
    shape = record["shape"]
    g, j = len(manifest.catalog), manifest.features
    expected_shape = (g, j) if field == "mean_compositions" else (g+1,) if field == "offsets" else (g,)
    if (record["dtype"] != dtype.str or type(shape) is not list or not shape
            or any(type(n) is not int or n < 0 for n in shape)
            or (field == "states" and (len(shape) != 2 or shape[1] != 48))
            or (field != "states" and tuple(shape) != expected_shape)):
        raise ValueError("Unexpected exact numeric dtype/shape; no object arrays or pickle")
    nbytes = math.prod(shape)*dtype.itemsize
    header = _header(dtype, shape)
    if (any(type(record[k]) is not int for k in ("data_offset", "nbytes", "file_bytes"))
            or record["nbytes"] != nbytes or record["data_offset"] != len(header)
            or record["file_bytes"] != len(header)+nbytes or record["file_bytes"] > output_cap
            or record["header_sha256"] != hashlib.sha256(header).hexdigest()):
        raise ValueError("Array bytes/header/cap disagree before mmap")
    return dtype, tuple(shape), header


class MappedFoldCache(FoldFittingCache):
    """Use only inside its context; never retain array views past that context.

    Array mmap objects are read-only and descriptor-backed. Closing invalidates
    this cache and drops its array references; borrowed external views may keep
    an OS mapping alive until released, but are no longer admitted for use.
    """

    @property
    def publication_identity(self):
        """SHA256 of the exact caller-bound, verified COMPLETE bytes."""
        return self._publication_reference["complete_sha256"]

    @property
    def publication_reference(self):
        """Immutable publication identities, sizes and source reference."""
        return self._publication_reference

    def _lease_check(self, *, full_hash=False):
        if self._closed or self._pid != os.getpid():
            raise RuntimeError("A live same-process mapped-cache lease is required")
        root = _path(self._root)
        if _stamp(root.lstat()) != self._root_stamp or _stamp(os.fstat(self._root_fd)) != self._root_stamp:
            raise ValueError("Cache directory identity changed")
        if set(os.listdir(self._root_fd)) != self._names:
            raise ValueError("Cache directory inventory changed")
        for file in self._leases:
            if (file.stream.closed or _stamp(os.fstat(file.stream.fileno())) != file.stamp
                    or _stamp(os.stat(file.name, dir_fd=self._root_fd, follow_symlinks=False)) != file.stamp):
                raise ValueError("Cache file identity changed: " + file.name)
            if full_hash:
                if _digest_stream(file.stream) != file.sha256:
                    raise ValueError("Cache content changed on full rehash: " + file.name)
                self.verification["exit_file_hash_bytes"] += file.size
                if _stamp(os.fstat(file.stream.fileno())) != file.stamp:
                    raise ValueError("Cache changed during exit rehash: " + file.name)
        if full_hash:
            # Recheck every path and the root after the whole hash pass, so a
            # file replacement during an earlier file's read cannot escape.
            self._lease_check(full_hash=False)

    def assert_unchanged(self):
        if not self._entered:
            raise RuntimeError("Consume the mapped cache only inside its context manager")
        self._lease_check()
        super().assert_unchanged()

    def __enter__(self):
        if self._entered:
            raise RuntimeError("Mapped-cache context cannot be entered twice")
        try:
            self._lease_check()
            self._entered = True
            self.assert_unchanged()
            return self
        except BaseException:
            try:
                self.close(verify_hashes=False)
            except BaseException:
                pass  # close's finally released descriptors; retain first error
            raise

    def close(self, *, verify_hashes=True):
        if self._closed:
            return
        try:
            self._lease_check(full_hash=verify_hashes)
            super().assert_unchanged()
            self.verification["exit_full_hash_verified"] = verify_hashes
        finally:
            self._entered, self._closed = False, True
            for file in self._leases:
                file.stream.close()
            os.close(self._root_fd)
            # Do not forcibly close an mmap while an external ndarray view may
            # reference it: that can segfault. All such views are out of scope.
            self.populations = MappingProxyType({})
            self.pool_laws, self.pool_truth = (), ()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close(verify_hashes=True)
        return False


def load_fold_cache(root, *, expected_manifest_sha256, expected_complete_sha256, output_cap_bytes, limits):
    """Open/verify exact files, then return a context-managed read-only cache.

    No mmap is created until its entire file hash and exact canonical header,
    payload length, dtype and shape have been checked. Existing provider content
    validation reconciles actual array hashes with the full caller-bound manifest.
    Entry and default exit hash-I/O bytes are recorded in cache.verification.
    """
    _sha(expected_manifest_sha256); _sha(expected_complete_sha256)
    if not isinstance(limits, ProviderLimits) or type(output_cap_bytes) is not int or output_cap_bytes < 1:
        raise ValueError("Explicit provider limits and positive publication-byte cap required")
    root = _path(root)
    before = root.lstat()
    if not stat.S_ISDIR(before.st_mode):
        raise ValueError("Cache root must be a regular directory")
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    leases = []
    try:
        if _stamp(os.fstat(root_fd)) != _stamp(before):
            raise ValueError("Cache directory changed while opening")
        complete = _open_verified(root_fd, "COMPLETE.json", expected_complete_sha256,
            maximum=min(output_cap_bytes, MAXIMUM_METADATA_BYTES))
        leases.append(complete)
        completion = _parse(complete.stream.read())
        keys = {"schema", "status", "manifest_sha256", "producer_sha256", "manifest_file",
                "array_files", "output_cap_bytes", "total_bytes"}
        if (set(completion) != keys or completion["schema"] != COMPLETE_SCHEMA or completion["status"] != "complete"
                or completion["manifest_sha256"] != expected_manifest_sha256
                or any(type(completion[k]) is not int or completion[k] < 1 for k in ("total_bytes", "output_cap_bytes", "array_files"))
                or completion["total_bytes"] > min(output_cap_bytes, completion["output_cap_bytes"])):
            raise ValueError("Cache completion identity/status/byte cap differs")
        _sha(completion["producer_sha256"])
        ref = completion["manifest_file"]
        if (not isinstance(ref, dict) or set(ref) != {"relative_path", "sha256", "size_bytes"}
                or ref["relative_path"] != "MANIFEST.json" or type(ref["size_bytes"]) is not int or ref["size_bytes"] < 1):
            raise ValueError("Exact manifest-file reference required")
        manifest_file = _open_verified(root_fd, "MANIFEST.json", ref["sha256"], size=ref["size_bytes"],
            maximum=min(output_cap_bytes, MAXIMUM_METADATA_BYTES))
        leases.append(manifest_file)
        metadata = _parse(manifest_file.stream.read())
        if (set(metadata) != {"schema", "manifest", "manifest_sha256", "producer_sha256", "arrays", "layout"}
                or metadata["schema"] != STORAGE_SCHEMA or metadata["layout"] != "per_fold_copies_no_shared_raw_store"
                or metadata["manifest_sha256"] != expected_manifest_sha256
                or metadata["producer_sha256"] != completion["producer_sha256"]):
            raise ValueError("Storage manifest identity/schema differs")
        manifest = _decode_manifest(metadata["manifest"])
        if manifest.identity != expected_manifest_sha256:
            raise ValueError("Full decoded manifest differs from caller expectation")
        records = metadata["arrays"]
        if not isinstance(records, list) or len(records) != len(manifest.populations)*len(ARRAY_FIELDS) or len(records) != completion["array_files"]:
            raise ValueError("Complete ordered array inventory required")
        plans, names, payload_bytes = [], {"MANIFEST.json", "COMPLETE.json"}, 0
        for index, reference in enumerate(manifest.populations):
            for position, field in enumerate(ARRAY_FIELDS):
                record = records[index*len(ARRAY_FIELDS)+position]
                dtype, shape, header = _validate_array_record(record, index, field, manifest, output_cap_bytes)
                plans.append((record, dtype, shape, header))
                names.add(record["relative_path"])
                payload_bytes += record["nbytes"]
        total = complete.size+manifest_file.size+sum(record["file_bytes"] for record in records)
        if total != completion["total_bytes"] or total > output_cap_bytes or payload_bytes > limits.maximum_host_bytes:
            raise ValueError("Whole cache bytes differ or exceed admitted storage/host cap")
        if set(os.listdir(root_fd)) != names:
            raise ValueError("Unexpected or missing cache files")
        arrays = []
        for record, dtype, shape, header in plans:
            file = _open_verified(root_fd, record["relative_path"], record["file_sha256"],
                                  size=record["file_bytes"], maximum=output_cap_bytes)
            leases.append(file)
            if file.stream.read(len(header)) != header:
                raise ValueError("Exact canonical numeric .npy header differs before mmap")
            array = np.memmap(file.stream, mode="r", dtype=dtype, shape=shape, offset=len(header), order="C")
            if _array_hash(array) != record["array_sha256"]:
                raise ValueError("Mapped array content hash differs")
            arrays.append(array)
        populations = tuple(PopulationArrays(reference, *arrays[i*len(ARRAY_FIELDS):(i+1)*len(ARRAY_FIELDS)])
                            for i, reference in enumerate(manifest.populations))
        cache = MappedFoldCache(manifest, populations, expected_manifest_sha256=expected_manifest_sha256, limits=limits)
        cache._root, cache._root_fd, cache._root_stamp = root, root_fd, _stamp(before)
        cache._leases, cache._names, cache._pid = tuple(leases), names, os.getpid()
        cache._closed, cache._entered = False, False
        cache._publication_reference = MappingProxyType(dict(root=str(root),
            complete_sha256=expected_complete_sha256, complete_bytes=complete.size,
            manifest_sha256=expected_manifest_sha256, manifest_file_sha256=ref["sha256"],
            producer_sha256=completion["producer_sha256"], total_bytes=total,
            array_files=len(records), output_cap_bytes=completion["output_cap_bytes"]))
        cache.verification = dict(entry_file_hash_bytes=total, entry_array_hash_bytes=payload_bytes,
            constructor_validation_array_hash_bytes=payload_bytes,
            exit_file_hash_bytes=0, exit_full_hash_verified=False,
            producer_sha256=completion["producer_sha256"], expected_complete_sha256=expected_complete_sha256,
            publication_bytes=total, readonly_mmap=True, scientific_launch_authority=False)
        cache._lease_check()
        return cache
    except BaseException:
        for file in leases:
            file.stream.close()
        os.close(root_fd)
        raise
