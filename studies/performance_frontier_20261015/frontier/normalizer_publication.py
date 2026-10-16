"""Accepted fitting cache -> one persistence-normalizer publication.

No optimizer, candidate prediction, query role or scheduler is used. This is a
worker component, not scientific launch authority. The caller must separately
admit the real operation inventory and resources. Completion follows the full
prepared-cache exit checks; failed outputs are retained and never adopted.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path

import torch

from . import (cache_storage, evaluation, fitting_factory, fitting_preparation,
               fitting_provider, geometry as geometry_module, measure_ops,
               molecular, normalization, sampling)
from .cache_storage import (_digest_stream, _json_bytes, _open_verified, _parse,
                            _path, _stamp, _write_once)
from .fitting_factory import NormalizerArtifact, bind_normalizers
from .fitting_preparation import open_prepared_cache
from .fitting_provider import ProviderLimits, _sha
from .geometry import RaggedSinkhornGeometry


MIB = 1024**2
COMPLETE_SCHEMA = "credo.frontier.normalizer-publication-complete.v1"
FILES = {"NORMALIZERS.json", "PUBLICATION.json", "COMPLETE.json"}


@dataclass(frozen=True)
class NormalizationLimits:
    provider: ProviderLimits
    maximum_paired_laws: int
    maximum_artifact_bytes: int
    output_cap_bytes: int

    def __post_init__(self):
        if not isinstance(self.provider, ProviderLimits):
            raise TypeError("Explicit cache-provider bounds required")
        if any(type(v) is not int or v < 1 for v in
               (self.maximum_paired_laws, self.maximum_artifact_bytes, self.output_cap_bytes)):
            raise ValueError("Positive integer normalization/output bounds required")
        # Per-law scalar evidence is data, not small metadata. Do not force a
        # real complete catalog into the cache's 32-MiB metadata-file limit.
        if (self.maximum_paired_laws > 3*26504 or self.maximum_artifact_bytes > 256*MIB
                or self.output_cap_bytes < self.maximum_artifact_bytes + 3*MIB):
            raise ValueError("Artifact plus publication/failure allowances exceed bounds")


def _sources():
    modules = (normalization, sampling, evaluation, measure_ops, molecular,
               geometry_module, fitting_factory, fitting_preparation,
               cache_storage, fitting_provider)
    paths = [Path(m.__file__) for m in modules] + [Path(__file__)]
    return {"frontier/"+p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def _put(fd, name, raw):
    sha = hashlib.sha256(raw).hexdigest()
    stamp = _write_once(fd, name, (raw,), expected_sha256=sha, expected_bytes=len(raw))
    return dict(relative_path=name, sha256=sha, size_bytes=len(raw)), stamp


def _inventory(cache, geometry, particles, block, limits, device):
    if not isinstance(geometry, RaggedSinkhornGeometry):
        raise TypeError("Retained geometry required")
    if (type(particles) is not int or type(block) is not int or particles < 1
            or block < 1 or block > 128):
        raise ValueError("Explicit bounded particle/block dimensions required")
    dev = torch.device(device)
    if cache.manifest.synthetic_fixture:
        if (dev.type != "cpu" or particles > 128 or len(cache.paired) > 64
                or limits.maximum_artifact_bytes > 8*MIB
                or limits.output_cap_bytes > 1024*MIB
                or limits.provider.maximum_host_bytes > 8*1024*MIB):
            raise ValueError("Fabricated normalization retains bounded CPU scope")
    elif particles != 512 or dev.type not in ("cpu", "cuda") or (dev.type == "cuda" and dev.index is None):
        raise ValueError("Scientific scales require512 particles and an explicit device")
    if len(cache.paired) > limits.maximum_paired_laws:
        raise ValueError("Complete paired support exceeds admitted law count; no truncation")
    sizes = [int(cache.populations[d, "Stim48hr"].positive_counts[g])
             for i in cache.paired for d, g in (cache.laws[i],)]
    costs = [(particles+n)**2 for n in sizes]
    if not sizes or max(costs) > geometry.options.maximum_geometry_elements:
        raise ValueError("Full endpoint support exceeds retained geometry workspace")
    return dict(paired_laws=len(sizes), fitting_donors=list(cache.manifest.fold.training_donors),
        catalog_categories=len(cache.manifest.catalog), source_particles=particles,
        pairwise_block=block, observed_endpoint_atoms=sum(sizes), maximum_endpoint_atoms=max(sizes),
        nominal_four_cost_matrix_elements=sum(costs),
        geometry_options=asdict(geometry.options), allocator_peak_or_runtime_measured=False)


def publish_normalizers(preparation_root, destination, *, preparation_complete_sha256,
        preparation_output_cap_bytes, limits, geometry, study_id, study_seed,
        protocol_payload, source_particles=512, block=128, device="cpu"):
    """Compute once, reconcile saved scores, and publish without recomputation.

    No numeric floor, changed loss or favorable subset is introduced. All14
    scales must pass the existing strict-positive rule. Scalar reductions are
    independently rebound through the same consumer used by the fitting factory.
    """
    if not isinstance(limits, NormalizationLimits):
        raise TypeError("Explicit normalization resource limits required")
    limits.__post_init__()
    _sha(preparation_complete_sha256)
    if type(protocol_payload) is not bytes or hashlib.sha256(protocol_payload).hexdigest() != normalization.PROTOCOL_SHA256:
        raise ValueError("Exact frozen normalization protocol required")
    root, prepared = _path(destination), _path(preparation_root)
    if root == prepared or prepared in root.parents or root in prepared.parents:
        raise ValueError("Normalizer output must be separate from its input preparation")
    if os.path.lexists(root):
        raise FileExistsError("Normalizer destination exists; no overwrite or automatic retry")
    source_files = _sources()
    parent = os.open(_path(root.parent), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    fd = None
    try:
        with open_prepared_cache(prepared, expected_complete_sha256=preparation_complete_sha256,
                output_cap_bytes=preparation_output_cap_bytes, limits=limits.provider) as mapped:
            operations = _inventory(mapped, geometry, source_particles, block, limits, device)
            if _stamp(os.fstat(parent)) != _stamp(root.parent.lstat()):
                raise ValueError("Normalizer parent changed while opening")
            os.mkdir(root.name, 0o700, dir_fd=parent)
            fd = os.open(root.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            os.fsync(parent)
            input_stamps = {prepared: _stamp(prepared.lstat()), prepared/"cache": mapped._root_stamp}
            input_stamps.update({prepared/"cache"/x.name: x.stamp for x in mapped._leases})
            input_stamps.update({prepared/name: _stamp((prepared/name).lstat())
                                 for name in ("PREPARATION.json", "COMPLETE.json")})
            result = normalization.compute_normalizers(mapped, geometry, study_id=study_id,
                study_seed=study_seed, producer_sha256=source_files["frontier/normalization.py"],
                sampler_source_sha256=source_files["frontier/sampling.py"], protocol_payload=protocol_payload,
                device=device, source_particles=source_particles, block=block)
            raw = (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()
            if len(raw) > limits.maximum_artifact_bytes:
                raise ValueError("Complete normalizer evidence exceeds artifact cap; no row deletion")
            artifact = NormalizerArtifact(raw, hashlib.sha256(raw).hexdigest(), limits.maximum_artifact_bytes)
            _, _, content_sha = bind_normalizers(mapped, artifact, study_id=study_id,
                                                 study_seed=study_seed, geometry=geometry)
            manifest_sha = mapped.manifest.identity
            # Exit rehashes every array and both enclosing preparation records.
        if mapped.verification.get("exit_full_hash_verified") is not True:
            raise ValueError("Full prepared-cache exit verification required")
        def check_inputs():
            if any(_stamp(_path(p).lstat()) != s for p,s in input_stamps.items()):
                raise ValueError("Prepared input identity changed before normalizer completion")
            if _sources() != source_files:
                raise ValueError("Normalizer implementation changed during publication")
        check_inputs()
        normalizer_ref, normalizer_stamp = _put(fd, "NORMALIZERS.json", raw)
        evidence = dict(schema="credo.frontier.normalizer-publication.v1", status="complete_calculation",
            preparation_root=str(prepared), preparation_complete_sha256=preparation_complete_sha256,
            cache_manifest_sha256=manifest_sha, operations=operations, limits=asdict(limits),
            cache_verification=dict(mapped.verification), implementation_sources=source_files,
            normalizer_content_sha256=content_sha, normalizer_file=normalizer_ref,
            synthetic_fixture=result["synthetic_fixture"], scalar_reconciliation_passed=True,
            normalizer_calculations=1, optimizer_steps=0, scientific_updates=0,
            query_outcomes_opened=False, scientific_launch_authority=False)
        publication = _json_bytes(evidence)
        if len(publication) > MIB:
            raise ValueError("Publication receipt exceeds reserved metadata allowance")
        pub_ref, pub_stamp = _put(fd, "PUBLICATION.json", publication)
        complete = dict(schema=COMPLETE_SCHEMA, status="complete", files=[normalizer_ref, pub_ref],
            cache_manifest_sha256=manifest_sha, normalizer_content_sha256=content_sha,
            output_cap_bytes=limits.output_cap_bytes, total_bytes=0, scientific_launch_authority=False)
        while True:
            final = _json_bytes(complete)
            total = len(raw)+len(publication)+len(final)
            if complete["total_bytes"] == total: break
            complete["total_bytes"] = total
        if len(final) > MIB or total+MIB > limits.output_cap_bytes:
            raise ValueError("Normalizer output and failure reserve exceed cap")
        def check_output(expected):
            current, opened = _path(root).lstat(), os.fstat(fd)
            if ((current.st_dev, current.st_ino, current.st_mode) !=
                    (opened.st_dev, opened.st_ino, opened.st_mode)
                    or set(os.listdir(fd)) != expected
                    or any(_stamp(os.stat(n, dir_fd=fd, follow_symlinks=False)) != s
                           for n,s in (("NORMALIZERS.json", normalizer_stamp), ("PUBLICATION.json", pub_stamp)))):
                raise ValueError("Normalizer output identity changed during publication")
        check_inputs(); check_output(FILES-{"COMPLETE.json"})
        complete_ref, _ = _put(fd, ".COMPLETE.pending", final)
        os.link(".COMPLETE.pending", "COMPLETE.json", src_dir_fd=fd, dst_dir_fd=fd, follow_symlinks=False)
        os.unlink(".COMPLETE.pending", dir_fd=fd); os.fsync(fd)
        check_inputs(); check_output(FILES)
        # The atomic link changes nlink/ctime. Reopen the final inode by its
        # expected bytes instead of comparing its temporary-name timestamp.
        completion_check = _open_verified(fd, "COMPLETE.json", complete_ref["sha256"],
                                          size=len(final), maximum=MIB)
        completion_check.stream.close()
        return dict(root=str(root), complete_sha256=complete_ref["sha256"], total_bytes=total,
            normalizer_file_sha256=artifact.file_sha256, normalizer_content_sha256=content_sha,
            cache_manifest_sha256=manifest_sha, scientific_launch_authority=False)
    except BaseException as error:
        if fd is not None:
            failure = _json_bytes(dict(status="failed", exception_type=type(error).__name__, automatic_retry=False))
            try: _put(fd, "FAILED.json", failure)
            except BaseException: pass
        raise
    finally:
        if fd is not None: os.close(fd)
        os.close(parent)


def load_normalizer_artifact(root, *, expected_complete_sha256, expected_cache_manifest_sha256,
                             maximum_artifact_bytes, output_cap_bytes):
    """Read bounded immutable bytes, not raw populations or new calculations.

    The fitting factory must still bind the returned artifact to its current
    cache, representation, sampler and geometry. This loader checks publication.
    """
    _sha(expected_complete_sha256); _sha(expected_cache_manifest_sha256)
    if (type(maximum_artifact_bytes) is not int or not 1 <= maximum_artifact_bytes <= 256*MIB
            or type(output_cap_bytes) is not int or output_cap_bytes < 1):
        raise ValueError("Explicit normalizer input/output bounds required")
    root = _path(root)
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    leases = []
    try:
        stamp = _stamp(os.fstat(fd))
        if _stamp(root.lstat()) != stamp or set(os.listdir(fd)) != FILES:
            raise ValueError("Exact accepted normalizer inventory required")
        completed = _open_verified(fd, "COMPLETE.json", expected_complete_sha256, maximum=min(MIB,output_cap_bytes))
        leases.append(completed)
        complete = _parse(completed.stream.read())
        if (complete.get("schema") != COMPLETE_SCHEMA or complete.get("status") != "complete"
                or complete.get("cache_manifest_sha256") != expected_cache_manifest_sha256
                or complete.get("scientific_launch_authority") is not False
                or any(type(complete.get(k)) is not int or complete[k] < 1 for k in ("total_bytes", "output_cap_bytes"))
                or complete["total_bytes"] > min(output_cap_bytes, complete["output_cap_bytes"])):
            raise ValueError("Normalizer completion binding/status/cap differs")
        refs = complete.get("files", [])
        if (not isinstance(refs, list) or len(refs) != 2
                or [x.get("relative_path") for x in refs] != ["NORMALIZERS.json", "PUBLICATION.json"]):
            raise ValueError("Exact normalizer and publication files required")
        for ref, cap in zip(refs, (maximum_artifact_bytes, MIB)):
            leases.append(_open_verified(fd, ref["relative_path"], ref["sha256"], size=ref["size_bytes"], maximum=cap))
        raw = leases[1].stream.read()
        record = _parse(leases[2].stream.read())
        if (record.get("schema") != "credo.frontier.normalizer-publication.v1"
                or record.get("status") != "complete_calculation"
                or record.get("cache_manifest_sha256") != expected_cache_manifest_sha256
                or record.get("normalizer_file") != refs[0]
                or record.get("normalizer_content_sha256") != complete.get("normalizer_content_sha256")
                or record.get("cache_verification", {}).get("exit_full_hash_verified") is not True
                or record.get("scalar_reconciliation_passed") is not True
                or record.get("normalizer_calculations") != 1 or record.get("optimizer_steps") != 0
                or record.get("scientific_updates") != 0 or record.get("query_outcomes_opened") is not False
                or record.get("scientific_launch_authority") is not False
                or sum(x.size for x in leases) != complete["total_bytes"]):
            raise ValueError("Normalizer publication evidence does not reconcile")
        for lease in leases:
            if (_digest_stream(lease.stream) != lease.sha256
                    or _stamp(os.fstat(lease.stream.fileno())) != lease.stamp
                    or _stamp(os.stat(lease.name, dir_fd=fd, follow_symlinks=False)) != lease.stamp):
                raise ValueError("Normalizer input changed while reading")
        if _stamp(root.lstat()) != stamp or set(os.listdir(fd)) != FILES:
            raise ValueError("Normalizer root changed while reading")
        return NormalizerArtifact(raw, refs[0]["sha256"], maximum_artifact_bytes)
    finally:
        for lease in leases: lease.stream.close()
        os.close(fd)
