"""Fitting-only native capabilities -> selected-observation cache publication.

This owner composes existing loading, compilation and array publication paths.
It does not train, normalize losses, open query roles, launch workers or infer
biological culture identities. An accepted selected observation artifact and
two already-admitted fitting handles are required. Scientific use still needs
source-bound worker qualification and actual resource/output admission.

An inner cache COMPLETE proves its array publication only. The enclosing
COMPLETE is accepted only after full cache readback and final native-input and
selected-artifact checks. Failed directories remain in place, are never resumed
or overwritten, and cannot be consumed through open_prepared_cache.
"""

from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from . import (cache_compiler, cache_storage, fitting_provider, native_inputs,
               observation, observation_loader, training_owner)
from .cache_compiler import (CompilerLimits, NativeShard, SourceCompileSpec,
    compile_population, native_source_input_identity, ordered_native_chunks)
from .cache_storage import (ARRAY_FIELDS, IO_BLOCK, MAXIMUM_METADATA_BYTES,
    _digest_stream, _json_bytes, _open_verified, _parse, _path, _stamp, _write_once,
    load_fold_cache, publish_fold_cache)
from .fitting_provider import (CatalogGuide, FoldCacheManifest, FoldFittingCache,
    ProviderLimits, _hash, _text)
from .native_inputs import FoldInputs, NativeHandle
from .observation_loader import ObservationArtifactAdmission, _verified_file, load_observation
from .training_owner import observation_state_digest


SCHEMA = "credo.frontier.fitting-preparation.v1"
COMPLETE_SCHEMA = "credo.frontier.fitting-preparation-complete.v1"
ROLES = ("dynamics_source", "dynamics_supervision")
_PREPARATION_FACTORY = object()


@dataclass(frozen=True)
class FittingPopulation:
    donor: str
    condition: str
    source_id: str
    replicate_id: str

    def __post_init__(self):
        if self.donor not in ("D1", "D2", "D3", "D4") or self.condition not in ("Stim8hr", "Stim48hr"):
            raise ValueError("Exact stimulated donor/condition required")
        _text(self.source_id); _text(self.replicate_id)


@dataclass(frozen=True)
class PreparationLimits:
    compiler: CompilerLimits
    provider: ProviderLimits
    maximum_host_bytes: int
    maximum_metadata_bytes: int
    batch_rows: int

    def __post_init__(self):
        if not isinstance(self.compiler, CompilerLimits) or not isinstance(self.provider, ProviderLimits):
            raise TypeError("Existing compiler and provider limits required")
        if any(type(v) is not int or v < 1 for v in
               (self.maximum_host_bytes, self.maximum_metadata_bytes, self.batch_rows)):
            raise ValueError("Explicit positive preparation limits required")
        if (self.maximum_host_bytes > 200*1024**3
                or self.maximum_metadata_bytes > MAXIMUM_METADATA_BYTES
                or self.batch_rows > min(8192, self.compiler.maximum_chunk_rows)):
            raise ValueError("Preparation exceeds worker/metadata/reader bounds")


def _sources():
    modules = (cache_compiler, cache_storage, fitting_provider, native_inputs,
               observation, observation_loader, training_owner)
    result = {"frontier/"+Path(m.__file__).name: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
              for m in modules}
    result["frontier/fitting_preparation.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def _reader_bounds(handle):
    reader = handle._reader  # Exact retained reader; no payload or cache mutation.
    return {key: getattr(reader, key) for key in
            ("max_uncompressed_bytes", "max_cached_bytes", "max_output_bytes", "max_batch_rows")}


def _admit(handles, admission, populations, limits):
    if (type(handles) is not tuple or len(handles) != 2
            or any(type(h) is not NativeHandle for h in handles)
            or not isinstance(admission, ObservationArtifactAdmission)
            or not isinstance(limits, PreparationLimits)):
        raise TypeError("Exact two fitting handles, selected observation and preparation limits required")
    admission.__post_init__(); limits.__post_init__()
    index = handles[0].index
    fold = FoldInputs(admission.fold.fold_id, admission.fold.training_donors)
    expected = tuple((d, c) for d in fold.training_donors for c in ("Stim8hr", "Stim48hr"))
    if (type(populations) is not tuple or any(type(p) is not FittingPopulation for p in populations)
            or tuple((p.donor, p.condition) for p in populations) != expected
            or len({p.source_id for p in populations}) != len(expected)):
        raise ValueError("Exactly one ordered early/late population per fitting donor required")
    if (index.binding.synthetic_fixture != admission.specification.synthetic_fixture
            or index.rna_axis_sha256 != admission.fold.feature_axis_sha256
            or index.binding.rna_features != admission.specification.features
            or admission.specification.latent_dim != 48
            or index.access(fold, "representation_fit").identity() != admission.trajectory_identity["input_sha256"]):
        raise ValueError("Actual observation admission differs from fitting subset/axis/role")
    if index.binding.synthetic_fixture and limits.maximum_host_bytes > 8*1024**3:
        raise ValueError("Fabricated preparation retains the 8-GiB CPU limit")
    bounds = []
    for handle, role, condition in zip(handles, ROLES, ("Stim8hr", "Stim48hr")):
        wanted = index.access(fold, role)
        rows = tuple(p.source_id for p in populations if p.condition == condition)
        if (handle.index.identity != index.identity or handle.root != handles[0].root
                or handle.access.identity() != wanted.identity()
                or set(wanted.source_ids) != set(rows)
                or any(index.source_roles.get(p.source_id) != (p.donor, p.condition)
                       for p in populations if p.condition == condition)):
            raise ValueError("Exact complete fitting-role handles required; no query or foreign source")
        b = _reader_bounds(handle)
        if limits.batch_rows > b["max_batch_rows"]:
            raise ValueError("Preparation batch exceeds the actual reader limit")
        bounds.append(b)
    catalog = tuple(CatalogGuide(str(r.guide_id), str(r.target_id), bool(r.is_control))
                    for r in index.catalog().itertuples(index=False))
    g, j = len(catalog), index.binding.rna_features
    # Conservative before payload access: every captured row could have RNA.
    array_bytes = sum(sum(s.rows for s in h.access.shards)*48*4
                      + len(h.access.source_ids)*(g*j*8+(3*g+1)*8) for h in handles)
    pool_bytes = sum(s.rows for s in handles[0].access.shards)*(48*4+8) + len(fold.training_donors)*j*8
    reader_bytes = sum(b["max_uncompressed_bytes"]+b["max_cached_bytes"]+3*b["max_output_bytes"] for b in bounds)
    # Charge both original arrays and readback mappings/pools during publication.
    # Compiler cap already charges its model and current output; double-charging
    # that output here avoids treating independent per-population caps as a sum.
    working = max(array_bytes+pool_bytes+limits.compiler.maximum_host_bytes+reader_bytes,
                  2*(array_bytes+pool_bytes)+reader_bytes+2*admission.artifact.size_bytes
                  +8*limits.maximum_metadata_bytes+2*IO_BLOCK)
    if array_bytes+pool_bytes > limits.provider.maximum_host_bytes or working > limits.maximum_host_bytes:
        raise ValueError("Aggregate whole-fold preparation estimate exceeds host/provider cap")
    estimate = dict(all_captured_rows_positive_upper_bound=True, array_payload_bytes=array_bytes,
        pooled_control_bytes=pool_bytes, native_reader_allowance_bytes=reader_bytes,
        conservative_host_estimate_bytes=working, maximum_host_bytes=limits.maximum_host_bytes,
        actual_process_RSS_or_GPU_memory_measured=False, reader_bounds=bounds)
    return index, fold, catalog, estimate


def _artifact_references(admission):
    return (admission.artifact, admission.receipt, admission.completion,
            admission.worker_contract, admission.protocol)


class PreparedFittingCache:
    """Same-process, one-publication owner; not a resumable training checkpoint."""
    def __init__(self, *, cache, observation, handles, snapshots, admission, receipt, limits, _factory=None):
        if _factory is not _PREPARATION_FACTORY:
            raise TypeError("Use prepare_fitting_cache; anonymous caches are not accepted preparations")
        self.cache, self.observation = cache, observation
        self._handles, self._snapshots, self._admission = handles, snapshots, admission
        self._receipt = _json_bytes(receipt)
        self._limits, self._pid, self._publication_attempted = limits, os.getpid(), False

    @property
    def receipt(self):
        return _parse(self._receipt)

    def _check(self):
        if self._pid != os.getpid():
            raise ValueError("Fitting preparation cannot cross a process boundary")
        self.cache.assert_unchanged()
        self.cache.verify_decoder(self.observation, self.cache.manifest.observation_binding)
        receipt = self.receipt
        if _sources() != receipt["implementation_sources"]:
            raise ValueError("Preparation source changed during ownership")
        if [_reader_bounds(h) for h in self._handles] != receipt["estimate"]["reader_bounds"]:
            raise ValueError("Native reader bounds changed during preparation")
        return [h.verify_publication_identity(s) for h, s in zip(self._handles, self._snapshots)]

    def publish(self, destination, *, output_cap_bytes):
        """New root/cache + preparation evidence; completion after all exit checks.

        A failed attempt leaves its exact provisional outputs. There is no retry
        on this owner, directory adoption, deletion or optimizer operation.
        """
        if self._publication_attempted:
            raise ValueError("One preparation publication attempt only; no automatic retry")
        root = _path(destination)
        if os.path.lexists(root):
            raise FileExistsError("Preparation destination exists; no overwrite or resume")
        meta = self._limits.maximum_metadata_bytes
        if type(output_cap_bytes) is not int or output_cap_bytes <= 3*meta:
            raise ValueError("Output allowance must include bounded preparation metadata")
        payload = sum(getattr(p, field).nbytes for p in self.cache.populations.values() for field in ARRAY_FIELDS)
        # Cache metadata/header bound, plus enclosing receipt and completion.
        if payload+3*meta+2*MAXIMUM_METADATA_BYTES+len(self.cache.populations)*len(ARRAY_FIELDS)*256 > output_cap_bytes:
            raise ValueError("Whole cache plus publication metadata exceeds output cap")
        self._check()
        self._publication_attempted = True
        parent_fd = os.open(_path(root.parent), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        root_fd = None
        try:
            if _stamp(os.fstat(parent_fd)) != _stamp(root.parent.lstat()):
                raise ValueError("Preparation parent changed while opening")
            os.mkdir(root.name, 0o700, dir_fd=parent_fd)
            root_fd = os.open(root.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
            os.fsync(parent_fd)
            producer = self.receipt["implementation_sources"]["frontier/fitting_preparation.py"]
            with ExitStack() as admitted:
                # Keep original selected-artifact descriptors/path identities
                # live through cache publication and enclosing completion.
                for ref in _artifact_references(self._admission):
                    admitted.enter_context(_verified_file(ref, maximum=observation_loader.MAXIMUM_ARTIFACT_BYTES))
                published = publish_fold_cache(self.cache, root/"cache",
                    expected_manifest_sha256=self.cache.manifest.identity,
                    producer_sha256=producer, output_cap_bytes=output_cap_bytes-3*meta)
                with load_fold_cache(root/"cache", expected_manifest_sha256=self.cache.manifest.identity,
                        expected_complete_sha256=published["complete_sha256"], output_cap_bytes=output_cap_bytes-3*meta,
                        limits=self._limits.provider) as mapped:
                    mapped.verify_decoder(self.observation, self.cache.manifest.observation_binding)
                    output_stamps = {f.name: f.stamp for f in mapped._leases}
                    cache_stamp = mapped._root_stamp
                verification = dict(mapped.verification)
                if verification.get("exit_full_hash_verified") is not True:
                    raise ValueError("Full cache exit hash verification required")
                final_input = self._check()
                record = dict(self.receipt, status="cache_published_and_readback_verified",
                    cache_publication=published, cache_verification=verification,
                    final_input_exit=final_input, publication_claimed=True,
                    loss_normalization_performed=False, main_fitting_performed=False)
                raw = _json_bytes(record)
                if len(raw) > meta:
                    raise ValueError("Preparation receipt exceeds metadata allowance")
                digest = hashlib.sha256(raw).hexdigest()
                stamp = _write_once(root_fd, "PREPARATION.json", (raw,), expected_sha256=digest, expected_bytes=len(raw))
                complete = dict(schema=COMPLETE_SCHEMA, status="complete", fold=asdict(self.cache.manifest.fold),
                    cache_manifest_sha256=self.cache.manifest.identity, cache_complete_sha256=published["complete_sha256"],
                    cache_total_bytes=published["total_bytes"],
                    receipt=dict(relative_path="PREPARATION.json", sha256=digest, size_bytes=len(raw)),
                    output_cap_bytes=output_cap_bytes, total_bytes=0, scientific_launch_authority=False)
                while True:
                    final = _json_bytes(complete)
                    total = published["total_bytes"]+len(raw)+len(final)
                    if total == complete["total_bytes"]: break
                    complete["total_bytes"] = total
                if len(final) > meta or total > output_cap_bytes:
                    raise ValueError("Complete preparation output exceeds declared cap")
                self._check()
                def verify_paths():
                    if (_stamp(os.fstat(root_fd)) != _stamp(_path(root).lstat())
                            or _stamp(_path(root/"cache").lstat()) != cache_stamp
                            or set(os.listdir(root/"cache")) != set(output_stamps)
                            or any(_stamp((root/"cache"/name).lstat()) != s for name, s in output_stamps.items())
                            or _stamp(os.stat("PREPARATION.json", dir_fd=root_fd, follow_symlinks=False)) != stamp):
                        raise ValueError("Published preparation path identity changed")
                verify_paths()
                if set(os.listdir(root_fd)) != {"cache", "PREPARATION.json"}:
                    raise ValueError("Unexpected preparation output before completion")
                complete_sha = hashlib.sha256(final).hexdigest()
                _write_once(root_fd, ".COMPLETE.pending", (final,), expected_sha256=complete_sha, expected_bytes=len(final))
                os.link(".COMPLETE.pending", "COMPLETE.json", src_dir_fd=root_fd, dst_dir_fd=root_fd, follow_symlinks=False)
                os.unlink(".COMPLETE.pending", dir_fd=root_fd); os.fsync(root_fd)
                verify_paths()
                self._check()
                if set(os.listdir(root_fd)) != {"cache", "PREPARATION.json", "COMPLETE.json"}:
                    raise ValueError("Unexpected preparation output at completion")
                return dict(root=str(root), complete_sha256=complete_sha, total_bytes=total,
                    cache_publication=published, input_exit_verified=True, cache_exit_hash_verified=True,
                    normalizers_published=False, scientific_launch_authority=False)
        except BaseException as error:
            if root_fd is not None:
                # Preserve failure even if a post-publication lease check fails.
                # Consumers require an exact three-member accepted root, so a
                # leftover inner/outer COMPLETE cannot override this failure.
                failure = _json_bytes(dict(schema="credo.frontier.fitting-preparation-failure.v1",
                    status="failed", exception_type=type(error).__name__, automatic_retry=False))
                try:
                    _write_once(root_fd, "FAILED.json", (failure,),
                        expected_sha256=hashlib.sha256(failure).hexdigest(), expected_bytes=len(failure))
                except BaseException:
                    pass  # Preserve original failure; missing parent acceptance is still not success.
            raise
        finally:
            if root_fd is not None: os.close(root_fd)
            os.close(parent_fd)


def prepare_fitting_cache(handles, *, observation_admission, populations, limits, device="cpu"):
    """Compile every complete fitting source exactly once, with no optimization.

    Call only under a numerical preparation admission. Replicate labels are
    explicit claims supplied by that admission, not inferred biological units.
    """
    index, fold, catalog, estimate = _admit(handles, observation_admission, populations, limits)
    destination = torch.device(device)
    if (destination.type not in ("cpu", "cuda") or (destination.type == "cuda" and destination.index is None)
            or (index.binding.synthetic_fixture and destination.type != "cpu")):
        raise ValueError("Explicit CPU/indexed CUDA destination; fabricated work is CPU only")
    source_files = _sources()
    model, binding = load_observation(observation_admission)
    model.to(destination)
    if (limits.compiler.encoding_rows > model.maximum_rows
            or observation_state_digest(model) != binding.state_sha256):
        raise ValueError("Actual selected observation or encoder batch differs")
    snapshots = tuple(handle.capture_publication_identity() for handle in handles)
    if len(_json_bytes(dict(input_leases=list(snapshots)))) > limits.maximum_metadata_bytes:
        raise ValueError("Native input lease evidence exceeds preparation metadata allowance")
    arrays, compilations = [], []
    for population in populations:
        role_index = 0 if population.condition == "Stim8hr" else 1
        handle = handles[role_index]
        support = handle.support_counts(population.source_id)
        counts = handle.captured_counts(population.source_id)
        if not np.array_equal(support.captured_cells.to_numpy(), counts.n_cells.to_numpy()):
            raise ValueError("Captured accounting and complete cell metadata disagree")
        spec = SourceCompileSpec(fold, catalog, index.binding.rna_features, index.rna_axis_sha256,
            binding, population.donor, population.condition, population.source_id, population.replicate_id,
            ROLES[role_index], handle.access.identity(), native_source_input_identity(index, population.source_id),
            tuple(NativeShard(s.shard, s.rows) for s in handle.access.shards if s.source_id == population.source_id),
            tuple(map(int, support.captured_cells)), tuple(map(int, support.positive_RNA_cells)),
            index.binding.synthetic_fixture)
        result = compile_population(spec, ordered_native_chunks(handle, spec, batch_rows=limits.batch_rows),
                                    model, limits=limits.compiler)
        arrays.append(result.population); compilations.append(result.receipt)
    manifest = FoldCacheManifest(fold, catalog, index.rna_axis_sha256, index.binding.rna_features,
        binding, handles[0].access.identity(), handles[1].access.identity(),
        tuple(p.reference for p in arrays), synthetic_fixture=index.binding.synthetic_fixture)
    cache = FoldFittingCache(manifest, arrays, expected_manifest_sha256=manifest.identity, limits=limits.provider)
    cache.verify_decoder(model, binding)
    exits = [h.verify_publication_identity(s) for h, s in zip(handles, snapshots)]
    if source_files != _sources():
        raise ValueError("Preparation source changed during compilation")
    receipt = dict(schema=SCHEMA, status="in_memory_complete_not_published", fold=asdict(fold),
        populations=[asdict(p) for p in populations], native_metadata_identity=index.identity,
        observation_admission_files=[asdict(r) for r in _artifact_references(observation_admission)],
        observation_binding=asdict(binding), cache_manifest_sha256=manifest.identity,
        input_leases=list(snapshots), input_exit=exits, compilations=compilations,
        estimate=estimate, limits=asdict(limits), actual_cache_retained_bytes=cache.host_bytes,
        implementation_sources=source_files, selected_observation_refitted=False,
        observation_technical_holdout_filter_applied=False, held_out_donor_payloads_opened=False,
        biological_replication_inferred=False, captured_zeros_preserved=True,
        synthetic_fixture=index.binding.synthetic_fixture, optimizer_calls=0,
        scientific_updates=0, publication_claimed=False, scientific_launch_authority=False)
    if len(_json_bytes(receipt)) > limits.maximum_metadata_bytes:
        raise ValueError("Preparation evidence exceeds the metadata allowance")
    return PreparedFittingCache(cache=cache, observation=model, handles=handles, snapshots=snapshots,
        admission=observation_admission, receipt=receipt, limits=limits, _factory=_PREPARATION_FACTORY)


@contextmanager
def open_prepared_cache(root, *, expected_complete_sha256, output_cap_bytes, limits):
    """Require enclosing preparation acceptance and hold all output leases.

    This verifies saved preparation evidence and arrays, not the original raw
    data or the correctness of a claimed biological culture assignment.
    Normalizer admission and main-fit worker qualification remain separate.
    """
    root = _path(root)
    if type(output_cap_bytes) is not int or output_cap_bytes < 1 or not isinstance(limits, ProviderLimits):
        raise ValueError("Explicit preparation output and provider limits required")
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    leases = []
    try:
        stamp = _stamp(os.fstat(root_fd))
        if stamp != _stamp(root.lstat()) or set(os.listdir(root_fd)) != {"cache", "PREPARATION.json", "COMPLETE.json"}:
            raise ValueError("Exact complete preparation inventory required")
        complete_file = _open_verified(root_fd, "COMPLETE.json", expected_complete_sha256,
                                      maximum=min(output_cap_bytes, MAXIMUM_METADATA_BYTES))
        leases.append(complete_file)
        complete = _parse(complete_file.stream.read())
        if (set(complete) != {"schema", "status", "fold", "cache_manifest_sha256", "cache_complete_sha256",
                "cache_total_bytes", "receipt", "output_cap_bytes", "total_bytes", "scientific_launch_authority"}
                or complete["schema"] != COMPLETE_SCHEMA or complete["status"] != "complete"
                or complete["scientific_launch_authority"] is not False
                or any(type(complete[k]) is not int or complete[k] < 1
                       for k in ("total_bytes", "cache_total_bytes", "output_cap_bytes"))
                or complete["total_bytes"] > min(output_cap_bytes, complete["output_cap_bytes"])):
            raise ValueError("Preparation completion schema/status/cap differs")
        ref = complete["receipt"]
        if set(ref) != {"relative_path", "sha256", "size_bytes"} or ref["relative_path"] != "PREPARATION.json":
            raise ValueError("Exact preparation receipt required")
        receipt_file = _open_verified(root_fd, "PREPARATION.json", ref["sha256"], size=ref["size_bytes"],
                                     maximum=min(output_cap_bytes, MAXIMUM_METADATA_BYTES))
        leases.append(receipt_file)
        record = _parse(receipt_file.stream.read())
        publication = record.get("cache_publication", {})
        if (record.get("schema") != SCHEMA or record.get("status") != "cache_published_and_readback_verified"
                or record.get("fold") != complete["fold"] or record.get("cache_manifest_sha256") != complete["cache_manifest_sha256"]
                or publication.get("complete_sha256") != complete["cache_complete_sha256"]
                or publication.get("root") != str(root/"cache") or publication.get("total_bytes") != complete["cache_total_bytes"]
                or record.get("cache_verification", {}).get("exit_full_hash_verified") is not True
                or record.get("publication_claimed") is not True or record.get("optimizer_calls") != 0
                or record.get("scientific_updates") != 0 or record.get("held_out_donor_payloads_opened") is not False
                or record.get("selected_observation_refitted") is not False):
            raise ValueError("Preparation evidence differs from accepted cache publication")
        entries, exits = record.get("input_leases"), record.get("final_input_exit")
        if (not isinstance(entries, list) or not isinstance(exits, list) or len(entries) != 2 or len(exits) != 2
                or any(x.get("status") != "unchanged" or x.get("input_lease_sha256") != _hash(s)
                       or x.get("access_sha256") != s.get("access_sha256") for s,x in zip(entries, exits))):
            raise ValueError("Both exact fitting input lease exits required")
        if complete["cache_total_bytes"]+complete_file.size+receipt_file.size != complete["total_bytes"]:
            raise ValueError("Preparation total output bytes differ")
        try:
            with load_fold_cache(root/"cache", expected_manifest_sha256=complete["cache_manifest_sha256"],
                    expected_complete_sha256=complete["cache_complete_sha256"], output_cap_bytes=output_cap_bytes,
                    limits=limits) as mapped:
                manifest = mapped.manifest
                if (asdict(manifest.fold) != dict(complete["fold"], training_donors=tuple(complete["fold"]["training_donors"]))
                        or [manifest.source_access_sha256, manifest.supervision_access_sha256] != [s["access_sha256"] for s in entries]
                        or asdict(manifest.observation_binding) != record["observation_binding"]):
                    raise ValueError("Prepared cache fold/role/representation differs")
                yield mapped
        finally:
            for lease in leases:
                if (_digest_stream(lease.stream) != lease.sha256
                        or _stamp(os.fstat(lease.stream.fileno())) != lease.stamp
                        or _stamp(os.stat(lease.name, dir_fd=root_fd, follow_symlinks=False)) != lease.stamp):
                    raise ValueError("Preparation metadata changed during consumption")
            if (_stamp(os.fstat(root_fd)) != stamp or _stamp(_path(root).lstat()) != stamp
                    or set(os.listdir(root_fd)) != {"cache", "PREPARATION.json", "COMPLETE.json"}):
                raise ValueError("Preparation root changed during consumption")
    finally:
        for lease in leases: lease.stream.close()
        os.close(root_fd)
