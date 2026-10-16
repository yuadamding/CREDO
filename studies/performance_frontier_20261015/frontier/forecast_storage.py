"""Bounded, completion-last single-seed forecast publication and mmap reading.

Only the maximum *raw* particle states are stored. Each effective prefix keeps
its own corrected normalized weights and mass: these are not obtained by
truncating maximum-prefix weights. Exact raw-prefix nesting and CPU spread
reconstruction are checked on every emission and again on readback. Final
ensembles are NOT prefix-nested and are intentionally excluded from this format.

The writer has no model, outcome, optimizer, checkpoint loader or authority to
run science. Its final record requires the source-only core's successful return
receipt. Identity strings supplied by a caller do not prove artifact custody.
Numeric arrays use explicit canonical NPY headers, never pickle. Writer arrays
use bounded positional writes, not a writable whole-bank mapping. One bounded
law/batch is live in working memory; read-only mmap/page-cache RSS still needs
worker qualification. This is not a claim that any real inventory fits 64 GiB.

Existing roots are never reused. Failed/incomplete publications are left in
place, never completed by another writer and never automatically retried.
"""

from dataclasses import asdict, dataclass
from functools import wraps
import hashlib
import json
import math
import os
import stat
from types import MappingProxyType

import numpy as np
import torch

from .cache_storage import (_digest_stream, _header, _json_bytes, _open_verified,
    _parse, _path, _stamp, _write_once, MAXIMUM_METADATA_BYTES, IO_BLOCK)
from .configuration import STAGE_UPDATES, get_configuration
from .fitting_provider import _array_hash, _hash, _sha
from .forecast import (ConditionalForecast, ForecastModelBinding, ForecastSource,
                       ensemble_prefix_plan)
from .measure_ops import calibrate_abundance, calibrate_dispersion, catalog_log_frequencies, source_log_frequencies
from .sampling import PooledSourceIdentity, SamplingContext
from .selector_contract import Recipe


SCHEMA = "credo.frontier.forecast-storage.v1"
COMPLETE_SCHEMA = "credo.frontier.forecast-storage-complete.v1"
POOL = "pooled_control"
ARRAY_NAMES = ("raw_states", "log_weights", "log_mass", "catalogs")
CATALOG_FIELDS = ("log_mass_increments", "log_source_frequencies", "uncalibrated_log_frequencies", "log_frequencies")
_SPEC_FACTORY = object()


def _runtime():
    return dict(torch_version=str(torch.__version__), numpy_version=np.__version__,
                torch_threads=torch.get_num_threads(), calibration_device="cpu",
                state_dtype="float32", reduction_dtype="float64")


@dataclass(frozen=True, init=False)
class ForecastStorageSpec:
    """Canonical immutable metadata exported from an admitted source-only view."""
    _encoded: bytes

    def __init__(self, encoded, *, _factory=None):
        if _factory is not _SPEC_FACTORY:
            raise TypeError("Storage specifications must be constructed from the validated source factory")
        object.__setattr__(self, "_encoded", encoded)
        self.__post_init__()

    @classmethod
    def from_source(cls, source, binding, context, recipe, *, prefixes, ensemble_component_totals=()):
        if (not isinstance(source, ForecastSource) or not isinstance(binding, ForecastModelBinding)
                or not isinstance(context, SamplingContext) or not isinstance(recipe, Recipe)):
            raise TypeError("Exact typed forecast source/model/context/recipe required")
        if (source.fold != binding.fold or source.observation_binding != binding.observation_binding
                or context.training_donors != source.fold.training_donors or context.purpose != "forecast"
                or context.configuration_id != recipe.configuration_id or recipe.configuration_id != binding.configuration_id
                or context.model_seed != binding.model_seed or context.accepted_update != binding.accepted_updates):
            raise ValueError("Source/model/sampler/recipe identity mismatch")
        prefixes, totals = tuple(prefixes), tuple(ensemble_component_totals)
        plan = ensemble_prefix_plan(binding.model_seed, totals, prefixes, synthetic_fixture=source.synthetic_fixture)
        if plan and (len(source.fold.training_donors) != 3 or binding.accepted_updates != 9216):
            raise ValueError("Ensemble component prefixes require the outer final recipe")
        effective = tuple(sorted(set(prefixes) | set(plan.values())))
        if min(effective) < get_configuration(recipe.configuration_id).mixture_components:
            raise ValueError("Every prefix must contain all mixture components")
        if not source.synthetic_fixture and binding.accepted_updates not in STAGE_UPDATES:
            raise ValueError("Scientific publication requires a registered training milestone")
        source.assert_unchanged()
        supported = [i for i, n in enumerate(source.positive_counts) if n]
        missing = [i for i, n in enumerate(source.positive_counts) if not n]
        controls = [i for i, g in enumerate(source.catalog) if g.is_control]
        pool = None
        if sum(int(source.positive_counts[i]) for i in controls):
            catalog_sha = _hash([asdict(source.catalog[i]) for i in controls])
            ref = source.reference
            definition = _hash(dict(policy="all_NTC_positive_RNA_source_cells_equal", source=ref.latent_sha256,
                                    raw_summary=ref.raw_summary_sha256, catalog=catalog_sha))
            pool = asdict(PooledSourceIdentity(ref.donor, "Stim8hr", ref.source_id, ref.replicate_id,
                       f"{ref.donor}::pooled-NTC-RNA", catalog_sha, definition))
        document = dict(schema="credo.frontier.forecast-storage-spec.v1", source_identity=source.identity,
            model_binding=asdict(binding), context=asdict(context), recipe=asdict(recipe),
            source_reference=asdict(source.reference), catalog=[asdict(g) for g in source.catalog],
            catalog_sha256=source.catalog_sha256, feature_axis_sha256=source.feature_axis_sha256,
            captured_counts=source.captured_counts.tolist(), positive_counts=source.positive_counts.tolist(),
            supported_guide_indices=supported, unavailable_source_guide_indices=missing,
            pooled_control_identity=pool, pooled_control_unavailable_reason=None if pool else "missing_positive_RNA_source_NTC_pool",
            single_seed_prefixes=list(prefixes), ensemble_component_totals=list(totals),
            ensemble_component_prefixes=plan, prefixes=list(effective), maximum_generated_particles=max(prefixes),
            synthetic_fixture=source.synthetic_fixture, reconstruction_runtime=_runtime())
        return cls(_json_bytes(document), _factory=_SPEC_FACTORY)

    def __post_init__(self):
        if type(self._encoded) is not bytes or _json_bytes(_parse(self._encoded)) != self._encoded:
            raise ValueError("Canonical immutable storage specification required")

    @property
    def document(self):
        return _parse(self._encoded)

    @property
    def identity(self):
        return hashlib.sha256(self._encoded).hexdigest()

    @property
    def effective_prefixes(self):
        return tuple(self.document["prefixes"])

    @property
    def maximum_particles(self):
        return self.document["maximum_generated_particles"]


@dataclass(frozen=True)
class ForecastStorageLimits:
    output_cap_bytes: int
    working_cap_bytes: int
    maximum_batch_laws: int

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in asdict(self).values()) or self.maximum_batch_laws > 1024:
            raise ValueError("Explicit positive storage/working/batch limits required")


def _layout(spec):
    doc = spec.document
    laws = len(doc["supported_guide_indices"]) + int(doc["pooled_control_identity"] is not None)
    k, g, p = len(spec.effective_prefixes), len(doc["catalog"]), spec.maximum_particles
    return dict(raw_states=(np.dtype("<f4"), (laws, p, 48)),
                log_weights=(np.dtype("<f8"), (laws, sum(spec.effective_prefixes))),
                log_mass=(np.dtype("<f8"), (k, laws)), catalogs=(np.dtype("<f8"), (k, g, 4)))


def _metadata_cap(spec):
    # Full scientific sampler receipts plus all prefix evidence can exceed the
    # cache publisher's 32-MiB cap. This separate bound is prospectively charged.
    return (8 if spec.document["synthetic_fixture"] else 128)*1024**2


def _encode_manifest(value, cap):
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()
    if len(raw) > cap: raise ValueError("Forecast metadata exceeds its declared publication bound")
    return raw


def storage_estimate(spec, limits):
    if not isinstance(spec, ForecastStorageSpec) or not isinstance(limits, ForecastStorageLimits):
        raise TypeError("Typed publication spec and byte limits required")
    layout = _layout(spec)
    array_bytes = sum(len(_header(dtype, shape)) + math.prod(shape)*dtype.itemsize for dtype, shape in layout.values())
    # Full metadata bounds, hash block, all catalog vectors, bounded emission/
    # reconstruction copies, and per-law/prefix hashes+bookkeeping. No mapped
    # bank is materialized. OS page cache is separately qualified, not hidden.
    doc = spec.document
    bookkeeping = max(1, layout["raw_states"][1][0]) * len(spec.effective_prefixes) * 1024
    metadata_cap = _metadata_cap(spec)
    working = (8*metadata_cap + IO_BLOCK + bookkeeping +
               len(doc["catalog"])*len(spec.effective_prefixes)*8*16 +
               limits.maximum_batch_laws*spec.maximum_particles*(48*4*8 + 8*8))
    result = dict(array_file_bytes=array_bytes, output_upper_bound_bytes=array_bytes+2*metadata_cap,
                  maximum_working_bytes=working, maximum_batch_laws=limits.maximum_batch_laws,
                  maximum_metadata_file_bytes=metadata_cap,
                  mapped_file_page_cache_is_not_a_measured_RSS_bound=True)
    if result["output_upper_bound_bytes"] > limits.output_cap_bytes or working > limits.working_cap_bytes:
        raise ValueError("Publication preflight exceeds explicit output/working cap")
    if doc["synthetic_fixture"] and (limits.maximum_batch_laws > 64 or limits.working_cap_bytes > 8*1024**3):
        raise ValueError("Fabricated publication is bounded to 64 laws and 8 GiB working memory")
    return result


def _receipt_check(receipt, spec):
    if not isinstance(receipt, dict) or receipt.get("schema") not in (
            "credo.frontier.in-memory-forecast.v1", "credo.frontier.streaming-forecast.v1"):
        raise ValueError("Successful whole-core forecast receipt required")
    doc = spec.document
    for key in ("source_identity", "model_binding", "context", "recipe", "prefixes", "single_seed_prefixes",
                "ensemble_component_totals", "ensemble_component_prefixes", "maximum_generated_particles",
                "synthetic_fixture", "supported_guide_indices", "unavailable_source_guide_indices", "pooled_control_identity"):
        if key not in receipt or _json_bytes({"v": receipt[key]}) != _json_bytes({"v": doc[key]}):
            raise ValueError("Whole-core receipt differs from publication identity: " + key)
    for key, value in dict(complement_mass_multiplier=1, pooled_control_is_catalog_entry=False,
            source_only_support=True, endpoint_inputs_used=False, optimizer_calls=0,
            input_artifact_custody_verified_here=False, checkpoint_loading_verified_here=False,
            publication_claimed=False, scientific_launch_authority=False).items():
        if key not in receipt or type(receipt[key]) is not type(value) or receipt[key] != value:
            raise ValueError("Wrong core completion/scope assertion: " + key)
    if set(receipt.get("control_ids", [])) != {g["target_id"] for g in doc["catalog"] if g["is_control"]}:
        raise ValueError("Whole-core control inventory differs")
    if receipt.get("query_donor") != doc["source_reference"]["donor"]:
        raise ValueError("Whole-core query donor differs")
    if not isinstance(receipt.get("sampler_receipts"), list):
        raise ValueError("Core sampler inventory is missing")
    if receipt.get("complete_prediction_checks_passed") is not True:
        raise ValueError("Whole-core final model/source/RNG checks have not passed")
    if receipt.get("law_storage") == "streamed_to_explicit_sink":
        if receipt.get("sink_spec_sha256") != spec.identity:
            raise ValueError("Core streaming sink identity differs")
    elif receipt.get("law_storage") != "in_memory" or receipt.get("sink_spec_sha256") is not None:
        raise ValueError("Unknown core law publication route")


def _catalog_check(spec, prefix, values, emitted_masses):
    doc = spec.document
    g = len(doc["catalog"])
    for value in values:
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.requires_grad
                or value.dtype != torch.float64 or value.shape != (g,) or not bool(torch.isfinite(value).all())):
            raise ValueError("Finite complete CPU FP64 catalog vectors required")
    masses, source, raw, calibrated = values
    expected_masses = torch.zeros(g, dtype=torch.float64, device="cpu")
    indices = torch.tensor(doc["supported_guide_indices"], dtype=torch.int64, device="cpu")
    expected_masses[indices] = emitted_masses
    expected_source = source_log_frequencies(torch.tensor(doc["captured_counts"], dtype=torch.int64, device="cpu"))
    expected_raw = catalog_log_frequencies(expected_source, indices, expected_masses[indices])
    expected_calibrated = calibrate_abundance(expected_source, expected_raw, doc["recipe"]["alpha"])
    if any(not torch.equal(a, b) for a, b in zip(values, (expected_masses, expected_source, expected_raw, expected_calibrated))):
        raise ValueError("Complete catalog does not reconcile with source counts/emitted masses/fixed calibration")
    if bool((calibrated.exp() <= 0).any()) or bool((raw.exp() <= 0).any()):
        raise ValueError("Catalog probabilities underflow; no undeclared floor")


def _fail_closed(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        self._live()
        try:
            return method(self, *args, **kwargs)
        except BaseException:
            self._failed = True
            raise
    return wrapped


class ForecastWriter:
    """Exclusive streaming writer. Call finish(core_receipt) only after core exit checks."""
    def __init__(self, destination, *, spec, producer_sha256, limits):
        _sha(producer_sha256)
        self.estimate = MappingProxyType(storage_estimate(spec, limits))
        self.spec, self.limits, self.producer_sha256 = spec, limits, producer_sha256
        self._doc = spec.document
        self._runtime_binding = self._doc["reconstruction_runtime"]
        self._spread = self._doc["recipe"]["spread_scale"]
        self._prefixes, self._pmax = spec.effective_prefixes, spec.maximum_particles
        self._guide_count = len(self._doc["catalog"])
        self._supported_count = len(self._doc["supported_guide_indices"])
        self.root = _path(destination)
        if os.path.lexists(self.root):
            raise FileExistsError("Forecast destination exists; no overwrite or retry")
        if self._runtime_binding != _runtime():
            raise ValueError("Publication runtime differs from exact reconstruction binding")
        self._pid, self._entered, self._failed, self._finished = os.getpid(), False, False, False
        self._closed, self._root_fd, self._streams = False, None, {}
        self._original = (spec, limits, self.estimate, producer_sha256, self.root)
        self._seen, self._raw_hashes, self._weight_hashes, self._catalog_seen = set(), {}, {}, set()
        self._catalog_hashes, self._file_ids = {}, {}
        self._layout = _layout(spec)
        self._keys = tuple(self._doc["supported_guide_indices"]) + ((POOL,) if self._doc["pooled_control_identity"] else ())
        self._rows = {key: i for i, key in enumerate(self._keys)}
        self._prefix_rows = {p: i for i, p in enumerate(spec.effective_prefixes)}
        self._masses = np.zeros((len(spec.effective_prefixes), len(self._keys)), dtype=np.float64)
        cursor = 0
        self._weight_slices = {}
        for p in spec.effective_prefixes:
            self._weight_slices[p] = slice(cursor, cursor+p)
            cursor += p

    def __enter__(self):
        if self._entered or self._closed:
            raise RuntimeError("Writer is single-use")
        parent_fd = os.open(_path(self.root.parent), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            if _stamp(os.fstat(parent_fd)) != _stamp(self.root.parent.lstat()):
                raise ValueError("Publication parent changed")
            os.mkdir(self.root.name, 0o700, dir_fd=parent_fd)
            self._root_fd = os.open(self.root.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
            self._root_inode = (os.fstat(self._root_fd).st_dev, os.fstat(self._root_fd).st_ino)
            for name, (dtype, shape) in self._layout.items():
                fd = os.open(name+".npy", os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=self._root_fd)
                stream = os.fdopen(fd, "w+b")
                self._streams[name] = stream
                header = _header(dtype, shape)
                stream.write(header); stream.flush()
                stream.truncate(len(header)+math.prod(shape)*dtype.itemsize)
                info = os.fstat(stream.fileno())
                self._file_ids[name] = (info.st_dev, info.st_ino, info.st_size, info.st_mode, info.st_nlink)
            os.fsync(parent_fd)
            self._entered = True
            return self
        except BaseException:
            self._failed = True
            self.close()
            raise
        finally:
            os.close(parent_fd)

    def _live(self):
        if (not self._entered or self._closed or self._failed or self._finished or self._pid != os.getpid()):
            raise RuntimeError("A healthy active same-process exclusive writer is required")
        root = _path(self.root)
        info = root.lstat()
        if (info.st_dev, info.st_ino) != self._root_inode or not stat.S_ISDIR(info.st_mode):
            self._failed = True
            raise ValueError("Publication root changed")
        if self._runtime_binding != _runtime():
            self._failed = True
            raise ValueError("Exact reconstruction runtime changed")
        if (self.spec is not self._original[0] or self.limits is not self._original[1]
                or self.estimate is not self._original[2] or self.producer_sha256 != self._original[3]
                or self.root != self._original[4]):
            self._failed = True
            raise ValueError("Writer source/limits/resource estimate changed")
        for name, stream in self._streams.items():
            for info in (os.fstat(stream.fileno()), os.stat(name+".npy", dir_fd=self._root_fd, follow_symlinks=False)):
                if (info.st_dev, info.st_ino, info.st_size, info.st_mode, info.st_nlink) != self._file_ids[name]:
                    self._failed = True
                    raise ValueError("Exclusive forecast array identity changed")

    def assert_active(self):
        self._live()

    def _write_region(self, name, element_offset, array):
        dtype, shape = self._layout[name]
        if array.dtype != dtype or not array.flags.c_contiguous:
            raise ValueError("Contiguous exact numeric write required")
        raw = memoryview(array).cast("B") if array.size else memoryview(b"")
        offset = len(_header(dtype, shape)) + element_offset*dtype.itemsize
        if element_offset < 0 or offset+len(raw) > len(_header(dtype, shape))+math.prod(shape)*dtype.itemsize:
            raise ValueError("Bounded write exceeds fixed packed array")
        for start in range(0, len(raw), IO_BLOCK):
            chunk = raw[start:start+IO_BLOCK]
            done = 0
            while done < len(chunk):
                n = os.pwrite(self._streams[name].fileno(), chunk[done:], offset+start+done)
                if n <= 0: raise OSError("Short forecast array write")
                done += n

    def _read_raw(self, row, prefix):
        dtype, shape = self._layout["raw_states"]
        size = prefix*48*dtype.itemsize
        offset = len(_header(dtype, shape))+row*self._pmax*48*dtype.itemsize
        raw = os.pread(self._streams["raw_states"].fileno(), size, offset)
        if len(raw) != size: raise ValueError("Short forecast raw-state readback")
        return np.frombuffer(raw, dtype=dtype).reshape(prefix, 48)

    def _read_region(self, name, offset, shape):
        dtype, full_shape = self._layout[name]
        size = math.prod(shape)*dtype.itemsize
        raw = os.pread(self._streams[name].fileno(), size, len(_header(dtype, full_shape))+offset*dtype.itemsize)
        if len(raw) != size: raise ValueError("Short numeric forecast readback")
        return np.frombuffer(raw, dtype=dtype).reshape(shape)

    @_fail_closed
    def emit(self, prefix, index, law):
        if (type(prefix) is not int or prefix not in self._prefix_rows
                or (type(index) is not int and index != POOL) or index not in self._rows
                or (prefix, index) in self._seen or not isinstance(law, ConditionalForecast)):
            raise ValueError("Each expected prefix/law must be emitted exactly once")
        raw, states, weights = law.uncalibrated_states, law.states, law.log_weights
        for tensor, dtype, shape in ((raw, torch.float32, (prefix, 48)), (states, torch.float32, (prefix, 48)),
                                      (weights, torch.float64, (prefix,))):
            if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tensor.dtype != dtype
                    or tensor.shape != shape or tensor.requires_grad or not tensor.is_contiguous()
                    or not bool(torch.isfinite(tensor).all())):
                raise ValueError("Exact finite contiguous CPU conditional-law arrays required")
        if type(law.log_mass_increment) is not float or not math.isfinite(law.log_mass_increment):
            raise ValueError("Finite prefix-specific log mass required")
        if abs(float(torch.logsumexp(weights, 0))) > 1e-10:
            raise ValueError("Prefix weights must be correctly normalized")
        reconstructed = calibrate_dispersion(raw, weights, self._spread)
        if not torch.equal(reconstructed, states):
            raise ValueError("Calibrated states are not exactly reconstructible from this prefix")
        row, k = self._rows[index], self._prefix_rows[prefix]
        digest = _array_hash(raw.numpy())
        if prefix == self._pmax:
            for p in self._prefixes:
                previous = self._raw_hashes.get((p, index))
                if previous is not None and previous != _array_hash(raw[:p].numpy()):
                    raise ValueError("Raw states are not exactly prefix-nested")
            self._write_region("raw_states", row*self._pmax*48, raw.numpy())
        elif (self._pmax, index) in self._seen:
            if digest != _array_hash(self._read_raw(row, prefix)):
                raise ValueError("Raw states are not exactly prefix-nested")
        self._write_region("log_weights", row*sum(self._prefixes)+self._weight_slices[prefix].start, weights.numpy())
        self._write_region("log_mass", k*len(self._keys)+row, np.asarray([law.log_mass_increment], dtype=np.float64))
        self._masses[k, row] = law.log_mass_increment
        self._raw_hashes[prefix, index] = digest
        self._weight_hashes[prefix, index] = _array_hash(weights.numpy())
        self._seen.add((prefix, index))

    @_fail_closed
    def emit_batch(self, prefix, laws):
        if not hasattr(laws, "items") or len(laws) > self.limits.maximum_batch_laws:
            raise ValueError("Explicit bounded mapping of law emissions required")
        for index, law in laws.items():
            self.emit(prefix, index, law)

    @_fail_closed
    def write_catalog(self, prefix, *, log_mass_increments, log_source_frequencies,
                      uncalibrated_log_frequencies, log_frequencies):
        if (type(prefix) is not int or prefix not in self._prefix_rows or prefix in self._catalog_seen
                or any((prefix, key) not in self._seen for key in self._keys)):
            raise ValueError("One complete catalog is required after all of its laws")
        values = (log_mass_increments, log_source_frequencies, uncalibrated_log_frequencies, log_frequencies)
        k = self._prefix_rows[prefix]
        guide_masses = torch.from_numpy(np.array(self._masses[k, :self._supported_count], copy=True))
        _catalog_check(self.spec, prefix, values, guide_masses)
        catalog = np.stack([value.numpy() for value in values], axis=-1)
        self._write_region("catalogs", k*self._guide_count*4, catalog)
        self._catalog_hashes[prefix] = _array_hash(catalog)
        self._catalog_seen.add(prefix)

    @_fail_closed
    def finish(self, forecast_receipt):
        if (len(self._seen) != len(self._keys)*len(self._prefixes)
                or self._catalog_seen != set(self._prefixes)):
            raise ValueError("Cannot complete a partial law/catalog inventory")
        _receipt_check(forecast_receipt, self.spec)
        raw_hash_rows = []
        for prefix in self._prefixes:
            for key in self._keys:
                digest = _array_hash(self._read_raw(self._rows[key], prefix))
                if digest != self._raw_hashes[prefix, key]:
                    raise ValueError("Stored raw bank changed or prefix nesting differs")
                row = self._rows[key]
                weights = self._read_region("log_weights", row*sum(self._prefixes)+self._weight_slices[prefix].start, (prefix,))
                if _array_hash(weights) != self._weight_hashes[prefix, key]:
                    raise ValueError("Stored prefix weights differ from emission")
                raw_hash_rows.append(dict(prefix=prefix, law=key, sha256=digest))
            catalog = self._read_region("catalogs", self._prefix_rows[prefix]*self._guide_count*4, (self._guide_count, 4))
            if _array_hash(catalog) != self._catalog_hashes[prefix]:
                raise ValueError("Stored catalog differs from emission")
        if not np.array_equal(self._read_region("log_mass", 0, self._masses.shape), self._masses):
            raise ValueError("Stored masses differ from emission")
        records, stamps = [], {}
        for name, (dtype, shape) in self._layout.items():
            stream = self._streams[name]
            stream.flush(); os.fsync(stream.fileno())
            header = _header(dtype, shape)
            stream.seek(0)
            if stream.read(len(header)) != header:
                raise ValueError("Stored numeric header differs")
            stamp = _stamp(os.fstat(stream.fileno()))
            path_stamp = _stamp(os.stat(name+".npy", dir_fd=self._root_fd, follow_symlinks=False))
            info = os.fstat(stream.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or path_stamp != stamp
                    or info.st_size != len(header)+math.prod(shape)*dtype.itemsize):
                raise ValueError("Array publication identity/size changed")
            digest = _digest_stream(stream)
            array_digest = hashlib.sha256(json.dumps([dtype.str, list(shape)]).encode())
            stream.seek(len(header))
            while chunk := stream.read(IO_BLOCK): array_digest.update(chunk)
            stream.seek(0)
            if _stamp(os.fstat(stream.fileno())) != stamp:
                raise ValueError("Array changed during hash readback")
            records.append(dict(name=name, relative_path=name+".npy", dtype=dtype.str, shape=list(shape),
                data_offset=len(header), file_bytes=info.st_size, file_sha256=digest,
                array_sha256=array_digest.hexdigest(), header_sha256=hashlib.sha256(header).hexdigest()))
            stamps[name+".npy"] = stamp
        metadata = _encode_manifest(dict(schema=SCHEMA, spec=self.spec.document, spec_sha256=self.spec.identity,
            producer_sha256=self.producer_sha256, forecast_receipt=forecast_receipt,
            forecast_receipt_sha256=_hash(forecast_receipt), arrays=records, raw_prefix_hashes=raw_hash_rows,
            estimate=dict(self.estimate), limits=asdict(self.limits),
            layout="maximum_raw_states_plus_prefix_specific_corrected_weights_and_complete_catalogs",
            calibrated_states_stored=False, final_ensemble_stored=False,
            checkpoint_loading_verified_here=False, scientific_launch_authority=False), self.estimate["maximum_metadata_file_bytes"])
        manifest_sha = hashlib.sha256(metadata).hexdigest()
        completion = dict(schema=COMPLETE_SCHEMA, status="complete", spec_sha256=self.spec.identity,
            manifest_sha256=manifest_sha, manifest_bytes=len(metadata), producer_sha256=self.producer_sha256,
            output_cap_bytes=self.limits.output_cap_bytes, total_bytes=0,
            effective_prefixes=list(self.spec.effective_prefixes), law_count=len(self._keys))
        while True:
            complete_bytes = _json_bytes(completion)
            total = sum(r["file_bytes"] for r in records)+len(metadata)+len(complete_bytes)
            if completion["total_bytes"] == total: break
            completion["total_bytes"] = total
        if total > self.limits.output_cap_bytes:
            raise ValueError("Complete forecast exceeds the explicit output cap")
        stamps["MANIFEST.json"] = _write_once(self._root_fd, "MANIFEST.json", (metadata,),
            expected_sha256=manifest_sha, expected_bytes=len(metadata))
        if set(os.listdir(self._root_fd)) != set(stamps):
            raise ValueError("Unexpected file in forecast publication")
        for name, stamp in stamps.items():
            if _stamp(os.stat(name, dir_fd=self._root_fd, follow_symlinks=False)) != stamp:
                raise ValueError("Published forecast file changed before completion")
        self._live()
        complete_sha = hashlib.sha256(complete_bytes).hexdigest()
        _write_once(self._root_fd, ".COMPLETE.pending", (complete_bytes,),
            expected_sha256=complete_sha, expected_bytes=len(complete_bytes))
        os.link(".COMPLETE.pending", "COMPLETE.json", src_dir_fd=self._root_fd, dst_dir_fd=self._root_fd, follow_symlinks=False)
        os.unlink(".COMPLETE.pending", dir_fd=self._root_fd)
        os.fsync(self._root_fd)
        self._live()
        if set(os.listdir(self._root_fd)) != set(stamps) | {"COMPLETE.json"}:
            raise ValueError("Forecast inventory changed during final completion publication")
        self._finished = True
        return dict(complete_sha256=complete_sha, manifest_sha256=manifest_sha, spec_sha256=self.spec.identity,
                    total_bytes=total, output_cap_bytes=self.limits.output_cap_bytes, root=str(self.root))

    def close(self):
        if self._closed: return
        self._masses = None
        for stream in self._streams.values(): stream.close()
        self._streams.clear()
        if self._root_fd is not None: os.close(self._root_fd)
        self._closed, self._entered = True, False

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None: self._failed = True
        unfinished = not self._finished and not self._failed
        self.close()
        if unfinished: raise RuntimeError("Forecast writer exited without completion; incomplete root retained")


class ForecastReader:
    """Read-only descriptor-bound access; law() copies just one prefix law.

    Use inside a context. No complete bank is copied. Each borrowed catalog
    view remains read-only and must not be used after the lease is closed.
    """
    def __init__(self, root, *, spec, expected_complete_sha256, expected_manifest_sha256,
                 maximum_output_bytes, maximum_working_bytes):
        if not isinstance(spec, ForecastStorageSpec): raise TypeError("Expected typed specification required")
        _sha(expected_complete_sha256); _sha(expected_manifest_sha256)
        if (type(maximum_output_bytes) is not int or maximum_output_bytes < 1
                or type(maximum_working_bytes) is not int or maximum_working_bytes < 1):
            raise ValueError("Explicit reader byte limits required")
        self.spec, self.root = spec, _path(root)
        self._original_spec, self._original_root = spec, self.root
        self._doc = spec.document
        self._runtime_binding = self._doc["reconstruction_runtime"]
        self._spread = self._doc["recipe"]["spread_scale"]
        self._missing = frozenset(self._doc["unavailable_source_guide_indices"])
        self._supported_count = len(self._doc["supported_guide_indices"])
        self._pid, self._entered, self._closed = os.getpid(), False, False
        self._leases, self._maps, self._root_fd = [], {}, None
        self._names = {name+".npy" for name in ARRAY_NAMES} | {"MANIFEST.json", "COMPLETE.json"}
        try:
            if self._runtime_binding != _runtime():
                raise ValueError("Reader runtime differs from exact calibrated-state reconstruction")
            self._root_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            self._root_stamp = _stamp(os.fstat(self._root_fd))
            if self._root_stamp != _stamp(self.root.lstat()) or set(os.listdir(self._root_fd)) != self._names:
                raise ValueError("Forecast publication root/inventory mismatch")
            complete_file = _open_verified(self._root_fd, "COMPLETE.json", expected_complete_sha256, maximum=MAXIMUM_METADATA_BYTES)
            self._leases.append(complete_file)
            complete = _parse(complete_file.stream.read())
            if (complete.get("schema") != COMPLETE_SCHEMA or complete.get("status") != "complete"
                    or complete.get("spec_sha256") != spec.identity or complete.get("manifest_sha256") != expected_manifest_sha256
                    or type(complete.get("total_bytes")) is not int or not 0 < complete["total_bytes"] <= maximum_output_bytes
                    or type(complete.get("manifest_bytes")) is not int):
                raise ValueError("Exact complete forecast identity and output bounds required")
            manifest_file = _open_verified(self._root_fd, "MANIFEST.json", expected_manifest_sha256,
                size=complete["manifest_bytes"], maximum=_metadata_cap(spec))
            self._leases.append(manifest_file)
            manifest = _parse(manifest_file.stream.read())
            if (manifest.get("schema") != SCHEMA or manifest.get("spec_sha256") != spec.identity
                    or _json_bytes(manifest.get("spec")) != spec._encoded
                    or manifest.get("producer_sha256") != complete.get("producer_sha256")
                    or manifest.get("calibrated_states_stored") is not False or manifest.get("final_ensemble_stored") is not False):
                raise ValueError("Forecast manifest/specification identity mismatch")
            _sha(manifest["producer_sha256"])
            _receipt_check(manifest["forecast_receipt"], spec)
            if _hash(manifest["forecast_receipt"]) != manifest.get("forecast_receipt_sha256"):
                raise ValueError("Whole-core receipt digest differs")
            self._core_receipt_bytes = _encode_manifest(manifest["forecast_receipt"], _metadata_cap(spec))
            limits = ForecastStorageLimits(**manifest["limits"])
            estimate = storage_estimate(spec, limits)
            if (estimate != manifest.get("estimate") or estimate["maximum_working_bytes"] > maximum_working_bytes
                    or complete.get("output_cap_bytes") != limits.output_cap_bytes):
                raise ValueError("Reader working limit/publication estimate differs")
            self._keys = tuple(spec.document["supported_guide_indices"]) + ((POOL,) if spec.document["pooled_control_identity"] else ())
            self._rows = {key: i for i, key in enumerate(self._keys)}
            self._prefix_rows = {p: i for i, p in enumerate(spec.effective_prefixes)}
            self._weight_slices, cursor = {}, 0
            for p in spec.effective_prefixes:
                self._weight_slices[p] = slice(cursor, cursor+p); cursor += p
            if complete.get("effective_prefixes") != list(spec.effective_prefixes) or complete.get("law_count") != len(self._keys):
                raise ValueError("Completion inventory differs")
            records = manifest.get("arrays")
            if not isinstance(records, list) or len(records) != len(ARRAY_NAMES): raise ValueError("Exact packed array inventory required")
            for record, (name, (dtype, shape)) in zip(records, _layout(spec).items()):
                header = _header(dtype, shape)
                expected = dict(name=name, relative_path=name+".npy", dtype=dtype.str, shape=list(shape),
                    data_offset=len(header), file_bytes=len(header)+math.prod(shape)*dtype.itemsize,
                    header_sha256=hashlib.sha256(header).hexdigest())
                if set(record) != set(expected) | {"file_sha256", "array_sha256"} or any(record[k] != v for k, v in expected.items()):
                    raise ValueError("Exact numeric array schema/header required; no pickle")
                file = _open_verified(self._root_fd, name+".npy", record["file_sha256"],
                    size=expected["file_bytes"], maximum=maximum_output_bytes)
                self._leases.append(file)
                if file.stream.read(len(header)) != header: raise ValueError("Canonical array header differs")
                array = (np.memmap(file.stream, mode="r", dtype=dtype, shape=shape, offset=len(header))
                         if math.prod(shape) else np.empty(shape, dtype=dtype))
                array.setflags(write=False)
                if _array_hash(array) != record["array_sha256"]: raise ValueError("Array content identity differs")
                self._maps[name] = array
            if sum(f.size for f in self._leases) != complete["total_bytes"]:
                raise ValueError("Complete byte inventory differs")
            expected_hashes = [(p, key) for p in spec.effective_prefixes for key in self._keys]
            hashes = manifest.get("raw_prefix_hashes")
            if not isinstance(hashes, list) or len(hashes) != len(expected_hashes): raise ValueError("Complete raw-prefix evidence required")
            self._raw_hashes = {}
            for entry, (p, key) in zip(hashes, expected_hashes):
                if set(entry) != {"prefix", "law", "sha256"} or entry["prefix"] != p or entry["law"] != key:
                    raise ValueError("Raw-prefix inventory differs")
                _sha(entry["sha256"])
                if _array_hash(self._maps["raw_states"][self._rows[key], :p]) != entry["sha256"]:
                    raise ValueError("Raw-prefix bank reconstruction differs")
                self._raw_hashes[p, key] = entry["sha256"]
            self.publication_reference = MappingProxyType(dict(root=str(self.root), complete_sha256=expected_complete_sha256,
                manifest_sha256=expected_manifest_sha256, spec_sha256=spec.identity, total_bytes=complete["total_bytes"]))
            self._lease_check()
        except BaseException:
            self.close(verify_hashes=False)
            raise

    def _lease_check(self, full_hash=False):
        if self._closed or self._pid != os.getpid(): raise RuntimeError("Live same-process forecast lease required")
        if self.spec is not self._original_spec or self.root != self._original_root:
            raise ValueError("Forecast reader source/root binding changed")
        if (_stamp(_path(self.root).lstat()) != self._root_stamp or _stamp(os.fstat(self._root_fd)) != self._root_stamp
                or set(os.listdir(self._root_fd)) != self._names):
            raise ValueError("Forecast directory identity/inventory changed")
        if self._runtime_binding != _runtime(): raise ValueError("Reader reconstruction runtime changed")
        for file in self._leases:
            if (_stamp(os.fstat(file.stream.fileno())) != file.stamp or
                    _stamp(os.stat(file.name, dir_fd=self._root_fd, follow_symlinks=False)) != file.stamp):
                raise ValueError("Forecast file changed: " + file.name)
            if full_hash and _digest_stream(file.stream) != file.sha256: raise ValueError("Forecast bytes changed: " + file.name)
        if full_hash: self._lease_check()

    def __enter__(self):
        if self._entered: raise RuntimeError("Reader cannot be entered twice")
        try:
            self._lease_check(); self._entered = True
            return self
        except BaseException:
            self.close(verify_hashes=False)
            raise

    @property
    def forecast_receipt(self):
        self._access()
        return _parse(self._core_receipt_bytes)

    def _access(self):
        if not self._entered: raise RuntimeError("Read forecast only inside its context")
        self._lease_check()

    def law(self, prefix, index):
        self._access()
        if type(prefix) is not int or prefix not in self._prefix_rows or (type(index) is not int and index != POOL):
            raise ValueError("Exact registered prefix and law index required")
        if index not in self._rows:
            if index == POOL or (type(index) is int and index in self._missing): return None
            raise ValueError("Law is outside the complete forecast inventory")
        row, k = self._rows[index], self._prefix_rows[prefix]
        raw = torch.from_numpy(np.array(self._maps["raw_states"][row, :prefix], copy=True))
        weights = torch.from_numpy(np.array(self._maps["log_weights"][row, self._weight_slices[prefix]], copy=True))
        mass = float(self._maps["log_mass"][k, row])
        if (not bool(torch.isfinite(raw).all()) or not bool(torch.isfinite(weights).all()) or not math.isfinite(mass)
                or abs(float(torch.logsumexp(weights, 0))) > 1e-10
                or _array_hash(raw.numpy()) != self._raw_hashes[prefix, index]):
            raise ValueError("Stored conditional law is invalid")
        states = calibrate_dispersion(raw, weights, self._spread)
        self._lease_check()
        return ConditionalForecast(raw, states, weights, mass)

    def catalog(self, prefix):
        self._access()
        if type(prefix) is not int or prefix not in self._prefix_rows: raise ValueError("Exact registered prefix required")
        k = self._prefix_rows[prefix]
        values = tuple(torch.from_numpy(np.array(self._maps["catalogs"][k, :, c], copy=True)) for c in range(4))
        masses = torch.from_numpy(np.array(self._maps["log_mass"][k, :self._supported_count], copy=True))
        _catalog_check(self.spec, prefix, values, masses)
        self._lease_check()
        return dict(zip(CATALOG_FIELDS, values))

    def close(self, *, verify_hashes=True):
        if self._closed: return
        try:
            if verify_hashes and self._root_fd is not None: self._lease_check(full_hash=True)
        finally:
            self._maps.clear()
            for file in self._leases: file.stream.close()
            if self._root_fd is not None: os.close(self._root_fd)
            self._closed, self._entered = True, False

    def __exit__(self, exc_type, exc, tb):
        self.close()
