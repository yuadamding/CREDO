"""Ordered sparse native rows -> full-support fold-bound population arrays.

This is an in-memory transformation, not an input-publication or output owner.
The caller supplies role-admitted chunks and the COMPLETE-selected frozen
observation module (normally from observation_loader.load_observation). No file,
checkpoint, network, optimizer or scheduler operation occurs here. A matching
hash assertion does not itself prove that the caller obtained those admissions.

NativeHandle.iterate_rows SHUFFLES rows. Do not pass its output as an ordered
stream: supply NativeChunk objects from bounded read_rows calls in the declared
shard/row order instead. Every captured row is checked, including zero-RNA rows.
All positive-RNA rows, including observation technical holdouts, are encoded
after observation selection without changing its scaler or selection.

Raw equal-positive-cell mean compositions can be reused across fitting folds;
latent arrays are always recomputed with the exact fold-native encoder/scaler.
Caller-owned mmap arrays may be used as sinks, but publication, unique ownership,
input leases, durability and failure cleanup remain an external worker's job.
"""

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json

import numpy as np
from scipy import sparse
import torch

from .fitting_provider import (CatalogGuide, PopulationArrays, PopulationReference,
                               _hash, _sha, _text)
from .configuration import DONORS
from .native_inputs import FoldInputs, NativeBatch, NativeHandle, VerifiedForecastRelease
from .observation import SparseR48, validate_counts
from .training_owner import ObservationBinding, observation_state_digest


SCHEMA = "credo.frontier.streaming-fitting-cache.v1"


@dataclass(frozen=True)
class NativeShard:
    shard: int
    rows: int

    def __post_init__(self):
        if type(self.shard) is not int or self.shard < 0 or type(self.rows) is not int or self.rows < 1:
            raise ValueError("Explicit nonempty native shard and row count required")


@dataclass(frozen=True)
class SourceCompileSpec:
    fold: FoldInputs
    catalog: tuple[CatalogGuide, ...]
    features: int
    feature_axis_sha256: str
    observation_binding: ObservationBinding
    donor: str
    condition: str
    source_id: str
    replicate_id: str
    access_role: str
    source_access_sha256: str
    source_input_sha256: str
    shards: tuple[NativeShard, ...]
    captured_counts: tuple[int, ...]
    positive_counts: tuple[int, ...]
    synthetic_fixture: bool = False

    def __post_init__(self):
        if (not isinstance(self.fold, FoldInputs) or not isinstance(self.observation_binding, ObservationBinding)
                or self.donor not in self.fold.training_donors
                or self.condition not in ("Stim8hr", "Stim48hr")
                or self.access_role != {"Stim8hr": "dynamics_source", "Stim48hr": "dynamics_supervision"}[self.condition]):
            raise ValueError("Exact fitting donor, condition and dynamics role required")
        self._validate_fields()

    def _validate_fields(self):
        for field in ("source_id", "replicate_id"):
            _text(getattr(self, field))
        for field in ("feature_axis_sha256", "source_access_sha256", "source_input_sha256"):
            _sha(getattr(self, field))
        if (type(self.catalog) is not tuple or not self.catalog
                or not all(isinstance(g, CatalogGuide) for g in self.catalog)
                or len({g.guide_id for g in self.catalog}) != len(self.catalog)
                or type(self.features) is not int or self.features < 1
                or type(self.synthetic_fixture) is not bool):
            raise ValueError("Exact complete catalog and feature count required")
        if not self.synthetic_fixture and (len(self.catalog), self.features) != (26504, 18129):
            raise ValueError("Scientific compilation retains all 26,504 guides and 18,129 RNA genes")
        if self.synthetic_fixture and (len(self.catalog) > 64 or self.features > 64):
            raise ValueError("Fabricated compilation exceeds its small law/feature scope")
        if (type(self.shards) is not tuple or not self.shards
                or not all(isinstance(s, NativeShard) for s in self.shards)
                or tuple(s.shard for s in self.shards) != tuple(sorted({s.shard for s in self.shards}))):
            raise ValueError("Explicit ascending unique native shard inventory required")
        for counts in (self.captured_counts, self.positive_counts):
            if (type(counts) is not tuple or len(counts) != len(self.catalog)
                    or any(type(n) is not int or n < 0 or n > np.iinfo(np.int64).max for n in counts)):
                raise ValueError("Complete nonnegative int64-range catalog support vectors required")
        if (sum(self.captured_counts) != sum(s.rows for s in self.shards)
                or sum(self.captured_counts) > np.iinfo(np.int64).max
                or any(p > n for p, n in zip(self.positive_counts, self.captured_counts))):
            raise ValueError("Complete captured denominator and positive-RNA support disagree")

    @property
    def catalog_sha256(self):
        return _hash([asdict(g) for g in self.catalog])

    @property
    def identity(self):
        return _hash(dict(schema=SCHEMA, **asdict(self)))


@dataclass(frozen=True)
class QuerySourceCompileSpec(SourceCompileSpec):
    """Explicit held-out early-source capability, never an endpoint capability.

    Inheriting shared array validation does not relax SourceCompileSpec's
    fitting-only role gate. The observation fold remains the training fold.
    """
    def __post_init__(self):
        if (not isinstance(self.fold, FoldInputs) or not isinstance(self.observation_binding, ObservationBinding)
                or self.donor not in DONORS or self.donor in self.fold.training_donors
                or self.condition != "Stim8hr" or self.access_role != "query"):
            raise ValueError("Exact held-out Stim8hr query source only; no endpoint or fitting role")
        self._validate_fields()

    @property
    def identity(self):
        return _hash(dict(schema="credo.frontier.streaming-query-source.v1", **asdict(self)))


@dataclass(frozen=True)
class EvaluationTruthCompileSpec(SourceCompileSpec):
    """Late held-out observations only after the exact forecast release.

    This separate role never broadens fitting or early-source specifications.
    NativeInputIndex additionally checks the release against its query identity.
    """
    release: VerifiedForecastRelease = field(kw_only=True)

    def __post_init__(self):
        if (not isinstance(self.fold, FoldInputs) or not isinstance(self.observation_binding, ObservationBinding)
                or self.donor not in DONORS or self.donor in self.fold.training_donors
                or self.condition != "Stim48hr" or self.access_role != "evaluation_truth"
                or not isinstance(self.release, VerifiedForecastRelease)
                or self.release.expectation.fold_sha256 != self.fold.identity
                or self.release.expectation.representation_sha256 != self.observation_binding.artifact_sha256):
            raise ValueError("Exact released held-out Stim48hr evaluation and fitting representation required")
        self._validate_fields()

    @property
    def identity(self):
        return _hash(dict(schema="credo.frontier.streaming-evaluation-truth.v1", **asdict(self)))


@dataclass(frozen=True)
class NativeChunk:
    """Explicit coordinates, not the reader's shuffled StreamCursor."""
    shard: int
    start_row: int
    batch: NativeBatch

    def __post_init__(self):
        if (type(self.shard) is not int or self.shard < 0
                or type(self.start_row) is not int or self.start_row < 0
                or not isinstance(self.batch, NativeBatch)):
            raise ValueError("NativeChunk needs exact shard/start coordinates and NativeBatch")


@dataclass(frozen=True)
class CompilerLimits:
    maximum_host_bytes: int
    maximum_chunk_rows: int
    maximum_chunk_bytes: int
    encoding_rows: int
    maximum_identity_bytes: int

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in asdict(self).values()):
            raise ValueError("Explicit positive compiler byte/row limits required")


@dataclass(frozen=True)
class ArraySink:
    """Fresh exclusively caller-owned writable arrays; no publication implied."""
    states: np.ndarray
    mean_compositions: np.ndarray | None


@dataclass(frozen=True)
class Compilation:
    population: PopulationArrays
    receipt: dict


def native_source_input_identity(index, source_id):
    """Metadata-only, fold-independent identity for the actual native inputs."""
    if source_id not in index.source_roles:
        raise ValueError("Unknown native source identity")
    shards = tuple(s for s in index.shards if s.source_id == source_id)
    if not shards or any(s.allowed_rows is not None for s in shards):
        raise ValueError("Only complete source shards may supply fitting caches")
    return _hash(dict(schema="credo.frontier.native-cache-source.v1", source_id=source_id,
        source_role=index.source_roles[source_id], package_binding=asdict(index.binding),
        rna_axis_sha256=index.rna_axis_sha256,
        catalog=index.catalog().to_dict("records"),
        accounting_sha256=index.inventory[f"source/{source_id}/population_accounting.parquet"],
        shards=[s.model_dump(mode="json") for s in shards]))


def ordered_native_chunks(handle, spec, *, batch_rows):
    """Explicit payload boundary, reading only one role-admitted native source.

    Checks metadata/capability identity before the first bounded read_rows call;
    never uses shuffled iterate_rows. Input leases and final exit verification
    still belong to the outer worker. Merely constructing this generator reads
    nothing, and no filesystem access occurs in this module itself.
    """
    if not isinstance(handle, NativeHandle) or not isinstance(spec, SourceCompileSpec):
        raise TypeError("Exact native handle and source compilation specification required")
    if type(batch_rows) is not int or not 0 < batch_rows <= 8192:
        raise ValueError("Explicit native reader batch size in 1..8192 required")
    index = handle.index
    query = {"query_donor": spec.donor} if isinstance(spec, (QuerySourceCompileSpec, EvaluationTruthCompileSpec)) else {}
    if isinstance(spec, EvaluationTruthCompileSpec):
        query["release"] = spec.release
    expected = index.access(spec.fold, spec.access_role, **query)
    selected = tuple(s for s in handle.access.shards if s.source_id == spec.source_id)
    if (handle.access.identity() != expected.identity()
            or expected.identity() != spec.source_access_sha256
            or spec.source_id not in handle.access.source_ids
            or index.source_roles.get(spec.source_id) != (spec.donor, spec.condition)
            or index.rna_axis_sha256 != spec.feature_axis_sha256
            or index.binding.synthetic_fixture != spec.synthetic_fixture
            or tuple((s.shard, s.rows) for s in selected) != tuple((s.shard, s.rows) for s in spec.shards)
            or any(s.allowed_rows is not None for s in selected)
            or native_source_input_identity(index, spec.source_id) != spec.source_input_sha256):
        raise ValueError("Native source role/fold/feature/input/shard identity differs")
    actual_catalog = tuple(CatalogGuide(str(row.guide_id), str(row.target_id), bool(row.is_control))
                           for row in index.catalog().itertuples(index=False))
    if actual_catalog != spec.catalog:
        raise ValueError("Native complete catalog differs from compiler catalog")
    for shard in spec.shards:
        for start in range(0, shard.rows, batch_rows):
            addresses = [(spec.source_id, shard.shard, row)
                         for row in range(start, min(start + batch_rows, shard.rows))]
            yield NativeChunk(shard.shard, start, handle.read_rows(addresses))


def _array(value, dtype, shape, *, writable):
    if (not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype)
            or value.shape != shape or not value.flags.c_contiguous
            or value.flags.writeable != writable):
        raise ValueError("Array sink/shared summary has wrong shape, dtype, layout or mutability")


def _row_record(spec, shard, row, cell, guide, depth):
    return json.dumps([spec.source_id, shard, row, cell, guide, depth],
                      ensure_ascii=False, separators=(",", ":")).encode() + b"\n"


def _verify_reuse(spec, population):
    if not isinstance(population, PopulationArrays):
        raise ValueError("Raw-summary reuse requires an admitted PopulationArrays reference")
    ref = population.reference
    expected = (spec.donor, spec.condition, spec.source_id, spec.replicate_id, spec.source_input_sha256)
    if (ref.donor, ref.condition, ref.source_id, ref.replicate_id, ref.source_input_sha256) != expected:
        raise ValueError("Reused raw summary belongs to another source/input or population definition")
    g, j = len(spec.catalog), spec.features
    for values in (population.captured_counts, population.positive_counts):
        _array(values, np.int64, (g,), writable=False)
    _array(population.mean_compositions, np.float64, (g, j), writable=False)
    if (not np.array_equal(population.captured_counts, spec.captured_counts)
            or not np.array_equal(population.positive_counts, spec.positive_counts)
            or population.raw_identity(catalog_sha256=spec.catalog_sha256,
                feature_axis_sha256=spec.feature_axis_sha256) != ref.raw_summary_sha256):
        raise ValueError("Reused raw summary content/support/feature catalog identity differs")
    for start in range(0, g, 64):
        means = population.mean_compositions[start:start + 64]
        supported = population.positive_counts[start:start + 64] > 0
        if (not np.isfinite(means).all() or np.any(means < 0)
                or not np.allclose(means[supported].sum(1), 1, rtol=0, atol=1e-10)
                or np.any(means[~supported] != 0)):
            raise ValueError("Reused raw summary has invalid unavailable/normalized mean rows")


def compile_population(spec, chunks, observation, *, limits, sink=None, reuse_raw=None):
    """Compile exactly one complete source, returning read-only provider arrays.

    No input stream is opened here. Admission checks precede its iteration.
    On any exception the supplied mutable sink is provisional and MUST NOT be
    published. The owner is responsible for its cleanup/no-clobber semantics.
    The conservative working estimate is not a measured RSS/GPU qualification.
    """
    if not isinstance(spec, SourceCompileSpec) or not isinstance(limits, CompilerLimits):
        raise TypeError("Typed source specification and resource limits required")
    if (not isinstance(observation, SparseR48) or observation.features != spec.features
            or observation.latent_dim != 48
            or observation.fold.fold_id != spec.fold.fold_id
            or observation.fold.training_donors != spec.fold.training_donors
            or observation.fold.feature_axis_sha256 != spec.feature_axis_sha256
            or limits.encoding_rows > observation.maximum_rows):
        raise ValueError("Exact fold-native observation model/scaler/encoding limit required")
    if spec.synthetic_fixture and observation.device.type != "cpu":
        raise ValueError("Fabricated cache compiler scope is CPU-only")
    if observation_state_digest(observation) != spec.observation_binding.state_sha256:
        raise ValueError("Actual frozen observation weights/scaler differ from binding")
    observation._budget(limits.encoding_rows)
    g, j, n = len(spec.catalog), spec.features, sum(spec.positive_counts)
    if reuse_raw is not None:
        _verify_reuse(spec, reuse_raw)
    # Include referenced shared arrays conservatively, even if mmap-backed. Six
    # chunk allowances cover admitted CSR/metadata, selected CSR, buffering,
    # vstack and sparse normalization/index temporaries. The identity set uses
    # fixed 32-byte digests with a conservative 192-byte allowance per row.
    identity_bytes = sum(spec.captured_counts) * 192
    if identity_bytes > limits.maximum_identity_bytes:
        raise ValueError("Complete duplicate-row identity guard exceeds cap; no truncation")
    output_bytes = n * 48 * 4 + g * j * 8 + (5 * g + 1) * 8
    # PopulationArrays retains the old fold's states/offsets while this call
    # writes fresh latent arrays. Count those references even when mmap-backed;
    # the shared raw means already appear in output_bytes above.
    retained_latent_bytes = (0 if reuse_raw is None else
                            reuse_raw.states.nbytes + reuse_raw.offsets.nbytes)
    model_bytes = sum(t.numel() * t.element_size() for t in observation.state_dict().values())
    encoder_bytes = limits.encoding_rows * (j + 16 * sum(observation.hidden_dims) + 8 * 48) * 4
    # Buffered encoding batches may span MANY tiny input chunks, so their size
    # cannot be bounded by maximum_chunk_bytes. Worst-case dense CSR support
    # allows 80 bytes/nonzero for retained pieces, vstack, sparse coordinates,
    # normalization and projection temporaries (counts themselves stay sparse).
    sparse_buffer_bytes = limits.encoding_rows * j * 80 + (limits.encoding_rows + 1) * 16
    summary_validation_bytes = min(g, 64) * j * 24  # finite masks, supported-row copy and reductions
    estimate = (output_bytes + retained_latent_bytes + 6 * limits.maximum_chunk_bytes + identity_bytes
                + 2 * model_bytes + 2 * encoder_bytes + sparse_buffer_bytes + summary_validation_bytes)
    if estimate > limits.maximum_host_bytes:
        raise ValueError("Complete compiler output/working estimate exceeds explicit host-byte cap")
    if sink is None:
        sink = ArraySink(np.empty((n, 48), np.float32), None if reuse_raw is not None else np.empty((g, j), np.float64))
    if not isinstance(sink, ArraySink):
        raise TypeError("Explicit ArraySink required")
    _array(sink.states, np.float32, (n, 48), writable=True)
    if reuse_raw is not None:
        reused = (reuse_raw.states, reuse_raw.mean_compositions, reuse_raw.captured_counts,
                  reuse_raw.positive_counts, reuse_raw.offsets)
        if sink.mean_compositions is not None or any(np.shares_memory(sink.states, a) for a in reused):
            raise ValueError("Raw reuse needs a fresh latent sink and no duplicate raw-summary sink")
        means = reuse_raw.mean_compositions
    else:
        _array(sink.mean_compositions, np.float64, (g, j), writable=True)
        if np.shares_memory(sink.states, sink.mean_compositions):
            raise ValueError("Latent and raw-summary sink storage must not overlap")
        means = sink.mean_compositions
        means.fill(0)
    offsets = np.r_[0, np.cumsum(np.asarray(spec.positive_counts, np.int64))]
    written = np.zeros(g, np.int64)
    captured, positive = np.zeros(g, np.int64), np.zeros(g, np.int64)
    seen = set()
    digest = hashlib.sha256(b"credo.frontier.canonical-native-rows.v1\n")
    pieces, guide_pieces, buffered = [], [], 0
    encoded_batches = 0
    max_chunk_bytes = 0

    def flush():
        nonlocal buffered, encoded_batches
        if not buffered:
            return
        counts = sparse.vstack(pieces, format="csr")
        guides = np.concatenate(guide_pieces)
        with torch.no_grad():
            z = observation.encode(counts).detach().to("cpu").numpy()
        if z.dtype != np.float32 or z.shape != (buffered, 48) or not np.isfinite(z).all():
            raise ValueError("Encoder returned invalid standardized states")
        # Group only inside a bounded encoder batch; within-guide native order
        # is retained and no endpoint support is subsampled or deduplicated.
        for guide in np.unique(guides):
            rows = z[guides == guide]
            left = int(offsets[guide] + written[guide])
            if written[guide] + len(rows) > spec.positive_counts[guide]:
                raise ValueError("Observed positive-RNA support exceeds admitted complete inventory")
            sink.states[left:left + len(rows)] = rows
            written[guide] += len(rows)
        pieces.clear(); guide_pieces.clear()
        buffered = 0
        encoded_batches += 1

    shard_position, next_row = 0, 0
    for chunk in chunks:
        if not isinstance(chunk, NativeChunk) or shard_position >= len(spec.shards):
            raise ValueError("Extra/non-native chunk outside complete source inventory")
        expected = spec.shards[shard_position]
        if (chunk.shard, chunk.start_row) != (expected.shard, next_row):
            raise ValueError("Native chunks must be contiguous, complete and in canonical shard/row order")
        b, counts = chunk.batch, chunk.batch.rna_counts
        validate_counts(counts, features=j, maximum_rows=limits.maximum_chunk_rows)
        rows = counts.shape[0]
        if next_row + rows > expected.rows or len(b.cells) != rows or len(b.row_ids) != rows:
            raise ValueError("Chunk row count crosses a native shard or metadata boundary")
        required = {"source_id", "cell_id", "row_in_shard", "guide_index", "donor_id", "condition", "RNA_UMIs"}
        if not required <= set(b.cells.columns):
            raise ValueError("Native cell table lacks explicit source/row/catalog/RNA identities")
        byte_count = counts.data.nbytes + counts.indices.nbytes + counts.indptr.nbytes
        byte_count += int(b.cells.memory_usage(index=True, deep=True).sum())
        byte_count += sum(len(r.encode()) + 64 for r in b.row_ids if isinstance(r, str))
        byte_count += np.asarray(b.rna_depth).nbytes + np.asarray(b.positive_rna).nbytes
        if byte_count > limits.maximum_chunk_bytes:
            raise ValueError("Admitted sparse chunk and metadata exceed compiler byte cap")
        max_chunk_bytes = max(max_chunk_bytes, byte_count)
        cells = b.cells
        guides = cells.guide_index.to_numpy()
        native_rows = cells.row_in_shard.to_numpy()
        if (guides.dtype.kind not in "iu" or np.any(guides >= g) or np.any(guides < 0)
                or native_rows.dtype.kind not in "iu"
                or not np.array_equal(native_rows, np.arange(next_row, next_row + rows))
                or not cells.source_id.eq(spec.source_id).all()
                or not cells.donor_id.eq(spec.donor).all() or not cells.condition.eq(spec.condition).all()):
            raise ValueError("Native row/guide/donor/condition identity differs from exact source")
        guides = guides.astype(np.int64, copy=False)
        for column, attr in (("guide_id", "guide_id"), ("primary_target_id", "target_id"), ("is_control", "is_control")):
            if column in cells and not np.array_equal(cells[column], [getattr(spec.catalog[i], attr) for i in guides]):
                raise ValueError("Native guide/target/control labels differ from canonical catalog")
        depth = np.asarray(counts.sum(1, dtype=np.uint64)).ravel()
        if (not isinstance(b.rna_depth, np.ndarray) or not isinstance(b.positive_rna, np.ndarray)
                or cells.RNA_UMIs.dtype.kind not in "iu" or b.rna_depth.dtype.kind not in "iu"
                or b.positive_rna.dtype != np.bool_ or b.rna_depth.shape != (rows,)
                or b.positive_rna.shape != (rows,) or not np.array_equal(cells.RNA_UMIs, depth)
                or not np.array_equal(b.rna_depth, depth) or not np.array_equal(b.positive_rna, depth > 0)):
            raise ValueError("Sparse RNA depth/positive support differs from native metadata")
        for index, cell in enumerate(cells.cell_id):
            if type(cell) is not str or not cell:
                raise ValueError("Explicit nonempty native cell identifiers required")
            row_id = json.dumps([spec.source_id, cell], separators=(",", ":"))
            if b.row_ids[index] != row_id:
                raise ValueError("Native source-scoped row identity differs")
            key = hashlib.sha256(row_id.encode()).digest()
            if key in seen:
                raise ValueError("Duplicate cell identity across native chunks or shards")
            seen.add(key)
            digest.update(_row_record(spec, chunk.shard, next_row + index, cell, int(guides[index]), int(depth[index])))
        captured += np.bincount(guides, minlength=g)
        live = depth > 0
        positive += np.bincount(guides[live], minlength=g)
        if (np.any(captured > spec.captured_counts) or np.any(positive > spec.positive_counts)):
            raise ValueError("Streamed support exceeds complete admitted count vectors")
        if reuse_raw is None and counts.nnz:
            # One contribution per nonzero entry, ordered by native cell then
            # canonical gene. Explicit stored zeros in zero-RNA rows contribute
            # zero and never enter the equal-positive-cell denominator.
            entry_rows = np.repeat(np.arange(rows), np.diff(counts.indptr))
            contribution = counts.data.astype(np.float64) / np.maximum(depth[entry_rows], 1)
            np.add.at(means, (guides[entry_rows], counts.indices), contribution)
        live_counts, live_guides = counts[live], guides[live]
        start = 0
        while start < len(live_guides):
            take = min(limits.encoding_rows - buffered, len(live_guides) - start)
            pieces.append(live_counts[start:start + take])
            guide_pieces.append(live_guides[start:start + take].copy())
            buffered += take
            start += take
            if buffered == limits.encoding_rows:
                flush()
        next_row += rows
        if next_row == expected.rows:
            shard_position += 1
            next_row = 0
    if (shard_position != len(spec.shards) or next_row
            or not np.array_equal(captured, spec.captured_counts)
            or not np.array_equal(positive, spec.positive_counts)):
        raise ValueError("Incomplete native stream or changed complete captured/RNA support")
    flush()
    if not np.array_equal(written, positive):
        raise ValueError("Not every positive-RNA row produced exactly one latent state")
    row_sha = digest.hexdigest()
    if reuse_raw is not None:
        if row_sha != reuse_raw.reference.row_identity_sha256:
            raise ValueError("Raw-summary reuse has a different canonical native row inventory")
        captured, positive = reuse_raw.captured_counts, reuse_raw.positive_counts
        _verify_reuse(spec, reuse_raw)
    else:
        np.divide(means, np.maximum(positive, 1)[:, None], out=means)
    if observation_state_digest(observation) != spec.observation_binding.state_sha256:
        raise ValueError("Frozen observation/scaler changed during cache compilation")
    for value in (captured, positive, offsets, sink.states, means):
        value.setflags(write=False)
    ref = PopulationReference(spec.donor, spec.condition, spec.source_id, spec.replicate_id,
        spec.source_access_sha256, spec.source_input_sha256, row_sha, "0" * 64, "0" * 64)
    population = PopulationArrays(ref, captured, positive, offsets, sink.states, means)
    ref = replace(ref, raw_summary_sha256=population.raw_identity(catalog_sha256=spec.catalog_sha256,
                  feature_axis_sha256=spec.feature_axis_sha256),
                  latent_sha256=population.latent_identity(catalog_sha256=spec.catalog_sha256,
                  observation_binding=spec.observation_binding))
    population = replace(population, reference=ref)
    receipt = dict(schema=("credo.frontier.streaming-evaluation-truth.v1"
        if isinstance(spec, EvaluationTruthCompileSpec) else "credo.frontier.streaming-query-source.v1"
        if isinstance(spec, QuerySourceCompileSpec) else SCHEMA),
        status="in_memory_complete", specification_sha256=spec.identity,
        observation_binding=asdict(spec.observation_binding), source_id=spec.source_id,
        donor=spec.donor, condition=spec.condition, access_role=spec.access_role,
        captured_rows=int(captured.sum()), positive_RNA_rows=int(positive.sum()),
        catalog_categories=g, RNA_features=j, complete_native_shards=len(spec.shards),
        row_identity_sha256=row_sha, raw_summary_sha256=ref.raw_summary_sha256,
        latent_sha256=ref.latent_sha256, reused_raw_summary=reuse_raw is not None,
        encoding_rows=limits.encoding_rows, encoding_batches=encoded_batches,
        actual_maximum_chunk_bytes=max_chunk_bytes, conservative_host_estimate_bytes=estimate,
        encoder_device=str(observation.device), optimizer_calls=0,
        observation_technical_holdout_filter_applied=False, publication_claimed=False,
        input_lease_verified=False)
    return Compilation(population, receipt)
