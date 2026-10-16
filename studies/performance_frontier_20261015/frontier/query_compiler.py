"""One admitted held-out Stim8hr source -> fold-native forecast population.

This is an explicit numerical payload boundary, not scheduler permission.
It consumes only an already-created query NativeHandle, checks its exact role,
captures a same-worker input lease, and uses the selected fitting observation
model unchanged. It does not accept a release, endpoint handle or fitting donor.
Source RNA means remain allowed early inputs; no endpoint is reconstructed here.
"""

from dataclasses import asdict, dataclass

from .cache_compiler import (CompilerLimits, NativeShard, QuerySourceCompileSpec,
    compile_population, native_source_input_identity, ordered_native_chunks)
from .fitting_provider import CatalogGuide
from .forecast import ForecastSource
from .native_inputs import FoldInputs, NativeHandle
from .observation import SparseR48
from .training_owner import ObservationBinding, observation_state_digest


@dataclass(frozen=True)
class CompiledQuery:
    """Source plus early RNA summaries; these share, rather than copy, latents.

    A forecast core charges its ForecastSource arrays, not this object's extra
    RNA matrix or the caller's observation model. The owner must either release
    these extras or charge them to its aggregate worker-memory admission.
    """
    source: ForecastSource
    population: object
    receipt: dict


def compile_query_source(handle, *, fold, donor, source_id, replicate_id,
                         observation, observation_binding, limits, batch_rows,
                         sink=None, reuse_raw=None):
    """Compile all positive-RNA early cells and retain complete captured counts.

    The caller binds culture/replicate metadata explicitly; this routine does
    not infer biological replication from an accession or a capture filename.
    A single-source query denominator is required, not an arbitrary subset of
    multiple populations. Publication and retention are separate operations.
    """
    if (not isinstance(handle, NativeHandle) or not isinstance(fold, FoldInputs)
            or not isinstance(observation, SparseR48)
            or not isinstance(observation_binding, ObservationBinding)
            or not isinstance(limits, CompilerLimits)):
        raise TypeError("Exact query handle, fold, selected observation and resource bindings required")
    index = handle.index
    expected = index.access(fold, "query", query_donor=donor)
    if (handle.access.identity() != expected.identity()
            or tuple(expected.source_ids) != (source_id,)
            or index.source_roles.get(source_id) != (donor, "Stim8hr")):
        raise ValueError("Complete held-out Stim8hr query capability required before payload access")
    if (observation.fold.fold_id != fold.fold_id
            or observation.fold.training_donors != fold.training_donors
            or observation.fold.feature_axis_sha256 != index.rna_axis_sha256
            or observation.features != index.binding.rna_features
            or observation.latent_dim != 48
            or observation_state_digest(observation) != observation_binding.state_sha256):
        raise ValueError("Actual selected observation differs from the fitting fold/query axis")
    if (type(batch_rows) is not int or not 1 <= batch_rows <= min(8192, limits.maximum_chunk_rows)
            or type(replicate_id) is not str or not replicate_id):
        raise ValueError("Explicit bounded batch rows and culture/replicate identity required")
    if index.binding.synthetic_fixture and observation.device.type != "cpu":
        raise ValueError("Fabricated query compilation is CPU only")
    snapshot = handle.capture_publication_identity()
    support = handle.support_counts(source_id)
    catalog = tuple(CatalogGuide(row.guide_id, row.target_id, bool(row.is_control))
                    for row in index.catalog().itertuples(index=False))
    spec = QuerySourceCompileSpec(fold, catalog, index.binding.rna_features,
        index.rna_axis_sha256, observation_binding, donor, "Stim8hr", source_id,
        replicate_id, "query", expected.identity(), native_source_input_identity(index, source_id),
        tuple(NativeShard(s.shard, s.rows) for s in expected.shards),
        tuple(map(int, support.captured_cells)), tuple(map(int, support.positive_RNA_cells)),
        index.binding.synthetic_fixture)
    result = compile_population(spec, ordered_native_chunks(handle, spec, batch_rows=batch_rows),
        observation, limits=limits, sink=sink, reuse_raw=reuse_raw)
    population = result.population
    source = ForecastSource(fold, catalog, observation_binding, index.rna_axis_sha256,
        population.reference, population.captured_counts, population.positive_counts,
        population.offsets, population.states, index.binding.synthetic_fixture)
    exit_record = handle.verify_publication_identity(snapshot)
    receipt = dict(schema="credo.frontier.compiled-query.v1", status="in_memory_complete",
        fitting_fold=asdict(fold), query_donor=donor, source_id=source_id,
        source_identity=source.identity, query_access_sha256=expected.identity(),
        native_metadata_identity=index.identity, compilation=result.receipt,
        retained_early_RNA_summary_bytes=(0 if population.mean_compositions is None
                                          else population.mean_compositions.nbytes),
        retained_early_RNA_summary_charged_by_forecast_core=False,
        input_lease_sha256=exit_record["input_lease_sha256"], input_lease_exit=exit_record,
        representation_refitted=False, query_endpoint_used=False, optimizer_calls=0,
        biological_replication_inferred=False, publication_claimed=False,
        scientific_launch_authority=False)
    return CompiledQuery(source, population, receipt)
