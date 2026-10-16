"""Released late observations -> unchanged fold-native state and RNA laws.

This consumes an already verified exact forecast release and an already-created
evaluation capability. It never creates a release, fits a representation,
selects guide support, or reads another donor/condition. Callers retain the
forecast publication leases and the separate scientific operation admission.
"""

from dataclasses import asdict, dataclass
import json

from .cache_compiler import (CompilerLimits, EvaluationTruthCompileSpec, NativeShard,
    compile_population, native_source_input_identity, ordered_native_chunks)
from .fitting_provider import CatalogGuide, PopulationArrays, _hash
from .native_inputs import FoldInputs, NativeHandle, VerifiedForecastRelease
from .observation import SparseR48
from .training_owner import ObservationBinding, observation_state_digest


_TRUTH_FACTORY = object()


@dataclass(frozen=True, init=False)
class CompiledTruth:
    fold: FoldInputs
    catalog: tuple[CatalogGuide, ...]
    feature_axis_sha256: str
    observation_binding: ObservationBinding
    population: PopulationArrays
    release: VerifiedForecastRelease
    _receipt_bytes: bytes

    def __init__(self, fold, catalog, feature_axis_sha256, observation_binding,
                 population, release, receipt, *, _factory=None):
        if _factory is not _TRUTH_FACTORY:
            raise TypeError("Use the released endpoint compiler, not a constructed truth admission")
        for name, value in (("fold", fold), ("catalog", catalog), ("feature_axis_sha256", feature_axis_sha256),
                            ("observation_binding", observation_binding), ("population", population), ("release", release)):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_receipt_bytes", json.dumps(receipt, sort_keys=True, allow_nan=False).encode())

    @property
    def receipt(self):
        return json.loads(self._receipt_bytes)

    def assert_unchanged(self):
        p = self.population
        receipt = self.receipt
        if (p.reference.condition != "Stim48hr" or p.reference.donor in self.fold.training_donors
                or self.release.expectation.fold_sha256 != self.fold.identity
                or self.release.expectation.representation_sha256 != self.observation_binding.artifact_sha256
                or p.reference.donor != receipt["query_donor"] or p.reference.source_id != receipt["source_id"]
                or p.reference.source_access_sha256 != receipt["evaluation_access_sha256"]
                or self.release.manifest_sha256 != receipt["forecast_seal_sha256"]
                or asdict(self.observation_binding) != receipt["observation_binding"]
                or any(getattr(p, name).flags.writeable for name in
                       ("captured_counts", "positive_counts", "offsets", "states", "mean_compositions"))):
            raise ValueError("Released immutable evaluation population required")
        catalog_sha = _hash([asdict(g) for g in self.catalog])
        if (p.raw_identity(catalog_sha256=catalog_sha, feature_axis_sha256=self.feature_axis_sha256)
                != p.reference.raw_summary_sha256
                or p.latent_identity(catalog_sha256=catalog_sha, observation_binding=self.observation_binding)
                != p.reference.latent_sha256):
            raise ValueError("Evaluation population content identity changed")


def compile_evaluation_truth(handle, *, fold, donor, source_id, replicate_id,
                             observation, observation_binding, release, limits,
                             batch_rows, sink=None, reuse_raw=None):
    """Complete captured denominator and all positive-RNA endpoint cells.

    Missing guide RNA remains missing; captured zeros remain valid categories.
    Biological repeat/culture identity is supplied, never inferred here.
    """
    if (not isinstance(handle, NativeHandle) or not isinstance(fold, FoldInputs)
            or not isinstance(observation, SparseR48)
            or not isinstance(observation_binding, ObservationBinding)
            or not isinstance(release, VerifiedForecastRelease) or not isinstance(limits, CompilerLimits)):
        raise TypeError("Exact verified release, evaluation handle, representation and limits required")
    index = handle.index
    expected = index.access(fold, "evaluation_truth", query_donor=donor, release=release)
    if (handle.access.identity() != expected.identity() or tuple(expected.source_ids) != (source_id,)
            or index.source_roles.get(source_id) != (donor, "Stim48hr")
            or release.expectation.representation_sha256 != observation_binding.artifact_sha256):
        raise ValueError("Exact released held-out endpoint capability and representation required")
    if (observation.fold.fold_id != fold.fold_id or observation.fold.training_donors != fold.training_donors
            or observation.fold.feature_axis_sha256 != index.rna_axis_sha256
            or observation.features != index.binding.rna_features or observation.latent_dim != 48
            or observation_state_digest(observation) != observation_binding.state_sha256):
        raise ValueError("Actual selected fitting representation differs from released truth axis")
    if (type(batch_rows) is not int or not 1 <= batch_rows <= min(8192, limits.maximum_chunk_rows)
            or type(replicate_id) is not str or not replicate_id):
        raise ValueError("Explicit bounded batch rows and culture/replicate identity required")
    if index.binding.synthetic_fixture and observation.device.type != "cpu":
        raise ValueError("Fabricated endpoint compilation is CPU only")
    snapshot = handle.capture_publication_identity()
    support = handle.support_counts(source_id)
    catalog = tuple(CatalogGuide(row.guide_id, row.target_id, bool(row.is_control))
                    for row in index.catalog().itertuples(index=False))
    spec = EvaluationTruthCompileSpec(fold, catalog, index.binding.rna_features,
        index.rna_axis_sha256, observation_binding, donor, "Stim48hr", source_id,
        replicate_id, "evaluation_truth", expected.identity(), native_source_input_identity(index, source_id),
        tuple(NativeShard(s.shard, s.rows) for s in expected.shards),
        tuple(map(int, support.captured_cells)), tuple(map(int, support.positive_RNA_cells)),
        index.binding.synthetic_fixture, release=release)
    compiled = compile_population(spec, ordered_native_chunks(handle, spec, batch_rows=batch_rows),
        observation, limits=limits, sink=sink, reuse_raw=reuse_raw)
    exit_record = handle.verify_publication_identity(snapshot)
    receipt = dict(schema="credo.frontier.compiled-evaluation-truth.v1", status="in_memory_complete",
        fitting_fold=asdict(fold), query_donor=donor, source_id=source_id,
        forecast_seal_sha256=release.manifest_sha256, evaluation_access_sha256=expected.identity(),
        observation_binding=asdict(observation_binding), native_metadata_identity=index.identity,
        compilation=compiled.receipt, input_lease_exit=exit_record,
        source_based_eligibility_changed=False, representation_refitted=False,
        biological_replication_inferred=False, optimizer_calls=0, publication_claimed=False,
        scientific_launch_authority=False)
    result = CompiledTruth(fold, catalog, index.rna_axis_sha256, observation_binding,
                           compiled.population, release, receipt, _factory=_TRUTH_FACTORY)
    result.assert_unchanged()
    return result
