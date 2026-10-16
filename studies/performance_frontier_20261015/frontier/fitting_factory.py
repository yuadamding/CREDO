"""Verified selected observation + leased cache + saved scales -> fit owner.

No training step, scheduler call, outcome release or recovery occurs here.
Preparation loads the actual selected observation export and constructs a fresh
registered model. The owner is opened only after exact identity reconciliation.
Scientific use still requires worker/resource/input admission and H100 numerical
qualification. Hash verification proves retained bytes, not biological validity.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import torch

from . import configuration, models, normalization, sampling
from .cache_compiler import native_source_input_identity
from .checkpoints import file_sha256
from .configuration import SEEDS, get_configuration
from .fitting_provider import CatalogGuide, FoldFittingProvider, _hash, _sha
from .models import build_model
from .native_inputs import NativeInputIndex
from .objective_registry import (FittingScaleBinding, ObjectiveRules, SourceSupportRecord,
                                 build_objective_protocol, source_support_factors)
from .observation_loader import ObservationArtifactAdmission, load_observation
from .sampling import SamplingContext, SamplerSpecification
from .training_owner import TrainingOwner, owner_binding


def forecast_model_sources(producer_source_sha256):
    """Producer-bound equations required for later frozen model loading.

    This is part of the checkpoint identity, not a caller-supplied inference
    manifest. The worker must additionally execute its admitted immutable source
    bundle; this small record permits new publication code without relabeling
    changed forward equations as the fitted predictor.
    """
    _sha(producer_source_sha256)
    files = {}
    for name, module in (("frontier/models.py", models),
                         ("frontier/configuration.py", configuration)):
        payload = Path(module.__file__).read_bytes()
        files[name] = dict(sha256=hashlib.sha256(payload).hexdigest(), size_bytes=len(payload))
    return dict(schema="credo.frontier.forecast-model-sources.v1",
        producer_source_sha256=producer_source_sha256, files=files)


@dataclass(frozen=True)
class NormalizerArtifact:
    """Already read immutable bytes with an independently admitted file hash.

    This is intentionally not a path-based loader or a request to recompute.
    maximum_bytes is the caller's explicit input budget, not a new allocation.
    """
    payload: bytes
    file_sha256: str
    maximum_bytes: int

    def __post_init__(self):
        _sha(self.file_sha256)
        if (type(self.payload) is not bytes or not self.payload
                or type(self.maximum_bytes) is not int or self.maximum_bytes < 1
                or len(self.payload) > self.maximum_bytes
                or hashlib.sha256(self.payload).hexdigest() != self.file_sha256):
            raise ValueError("Exact bounded immutable normalizer artifact bytes required")


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate normalizer JSON key")
        result[key] = value
    return result


def bind_normalizers(cache, artifact, *, study_id, study_seed, geometry):
    """Verify bytes, all scalar reductions, support and actual producer binding.

    Reconciles saved scalar scores only; never recalculates population geometry
    or molecular means. Upstream numerical correctness remains separate evidence.
    """
    if not isinstance(artifact, NormalizerArtifact):
        raise TypeError("Exact normalizer artifact required")
    artifact.__post_init__()
    cache.assert_unchanged()
    record = json.loads(artifact.payload, object_pairs_hook=_unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite normalizer metadata")))
    if not isinstance(record, dict) or "content_sha256" not in record:
        raise ValueError("Complete computed normalizer record required")
    body = {key: value for key, value in record.items() if key != "content_sha256"}
    if _hash(body) != record["content_sha256"]:
        raise ValueError("Normalizer content identity differs from verified bytes")
    manifest = cache.manifest
    expected = dict(schema="credo.frontier.computed-normalizers.v1",
        protocol_sha256=normalization.PROTOCOL_SHA256, objective_rules_sha256=manifest.rules_sha256,
        fitting_donors=list(manifest.fold.training_donors), fitting_input_sha256=manifest.input_sha256,
        cache_manifest_sha256=manifest.identity, observation_binding=asdict(manifest.observation_binding),
        producer_sha256=file_sha256(normalization.__file__), synthetic_fixture=manifest.synthetic_fixture,
        scientific_updates=0, query_outcomes_used=False, scientific_launch_authority=False)
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("Normalizer producer, fold, representation, role or scope differs")
    sampler = record.get("sampler_binding", {})
    context = SamplingContext(study_id, manifest.fold.training_donors, "F0-V0", study_seed, 0, 0, "scales")
    required = dict(context=asdict(context), specification=asdict(SamplerSpecification(maximum_laws=1)),
        source_sha256=file_sha256(sampling.__file__), geometry=geometry.backend_identity,
        unused_noise_omitted=True, unused_full_source_diagnostics_omitted=True)
    if any(key not in sampler or _hash(sampler[key]) != _hash(value) for key, value in required.items()):
        raise ValueError("Normalizer source draw/geometry binding differs")
    particles, block = sampler.get("particles"), sampler.get("pairwise_block")
    if (type(particles) is not int or type(block) is not int or block < 1
            or (manifest.synthetic_fixture and not (1 <= particles <= 128 and block <= 128))
            or (not manifest.synthetic_fixture and particles != 512)):
        raise ValueError("Normalizer numerical dimensions differ from admitted scope")
    rebuilt = normalization.reduce_normalizers(cache, record.get("per_law", ()), record.get("per_donor", ()),
        sampler_binding=sampler, producer_sha256=expected["producer_sha256"])
    if _hash(rebuilt) != _hash(record):
        raise ValueError("Saved normalizer reductions or complete support do not reconcile")
    binding = FittingScaleBinding(manifest.fold.training_donors, manifest.rules_sha256,
        manifest.observation_binding.artifact_sha256, manifest.input_sha256,
        record["support_inventory_sha256"], artifact.file_sha256, expected["producer_sha256"],
        _hash(sampler), record["training_values"])
    return binding, MappingProxyType(dict(record["selection_values"])), record["content_sha256"]


def _native_bindings(index, cache, admission):
    if not isinstance(index, NativeInputIndex) or not isinstance(admission, ObservationArtifactAdmission):
        raise TypeError("Verified native metadata index and selected observation admission required")
    manifest = cache.manifest
    fold = manifest.fold
    if (index.binding.synthetic_fixture != manifest.synthetic_fixture
            or admission.specification.synthetic_fixture != manifest.synthetic_fixture
            or admission.fold.fold_id != fold.fold_id
            or admission.fold.training_donors != fold.training_donors
            or index.rna_axis_sha256 != manifest.feature_axis_sha256
            or admission.fold.feature_axis_sha256 != manifest.feature_axis_sha256
            or index.binding.rna_features != manifest.features):
        raise ValueError("Native metadata, cache and observation fold/axis/scope differ")
    catalog = tuple(CatalogGuide(row.guide_id, row.target_id, bool(row.is_control))
                    for row in index.catalog().itertuples(index=False))
    if catalog != manifest.catalog:
        raise ValueError("Native complete catalog differs from cache order/identity")
    roles = {role: index.access(fold, role) for role in
             ("representation_fit", "dynamics_source", "dynamics_supervision")}
    if (roles["representation_fit"].identity() != admission.trajectory_identity["input_sha256"]
            or roles["dynamics_source"].identity() != manifest.source_access_sha256
            or roles["dynamics_supervision"].identity() != manifest.supervision_access_sha256):
        raise ValueError("Native fitting capabilities differ from admitted observation/cache roles")
    for ref in manifest.populations:
        if (index.source_roles.get(ref.source_id) != (ref.donor, ref.condition)
                or native_source_input_identity(index, ref.source_id) != ref.source_input_sha256):
            raise ValueError("Native source inventory differs from compiled source binding")
    return {role: access.identity() for role, access in roles.items()}


class PreparedFitting:
    """Prepared components with no optimizer until an exact lease is supplied.

    Keep the supplied MappedFoldCache context open through the owner's lifetime.
    Provider boundary checks then include its descriptor/path lease checks.
    """
    def __init__(self, *, cache, identity, arguments, scale_binding, selection_scales,
                 support_factors, initialization_sha256):
        self.cache = cache
        self.identity = json.loads(json.dumps(identity, sort_keys=True, allow_nan=False))
        self._identity_sha256 = _hash(self.identity)
        self.arguments = MappingProxyType(dict(arguments))
        self.scale_binding, self.selection_scales = scale_binding, selection_scales
        self.support_factors = support_factors
        self.initialization_sha256 = initialization_sha256
        self._count_inputs_sha256 = _count_inputs_identity(arguments["blocks"], arguments["cache"])
        self._opened = False

    def open_owner(self, lease, *, resume=None, retention_policy=None, resource_guard=None):
        self.cache.assert_unchanged()
        if (self._opened or _hash(self.identity) != self._identity_sha256
                or lease.identity != self.identity):
            raise ValueError("Exact unchanged fold/seed/source owner lease required once")
        if forecast_model_sources(self.identity["source_sha256"]) != self.identity["forecast_model_sources"]:
            raise ValueError("Model equation source changed after fitting preparation")
        args = self.arguments
        self.cache.verify_decoder(args["decoder"], args["observation_binding"])
        if _model_initialization_identity(args["model"]) != self.initialization_sha256:
            raise ValueError("Fresh model changed between preparation and owner opening")
        if (_count_inputs_identity(args["blocks"], args["cache"]) != self._count_inputs_sha256
                or args["provider"].state_dict()["position"] != 0):
            raise ValueError("Prepared count inputs or provider position changed")
        current = owner_binding(model=args["model"], protocol=args["protocol"], execution=args["execution"],
            geometry=args["geometry"], inventory=args["provider"].inventory, schedule=args["schedule"],
            decoder=args["decoder"], observation_binding=args["observation_binding"],
            maximum_checkpoint_bytes=args["maximum_checkpoint_bytes"])
        if current != self.identity["training_owner"]:
            raise ValueError("Prepared owner components changed before lease consumption")
        owner = TrainingOwner(lease=lease, resume=resume, retention_policy=retention_policy,
                              resource_guard=resource_guard, **args)
        self._opened = True
        return owner


def _model_initialization_identity(model):
    digest = hashlib.sha256()
    digest.update(json.dumps(asdict(model.config), sort_keys=True).encode())
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().to("cpu").contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()).cast("B"))
    return digest.hexdigest()


def _count_inputs_identity(blocks, cache):
    digest = hashlib.sha256()
    def update(value):
        if isinstance(value, torch.Tensor):
            tensor = value.detach().to("cpu").contiguous()
            digest.update(json.dumps([str(tensor.dtype), list(tensor.shape)]).encode())
            if tensor.numel():
                digest.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()).cast("B"))
        elif isinstance(value, dict):
            for key in sorted(value):
                digest.update(json.dumps(key).encode()); update(value[key])
        elif isinstance(value, (list, tuple)):
            digest.update(str(len(value)).encode())
            for item in value:
                update(item)
        else:
            digest.update(json.dumps(value, sort_keys=True, allow_nan=False).encode())
    update([dict(donor_id=b.donor_id, catalog_id=b.catalog_id, log_source=b.log_source,
        endpoint_counts=b.endpoint_counts, modeled_indices=b.modeled_indices, cache_indices=b.cache_indices)
        for b in blocks])
    update(cache.state_dict())
    return digest.hexdigest()


def prepare_fitting_components(*, observation_admission, cache, native_index, normalizers,
        rules, configuration_id, model_seed, study_id, study_seed, trajectory_id,
        source_sha256, schedule, execution, geometry, device="cpu", pin_memory=False,
        maximum_checkpoint_bytes=None):
    """Load selected observation and prepare a fully bound fresh model/provider.

    No arbitrary model, decoder, target table or scale vector may be injected.
    Healthy checkpoint restoration remains TrainingOwner's explicit resume path.
    """
    from .cache_storage import MappedFoldCache
    if not isinstance(cache, MappedFoldCache) or not isinstance(rules, ObjectiveRules):
        raise TypeError("Live verified mmap cache and frozen objective rules required")
    cache.assert_unchanged()
    destination = torch.device(device)
    if cache.manifest.synthetic_fixture != (schedule.execution_scope in ("synthetic_cpu_only", "synthetic_cuda_qualification")):
        raise ValueError("Factory input and owner execution scopes differ")
    if cache.manifest.synthetic_fixture and schedule.execution_scope == "synthetic_cuda_qualification":
        from .main_qualification_worker_plan import worker_schedule
        if schedule != worker_schedule() or destination != torch.device("cuda:0") or torch.get_num_threads() != 8:
            raise ValueError("Exact compact CUDA worker fixture and eight threads required")
    elif cache.manifest.synthetic_fixture and (destination.type != "cpu" or torch.get_num_threads() > 2):
        raise ValueError("Fabricated factory remains CPU-only with at most two threads")
    if not cache.manifest.synthetic_fixture and (destination.type != "cuda" or destination.index is None):
        raise ValueError("Scientific fitting preparation requires an explicitly admitted indexed CUDA device")
    if type(model_seed) is not int or model_seed not in SEEDS:
        raise ValueError("One exact registered model/provider/lease seed required")
    if (maximum_checkpoint_bytes is not None and
            (type(maximum_checkpoint_bytes) is not int or maximum_checkpoint_bytes < 1)):
        raise ValueError("Positive integer checkpoint payload cap required")
    if not cache.manifest.synthetic_fixture and maximum_checkpoint_bytes is None:
        raise ValueError("Scientific fitting requires an admitted checkpoint payload cap")
    _sha(source_sha256)
    equation_sources = forecast_model_sources(source_sha256)
    roles = _native_bindings(native_index, cache, observation_admission)
    if rules.identity != cache.manifest.rules_sha256:
        raise ValueError("Fitting rules differ from compiled cache")
    config = get_configuration(configuration_id)
    decoder, binding = load_observation(observation_admission)
    cache.verify_decoder(decoder, binding)  # Mandatory even for F0/F1.
    scales, selection, content_sha = bind_normalizers(cache, normalizers,
        study_id=study_id, study_seed=study_seed, geometry=geometry)
    records = tuple(SourceSupportRecord(donor, "Stim8hr", guide.guide_id, guide.target_id,
        int(cache.populations[(donor, "Stim8hr")].positive_counts[index]), guide.is_control)
        for donor in cache.manifest.fold.training_donors for index, guide in enumerate(cache.manifest.catalog))
    support = source_support_factors(records, fitting_donors=cache.manifest.fold.training_donors,
        control_target_ids=cache.control_target_ids, source_inventory_sha256=cache.manifest.source_access_sha256,
        rules=rules)
    private = dict(target_ids=support.target_ids, support_factors=support.factors) if config.target_residual else {}
    model = build_model(configuration_id, seed=model_seed, control_ids=support.control_target_ids, **private)
    model.to(destination)
    decoder.to(destination)
    cache.verify_decoder(decoder, binding)
    provider = FoldFittingProvider(cache, rules=rules, configuration_id=configuration_id,
        model_seed=model_seed, study_id=study_id, study_seed=study_seed, schedule=schedule,
        device=destination, pin_memory=pin_memory)
    blocks, count_cache = provider.count_inputs()
    protocol = build_objective_protocol(configuration_id, rules=rules, scales=scales,
        fitting_donors=cache.manifest.fold.training_donors,
        observation_artifact_sha256=binding.artifact_sha256, fitting_input_sha256=cache.manifest.input_sha256)
    arguments = dict(model=model, provider=provider, protocol=protocol, execution=execution, geometry=geometry,
        blocks=blocks, cache=count_cache, schedule=schedule, decoder=decoder, observation_binding=binding,
        maximum_checkpoint_bytes=maximum_checkpoint_bytes)
    bound = owner_binding(model=model, protocol=protocol, execution=execution, geometry=geometry,
        inventory=provider.inventory, schedule=schedule, decoder=decoder, observation_binding=binding,
        maximum_checkpoint_bytes=maximum_checkpoint_bytes)
    initial = _model_initialization_identity(model)
    identity = dict(trajectory_id=trajectory_id, fold_id=cache.manifest.fold.fold_id, seed=model_seed,
        input_sha256=provider.inventory.input_sha256, configuration_sha256=bound["configuration_sha256"],
        source_sha256=source_sha256, training_owner=bound,
        forecast_model_sources=equation_sources,
        fitting_factory_sha256=file_sha256(__file__), native_metadata_identity=native_index.identity,
        native_roles=roles, cache_publication_identity=cache.publication_identity,
        normalizer_file_sha256=normalizers.file_sha256, normalizer_content_sha256=content_sha,
        target_support_sha256=support.identity, fresh_initialization_sha256=initial)
    cache.assert_unchanged()
    if forecast_model_sources(source_sha256) != equation_sources:
        raise ValueError("Model equation source changed during fitting preparation")
    return PreparedFitting(cache=cache, identity=identity, arguments=arguments,
        scale_binding=scales, selection_scales=selection, support_factors=support,
        initialization_sha256=initial)
