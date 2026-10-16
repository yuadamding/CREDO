"""Read a healthy current main-fit milestone into a frozen CPU predictor.

Only the exact admitted checkpoint, its COMPLETE receipt, the current
stage-complete owner journal and two producer-bound model sources are opened.
A later/failed owner journal cannot
certify an earlier stage here. No discovery, optimizer restoration, RNG restore,
checkpoint selection, query access, scheduler operation or update is performed.

The producer checkpoint is monolithic: restricted deserialization reads its
tensor payload, including saved optimizer tensors, within an explicit archive
bound. Only model weights are installed; no optimizer is created or loaded.
An inference-only producer export would be needed to avoid reading those bytes.
File verification is not worker/lifecycle admission or predictive validation.
"""

from contextlib import ExitStack
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import zipfile

import torch

from . import checkpoints, configuration as configuration_implementation, models as model_implementation
from .configuration import SEEDS, STAGE_UPDATES, get_configuration
from .fitting_provider import _hash, _sha
from .forecast import ForecastModelBinding, model_state_digest
from .models import build_model
from .native_inputs import FoldInputs
from .objective_registry import TargetSupportFactors
from .observation_loader import STUDY, _path, _unique_object, _verified_file
from .training_owner import ObservationBinding, OwnerSchedule, SCHEMA as OWNER_SCHEMA


@dataclass(frozen=True)
class ForecastFile:
    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self):
        _path(self.path)
        _sha(self.sha256)
        if type(self.size_bytes) is not int or self.size_bytes < 1:
            raise ValueError("Exact positive forecast-input file size required")


@dataclass(frozen=True)
class ForecastArtifactAdmission:
    checkpoint: ForecastFile
    completion: ForecastFile
    owner_status: ForecastFile
    trajectory_identity: dict
    fold: FoldInputs
    observation_binding: ObservationBinding
    target_support: TargetSupportFactors
    configuration_id: str
    model_seed: int
    accepted_updates: int
    schedule: OwnerSchedule
    maximum_checkpoint_bytes: int
    maximum_metadata_bytes: int

    def __post_init__(self):
        if (not all(isinstance(v, ForecastFile) for v in (self.checkpoint, self.completion, self.owner_status))
                or not isinstance(self.fold, FoldInputs) or not isinstance(self.observation_binding, ObservationBinding)
                or not isinstance(self.target_support, TargetSupportFactors) or not isinstance(self.schedule, OwnerSchedule)
                or not isinstance(self.trajectory_identity, dict)):
            raise TypeError("Exact main-checkpoint, fold, support and observation admissions required")
        for v in (self.checkpoint, self.completion, self.owner_status):
            v.__post_init__()
        self.target_support.__post_init__()
        self.schedule.__post_init__()
        if (type(self.model_seed) is not int or self.model_seed not in SEEDS
                or type(self.accepted_updates) is not int or self.accepted_updates not in self.schedule.stages
                or self.schedule.phase_after(self.accepted_updates) != "joint"
                or any(type(v) is not int or v < 1 for v in
                       (self.maximum_checkpoint_bytes, self.maximum_metadata_bytes))):
            raise ValueError("Registered seed, completed joint milestone and explicit byte limits required")
        if self.schedule.execution_scope == "scientific" and self.accepted_updates not in STAGE_UPDATES:
            raise ValueError("Only registered scientific allocation milestones can forecast")
        if self.schedule.execution_scope == "synthetic_cpu_only" and (
                self.maximum_checkpoint_bytes > 64 * 1024**2 or self.maximum_metadata_bytes > 4 * 1024**2
                or len(self.target_support.target_ids) > 64):
            raise ValueError("Fabricated forecast loader exceeds bounded CPU input scope")
        if len(self.target_support.target_ids) > 26504:
            raise ValueError("Target support cannot exceed the complete guide catalog")
        if (self.checkpoint.size_bytes > self.maximum_checkpoint_bytes
                or max(self.completion.size_bytes, self.owner_status.size_bytes) > self.maximum_metadata_bytes):
            raise ValueError("Admitted checkpoint or metadata exceeds its explicit byte limit")
        directory = _path(self.checkpoint.path).parent
        root = directory.parent.parent
        if (STUDY not in root.parts or Path(self.checkpoint.path).name != "checkpoint.pt"
                or directory != root / "checkpoints" / f"update-{self.accepted_updates:09d}"
                or _path(self.completion.path) != directory / "COMPLETE.json"
                or _path(self.owner_status.path) != root / "TRAINING_OWNER_STATE.json"):
            raise ValueError("Exact current-study main checkpoint and owner journal paths required")
        identity = self.trajectory_identity
        required = checkpoints.IDENTITY_KEYS | {"training_owner", "target_support_sha256",
            "fitting_factory_sha256", "native_metadata_identity", "native_roles", "cache_publication_identity",
            "normalizer_file_sha256", "normalizer_content_sha256", "fresh_initialization_sha256",
            "forecast_model_sources"}
        if not required <= set(identity):
            raise ValueError("Complete factory-bound main trajectory identity required; no historical/observation weights")
        checkpoints._identity(identity)
        configuration = get_configuration(self.configuration_id)
        if (identity["fold_id"] != self.fold.fold_id or identity["seed"] != self.model_seed
                or identity["configuration_sha256"] != _hash(asdict(configuration))
                or self.target_support.fitting_donors != self.fold.training_donors
                or identity["target_support_sha256"] != self.target_support.identity):
            raise ValueError("Admitted fold, seed, configuration or fitting target support differs")
        for key in required - checkpoints.IDENTITY_KEYS - {"training_owner", "native_roles", "forecast_model_sources"}:
            _sha(identity[key])
        _model_source_references(identity, self.maximum_metadata_bytes)
        owner = identity["training_owner"]
        expected_owner_keys = {"schema", "protocol_sha256", "objective_source_sha256", "owner_source_sha256",
            "configuration_sha256", "model_target_order_sha256", "model_support_sha256", "fitting_inventory_sha256",
            "execution_sha256", "geometry_sha256", "schedule_sha256", "observation_binding",
            "maximum_checkpoint_bytes"}
        if (not isinstance(owner, dict) or set(owner) != expected_owner_keys or owner["schema"] != OWNER_SCHEMA
                or owner["configuration_sha256"] != identity["configuration_sha256"]
                or owner["schedule_sha256"] != _hash(asdict(self.schedule))
                or owner["observation_binding"] != asdict(self.observation_binding)):
            raise ValueError("Owner scientific protocol/schedule/observation binding differs")
        producer_cap = owner["maximum_checkpoint_bytes"]
        if ((producer_cap is not None and (type(producer_cap) is not int or producer_cap < 1
                                           or self.checkpoint.size_bytes > producer_cap))
                or (self.schedule.execution_scope == "scientific" and producer_cap is None)):
            raise ValueError("Owner checkpoint write cap is invalid or exceeded")
        for key in expected_owner_keys - {"schema", "observation_binding", "maximum_checkpoint_bytes"}:
            _sha(owner[key])
        roles = identity["native_roles"]
        if not isinstance(roles, dict) or set(roles) != {"representation_fit", "dynamics_source", "dynamics_supervision"}:
            raise ValueError("Only exact fitting-role identities belong in model admission")
        for value in roles.values():
            _sha(value)
        if (self.target_support.source_inventory_sha256 != roles["dynamics_source"]
                or identity["input_sha256"] != _hash(dict(fold=self.fold.identity,
                    source=roles["dynamics_source"], supervision=roles["dynamics_supervision"]))):
            raise ValueError("Main input and target support do not share the admitted fitting-source roles")


def _model_source_references(identity, maximum_bytes):
    """Read producer-recorded dependencies, never infer missing provenance.

    Paths come from the imported implementation modules, not from checkpoint
    metadata. Only their two equation/configuration dependencies must match the
    fitting bytes; this does not require new forecast publication code to be
    identical to the fitting package. Immutable worker source admission remains
    responsible for binding these imported modules to the executed package.
    """
    sources = identity.get("forecast_model_sources")
    modules = {"frontier/models.py": model_implementation,
               "frontier/configuration.py": configuration_implementation}
    if (not isinstance(sources, dict)
            or set(sources) != {"schema", "producer_source_sha256", "files"}
            or sources["schema"] != "credo.frontier.forecast-model-sources.v1"
            or sources["producer_source_sha256"] != identity["source_sha256"]
            or not isinstance(sources["files"], dict) or set(sources["files"]) != set(modules)):
        raise ValueError("Exact producer-bound forecast model source inventory required")
    references = []
    for name, module in modules.items():
        item = sources["files"][name]
        if (not isinstance(item, dict) or set(item) != {"sha256", "size_bytes"}
                or type(item["size_bytes"]) is not int or not 0 < item["size_bytes"] <= maximum_bytes):
            raise ValueError("Exact bounded forecast model source hash/size required: " + name)
        references.append(ForecastFile(str(Path(module.__file__).absolute()), item["sha256"], item["size_bytes"]))
    return tuple(references)


def _rehash_model_source(stream, reference):
    """Recheck the same verified descriptor before returning a loaded model."""
    stream.seek(0)
    digest = hashlib.sha256()
    while block := stream.read(1024 * 1024):
        digest.update(block)
    if digest.hexdigest() != reference.sha256:
        raise ValueError("Forecast model source SHA-256 changed during loading: " + reference.path)


def _json(stream):
    result = json.load(stream, object_pairs_hook=_unique_object,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite forecast metadata")))
    if not isinstance(result, dict):
        raise ValueError("Forecast admission metadata must be an object")
    return result


def _completion(admission, receipt, status):
    u, identity = admission.accepted_updates, admission.trajectory_identity
    expected = dict(schema=checkpoints.SCHEMA, identity=identity,
        checkpoint_sha256=admission.checkpoint.sha256, checkpoint_bytes=admission.checkpoint.size_bytes,
        accepted_updates=u, attempted_updates=u, phase="joint",
        execution_scope=admission.schedule.execution_scope, milestone=True)
    producer_cap = identity["training_owner"]["maximum_checkpoint_bytes"]
    if producer_cap is not None:
        expected["maximum_checkpoint_bytes"] = producer_cap
    if (set(receipt) != set(expected) | {"claim_id", "worker_id"}
            or any(receipt.get(k) != v for k, v in expected.items())
            or any(type(receipt[k]) is not str or not receipt[k] for k in ("claim_id", "worker_id"))):
        raise ValueError("Checkpoint completion is not the exact admitted healthy main milestone")
    refreshes = list(range(admission.schedule.state_updates, u, admission.schedule.cache_period))
    required_status = dict(schema=OWNER_SCHEMA, identity=identity, status="stage_complete",
        accepted_updates=u, attempted_updates=u, optimizer_calls=u, healthy_optimizer_updates=u,
        phase="joint", joint_entries=1, refresh_updates=refreshes, failure=None, automatic_retry=False,
        scientific_updates=u if admission.schedule.execution_scope == "scientific" else 0,
        last_checkpoint={"directory": str(Path(admission.checkpoint.path).parent), **receipt})
    if any(status.get(k) != v for k, v in required_status.items()):
        raise ValueError("Current owner is failed, advanced, incomplete or differently bound; no earlier-stage inference")
    return refreshes


def _archive_bound(stream, limit):
    # The declared producer uses uncompressed torch zip storage. Reject other
    # formats/compressed storage before allocating tensors; no archive extraction.
    with zipfile.ZipFile(stream) as archive:
        entries = archive.infolist()
        if (not entries or any(e.compress_type != zipfile.ZIP_STORED or e.file_size < 0 for e in entries)
                or sum(e.file_size for e in entries) > limit):
            raise ValueError("Checkpoint expanded tensor archive exceeds the admitted byte bound")
    stream.seek(0)


def load_forecast_model(admission):
    """Return (frozen CPU FrontierModel, exact ForecastModelBinding).

    CPU loading is deliberate even for an admitted scientific CUDA checkpoint.
    The separately qualified forecast owner controls any later device transfer.
    Global RNG and all training/optimizer state are left untouched.
    """
    if not isinstance(admission, ForecastArtifactAdmission):
        raise TypeError("Explicit main forecast artifact admission required")
    admission.__post_init__()
    if admission.schedule.execution_scope == "synthetic_cpu_only" and torch.get_num_threads() > 2:
        raise ValueError("Fabricated forecast loading requires at most two CPU threads")
    with ExitStack() as stack:
        model_sources = [(reference, stack.enter_context(_verified_file(reference,
            maximum=admission.maximum_metadata_bytes))) for reference in
            _model_source_references(admission.trajectory_identity, admission.maximum_metadata_bytes)]
        receipt_stream = stack.enter_context(_verified_file(admission.completion, maximum=admission.maximum_metadata_bytes))
        status_stream = stack.enter_context(_verified_file(admission.owner_status, maximum=admission.maximum_metadata_bytes))
        receipt, status = _json(receipt_stream), _json(status_stream)
        refreshes = _completion(admission, receipt, status)
        stream = stack.enter_context(_verified_file(admission.checkpoint, maximum=admission.maximum_checkpoint_bytes))
        _archive_bound(stream, admission.maximum_checkpoint_bytes)
        payload = torch.load(stream, map_location="cpu", weights_only=True)
        u, identity, scope = admission.accepted_updates, admission.trajectory_identity, admission.schedule.execution_scope
        expected = dict(schema=checkpoints.SCHEMA, resume_capable=True, identity=identity,
            runtime=checkpoints._runtime(), execution_scope=scope,
            model_class="frontier.models.FrontierModel", optimizer_class="torch.optim.adam.Adam",
            scheduler_class=None, scheduler_state=None, model_device_types=["cuda"] if scope == "scientific" else ["cpu"],
            model_training=True, phase="joint", accepted_updates=u, attempted_updates=u, last_attempt_status="accepted",
            scientific_updates=u if scope == "scientific" else 0,
            disposable_optimizer_calls=0 if scope == "scientific" else u)
        retained_keys = {"model_constructor_metadata", "requires_grad_mask", "model_state", "model_state_metadata",
            "optimizer_parameter_layout", "optimizer_state", "rng", "sampler_state", "count_cache"}
        if (not isinstance(payload, dict) or set(payload) != set(expected) | retained_keys
                or any(payload.get(k) != v for k, v in expected.items())):
            raise ValueError("Not the exact source-bound healthy main checkpoint payload/runtime")
        sampler = payload.get("sampler_state", {})
        provider = sampler.get("provider_state") if isinstance(sampler, dict) else None
        if (not isinstance(sampler, dict) or sampler.get("position") != u
                or sampler.get("binding") != identity["training_owner"] or sampler.get("joint_entries") != 1
                or sampler.get("refresh_updates") != refreshes
                or sampler.get("resource_policy_sha256") != status.get("resource_policy_sha256")
                or not isinstance(provider, dict) or provider.get("position") != u
                or provider.get("schema") != "credo.frontier.fitting-provider-state.v1"):
            raise ValueError("Checkpoint sampler/owner accepted boundary differs")
        if scope == "scientific":
            _sha(sampler.get("resource_policy_sha256"))
        configuration = get_configuration(admission.configuration_id)
        support = admission.target_support
        private = dict(target_ids=support.target_ids, support_factors=support.factors) if configuration.target_residual else {}
        model = build_model(admission.configuration_id, seed=admission.model_seed,
            control_ids=support.control_target_ids, **private)
        if payload.get("model_constructor_metadata") != checkpoints._constructor_metadata(model):
            raise ValueError("Checkpoint model constructor/target routing/support differs")
        owner = identity["training_owner"]
        actual_support = model.support_factors.tolist() if configuration.target_residual else None
        if (owner["model_target_order_sha256"] != _hash([list(model.target_ids), list(model.control_ids)])
                or owner["model_support_sha256"] != _hash(actual_support)):
            raise ValueError("Model target routing differs from admitted owner identity")
        saved, expected_state = payload.get("model_state"), model.state_dict()
        if not isinstance(saved, dict) or set(saved) != set(expected_state):
            raise ValueError("Exact registered model tensor inventory required")
        for name, tensor in expected_state.items():
            value = saved[name]
            if (not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.layout != torch.strided
                    or value.dtype != torch.float32 or value.shape != tensor.shape or not bool(torch.isfinite(value).all())):
                raise ValueError("Model tensor shape/dtype/finite-value identity differs: " + name)
            if name == "support_factors" and not torch.equal(value, tensor):
                raise ValueError("Checkpoint support buffer disagrees with admitted source support")
        mask = payload.get("requires_grad_mask")
        if mask != {name: True for name, _ in model.named_parameters()}:
            raise ValueError("Checkpoint is not the full joint-phase main model")
        model.load_state_dict(saved, strict=True)
        model.set_phase("joint")
        model.eval()
        model.requires_grad_(False)
        binding = ForecastModelBinding(admission.fold, admission.observation_binding,
            admission.configuration_id, admission.checkpoint.sha256, model_state_digest(model),
            support.identity, identity["source_sha256"], admission.model_seed, u)
        for reference, source_stream in model_sources:
            _rehash_model_source(source_stream, reference)
    # All five same-descriptor/path identity checks complete before publication.
    return model, binding
