"""One exact worker input contract -> prepared main fitting components.

This is the input side of the main worker, not a scheduler or launch authority.
It opens accepted fold-native caches and saved normalizers; it never compiles
raw data, recomputes normalizers, selects a model, opens query payloads or makes
an optimizer. The caller must separately admit the worker source/runtime,
resources and task/selection, and enter FittingStage outside this context so
all input leases exit before stage completion is published.
"""

from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, fields
import hashlib
import json

import torch

from .fitting_factory import prepare_fitting_components
from .fitting_preparation import open_prepared_cache
from .fitting_provider import ProviderLimits
from .geometry import GeometryOptions, RaggedSinkhornGeometry
from .native_inputs import (NativeInputIndex, PackageBinding, READER_SOURCE_HASHES,
                            TABLES, _verified)
from .normalizer_publication import load_normalizer_artifact
from .objective_registry import ObjectiveRules, RULES_SHA256
from .observation import ObservationFold
from .observation_loader import (ObservationArtifactAdmission, ObservationFile,
    STUDY, _path, _sha, _unique_object, _verified_file)
from .observation_training import ObservationTrainingSpec
from .planner import TrainingTask, TrajectoryKey
from .training_objective import ObjectiveExecution
from .training_owner import OwnerSchedule


SCHEMA = "credo.frontier.main-fitting-inputs.v1"
MAXIMUM_CONTRACT_BYTES = 4*1024**2
MIB = 1024**2
GIB = 1024**3


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _object(value, names):
    if type(value) is not dict or set(value) != set(names):
        raise ValueError("Exact fitting-input fields required: "+", ".join(sorted(names)))
    return value


def _typed(cls, value, *, tuples=()):
    value = dict(_object(value, (f.name for f in fields(cls))))
    for name in tuples:
        if type(value[name]) is not list:
            raise ValueError("Explicit serialized list required: "+name)
        value[name] = tuple(value[name])
    result = cls(**value)
    return result


def _positive(value, maximum, label):
    if type(value) is not int or not 0 < value <= maximum:
        raise ValueError("Positive bounded "+label+" required")


def _decode(record):
    _object(record, ("schema", "study_id", "study_seed", "source_sha256", "task",
        "native", "observation", "preparation", "normalizers", "rules",
        "schedule", "execution", "geometry", "maximum_checkpoint_bytes"))
    if record["schema"] != SCHEMA:
        raise ValueError("Exact main fitting-input schema required")
    if type(record["study_id"]) is not str or not record["study_id"]:
        raise ValueError("Explicit sampling study identity required")
    if type(record["study_seed"]) is not int or record["study_seed"] < 0:
        raise ValueError("Explicit nonnegative sampling study seed required")
    _sha(record["source_sha256"])
    task_doc = _object(record["task"], ("key", "start_updates", "target_updates", "requesting_selectors"))
    key = _typed(TrajectoryKey, task_doc["key"], tuples=("training_donors",))
    if type(task_doc["requesting_selectors"]) is not list:
        raise ValueError("Serialized requesting selectors required")
    task = TrainingTask(key, task_doc["start_updates"], task_doc["target_updates"],
                        tuple(task_doc["requesting_selectors"]))
    if _encoded(asdict(task)) != _encoded(task_doc):
        raise ValueError("Canonical task donor/selector ordering required; no silent reordering")
    native = _object(record["native"], ("metadata_root", "reader_source_root", "binding", "metadata_index_sha256"))
    for name in ("metadata_root", "reader_source_root"):
        _path(native[name])
    _sha(native["metadata_index_sha256"])
    binding = _typed(PackageBinding, native["binding"])
    if type(binding.synthetic_fixture) is not bool:
        raise ValueError("Explicit Boolean synthetic input scope required")
    observed = _object(record["observation"], (f.name for f in fields(ObservationArtifactAdmission)))
    obs_args = {name: _typed(ObservationFile, observed[name])
                for name in ("artifact", "receipt", "completion", "worker_contract", "protocol")}
    obs_args.update(fold=_typed(ObservationFold, observed["fold"], tuples=("training_donors",)),
        specification=_typed(ObservationTrainingSpec, observed["specification"],
                              tuples=("hidden_dims", "audit_updates")),
        trajectory_identity=observed["trajectory_identity"])
    observation = ObservationArtifactAdmission(**obs_args)
    schedule = _typed(OwnerSchedule, record["schedule"], tuples=("stages",))
    if schedule.execution_scope == "synthetic_cuda_qualification":
        from .main_qualification_worker_plan import worker_schedule
        if schedule != worker_schedule():
            raise ValueError("File-backed CUDA fixture requires the compact worker schedule")
    execution = _typed(ObjectiveExecution, record["execution"])
    geometry = _typed(GeometryOptions, record["geometry"])
    if (type(observation.specification.synthetic_fixture) is not bool
            or observation.specification.synthetic_fixture is not binding.synthetic_fixture
            or (schedule.execution_scope in ("synthetic_cpu_only", "synthetic_cuda_qualification")) is not binding.synthetic_fixture
            or observation.fold.training_donors != key.training_donors
            or observation.specification.features != binding.rna_features
            or observation.specification.latent_dim != 48 or len(schedule.stages) != 3):
        raise ValueError("Same task/observation/native fold,48D representation and three-stage scope required")
    if not binding.synthetic_fixture and record["study_id"] != STUDY:
        raise ValueError("Scientific sampling must use the actual study identity")
    prep = _object(record["preparation"], ("root", "complete_sha256", "output_cap_bytes", "provider_limits"))
    norms = _object(record["normalizers"], ("root", "complete_sha256", "maximum_artifact_bytes", "output_cap_bytes"))
    for item in (prep, norms):
        _path(item["root"]); _sha(item["complete_sha256"])
        _positive(item["output_cap_bytes"], GIB if binding.synthetic_fixture else 384*GIB, "input publication cap")
    if (_path(prep["root"]).is_relative_to(_path(norms["root"]))
            or _path(norms["root"]).is_relative_to(_path(prep["root"]))):
        raise ValueError("Distinct nonnested cache and normalizer publications required")
    provider = _typed(ProviderLimits, prep["provider_limits"])
    host_cap = 8*GIB if binding.synthetic_fixture else 200*GIB
    if max(provider.maximum_host_bytes, provider.maximum_prepared_bytes) > host_cap:
        raise ValueError("Input provider exceeds worker host-memory ceiling")
    _positive(norms["maximum_artifact_bytes"], 8*MIB if binding.synthetic_fixture else 256*MIB,
              "normalizer artifact cap")
    _positive(record["maximum_checkpoint_bytes"], 64*MIB if binding.synthetic_fixture else 384*GIB,
              "checkpoint cap")
    rules = _typed(ObservationFile, record["rules"])
    if rules.sha256 != RULES_SHA256 or rules.size_bytes > MAXIMUM_CONTRACT_BYTES:
        raise ValueError("Exact frozen main objective rules required")
    return dict(task=task, native_binding=binding, observation=observation, schedule=schedule,
                execution=execution, geometry_options=geometry, provider_limits=provider, rules=rules)


@dataclass(frozen=True, init=False)
class FittingInputContract:
    """Immutable bytes; parsed objects are rebuilt rather than shared mutable state."""
    _payload: bytes

    def __init__(self, payload):
        if type(payload) is not bytes or not 0 < len(payload) <= MAXIMUM_CONTRACT_BYTES:
            raise ValueError("Bounded immutable main fitting-input contract bytes required")
        record = json.loads(payload, object_pairs_hook=_unique_object,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite input metadata")))
        _decode(record)
        object.__setattr__(self, "_payload", payload)

    @property
    def identity(self):
        return hashlib.sha256(self._payload).hexdigest()

    def record(self):
        return json.loads(self._payload)

    def components(self):
        return _decode(self.record())


@dataclass(frozen=True)
class OpenFittingInputs:
    prepared: object
    task: TrainingTask
    input_contract_sha256: str
    input_reference: ObservationFile


def _admit_device(contract, device, pin_memory):
    parts = contract.components()
    native, schedule = parts["native_binding"], parts["schedule"]
    destination = torch.device(device)
    if type(pin_memory) is not bool:
        raise ValueError("Explicit pin-memory choice required")
    if native.synthetic_fixture and schedule.execution_scope == "synthetic_cuda_qualification":
        from .main_qualification_worker_plan import worker_schedule
        if schedule != worker_schedule() or destination != torch.device("cuda:0") or torch.get_num_threads() != 8:
            raise ValueError("Exact compact CUDA worker fixture and eight threads required")
    elif native.synthetic_fixture:
        if destination.type != "cpu" or torch.get_num_threads() > 2 or pin_memory:
            raise ValueError("Fabricated worker inputs remain unpinned CPU-only with two threads")
    elif destination != torch.device("cuda:0"):
        raise ValueError("Scientific worker inputs require the separately admitted cuda:0")
    return destination


@contextmanager
def open_fitting_inputs(reference, *, device="cpu", pin_memory=False):
    """Read-only inputs to the actual factory; no optimizer or launch.

    Metadata, reader-source, selected-observation, contract and prepared-cache
    leases cover the caller's use. Normalizer bytes are verified and copied once
    into an immutable artifact, then reconciled by the factory without numerical
    replay. Native count/cell handles are never opened in this consumer.
    """
    if type(reference) is not ObservationFile:
        raise TypeError("Exact path/hash/size contract reference required")
    reference.__post_init__()
    with ExitStack() as stack:
        stream = stack.enter_context(_verified_file(reference, maximum=MAXIMUM_CONTRACT_BYTES))
        contract = FittingInputContract(stream.read())
        record, parts = contract.record(), contract.components()
        destination = _admit_device(contract, device, pin_memory)
        native, binding = record["native"], parts["native_binding"]
        # Lease code and completion/catalog metadata before the index uses them.
        for name, sha in READER_SOURCE_HASHES.items():
            stack.enter_context(_verified(native["reader_source_root"], name, sha, maximum_bytes=2*MIB))
        for name, sha, limit in (("INPUT_DATA_COMPLETE.json", binding.completion_sha256, MIB),
                                 ("SHA256SUMS", binding.inventory_sha256, 16*MIB)):
            stack.enter_context(_verified(native["metadata_root"], name, sha, maximum_bytes=limit))
        index = NativeInputIndex(native["metadata_root"], native["reader_source_root"], binding=binding)
        if index.identity != native["metadata_index_sha256"]:
            raise ValueError("Native metadata differs from the worker input contract")
        for name in TABLES:
            member = "source/"+name
            stack.enter_context(_verified(native["metadata_root"], member, index.inventory[member], maximum_bytes=64*MIB))
        for name in ("artifact", "receipt", "completion", "worker_contract", "protocol"):
            item = getattr(parts["observation"], name)
            stack.enter_context(_verified_file(item, maximum=256*MIB if name == "artifact" else 32*MIB))
        rules_stream = stack.enter_context(_verified_file(parts["rules"], maximum=MAXIMUM_CONTRACT_BYTES))
        rules = ObjectiveRules(rules_stream.read(), parts["rules"].sha256)
        prep = record["preparation"]
        cache = stack.enter_context(open_prepared_cache(prep["root"],
            expected_complete_sha256=prep["complete_sha256"], output_cap_bytes=prep["output_cap_bytes"],
            limits=parts["provider_limits"]))
        if (cache.manifest.fold.training_donors != parts["task"].key.training_donors
                or cache.manifest.fold.fold_id != parts["observation"].fold.fold_id
                or cache.manifest.synthetic_fixture is not binding.synthetic_fixture):
            raise ValueError("Prepared cache has a different admitted fold or execution scope")
        norms = record["normalizers"]
        artifact = load_normalizer_artifact(norms["root"], expected_complete_sha256=norms["complete_sha256"],
            expected_cache_manifest_sha256=cache.manifest.identity,
            maximum_artifact_bytes=norms["maximum_artifact_bytes"], output_cap_bytes=norms["output_cap_bytes"])
        key = parts["task"].key
        prepared = prepare_fitting_components(observation_admission=parts["observation"], cache=cache,
            native_index=index, normalizers=artifact, rules=rules, configuration_id=key.configuration_id,
            model_seed=key.seed, study_id=record["study_id"], study_seed=record["study_seed"],
            trajectory_id=key.storage_identifier, source_sha256=record["source_sha256"],
            schedule=parts["schedule"], execution=parts["execution"],
            geometry=RaggedSinkhornGeometry(options=parts["geometry_options"]),
            device=destination, pin_memory=pin_memory, maximum_checkpoint_bytes=record["maximum_checkpoint_bytes"])
        yield OpenFittingInputs(prepared, parts["task"], contract.identity, reference)
