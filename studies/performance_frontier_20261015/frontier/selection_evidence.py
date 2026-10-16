"""Verified saved scalar assessments -> one allowed inner edge's evidence.

The requested StagePlan/ScoreTask is checked before any filesystem operation.
Only the named seal, its readable plan, the model checkpoint COMPLETE metadata,
and the completed assessment's three scalar files are opened. Particle banks,
checkpoint tensors, raw inputs and outcomes are never reopened. Named hashes
are caller-admitted custody bindings, not an operating-system access boundary
or independent proof of the original experiment. No selection occurs here.

Fabricated scalar fixtures have their own explicit checkpoint scope. Nominal
512/4096 metadata in those fixtures is not a small-particle numerical result
relabeled as a scientific result, nor evidence of training or forecasting.
"""

from contextlib import ExitStack
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from .assessment import scientific_recipe_identity
from .assessment_storage import verify_assessment
from .checkpoints import SCHEMA as CHECKPOINT_SCHEMA, _identity as checkpoint_identity
from .configuration import ALPHA_GRID, SPREAD_GRID, get_configuration
from .fitting_provider import _hash, _sha
from .forecast_storage import ARRAY_NAMES
from .native_inputs import FoldInputs, ForecastExpectation
from .observation_loader import _verified_file
from .planner import ScoreTask, StagePlan
from .selection import CHANNELS, UnavailableSelection
from .selector_contract import ChannelCoverage, EvidenceContext, Recipe, ScoreEvidence


SCHEMA = "credo.frontier.saved-inner-score-evidence.v1"
FABRICATED_SCOPE = "fabricated_scalar_metadata_only"


def _encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()


def _plain(value):
    return json.loads(_encode(value))


def _lexical_path(value):
    # No exists/is_symlink/stat here: eligibility must precede ANY file access.
    if not isinstance(value, str):
        raise ValueError("Explicit absolute metadata path required")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or path.name in ("", ".", ".."):
        raise ValueError("Absolute nontraversing metadata path required")
    return path


def _relative(value):
    if (not isinstance(value, str) or not value or Path(value).is_absolute()
            or any(x in ("", ".", "..") for x in value.split("/"))):
        raise ValueError("Canonical relative artifact path required")
    return value


def _positive(value):
    if type(value) is not int or value < 1:
        raise ValueError("Positive exact byte/count bound required")
    return value


def _parse(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate metadata key")
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite metadata")))
    if not isinstance(value, dict):
        raise ValueError("Metadata object required")
    return value


@dataclass(frozen=True)
class SelectionMetadataFile:
    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self):
        _lexical_path(self.path); _sha(self.sha256); _positive(self.size_bytes)


@dataclass(frozen=True)
class SelectionEvidenceAdmission:
    assessment_root: str
    assessment_complete_sha256: str
    maximum_assessment_bytes: int
    seal: SelectionMetadataFile
    seal_plan: SelectionMetadataFile
    checkpoint_complete: SelectionMetadataFile
    maximum_metadata_bytes: int
    fabricated_scalar_metadata: bool = False

    def __post_init__(self):
        _lexical_path(self.assessment_root); _sha(self.assessment_complete_sha256)
        _positive(self.maximum_assessment_bytes); _positive(self.maximum_metadata_bytes)
        if type(self.fabricated_scalar_metadata) is not bool:
            raise ValueError("Explicit fabricated-scalar scope required")
        refs = (self.seal, self.seal_plan, self.checkpoint_complete)
        if any(type(ref) is not SelectionMetadataFile for ref in refs):
            raise TypeError("Exact three metadata file admissions required")
        for ref in refs: ref.__post_init__()
        if sum(ref.size_bytes for ref in refs) > self.maximum_metadata_bytes:
            raise ValueError("Aggregate metadata byte allowance exceeded")
        if (self.seal_plan.path != self.seal.path+".plan.json"
                or Path(self.checkpoint_complete.path).name != "COMPLETE.json"
                or len({ref.path for ref in refs}) != 3):
            raise ValueError("Exact seal/plan and separate checkpoint COMPLETE paths required")


@dataclass(frozen=True, init=False)
class BoundAssessmentEvidence:
    evidence: tuple[ScoreEvidence, ...]
    _provenance: bytes

    def __init__(self, *args, **kwargs):
        raise TypeError("Use read_inner_assessment; anonymous scores are not verified evidence")

    @property
    def provenance(self):
        return json.loads(self._provenance)

    @property
    def identity(self):
        return hashlib.sha256(self._provenance).hexdigest()


def _allowed(stage_plan, task):
    if type(stage_plan) is not StagePlan or type(task) is not ScoreTask:
        raise TypeError("Frozen StagePlan and exact ScoreTask required")
    stage_plan.__post_init__(); task.__post_init__()
    if (stage_plan.name not in ("screen", "family", "finalist") or task.phase != "inner"
            or task not in stage_plan.scores_for(task.selector) or task.key.seed != 0
            or task.updates != {"screen": 1024, "family": 3072, "finalist": 9216}[stage_plan.name]):
        raise ValueError("Only this selector's exact eligible inner task may open saved scores")
    return 4096 if stage_plan.name == "finalist" else 512


def _checkpoint_prefix(admission, task, qualification_prefix):
    """Keep an actual compact checkpoint distinct from a nominal study stage.

    Only continuation custody has an edge_id. Ordinary assessment admissions
    cannot opt into this path, and unrelated metadata group members retain
    their explicitly fabricated nominal prefixes.
    """
    if qualification_prefix is None:
        return task.updates
    if (type(qualification_prefix) is not int or qualification_prefix != 2
            or admission.fabricated_scalar_metadata is not True
            or not getattr(admission, "edge_id", None)):
        raise ValueError("Only explicit two-update fabricated continuation custody is admitted")
    if task.identifier != admission.edge_id:
        return task.updates
    if (task.key.training_donors != ("D3", "D4") or task.key.configuration_id != "F0-V2"
            or task.key.seed != 0 or task.updates != 1024):
        raise ValueError("Compact checkpoint prefix is limited to the fixed worker-entry case")
    return qualification_prefix


def _seal_metadata(admission, seal, plan, stage_plan, task, *, qualification_prefix=None):
    if (set(seal) != {"schema", "status", "bindings", "forecasts"}
            or seal["schema"] != "credo.frontier.forecast-seal.v1" or seal["status"] != "complete"):
        raise ValueError("Exact completed forecast-seal metadata required")
    plan_keys = {"schema", "fold", "query_donor", "query_access_sha256", "expectation",
        "configuration_inventory", "entries", "inventory_completeness_is_controller_obligation",
        "whole_study_seal", "scientific_launch_authority"}
    fold = FoldInputs("subset-"+"-".join(task.key.training_donors), task.key.training_donors)
    if (set(plan) != plan_keys or plan["schema"] != "credo.frontier.forecast-seal-plan.v1"
            or plan["fold"] != _plain(asdict(fold)) or plan["query_donor"] != task.validation_donor
            or plan["whole_study_seal"] is not False or plan["scientific_launch_authority"] is not False
            or plan["inventory_completeness_is_controller_obligation"] is not True):
        raise ValueError("Saved seal plan is not the exact allowed fold/query")
    inventory = plan["configuration_inventory"]
    if (not isinstance(inventory, dict) or set(inventory) != {"schema", "entries"}
            or inventory["schema"] != "credo.frontier.forecast-seal-configurations.v1"):
        raise ValueError("Inner plan cannot include ensemble or unknown configuration inventory")
    entries, configurations, forecasts = plan["entries"], inventory["entries"], seal["forecasts"]
    expected_tasks = {t.key.configuration_id: t for t in stage_plan.scores_for(task.selector)
        if t.key.training_donors == task.key.training_donors and t.validation_donor == task.validation_donor}
    if (not isinstance(entries, list) or not entries or not isinstance(configurations, list)
            or not isinstance(forecasts, list) or len(entries) != len(configurations) or len(entries) != len(forecasts)
            or len(entries) != len(expected_tasks)):
        raise ValueError("Complete corresponding metadata inventories required")
    ids = [entry.get("forecast_id") for entry in entries]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Unique nonempty forecast identifiers required")
    models = tuple(dict.fromkeys(e["model_binding"]["model_artifact_sha256"] for e in entries))
    first_binding = entries[0]["model_binding"]
    expectation = ForecastExpectation(fold.identity, plan["query_access_sha256"], models,
        _hash(inventory), first_binding["numerical_source_sha256"],
        first_binding["observation_binding"]["artifact_sha256"], tuple(ids))
    if plan["expectation"] != expectation.record() or seal["bindings"] != expectation.record():
        raise ValueError("Seal, plan and ordered inventory bindings disagree")
    selected = []
    seen_configurations = set()
    paths = set()
    plan_path = Path(admission.seal_plan.path).name
    for position, (entry, config, row) in enumerate(zip(entries, configurations, forecasts)):
        entry_keys = {"forecast_id", "bank_relative_root", "spec_sha256", "complete_sha256", "manifest_sha256",
            "model_binding", "source_identity", "source_reference", "catalog_sha256", "feature_axis_sha256"}
        config_keys = {"forecast_id", "configuration", "recipe", "prefixes", "single_seed_prefixes",
            "ensemble_component_totals", "ensemble_component_prefixes", "context", "derivation_manifest"}
        if (set(entry) != entry_keys or set(config) != config_keys or set(row) != {"forecast_id", "artifacts"}
                or config["forecast_id"] != entry["forecast_id"] or row["forecast_id"] != entry["forecast_id"]):
            raise ValueError("Exact corresponding seal entry schemas required")
        bank = _relative(entry["bank_relative_root"])
        for key in ("spec_sha256", "complete_sha256", "manifest_sha256", "source_identity", "catalog_sha256", "feature_axis_sha256"):
            _sha(entry[key])
        binding, reference = entry["model_binding"], entry["source_reference"]
        configuration_id = binding["configuration_id"]
        item_task = expected_tasks.get(configuration_id)
        if item_task is None or configuration_id in seen_configurations:
            raise ValueError("Complete eligible group requires each allowed configuration exactly once")
        seen_configurations.add(configuration_id)
        if (binding["fold"] != plan["fold"] or reference["donor"] != task.validation_donor
                or reference["condition"] != "Stim8hr" or reference["source_access_sha256"] != plan["query_access_sha256"]
                or binding["numerical_source_sha256"] != first_binding["numerical_source_sha256"]
                or binding["observation_binding"] != first_binding["observation_binding"]
                or any(entry[k] != entries[0][k] for k in
                    ("source_identity", "source_reference", "catalog_sha256", "feature_axis_sha256"))):
            raise ValueError("A sealed entry names a foreign fold/query/representation")
        artifacts = row["artifacts"]
        names = [bank+"/"+name for name in ("COMPLETE.json", "MANIFEST.json", *(n+".npy" for n in ARRAY_NAMES))]
        if position == 0: names.append(plan_path)
        if not isinstance(artifacts, list) or len(artifacts) != len(names):
            raise ValueError("Exact six bank references and one plan reference required")
        for artifact, name in zip(artifacts, names):
            if (set(artifact) != {"relative_uri", "sha256", "size_bytes"}
                    or _relative(artifact["relative_uri"]) != name or name in paths):
                raise ValueError("Duplicate or unexpected sealed artifact path")
            paths.add(name); _sha(artifact["sha256"]); _positive(artifact["size_bytes"])
        if (artifacts[0]["sha256"] != entry["complete_sha256"]
                or artifacts[1]["sha256"] != entry["manifest_sha256"]):
            raise ValueError("Parent completion/manifest differs from seal")
        if position == 0 and artifacts[-1] != dict(relative_uri=plan_path,
                sha256=admission.seal_plan.sha256, size_bytes=admission.seal_plan.size_bytes):
            raise ValueError("Readable plan artifact is not the exact sealed metadata")
        parent = dict(root=str(Path(admission.seal.path).parent/bank), complete_sha256=entry["complete_sha256"],
            manifest_sha256=entry["manifest_sha256"], spec_sha256=entry["spec_sha256"],
            total_bytes=sum(a["size_bytes"] for a in artifacts[:6]))
        derivation = config["derivation_manifest"]
        context = config["context"]
        if (not isinstance(context, dict)
                or any(context.get(k) != v for k, v in dict(configuration_id=configuration_id,
                    training_donors=list(item_task.key.training_donors), model_seed=0,
                    accepted_update=_checkpoint_prefix(admission, item_task, qualification_prefix), purpose="forecast").items())):
            raise ValueError("Saved forecast context is not this exact inner task")
        role = dict(phase="inner", stage=stage_plan.name, selector=task.selector, task=_plain(asdict(item_task)),
            stage_plan_sha256=_hash(asdict(stage_plan)), eligible_configurations=list(stage_plan.eligible_for(task.selector)),
            required_recipe_count=24, outer_outcome_selection_allowed=False)
        if (not isinstance(derivation, dict) or derivation.get("schema") != "credo.frontier.calibration-derivation.v1"
                or derivation.get("status") != "source_only_inventory_bound" or derivation.get("role") != role
                or derivation.get("parent_publication") != parent or derivation.get("model_binding") != binding
                or derivation.get("context") != context
                or derivation.get("source_identity") != entry["source_identity"]
                or derivation.get("catalog_sha256") != entry["catalog_sha256"]
                or derivation.get("feature_axis_sha256") != entry["feature_axis_sha256"]
                or binding["model_seed"] != 0 or binding["accepted_updates"] != _checkpoint_prefix(admission, item_task, qualification_prefix)
                or derivation.get("synthetic_fixture") is not admission.fabricated_scalar_metadata):
            raise ValueError("Saved calibration role is not this exact allowed task")
        if config["configuration"] != _plain(asdict(get_configuration(configuration_id))):
            raise ValueError("Saved configuration differs from registered candidate")
        prefix = 4096 if stage_plan.name == "finalist" else 512
        recipes = [_plain(asdict(Recipe(configuration_id, a, s))) for a in ALPHA_GRID for s in SPREAD_GRID]
        if (derivation.get("recipes") != recipes or derivation.get("permitted_prefixes") != [prefix]
                or config["single_seed_prefixes"] != [prefix] or config["prefixes"] != [prefix]
                or config["ensemble_component_totals"] or config["ensemble_component_prefixes"]
                or config["recipe"] != _plain(asdict(Recipe(configuration_id)))):
            raise ValueError("Every group member requires the full calibration grid and registered scoring prefix")
        if configuration_id == task.key.configuration_id:
            selected.append((entry, config, parent))
    if seen_configurations != set(expected_tasks):
        raise ValueError("Complete eligible fold/query group required, not successful-candidate subset")
    if len(selected) != 1:
        raise ValueError("Exactly one admitted parent for the requested task required")
    return selected[0], fold


def _checkpoint(record, admission, task, fold, binding, *, qualification_prefix=None):
    keys = {"schema", "identity", "checkpoint_sha256", "checkpoint_bytes", "accepted_updates",
        "attempted_updates", "phase", "execution_scope", "milestone", "claim_id", "worker_id"}
    if not isinstance(record,dict):
        raise ValueError("Checkpoint COMPLETE must be an exact metadata object")
    identity=checkpoint_identity(record.get("identity"))
    owner=identity.get("training_owner")
    if not isinstance(owner,dict):
        raise ValueError("Checkpoint must retain its actual fitting-owner binding")
    producer_cap=owner.get("maximum_checkpoint_bytes")
    if producer_cap is not None:
        if (type(producer_cap) is not int or producer_cap<1
                or type(record.get("maximum_checkpoint_bytes")) is not int
                or record["maximum_checkpoint_bytes"]!=producer_cap):
            raise ValueError("Checkpoint payload bound differs from the producer owner")
        keys.add("maximum_checkpoint_bytes")
    elif not admission.fabricated_scalar_metadata:
        raise ValueError("Scientific checkpoint requires its producer-bound payload limit")
    scope = FABRICATED_SCOPE if admission.fabricated_scalar_metadata else "scientific"
    prefix = _checkpoint_prefix(admission, task, qualification_prefix)
    if qualification_prefix is not None:
        scope = "synthetic_cuda_qualification"
    if (set(record) != keys or record["schema"] != CHECKPOINT_SCHEMA or record["phase"] != "joint"
            or type(record["accepted_updates"]) is not int or record["accepted_updates"] != prefix
            or type(record["attempted_updates"]) is not int or record["attempted_updates"] != prefix
            or record["milestone"] is not True or record["execution_scope"] != scope
            or record["checkpoint_sha256"] != binding["model_artifact_sha256"]):
        raise ValueError("Checkpoint COMPLETE must identify this healthy exact stage, not a failed attempt")
    _positive(record["checkpoint_bytes"])
    if producer_cap is not None and record["checkpoint_bytes"]>producer_cap:
        raise ValueError("Checkpoint payload exceeds its producer-bound limit")
    for name in ("claim_id", "worker_id"):
        if not isinstance(record[name], str) or not record[name]:
            raise ValueError("Exact checkpoint owner identity required")
    if (identity["fold_id"] != fold.fold_id or identity["seed"] != 0
            or identity["configuration_sha256"] != _hash(asdict(get_configuration(task.key.configuration_id)))
            or identity["source_sha256"] != binding["numerical_source_sha256"]):
        raise ValueError("Checkpoint fitting fold/configuration/source differs")
    roles = identity.get("native_roles")
    if (not isinstance(roles, dict) or set(roles) != {"representation_fit", "dynamics_source", "dynamics_supervision"}
            or not isinstance(owner, dict) or owner.get("observation_binding") != binding["observation_binding"]):
        raise ValueError("Checkpoint must retain actual fitting-role and observation bindings")
    for value in roles.values(): _sha(value)
    if identity["input_sha256"] != _hash(dict(fold=fold.identity,
            source=roles["dynamics_source"], supervision=roles["dynamics_supervision"])):
        raise ValueError("Checkpoint fitting input digest differs from its role bindings")
    return identity


def read_inner_assessment(admission, *, stage_plan, task):
    """Read one explicitly allowed edge; return all 24 recipes, never a winner."""
    prefix = _allowed(stage_plan, task)  # MUST remain before any file operation.
    if type(admission) is not SelectionEvidenceAdmission:
        raise TypeError("Exact saved assessment admission required")
    admission.__post_init__()
    refs = (admission.seal, admission.seal_plan, admission.checkpoint_complete)
    with ExitStack() as stack:
        streams = [stack.enter_context(_verified_file(ref, maximum=admission.maximum_metadata_bytes)) for ref in refs]
        metadata = [_parse(stream.read()) for stream in streams]
        (entry, config, parent), fold = _seal_metadata(admission, metadata[0], metadata[1], stage_plan, task)
        binding = entry["model_binding"]
        checkpoint = _checkpoint(metadata[2], admission, task, fold, binding)
        derivation = config["derivation_manifest"]
        expected_recipes = tuple(Recipe(task.key.configuration_id, a, s) for a in ALPHA_GRID for s in SPREAD_GRID)
        if (derivation.get("recipes") != [_plain(asdict(r)) for r in expected_recipes]
                or prefix not in config["single_seed_prefixes"] or prefix not in derivation.get("permitted_prefixes", ())):
            raise ValueError("Exact full calibration grid and registered scoring prefix required")
        # Only now is the scalar assessment payload admitted for this selector.
        saved = verify_assessment(admission.assessment_root,
            expected_complete_sha256=admission.assessment_complete_sha256, maxbytes=admission.maximum_assessment_bytes)
        summary, publication = saved["summary"], saved["publication"]
        receipt = summary["receipt"]
        derivation_sha = hashlib.sha256(_encode(derivation).rstrip(b"\n")).hexdigest()
        if (receipt["predictor_kind"] != "single_seed" or receipt["parent_publications"] != [parent]
                or receipt["forecast_seal_sha256"] != admission.seal.sha256
                or receipt["derivation_sha256"] != derivation_sha
                or receipt["early_source_identity"] != entry["source_identity"]
                or receipt["observation_binding"] != binding["observation_binding"]
                or receipt["prefix"] != prefix or receipt["recipe_count"] != 24):
            raise ValueError("Saved assessment differs from admitted inner source/model/derivation")
        reductions = {Recipe(**row["recipe"]): row["scores"] for row in summary["scores"]}
        if len(summary["scores"]) != 24 or set(reductions) != set(expected_recipes):
            raise ValueError("All 24 calibrated reductions required exactly once")
        contexts, coverages, evidence = [], [], []
        for recipe in expected_recipes:
            row = reductions[recipe]
            sid = row["identity"]
            expected_identity = dict(fold_id=fold.fold_id, query_donor=task.validation_donor,
                recipe_id=scientific_recipe_identity(recipe), seed=0, particle_prefix=prefix,
                source_sha256=binding["numerical_source_sha256"],
                representation_sha256=binding["observation_binding"]["artifact_sha256"],
                model_sha256=binding["model_artifact_sha256"], catalog_sha256=entry["catalog_sha256"],
                gene_axis_sha256=entry["feature_axis_sha256"], predictor_kind="single_seed")
            if any(sid.get(key) != value for key, value in expected_identity.items()):
                raise ValueError("Saved score identity differs from exact task/model")
            if row.get("selection_available") is not True or row.get("unavailable"):
                raise UnavailableSelection("A required saved channel is unavailable; no recipe or support deletion")
            coverage = {}
            for channel in CHANNELS:
                c = row["coverage"][channel]
                if (type(c["expected_rows"]) is not int or c["expected_rows"] < 1
                        or c["available_rows"] != c["expected_rows"] or c["missing_rows"]
                        or abs(c["available_weight"]-1.) > 1e-10):
                    raise UnavailableSelection("Incomplete required fixed channel support")
                coverage[channel] = ChannelCoverage(_hash(dict(schema="credo.frontier.selection-channel-support.v1",
                    support_sha256=receipt["support_sha256"], channel=channel, fixed_weights=c["fixed_weights"])), c["expected_rows"])
            context = EvidenceContext(stage_plan.name, _hash(receipt["runtime"]),
                sid["representation_sha256"], checkpoint["input_sha256"], receipt["early_source_identity"],
                _hash(dict(source_endpoint_support_sha256=sid["input_sha256"], complete_group_seal_sha256=admission.seal.sha256)),
                _hash(dict(schema="credo.frontier.selection-scoring-context.v1", sources=receipt["scoring_implementation_sources"],
                    blocks={key: receipt["limits"][key] for key in ("pair_block", "gene_block", "row_block")})),
                prefix, task.key.training_donors)
            contexts.append(context); coverages.append(coverage)
            evidence.append(ScoreEvidence(task, recipe, context, coverage, {c: row[c] for c in CHANNELS}))
        if any(c != contexts[0] for c in contexts[1:]) or any(c != coverages[0] for c in coverages[1:]):
            raise ValueError("Calibration recipes must retain one fixed context and channel support")
        # Context-manager exits retain descriptor/path leases; additionally hash
        # the small source metadata again before returning the immutable result.
        for ref, stream in zip(refs, streams):
            stream.seek(0)
            if hashlib.sha256(stream.read()).hexdigest() != ref.sha256:
                raise ValueError("Selection metadata changed during consumption")
        provenance = dict(schema=SCHEMA, task=asdict(task), stage_plan_sha256=_hash(asdict(stage_plan)),
            metadata_files=[asdict(ref) for ref in refs], assessment_publication=publication,
            model_artifact_sha256=binding["model_artifact_sha256"], checkpoint_identity=checkpoint,
            prediction_derivations=receipt["prediction_derivations"], forecast_seal_sha256=admission.seal.sha256,
            contexts=[asdict(contexts[0])], source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            recipes=24, score_channels=list(CHANNELS), fabricated_scalar_metadata=admission.fabricated_scalar_metadata,
            metadata_byte_limit=admission.maximum_metadata_bytes, metadata_bytes=sum(r.size_bytes for r in refs),
            assessment_byte_limit=admission.maximum_assessment_bytes, optimizer_calls=0,
            bank_or_outcome_or_checkpoint_tensor_reads=0, selection_performed=False,
            caller_admitted_metadata_origin_not_independently_proved=True, scientific_launch_authority=False)
        provenance["truth_context_semantics"] = "Source/endpoint/support composite plus one common complete group seal; physical content and custody remain distinct"
        result = object.__new__(BoundAssessmentEvidence)
        object.__setattr__(result, "evidence", tuple(evidence))
        object.__setattr__(result, "_provenance", _encode(provenance))
    return result
