"""Explicitly opaque forecast-custody fixtures around a real compact checkpoint.

This publishes NO numerical forecast or score and opens NO outcome. Only the
selected checkpoint is real; unrelated configuration entries are labelled
fabricated group metadata. The selected checkpoint retains its actual two
updates and CUDA-fixture scope, never a nominal 1,024 scientific updates.
"""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .configuration import ALPHA_GRID, SPREAD_GRID, get_configuration
from .continuation_custody import ContinuationForecastAdmission, verify_continuation_custody
from .forecast_storage import ARRAY_NAMES
from .main_qualification_artifacts import file_reference, write_json, stage_task
from .main_qualification_worker_plan import cases
from .native_inputs import FoldInputs, ForecastExpectation
from .observation_loader import _path, _verified_file, ObservationFile
from .retention import identity_sha256
from .selection_evidence import FABRICATED_SCOPE, SelectionMetadataFile
from .selector_contract import Recipe


MIB = 1024**2


def plain(value):
    return json.loads(json.dumps(value,sort_keys=True,allow_nan=False))


def metadata_ref(reference):
    return SelectionMetadataFile(reference.path,reference.sha256,reference.size_bytes)


def publish_custody(root, *, checkpoint_complete):
    root = _path(root)
    if (not root.parent.is_dir() or type(checkpoint_complete) is not ObservationFile
            or Path(checkpoint_complete.path).name != "COMPLETE.json"):
        raise ValueError("Prepared absent custody root and exact prior checkpoint COMPLETE required")
    with _verified_file(checkpoint_complete,maximum=4*MIB) as stream:
        checkpoint = json.load(stream)
    if (checkpoint.get("execution_scope") != "synthetic_cuda_qualification"
            or checkpoint.get("accepted_updates") != 2 or checkpoint.get("attempted_updates") != 2
            or checkpoint.get("milestone") is not True or checkpoint.get("phase") != "joint"):
        raise ValueError("Actual healthy two-update CUDA-fixture checkpoint required")
    tensor = ObservationFile(str(Path(checkpoint_complete.path).parent / "checkpoint.pt"),
        checkpoint["checkpoint_sha256"],checkpoint["checkpoint_bytes"])
    with _verified_file(tensor,maximum=64*MIB):
        pass  # Byte custody, not deserialization or a new optimizer call.
    owner_identity = checkpoint["identity"]
    observation = owner_identity["training_owner"]["observation_binding"]
    plan, _, task = stage_task(cases()[0])
    following, _, next_task = stage_task(cases()[1])
    fold = FoldInputs("subset-D3-D4",("D3","D4"))
    if owner_identity["fold_id"] != fold.fold_id or owner_identity["seed"] != 0:
        raise ValueError("Checkpoint fold/seed differs from the fixed continuing worker")
    root.mkdir()
    write_json(root / "FIXTURE_ONLY.json",dict(fixture_only=True,numerical_forecasts=0,optimizer_calls=0,
        selected_checkpoint_actual_updates=2,unrelated_candidate_metadata_is_fabricated=True,
        opaque_payloads_are_not_arrays=True,scientific_launch_authority=False))
    admissions = []
    for donor in ("D1","D2"):
        folder = root / donor; folder.mkdir()
        group = [t for t in plan.scores_for(donor) if t.key.training_donors == task.key.training_donors]
        if len(group) != 24 or len({t.validation_donor for t in group}) != 1:
            raise ValueError("Complete fixed inner metadata group required")
        query = group[0].validation_donor
        source_id = identity_sha256(["fabricated-query-source",query])
        query_access = identity_sha256(["fabricated-query-access",query])
        source_ref = dict(donor=query,condition="Stim8hr",source_id=query+"_Stim8hr",
            source_access_sha256=query_access,source_input_sha256=identity_sha256(["input",query]))
        entries, configs, forecasts = [], [], []
        for edge in group:
            config_id = edge.key.configuration_id
            selected = config_id == "F0-V2"
            actual_prefix = 2 if selected else edge.updates
            bank_name = "opaque-custody-"+config_id
            forecast_id = "fabricated-bank-"+config_id
            binding = dict(fold=plain(asdict(fold)),observation_binding=observation,configuration_id=config_id,
                model_artifact_sha256=checkpoint["checkpoint_sha256"] if selected else identity_sha256(["unrun",config_id]),
                model_state_sha256=identity_sha256(["opaque-state-marker",config_id]),
                target_support_sha256=owner_identity.get("target_support_sha256",identity_sha256(["fixture-support"])),
                numerical_source_sha256=owner_identity["source_sha256"],model_seed=0,accepted_updates=actual_prefix)
            names = ("COMPLETE.json","MANIFEST.json",*(n+".npy" for n in ARRAY_NAMES))
            artifacts = []
            for name in names:
                raw = ("OPAQUE-CUSTODY-FIXTURE-NOT-A-NUMERICAL-FORECAST:"+query+":"+config_id+":"+name).encode()
                if selected:
                    path = folder / bank_name / name
                    path.parent.mkdir(exist_ok=True)
                    with path.open("xb") as out: out.write(raw)
                artifacts.append(dict(relative_uri=bank_name+"/"+name,
                    sha256=hashlib.sha256(raw).hexdigest(),size_bytes=len(raw)))
            parent = dict(root=str(folder / bank_name),complete_sha256=artifacts[0]["sha256"],
                manifest_sha256=artifacts[1]["sha256"],spec_sha256=identity_sha256(["opaque-spec",config_id]),
                total_bytes=sum(a["size_bytes"] for a in artifacts))
            context = dict(configuration_id=config_id,training_donors=list(fold.training_donors),model_seed=0,
                accepted_update=actual_prefix,purpose="forecast")
            role = dict(phase="inner",stage="screen",selector=edge.selector,task=plain(asdict(edge)),
                stage_plan_sha256=identity_sha256(asdict(plan)),eligible_configurations=list(plan.eligible_for(edge.selector)),
                required_recipe_count=24,outer_outcome_selection_allowed=False)
            catalog,axis = identity_sha256(["fabricated-catalog"]),identity_sha256(["fabricated-axis"])
            derivation = dict(schema="credo.frontier.calibration-derivation.v1",status="source_only_inventory_bound",
                parent_publication=parent,model_binding=binding,context=context,source_identity=source_id,
                catalog_sha256=catalog,feature_axis_sha256=axis,role=role,
                recipes=[asdict(Recipe(config_id,a,s)) for a in ALPHA_GRID for s in SPREAD_GRID],
                permitted_prefixes=[512],synthetic_fixture=True,fixture_scope=FABRICATED_SCOPE)
            entries.append(dict(forecast_id=forecast_id,bank_relative_root=bank_name,spec_sha256=parent["spec_sha256"],
                complete_sha256=parent["complete_sha256"],manifest_sha256=parent["manifest_sha256"],
                model_binding=binding,source_identity=source_id,source_reference=source_ref,
                catalog_sha256=catalog,feature_axis_sha256=axis))
            configs.append(dict(forecast_id=forecast_id,configuration=plain(asdict(get_configuration(config_id))),
                recipe=asdict(Recipe(config_id)),prefixes=[512],single_seed_prefixes=[512],
                ensemble_component_totals=[],ensemble_component_prefixes={},context=context,derivation_manifest=derivation))
            forecasts.append(dict(forecast_id=forecast_id,artifacts=artifacts))
        inventory = dict(schema="credo.frontier.forecast-seal-configurations.v1",entries=configs)
        expectation = ForecastExpectation(fold.identity,query_access,
            tuple(dict.fromkeys(e["model_binding"]["model_artifact_sha256"] for e in entries)),
            identity_sha256(inventory),owner_identity["source_sha256"],observation["artifact_sha256"],
            tuple(e["forecast_id"] for e in entries))
        seal_plan = write_json(folder / "SEAL.json.plan.json",dict(schema="credo.frontier.forecast-seal-plan.v1",
            fold=plain(asdict(fold)),query_donor=query,query_access_sha256=query_access,expectation=expectation.record(),
            configuration_inventory=inventory,entries=entries,inventory_completeness_is_controller_obligation=True,
            whole_study_seal=False,scientific_launch_authority=False))
        forecasts[0]["artifacts"].append(dict(relative_uri="SEAL.json.plan.json",sha256=seal_plan.sha256,
            size_bytes=seal_plan.size_bytes))
        seal = write_json(folder / "SEAL.json",dict(schema="credo.frontier.forecast-seal.v1",status="complete",
            bindings=expectation.record(),forecasts=forecasts))
        selected_edge = next(t for t in group if t.key.configuration_id == "F0-V2")
        admissions.append(ContinuationForecastAdmission(selected_edge.identifier,metadata_ref(seal),metadata_ref(seal_plan),
            metadata_ref(checkpoint_complete),4*MIB,4*MIB,True))
    admissions = tuple(sorted(admissions,key=lambda a:a.edge_id))
    verified = verify_continuation_custody(plan,next_task,admissions,checkpoint_identity=owner_identity,
        checkpoint_sha256=checkpoint["checkpoint_sha256"],qualification_prefix=2)
    receipt = write_json(root / "CUSTODY_COMPLETE.json",dict(schema="credo.frontier.main-qualification-custody.v1",
        status="opaque_fixture_custody_verified",admissions=[asdict(a) for a in admissions],
        checkpoint_complete=asdict(checkpoint_complete),verification=verified.record(),
        numerical_forecasts=0,scientific_updates=0,optimizer_calls=0,scientific_launch_authority=False))
    return admissions,receipt
