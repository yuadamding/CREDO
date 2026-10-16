"""Independent complete-case numerical/publication reconciliation.

Reads saved statistics and hashes checkpoint bytes; never instantiates a model,
deserializes a checkpoint, applies an optimizer update or accesses CUDA. It does
not establish scheduler exit or issue a blanket scientific-shape qualification.
"""

from contextlib import ExitStack
from dataclasses import asdict
from hashlib import sha256
import math
from pathlib import Path

from .configuration import get_configuration
from .geometry import GeometryOptions
from .main_qualification_contract import decode, exact, settings
from .main_qualification_fixture import FixtureDimensions
from .main_qualification_plan import cases, plan_identity, qualification_schedule, SCOPE
from .main_qualification_precision import PHASES, TOLERANCES, precision_dimensions, precision_plan
from .main_qualification_readback import verify_publication
from .main_qualification_worker_plan import cases as worker_cases, identity as worker_identity, worker_schedule
from .main_worker_contract import MAXIMUM_BYTES, parse, decode_contract, IMAGE
from .main_worker_source import source_closure, manifest_payload, QUALIFICATION_ENTRY
from .observation_loader import ObservationFile, _path, _verified_file
from .qualification_inventory import OutputInventory
from .retention import identity_sha256, STUDY_ID


GIB,MIB=1024**3,1024**2


def finite(value,*,positive=False):
    if type(value) not in (int,float) or not math.isfinite(value) or value<0 or (positive and value==0):
        raise ValueError("Finite nonnegative numerical statistic required")
    return value


def count(value,*,positive=False):
    if type(value) is not int or value<0 or (positive and value==0):
        raise ValueError("Exact nonnegative count required")
    return value


def hardware(value):
    exact(value,dict(device="cuda:0",name="NVIDIA H100 80GB HBM3",capability=[9,0],visible_devices=1,
        precision="FP32",tf32=False,deterministic_algorithms=True,required_headroom_bytes=8*GIB))
    if not 75*GIB<=count(value["total_memory_bytes"])<=82*GIB:
        raise ValueError("Required full H100 hardware class differs")


def resources(value,total):
    if count(value["maximum_rss_bytes"])>200*GIB:
        raise ValueError("Measured host memory exceeded bound")
    for kind in ("allocated","reserved"):
        current=count(value[kind+"_bytes"]);peak=count(value["peak_"+kind+"_bytes"])
        if current>peak or peak>total-8*GIB:raise ValueError("GPU resource peak/headroom differs")
    if value["peak_allocated_bytes"]>value["peak_reserved_bytes"]:
        raise ValueError("GPU allocated/reserved peaks are inconsistent")
    if value["physical_headroom_at_peak_reserved_bytes"]!=total-value["peak_reserved_bytes"]:
        raise ValueError("Saved headroom arithmetic differs")


def comparison(value):
    exact(value,dict(status="passed",all_parameter_gradients_checked=True))
    channels,gradients=value["channels"],value["parameter_gradients"]
    if not isinstance(channels,dict) or not isinstance(gradients,dict) or not channels or not gradients:
        raise ValueError("Complete nonempty precision channels and gradients required")
    absent=value["absent_gradients"]
    if (type(absent) is not list or any(type(k) is not str for k in absent)
            or len(absent)!=len(set(absent)) or set(absent)&set(gradients)):
        raise ValueError("Exact distinct gradient participation required")
    for item in (*channels.values(),*gradients.values()):
        if set(item)!={"elements","maximum_absolute_error","maximum_reference_magnitude",
                        "maximum_tolerance_normalized_error","passed"}:
            raise ValueError("Precision element summary fields differ")
        count(item["elements"],positive=True)
        for key in ("maximum_absolute_error","maximum_reference_magnitude","maximum_tolerance_normalized_error"):
            finite(item[key])
        if item["passed"] is not True or item["maximum_tolerance_normalized_error"]>1:
            raise ValueError("A saved elementwise precision tolerance failed")
    exact(value,dict(compared_gradient_elements=sum(g["elements"] for g in gradients.values()),
        maximum_gradient_absolute_error=max(g["maximum_absolute_error"] for g in gradients.values()),
        maximum_channel_absolute_error=max(g["maximum_absolute_error"] for g in channels.values()),
        maximum_tolerance_normalized_error=max(g["maximum_tolerance_normalized_error"]
            for g in (*channels.values(),*gradients.values()))))
    if not {"objective","global_regularizer","active_log_mass"}<=set(channels):
        raise ValueError("Missing mandatory objective channels")
    return set(channels),set(gradients),set(absent)


def verify_precision(value,case,phase,*,source,rules,geometry):
    dims=precision_dimensions()
    exact(value,dict(schema="credo.frontier.main-paired-precision-case.v1",status="passed",
        case=asdict(case),phase=phase,input_dimensions=asdict(dims),tolerances=dict(TOLERANCES),
        source_sha256=source,rules_sha256=rules,geometry=geometry,
        input_identity=identity_sha256(dict(schema="credo.main-qualification-fabricated-inputs.v1",
            donors=case.training_donors,dimensions=asdict(dims),seed=419)),
        precision_plan_sha256=identity_sha256(precision_plan()),optimizer_calls=0,scientific_updates=0,
        same_inputs_and_initial_weights=True,real_biological_payload_access=False,
        full_worker_entry_verified=False,full_shape_memory_qualified=False,main_worker_qualification_pass=False))
    a,b=comparison(value["cpu_cuda"]),comparison(value["cuda_microbatch_accumulation"])
    if a!=b:raise ValueError("CPU/CUDA and microbatch comparison inventories differ")
    active=count(value["active_parameter_count"],positive=True)
    if active>count(value["parameter_count"],positive=True):raise ValueError("Active parameter count exceeds total")
    if not 0<value["cpu_cuda"]["compared_gradient_elements"]<=active:
        raise ValueError("Compared parameter gradient count differs")
    if phase=="joint" and not {"count_ce","donor_count_ce","count_gradient_coefficients"}<=a[0]:
        raise ValueError("Missing joint catalog precision channels")


def verify_owner(inventory,value,case,*,source,geometry,total_gpu_bytes):
    exact(value,dict(schema="credo.frontier.main-cuda-owner-case.v1",status="owner_case_passed",case=asdict(case),
        plan_sha256=plan_identity(),source_sha256=source,input_dimensions=asdict(FixtureDimensions()),
        execution=settings()["owner_execution"],geometry=geometry,optimizer_calls=case.optimizer_calls,
        scientific_updates=0,real_biological_payload_access=False,restart_verified=case.restart_comparison,
        full_worker_entry_verified=False,main_worker_qualification_pass=False,automatic_retry=False,automatic_successor=False))
    expected=[("primary",2,2)]
    if case.restart_comparison:expected += [("primary",3,1),("continuous-reference",3,3)]
    rows=value["records"]
    if len(rows)!=len(expected):raise ValueError("Owner/restart record inventory differs")
    base=inventory.root/"numerics"/case.identifier
    checkpoints=set()
    for row,(label,updates,calls_) in zip(rows,expected,strict=True):
        exact(row,dict(label=label,accepted=updates,disposable_calls=calls_,scientific_updates=0))
        count(row["active_parameters"],positive=True)
        metrics=row["last_metrics"]
        exact(metrics,dict(update=updates,phase="joint"))
        finite(metrics["gradient_norm_before_clip"],positive=True)
        # Losses may be signed; only finiteness is imposed, never performance.
        for key in ("objective","count_ce"):
            if type(metrics[key]) not in (float,int) or not math.isfinite(metrics[key]):
                raise ValueError("Finite actual joint loss required")
        saved=row["checkpoint"];directory=base/label/"checkpoints"/f"update-{updates:09d}"
        if _path(saved["directory"])!=directory:raise ValueError("Owner checkpoint escaped its case/prefix")
        cp=inventory.read(directory/"COMPLETE.json")
        if cp!={k:v for k,v in saved.items() if k!="directory"}:raise ValueError("Owner checkpoint metadata differs")
        exact(cp,dict(schema="credo.frontier.resumable_checkpoint.v1",accepted_updates=updates,
            attempted_updates=updates,phase="joint",execution_scope=SCOPE,milestone=True,maximum_checkpoint_bytes=64*MIB))
        exact(cp["identity"],dict(trajectory_id=label,fold_id="fabricated-"+"-".join(case.training_donors),
            seed=419,source_sha256=source,configuration_sha256=identity_sha256(asdict(get_configuration(case.configuration_id)))))
        exact(cp["identity"]["training_owner"],dict(schedule_sha256=identity_sha256(asdict(qualification_schedule())),
            maximum_checkpoint_bytes=64*MIB))
        tref=inventory.reference(directory/"checkpoint.pt",maximum=64*MIB)
        if (tref.sha256,tref.size_bytes)!=(cp["checkpoint_sha256"],cp["checkpoint_bytes"]):
            raise ValueError("Checkpoint payload digest or bytes differ")
        release=inventory.read(base/label/"claims"/(cp["claim_id"]+".released.json"))
        exact(release,dict(claim_id=cp["claim_id"],exception_type=None,automatic_restart_authorized=False))
        guard=row["resources"]
        exact(guard,dict(threads=8,accepted_updates=updates,event="stage_complete",
            gpu_name="NVIDIA H100 80GB HBM3",total_gpu_bytes=total_gpu_bytes))
        if (count(guard["maximum_rss_bytes"])>200*GIB or count(guard["output_bytes"])>256*MIB
                or count(guard["peak_reserved_gpu_bytes"])>count(guard["total_gpu_bytes"])-8*GIB):
            raise ValueError("Owner resource guard bounds differ")
        checkpoints.add(tref.path)
    expected_times=[("primary",2)] + ([("primary",3),("continuous-reference",2),("continuous-reference",3)]
                                      if case.restart_comparison else [])
    if len(value["timings"])!=len(expected_times):raise ValueError("Owner timing stage inventory differs")
    for row,(label,updates) in zip(value["timings"],expected_times,strict=True):
        exact(row,dict(label=label,accepted_target=updates));finite(row["elapsed_seconds"],positive=True)
    all_checkpoints=0
    trajectories=[("primary",3 if case.restart_comparison else 2)]
    if case.restart_comparison:trajectories.append(("continuous-reference",3))
    for label,stop in trajectories:
        # Every accepted disposable prefix is retained in this qualification.
        # Hashing a terminal checkpoint alone cannot prove complete publication.
        for update in range(1,stop+1):
            directory=base/label/"checkpoints"/f"update-{update:09d}"
            cp=inventory.read(directory/"COMPLETE.json")
            exact(cp,dict(schema="credo.frontier.resumable_checkpoint.v1",accepted_updates=update,
                attempted_updates=update,phase="state" if update==1 else "joint",execution_scope=SCOPE,
                milestone=update>=2,maximum_checkpoint_bytes=64*MIB))
            tref=inventory.reference(directory/"checkpoint.pt",maximum=64*MIB)
            if (tref.sha256,tref.size_bytes)!=(cp["checkpoint_sha256"],cp["checkpoint_bytes"]):
                raise ValueError("A retained checkpoint prefix has changed bytes")
            exact(cp["identity"],dict(trajectory_id=label,source_sha256=source,
                configuration_sha256=identity_sha256(asdict(get_configuration(case.configuration_id)))))
            exact(inventory.read(base/label/"claims"/(cp["claim_id"]+".released.json")),dict(
                claim_id=cp["claim_id"],exception_type=None,automatic_restart_authorized=False))
            all_checkpoints+=1
    return len(checkpoints),all_checkpoints


def verify(contract_ref,execution_ref):
    """No scheduler evidence or scientific permission is fabricated by reading."""
    with ExitStack() as leases:
        def metadata(ref):
            return parse(leases.enter_context(_verified_file(ref,maximum=MAXIMUM_BYTES)).read())
        raw=leases.enter_context(_verified_file(contract_ref,maximum=MAXIMUM_BYTES)).read()
        record,refs=decode(raw)
        root=_path(record["output_root"])
        inventory=OutputInventory(root,limit=16*GIB,
            excluded=("READBACK.log","READBACK_EXIT.json","TERMINAL_READBACK.json","COMPLETE.json"))
        if execution_ref.path!=str(root/"EXECUTION_COMPLETE.json"):
            raise ValueError("Exact parent execution publication required")
        done=inventory.read(root/"EXECUTION_COMPLETE.json",execution_ref)
        exact(done,dict(schema="credo.frontier.main-qualification-execution.v1",
            status="components_complete_requires_independent_terminal_readback",contract=asdict(contract_ref),
            source_manifest_sha256=refs["source_manifest"].sha256,
            worker_source_manifest_sha256=refs["worker_source_manifest"].sha256,
            reported_disposable_optimizer_calls=111,maximum_disposable_optimizer_calls=112,scientific_updates=0,
            real_biological_payload_access=False,scientific_launch_authority=False,main_worker_qualification_pass=False,
            automatic_retry=False,automatic_successor=False,all_parent_source_and_metadata_exits_verified=True))
        source=source_closure(_path(record["source_root"]),entry=QUALIFICATION_ENTRY)
        worker=source_closure(_path(record["source_root"]))
        for entry,files,key in ((QUALIFICATION_ENTRY,source,"source_manifest"),
                                ("frontier/main_worker.py",worker,"worker_source_manifest")):
            raw_manifest=leases.enter_context(_verified_file(refs[key],maximum=MAXIMUM_BYTES)).read()
            if raw_manifest!=manifest_payload(files,entry=entry):raise ValueError("Terminal source closure differs")
        # Short metadata/code leases only. Checkpoint bytes were hashed by the
        # streaming inventory; no model load or long-lived tensor descriptor.
        for name,raw_source in source.items():
            leases.enter_context(_verified_file(ObservationFile(str(Path(record["source_root"])/name),
                sha256(raw_source).hexdigest(),len(raw_source)),maximum=2*MIB))
        release_ref=ObservationFile(**done["release"]);release=metadata(release_ref)
        exact(release,dict(run_id=record["run_id"],worker_contract_sha256=contract_ref.sha256,
            source_manifest_sha256=refs["source_manifest"].sha256,image=IMAGE,
            scientific_fit_authorized=False,disposable_qualification_authorized=True))
        exact(inventory.read(root/"STARTED.json"),dict(run_id=record["run_id"],contract=asdict(contract_ref),
            release=asdict(release_ref),scientific_updates=0,automatic_retry=False))
        exact(inventory.read(root/"NUMERICS_EXIT.json"),dict(returncode=0,automatic_retry=False))
        numerical_ref=ObservationFile(**done["numerics"])
        numerical=inventory.read(root/"numerics/COMPLETE.json",numerical_ref)
        exact(numerical,dict(schema="credo.frontier.main-qualification-numerics.v1",status="numerical_components_passed",
            plan_sha256=plan_identity(),source_manifest_sha256=refs["source_manifest"].sha256,
            disposable_optimizer_calls=104,scientific_updates=0,runtime=record["expected_runtime"],
            real_biological_payload_access=False,scientific_launch_authority=False,full_worker_entry_verified=False,
            automatic_retry=False))
        hardware(numerical["hardware"])
        resources(numerical["resources"],numerical["hardware"]["total_memory_bytes"])
        backend=numerical["backend"]
        if (set(backend)!={"version","scalar_module","divergence_module","utilities_module"}
                or backend["scalar_module"] not in ("geomloss.samples_loss","geomloss._legacy.samples_loss")):
            raise ValueError("Exact public GeomLoss backend binding required")
        namespace=backend["scalar_module"].rsplit(".",1)[0]
        exact(backend,dict(divergence_module=namespace+".sinkhorn_divergence",utilities_module=namespace+".utils"))
        fixed=settings()
        geometry=asdict(GeometryOptions(fixed["maximum_geometry_elements"],fixed["maximum_geometry_guides"],
                                        numerical["backend"]["version"]))
        started=inventory.read(root/"numerics/STARTED.json")
        exact(started,{k:numerical[k] for k in ("hardware","runtime","backend","plan_sha256","scientific_updates","automatic_retry")})
        precision_refs=[ObservationFile(**r) for r in numerical["precision"]]
        owner_refs=[ObservationFile(**r) for r in numerical["owners"]]
        expected_precision=[(case,phase) for case in cases() for phase in PHASES]
        if len(precision_refs)!=96 or len(owner_refs)!=48:raise ValueError("Complete 96/48 case inventory required")
        for ref,(case,phase) in zip(precision_refs,expected_precision,strict=True):
            value=inventory.read(root/"numerics"/(case.identifier+"-"+phase+"-precision.json"),ref)
            verify_precision(value,case,phase,source=refs["source_manifest"].sha256,
                             rules=refs["rules"].sha256,geometry=geometry)
        checkpoint_count,all_checkpoint_count=0,0
        for ref,case in zip(owner_refs,cases(),strict=True):
            value=inventory.read(root/"numerics"/case.identifier/"OWNER_CASE_COMPLETE.json",ref)
            terminal_count,full_count=verify_owner(inventory,value,case,source=refs["source_manifest"].sha256,
                geometry=geometry,total_gpu_bytes=numerical["hardware"]["total_memory_bytes"])
            checkpoint_count+=terminal_count;all_checkpoint_count+=full_count
        fixture=root/"qualification-fixtures"/record["run_id"]/STUDY_ID
        sequence_ref=ObservationFile(**done["worker_sequence"])
        sequence=inventory.read(fixture/"worker_sequence/COMPLETE.json",sequence_ref)
        exact(sequence,dict(schema="credo.frontier.main-qualification-worker-sequence.v1",
            status="three_worker_publications_verified",disposable_optimizer_calls_reported=7,scientific_updates=0,
            worker_plan_sha256=worker_identity(),same_assigned_gpu_sequential_workers=True,
            checkpoint_continuation_case_included=True,full_main_qualification_pass=False,scientific_launch_authority=False,
            automatic_retry=False))
        if len(sequence["readbacks"])!=3:raise ValueError("All three worker cases required")
        worker_checkpoint_count=0
        for ref,case in zip(sequence["readbacks"],worker_cases(),strict=True):
            folder=fixture/"worker_requests"/case.stage
            saved=inventory.read(folder/"READBACK.json",ObservationFile(**ref))
            cref=inventory.reference(folder/"CONTRACT.json")
            worker_record,parts=decode_contract(Path(cref.path).read_bytes(),qualification=True)
            if parts["qualification_case"]!=case or parts["source_manifest"].sha256!=refs["worker_source_manifest"].sha256:
                raise ValueError("Worker case/source differs from parent")
            for key in ("source_manifest","input_reference","admission","qualification"):
                inventory.read(parts[key].path,parts[key])
            cpath=Path(worker_record["receipt_root"])/"COMPLETE.json"
            fresh=verify_publication(cref,inventory.reference(cpath))
            if identity_sha256(fresh)!=identity_sha256(saved):
                raise ValueError("Fresh independent worker readback differs")
            started=inventory.read(fresh["started"]["path"],ObservationFile(**fresh["started"]))
            exact(started,dict(hardware=numerical["hardware"],runtime=record["expected_runtime"]))
            worker_release_ref=ObservationFile(**started["startup_release"])
            worker_release=inventory.read(folder/"RELEASE.json",worker_release_ref)
            exact(worker_release,dict(schema="credo.frontier.main-worker-qualification-startup-release.v1",
                parent_run_id=record["run_id"],run_id=worker_record["worker_id"],worker_contract_sha256=cref.sha256,
                pod_name=release["pod_name"],pod_uid=release["pod_uid"],job_uid=release["job_uid"],
                scientific_fit_authorized=False,disposable_qualification_authorized=True,
                disposable_optimizer_calls=case.optimizer_calls))
            exact(inventory.read(folder/"PROCESS_EXIT.json"),dict(returncode=0,automatic_retry=False))
            trajectory=fixture/"trajectories"/parts["task"].key.storage_identifier
            for update in range(case.accepted_start+1,case.accepted_target+1):
                directory=trajectory/"checkpoints"/f"update-{update:09d}"
                cp=inventory.read(directory/"COMPLETE.json")
                exact(cp,dict(schema="credo.frontier.resumable_checkpoint.v1",accepted_updates=update,
                    attempted_updates=update,phase="state" if update==1 else "joint",execution_scope=SCOPE,
                    milestone=update>=2,maximum_checkpoint_bytes=64*MIB,worker_id=worker_record["worker_id"]))
                exact(cp["identity"],dict(trajectory_id=parts["task"].key.storage_identifier,seed=0,
                    source_sha256=refs["worker_source_manifest"].sha256,
                    configuration_sha256=identity_sha256(asdict(get_configuration(case.configuration_id)))))
                exact(cp["identity"]["training_owner"],dict(schedule_sha256=identity_sha256(asdict(worker_schedule()))))
                tref=inventory.reference(directory/"checkpoint.pt",maximum=64*MIB)
                if (tref.sha256,tref.size_bytes)!=(cp["checkpoint_sha256"],cp["checkpoint_bytes"]):
                    raise ValueError("Worker checkpoint prefix payload differs")
                exact(inventory.read(trajectory/"claims"/(cp["claim_id"]+".released.json")),dict(
                    claim_id=cp["claim_id"],exception_type=None,automatic_restart_authorized=False))
                worker_checkpoint_count+=1
        inventory.verify_exit()
        result=dict(schema="credo.frontier.main-qualification-terminal-readback.v1",
            status="complete_numerical_publication_inventory_verified",contract=asdict(contract_ref),
            execution=asdict(execution_ref),files=inventory.files,total_bytes=inventory.total_bytes,
            precision_cases=96,owner_cases=48,owner_terminal_checkpoint_payloads=checkpoint_count,
            owner_all_checkpoint_payloads=all_checkpoint_count,
            worker_all_checkpoint_payloads=worker_checkpoint_count,
            worker_cases=3,disposable_optimizer_calls_reported=111,scientific_updates=0,
            checkpoint_deserializations=0,numerical_replay=False,real_biological_payload_access=False,
            source_manifest_sha256=refs["source_manifest"].sha256,
            worker_source_manifest_sha256=refs["worker_source_manifest"].sha256,
            scheduler_terminal_verified=False,main_worker_qualification_pass=False,scientific_launch_authority=False)
    return result
