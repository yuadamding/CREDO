"""One explicitly approved, zero-update observation replacement batch.

Read-only remote intake precedes a tested local journal amendment. This does
not submit Jobs, retry failed replacements, delete old evidence, or train.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from observation import package, HOME, SCRATCH
from reserve_observation_fits import (qualification_evidence, _capacity, _requests,
    sha, read_json, publish, OUTPUT_BYTES, UPDATES, GIB)
from transport import remote, record, capacity
from frontier.coordination import AdmissionJournal, LIVE, TERMINAL

AUTHORITY = "OBSERVATION_REPLACEMENT_FITS_V2_AUTHORITY.json"
REQUEST = "OBSERVATION_REPLACEMENT_FIT_BATCH_V2.json"
STATUS = "receipts/OBSERVATION_QUALIFICATION_V2_COMPLETION_STATUS_20261015.json"
SUBSETS = [["D1","D2"],["D1","D3"],["D1","D4"],["D2","D3"]]


def bindings():
    e = qualification_evidence(ROOT, ROOT/"observation_qualification_launch_v2", ROOT/STATUS)
    a = read_json(ROOT/AUTHORITY)
    expected = dict(schema="credo.frontier.observation-zero-update-replacement-authority.v1",
        user_instruction="Approve the four replacement observation fits",
        qualification_run_id=e["run_id"], qualification_sha256=e["qualification_sha256"],
        qualification_snapshot_sha256=e["snapshot_sha256"], source_manifest_sha256=e["source_manifest_sha256"],
        training_subsets=SUBSETS, updates_per_fit=20000, total_scientific_updates=80000,
        additional_study_trajectories=0, resources_per_worker=dict(H100=1,CPUs=24,host_memory_GiB=200,
            application_threads=8,output_GiB=4), maximum_concurrent_H100s=4,
        aggregate_study_output_cap_GiB=384,unallocated_fileset_reserve_GiB=16,
        preserve_old_attempts_and_storage_reservations=True,automatic_retries=False,
        automatic_successor=False,main_model_fitting=False,new_variants_or_seeds=False,
        elapsed_time_deadline=None,LARRY_access=False)
    from deployment.common import encoded
    if any(k not in a or encoded(a[k]) != encoded(v) for k,v in expected.items()):
        raise ValueError("Exact current user replacement authority required")
    rows = _requests(ROOT, ROOT/REQUEST, e)
    old = read_json(ROOT/"OBSERVATION_FIRST_FIT_BATCH.json")["requested_fits"]
    if [r["training_donors"] for r in rows] != SUBSETS or [r["training_donors"] for r in old] != SUBSETS:
        raise ValueError("Only the approved first four observation subsets may be replaced")
    for prior,new in zip(old,rows):
        if new["run_id"] != prior["run_id"].removesuffix("v1")+"v2" or new["phase"] != prior["phase"].removesuffix("v1")+"v2":
            raise ValueError("Exact disjoint v2 attempt identities required")
        p,b,plan = package(ROOT/(prior["phase"]+"_launch"))
        if b["run_id"] != prior["run_id"] or plan["mode"] != "fit":
            raise ValueError("Original source-bound fit package differs")
        if prior["training_donors"] != ["D1","D2"] and (p/"CREATE.json").exists():
            raise ValueError("Unsubmitted predecessor has a create receipt")
        new["predecessor_id"] = prior["run_id"]
        new["predecessor_plan_sha256"] = b["plan_sha256"]
    failure = read_json(ROOT/"receipts/OBSERVATION_D1D2_FAILURE_TERMINAL_20261015_V1.json")
    log = read_json(ROOT/"receipts/OBSERVATION_D1D2_FAILURE_LOG_20261015_V1.json")
    if (failure["uid"] != "2587901f-4deb-4518-997b-2e3c80fccb09" or failure["job"] is not None
            or failure["pods"] != [] or failure["records"]["output/WORKER_STARTED.json"] is not None
            or failure["records"]["terminal.json"]["value"]["application_pass"] is not False
            or "qualification = verify_qualification(contract, hardware)" not in log["text"]
            or "Missing, failed or mismatched same-source H100 qualification" not in log["text"]):
        raise ValueError("Original pre-input/pre-optimizer failure proof differs")
    return e,a,rows,old,log


INTAKE = r'''
import datetime,hashlib,json,os,stat,subprocess
from pathlib import Path
k=['/risapps/noarch/kubectl/1.28.4/bin/kubectl','--context','yding4_yn-gpu-workload@kubernetes-admin@kubernetes','-n','yn-gpu-workload','--request-timeout=30s']
config='/rsrch8/home/bcb/yding4/.kube/config'
s=os.lstat(config)
if not stat.S_ISREG(s.st_mode) or s.st_uid!=307469 or stat.S_IMODE(s.st_mode)!=0o600:raise RuntimeError('Kubeconfig metadata differs')
def query(args):
 r=subprocess.run(k+args,env=dict(os.environ,KUBECONFIG=config),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=45)
 if r.returncode:raise RuntimeError(r.stderr.decode()[-2000:])
 return json.loads(r.stdout) if r.stdout.strip() else None
checks=[]
for prior,new in zip(OLD,NEW):
 for row in (prior,new):
  j=query(['get','job/'+row['run_id'],'--ignore-not-found=true','-o','json'])
  p=query(['get','pods','-l','job-name='+row['run_id'],'-o','json'])['items']
  if j is not None or p:raise RuntimeError('Old or replacement exact Job/Pods still exist: '+row['run_id'])
 old_home=Path(HOME)/prior['phase']; old_control=Path(SCRATCH)/prior['phase']
 raw=(old_home/'bundle/PLAN.json').read_bytes()
 if hashlib.sha256(raw).hexdigest()!=new['predecessor_plan_sha256']:raise RuntimeError('Original remote plan changed')
 if os.path.lexists(str(old_home/'output')):raise RuntimeError('Original attempt has scientific output; zero-work recovery cannot assume absence')
 if prior['training_donors']!=['D1','D2']:
  for p in [old_control/'gpu/live-binding.json',old_control/'gpu/launch-success.json',old_home/'handshake/STARTUP.json']:
   if os.path.lexists(str(p)):raise RuntimeError('Unsubmitted attempt has execution evidence')
 for parent in (Path(HOME),Path(SCRATCH)):
  if os.path.lexists(str(parent/new['phase'])):raise RuntimeError('Replacement root already exists')
 checks.append(dict(predecessor_id=prior['run_id'],replacement_id=new['run_id'],
  old_job_absent=True,old_owned_pods=0,new_job_absent=True,new_owned_pods=0,
  old_scientific_output_absent=True,new_roots_absent=True,
  predecessor_plan_sha256=new['predecessor_plan_sha256']))
log=Path(FAILURE_LOG_PATH)
if log.stat().st_uid!=307469 or log.is_symlink() or log.stat().st_size>1024**2 or hashlib.sha256(log.read_bytes()).hexdigest()!=FAILURE_LOG_SHA:
 raise RuntimeError('Retained zero-work failure log changed')
print('RECEIPT='+json.dumps(dict(schema='credo.frontier.observation-replacement-intake.v1',
 at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='passed',
 authority_sha256=AUTHORITY_SHA,request_sha256=REQUEST_SHA,checks=checks,
 failure_log_sha256=FAILURE_LOG_SHA,remote_writes=False,biological_payloads_opened=False)))
'''


def intake(output, capacity_output):
    if output.exists() or capacity_output.exists(): raise ValueError("New receipt identities required")
    e,a,rows,old,log=bindings()
    current=remote(INTAKE,dict(OLD=old,NEW=[{k:v for k,v in r.items() if k not in ("work_binding","fit_bindings")} for r in rows],
        HOME=HOME,SCRATCH=SCRATCH,FAILURE_LOG_PATH=log["path"],FAILURE_LOG_SHA=log["sha256"],
        AUTHORITY_SHA=sha(ROOT/AUTHORITY),REQUEST_SHA=sha(ROOT/REQUEST)))
    record(output,current)
    cap=capacity(); record(capacity_output,cap)
    return dict(status="read_only_intake_passed",headroom_bytes=_capacity(capacity_output,e),receipt=str(output))


def reserve(intake_path, capacity_path, output):
    if output.exists() or output.parent != ROOT: raise ValueError("New exact study-local export directory required")
    e,a,rows,old,log=bindings(); check=read_json(intake_path)
    authority_sha,proof_sha=sha(ROOT/AUTHORITY),sha(intake_path)
    if (check.get("status")!="passed" or check.get("authority_sha256")!=authority_sha
            or check.get("request_sha256")!=sha(ROOT/REQUEST) or check.get("remote_writes") is not False
            or check.get("failure_log_sha256")!=log["sha256"] or len(check.get("checks",[]))!=4):
        raise ValueError("Exact approved read-only zero-work intake required")
    for row,proof in zip(rows,check["checks"]):
        expected=dict(predecessor_id=row["predecessor_id"],replacement_id=row["run_id"],
            old_job_absent=True,old_owned_pods=0,new_job_absent=True,new_owned_pods=0,
            old_scientific_output_absent=True,new_roots_absent=True,
            predecessor_plan_sha256=row["predecessor_plan_sha256"])
        if proof!=expected: raise ValueError("Exact predecessor/replacement absence proof required")
    headroom=_capacity(capacity_path,e); cap_sha=sha(capacity_path)
    with AdmissionJournal(ROOT/"admission",study_identity_sha256=sha(ROOT/"AUTHORITY.json")) as journal:
        before=journal.snapshot(); tasks=before["tasks"]
        if tasks[e["run_id"]]["status"]!="complete": raise ValueError("Qualification must be reconciled complete")
        for row in rows:
            t=tasks[row["predecessor_id"]]
            if (row["run_id"] in tasks or t["status"] != ("failed" if row["training_donors"]==["D1","D2"] else "reserved")
                    or t["work_binding"]!=row["work_binding"].to_dict()
                    or {"replacement_task_id","replaces_task_id"}&set(t)):
                raise ValueError("Original zero-work batch reservation differs")
        if headroom<sum(t["storage_reserved_bytes"] for t in tasks.values())+4*OUTPUT_BYTES+16*GIB:
            raise ValueError("Whole recovery batch plus historical storage and reserve does not fit")
        output.mkdir(mode=0o700)
        publish(output/"BATCH_INTENT.json",dict(authority_sha256=authority_sha,zero_update_evidence_sha256=proof_sha,
            qualification_evidence=e,capacity_receipt_sha256=cap_sha,predecessor_journal_sha256=before["last_sha256"],
            automatic_retry=False,requested_runs=[r["run_id"] for r in rows]))
        for row in rows:
            state=journal.replace_zero_update_representation(row["predecessor_id"],row["run_id"],
                authority_sha256=authority_sha,zero_update_evidence_sha256=proof_sha,
                admission_sha256=e["phase_authority_sha256"],capacity_receipt_sha256=cap_sha,
                fileset_headroom_bytes=headroom)
        aggregate=sum(t["storage_reserved_bytes"] for t in state["tasks"].values())
        live=sum(t["status"] in LIVE for t in state["tasks"].values()); exports=[]
        for row in rows:
            admission=dict(schema="credo.frontier.observation-admission-export.v1",run_id=row["run_id"],
                status="reserved",storage_reserved_bytes=OUTPUT_BYTES,scientific_update_reservation=UPDATES,
                aggregate_reserved_output_bytes=aggregate,live_h100_reservations=live,
                journal_event_sha256=state["last_sha256"],journal_identity_sha256=sha(ROOT/"admission/IDENTITY.json"),
                phase_authority_sha256=e["phase_authority_sha256"],source_manifest_sha256=e["source_manifest_sha256"],
                qualification_sha256=e["qualification_sha256"],qualified_hardware=e["qualified_hardware"],
                qualified_peak_reserved_bytes=e["qualified_peak_reserved_bytes"],capacity_receipt_sha256=cap_sha,
                work_binding=row["work_binding"].to_dict(),scheduler_submitted=False,scientific_updates=0,
                replacement_authority_sha256=authority_sha,replaces_task_id=row["predecessor_id"],
                zero_update_evidence_sha256=proof_sha,replacement_authority=a)
            member=output/row["run_id"]; member.mkdir(mode=0o700)
            ar=publish(member/"ADMISSION.json",admission); br=publish(member/"FIT_BINDINGS.json",row["fit_bindings"])
            exports.append(dict(run_id=row["run_id"],phase=row["phase"],training_donors=row["training_donors"],
                admission=dict(path=str(member/"ADMISSION.json"),**ar),fit_bindings=dict(path=str(member/"FIT_BINDINGS.json"),**br)))
        result=dict(schema="credo.frontier.observation-replacement-batch-reservation.v1",status="reserved_not_launched",
            fits=exports,journal_event_sha256=state["last_sha256"],qualification_evidence=e,
            replacement_authority_sha256=authority_sha,zero_update_evidence_sha256=proof_sha,
            aggregate_reserved_output_bytes=aggregate,live_h100_reservations=live,
            scientific_reservation_transferred=80000,additional_scientific_trajectories=0,
            old_attempts_preserved=True,automatic_retry=False,scheduler_calls=0)
        publish(output/"BATCH_RESERVED.json",result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("action",choices=("intake","reserve"))
    p.add_argument("--intake",type=Path,required=True); p.add_argument("--capacity",type=Path,required=True)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    value=intake(a.intake.absolute(),a.capacity.absolute()) if a.action=="intake" else reserve(a.intake.absolute(),a.capacity.absolute(),a.output.absolute())
    print(json.dumps(value,sort_keys=True))

if __name__=="__main__": main()
