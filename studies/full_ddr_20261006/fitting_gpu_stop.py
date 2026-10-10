"""One explicitly authorized exact-UID stop; never submits or restarts a Job.

Uses the frozen lifecycle's reviewed foreground cleanup with UID/resourceVersion
preconditions. The new stop evidence is disjoint from every original receipt.
Published models remain provisional: cancellation cannot fabricate lease exit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import fitting_operate
import prepare_operate as ops

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "gpu_native_transition_v1"
JOB = "credo-full-ddr-fitting-20261007-v1"
UID = "d607a5fa-48a5-40b6-b04f-59c2cc631dbc"
CORE_SHA = "3da1a0a5b0d0d26019918ee48e2ce24cd2e616272de7854f4c82ba2e37e7ebd2"
REMOTE_EVIDENCE = "/rsrch8/scratch/bcb/yding4/perturbseq/credo_full_ddr_gse314342_20261006_v1/gpu_native_transition_v1"


def account_progress(fits, records):
    """Durable counters, not a fabricated exact update for a killed step."""
    published = sum(row["optimizer_updates"] for row in fits)
    last = records[-1] if records else {}
    current = last.get("fit")
    rolled_up = any(current == {k: row[k] for k in ("candidate_id", "seed")} for row in fits)
    relevant = [row for row in records if row.get("fit") == current]
    applied = max((row.get("optimizer_updates", 0) for row in relevant), default=0)
    attempted = max((row.get("attempted_optimizer_updates", row.get("optimizer_updates", 0))
                     for row in relevant), default=0)
    lower = published + (0 if rolled_up else applied)
    upper = published + (0 if rolled_up else max(applied, attempted))
    return dict(published_fits=len(fits), published_fit_optimizer_updates=published,
        unpublished_fit=current if not rolled_up else None,
        durably_recorded_optimizer_updates=lower, attempted_updates_upper_bound=upper,
        exact_unpublished_optimizer_completion_verified=False,
        reuse_unpublished_model=False, optimizer_resume=False)


REMOTE_STOP = r'''
import hashlib,importlib.util,json,os,stat,sys
from pathlib import Path
from datetime import datetime,timezone
if (os.getuid(),os.getgid())!=(307469,600815):raise RuntimeError('Stop writer owner differs')
def checked(path,expected=None,retain=True):
 p=Path(path)
 for parent in p.parents:
  if not stat.S_ISDIR(parent.lstat().st_mode):raise RuntimeError('Unsafe evidence ancestor')
 before=p.lstat()
 if not stat.S_ISREG(before.st_mode) or before.st_uid!=307469:raise RuntimeError('Unsafe evidence member')
 if retain and before.st_size>64*1024**2:raise RuntimeError('Metadata exceeds bound')
 digest,chunks=hashlib.sha256(),[]
 signature=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 with os.fdopen(os.open(str(p),os.O_RDONLY|os.O_NOFOLLOW),'rb') as handle:
  if signature(os.fstat(handle.fileno()))!=signature(before):raise RuntimeError('Evidence changed at open')
  for part in iter(lambda:handle.read(8*1024**2),b''):
   digest.update(part)
   if retain:chunks.append(part)
  if signature(os.fstat(handle.fileno()))!=signature(before):raise RuntimeError('Evidence changed during read')
 if signature(p.lstat())!=signature(before):raise RuntimeError('Evidence path changed')
 reference=dict(sha256=digest.hexdigest(),size_bytes=before.st_size)
 if expected is not None and reference['sha256']!=expected:raise RuntimeError('Evidence digest differs')
 return reference,b''.join(chunks)
bundle=Path(BUNDLE);control=Path(CONTROL);output=Path(OUTPUT);root=Path(NEW_EVIDENCE)
_,raw=checked(bundle/'PLAN.json',PLAN_SHA);plan=json.loads(raw)
if plan['run_id']!=JOB or plan['worker_output']!=str(output):raise RuntimeError('Wrong stop scope')
_,raw=checked(control/'live-binding.json',BINDING_SHA);binding=json.loads(raw)
if binding['job']!=JOB or binding['uid']!=UID or binding['run_id']!=JOB or binding['plan_sha256']!=PLAN_SHA:raise RuntimeError('Wrong bound Job')
checked(bundle/'lifecycle_core.py',CORE_SHA)
sys.path.insert(0,str(bundle))
spec=importlib.util.spec_from_file_location('exact_stop_core',str(bundle/'lifecycle_core.py'))
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
core.MODE='supervise'
original_kube=core.kube
def stop_only(*args):
 if args[0] not in ('get','logs','delete'):raise RuntimeError('Stop cannot create, patch or launch')
 return original_kube(*args)
core.kube=stop_only
job=core.get_job(JOB);core.verify(job,binding,False)
parent=root.parent
if parent.stat().st_uid!=307469 or not os.access(str(parent),os.W_OK):raise RuntimeError('Evidence parent not owned/writable')
for path in (parent,)+tuple(parent.parents):
 if not stat.S_ISDIR(path.lstat().st_mode):raise RuntimeError('Unsafe stop evidence parent')
fs=os.statvfs(str(parent))
if fs.f_bavail*fs.f_frsize<16*1024**2:raise RuntimeError('Stop evidence lacks headroom')
root.mkdir(mode=0o700)
def published():
 records=[]
 for item in plan['fits']:
  relative=Path('fits')/item['candidate_id']/('seed-'+str(item['seed']))
  path=output/relative/'FIT_COMPLETE.json'
  if not os.path.lexists(str(path)):continue
  reference,raw=checked(path);row=json.loads(raw)
  checkpoint=relative/'state/checkpoint.pt'
  if (any(row.get(k)!=v for k,v in item.items()) or row.get('checkpoint_path')!=checkpoint.as_posix()
      or row.get('fitting_contract_sha256')!=plan['fitting_contract_sha256']
      or row.get('status')!='terminal_fitting_complete_pending_input_lease_exit'):
   raise RuntimeError('Published fit binding differs')
  actual,_=checked(output/checkpoint,retain=False)
  if actual!=row['checkpoint']:raise RuntimeError('Published checkpoint bytes differ')
  records.append(dict(row,fit_completion_reference=reference))
 return records
def progress():
 path=output/'PROGRESS.jsonl';info=path.lstat()
 if not stat.S_ISREG(info.st_mode) or info.st_uid!=307469:raise RuntimeError('Unsafe progress')
 with os.fdopen(os.open(str(path),os.O_RDONLY|os.O_NOFOLLOW),'rb') as handle:
  offset=max(0,info.st_size-128*1024);handle.seek(offset);raw=handle.read(128*1024)
 lines=raw.splitlines()
 if offset:lines=lines[1:]
 parsed=[]
 for line in lines:
  try:parsed.append(json.loads(line))
  except ValueError:continue
 return dict(offset=offset,tail_sha256=hashlib.sha256(raw).hexdigest(),records=parsed)
before=published()
core.publish(root/'STOP_INTENT.json',dict(status='authorized_user_stop',utc=datetime.now(timezone.utc).isoformat(),
 job=JOB,uid=UID,plan_sha256=PLAN_SHA,binding_sha256=BINDING_SHA,core_sha256=CORE_SHA,
 preserved_fits=before,progress=progress(),automatic_retry=False,new_jobs=0,reason='GPU-first source transition'))
capture=root/'pre_stop';capture.mkdir(mode=0o700)
core.capture(job,capture)
core.verify(core.get_job(JOB),binding,False)
core.cleanup(binding,root)
after=published()
for record in before:
 if record not in after:raise RuntimeError('Published model changed during stop')
cleanup_ref,raw=checked(root/'cleanup.json');cleanup=json.loads(raw)
if cleanup.get('status')!='complete' or cleanup.get('uid')!=UID or cleanup.get('reads',[])[-2:]!=[dict(job_absent=True,owned_pods=0)]*2:raise RuntimeError('Stop absence proof incomplete')
result=dict(status='user_stopped_exact_job',utc=datetime.now(timezone.utc).isoformat(),job=JOB,uid=UID,
 preserved_fits=after,final_progress=progress(),cleanup=cleanup,cleanup_reference=cleanup_ref,
 plan_sha256=PLAN_SHA,binding_sha256=BINDING_SHA,core_sha256=CORE_SHA,
 input_lease_exit_verified=False,old_pipeline_completion_claimed=False,
 numerical_failure_claimed=False,new_jobs=0,automatic_retry=False,output_files_deleted=0)
core.publish(root/'STOP_COMPLETE.json',result)
print('RECEIPT='+json.dumps(result))
'''


def publish(path, value):
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def stop():
    if (EVIDENCE / "STOP_RESULT.json").exists() or (EVIDENCE / "STOP_SESSION_PREFLIGHT.json").exists():
        raise FileExistsError("Consumed stop attempt: reconcile, never repeat")
    fitting_operate.bind_context()
    build, plan = fitting_operate.verified_build()
    created = json.loads(ops.read_regular(ops.BASE / "CREATE_RESULT.json"))["result"]
    launch = json.loads(ops.read_regular(ops.BASE / "LAUNCH_RESULT.json"))
    if created["uid"] != UID or launch["result"]["uid"] != UID or created["job"] != JOB:
        raise ValueError("Stop identities disagree")
    preflight = subprocess.run(["/home/yding1995/.agents/skills/seadragon/scripts/session-preflight.sh",
                               "--mode", "mutate"], capture_output=True, timeout=260)
    lines = [line.split("=", 1)[1] for line in preflight.stdout.decode().splitlines()
             if line.startswith("SEADRAGON_PREFLIGHT_JSON=")]
    if preflight.returncode or len(lines) != 1:
        raise RuntimeError("Stop mutation preflight failed; no mutation performed")
    session = json.loads(lines[0])
    if (session.get("status") != "ready" or session.get("mode") != "mutate"
            or session.get("context") != "yding4_yn-gpu-workload@kubernetes-admin@kubernetes"
            or session.get("namespace") != "yn-gpu-workload"
            or not all(row.get("allowed") is True for row in session["required"].values())):
        raise ValueError("Stop mutation preflight does not admit exact operations")
    publish(EVIDENCE / "STOP_SESSION_PREFLIGHT.json", session)
    receipt = ops.remote(REMOTE_STOP, dict(JOB=JOB, UID=UID, CORE_SHA=CORE_SHA,
        BUNDLE=plan["bundle_root"], CONTROL=plan["control_root"]+"/gpu", OUTPUT=plan["worker_output"],
        NEW_EVIDENCE=REMOTE_EVIDENCE, PLAN_SHA=build["plan_sha256"],
        BINDING_SHA="331f5de03a401974b5bee67cf89fe51499cd26f981d730d4e21b7b7fb13b87f5"), timeout=240)
    receipt["update_accounting"] = account_progress(receipt["preserved_fits"], receipt["final_progress"]["records"])
    receipt["stop_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    publish(EVIDENCE / "STOP_RESULT.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorized-user-stop", action="store_true", required=True)
    parser.parse_args()
    result = stop()
    print(json.dumps({k: v for k, v in result.items() if k not in {"preserved_fits", "final_progress", "cleanup"}}, indent=2))
