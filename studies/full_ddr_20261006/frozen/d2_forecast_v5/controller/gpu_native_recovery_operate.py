"""One authorized, disjoint qualification recovery and conditional successor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import prepare_operate as shared
import qualify_operate as inherited
from gpu_native_recovery_launch import FIT_NAME, QUAL_NAME, RECOVERY, ROOT, ROOT_NAMES, roots
from prepare_launch import read_regular

PHASE = None


def verified_build():
    base = ROOT / ROOT_NAMES[PHASE]
    build = json.loads(read_regular(base / "BUILD.json"))
    run_id = QUAL_NAME if PHASE == "qualification" else FIT_NAME
    home, control = roots(PHASE)
    if build.get("run_id") != run_id or build.get("phase") != PHASE:
        raise ValueError("Wrong immutable recovery build")
    for name, expected in build["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Unsafe recovery member")
        raw = read_regular(base / "bundle" / name)
        if expected != dict(sha256=shared.digest(raw), size_bytes=len(raw)):
            raise ValueError("Immutable recovery source differs: " + name)
    for name in ("gpu_native_recovery_operate.py", "gpu_native_recovery_launch.py"):
        raw = read_regular(ROOT / name)
        if build["files"].get("controller/" + name) != dict(sha256=shared.digest(raw), size_bytes=len(raw)):
            raise ValueError("Current recovery controller differs")
    plan_raw = read_regular(base / "bundle/PLAN.json")
    plan = json.loads(plan_raw)
    if (shared.digest(plan_raw) != build["plan_sha256"] or plan["run_id"] != run_id
            or plan["bundle_root"] != home + "/bundle" or plan["control_root"] != control
            or plan["phase"] != PHASE or shared.digest(read_regular(base / "bundle.tar.gz")) != build["archive_sha256"]):
        raise ValueError("Recovery plan/archive/root differs")
    return build, plan


def bind_context(phase):
    global PHASE
    PHASE = phase
    home, control = roots(phase)
    run_id = QUAL_NAME if phase == "qualification" else FIT_NAME
    values = dict(BASE=ROOT / ROOT_NAMES[phase], ROOT=ROOT, RUN_ID=run_id, HOME_RUN=home, CONTROL_ROOT=control,
        BRIDGE="/Users/yding4/" + run_id + "-transfer", TRANSFER="/rsrch8/scratch/bcb/yding4/perturbseq/" + run_id + "-transfer")
    for module in (shared, inherited):
        for name, value in values.items():
            setattr(module, name, value)
        module.verified_build = verified_build


PARENT_AUDIT = r'''
import hashlib,json,os,socket,stat,subprocess
from pathlib import Path
from datetime import datetime,timezone
if (os.getuid(),os.getgid())!=(307469,600815):raise RuntimeError('Recovery audit owner differs')
if socket.gethostname()!='ldragon5':raise RuntimeError('Exact original supervisor host ldragon5 required')
def process_absence(pid,ticks,command):
 p=Path('/proc')/str(pid)
 result=dict(expected_host='ldragon5',pid=pid,original_process_absent=True)
 if p.exists():
  raw=(p/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
  if int(fields[19])==ticks and fields[0]!='Z':
   digest=hashlib.sha256((p/'cmdline').read_bytes()).hexdigest()
   raise RuntimeError('Original supervisor remains or changed its command; absence not established: '+str(pid)+',same_command='+str(digest==command))
  result['pid_reused_or_zombie']=True
 return result
parent=process_absence(2936393,175668333,'23be8dcd4061bd81cfe1aa00212713399ab51b60df7bbd04f3f6dd9813a5b816')
qualifier=process_absence(1713504,184829098,'c209e9da8ace02b37b9e18329d4c5c1a4b609c8c32a4ded2d73cb3e59e1f3e50')
def read(path):
 p=Path(path)
 for ancestor in p.parents:
  if not stat.S_ISDIR(ancestor.lstat().st_mode):raise RuntimeError('Unsafe parent evidence ancestor')
 info=p.lstat()
 if not stat.S_ISREG(info.st_mode) or info.st_uid!=307469 or info.st_size>64*1024**2:raise RuntimeError('Unqualified parent evidence')
 raw=p.read_bytes()
 return json.loads(raw),dict(sha256=hashlib.sha256(raw).hexdigest(),size_bytes=len(raw))
control=Path('/rsrch8/scratch/bcb/yding4/perturbseq/credo_full_ddr_gse314342_20261006_v1/gpu_native_qualification_v1/gpu')
output=Path('/rsrch8/home/bcb/yding4/perturbseq/credo_full_ddr_gse314342_20261006_v1/gpu_native_qualification_v1/output')
terminal,tref=read(control/'terminal.json');cleanup,cref=read(control/'cleanup.json');failed,fref=read(output/'FAILED.json')
if (terminal.get('uid')!='0f1ed046-8dc9-4e05-80af-04b96b5e74d7' or terminal.get('plan_sha256')!='b114ebac44019cfdec89357ed39c0e1e951cf7c4085aee835810e37d31653032'
    or terminal.get('status')!='failed' or cleanup.get('uid')!=terminal['uid'] or cleanup.get('status')!='complete'
    or cleanup.get('reads',[])[-2:]!=[dict(job_absent=True,owned_pods=0)]*2
    or failed.get('run_id')!='credo-full-ddr-gpu-native-qual-20261008-v1' or failed.get('scientific_optimizer_updates')!=0
    or failed.get('budget_used')!=dict(full=11,partial=2,steps=2,refresh=1,chunks=183)):
 raise RuntimeError('Failed qualifier identity, consumed work or cleanup differs')
for name in ('successor-intent.json','successor-create.json','successor-launch.json','successor-failure.json'):
 if os.path.lexists(str(control/name)):raise RuntimeError('Original successor edge is not unused')
stop,sref=read('/rsrch8/scratch/bcb/yding4/perturbseq/credo_full_ddr_gse314342_20261006_v1/gpu_native_transition_v1/STOP_COMPLETE.json')
if sref['sha256']!='b23fb1681197e501bcd2cc33bfae8463165ad3759d0e9689031d8baf9f0a1411':raise RuntimeError('Original stop proof differs')
env=dict(os.environ,KUBECONFIG='/rsrch8/home/bcb/yding4/.kube/config')
k=['/risapps/noarch/kubectl/1.28.4/bin/kubectl','--context','yding4_yn-gpu-workload@kubernetes-admin@kubernetes','-n','yn-gpu-workload','--request-timeout=30s']
jobs=['credo-full-ddr-fitting-20261007-v1','credo-full-ddr-gpu-native-qual-20261008-v1','credo-full-ddr-fitting-gpu-native-20261008-v1']
for _ in range(2):
 for name in jobs:
  for kind,extra in [('job',[name,'--ignore-not-found=true']),('pods',['-l','job-name='+name])]:
   r=subprocess.run(k+['get',kind]+extra+['-o','json'],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=45)
   if r.returncode or (bool(r.stdout.strip()) if kind=='job' else bool(json.loads(r.stdout)['items'])):raise RuntimeError('Exact old Job or Pods remain')
log=control/'pod-00.log';log_tail=None
if os.path.lexists(str(log)):
 info=log.lstat()
 if not stat.S_ISREG(info.st_mode) or info.st_uid!=307469 or info.st_size>2*1024**2:raise RuntimeError('Unqualified failure log')
 with log.open('rb') as f:
  f.seek(max(0,info.st_size-4096));log_tail=f.read().decode(errors='replace')
print('RECEIPT='+json.dumps(dict(status='parent_absence_verified',utc=datetime.now(timezone.utc).isoformat(),process=parent,
 failed_qualification_process=qualifier,failed_qualification_run_id=jobs[1],failed_qualification_job_uid=terminal['uid'],
 failed_qualification_consumed_work=failed['budget_used'],failed_qualification_scientific_updates=0,
 failed_terminal_reference=tref,failed_cleanup_reference=cref,failed_worker_reference=fref,original_stop_reference=sref,
 original_parent_job_absent=True,failed_qualification_job_absent=True,previous_successor_absent=True,
 two_read_old_job_and_pod_absence_verified=True,failure_log_tail=log_tail,remote_writes=False)))
'''


def audit_parent():
    # Read-only proof precedes bundle construction and mutation admission.
    result = shared.remote(PARENT_AUDIT, timeout=300)
    previous_base = shared.BASE
    shared.BASE = RECOVERY
    try:
        shared.record("PARENT_VERIFIED_RUNTIME.json", result)
    finally:
        shared.BASE = previous_base
    return result


def preflight():
    shared.session_gate()
    _, plan = verified_build()
    raw = read_regular(shared.BASE / "bundle/PARENT_ABSENCE.json")
    proof = json.loads(raw)
    if (shared.digest(raw) != plan["parent_absence_sha256"] or proof.get("status") != "parent_absence_verified"
            or proof.get("process", {}).get("original_process_absent") is not True
            or proof.get("failed_qualification_process", {}).get("original_process_absent") is not True
            or proof.get("two_read_old_job_and_pod_absence_verified") is not True
            or proof.get("previous_successor_absent") is not True):
        raise ValueError("Exact old supervisor/Jobs absence required")
    shared.record("PARENT_VERIFIED.json", proof)
    return inherited.preflight()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=tuple(ROOT_NAMES))
    parser.add_argument("action", choices=("audit-parent", "session", "preflight", "stage", "dryrun", "create", "launch"))
    args = parser.parse_args()
    bind_context(args.phase)
    if args.phase == "fitting" and args.action in {"create", "launch", "audit-parent"}:
        raise ValueError("Only the qualified phase-success owner can launch fitting")
    if args.action == "audit-parent":
        result = audit_parent()
    elif args.action == "session" and args.phase == "fitting":
        verified_build()
        proof = json.loads(read_regular(ROOT / ROOT_NAMES["qualification"] / "SESSION_PREFLIGHT.json"))
        shared.record("SESSION_PREFLIGHT.json", proof)
        shared.session_gate()
        result = dict(status="ready", reused_current_related_session=True, remote_mutations=False)
    else:
        result = {"session": shared.session, "preflight": preflight, "stage": inherited.stage}.get(
            args.action, lambda: shared.action(args.action))()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
