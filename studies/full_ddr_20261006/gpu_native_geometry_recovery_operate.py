"""Exact failed-fit recovery; no duplicate qualifier or training launch."""
from __future__ import annotations

import json
import sys

import gpu_native_compiler_operate as inherited
import gpu_native_adam_recovery_operate as previous
from gpu_native_geometry_recovery_launch import CPU_NAME, FIT_NAME, QUAL_NAME, RECOVERY, ROOT, ROOT_NAMES, SOURCES, roots
from prepare_launch import read_regular

original_verified_build = inherited.verified_build


def verified_build():
    build, plan = original_verified_build()
    for name in SOURCES:
        raw = read_regular(ROOT / name)
        if build["files"].get("controller/" + name) != dict(sha256=inherited.shared.digest(raw), size_bytes=len(raw)):
            raise ValueError("Current geometry recovery controller differs")
    if plan.get("full_support_geometry_admission_required") is not True:
        raise ValueError("All-support admission is not bound")
    return build, plan


PARENT_AUDIT = r'''
import hashlib,json,os,socket,stat,subprocess
from pathlib import Path
from datetime import datetime,timezone
if (os.getuid(),os.getgid())!=(307469,600815):raise RuntimeError('Geometry recovery audit owner differs')
host=socket.gethostname()
if host!='ldragon5':
 print('RECEIPT='+json.dumps(dict(status='different_approved_backend',host=host,required_host='ldragon5',remote_writes=False)))
 raise SystemExit(0)
def process_absence(pid,ticks,command):
 p=Path('/proc')/str(pid);result=dict(expected_host='ldragon5',pid=pid,original_process_absent=True)
 if p.exists():
  raw=(p/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
  if int(fields[19])==ticks and fields[0]!='Z':
   same=hashlib.sha256((p/'cmdline').read_bytes()).hexdigest()==command
   raise RuntimeError('Exact prior supervisor remains; same command='+str(same))
  result['pid_reused_or_zombie']=True
 return result
qual_process=process_absence(593695,191245791,'9f25f0255848e6fecf6995d4b516e20068f61de2e815af5da1237251b61f2e87')
fit_process=process_absence(677265,191327801,'8c760fc01ae283f908a0f99882c81e573a6c69b7c8a806d6e427cca57f54b128')
def read(path):
 p=Path(path)
 for ancestor in p.parents:
  if not stat.S_ISDIR(ancestor.lstat().st_mode):raise RuntimeError('Unsafe parent evidence ancestor')
 before=p.lstat()
 if not stat.S_ISREG(before.st_mode) or before.st_uid!=307469 or before.st_size>16*1024**2:raise RuntimeError('Unqualified parent evidence')
 with os.fdopen(os.open(str(p),os.O_RDONLY|os.O_NOFOLLOW),'rb') as stream:raw=stream.read(16*1024**2+1)
 after=p.lstat()
 if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise RuntimeError('Parent evidence changed')
 return json.loads(raw),dict(sha256=hashlib.sha256(raw).hexdigest(),size_bytes=len(raw))
stem='/bcb/yding4/perturbseq/credo_full_ddr_gse314342_20261006_v1/'
home=Path('/rsrch8/home'+stem);control=Path('/rsrch8/scratch'+stem)
terminal,tref=read(control/'fitting_gpu_native_v5/gpu/terminal.json')
cleanup,cref=read(control/'fitting_gpu_native_v5/gpu/cleanup.json')
failed,fref=read(home/'fitting_gpu_native_v5/output/FAILED.json')
if (terminal.get('uid')!='eaac6c96-0218-40fb-bab8-d5517c1acec7' or terminal.get('plan_sha256')!='384d73610c839fdb1c667b538526d8acb889592e0e2a119d732754abb590d7c7'
    or terminal.get('status')!='failed' or cleanup.get('uid')!=terminal['uid'] or cleanup.get('status')!='complete'
    or cleanup.get('reads',[])[-2:]!=[dict(job_absent=True,owned_pods=0)]*2
    or failed.get('run_id')!='credo-full-ddr-fitting-gpu-native-20261009-v5'
    or failed.get('scientific_optimizer_updates')!=18 or failed.get('attempted_optimizer_updates')!=19
    or failed.get('completed_fits')!=0 or failed.get('published_fit_optimizer_updates')!=0
    or 'Single full-support geometry exceeds workspace budget' not in failed.get('error','')
    or os.path.lexists(str(home/'fitting_gpu_native_v5/output/COMPLETE.json'))):
 raise RuntimeError('Failed fitting identity, work or cleanup differs')
qterm,qtref=read(control/'gpu_native_qualification_v5/gpu/terminal.json')
qclean,qcref=read(control/'gpu_native_qualification_v5/gpu/cleanup.json')
qdone,qdref=read(home/'gpu_native_qualification_v5/output/COMPLETE.json')
if (qterm.get('uid')!='ba1c3399-810f-424e-8fb4-54db0e1f21f4' or qterm.get('status')!='passed'
    or qterm.get('plan_sha256')!='3fd0d310fc1ebcc465a3d798b591b2d49195900b63c8ce8bcc0f9f082711039b'
    or qdref['sha256']!='1b00a1692c3e73e394f7cf77f27e0590bdafdca78d4b0bda49e052f16511ac9a'
    or not qdone.get('input_lease_exit_verified') or qdone.get('budget_used')!=dict(full=90,partial=12,steps=18,refresh=3,chunks=549)
    or qclean.get('uid')!=qterm['uid'] or qclean.get('status')!='complete'
    or qclean.get('reads',[])[-2:]!=[dict(job_absent=True,owned_pods=0)]*2):
 raise RuntimeError('Successful qualification lineage differs')
launch,lref=read(control/'gpu_native_qualification_v5/gpu/successor-launch.json')
if launch.get('uid')!=terminal['uid'] or launch.get('process',{}).get('pid')!=677265 or launch['process'].get('start_ticks')!=191327801:
 raise RuntimeError('Failed fitting launch process differs')
stop,sref=read(control/'gpu_native_transition_v1/STOP_COMPLETE.json')
if sref['sha256']!='b23fb1681197e501bcd2cc33bfae8463165ad3759d0e9689031d8baf9f0a1411':raise RuntimeError('Original stop proof differs')
env=dict(os.environ,KUBECONFIG='/rsrch8/home/bcb/yding4/.kube/config')
k=['/risapps/noarch/kubectl/1.28.4/bin/kubectl','--context','yding4_yn-gpu-workload@kubernetes-admin@kubernetes','-n','yn-gpu-workload','--request-timeout=30s']
jobs=['credo-full-ddr-gpu-native-qual-20261009-v5','credo-full-ddr-fitting-gpu-native-20261009-v5']
for _ in range(2):
 for name in jobs:
  for kind,extra in [('job',[name,'--ignore-not-found=true']),('pods',['-l','job-name='+name])]:
   r=subprocess.run(k+['get',kind]+extra+['-o','json'],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=45)
   if r.returncode or (bool(r.stdout.strip()) if kind=='job' else bool(json.loads(r.stdout)['items'])):raise RuntimeError('Exact previous Job or Pods remain')
print('RECEIPT='+json.dumps(dict(status='parent_absence_verified',utc=datetime.now(timezone.utc).isoformat(),
 process=fit_process,qualification_process=qual_process,failed_fitting_run_id=jobs[1],failed_fitting_job_uid=terminal['uid'],
 failed_fitting_scientific_updates=18,failed_fitting_attempted_updates=19,failed_fitting_completed_fits=0,
 failed_terminal_reference=tref,failed_cleanup_reference=cref,failed_worker_reference=fref,original_stop_reference=sref,
 successful_qualification_references=dict(terminal=qtref,cleanup=qcref,complete=qdref),failed_launch_reference=lref,
 two_read_old_job_and_pod_absence_verified=True,previous_successor_absent=True,previous_successor_was_launched=True,remote_writes=False)))
'''


def audit_parent():
    observations = []
    for _ in range(6):
        result = inherited.shared.remote(PARENT_AUDIT, timeout=300)
        if result.get("status") == "parent_absence_verified":
            result["different_backend_observations"] = observations
            old = inherited.shared.BASE
            inherited.shared.BASE = RECOVERY
            try:
                inherited.shared.record("PARENT_VERIFIED_RUNTIME.json", result)
            finally:
                inherited.shared.BASE = old
            return result
        if result.get("status") != "different_approved_backend" or result.get("host") not in {"ldragon3", "ldragon4"}:
            raise RuntimeError("Approved supervisor-host resolution failed")
        observations.append(result)
    raise RuntimeError("Exact ldragon5 owner host unavailable; no absence claim or mutation")


def preflight():
    inherited.shared.session_gate()
    _, plan = verified_build()
    raw = read_regular(inherited.shared.BASE / "bundle/PARENT_ABSENCE.json")
    proof = json.loads(raw)
    if (inherited.shared.digest(raw) != plan["parent_absence_sha256"]
            or proof.get("status") != "parent_absence_verified"
            or proof.get("process", {}).get("original_process_absent") is not True
            or proof.get("qualification_process", {}).get("original_process_absent") is not True
            or proof.get("two_read_old_job_and_pod_absence_verified") is not True
            or proof.get("previous_successor_absent") is not True):
        raise ValueError("Exact failed-fit cleanup and owner absence required")
    inherited.shared.record("PARENT_VERIFIED.json", proof)
    return inherited.inherited.preflight()


def bind_context(phase):
    inherited.RECOVERY, inherited.ROOT_NAMES = RECOVERY, ROOT_NAMES
    inherited.QUAL_NAME, inherited.FIT_NAME, inherited.CPU_NAME = QUAL_NAME, FIT_NAME, CPU_NAME
    inherited.roots, inherited.verified_build = roots, verified_build
    inherited.audit_parent, inherited.preflight = audit_parent, preflight
    inherited.compiler_check = previous.compiler_check
    previous.verified_build = verified_build
    inherited.compiler_gate = previous.compiler_gate
    inherited.bind_context(phase)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ROOT_NAMES:
        raise ValueError("Exact geometry recovery phase required")
    bind_context(sys.argv[1])
    inherited.main()
