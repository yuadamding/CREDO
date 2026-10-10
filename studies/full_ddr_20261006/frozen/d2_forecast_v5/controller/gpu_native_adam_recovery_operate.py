"""Serialized v5 Adam recovery using the existing qualified compiler proof."""
from __future__ import annotations

import json
import sys

import gpu_native_compiler_operate as inherited
import gpu_native_compiler_reuse as reuse
from gpu_native_adam_recovery_launch import CPU_NAME, FIT_NAME, QUAL_NAME, RECOVERY, ROOT, ROOT_NAMES, SOURCES, roots
from prepare_launch import read_regular

original_verified_build = inherited.verified_build


def verified_build():
    build, plan = original_verified_build()
    for name in SOURCES:
        raw = read_regular(ROOT / name)
        if build["files"].get("controller/" + name) != dict(sha256=inherited.shared.digest(raw), size_bytes=len(raw)):
            raise ValueError("Current Adam recovery controller differs")
    return build, plan


PARENT_AUDIT = r'''
import hashlib,json,os,socket,stat,subprocess
from pathlib import Path
from datetime import datetime,timezone
if (os.getuid(),os.getgid())!=(307469,600815):raise RuntimeError('Adam recovery audit owner differs')
host=socket.gethostname()
if host!='ldragon5':
 print('RECEIPT='+json.dumps(dict(status='different_approved_backend',host=host,required_host='ldragon5',remote_writes=False)))
 raise SystemExit(0)
p=Path('/proc/197662');process=dict(expected_host='ldragon5',pid=197662,original_process_absent=True)
if p.exists():
 raw=(p/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
 if int(fields[19])==190840164 and fields[0]!='Z':
  same=hashlib.sha256((p/'cmdline').read_bytes()).hexdigest()=='6ddc45ea3175b61dd90a18476fc767f6899fc728bb83d9901cdac55ac6aed0e4'
  raise RuntimeError('Exact Adam-failed supervisor remains; same command='+str(same))
 process['pid_reused_or_zombie']=True
def read(path):
 p=Path(path)
 for ancestor in p.parents:
  if not stat.S_ISDIR(ancestor.lstat().st_mode):raise RuntimeError('Unsafe failed-parent evidence ancestor')
 before=p.lstat()
 if not stat.S_ISREG(before.st_mode) or before.st_uid!=307469 or before.st_size>16*1024**2:raise RuntimeError('Unqualified failed-parent evidence')
 with os.fdopen(os.open(str(p),os.O_RDONLY|os.O_NOFOLLOW),'rb') as stream:raw=stream.read(16*1024**2+1)
 after=p.lstat()
 if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise RuntimeError('Failed-parent evidence changed')
 return json.loads(raw),dict(sha256=hashlib.sha256(raw).hexdigest(),size_bytes=len(raw))
prefix='/rsrch8/scratch/bcb/yding4/perturbseq/credo_full_ddr_gse314342_20261006_v1/'
control=Path(prefix+'gpu_native_qualification_v4/gpu')
output=Path('/rsrch8/home/bcb/yding4/perturbseq/credo_full_ddr_gse314342_20261006_v1/gpu_native_qualification_v4/output')
terminal,tref=read(control/'terminal.json');cleanup,cref=read(control/'cleanup.json');failed,fref=read(output/'FAILED.json')
if (terminal.get('uid')!='6f0f05f4-bfa3-4bd9-b255-b24d46d748ea' or terminal.get('plan_sha256')!='f398af95da33ab9e46420b36aca1f3861c4218207d4bf93c5158a912ecf1c81f'
    or terminal.get('status')!='failed' or cleanup.get('uid')!=terminal['uid'] or cleanup.get('status')!='complete'
    or cleanup.get('reads',[])[-2:]!=[dict(job_absent=True,owned_pods=0)]*2
    or failed.get('run_id')!='credo-full-ddr-gpu-native-qual-20261009-v4' or failed.get('scientific_optimizer_updates')!=0
    or failed.get('budget_used')!=dict(full=50,partial=6,steps=10,refresh=2,chunks=366)
    or 'disposable Adam parameters allclose failed: trunk.0.weight' not in failed.get('error','') or os.path.lexists(str(output/'COMPLETE.json'))):
 raise RuntimeError('Failed-parent Adam, consumed-work or cleanup identity differs')
for name in ('successor-intent.json','successor-create.json','successor-launch.json','successor-failure.json'):
 if os.path.lexists(str(control/name)):raise RuntimeError('Previous scientific successor edge was used')
stop,sref=read(prefix+'gpu_native_transition_v1/STOP_COMPLETE.json')
if sref['sha256']!='b23fb1681197e501bcd2cc33bfae8463165ad3759d0e9689031d8baf9f0a1411':raise RuntimeError('Original stop proof differs')
env=dict(os.environ,KUBECONFIG='/rsrch8/home/bcb/yding4/.kube/config')
k=['/risapps/noarch/kubectl/1.28.4/bin/kubectl','--context','yding4_yn-gpu-workload@kubernetes-admin@kubernetes','-n','yn-gpu-workload','--request-timeout=30s']
jobs=['credo-full-ddr-gpu-native-qual-20261009-v4','credo-full-ddr-fitting-gpu-native-20261009-v4']
for _ in range(2):
 for name in jobs:
  for kind,extra in [('job',[name,'--ignore-not-found=true']),('pods',['-l','job-name='+name])]:
   r=subprocess.run(k+['get',kind]+extra+['-o','json'],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=45)
   if r.returncode or (bool(r.stdout.strip()) if kind=='job' else bool(json.loads(r.stdout)['items'])):raise RuntimeError('Exact previous Job or Pods remain')
print('RECEIPT='+json.dumps(dict(status='parent_absence_verified',utc=datetime.now(timezone.utc).isoformat(),process=process,
 failed_qualification_process=process,failed_qualification_run_id=jobs[0],failed_qualification_job_uid=terminal['uid'],
 failed_qualification_consumed_work=failed['budget_used'],failed_qualification_scientific_updates=0,
 failed_terminal_reference=tref,failed_cleanup_reference=cref,failed_worker_reference=fref,original_stop_reference=sref,
 two_read_old_job_and_pod_absence_verified=True,previous_successor_absent=True,remote_writes=False)))
'''


def audit_parent():
    observations = []
    for _ in range(6):
        result = inherited.shared.remote(PARENT_AUDIT, timeout=300)
        if result.get("status") == "parent_absence_verified":
            result["different_backend_observations"] = observations
            previous_base = inherited.shared.BASE
            inherited.shared.BASE = RECOVERY
            try:
                inherited.shared.record("PARENT_VERIFIED_RUNTIME.json", result)
            finally:
                inherited.shared.BASE = previous_base
            return result
        if result.get("status") != "different_approved_backend" or result.get("host") not in {"ldragon3", "ldragon4"}:
            raise RuntimeError("Approved supervisor-host resolution failed")
        observations.append(result)
    raise RuntimeError("Exact ldragon5 supervisor host unavailable; no absence claim or mutation")


def compiler_check():
    raise RuntimeError("v5 reuses the completed compiler prerequisite; no new compiler Job is authorized")


def compiler_gate():
    _, plan = verified_build()
    if plan.get("compiler_preflight_import") != reuse.import_binding():
        raise ValueError("Bound compiler import differs")
    proof = json.loads(read_regular(ROOT / "gpu_native_qualification_v4/COMPILER_CHECK_RESULT.json"))
    live_gate = inherited.LIVE_COMPILER_GATE.replace("print('RECEIPT='", """cleanup,cleanup_sha=read(Path(CLEANUP))
if (cleanup_sha!=CLEANUP_SHA or cleanup.get('uid')!=UID or cleanup.get('status')!='complete'
    or cleanup.get('reads',[])[-2:]!=[dict(job_absent=True,owned_pods=0)]*2):raise RuntimeError('Imported compiler cleanup differs')
print('RECEIPT='""")
    result = inherited.shared.remote(live_gate, dict(
        GATE=plan["compiler_preflight_gate_path"], EXPECTED=proof,
        SPEC=reuse.OLD_SPEC, ARCHIVE=reuse.tool.ARCHIVE_SHA, IMAGE=plan["image"], PLAN=reuse.OLD_PLAN,
        CLEANUP=reuse.import_binding()["cleanup_source_path"], CLEANUP_SHA=reuse.OLD_CLEANUP_SHA, UID=reuse.OLD_UID))
    if result.get("gate_sha256") != reuse.OLD_GATE_SHA or result.get("compiler_complete_sha256") != reuse.OLD_COMPLETE_SHA:
        raise ValueError("Live compiler prerequisite bytes changed")
    return dict(result, imported_completed_prerequisite=True, no_new_static_job=True)


def bind_context(phase):
    inherited.RECOVERY, inherited.ROOT_NAMES = RECOVERY, ROOT_NAMES
    inherited.QUAL_NAME, inherited.FIT_NAME, inherited.CPU_NAME = QUAL_NAME, FIT_NAME, CPU_NAME
    inherited.roots, inherited.verified_build = roots, verified_build
    inherited.audit_parent, inherited.compiler_check, inherited.compiler_gate = audit_parent, compiler_check, compiler_gate
    inherited.bind_context(phase)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ROOT_NAMES:
        raise ValueError("Exact Adam recovery phase required")
    bind_context(sys.argv[1])
    inherited.main()
