"""Correct an unsubmitted compiler-controller owner, preserving all evidence."""
import json
import sys

import gpu_native_compiler_operate as inherited
from gpu_native_compiler_recovery_launch import CPU_NAME, FIT_NAME, QUAL_NAME, RECOVERY, ROOT, ROOT_NAMES, roots
from prepare_launch import read_regular

old_verified_build = inherited.verified_build


def verified_build():
    build, plan = old_verified_build()
    for name in ("gpu_native_compiler_recovery_launch.py", "gpu_native_compiler_recovery_operate.py"):
        raw = read_regular(ROOT / name)
        if build["files"].get("controller/" + name) != dict(sha256=inherited.shared.digest(raw), size_bytes=len(raw)):
            raise ValueError("Current pre-submission correction controller differs")
    return build, plan


AUDIT = r'''
import hashlib,json,os,stat,subprocess
from pathlib import Path
from datetime import datetime,timezone
if (os.getuid(),os.getgid())!=(307469,600815):raise RuntimeError('Presubmission audit owner differs')
prefix='/rsrch8/scratch/bcb/yding4/perturbseq/credo_full_ddr_gse314342_20261006_v1/'
if os.path.lexists(prefix+'gpu_native_qualification_v3/compiler'):raise RuntimeError('Previous compiler controller obtained an owner; reconcile')
for suffix in ('gpu_native_qualification_v3/gpu','fitting_gpu_native_v3/gpu'):
 for name in ('create-intent.json','live-binding.json','create-uncertain.json','activation.json','launch-success.json','successor-intent.json'):
  if os.path.lexists(prefix+suffix+'/'+name):raise RuntimeError('Previous presubmission GPU owner used')
env=dict(os.environ,KUBECONFIG='/rsrch8/home/bcb/yding4/.kube/config')
k=['/risapps/noarch/kubectl/1.28.4/bin/kubectl','--context','yding4_yn-gpu-workload@kubernetes-admin@kubernetes','-n','yn-gpu-workload','--request-timeout=30s']
jobs=['credo-full-ddr-fitting-20261007-v1','credo-full-ddr-gpu-native-qual-20261008-v1','credo-full-ddr-fitting-gpu-native-20261008-v1',
 'credo-full-ddr-gpu-native-qual-20261009-v2','credo-full-ddr-fitting-gpu-native-20261009-v2',
 'credo-full-ddr-compiler-static-20261009-v3','credo-full-ddr-gpu-native-qual-20261009-v3','credo-full-ddr-fitting-gpu-native-20261009-v3']
for _ in range(2):
 for name in jobs:
  for kind,extra in [('job',[name,'--ignore-not-found=true']),('pods',['-l','job-name='+name])]:
   r=subprocess.run(k+['get',kind]+extra+['-o','json'],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=45)
   if r.returncode or (bool(r.stdout.strip()) if kind=='job' else bool(json.loads(r.stdout)['items'])):raise RuntimeError('Exact previous Job/Pod remains')
print('RECEIPT='+json.dumps(dict(status='presubmission_zero_work_absence_verified',utc=datetime.now(timezone.utc).isoformat(),
 prior_compiler_owner_absent=True,two_read_old_job_and_pod_absence_verified=True,previous_successor_absent=True,
 previous_presubmission_jobs_created=0,previous_presubmission_scientific_updates=0,remote_writes=False)))
'''


def audit_parent():
    previous_raw = read_regular(ROOT / "gpu_native_compiler_recovery_v3/PARENT_VERIFIED_RUNTIME.json")
    previous = json.loads(previous_raw)
    if (previous.get("status") != "parent_absence_verified" or previous.get("process", {}).get("original_process_absent") is not True
            or previous.get("failed_qualification_job_uid") != "248c4a05-274e-424c-8c8d-640173ca03cb"):
        raise ValueError("Immutable original supervisor absence proof differs")
    fresh = inherited.shared.remote(AUDIT, timeout=300)
    if fresh.get("status") != "presubmission_zero_work_absence_verified":
        raise RuntimeError("Zero-work replacement is not admitted")
    previous.update(utc=fresh["utc"], presubmission_absence=fresh,
        original_process_absence_reused_sha256=inherited.shared.digest(previous_raw))
    old_base = inherited.shared.BASE
    inherited.shared.BASE = RECOVERY
    try:
        inherited.shared.record("PARENT_VERIFIED_RUNTIME.json", previous)
    finally:
        inherited.shared.BASE = old_base
    return previous


def bind_context(phase):
    inherited.RECOVERY, inherited.ROOT_NAMES = RECOVERY, ROOT_NAMES
    inherited.QUAL_NAME, inherited.FIT_NAME, inherited.CPU_NAME = QUAL_NAME, FIT_NAME, CPU_NAME
    inherited.roots = roots
    inherited.verified_build = verified_build
    inherited.audit_parent = audit_parent
    inherited.bind_context(phase)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ROOT_NAMES:
        raise ValueError("Exact correction phase required")
    bind_context(sys.argv[1])
    inherited.main()
