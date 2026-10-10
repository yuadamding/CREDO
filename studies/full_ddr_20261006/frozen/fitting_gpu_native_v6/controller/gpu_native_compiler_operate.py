"""Serialized one-shot compiler recovery and conditional H100 fitting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import prepare_operate as shared
import qualify_operate as inherited
from gpu_native_compiler_launch import CPU_NAME, FIT_NAME, QUAL_NAME, RECOVERY, ROOT, ROOT_NAMES, roots
from prepare_launch import read_regular

PHASE = None


def verified_build():
    base = ROOT / ROOT_NAMES[PHASE]
    build = json.loads(read_regular(base / "BUILD.json"))
    run_id = QUAL_NAME if PHASE == "qualification" else FIT_NAME
    home, control = roots(PHASE)
    if build.get("run_id") != run_id or build.get("phase") != PHASE:
        raise ValueError("Wrong immutable compiler recovery")
    for name, expected in build["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Unsafe compiler recovery member")
        raw = read_regular(base / "bundle" / name)
        if expected != dict(sha256=shared.digest(raw), size_bytes=len(raw)):
            raise ValueError("Immutable compiler recovery differs: " + name)
    for name in ("gpu_native_compiler_launch.py", "gpu_native_compiler_operate.py"):
        raw = read_regular(ROOT / name)
        if build["files"].get("controller/" + name) != dict(sha256=shared.digest(raw), size_bytes=len(raw)):
            raise ValueError("Current compiler controller differs")
    plan_raw = read_regular(base / "bundle/PLAN.json")
    plan = json.loads(plan_raw)
    if (shared.digest(plan_raw) != build["plan_sha256"] or plan["run_id"] != run_id
            or plan["bundle_root"] != home + "/bundle" or plan["control_root"] != control
            or shared.digest(read_regular(base / "bundle.tar.gz")) != build["archive_sha256"]):
        raise ValueError("Compiler plan/archive/root differs")
    return build, plan


def bind_context(phase):
    global PHASE
    PHASE = phase
    home, control = roots(phase)
    run_id = QUAL_NAME if phase == "qualification" else FIT_NAME
    values = dict(BASE=ROOT / ROOT_NAMES[phase], ROOT=ROOT, RUN_ID=run_id, HOME_RUN=home, CONTROL_ROOT=control,
        BRIDGE="/Users/yding4/" + run_id + "-transfer", TRANSFER="/rsrch8/scratch/bcb/yding4/perturbseq/" + run_id + "-transfer")
    for module in (shared, inherited):
        for key, value in values.items():
            setattr(module, key, value)
        module.verified_build = verified_build


PARENT_AUDIT = r'''
import hashlib,json,os,socket,stat,subprocess
from pathlib import Path
from datetime import datetime,timezone
if (os.getuid(),os.getgid())!=(307469,600815):raise RuntimeError('Compiler recovery audit owner differs')
host=socket.gethostname()
if host!='ldragon4':
 print('RECEIPT='+json.dumps(dict(status='different_approved_backend',host=host,required_host='ldragon4',remote_writes=False)))
 raise SystemExit(0)
p=Path('/proc/3928225');process=dict(expected_host='ldragon4',pid=3928225,original_process_absent=True)
if p.exists():
 raw=(p/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
 if int(fields[19])==216470949 and fields[0]!='Z':
  same=hashlib.sha256((p/'cmdline').read_bytes()).hexdigest()=='a74dfa2efd3c8ac21a192c99d6a097f624d33b1f07c9c0925c7aedea0d9be107'
  raise RuntimeError('Exact compiler-failed supervisor remains; same command='+str(same))
 process['pid_reused_or_zombie']=True
def read(path):
 p=Path(path)
 for ancestor in p.parents:
  if not stat.S_ISDIR(ancestor.lstat().st_mode):raise RuntimeError('Unsafe failed-parent evidence ancestor')
 info=p.lstat()
 if not stat.S_ISREG(info.st_mode) or info.st_uid!=307469 or info.st_size>16*1024**2:raise RuntimeError('Unqualified failed-parent evidence')
 raw=p.read_bytes()
 return json.loads(raw),dict(sha256=hashlib.sha256(raw).hexdigest(),size_bytes=len(raw))
prefix='/rsrch8/scratch/bcb/yding4/perturbseq/credo_full_ddr_gse314342_20261006_v1/'
control=Path(prefix+'gpu_native_qualification_v2/gpu')
output=Path('/rsrch8/home/bcb/yding4/perturbseq/credo_full_ddr_gse314342_20261006_v1/gpu_native_qualification_v2/output')
terminal,tref=read(control/'terminal.json');cleanup,cref=read(control/'cleanup.json');failed,fref=read(output/'FAILED.json')
if (terminal.get('uid')!='248c4a05-274e-424c-8c8d-640173ca03cb' or terminal.get('plan_sha256')!='112841288c3455cbf13c3f83c75caf97b88c268311b669a75c86891a5b923491'
    or terminal.get('status')!='failed' or cleanup.get('uid')!=terminal['uid'] or cleanup.get('status')!='complete'
    or cleanup.get('reads',[])[-2:]!=[dict(job_absent=True,owned_pods=0)]*2
    or failed.get('run_id')!='credo-full-ddr-gpu-native-qual-20261009-v2' or failed.get('scientific_optimizer_updates')!=0
    or failed.get('budget_used')!=dict(full=21,partial=2,steps=4,refresh=1,chunks=183)
    or 'Failed to find C compiler' not in failed.get('error','') or os.path.lexists(str(output/'COMPLETE.json'))):
 raise RuntimeError('Failed-parent compiler, consumed-work or cleanup identity differs')
for name in ('successor-intent.json','successor-create.json','successor-launch.json','successor-failure.json'):
 if os.path.lexists(str(control/name)):raise RuntimeError('Previous scientific successor edge was used')
stop,sref=read(prefix+'gpu_native_transition_v1/STOP_COMPLETE.json')
if sref['sha256']!='b23fb1681197e501bcd2cc33bfae8463165ad3759d0e9689031d8baf9f0a1411':raise RuntimeError('Original stop proof differs')
env=dict(os.environ,KUBECONFIG='/rsrch8/home/bcb/yding4/.kube/config')
k=['/risapps/noarch/kubectl/1.28.4/bin/kubectl','--context','yding4_yn-gpu-workload@kubernetes-admin@kubernetes','-n','yn-gpu-workload','--request-timeout=30s']
jobs=['credo-full-ddr-fitting-20261007-v1','credo-full-ddr-gpu-native-qual-20261008-v1','credo-full-ddr-fitting-gpu-native-20261008-v1',
 'credo-full-ddr-gpu-native-qual-20261009-v2','credo-full-ddr-fitting-gpu-native-20261009-v2']
for _ in range(2):
 for name in jobs:
  for kind,extra in [('job',[name,'--ignore-not-found=true']),('pods',['-l','job-name='+name])]:
   r=subprocess.run(k+['get',kind]+extra+['-o','json'],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=45)
   if r.returncode or (bool(r.stdout.strip()) if kind=='job' else bool(json.loads(r.stdout)['items'])):raise RuntimeError('Exact old Job or Pods remain')
print('RECEIPT='+json.dumps(dict(status='parent_absence_verified',utc=datetime.now(timezone.utc).isoformat(),process=process,
 failed_qualification_process=process,failed_qualification_run_id=jobs[3],failed_qualification_job_uid=terminal['uid'],
 failed_qualification_consumed_work=failed['budget_used'],failed_qualification_scientific_updates=0,
 failed_terminal_reference=tref,failed_cleanup_reference=cref,failed_worker_reference=fref,original_stop_reference=sref,
 two_read_old_job_and_pod_absence_verified=True,previous_successor_absent=True,remote_writes=False)))
'''


def audit_parent():
    # The managed alias can land on any approved backend. Only the exact
    # supervisor host can prove its /proc identity; a different host is unknown.
    observations = []
    for _ in range(6):
        result = shared.remote(PARENT_AUDIT, timeout=300)
        if result.get("status") == "parent_absence_verified":
            result["different_backend_observations"] = observations
            previous_base = shared.BASE
            shared.BASE = RECOVERY
            try:
                shared.record("PARENT_VERIFIED_RUNTIME.json", result)
            finally:
                shared.BASE = previous_base
            return result
        if result.get("status") != "different_approved_backend" or result.get("host") not in {"ldragon3", "ldragon5"}:
            raise RuntimeError("Approved supervisor-host resolution failed")
        observations.append(result)
    raise RuntimeError("Exact ldragon4 supervisor host unavailable; no absence claim or mutation")


def preflight():
    shared.session_gate()
    _, plan = verified_build()
    raw = read_regular(shared.BASE / "bundle/PARENT_ABSENCE.json")
    proof = json.loads(raw)
    if (shared.digest(raw) != plan["parent_absence_sha256"] or proof.get("status") != "parent_absence_verified"
            or proof.get("process", {}).get("original_process_absent") is not True
            or proof.get("two_read_old_job_and_pod_absence_verified") is not True
            or proof.get("previous_successor_absent") is not True):
        raise ValueError("Exact prior-owner and unused-successor absence required")
    shared.record("PARENT_VERIFIED.json", proof)
    return inherited.preflight()


def compiler_check():
    if PHASE != "qualification":
        raise ValueError("Only qualification owns the compiler check")
    shared.session_gate()
    build, plan = verified_build()
    deployed = json.loads(read_regular(shared.BASE / "DEPLOYMENT.json"))
    if deployed.get("plan_sha256") != build["plan_sha256"]:
        raise ValueError("Compiler-check deployment differs")
    shared.record("COMPILER_CHECK_INVOCATION.json", dict(job=CPU_NAME, plan_sha256=build["plan_sha256"]))
    code = r'''
import hashlib,json,os,stat,subprocess
from pathlib import Path
root=Path(ROOT)
if hashlib.sha256((root/'INVENTORY.json').read_bytes()).hexdigest()!=INVENTORY_SHA:raise RuntimeError('Live compiler inventory differs')
inventory=json.loads((root/'INVENTORY.json').read_text())
for name,expected in inventory.items():
 p=root/name;info=p.lstat()
 if not stat.S_ISREG(info.st_mode) or info.st_uid!=307469:raise RuntimeError('Live compiler source type/owner differs')
 raw=p.read_bytes()
 if expected!=dict(sha256=hashlib.sha256(raw).hexdigest(),size_bytes=len(raw)):raise RuntimeError('Live compiler source differs')
if hashlib.sha256((root/'PLAN.json').read_bytes()).hexdigest()!=PLAN_SHA:raise RuntimeError('Compiler plan differs')
env=dict(os.environ,KUBECONFIG='/rsrch8/home/bcb/yding4/.kube/config')
result=subprocess.run([HOST_PY,str(root/'compiler_preflight_control.py')],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
if result.returncode:raise RuntimeError('Compiler prerequisite failed; preserve/reconcile without retry: '+result.stdout.decode()[-2000:]+result.stderr.decode()[-4000:])
value=json.loads(result.stdout.decode().splitlines()[-1])
if value.get('status')!='passed_same_image_compiler_prerequisite':raise RuntimeError('Compiler prerequisite unresolved')
print('RECEIPT='+json.dumps(value))
'''
    result = shared.remote(code, dict(ROOT=shared.HOME_RUN + "/bundle", INVENTORY_SHA=build["inventory_sha256"],
        PLAN_SHA=build["plan_sha256"], HOST_PY=shared.HOST_PY), timeout=900)
    shared.record("COMPILER_CHECK_RESULT.json", result)
    return result


LIVE_COMPILER_GATE = r'''
import hashlib,json,os,stat
from pathlib import Path
path=Path(GATE)
def read(p):
 for parent in p.parents:
  if not stat.S_ISDIR(parent.lstat().st_mode):raise RuntimeError('Unsafe compiler prerequisite ancestor')
 before=p.lstat()
 if not stat.S_ISREG(before.st_mode) or before.st_uid!=307469 or before.st_size>16*1024**2:raise RuntimeError('Unsafe compiler prerequisite file')
 with os.fdopen(os.open(str(p),os.O_RDONLY|os.O_NOFOLLOW),'rb') as stream:
  raw=stream.read(16*1024**2+1)
 after=p.lstat()
 if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise RuntimeError('Compiler prerequisite changed')
 return json.loads(raw),hashlib.sha256(raw).hexdigest()
gate,gsha=read(path)
if gate!=EXPECTED:raise RuntimeError('Live compiler prerequisite differs from qualified local receipt')
complete,csha=read(path.parent/'COMPLETE.json')
required=dict(status='compiler_dependency_surface_verified',compiler_spec_sha256=SPEC,archive_sha256=ARCHIVE,
 native_builds=2,python_extension_loaded=True,cpp_library_loaded=True,cohort_access=False,scientific_optimizer_updates=0,
 python_version='3.12.13',torch_package_version='2.6.0+cu124',triton_version='3.2.0',data_free=True,image=IMAGE,plan_sha256=PLAN)
if csha!=gate['compiler_complete_sha256'] or any(complete.get(k)!=v for k,v in required.items()):raise RuntimeError('Live compiler native/ABI prerequisite differs')
print('RECEIPT='+json.dumps(dict(status='live_compiler_prerequisite_verified',gate_sha256=gsha,compiler_complete_sha256=csha)))
'''


def compiler_gate():
    from compiler_toolchain import ARCHIVE_SHA
    build, plan = verified_build()
    proof = json.loads(read_regular(shared.BASE / "COMPILER_CHECK_RESULT.json"))
    required = dict(qualification_plan_sha256=build["plan_sha256"], status="passed_same_image_compiler_prerequisite",
        image=plan["image"], compiler_spec_sha256=plan["compiler_spec_sha256"], compiler_archive_sha256=ARCHIVE_SHA,
        native_builds=2, data_free=True, GPU_requests=0, visibility_mask="none", scientific_optimizer_updates=0,
        job=CPU_NAME)
    if (any(proof.get(k) != v for k, v in required.items()) or not proof.get("uid")
            or proof.get("cleanup_reads", [])[-2:] != [dict(job_absent=True, owned_pods=0)]*2):
        raise ValueError("No completed same-image compiler prerequisite")
    return shared.remote(LIVE_COMPILER_GATE, dict(GATE=plan["compiler_preflight_gate_path"], EXPECTED=proof,
        SPEC=plan["compiler_spec_sha256"], ARCHIVE=ARCHIVE_SHA, IMAGE=plan["image"], PLAN=build["plan_sha256"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=tuple(ROOT_NAMES))
    parser.add_argument("action", choices=("audit-parent", "session", "preflight", "stage", "compiler", "dryrun", "create", "launch"))
    args = parser.parse_args()
    bind_context(args.phase)
    if args.phase == "fitting" and args.action in {"audit-parent", "compiler", "create", "launch"}:
        raise ValueError("Only the qualified successor owner may create fitting")
    if args.action == "audit-parent":
        result = audit_parent()
    elif args.action == "compiler":
        result = compiler_check()
    elif args.action == "session" and args.phase == "fitting":
        verified_build()
        proof = json.loads(read_regular(ROOT / ROOT_NAMES["qualification"] / "SESSION_PREFLIGHT.json"))
        shared.record("SESSION_PREFLIGHT.json", proof)
        shared.session_gate()
        result = dict(status="ready", reused_current_related_session=True, remote_mutations=False)
    elif args.action in {"create", "launch"}:
        compiler_gate()
        result = shared.action(args.action)
    else:
        result = {"session": shared.session, "preflight": preflight, "stage": inherited.stage}.get(
            args.action, lambda: shared.action(args.action))()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
