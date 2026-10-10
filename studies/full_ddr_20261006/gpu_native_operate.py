"""Serialized, immutable two-phase operations; no automatic retry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import prepare_operate as shared
import qualify_operate as inherited
from gpu_native_launch import FIT_NAME, QUAL_NAME, ROOT, ROOT_NAMES, roots
from prepare_launch import read_regular

PHASE = None


def verified_build():
    base = ROOT / ROOT_NAMES[PHASE]
    build = json.loads(read_regular(base / "BUILD.json"))
    run_id = QUAL_NAME if PHASE == "qualification" else FIT_NAME
    home, control = roots(PHASE)
    if build.get("run_id") != run_id or build.get("phase") != PHASE:
        raise ValueError("Wrong immutable phase build")
    for name, expected in build["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Unsafe bundle member")
        raw = read_regular(base / "bundle" / name)
        if expected != dict(sha256=shared.digest(raw), size_bytes=len(raw)):
            raise ValueError("Immutable phase source differs: "+name)
    for name in ("gpu_native_operate.py", "gpu_native_launch.py"):
        raw = read_regular(ROOT / name)
        if build["files"].get("controller/"+name) != dict(sha256=shared.digest(raw), size_bytes=len(raw)):
            raise ValueError("Current phase controller differs")
    plan_raw = read_regular(base / "bundle/PLAN.json")
    plan = json.loads(plan_raw)
    if (shared.digest(plan_raw) != build["plan_sha256"] or plan["run_id"] != run_id
            or plan["phase"] != PHASE or plan["bundle_root"] != home+"/bundle"
            or plan["control_root"] != control or shared.digest(read_regular(base / "bundle.tar.gz")) != build["archive_sha256"]):
        raise ValueError("Phase plan/archive/root differs")
    return build, plan


def bind_context(phase):
    global PHASE
    PHASE = phase
    home, control = roots(phase)
    run_id = QUAL_NAME if phase == "qualification" else FIT_NAME
    values = dict(BASE=ROOT / ROOT_NAMES[phase], ROOT=ROOT, RUN_ID=run_id, HOME_RUN=home, CONTROL_ROOT=control,
        BRIDGE="/Users/yding4/"+run_id+"-transfer", TRANSFER="/rsrch8/scratch/bcb/yding4/perturbseq/"+run_id+"-transfer")
    for module in (shared, inherited):
        for name, value in values.items():
            setattr(module, name, value)
        module.verified_build = verified_build


PARENT_AUDIT = r'''
import hashlib,json,os,socket,stat,subprocess
from pathlib import Path
from datetime import datetime,timezone
if (os.getuid(),os.getgid())!=(307469,600815):raise RuntimeError('Parent audit writer differs')
if socket.gethostname()!='ldragon5':raise RuntimeError('Exact old supervisor host ldragon5 required; no absence claim from another host')
proc=Path('/proc/2936393')
process=dict(expected_host='ldragon5',pid=2936393,original_process_absent=True)
if proc.exists():
 raw=(proc/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
 ticks=int(fields[19]);cmd=hashlib.sha256((proc/'cmdline').read_bytes()).hexdigest()
 if ticks==175668333 and cmd=='23be8dcd4061bd81cfe1aa00212713399ab51b60df7bbd04f3f6dd9813a5b816' and fields[0]!='Z':raise RuntimeError('Original supervisor still alive; do not activate replacement')
 process['observed_pid_reused_or_zombie']=True
stop=Path('/rsrch8/scratch/bcb/yding4/perturbseq/credo_full_ddr_gse314342_20261006_v1/gpu_native_transition_v1/STOP_COMPLETE.json')
if not stat.S_ISREG(stop.lstat().st_mode) or stop.stat().st_uid!=307469:raise RuntimeError('Owned stop completion absent')
stop_digest=hashlib.sha256(stop.read_bytes()).hexdigest()
env=dict(os.environ,KUBECONFIG='/rsrch8/home/bcb/yding4/.kube/config')
k=['/risapps/noarch/kubectl/1.28.4/bin/kubectl','--context','yding4_yn-gpu-workload@kubernetes-admin@kubernetes','-n','yn-gpu-workload','--request-timeout=30s']
for _ in range(2):
 for kind,extra in [('job',['credo-full-ddr-fitting-20261007-v1','--ignore-not-found=true']),('pods',['-l','job-name=credo-full-ddr-fitting-20261007-v1'])]:
  v=subprocess.run(k+['get',kind]+extra+['-o','json'],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=45)
  if v.returncode or (bool(v.stdout.strip()) if kind=='job' else bool(json.loads(v.stdout)['items'])):raise RuntimeError('Old exact Job/Pods remain')
print('RECEIPT='+json.dumps(dict(status='parent_absence_verified',utc=datetime.now(timezone.utc).isoformat(),
 process=process,parent_job_absent=True,parent_owned_pods_absent=True,stop_completion_sha256=stop_digest,remote_writes=False)))
'''


def preflight():
    shared.session_gate()
    _, plan = verified_build()
    # Reuse the exact proof obtained on ldragon5 during this related session.
    # Never infer process absence from a load-balanced different login host.
    raw = read_regular(shared.BASE / "bundle/PARENT_ABSENCE.json")
    parent = json.loads(raw)
    if (shared.digest(raw) != plan["parent_absence_sha256"] or parent.get("status") != "parent_absence_verified"
            or parent.get("process", {}).get("original_process_absent") is not True
            or parent.get("process", {}).get("expected_host") != "ldragon5"):
        raise ValueError("Exact old supervisor/Job absence required")
    shared.record("PARENT_VERIFIED.json", parent)
    result = inherited.preflight()
    if plan["automatic_retries"] != 0:
        raise ValueError("Retry policy changed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=tuple(ROOT_NAMES))
    parser.add_argument("action", choices=("session", "preflight", "stage", "dryrun", "create", "launch"))
    args = parser.parse_args()
    bind_context(args.phase)
    if args.phase == "fitting" and args.action in {"create", "launch"}:
        raise ValueError("Only the qualified phase-success owner can launch fitting")
    if args.action == "session" and args.phase == "fitting":
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
