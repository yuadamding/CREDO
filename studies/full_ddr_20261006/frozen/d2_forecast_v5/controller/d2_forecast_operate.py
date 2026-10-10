"""Single new D2 inference attempt; no automatic scoring, retries or duplicates."""
import argparse
import json

import prepare_operate as shared
import qualify_operate as inherited
from d2_forecast_launch import ROOT, BASE, RUN_ID, HOME_RUN, CONTROL_ROOT, CONTROLLERS
from prepare_launch import read_regular

BRIDGE = "/Users/yding4/credo-full-ddr-d2-forecast-20261010-v5-transfer"
TRANSFER = "/rsrch8/scratch/bcb/yding4/perturbseq/credo-full-ddr-d2-forecast-20261010-v5-transfer"
original_verify = inherited.original_verify


def process_reconciliation(recovery):
    """Only lifecycle identity crosses the SSH argv boundary, never bank lists."""
    result = {key: recovery[key] for key in ("job", "uid", "host", "process")}
    if len(json.dumps(result).encode()) > 4096:
        raise ValueError("Unexpectedly large lifecycle reconciliation")
    return result


def preflight():
    shared.session_gate()
    _, plan = verified_build()
    handoff = json.loads(read_regular(BASE / "bundle/D2_HANDOFF.json"))
    recovery_raw = read_regular(BASE / "bundle/FAILED_V3_RECONCILIATION.json")
    if shared.digest(recovery_raw) != plan["failed_v3_reconciliation_sha256"]:
        raise ValueError("Failed attempt recovery proof changed")
    recovery = process_reconciliation(json.loads(recovery_raw))
    result = shared.remote(r'''
import hashlib,json,os,stat,subprocess,socket
from pathlib import Path
from datetime import datetime,timezone
p=Path(PARENT['output_root'])/'COMPLETE.json'
if not stat.S_ISREG(p.lstat().st_mode) or p.stat().st_uid!=307469 or p.stat().st_size>16*1024**2:
 raise RuntimeError('Parent completion type/owner differs')
raw=p.read_bytes()
if hashlib.sha256(raw).hexdigest()!=PARENT['completion_sha256']:raise RuntimeError('Parent completion changed')
c=json.loads(raw)
if c.get('scientific_training_complete') is not True or c.get('assessment_performed') is not False:
 raise RuntimeError('Parent fitting stage differs')
env=dict(os.environ,KUBECONFIG='/rsrch8/home/bcb/yding4/.kube/config')
k=['/risapps/noarch/kubectl/1.28.4/bin/kubectl','--context','yding4_yn-gpu-workload@kubernetes-admin@kubernetes','-n','yn-gpu-workload','--request-timeout=30s']
for job in (PARENT['run_id'],RECOVERY['job']):
 for kind,extra in [('job',[job,'--ignore-not-found=true']),('pods',['-l','job-name='+job])]:
  v=subprocess.run(k+['get',kind]+extra+['-o','json'],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=45)
  if v.returncode or (bool(v.stdout.strip()) if kind=='job' else bool(json.loads(v.stdout)['items'])):
   raise RuntimeError('Exact parent/failed Job/Pod absence not established')
if socket.gethostname()!=RECOVERY['host']:raise RuntimeError('Prior process must be reconciled on its bound controller host')
process=RECOVERY['process']['bound']
p=Path('/proc/%d'%process['pid'])
if p.exists():
 fields=(p/'stat').read_text().rsplit(')',1)[1].split()
 actual=dict(pid=process['pid'],start_ticks=int(fields[19]),pgrp=int(fields[2]),session=int(fields[3]),tty=int(fields[4]),cmdline_sha256=hashlib.sha256((p/'cmdline').read_bytes()).hexdigest())
 if actual==process and fields[0]!='Z':raise RuntimeError('Prior supervisor remains live')
print('RECEIPT='+json.dumps(dict(status='completed_parent_and_exact_job_absence_verified',utc=datetime.now(timezone.utc).isoformat(),parent=PARENT,failed_job=RECOVERY['job'],failed_uid=RECOVERY['uid'],prior_supervisor_absent=True,recovery_sha256=RECOVERY_SHA,remote_writes=False)))
''', dict(PARENT=handoff["parent"], RECOVERY=recovery, RECOVERY_SHA=plan["failed_v3_reconciliation_sha256"]))
    shared.record("PARENT_VERIFIED.json", result)
    return inherited.preflight()


def verified_build():
    build, plan = original_verify()
    for name in CONTROLLERS:
        raw = read_regular(ROOT / name)
        if build["files"].get("controller/"+name) != dict(sha256=shared.digest(raw), size_bytes=len(raw)):
            raise ValueError("Frozen D2 operation source differs: " + name)
    if plan.get("schema") != "credo.full_ddr.d2_forecast_execution.v1" or plan.get("parameter_updates") != 0:
        raise ValueError("Not the bound inference-only plan")
    return build, plan


def bind_context():
    values = dict(ROOT=ROOT, BASE=BASE, RUN_ID=RUN_ID, HOME_RUN=HOME_RUN,
                  CONTROL_ROOT=CONTROL_ROOT, BRIDGE=BRIDGE, TRANSFER=TRANSFER)
    for module in (shared, inherited):
        for key, value in values.items():
            setattr(module, key, value)
        module.verified_build = verified_build


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("session", "preflight", "stage", "dryrun", "create", "launch"))
    action = parser.parse_args().action
    bind_context()
    result = {"session": shared.session, "preflight": preflight,
              "stage": inherited.stage}.get(action, lambda: shared.action(action))()
    print(json.dumps(result, indent=2, sort_keys=True))
