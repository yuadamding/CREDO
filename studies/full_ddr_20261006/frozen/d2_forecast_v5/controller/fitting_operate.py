"""One serialized immutable fitting launch, never a capacity retry."""
import argparse
import json

import prepare_operate as shared
import qualify_operate as capacity_ops
from fitting_launch import ROOT, RUN_ID, HOME_RUN, CONTROL_ROOT
from prepare_launch import read_regular

BASE = ROOT / "fitting_v1"
BRIDGE = "/Users/yding4/credo-full-ddr-fitting-20261007-v1-transfer"
TRANSFER = "/rsrch8/scratch/bcb/yding4/perturbseq/credo-full-ddr-fitting-20261007-v1-transfer"


def verified_build():
    build, plan = capacity_ops.original_verify()
    for name in ("qualify_launch.py", "qualify_operate.py", "fitting_launch.py", "fitting_operate.py", "fitting_status.py"):
        raw = read_regular(ROOT / name)
        if build["files"].get("controller/"+name) != dict(sha256=shared.digest(raw), size_bytes=len(raw)):
            raise ValueError("Fitting controller differs from immutable source")
    if plan.get("schema") != "credo.all4_ddr.architecture_fitting_execution.v1" or plan.get("maximum_fits") != 9:
        raise ValueError("Not the exact fitting-only plan")
    return build, plan


def bind_context():
    for module in (shared, capacity_ops):
        for key, value in dict(BASE=BASE, ROOT=ROOT, RUN_ID=RUN_ID, HOME_RUN=HOME_RUN,
                              CONTROL_ROOT=CONTROL_ROOT, BRIDGE=BRIDGE, TRANSFER=TRANSFER).items():
            setattr(module, key, value)
        module.verified_build = verified_build


CAPACITY_AUDIT = r'''
import hashlib,json,os,stat,subprocess
from pathlib import Path
from datetime import datetime,timezone
if (os.getuid(),os.getgid())!=(307469,600815):raise RuntimeError('Prerequisite auditor owner differs')
def checked(path,expected=None):
 p=Path(path)
 for a in p.parents:
  if not stat.S_ISDIR(a.lstat().st_mode):raise RuntimeError('Unsafe prerequisite ancestor')
 i=p.lstat()
 if not stat.S_ISREG(i.st_mode) or i.st_uid!=307469 or i.st_size>64*1024**2:raise RuntimeError('Unsafe prerequisite metadata')
 raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest()
 if expected is not None and digest!=expected:raise RuntimeError('Prerequisite identity differs')
 return json.loads(raw),digest
home=Path(PARENT['home']);control=Path(PARENT['control'])
complete,_=checked(home/'output/COMPLETE.json',PARENT['complete_sha256'])
terminal,tsha=checked(control/'terminal.json');cleanup,csha=checked(control/'cleanup.json')
if (complete.get('status')!='qualified_full_ddr_capacity' or complete.get('scientific_optimizer_updates')!=0
 or terminal.get('status')!='passed' or terminal.get('uid')!=PARENT['job_uid']
 or terminal.get('result_complete_sha256')!=PARENT['complete_sha256']
 or cleanup.get('status')!='complete' or cleanup.get('uid')!=PARENT['job_uid']
 or cleanup.get('reads')!=[dict(job_absent=True,owned_pods=0)]*2):raise RuntimeError('Capacity terminal prerequisite unresolved')
_,psha=checked(home/'bundle/PLAN.json',PARENT['plan_sha256'])
support,ssha=checked(home/'output/SUPPORT.json',PARENT['support_sha256'])
if support.get('fitting_series')!=46809 or support.get('fitting_targets')!=12555:raise RuntimeError('Fitting support prerequisite changed')
shapes=[]
for result,bound in zip(complete['results'],PARENT['completed_capacity_shapes']):
 p=result['probes'][-1]
 shape=dict(candidate_id=result['candidate_id'],groups=p['groups'],allocated_bytes=p['peak_allocated_bytes'],reserved_bytes=p['peak_reserved_bytes'])
 if shape!=bound or p['particles']!=512 or p['steps']!=8 or p.get('geometry_backend')!='geomloss_tensorized' or p['groups']<1086:raise RuntimeError('Fitting shape prerequisite differs')
 shapes.append(shape)
if len(shapes)!=3:raise RuntimeError('Capacity coverage incomplete')
env=dict(os.environ,KUBECONFIG='/rsrch8/home/bcb/yding4/.kube/config')
k=['/risapps/noarch/kubectl/1.28.4/bin/kubectl','--context','yding4_yn-gpu-workload@kubernetes-admin@kubernetes','-n','yn-gpu-workload','--request-timeout=30s']
for _ in range(2):
 for kind,args in [('job',[PARENT['run_id'],'--ignore-not-found=true']),('pods',['-l','job-name='+PARENT['run_id']])]:
  v=subprocess.run(k+['get',kind]+args+['-o','json'],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=45)
  if v.returncode or (bool(v.stdout.strip()) if kind=='job' else bool(json.loads(v.stdout)['items'])):raise RuntimeError('Capacity exact cleanup absence not established')
print('RECEIPT='+json.dumps(dict(status='capacity_prerequisite_verified',utc=datetime.now(timezone.utc).isoformat(),
 fitting_plan_sha256=PLAN_SHA,capacity_complete_sha256=PARENT['complete_sha256'],terminal_sha256=tsha,
 cleanup_sha256=csha,support_sha256=ssha,capacity_plan_sha256=psha,completed_shapes=shapes,
 optimizer_memory_previously_qualified=False,new_capacity_probes=0,remote_writes=False)))
'''


def preflight():
    shared.session_gate()
    build, plan = verified_build()
    raw = read_regular(BASE / "bundle/FIT_CAPACITY_BINDING.json")
    if shared.digest(raw) != plan["capacity_binding_sha256"]:
        raise ValueError("Capacity binding differs")
    result = shared.remote(CAPACITY_AUDIT, dict(PARENT=json.loads(raw), PLAN_SHA=build["plan_sha256"]))
    shared.record("CAPACITY_PREREQUISITE_VERIFIED.json", result)
    return capacity_ops.preflight()


def stage():
    build, _ = verified_build()
    prerequisite = json.loads(read_regular(BASE / "CAPACITY_PREREQUISITE_VERIFIED.json"))
    if prerequisite.get("fitting_plan_sha256") != build["plan_sha256"] or prerequisite.get("status") != "capacity_prerequisite_verified":
        raise ValueError("Exact completed capacity prerequisite required")
    return capacity_ops.stage()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("session", "preflight", "stage", "dryrun", "create", "launch"))
    action = parser.parse_args().action
    bind_context()
    result = {"session": shared.session, "preflight": preflight, "stage": stage}.get(action, lambda: shared.action(action))()
    print(json.dumps(result, indent=2, sort_keys=True))
