"""One serialized, source-changing recovery. No parent mutation or automatic retry."""
import argparse
import json
import qualify_operate as prior
import prepare_operate as shared
from prepare_launch import read_regular
from capacity_relaunch import ROOT, RUN_ID, HOME_RUN, CONTROL_ROOT

BASE = ROOT / "capacity_v2"
BRIDGE = "/Users/yding4/credo-full-ddr-capacity-20261007-v2-transfer"
TRANSFER = "/rsrch8/scratch/bcb/yding4/perturbseq/credo-full-ddr-capacity-20261007-v2-transfer"


def verified_build():
    build, plan = prior.original_verify()
    for name in ("qualify_launch.py", "qualify_operate.py", "capacity_relaunch.py",
                 "capacity_reoperate.py", "capacity_restatus.py"):
        raw = read_regular(ROOT / name)
        if build["files"].get("controller/" + name) != dict(sha256=shared.digest(raw), size_bytes=len(raw)):
            raise ValueError("Recovery controller differs from frozen source")
    if plan.get("schema") != "credo.all4_ddr.capacity_execution.v2" or plan.get("parameter_updates") != 0:
        raise ValueError("Not the capacity-only recovery plan")
    return build, plan


def bind_context():
    for module in (shared, prior):
        for key, value in dict(BASE=BASE, ROOT=ROOT, RUN_ID=RUN_ID, HOME_RUN=HOME_RUN,
                              CONTROL_ROOT=CONTROL_ROOT, BRIDGE=BRIDGE, TRANSFER=TRANSFER).items():
            setattr(module, key, value)
        module.verified_build = verified_build


PARENT_AUDIT = r'''
import hashlib,json,os,socket,stat,subprocess
from pathlib import Path
from datetime import datetime,timezone
if socket.gethostname()!=PARENT['supervisor_host']:raise RuntimeError('Parent process audit requires its bound host')
if (os.getuid(),os.getgid())!=(307469,600815):raise RuntimeError('Parent auditor owner differs')
def checked(path,expected):
 p=Path(path)
 for ancestor in p.parents:
  if not stat.S_ISDIR(ancestor.lstat().st_mode):raise RuntimeError('Unsafe parent evidence ancestor')
 info=p.lstat()
 if not stat.S_ISREG(info.st_mode) or info.st_uid!=307469 or info.st_size>100*1024**2:raise RuntimeError('Unsafe parent evidence')
 raw=p.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=expected:raise RuntimeError('Parent evidence changed: '+str(p))
 return raw
home=Path(PARENT['parent_home']);control=Path(PARENT['parent_control'])
checked(PARENT['parent_transfer_archive'],PARENT['parent_archive_sha256'])
checked(home/'bundle/PLAN.json',PARENT['parent_plan_sha256'])
inventory=json.loads(checked(home/'bundle/INVENTORY.json',PARENT['parent_inventory_sha256']))
for name,item in inventory.items():
 rel=Path(name)
 if rel.is_absolute() or '..' in rel.parts or rel.as_posix()!=name:raise RuntimeError('Unsafe parent inventory')
 checked(home/'bundle'/rel,item['sha256'])
evidence={}
for name,digest in PARENT['evidence_sha256'].items():
 prefix,relative=name.split('/',1)
 evidence[name]=checked((home/'output' if prefix=='output' else control)/relative,digest)
failed=json.loads(evidence['output/FAILED.json'])
terminal=json.loads(evidence['control/terminal.json'])
cleanup=json.loads(evidence['control/cleanup.json'])
if (failed.get('status')!='failed' or failed.get('parameter_updates')!=0
 or terminal.get('status')!='failed' or terminal.get('uid')!=PARENT['parent_job_uid']
 or cleanup.get('status')!='complete' or cleanup.get('uid')!=PARENT['parent_job_uid']
 or (home/'output/COMPLETE.json').exists()):raise RuntimeError('Parent terminal scope differs')
rows=[json.loads(line) for line in evidence['output/PROGRESS.jsonl'].decode().splitlines()]
probes=[row for row in rows if row.get('phase')=='capacity_probe']
if len(probes)!=7 or any(row.get('candidate_id')!='ddr_mlp128_e8' or row.get('parameter_updates')!=0 for row in probes):raise RuntimeError('Parent probe accounting differs')
if 'checkpoint_geometry_mass_loss' not in evidence['control/pod-00.log'].decode():raise RuntimeError('Failure attribution absent')
process=PARENT['supervisor'];proc=Path('/proc')/str(process['pid'])
if proc.exists():
 raw=(proc/'stat').read_text().rsplit(')',1)[1].split()
 if int(raw[19])==process['start_ticks']:raise RuntimeError('Parent supervisor remains alive; no relaunch')
env=dict(os.environ,KUBECONFIG='/rsrch8/home/bcb/yding4/.kube/config')
k=['/risapps/noarch/kubectl/1.28.4/bin/kubectl','--context','yding4_yn-gpu-workload@kubernetes-admin@kubernetes','-n','yn-gpu-workload','--request-timeout=30s']
for _ in range(2):
 for args in [['get','job',PARENT['parent_run_id'],'--ignore-not-found=true','-o','json'],['get','pods','-l','job-name='+PARENT['parent_run_id'],'-o','json']]:
  v=subprocess.run(k+args,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=45)
  if v.returncode or (bool(v.stdout.strip()) if args[1]=='job' else bool(json.loads(v.stdout)['items'])):raise RuntimeError('Parent exact Job/Pod absence unproven')
print('RECEIPT='+json.dumps(dict(status='parent_verified',utc=datetime.now(timezone.utc).isoformat(),
 host=socket.gethostname(),recovery_parent_sha256=PARENT_SHA,new_plan_sha256=PLAN_SHA,
 parent_job_uid=PARENT['parent_job_uid'],parent_process_absent=True,parent_job_and_pods_absent_twice=True,
 parent_bundle_inventory_verified=True,parent_evidence_sha256=PARENT['evidence_sha256'],remote_writes=False)))
'''


def preflight():
    shared.session_gate()
    build, plan = verified_build()
    raw = read_regular(BASE / "bundle/RECOVERY_PARENT.json")
    if shared.digest(raw) != plan["recovery_parent_sha256"]:
        raise ValueError("Parent binding differs")
    result = shared.remote(PARENT_AUDIT, dict(PARENT=json.loads(raw), PARENT_SHA=shared.digest(raw),
                                             PLAN_SHA=build["plan_sha256"]))
    shared.record("RECOVERY_PARENT_VERIFIED.json", result)
    return prior.preflight()


def stage():
    build, plan = verified_build()
    parent = json.loads(read_regular(BASE / "RECOVERY_PARENT_VERIFIED.json"))
    if parent.get("status") != "parent_verified" or parent.get("new_plan_sha256") != build["plan_sha256"]:
        raise ValueError("Exact verified parent required")
    result = prior.stage()
    shared.record("RECOVERY_DEPLOYMENT_VERIFIED.json", dict(status="deployment_verified",
        plan_sha256=build["plan_sha256"], recovery_parent_sha256=plan["recovery_parent_sha256"],
        previous_receipt_sha256=shared.digest(read_regular(BASE / "RECOVERY_PARENT_VERIFIED.json")),
        deployment_receipt_sha256=shared.digest(read_regular(BASE / "DEPLOYMENT.json"))))
    return result


def action(name):
    build, _ = verified_build()
    deployed = json.loads(read_regular(BASE / "RECOVERY_DEPLOYMENT_VERIFIED.json"))
    if (deployed.get("status") != "deployment_verified" or deployed.get("plan_sha256") != build["plan_sha256"]
            or deployed.get("previous_receipt_sha256") != shared.digest(read_regular(BASE / "RECOVERY_PARENT_VERIFIED.json"))
            or deployed.get("deployment_receipt_sha256") != shared.digest(read_regular(BASE / "DEPLOYMENT.json"))):
        raise ValueError("Recovery deployment chain differs")
    result = shared.action(name)
    if name == "launch":
        shared.record("RECOVERY_LAUNCH_BOUND.json", dict(status="new_lifecycle_bound",
            plan_sha256=build["plan_sha256"],
            previous_receipt_sha256=shared.digest(read_regular(BASE / "RECOVERY_DEPLOYMENT_VERIFIED.json")),
            create_receipt_sha256=shared.digest(read_regular(BASE / "CREATE_RESULT.json")),
            launch_receipt_sha256=shared.digest(read_regular(BASE / "LAUNCH_RESULT.json"))))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("session", "preflight", "stage", "dryrun", "create", "launch"))
    name = parser.parse_args().action
    bind_context()
    result = {"session": shared.session, "preflight": preflight, "stage": stage}.get(name, lambda: action(name))()
    print(json.dumps(result, indent=2, sort_keys=True))
