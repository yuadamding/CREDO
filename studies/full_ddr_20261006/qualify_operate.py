"""Single fresh H100 capacity lifecycle; no fitting, retry, or prep mutation."""
import argparse
import json
from pathlib import Path
import shlex
import prepare_operate as shared
from qualify_launch import RUN_ID, HOME_RUN, CONTROL_ROOT
from prepare_launch import read_regular

ROOT = Path(__file__).absolute().parent
BASE = ROOT / "capacity_v1"
BRIDGE = "/Users/yding4/credo-full-ddr-capacity-20261007-v1-transfer"
TRANSFER = "/rsrch8/scratch/bcb/yding4/perturbseq/credo-full-ddr-capacity-20261007-v1-transfer"
original_verify = shared.verified_build


def verified_build():
    build, plan = original_verify()
    for name in ("qualify_operate.py", "qualify_launch.py"):
        raw = read_regular(ROOT / name)
        if build["files"].get("controller/" + name) != dict(sha256=shared.digest(raw), size_bytes=len(raw)):
            raise ValueError("Capacity-specific controller differs from frozen source")
    if plan.get("schema") != "credo.all4_ddr.capacity_execution.v1" or plan.get("parameter_updates") != 0:
        raise ValueError("Not the capacity-only plan")
    return build, plan


def bind_context():
    # Isolated CLI process; inherited helpers resolve only these bound task
    # constants. No historical files or active attempt settings are edited.
    for key, value in dict(BASE=BASE, ROOT=ROOT, RUN_ID=RUN_ID, HOME_RUN=HOME_RUN,
                           CONTROL_ROOT=CONTROL_ROOT, BRIDGE=BRIDGE, TRANSFER=TRANSFER).items():
        setattr(shared, key, value)
    shared.verified_build = verified_build


def preflight():
    shared.session_gate()
    build, plan = verified_build()
    publisher = read_regular(BASE / "bundle/deployment.py").decode()
    code = publisher + r'''
import socket
from datetime import datetime,timezone
if (os.getuid(),os.getgid())!=(307469,600815):raise RuntimeError('Publisher owner differs')
r=dict(status='passed',utc=datetime.now(timezone.utc).isoformat(),host=socket.gethostname(),paths={},remote_writes=False,plan_sha256=PLAN_SHA)
for target,fileset in [(CONTROL,'s_bcb-yding4'),(HOME_RUN,'h_bcb-yding4')]:
 root=Path(target)
 if os.path.lexists(root):raise RuntimeError('Capacity attempt exists; reconcile, do not retry')
 for p in (root.parent,)+tuple(root.parent.parents):
  if not stat.S_ISDIR(p.lstat().st_mode):raise RuntimeError('Unsafe ancestor')
 if root.parent.stat().st_uid!=307469 or not os.access(str(root.parent),os.W_OK):raise RuntimeError('Study parent not owned/writable')
 r['paths'][target]=capacity(root.parent,fileset,minimum_bytes=OUTPUT_CAP+2_000_000_000)
if os.path.lexists(TRANSFER):raise RuntimeError('Transfer root exists')
for value in (HOST_PY,'/risapps/noarch/kubectl/1.28.4/bin/kubectl'):
 p=Path(value)
 if p.is_symlink() or not p.is_file() or not os.access(str(p),os.X_OK):raise RuntimeError('Detached tool unavailable')
env=dict(os.environ,KUBECONFIG='/rsrch8/home/bcb/yding4/.kube/config')
info=os.lstat(env['KUBECONFIG'])
if not stat.S_ISREG(info.st_mode) or info.st_uid!=307469 or stat.S_IMODE(info.st_mode)!=0o600:raise RuntimeError('Configuration metadata invalid')
k=['/risapps/noarch/kubectl/1.28.4/bin/kubectl','--context','yding4_yn-gpu-workload@kubernetes-admin@kubernetes','-n','yn-gpu-workload','--request-timeout=30s']
for kind,extra in [('job',[JOB,'--ignore-not-found=true']),('pods',['-l','job-name='+JOB])]:
 v=subprocess.run(k+['get',kind]+extra+['-o','json'],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=45)
 if v.returncode or (bool(v.stdout.strip()) if kind=='job' else bool(json.loads(v.stdout)['items'])):raise RuntimeError('Exact Job/Pod absence not proven')
r['exact_job_absent']=JOB
print('RECEIPT='+json.dumps(r))
'''
    result = shared.remote(code, dict(CONTROL=CONTROL_ROOT, HOME_RUN=HOME_RUN, TRANSFER=TRANSFER,
        JOB=RUN_ID, HOST_PY=shared.HOST_PY, PLAN_SHA=build["plan_sha256"],
        OUTPUT_CAP=plan.get("maximum_new_output_bytes", plan["maximum_output_bytes"])))
    result["bridge"] = shared.remote(r'''
import os,stat,json
from pathlib import Path
p=Path(ROOT)
if os.getuid()!=1896174575 or os.path.lexists(p):raise RuntimeError('Bridge owner/absence differs')
for parent in p.parents:
 if not stat.S_ISDIR(parent.lstat().st_mode):raise RuntimeError('Unsafe bridge ancestor')
if p.parent.stat().st_uid!=os.getuid():raise RuntimeError('Bridge owner differs')
fs=os.statvfs(str(p.parent))
if fs.f_bavail*fs.f_frsize<REQUIRED:raise RuntimeError('Bridge capacity insufficient')
print('RECEIPT='+json.dumps(dict(status='passed',available_bytes=fs.f_bavail*fs.f_frsize)))
''', dict(ROOT=BRIDGE, REQUIRED=2 * (BASE / "bundle.tar.gz").stat().st_size + 100_000_000), bridge=True)
    shared.record("PREFLIGHT.json", result)
    return result


def stage():
    shared.session_gate()
    build, plan = verified_build()
    check = json.loads(read_regular(BASE / "PREFLIGHT.json"))
    if check.get("status") != "passed" or check.get("plan_sha256") != build["plan_sha256"]:
        raise ValueError("Exact successful preflight required")
    shared.record("STAGE_INTENT.json", dict(plan_sha256=build["plan_sha256"], archive_sha256=build["archive_sha256"],
                                           operator_sha256=shared.digest(read_regular(Path(__file__)))))
    result = shared.remote(shared.ARCHIVE_WRITER, dict(ROOT=BRIDGE, UID=1896174575, EXPECTED=build["archive_sha256"]),
                           data=read_regular(BASE / "bundle.tar.gz"), bridge=True)
    shared.record("BRIDGE_STAGE.json", result)
    code = "UID=307469\nROOT=" + repr(TRANSFER) + "\nEXPECTED=" + repr(build["archive_sha256"]) + "\n" + shared.ARCHIVE_WRITER
    command = shared.SECOND_HOP + ["seadragon", shlex.join([shared.HOST_PY, "-c", code])]
    result = shared.remote(r'''
import hashlib,json,stat,subprocess
from pathlib import Path
p=Path(ARCHIVE)
if not stat.S_ISREG(p.lstat().st_mode) or p.stat().st_uid!=1896174575:raise RuntimeError('Bridge archive owner/type differs')
raw=p.read_bytes()
if hashlib.sha256(raw).hexdigest()!=EXPECTED:raise RuntimeError('Bridge archive changed')
v=subprocess.run(COMMAND,input=raw,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=240)
if v.returncode:raise RuntimeError(v.stderr.decode()[-3000:])
lines=[x for x in v.stdout.decode().splitlines() if x.startswith('RECEIPT=')]
if len(lines)!=1:raise RuntimeError('Transfer uncertain')
print(lines[0])
''', dict(ARCHIVE=BRIDGE + "/bundle.tar.gz", EXPECTED=build["archive_sha256"], COMMAND=command), bridge=True)
    shared.record("REMOTE_ARCHIVE_STAGE.json", result)
    code = read_regular(BASE / "bundle/deployment.py").decode() + r'''
def validated_roots(plan):
 if (plan['run_id']!=JOB or plan['control_root']!=CONTROL or plan['bundle_root']!=HOME_RUN+'/bundle'
     or plan['worker_output']!=HOME_RUN+'/output'):raise RuntimeError('Capacity scope differs')
 return Path(CONTROL),Path(HOME_RUN)
p=Path(TRANSFER)/'bundle.tar.gz'
if not stat.S_ISREG(p.lstat().st_mode) or p.stat().st_uid!=307469:raise RuntimeError('Archive owner differs')
raw=p.read_bytes()
payload,inventory=archive_payload(raw,BUILD['archive_sha256'],BUILD['inventory_sha256'])
plan=json.loads(payload['PLAN.json'])
if sha(payload['PLAN.json'])!=BUILD['plan_sha256']:raise RuntimeError('Plan differs')
for root,fileset in zip(validated_roots(plan),('s_bcb-yding4','h_bcb-yding4')):
 if os.path.lexists(root):raise RuntimeError('Capacity attempt exists; no overwrite')
 for parent in (root.parent,)+tuple(root.parent.parents):
  if not stat.S_ISDIR(parent.lstat().st_mode):raise RuntimeError('Unsafe parent')
 if root.parent.stat().st_uid!=307469:raise RuntimeError('Parent owner differs')
 capacity(root.parent,fileset,minimum_bytes=OUTPUT_CAP+2_000_000_000)
 permission_canary(root.parent)
 receipt_link_canary(root.parent)
print('RECEIPT='+json.dumps(publish_archive(raw,BUILD)))
'''
    result = shared.remote(code, dict(TRANSFER=TRANSFER, BUILD=build, JOB=RUN_ID, CONTROL=CONTROL_ROOT,
                                      HOME_RUN=HOME_RUN,
                                      OUTPUT_CAP=plan.get("maximum_new_output_bytes", plan["maximum_output_bytes"])))
    shared.record("DEPLOYMENT.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("session", "preflight", "stage", "dryrun", "create", "launch"))
    action = parser.parse_args().action
    bind_context()
    result = {"session": shared.session, "preflight": preflight, "stage": stage}.get(action, lambda: shared.action(action))()
    print(json.dumps(result, indent=2, sort_keys=True))
