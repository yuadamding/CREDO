"""Explicit, serialized operations for one frozen all-four-donor preparation Job.

No action on import; no retry loop and no arbitrary remote command interface.
Only this task's immutable bundle, exact new roots and exact Job are in scope.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

from prepare_launch import CONTROL_ROOT, HOME_RUN, RUN_ID, canonical, read_regular

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "preparation_v1"
BRIDGE = "/Users/yding4/credo-full-ddr-prepare-20261006-v1-transfer"
TRANSFER = "/rsrch8/scratch/bcb/yding4/perturbseq/credo-full-ddr-prepare-20261006-v1-transfer"
HOST_PY = "/usr/libexec/platform-python3.6"
BRIDGE_PY = "/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9"
SSH_OPTIONS = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ForwardAgent=no",
               "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=6"]
SECOND_HOP = ["/usr/bin/ssh", "-F", "/Users/yding4/.ssh/config.d/codex-chain/company-seadragon.conf", *SSH_OPTIONS]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def record(name, value):
    if Path(name).name != name:
        raise ValueError("Receipt name must be one component")
    with os.fdopen(os.open(BASE / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb") as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())


def verified_build():
    build = json.loads(read_regular(BASE / "BUILD.json"))
    if build["run_id"] != RUN_ID:
        raise ValueError("Wrong preparation build")
    for name, expected in build["files"].items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != name:
            raise ValueError("Unsafe inventory member")
        raw = read_regular(BASE / "bundle" / path)
        if expected != {"sha256": digest(raw), "size_bytes": len(raw)}:
            raise ValueError("Immutable bundle differs: " + name)
    for name in ("prepare_operate.py", "prepare_launch.py"):
        raw = read_regular(ROOT / name)
        if build["files"].get("controller/" + name) != {"sha256": digest(raw), "size_bytes": len(raw)}:
            raise ValueError("Current controller differs from frozen source: " + name)
    plan = json.loads(read_regular(BASE / "bundle/PLAN.json"))
    if (digest(read_regular(BASE / "bundle/PLAN.json")) != build["plan_sha256"]
            or plan["run_id"] != RUN_ID or plan["control_root"] != CONTROL_ROOT
            or plan["bundle_root"] != HOME_RUN + "/bundle"
            or digest(read_regular(BASE / "bundle.tar.gz")) != build["archive_sha256"]):
        raise ValueError("Bound preparation plan/archive differs")
    return build, plan


def ssh_command(arguments, bridge=False):
    command = shlex.join([str(value) for value in arguments])
    if not bridge:
        command = shlex.join(SECOND_HOP + ["seadragon", command])
    return ["/usr/bin/ssh", *SSH_OPTIONS, "work-mac", command]


def call(arguments, *, data=None, bridge=False, timeout=300):
    result = subprocess.run(ssh_command(arguments, bridge), input=data, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=timeout)
    if result.returncode:
        raise RuntimeError("Exact call failed; reconcile without retry: "
                           + result.stdout.decode(errors="replace")[-3000:]
                           + result.stderr.decode(errors="replace")[-3000:])
    return result.stdout.decode()


def remote(code, values=None, *, data=None, bridge=False, timeout=300):
    prefix = "\n".join(key + "=" + repr(value) for key, value in (values or {}).items()) + "\n"
    raw = call([BRIDGE_PY if bridge else HOST_PY, "-c", prefix + code], data=data,
               bridge=bridge, timeout=timeout)
    lines = [line.split("=", 1)[1] for line in raw.splitlines() if line.startswith("RECEIPT=")]
    if len(lines) != 1:
        raise RuntimeError("Ambiguous receipt; reconcile exact attempt before any further mutation")
    return json.loads(lines[0])


def session_gate():
    value = json.loads(read_regular(BASE / "SESSION_PREFLIGHT.json"))
    if (value.get("status") != "ready" or value.get("mode") != "mutate"
            or value.get("context") != "yding4_yn-gpu-workload@kubernetes-admin@kubernetes"
            or value.get("namespace") != "yn-gpu-workload"
            or not all(row.get("allowed") is True for row in value["required"].values())):
        raise ValueError("Mutation session preflight is not qualified")


def session():
    verified_build()
    result = subprocess.run(["/home/yding1995/.agents/skills/seadragon/scripts/session-preflight.sh",
                             "--mode", "mutate"], capture_output=True, timeout=260)
    lines = [line.split("=", 1)[1] for line in result.stdout.decode().splitlines()
             if line.startswith("SEADRAGON_PREFLIGHT_JSON=")]
    if result.returncode or len(lines) != 1:
        raise RuntimeError("Session preflight failed; no mutation performed")
    record("SESSION_PREFLIGHT.json", json.loads(lines[0]))
    session_gate()
    return {"status": "ready", "remote_mutations": False}


def preflight():
    session_gate()
    build, plan = verified_build()
    publisher = read_regular(BASE / "bundle/deployment.py").decode()
    code = publisher + r'''
import socket
from datetime import datetime,timezone
if (os.getuid(),os.getgid())!=(307469,600815):raise RuntimeError('Publisher owner differs')
r=dict(status='passed',utc=datetime.now(timezone.utc).isoformat(),host=socket.gethostname(),paths={},remote_writes=False,plan_sha256=PLAN_SHA)
for target,fileset in [(CONTROL,'s_bcb-yding4'),(HOME_RUN,'h_bcb-yding4')]:
 root=Path(target)
 if os.path.lexists(root) or os.path.lexists(root.parent):raise RuntimeError('New study preparation parent already exists; reconcile')
 parent=root.parent.parent
 for p in (parent,)+tuple(parent.parents):
  if not stat.S_ISDIR(p.lstat().st_mode):raise RuntimeError('Unsafe ancestor')
 if parent.stat().st_uid!=307469 or not os.access(str(parent),os.W_OK):raise RuntimeError('Project parent not owned/writable')
 r['paths'][target]=capacity(parent,fileset,minimum_bytes=OUTPUT_CAP+2_000_000_000)
if os.path.lexists(TRANSFER):raise RuntimeError('Transfer root exists')
for value in (HOST_PY,'/risapps/noarch/kubectl/1.28.4/bin/kubectl'):
 p=Path(value)
 if p.is_symlink() or not p.is_file() or not os.access(str(p),os.X_OK):raise RuntimeError('Bound detached tool unavailable')
env=dict(os.environ,KUBECONFIG='/rsrch8/home/bcb/yding4/.kube/config')
info=os.lstat(env['KUBECONFIG'])
if not stat.S_ISREG(info.st_mode) or info.st_uid!=307469 or stat.S_IMODE(info.st_mode)!=0o600:raise RuntimeError('Configuration metadata invalid')
k=['/risapps/noarch/kubectl/1.28.4/bin/kubectl','--context','yding4_yn-gpu-workload@kubernetes-admin@kubernetes','-n','yn-gpu-workload','--request-timeout=30s']
for kind,extra in [('job',[JOB,'--ignore-not-found=true']),('pods',['-l','job-name='+JOB])]:
 v=subprocess.run(k+['get',kind]+extra+['-o','json'],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=45)
 if v.returncode or (bool(v.stdout.strip()) if kind=='job' else bool(json.loads(v.stdout)['items'])):raise RuntimeError('Exact Job/Pod absence not established')
r['exact_job_absent']=JOB
print('RECEIPT='+json.dumps(r))
'''
    result = remote(code, dict(CONTROL=CONTROL_ROOT, HOME_RUN=HOME_RUN, TRANSFER=TRANSFER,
        JOB=RUN_ID, HOST_PY=HOST_PY, PLAN_SHA=build["plan_sha256"], OUTPUT_CAP=plan["maximum_new_output_bytes"]))
    result["bridge"] = remote(r'''
import os,stat,json
from pathlib import Path
p=Path(ROOT)
if os.getuid()!=1896174575 or os.path.lexists(p):raise RuntimeError('Bridge owner/absence differs')
for parent in p.parents:
 if not stat.S_ISDIR(parent.lstat().st_mode):raise RuntimeError('Unsafe bridge ancestor')
if p.parent.stat().st_uid!=os.getuid():raise RuntimeError('Bridge parent owner differs')
fs=os.statvfs(str(p.parent))
if fs.f_bavail*fs.f_frsize<REQUIRED:raise RuntimeError('Bridge capacity insufficient')
print('RECEIPT='+json.dumps(dict(status='passed',available_bytes=fs.f_bavail*fs.f_frsize)))
''', dict(ROOT=BRIDGE, REQUIRED=2 * (BASE / "bundle.tar.gz").stat().st_size + 100_000_000), bridge=True)
    record("PREFLIGHT.json", result)
    return result


ARCHIVE_WRITER = r'''
import os,stat,hashlib,sys,json
from pathlib import Path
if os.getuid()!=UID:raise RuntimeError('Archive writer owner differs')
root=Path(ROOT)
for parent in root.parents:
 if not stat.S_ISDIR(parent.lstat().st_mode):raise RuntimeError('Unsafe transfer ancestor')
if root.parent.stat().st_uid!=UID:raise RuntimeError('Transfer parent owner differs')
root.mkdir(mode=0o700)
probe=root/'writer-canary'
with os.fdopen(os.open(str(probe),os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600),'wb') as f:
 f.write(b'owned');f.flush();os.fsync(f.fileno())
if probe.stat().st_uid!=UID or probe.read_bytes()!=b'owned':raise RuntimeError('Transfer canary failed')
probe.unlink()
raw=sys.stdin.buffer.read(100_000_001)
if len(raw)>100_000_000 or hashlib.sha256(raw).hexdigest()!=EXPECTED:raise RuntimeError('Archive bytes differ')
p=root/'bundle.tar.gz'
with os.fdopen(os.open(str(p),os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600),'wb') as f:
 f.write(raw);f.flush();os.fchmod(f.fileno(),0o600);os.fsync(f.fileno())
if p.stat().st_uid!=UID or stat.S_IMODE(p.stat().st_mode)!=0o600 or hashlib.sha256(p.read_bytes()).hexdigest()!=EXPECTED:raise RuntimeError('Archive readback differs')
print('RECEIPT='+json.dumps(dict(status='archive_staged_verified',root=ROOT,sha256=EXPECTED,size_bytes=len(raw))))
'''


def stage():
    session_gate()
    build, plan = verified_build()
    check = json.loads(read_regular(BASE / "PREFLIGHT.json"))
    if check.get("status") != "passed" or check.get("plan_sha256") != build["plan_sha256"]:
        raise ValueError("Exact successful preflight required")
    record("STAGE_INTENT.json", {"plan_sha256": build["plan_sha256"], "archive_sha256": build["archive_sha256"],
                                "operator_sha256": digest(read_regular(Path(__file__)))})
    raw = read_regular(BASE / "bundle.tar.gz")
    result = remote(ARCHIVE_WRITER, dict(ROOT=BRIDGE, UID=1896174575, EXPECTED=build["archive_sha256"]), data=raw, bridge=True)
    record("BRIDGE_STAGE.json", result)
    code = "UID=307469\nROOT=" + repr(TRANSFER) + "\nEXPECTED=" + repr(build["archive_sha256"]) + "\n" + ARCHIVE_WRITER
    command = SECOND_HOP + ["seadragon", shlex.join([HOST_PY, "-c", code])]
    result = remote(r'''
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
    record("REMOTE_ARCHIVE_STAGE.json", result)
    publisher = read_regular(BASE / "bundle/deployment.py").decode()
    code = publisher + r'''
def validated_roots(plan):
 if (plan['run_id']!=JOB or plan['control_root']!=CONTROL or plan['bundle_root']!=HOME_RUN+'/bundle'
     or plan['worker_output']!=HOME_RUN+'/output'):raise RuntimeError('New study root binding differs')
 return Path(CONTROL),Path(HOME_RUN)
p=Path(TRANSFER)/'bundle.tar.gz'
if not stat.S_ISREG(p.lstat().st_mode) or p.stat().st_uid!=307469:raise RuntimeError('Remote archive owner/type differs')
raw=p.read_bytes()
payload,inventory=archive_payload(raw,BUILD['archive_sha256'],BUILD['inventory_sha256'])
plan=json.loads(payload['PLAN.json'])
if sha(payload['PLAN.json'])!=BUILD['plan_sha256']:raise RuntimeError('Plan changed')
roots=validated_roots(plan)
for root,fileset in zip(roots,('s_bcb-yding4','h_bcb-yding4')):
 if os.path.lexists(root.parent):raise RuntimeError('New study parent exists; no overwrite')
 capacity(root.parent.parent,fileset,minimum_bytes=OUTPUT_CAP+2_000_000_000)
for root in roots:
 root.parent.mkdir(mode=0o700)
 permission_canary(root.parent)
 receipt_link_canary(root.parent)
print('RECEIPT='+json.dumps(publish_archive(raw,BUILD)))
'''
    result = remote(code, dict(TRANSFER=TRANSFER, BUILD=build, JOB=RUN_ID, CONTROL=CONTROL_ROOT,
                               HOME_RUN=HOME_RUN, OUTPUT_CAP=plan["maximum_new_output_bytes"]))
    record("DEPLOYMENT.json", result)
    # Duplicate transfer copies are retained until explicitly reconciled cleanup.
    return result


def action(name):
    if name not in {"dryrun", "create", "launch"}:
        raise ValueError("Unsupported action")
    session_gate()
    build, _ = verified_build()
    deployed = json.loads(read_regular(BASE / "DEPLOYMENT.json"))
    if deployed.get("plan_sha256") != build["plan_sha256"] or deployed.get("archive_sha256") != build["archive_sha256"]:
        raise ValueError("Deployment binding differs")
    record(name.upper() + "_INVOCATION.json", {"job": RUN_ID, "plan_sha256": build["plan_sha256"],
                                               "operator_sha256": digest(read_regular(Path(__file__)))})
    code = r'''
import hashlib,json,os,socket,stat,subprocess
from pathlib import Path
root=Path(ROOT)
for ancestor in (root,)+tuple(root.parents):
 if not stat.S_ISDIR(ancestor.lstat().st_mode):raise RuntimeError('Unsafe live bundle ancestor')
inventory_path=root/'INVENTORY.json'
if not stat.S_ISREG(inventory_path.lstat().st_mode) or hashlib.sha256(inventory_path.read_bytes()).hexdigest()!=INVENTORY_SHA:raise RuntimeError('Live inventory differs')
inventory=json.loads(inventory_path.read_text())
actual={}
for p in root.rglob('*'):
 mode=p.lstat().st_mode
 if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):raise RuntimeError('Special bundle member')
 if stat.S_ISREG(mode) and p!=inventory_path:
  raw=p.read_bytes();actual[p.relative_to(root).as_posix()]=dict(sha256=hashlib.sha256(raw).hexdigest(),size_bytes=len(raw))
if actual!=inventory:raise RuntimeError('Live operation payload differs')
if actual['PLAN.json']['sha256']!=PLAN_SHA:raise RuntimeError('Plan inventory identity differs')
env=dict(os.environ,KUBECONFIG='/rsrch8/home/bcb/yding4/.kube/config')
command=[HOST_PY,str(root/'control.py'),ACTION,'--plan',str(root/'PLAN.json')]
value=subprocess.run(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=240)
if value.returncode:raise RuntimeError('Lifecycle failed; retain/reconcile: '+value.stdout.decode()[-4000:]+value.stderr.decode()[-4000:])
lines=value.stdout.decode().splitlines()
record=json.loads(lines[-1])
if ACTION=='dryrun':record=dict(name=record['metadata']['name'],admission_checked=True)
elif ACTION=='create':record={k:v for k,v in record.items() if k!='contract'}
print('RECEIPT='+json.dumps(dict(action=ACTION,host=socket.gethostname(),result=record)))
'''
    result = remote(code, dict(ROOT=HOME_RUN + "/bundle", INVENTORY_SHA=build["inventory_sha256"],
                               PLAN_SHA=build["plan_sha256"], HOST_PY=HOST_PY, ACTION=name), timeout=280)
    record(name.upper() + "_RESULT.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("session", "preflight", "stage", "dryrun", "create", "launch"))
    name = parser.parse_args().action
    result = {"session": session, "preflight": preflight, "stage": stage}.get(name, lambda: action(name))()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
