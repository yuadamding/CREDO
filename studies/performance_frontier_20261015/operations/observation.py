"""Serialized source publication and exact scalar observation-job operations.

Requires a real local admission-journal reservation and the installed session
preflight. Historical jobs, inputs and results are never modified. No automatic
retry, scheduler substitution, duplicate submission or execution deadline.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from transport import (ROOT, WORKSPACE, STUDY, HOST_PYTHON, SECOND_HOP,
                       remote, record, capacity)


HELPER = WORKSPACE / "credo_full_ddr_gse314342_20261006_v1/final_refit_v1/bundle/deployment.py"
HELPER_SHA = "1c88af69851bb00028488d714bb4fdef014cf3f3cfd20c081cd470c9d7df0ae2"
HOME = "/rsrch8/home/bcb/yding4/perturbseq/" + STUDY
SCRATCH = "/rsrch8/scratch/bcb/yding4/perturbseq/" + STUDY
GIB = 1024 ** 3


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def package(path):
    path = Path(path).absolute()
    if not path.is_relative_to(ROOT) or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Exact regular local frontier package required")
    build = json.loads((path / "BUILD.json").read_bytes())
    plan = json.loads((path / "bundle/PLAN.json").read_bytes())
    if (build["schema"] != "credo.frontier.observation-build.v1"
            or build["plan_sha256"] != sha(path / "bundle/PLAN.json")
            or build["archive_sha256"] != sha(path / "bundle.tar.gz")
            or build["inventory_sha256"] != sha(path / "bundle/INVENTORY.json")
            or build["run_id"] != plan["run_id"]
            or plan["bundle_root"] != HOME + "/" + build["phase"] + "/bundle"
            or plan["control_root"] != SCRATCH + "/" + build["phase"]):
        raise ValueError("Built source/plan/root identity mismatch")
    inventory = json.loads((path / "bundle/INVENTORY.json").read_bytes())
    seen = {p.relative_to(path / "bundle").as_posix() for p in (path / "bundle").rglob("*") if p.is_file()}
    if seen != set(inventory) | {"INVENTORY.json"}:
        raise ValueError("Exact local source inventory differs")
    for name, item in inventory.items():
        member = path / "bundle" / name
        if member.is_symlink() or sha(member) != item["sha256"] or member.stat().st_size != item["size_bytes"]:
            raise ValueError("Source package changed: " + name)
    for build_key, plan_key, name in (
            ("source_manifest_sha256", "source_manifest", "SOURCE_MANIFEST.json"),
            ("phase_authority_sha256", "phase_authority", "OBSERVATION_PHASE_CONTRACT.json")):
        payload_sha = sha(path / "bundle" / name)
        reference = plan.get(plan_key)
        if (build.get(build_key) != payload_sha or not isinstance(reference, dict)
                or reference.get("sha256") != payload_sha
                or reference.get("path") != plan["bundle_root"] + "/" + name):
            raise ValueError("BUILD/PLAN/payload binding differs: " + name)
    return path, build, plan


def session_gate(path):
    value = json.loads(Path(path).read_bytes())
    if (value.get("status") != "ready" or value.get("mode") != "mutate"
            or value.get("context") != "yding4_yn-gpu-workload@kubernetes-admin@kubernetes"
            or value.get("namespace") != "yn-gpu-workload"
            or not all(row.get("allowed") is True for row in value["required"].values())):
        raise ValueError("Current standard mutation preflight required")


def admission_gate(path, build, plan):
    sys.path.insert(0, str(ROOT))
    from frontier.coordination import AdmissionJournal, LIVE
    admission = json.loads((path / "bundle/ADMISSION.json").read_bytes())
    if (admission.get("source_manifest_sha256") != build["source_manifest_sha256"]
            or admission.get("phase_authority_sha256") != build["phase_authority_sha256"]):
        raise ValueError("Reservation does not bind these exact numerical sources and phase authority")
    with AdmissionJournal(ROOT / "admission", study_identity_sha256=sha(ROOT / "AUTHORITY.json")) as journal:
        state = journal.snapshot()
        item = state["tasks"].get(build["run_id"])
        if (item is None or item["status"] not in LIVE or item["worker_id"] != build["run_id"]
                or item["storage_reserved_bytes"] != plan["maximum_output_bytes"]
                or item["work_binding"] != admission["work_binding"]
                or item["admission_sha256"] != admission["phase_authority_sha256"]):
            raise ValueError("Exact live journal reservation missing or changed")
        if "replaces_task_id" in item:
            recovery = ROOT / "OBSERVATION_REPLACEMENT_FITS_V2_AUTHORITY.json"
            if (admission.get("replaces_task_id") != item["replaces_task_id"]
                    or admission.get("replacement_authority_sha256") != item["replacement_authority_sha256"]
                    or sha(recovery) != item["replacement_authority_sha256"]
                    or json.loads(recovery.read_bytes()) != admission.get("replacement_authority")):
                raise ValueError("Replacement launch lacks its exact explicit recovery approval")
        reserved = sum(task["storage_reserved_bytes"] for task in state["tasks"].values())
        live = sum(task["status"] in LIVE for task in state["tasks"].values())
        if (admission["aggregate_reserved_output_bytes"] < reserved
                or admission["live_h100_reservations"] < live):
            raise ValueError("Stale admission export understates current reservations; bind a complete launch batch")
        events = [json.loads(event.read_bytes()) for event in (ROOT / "admission/events").iterdir()]
        cited = [event for event in events if hashlib.sha256(json.dumps(event, sort_keys=True,
                 allow_nan=False, separators=(",", ":")).encode()).hexdigest() == admission["journal_event_sha256"]]
        if (len(cited) != 1 or cited[0]["tasks"].get(build["run_id"], {}).get("work_binding") != item["work_binding"]
                or cited[0]["tasks"][build["run_id"]]["storage_reserved_bytes"] != item["storage_reserved_bytes"]):
            raise ValueError("Admission does not cite this task's actual journal event")
    return admission


def paths(build):
    return ("/Users/yding4/" + build["run_id"] + "-transfer",
            "/rsrch8/scratch/bcb/yding4/perturbseq/" + build["run_id"] + "-transfer")


PREFLIGHT = r'''
import os,stat,json,subprocess
from pathlib import Path
if (os.getuid(),os.getgid())!=(307469,600815):raise RuntimeError('Operator owner differs')
for target in ROOTS:
 p=Path(target)
 if os.path.lexists(str(p)):raise RuntimeError('New phase already exists; reconcile')
 parent=p.parent if p.parent.exists() else p.parent.parent
 for ancestor in (parent,)+tuple(parent.parents):
  if not stat.S_ISDIR(ancestor.lstat().st_mode):raise RuntimeError('Unsafe phase ancestor')
 if parent.stat().st_uid!=307469 or not os.access(str(parent),os.W_OK):raise RuntimeError('Phase parent not owned/writable')
if os.path.lexists(TRANSFER):raise RuntimeError('Transfer root already exists; reconcile')
for filename in [HOST_PY,'/risapps/noarch/kubectl/1.28.4/bin/kubectl']:
 p=Path(filename)
 if p.is_symlink() or not p.is_file() or not os.access(str(p),os.X_OK):raise RuntimeError('Bound host tool missing')
config='/rsrch8/home/bcb/yding4/.kube/config'
info=os.lstat(config)
if not stat.S_ISREG(info.st_mode) or info.st_uid!=307469 or stat.S_IMODE(info.st_mode)!=0o600:
 raise RuntimeError('Kubeconfig metadata differs')
k=['/risapps/noarch/kubectl/1.28.4/bin/kubectl','--context','yding4_yn-gpu-workload@kubernetes-admin@kubernetes','-n','yn-gpu-workload','--request-timeout=30s']
for kind,extra in [('job',[JOB,'--ignore-not-found=true']),('pods',['-l','job-name='+JOB])]:
 result=subprocess.run(k+['get',kind]+extra+['-o','json'],env=dict(os.environ,KUBECONFIG=config),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=45)
 if result.returncode or (bool(result.stdout.strip()) if kind=='job' else bool(json.loads(result.stdout)['items'])):
  raise RuntimeError('Exact new Job/Pod absence not established')
print('RECEIPT='+json.dumps(dict(status='passed',job=JOB,roots_absent=ROOTS,transfer_absent=TRANSFER,remote_writes=False)))
'''


ARCHIVE_WRITER = r'''
import os,stat,hashlib,sys,json
from pathlib import Path
if os.getuid()!=UID:raise RuntimeError('Archive writer owner differs')
root=Path(TRANSFER)
for p in root.parents:
 if not stat.S_ISDIR(p.lstat().st_mode):raise RuntimeError('Unsafe transfer ancestor')
if root.parent.stat().st_uid!=UID:raise RuntimeError('Transfer parent owner differs')
root.mkdir(mode=0o700)
probe=root/'writer-canary'
with probe.open('xb') as f:
 f.write(b'owned');f.flush();os.fsync(f.fileno())
if probe.stat().st_uid!=UID or probe.read_bytes()!=b'owned':raise RuntimeError('Transfer canary failed')
probe.unlink()
raw=sys.stdin.buffer.read(100_000_001)
if len(raw)>100_000_000 or hashlib.sha256(raw).hexdigest()!=EXPECTED:raise RuntimeError('Archive bytes differ')
p=root/'bundle.tar.gz'
with os.fdopen(os.open(str(p),os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600),'wb') as f:
 f.write(raw);f.flush();os.fchmod(f.fileno(),0o600);os.fsync(f.fileno())
if p.stat().st_uid!=UID or hashlib.sha256(p.read_bytes()).hexdigest()!=EXPECTED:raise RuntimeError('Archive readback differs')
print('RECEIPT='+json.dumps(dict(status='archive_staged_verified',root=TRANSFER,sha256=EXPECTED,size_bytes=len(raw))))
'''


def preflight(path, build, plan, admission):
    snapshot = capacity()
    required = admission["aggregate_reserved_output_bytes"] + 16 * GIB
    if min(snapshot["headroom_bytes"], snapshot["filesystem_available_bytes"]) < required:
        raise ValueError("Fresh aggregate output/reserve headroom unavailable")
    bridge, transfer = paths(build)
    roots = [plan["control_root"], plan["bundle_root"].removesuffix("/bundle")]
    result = remote(PREFLIGHT, dict(ROOTS=roots, TRANSFER=transfer, HOST_PY=HOST_PYTHON, JOB=build["run_id"]))
    result["capacity"] = snapshot
    result["plan_sha256"] = build["plan_sha256"]
    result["bridge"] = remote(r'''
import os,stat,json
from pathlib import Path
p=Path(TRANSFER)
if os.getuid()!=1896174575 or os.path.lexists(str(p)):raise RuntimeError('Bridge owner/absence differs')
for ancestor in p.parents:
 if not stat.S_ISDIR(ancestor.lstat().st_mode):raise RuntimeError('Unsafe bridge ancestor')
if p.parent.stat().st_uid!=os.getuid():raise RuntimeError('Bridge parent owner differs')
fs=os.statvfs(str(p.parent))
if fs.f_bavail*fs.f_frsize<REQUIRED:raise RuntimeError('Bridge capacity insufficient')
print('RECEIPT='+json.dumps(dict(status='passed',available_bytes=fs.f_bavail*fs.f_frsize)))
''', dict(TRANSFER=bridge, REQUIRED=2 * (path / "bundle.tar.gz").stat().st_size + 100_000_000), bridge=True)
    record(path / "PREFLIGHT.json", result)
    return result


def stage(path, build, plan):
    check = json.loads((path / "PREFLIGHT.json").read_bytes())
    if check.get("status") != "passed" or check.get("plan_sha256") != build["plan_sha256"]:
        raise ValueError("Exact successful preflight required")
    helper = HELPER.read_bytes()
    if hashlib.sha256(helper).hexdigest() != HELPER_SHA:
        raise ValueError("Retained exclusive publisher changed")
    record(path / "STAGE_INTENT.json", dict(plan_sha256=build["plan_sha256"], archive_sha256=build["archive_sha256"],
                                          operator_sha256=sha(__file__), transport_sha256=sha(Path(__file__).with_name("transport.py"))))
    bridge, transfer = paths(build)
    result = remote(ARCHIVE_WRITER, dict(TRANSFER=bridge, UID=1896174575, EXPECTED=build["archive_sha256"]),
                    data=(path / "bundle.tar.gz").read_bytes(), bridge=True)
    record(path / "BRIDGE_STAGE.json", result)
    import shlex
    script = "UID=307469\nTRANSFER=" + repr(transfer) + "\nEXPECTED=" + repr(build["archive_sha256"]) + "\n" + ARCHIVE_WRITER
    forwarding = SECOND_HOP + ["seadragon", shlex.join([HOST_PYTHON, "-c", script])]
    result = remote(r'''
import os,stat,hashlib,json,subprocess
from pathlib import Path
p=Path(ARCHIVE)
if not stat.S_ISREG(p.lstat().st_mode) or p.stat().st_uid!=1896174575:raise RuntimeError('Bridge archive owner/type differs')
raw=p.read_bytes()
if hashlib.sha256(raw).hexdigest()!=EXPECTED:raise RuntimeError('Bridge archive changed')
r=subprocess.run(COMMAND,input=raw,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=240)
if r.returncode:raise RuntimeError(r.stderr.decode()[-3000:])
lines=[line for line in r.stdout.decode().splitlines() if line.startswith('RECEIPT=')]
if len(lines)!=1:raise RuntimeError('Transfer uncertain; reconcile')
print(lines[0])
''', dict(ARCHIVE=bridge + "/bundle.tar.gz", EXPECTED=build["archive_sha256"], COMMAND=forwarding), bridge=True)
    record(path / "REMOTE_ARCHIVE_STAGE.json", result)
    result = remote(helper.decode() + r'''
if (os.getuid(),os.getgid())!=(307469,600815):raise RuntimeError('Publisher owner differs')
archive=Path(TRANSFER)/'bundle.tar.gz'
if not stat.S_ISREG(archive.lstat().st_mode) or archive.stat().st_uid!=307469:raise RuntimeError('Archive owner/type differs')
payload,inventory=archive_payload(archive.read_bytes(),BUILD['archive_sha256'],BUILD['inventory_sha256'])
plan=json.loads(payload['PLAN.json'])
if sha(payload['PLAN.json'])!=BUILD['plan_sha256'] or plan['run_id']!=JOB:raise RuntimeError('Publication plan differs')
roots=[Path(CONTROL),Path(HOME)]
if plan['control_root']!=CONTROL or plan['bundle_root']!=HOME+'/bundle':raise RuntimeError('Exact new phase differs')
for root,fileset in zip(roots,('s_bcb-yding4','h_bcb-yding4')):
 if os.path.lexists(str(root)):raise RuntimeError('Phase publication exists; no retry')
 parent=root.parent
 if not parent.exists():
  if parent.name!=STUDY or not parent.parent.is_dir():raise RuntimeError('Unsafe study parent')
  parent.mkdir(mode=0o700)
 if not stat.S_ISDIR(parent.lstat().st_mode) or parent.stat().st_uid!=307469:raise RuntimeError('Study root differs')
 capacity(parent,fileset,minimum_bytes=REQUIRED)
result=publish_both_roots(roots,payload,BUILD)
print('RECEIPT='+json.dumps(dict(status='published_verified',publications=result,run_id=JOB,plan_sha256=BUILD['plan_sha256'])))
''', dict(TRANSFER=transfer, BUILD=build, JOB=build["run_id"], STUDY=STUDY,
          CONTROL=plan["control_root"], HOME=plan["bundle_root"].removesuffix("/bundle"),
          REQUIRED=plan["maximum_output_bytes"] + 16 * GIB))
    record(path / "DEPLOYMENT.json", result)
    return result


def closeout(path, build):
    published = json.loads((path / "DEPLOYMENT.json").read_bytes())
    if published.get("status") != "published_verified" or published.get("plan_sha256") != build["plan_sha256"]:
        raise ValueError("Verified final publication required before transfer cleanup")
    record(path / "TRANSFER_CLOSEOUT_INTENT.json", dict(archive_sha256=build["archive_sha256"], publication_sha256=sha(path / "DEPLOYMENT.json")))
    results = []
    for bridge, transfer, owner in ((True, paths(build)[0], 1896174575), (False, paths(build)[1], 307469)):
        result = remote(r'''
import os,stat,json,hashlib
from pathlib import Path
root=Path(TRANSFER)
if os.getuid()!=UID or not stat.S_ISDIR(root.lstat().st_mode) or root.stat().st_uid!=UID:
 raise RuntimeError('Exact transfer owner/type differs')
if sorted(p.name for p in root.iterdir())!=['bundle.tar.gz']:raise RuntimeError('Transfer directory membership differs')
p=root/'bundle.tar.gz'
if not stat.S_ISREG(p.lstat().st_mode) or p.stat().st_uid!=UID or p.stat().st_nlink!=1:
 raise RuntimeError('Exact temporary archive differs')
if hashlib.sha256(p.read_bytes()).hexdigest()!=EXPECTED:raise RuntimeError('Temporary archive hash differs')
p.unlink();root.rmdir()
if os.path.lexists(str(root)):raise RuntimeError('Transfer root remains')
print('RECEIPT='+json.dumps(dict(status='verified_absent',removed_archive=str(p),removed_empty_directory=str(root),archive_sha256=EXPECTED,final_publication_retained=True)))
''', dict(TRANSFER=transfer, UID=owner, EXPECTED=build["archive_sha256"]), bridge=bridge)
        record(path / ("BRIDGE_CLOSEOUT.json" if bridge else "REMOTE_TRANSFER_CLOSEOUT.json"), result)
        results.append(result)
    return results


def lifecycle(path, build, plan, action):
    if action not in ("dryrun", "create", "launch"):
        raise ValueError("Exact supported scalar lifecycle action required")
    for name in ("DEPLOYMENT.json", "BRIDGE_CLOSEOUT.json", "REMOTE_TRANSFER_CLOSEOUT.json"):
        if not (path / name).is_file():
            raise ValueError("Complete publication and relay closeout required")
    record(path / (action.upper() + "_INTENT.json"), dict(run_id=build["run_id"], plan_sha256=build["plan_sha256"], action=action))
    result = remote(r'''
import os,json,subprocess,hashlib,stat
from pathlib import Path
p=Path(PLAN)
if hashlib.sha256(p.read_bytes()).hexdigest()!=EXPECTED:raise RuntimeError('Remote plan changed')
config='/rsrch8/home/bcb/yding4/.kube/config'
info=os.lstat(config)
if not stat.S_ISREG(info.st_mode) or info.st_uid!=307469 or stat.S_IMODE(info.st_mode)!=0o600:
 raise RuntimeError('Kubeconfig metadata differs')
r=subprocess.run([PYTHON,CONTROL,ACTION,'--plan',PLAN],env=dict(os.environ,KUBECONFIG=config),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
if r.returncode:raise RuntimeError('Lifecycle failed; preserve exact attempt: '+r.stderr.decode()[-4000:])
root=Path(CONTROL_ROOT)/'gpu'
records={}
for name in ['live-binding.json','supervisor-ready.json','activation.json','launch-success.json','launch-attention.json','terminal.json']:
 q=root/name
 if q.is_file():records[name]=json.loads(q.read_bytes())
print('RECEIPT='+json.dumps(dict(status='action_returned',action=ACTION,run_id=JOB,stdout=r.stdout.decode(),records=records)))
''', dict(PYTHON=HOST_PYTHON, CONTROL=plan["bundle_root"] + "/control.py", PLAN=plan["bundle_root"] + "/PLAN.json",
          EXPECTED=build["plan_sha256"], CONTROL_ROOT=plan["control_root"], ACTION=action, JOB=build["run_id"]), timeout=240)
    record(path / (action.upper() + ".json"), result)
    if action in ("create", "launch"):
        binding = result["records"].get("live-binding.json")
        if not binding or binding.get("job") != build["run_id"] or not binding.get("uid"):
            raise RuntimeError("Successful mutation lacks exact Job binding; reconcile, do not repeat")
        from frontier.coordination import AdmissionJournal
        with AdmissionJournal(ROOT / "admission", study_identity_sha256=sha(ROOT / "AUTHORITY.json")) as journal:
            journal.reconcile(build["run_id"], status="submitted", exact_scheduler_id=binding["uid"],
                              evidence_sha256=sha(path / (action.upper() + ".json")))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "stage", "closeout", "dryrun", "create", "launch"))
    parser.add_argument("--package", required=True)
    parser.add_argument("--session-receipt", required=True)
    args = parser.parse_args()
    path, build, plan = package(args.package)
    session_gate(args.session_receipt)
    admission = admission_gate(path, build, plan)
    if args.action == "preflight":
        result = preflight(path, build, plan, admission)
    elif args.action == "stage":
        result = stage(path, build, plan)
    elif args.action == "closeout":
        result = closeout(path, build)
    else:
        result = lifecycle(path, build, plan, args.action)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
