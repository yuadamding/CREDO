"""Exact approved SSH transport and read-only frontier storage intake.

No action on import; no credentials, broad workload enumeration or retry loop.
Callers must complete the installed session preflight before mutations. Per-call
network timeouts do not impose execution deadlines on scientific workers.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess


STUDY = "credo_performance_frontier_20261015_v1"
ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
HOST_PYTHON = "/usr/libexec/platform-python3.6"
BRIDGE_PYTHON = "/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9"
SSH_OPTIONS = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ForwardAgent=no",
               "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=6"]
SECOND_HOP = ["/usr/bin/ssh", "-F", "/Users/yding4/.ssh/config.d/codex-chain/company-seadragon.conf", *SSH_OPTIONS]
COLLECTOR = WORKSPACE / "credo_full_ddr_gse314342_20261006_v1/comparator_replay_assessment_launch_v1/package_v1/payload/comparator_replay_assessment_v1/bundle/capacity_metadata.py"
COLLECTOR_SHA = "a847dc3a31716a68f333155a07ca128842d6e04dbd25c8833b1de6608e695bf1"


def command(arguments, *, bridge=False):
    text = shlex.join([str(item) for item in arguments])
    if not bridge:
        text = shlex.join(SECOND_HOP + ["seadragon", text])
    return ["/usr/bin/ssh", *SSH_OPTIONS, "work-mac", text]


def call(arguments, *, data=None, bridge=False, timeout=240):
    result = subprocess.run(command(arguments, bridge=bridge), input=data, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError("Exact transport failed; reconcile without retry:\n" + result.stderr.decode(errors="replace")[-3000:])
    return result.stdout.decode()


def remote(code, values=None, *, data=None, bridge=False, timeout=240):
    values = values or {}
    if any(not key.isidentifier() or key.startswith("_") for key in values):
        raise ValueError("Named explicit script inputs required")
    script = "\n".join(key + "=" + repr(value) for key, value in values.items()) + "\n" + code
    interpreter = BRIDGE_PYTHON if bridge else HOST_PYTHON
    # Keep large source inventories out of SSH argv. Binary payload mode uses a
    # small bound writer and stdin only for the artifact, matching retained use.
    if data is None:
        raw = call([interpreter, "-"], data=script.encode(), bridge=bridge, timeout=timeout)
    else:
        if len(script.encode()) > 24000:
            raise ValueError("Binary transfer bootstrap exceeds bounded command size")
        raw = call([interpreter, "-c", script], data=data, bridge=bridge, timeout=timeout)
    lines = [line.split("=", 1)[1] for line in raw.splitlines() if line.startswith("RECEIPT=")]
    if len(lines) != 1:
        raise RuntimeError("Ambiguous exact receipt; reconcile without retry")
    return json.loads(lines[0])


def record(path, value):
    path = Path(path).absolute()
    if (not path.is_relative_to(ROOT) or not path.parent.is_dir()
            or any(parent.is_symlink() for parent in (path, *path.parents))):
        raise ValueError("Prepared local frontier receipt destination required")
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


CAPACITY_CODE = r'''
import hashlib,json,os,stat
from pathlib import Path
if hashlib.sha256(COLLECTOR_SOURCE.encode()).hexdigest()!=COLLECTOR_SHA:
 raise RuntimeError('Retained metadata collector source differs')
scope={'__name__':'frontier_retained_capacity'}
exec(COLLECTOR_SOURCE,scope)
scope['PROJECT']='/rsrch8/home/bcb/yding4/perturbseq'
scope['assess']=lambda snapshot,*args:snapshot
snapshot=scope['collect_capacity']({},'',{})
quota=scope['parse_quota'](snapshot['quota_output'])
limits=[quota[key] for key in ('soft_bytes','hard_bytes') if quota[key]>0]
if not limits or quota['block_grace']!='none' or quota['file_grace']!='none':
 raise RuntimeError('Positive exact quota and clear grace required')
snapshot['quota']=quota
snapshot['headroom_bytes']=min(limits)-quota['used_bytes']-quota['in_doubt_bytes']
snapshot['schema']='credo.frontier.storage-intake.v1'
snapshot['collector_sha256']=COLLECTOR_SHA
snapshot['study_id']=STUDY
snapshot['launch_admission']=False
snapshot['new_study_roots']={}
for parent in ['/rsrch8/home/bcb/yding4/perturbseq','/rsrch8/scratch/bcb/yding4/perturbseq']:
 p=Path(parent)
 for ancestor in (p,)+tuple(p.parents):
  if not stat.S_ISDIR(ancestor.lstat().st_mode):raise RuntimeError('Unsafe project ancestor')
 if p.stat().st_uid!=307469 or not os.access(str(p),os.W_OK):raise RuntimeError('Project parent not owned/writable')
 target=p/STUDY
 present=os.path.lexists(str(target))
 if present and (not stat.S_ISDIR(target.lstat().st_mode) or target.stat().st_uid!=307469):
  raise RuntimeError('Existing study root type/owner differs')
 snapshot['new_study_roots'][str(target)]={'exists':present,'parent_owned_writable':True}
print('RECEIPT='+json.dumps(snapshot))
'''


def capacity():
    source = COLLECTOR.read_bytes()
    if hashlib.sha256(source).hexdigest() != COLLECTOR_SHA:
        raise ValueError("Retained collector source changed")
    return remote(CAPACITY_CODE, {"COLLECTOR_SOURCE": source.decode(), "COLLECTOR_SHA": COLLECTOR_SHA, "STUDY": STUDY})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("capacity",))
    parser.add_argument("--receipt", type=Path, required=True)
    options = parser.parse_args()
    if options.receipt.exists():
        raise ValueError("Receipt already exists; do not repeat an uncertain operation")
    result = capacity()
    record(options.receipt, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
