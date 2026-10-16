"""One read-only exact-input check through the already verified access chain.

Reads only six small publication/annotation metadata files. It does not open
CSR arrays, cell tables, count summaries, outcomes, models or scheduler state.
Run the installed chain verification once before using this command. Results
are printed for a new study receipt; historical files are never rewritten.
"""

import json
import shlex
import subprocess


ROOT = "/rsrch8/home/bcb/yding4/perturbseq/inputs/GSE314342_count_native_v1"
EXPECTED = {
    "INPUT_DATA_COMPLETE.json": "77d4fab700449ac323ff750cef6946a3185c06653891c41123fb4c76a4d5f057",
    "SHA256SUMS": "9b6a046958ee587ad600dd3cae0cf4aaa6f719b38c436b25a6c8c5159f7dcc63",
    "source/features_and_source_positions.parquet": "064223f8766911dd8f6f6e5fb90c36a485bc0b761885e9037331092fc0e1e36b",
    "source/guides.parquet": "d831fba118113adc2234474078bea4f67ab879c1e98fbda469de6f5a5d20549b",
    "source/samples_and_culture_branches.parquet": "08b5b6f4cbddc6b41318ef966653f0ce74df327b3c5a732766ec56e26bb73580",
    "source/shard_manifest.parquet": "f7d1004d737fe2eca60af115e8c2b0422b9e844518f2735ab0ee63eab09ef203",
}
OPTIONS = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
           "-o", "ForwardAgent=no", "-o", "ConnectTimeout=20",
           "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=6"]


def collect():
    remote = '''
import hashlib,json,os,socket,stat
from pathlib import Path
from datetime import datetime,timezone
root=Path(REQUEST["root"])
if (os.getuid(),os.getgid())!=(307469,600815):
    raise RuntimeError("Input-check account differs")
for parent in (root,)+tuple(root.parents):
    if not stat.S_ISDIR(parent.lstat().st_mode):
        raise RuntimeError("Input parent is not a real directory")
if root.lstat().st_uid!=307469:
    raise RuntimeError("Input root owner differs")
rows=[]
for name,expected in sorted(REQUEST["expected"].items()):
    path=root/name
    for parent in path.parents:
        if not stat.S_ISDIR(parent.lstat().st_mode):
            raise RuntimeError("Metadata parent is not a real directory")
    before=path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size>10*1024**2:
        raise RuntimeError("Unexpected metadata type or size")
    descriptor=os.open(str(path),os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(descriptor,"rb") as handle:
        opened=os.fstat(handle.fileno())
        digest=hashlib.sha256()
        for block in iter(lambda:handle.read(1024**2),b""):
            digest.update(block)
        after=os.fstat(handle.fileno())
    signature=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_uid)
    if not signature(before)==signature(opened)==signature(after)==signature(path.lstat()):
        raise RuntimeError("Metadata changed during read")
    if digest.hexdigest()!=expected:
        raise RuntimeError("Metadata digest differs: "+name)
    rows.append({"relative_path":name,"sha256":expected,"bytes":before.st_size,"owner_uid":before.st_uid})
print("RECEIPT="+json.dumps({"schema":"credo.frontier.native_metadata_check.v1",
    "utc":datetime.now(timezone.utc).isoformat(),"host":socket.gethostname(),
    "root":str(root),"files":rows,"status":"metadata_identities_match",
    "numerical_payloads_opened":False,"remote_writes":False,
    "csr_payload_availability_or_hashes_verified":False,
    "worker_mount_verified":False,"launch_admission":False},sort_keys=True))
'''
    code = "REQUEST=" + repr({"root": ROOT, "expected": EXPECTED}) + "\n" + remote
    second = ["/usr/bin/ssh", "-F", "/Users/yding4/.ssh/config.d/codex-chain/company-seadragon.conf", *OPTIONS]
    command = ["/usr/bin/ssh", *OPTIONS, "work-mac", shlex.join(second + [
        "seadragon", shlex.join(["/usr/libexec/platform-python3.6", "-c", code])])]
    result = subprocess.run(command, capture_output=True, timeout=180, check=False)
    if result.returncode:
        raise RuntimeError("Read-only metadata check failed, no retry: " + result.stderr.decode(errors="replace")[-1800:])
    rows = [line[8:] for line in result.stdout.decode().splitlines() if line.startswith("RECEIPT=")]
    if len(rows) != 1:
        raise RuntimeError("Ambiguous metadata response")
    return json.loads(rows[0])


if __name__ == "__main__":
    print(json.dumps(collect(), indent=2))
