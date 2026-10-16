"""Bound the qualification producer and fixed inputs; never submit or authorize.

Only current import-closure code, three study protocol files and four pinned
historical reader sources are packaged. No fitted weights or biological data.
"""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

from frontier.main_worker_source import source_closure, manifest_payload, QUALIFICATION_ENTRY
from frontier.main_qualification_contract import settings
from frontier.main_qualification_plan import plan
from frontier.native_inputs import READER_SOURCE_HASHES
from frontier.objective_registry import ObjectiveRules
from .build import archive_bytes
from .common import encoded, publish, read


def construct(root, reader_root):
    root, reader_root = Path(root), Path(reader_root)
    source = source_closure(root, entry=QUALIFICATION_ENTRY)
    worker = source_closure(root)
    if any(source.get(k) != v for k,v in worker.items()):
        raise ValueError("Worker closure must be contained unchanged in qualification source")
    parent_manifest, worker_manifest = manifest_payload(source, entry=QUALIFICATION_ENTRY), manifest_payload(worker)
    payload = {"source/"+k:v for k,v in source.items()}
    payload.update({"SOURCE_MANIFEST.json":parent_manifest,"WORKER_SOURCE_MANIFEST.json":worker_manifest,
                    "QUALIFICATION_PLAN.json":encoded(plan()),"QUALIFICATION_SETTINGS.json":encoded(settings())})
    for name in ("OBSERVATION_PROTOCOL.json", "NORMALIZATION_PROTOCOL.json", "MAIN_OBJECTIVE_RULES.json"):
        payload[name] = read(root/name, maximum=4*1024**2)
    rules = payload["MAIN_OBJECTIVE_RULES.json"]
    ObjectiveRules(rules, sha256(rules).hexdigest())
    for name,digest in READER_SOURCE_HASHES.items():
        payload["retained_reader/"+name] = read(reader_root/name,digest,maximum=2*1024**2)
    if sum(map(len,payload.values())) > 16*1024**2:
        raise ValueError("Qualification package exceeds bounded code/protocol allowance")
    archive = archive_bytes(payload)
    record = dict(schema="credo.frontier.main-qualification-source-package.v1",
        entry="python -m frontier.main_qualification_worker", source_files=len(source),
        source_bytes=sum(map(len,source.values())), worker_source_files=len(worker),
        source_manifest_sha256=sha256(parent_manifest).hexdigest(),
        worker_source_manifest_sha256=sha256(worker_manifest).hexdigest(),
        archive_sha256=sha256(archive).hexdigest(), archive_bytes=len(archive),
        files={k:dict(sha256=sha256(v).hexdigest(),size_bytes=len(v)) for k,v in sorted(payload.items())},
        numerical_execution=False, scientific_launch_authority=False, disposable_launch_authority=False,
        authority_admission_contract_startup_release_included=False, cuda_qualification=False,
        biological_payloads_included=False, optimizer_calls=0)
    return archive, record, payload


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("study-root","reader-source-root","output"):
        parser.add_argument("--"+name,required=True)
    args=parser.parse_args(argv)
    output=Path(args.output)
    if (not output.is_absolute() or not output.parent.is_dir()
            or any(p.is_symlink() for p in (output,*output.parents))):
        raise ValueError("Prepared nonsymlink absolute output parent required")
    archive,receipt,payload=construct(args.study_root,args.reader_source_root)
    output.mkdir(mode=0o700)
    publications={"main-qualification-source.tar.gz":archive,
        **{k:payload[k] for k in ("SOURCE_MANIFEST.json","WORKER_SOURCE_MANIFEST.json",
                                  "QUALIFICATION_PLAN.json","QUALIFICATION_SETTINGS.json")}}
    for name,raw in publications.items():
        path=output/name
        with path.open("xb") as stream:
            stream.write(raw);stream.flush();os.fsync(stream.fileno())
        if read(path,sha256(raw).hexdigest())!=raw:
            raise ValueError("Qualification source package readback differs")
    publish(output/"PACKAGE.json",receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!="files"},sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
