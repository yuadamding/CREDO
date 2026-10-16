"""Publish an immutable local main-worker source package, never a remote job.

This package is an input to future CUDA qualification and controller assembly,
not a launch bundle. It cannot reuse observation qualification as main evidence.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from frontier.main_worker_source import source_closure, manifest_payload
from .build import archive_bytes
from .common import publish, read


def construct(root):
    source = source_closure(root)
    manifest = manifest_payload(source)
    payload = {"source/" + name: raw for name, raw in source.items()}
    payload["SOURCE_MANIFEST.json"] = manifest
    archive = archive_bytes(payload)
    return archive, dict(schema="credo.frontier.main-worker-source-package.v1",
        entry="python -m frontier.main_worker", source_files=len(source),
        source_bytes=sum(map(len, source.values())), archive_bytes=len(archive),
        archive_sha256=sha256(archive).hexdigest(), source_manifest_sha256=sha256(manifest).hexdigest(),
        scientific_launch_authority=False, cuda_qualification=False,
        source_manifest=json.loads(manifest)), payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root, output = Path(args.study_root), Path(args.output)
    if (not root.is_absolute() or not output.is_absolute()
            or any(p.is_symlink() for p in (output, *output.parents)) or not output.parent.is_dir()):
        raise ValueError("Prepared absolute nonsymlink local output parent required")
    archive, receipt, payload = construct(root)
    output.mkdir(mode=0o700)
    path = output / "main-source.tar.gz"
    with path.open("xb") as handle:
        handle.write(archive)
        handle.flush()
        import os
        os.fsync(handle.fileno())
    if read(path, receipt["archive_sha256"]) != archive:
        raise ValueError("Main source archive readback differs")
    with (output / "SOURCE_MANIFEST.json").open("xb") as handle:
        handle.write(payload["SOURCE_MANIFEST.json"])
        handle.flush()
        os.fsync(handle.fileno())
    read(output / "SOURCE_MANIFEST.json", receipt["source_manifest_sha256"])
    publish(output / "PACKAGE.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
