"""Verify archived source bytes without importing any scientific/run module."""

import ast
import hashlib
import json
from pathlib import Path


def verify(root=None):
    root = Path(__file__).resolve().parent if root is None else Path(root)
    manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text())
    if manifest["schema"] != "credo.frontier.source-publication.v1":
        raise ValueError("Unexpected publication schema")
    for name, record in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe publication member")
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError("Regular source file required: " + name)
        raw = path.read_bytes()
        if (len(raw) != record["size_bytes"]
                or hashlib.sha256(raw).hexdigest() != record["sha256"]):
            raise ValueError("Source identity mismatch: " + name)
        if relative.suffix == ".py":
            ast.parse(raw, filename=name)
    print("Verified %d archived source/protocol files" % len(manifest["files"]))


if __name__ == "__main__":
    verify()
