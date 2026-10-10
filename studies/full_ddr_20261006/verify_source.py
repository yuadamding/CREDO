"""Verify published run-source bytes without importing or executing run code."""

import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text())
    for name, expected in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe source path: {name}")
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Not a regular source file: {name}")
        raw = path.read_bytes()
        if len(raw) != expected["size_bytes"]:
            raise ValueError(f"Source size mismatch: {name}")
        if hashlib.sha256(raw).hexdigest() != expected["sha256"]:
            raise ValueError(f"Source hash mismatch: {name}")
    print(f"Verified {len(manifest['files'])} source files; no run code executed.")


if __name__ == "__main__":
    main()
