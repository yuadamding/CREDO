"""Publication integrity only; never import archived operational scripts."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "studies" / "full_ddr_20261006"


def test_full_ddr_source_snapshot_matches_provenance() -> None:
    manifest = json.loads((SNAPSHOT / "SOURCE_MANIFEST.json").read_text())
    assert manifest["schema"] == "credo.full_ddr.source_publication.v1"
    assert set(manifest["phases"]) == {"fitting_gpu_native_v6", "d2_forecast_v5"}
    assert len(manifest["files"]) == 258
    for name, record in manifest["files"].items():
        relative = Path(name)
        assert not relative.is_absolute() and ".." not in relative.parts
        path = SNAPSHOT / relative
        assert path.is_file() and not path.is_symlink()
        payload = path.read_bytes()
        assert len(payload) == record["size_bytes"] < 10_000_000
        assert hashlib.sha256(payload).hexdigest() == record["sha256"]
        assert record["origins"]
        if path.suffix == ".py":
            ast.parse(payload, filename=name)


def test_full_ddr_source_snapshot_does_not_replace_v4_or_vendor() -> None:
    assert (ROOT / "src" / "credo_count_sde_v4" / "__init__.py").is_file()
    assert not (ROOT / "src" / "credo").exists()
    assert (SNAPSHOT / "source" / "src" / "credo" / "__init__.py").is_file()
    expected = (ROOT / "vendor" / "SHA256SUMS").read_text().split()[0]
    actual = hashlib.sha256((ROOT / "vendor" / "credo-6f4f57c.tar").read_bytes()).hexdigest()
    assert actual == expected
