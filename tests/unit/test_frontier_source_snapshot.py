"""Byte-only publication tests; never import archived training or operations."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "studies" / "performance_frontier_20261015"


def test_frontier_source_snapshot_matches_manifest() -> None:
    manifest = json.loads((SNAPSHOT / "SOURCE_MANIFEST.json").read_text())
    assert manifest["schema"] == "credo.frontier.source-publication.v1"
    assert manifest["scientific_execution_performed_for_publication"] is False
    for name, record in manifest["files"].items():
        relative = Path(name)
        assert not relative.is_absolute() and ".." not in relative.parts
        path = SNAPSHOT / relative
        assert path.is_file() and not path.is_symlink()
        raw = path.read_bytes()
        assert len(raw) == record["size_bytes"]
        assert hashlib.sha256(raw).hexdigest() == record["sha256"]
        assert record["origin_relative_path"] == name
        if path.suffix == ".py":
            ast.parse(raw, filename=name)


def test_observation_bundle_identity_is_separate_from_working_source() -> None:
    manifest = json.loads((SNAPSHOT / "SOURCE_MANIFEST.json").read_text())
    assert set(manifest["observation_v4_bundle_checks"]) == {"D1D2", "D1D3", "D1D4", "D2D3"}
    for record in manifest["observation_v4_bundle_checks"].values():
        assert record["worker_source_matches_snapshot"] is True
        assert record["continuation_source_matches_snapshot"] is True
        assert len(record["source_manifest_sha256"]) == 64
    assert (ROOT / "src" / "credo_count_sde_v4" / "__init__.py").is_file()
    assert not (ROOT / "src" / "frontier").exists()
