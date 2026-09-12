from __future__ import annotations

import os
from pathlib import Path

import pytest

from credo_count_sde_v4.compat.credo3 import FROZEN_COMMIT, FROZEN_SOURCE_SHA256
from credo_count_sde_v4.compat.credo3 import verify as verify_module
from credo_count_sde_v4.compat.credo3.verify import verify_frozen_credo
from credo_count_sde_v4.errors import IntegrityError
from credo_count_sde_v4.recipe import recipe


def _available_workspace() -> Path:
    root = verify_module._workspace_root()
    if not os.environ.get("CREDO_V4_WORKSPACE_ROOT") and not (root / "CREDO").exists():
        pytest.skip("Sibling CREDO checkout is supplied by release CI.")
    return root


def test_frozen_credo_preflight_when_workspace_is_available() -> None:
    root = _available_workspace()
    receipt = verify_frozen_credo(root)
    assert receipt["commit"] == FROZEN_COMMIT
    assert receipt["artifact_sha256"] == FROZEN_SOURCE_SHA256


def test_development_descriptor_keeps_v4_loader_canonical() -> None:
    assert recipe.recipe_id == "credo.count_sde_v4"
    assert recipe.recipe_version == "4.0.dev37"
    assert recipe.loader == "credo-v4 open-run"
    assert recipe.discovery_only


def test_frozen_credo_preflight_without_git_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    root = _available_workspace()
    monkeypatch.setattr(verify_module.shutil, "which", lambda _: None)
    receipt = verify_frozen_credo(root)
    assert receipt["commit"] == FROZEN_COMMIT
    assert receipt["verification_method"] == "archive_byte_comparison"


@pytest.mark.parametrize("without_git", [False, True])
def test_nested_ci_workspace_uses_explicit_root(tmp_path, monkeypatch, without_git):
    source = _available_workspace()
    archive = next(
        path
        for path in (
            source / "vendor/credo-6f4f57c.tar",
            source / "credo-count-sde-v4/vendor/credo-6f4f57c.tar",
            source / "credo_biology_validation/vendor/credo-6f4f57c.tar",
        )
        if path.is_file()
    )
    nested = tmp_path / "work/CREDO/CREDO"
    (nested / "vendor").mkdir(parents=True)
    (nested / "CREDO").symlink_to(source / "CREDO", target_is_directory=True)
    (nested / "vendor/credo-6f4f57c.tar").symlink_to(archive)
    monkeypatch.setenv("CREDO_V4_WORKSPACE_ROOT", str(nested))
    if without_git:
        monkeypatch.setattr(verify_module.shutil, "which", lambda _: None)
    assert _available_workspace() == nested
    receipt = verify_frozen_credo()
    assert receipt["checkout"] == str(nested / "CREDO")
    assert receipt["commit"] == FROZEN_COMMIT
    assert receipt["artifact_sha256"] == FROZEN_SOURCE_SHA256


def test_explicit_missing_workspace_fails_not_skips(tmp_path, monkeypatch):
    monkeypatch.setenv("CREDO_V4_WORKSPACE_ROOT", str(tmp_path))
    assert _available_workspace() == tmp_path
    with pytest.raises(IntegrityError, match="unavailable"):
        verify_frozen_credo(_available_workspace())
