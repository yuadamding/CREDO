from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _env() -> dict[str, str]:
    env = os.environ.copy()
    path = str(ROOT / "package" / "src")
    env["PYTHONPATH"] = path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def test_verify_setup_without_data_succeeds() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verify_setup.py"), "--json"],
        cwd=ROOT,
        env=_env(),
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(result.stdout)
    assert report["ok"] is True
    assert report["data"]["checked"] is False
    assert report["environment"]["required_imports"]["credo"]["ok"] is True


def test_verify_setup_check_data_requires_existing_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing.h5ad"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "verify_setup.py"),
            "--json",
            "--check-data",
            "--data-path",
            str(missing),
        ],
        cwd=ROOT,
        env=_env(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["ok"] is False
    assert report["data"]["checked"] is True
    assert "Missing data file" in report["data"]["error"]
