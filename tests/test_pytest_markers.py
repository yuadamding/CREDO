from __future__ import annotations

import re
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


CATEGORY_MARKERS = {
    "unit",
    "semantic",
    "integration",
    "runner",
    "biology",
    "randomized",
}


def test_all_test_files_have_an_explicit_category_marker() -> None:
    root = Path(__file__).resolve().parents[1]
    missing: list[str] = []
    for path in sorted((root / "tests").glob("test_*.py")):
        text = path.read_text(encoding="utf-8")
        marks = set(re.findall(r"pytest\.mark\.([A-Za-z_][A-Za-z0-9_]*)", text))
        if not marks.intersection(CATEGORY_MARKERS):
            missing.append(str(path.relative_to(root)))

    assert not missing, "Missing test category marker: " + ", ".join(missing)
