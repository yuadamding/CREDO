from __future__ import annotations

from pathlib import Path

import pytest

from check_source_hygiene import MIN_PHYSICAL_LINES, collect_source_hygiene_offenders


pytestmark = pytest.mark.unit


def test_sources_have_plain_lf_text() -> None:
    offenders = collect_source_hygiene_offenders()
    assert not offenders, "\n".join(offenders)


def test_critical_sources_have_normal_physical_line_counts() -> None:
    root = Path(__file__).resolve().parents[1]
    for rel_path, min_lines in MIN_PHYSICAL_LINES.items():
        text = (root / rel_path).read_text(encoding="utf-8")
        assert text.count("\n") + 1 >= min_lines


def test_source_hygiene_checker_protects_itself() -> None:
    assert "tests/check_source_hygiene.py" in MIN_PHYSICAL_LINES
    assert "tests/test_source_hygiene.py" in MIN_PHYSICAL_LINES
