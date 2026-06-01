from __future__ import annotations

from pathlib import Path

from check_source_hygiene import MIN_PHYSICAL_LINES, collect_source_hygiene_offenders


def test_sources_have_plain_lf_text() -> None:
    offenders = collect_source_hygiene_offenders()
    assert not offenders, "\n".join(offenders)


def test_critical_sources_have_normal_physical_line_counts() -> None:
    root = Path(__file__).resolve().parents[1]
    for rel_path, min_lines in MIN_PHYSICAL_LINES.items():
        text = (root / rel_path).read_text(encoding="utf-8")
        assert text.count("\n") + 1 >= min_lines
