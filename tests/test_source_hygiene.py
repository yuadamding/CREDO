from __future__ import annotations

from check_source_hygiene import collect_source_hygiene_offenders


def test_sources_have_plain_lf_text() -> None:
    offenders = collect_source_hygiene_offenders()
    assert not offenders, "\n".join(offenders)
