from __future__ import annotations

from pathlib import Path


FORBIDDEN_UNICODE_CONTROLS = {
    0x202A: "LRE",
    0x202B: "RLE",
    0x202C: "PDF",
    0x202D: "LRO",
    0x202E: "RLO",
    0x2066: "LRI",
    0x2067: "RLI",
    0x2068: "FSI",
    0x2069: "PDI",
    0x2028: "LINE_SEPARATOR",
    0x2029: "PARAGRAPH_SEPARATOR",
}


def test_python_sources_have_plain_lf_text() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = (
        list((root / "package").rglob("*.py"))
        + list((root / "tests").rglob("*.py"))
        + list((root / "runners").rglob("*.py"))
    )

    offenders: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        hits = [
            f"{name}={text.count(chr(codepoint))}"
            for codepoint, name in FORBIDDEN_UNICODE_CONTROLS.items()
            if chr(codepoint) in text
        ]
        if "\r" in text:
            hits.append(f"CR={text.count(chr(13))}")
        if hits:
            offenders.append(f"{path.relative_to(root)}: {', '.join(hits)}")

    assert not offenders, "\n".join(offenders)
