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

SOURCE_PATTERNS = (
    "package/**/*.py",
    "runners/**/*.py",
    "tests/**/*.py",
    ".github/**/*.yml",
    ".github/**/*.yaml",
    "*.yml",
    "*.yaml",
)

MIN_PHYSICAL_LINES = {
    "package/src/credo/models/causal_context.py": 250,
    "package/src/credo/models/interventions.py": 80,
    "package/src/credo/training/trainer.py": 500,
    "tests/check_source_hygiene.py": 50,
    "tests/test_source_hygiene.py": 20,
    "tests/test_causal_attention_invariants.py": 100,
}


def iter_source_paths(root: Path) -> list[Path]:
    paths: set[Path] = set()
    for pattern in SOURCE_PATTERNS:
        paths.update(path for path in root.glob(pattern) if path.is_file())
    return sorted(paths)


def collect_source_hygiene_offenders(root: Path | None = None) -> list[str]:
    root = Path(__file__).resolve().parents[1] if root is None else root
    offenders: list[str] = []
    for path in iter_source_paths(root):
        text = path.read_text(encoding="utf-8")
        hits = [
            f"{name}={text.count(chr(codepoint))}"
            for codepoint, name in FORBIDDEN_UNICODE_CONTROLS.items()
            if chr(codepoint) in text
        ]
        if "\r" in text:
            hits.append(f"CR={text.count(chr(13))}")
        if text and not text.endswith("\n"):
            hits.append("NO_FINAL_NEWLINE=1")
        trailing_ws = 0
        for line in text.splitlines(keepends=True):
            body = line.rstrip("\r\n")
            if body.rstrip(" \t") != body:
                trailing_ws += 1
        if trailing_ws:
            hits.append(f"TRAILING_WS={trailing_ws}")
        rel_path = str(path.relative_to(root))
        min_lines = MIN_PHYSICAL_LINES.get(rel_path)
        if min_lines is not None:
            n_lines = text.count("\n") + (1 if text else 0)
            if n_lines < min_lines:
                hits.append(f"COLLAPSED_LINES={n_lines}<{min_lines}")
        if hits:
            offenders.append(f"{path.relative_to(root)}: {', '.join(hits)}")
    return offenders


def main() -> int:
    offenders = collect_source_hygiene_offenders()
    if offenders:
        print("\n".join(offenders))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
