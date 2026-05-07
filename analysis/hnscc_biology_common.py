from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


BUILTIN_SIGNATURES: dict[str, list[str]] = {
    "tnf_expansion": [
        "JUN",
        "FOS",
        "JUNB",
        "JUND",
        "EGR1",
        "ATF3",
        "NFKBIZ",
        "SOCS3",
        "CCN1",
        "DUSP1",
        "KLF4",
        "KLF6",
        "ZFP36",
        "BTG2",
    ],
    "autocrine_tnf_tsk": [
        "TNF",
        "MMP9",
        "MMP10",
        "TGFA",
        "IL1A",
        "LAMC2",
        "INHBA",
        "NT5E",
        "CD44",
        "VIM",
        "SNAI2",
        "SERPINE1",
        "ITGA5",
        "PLAU",
    ],
    "pemt": [
        "LAMC2",
        "ITGA5",
        "VIM",
        "SNAI2",
        "FN1",
        "TGFBI",
        "SERPINE1",
        "MMP10",
        "MMP9",
        "CD44",
        "INHBA",
    ],
    "cis_like": [
        "TP63",
        "KRT5",
        "KRT14",
        "KRT17",
        "SOX2",
        "EPCAM",
        "ATP1B3",
    ],
}


GENE_ALIASES: dict[str, list[str]] = {
    "TP63": ["TP63", "TRP63"],
    "TRP63": ["TRP63", "TP63"],
}


def normalize_gene_name(name: object) -> str:
    return str(name).strip().upper()


def candidate_gene_keys(name: object) -> list[str]:
    """Return normalized lookup keys, including simple mouse/human aliases."""
    key = normalize_gene_name(name)
    candidates = [normalize_gene_name(item) for item in GENE_ALIASES.get(key, [key])]
    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
    return out


def load_signature_sets(path: str | Path | None = None) -> dict[str, list[str]]:
    signatures = {name: list(genes) for name, genes in BUILTIN_SIGNATURES.items()}
    if path is None:
        return signatures
    custom_path = Path(path)
    if not custom_path.exists():
        raise FileNotFoundError(custom_path)
    if custom_path.suffix.lower() == ".json":
        payload = json.loads(custom_path.read_text())
        for name, genes in payload.items():
            signatures[str(name)] = [str(gene) for gene in genes]
        return signatures

    df = pd.read_csv(custom_path)
    required = {"signature", "gene"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Custom signature CSV missing columns: {sorted(missing)}")
    for name, group in df.groupby("signature", dropna=False):
        signatures[str(name)] = [str(gene) for gene in group["gene"].dropna()]
    return signatures


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    mean = values.mean(skipna=True)
    std = values.std(skipna=True, ddof=0)
    if pd.isna(std) or float(std) == 0.0:
        return pd.Series(0.0, index=series.index)
    return (values - mean) / std


def first_present(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def infer_target_gene(perturbation_id: object) -> str:
    text = str(perturbation_id)
    if not text:
        return ""
    lower = text.lower()
    if lower in {"control", "ctrl", "non-targeting", "non_targeting", "ntc"}:
        return "control"
    for prefix in ("sg", "gRNA", "guide"):
        text = re.sub(rf"^{prefix}[_:-]", "", text, flags=re.IGNORECASE)
    parts = re.split(r"[_|:;,\-\s]+", text)
    for part in parts:
        cleaned = re.sub(r"[^A-Za-z0-9]", "", part)
        if not cleaned:
            continue
        if cleaned.lower() in {"sg", "grna", "guide", "target"}:
            continue
        if cleaned.lower().startswith(("nt", "ctrl", "control")):
            return "control"
        if any(ch.isalpha() for ch in cleaned):
            return cleaned.upper()
    return text


def classify_priority(row: pd.Series) -> str:
    growth = float(row.get("z_delta_log_mass", 0.0) or 0.0)
    tsk = float(row.get("z_delta_autocrine_tnf_tsk_score", 0.0) or 0.0)
    tnf = float(row.get("z_delta_tnf_expansion_score", 0.0) or 0.0)
    diff = float(row.get("z_diffusion_action", 0.0) or 0.0)
    human = float(row.get("z_human_stage_trend", 0.0) or 0.0)
    if growth >= 0.5 and (tsk >= 0.5 or human >= 0.5):
        return "Class I"
    if growth >= 0.5 and tnf >= 0.5 and tsk < 0.5:
        return "Class II"
    if diff >= 0.75 or float(row.get("z_context_dependence", 0.0) or 0.0) >= 0.75:
        return "Class III"
    if human >= 0.75:
        return "Class IV"
    return "watch"


def write_markdown_table(df: pd.DataFrame, path: Path, *, title: str, max_rows: int = 30) -> None:
    preview = df.head(max_rows).copy()
    lines = [f"# {title}", ""]
    if preview.empty:
        lines.append("No rows.")
        path.write_text("\n".join(lines) + "\n")
        return
    columns = list(preview.columns)
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in preview.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n")
