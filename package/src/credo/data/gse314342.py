"""GSE314342 cohort adaptation for multi-time CREDO trajectories."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


ACCESSION = "GSE314342"
TIME_LABELS = ("Rest", "Stim8hr", "Stim48hr")
DEFAULT_PHYSICAL_TIMES = {"Rest": 0.0, "Stim8hr": 8.0, "Stim48hr": 48.0}
NTC_EMBEDDING_ID = "__NTC__"

OBS_ALIASES: dict[str, tuple[str, ...]] = {
    "guide_id": ("guide_id", "guide", "sgRNA", "sgrna", "guide_name"),
    "guide_type": ("guide_type", "sgRNA_type", "guide_class"),
    "guide_group": ("guide_group", "assignment_group", "guide_assignment"),
    "target_gene": (
        "perturbed_gene_name",
        "target_gene_name",
        "target_gene",
        "gene",
        "target",
    ),
    "target_gene_id": (
        "perturbed_gene_id",
        "target_gene_id",
        "designed_target_gene_id",
    ),
    "donor_id": ("donor_id", "donor", "Donor", "sample_id"),
    "time_label": ("condition", "time_label", "stimulation", "timepoint"),
    "lane_id": ("lane_id", "lane", "Lane"),
    "run_id": ("10xrun_id", "run_id", "run", "Library"),
    "low_quality": ("low_quality", "is_low_quality", "lowquality"),
    "guide_umi_count": (
        "top_guide_UMI_counts",
        "guide_umi_count",
        "guide_umis",
        "guide_UMI",
    ),
    "offtarget_flag": ("offtarget_flag", "off_target", "high_offtarget"),
    "guide_effective_rest": ("guide_effective_rest", "effective_Rest"),
    "guide_effective_8h": ("guide_effective_8h", "effective_Stim8hr"),
    "guide_effective_48h": ("guide_effective_48h", "effective_Stim48hr"),
}


@dataclass(frozen=True)
class LateTimeResolution:
    physical_time_hours: float
    status: str
    processed_label: str
    raw_label: str
    rationale: str
    source: str

    @classmethod
    def load(cls, path: str | Path, *, require_resolved: bool = True) -> "LateTimeResolution":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if str(payload.get("accession", ACCESSION)) != ACCESSION:
            raise ValueError(f"Late-time resolution does not describe {ACCESSION}.")
        required = {
            "physical_time_hours",
            "status",
            "processed_label",
            "raw_label",
            "rationale",
            "source",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise KeyError(f"Late-time resolution is missing fields: {missing}")
        result = cls(
            physical_time_hours=float(payload["physical_time_hours"]),
            status=str(payload["status"]),
            processed_label=str(payload["processed_label"]),
            raw_label=str(payload["raw_label"]),
            rationale=str(payload["rationale"]),
            source=str(payload["source"]),
        )
        if not np.isfinite(result.physical_time_hours) or result.physical_time_hours <= 8:
            raise ValueError("Resolved late physical time must be finite and greater than 8 hours.")
        if require_resolved and result.status != "resolved":
            raise ValueError("Final GSE314342 builds require late-time status='resolved'.")
        if result.processed_label != "Stim48hr":
            raise ValueError("The processed late condition must be recorded as 'Stim48hr'.")
        return result


def _resolve_column(frame: pd.DataFrame, canonical: str, *, required: bool = True) -> str | None:
    match = next((name for name in OBS_ALIASES[canonical] if name in frame.columns), None)
    if match is None and required:
        raise KeyError(
            f"GSE314342 metadata is missing {canonical!r}; tried {OBS_ALIASES[canonical]!r}."
        )
    return match


def _as_bool(series: pd.Series, *, default: bool = False) -> pd.Series:
    if series is None:
        return pd.Series(default, index=pd.RangeIndex(0), dtype=bool)
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(default).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(float(default)).astype(float).ne(0)
    normalized = series.fillna(str(default)).astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "t", "yes", "y"})


def normalize_time_label(value: object) -> str:
    text = str(value).strip().lower().replace(" ", "").replace("_", "")
    aliases = {
        "rest": "Rest",
        "unstim": "Rest",
        "unstimulated": "Rest",
        "stim8hr": "Stim8hr",
        "8hr": "Stim8hr",
        "8h": "Stim8hr",
        "stim48hr": "Stim48hr",
        "48hr": "Stim48hr",
        "48h": "Stim48hr",
    }
    if text not in aliases:
        raise ValueError(f"Unknown GSE314342 condition {value!r}.")
    return aliases[text]


def canonicalize_obs(
    obs: pd.DataFrame,
    *,
    donor_aliases: Mapping[str, str] | None,
    late_time: LateTimeResolution,
    source_name: str,
    sample_id: str | None = None,
    original_donor_id: str | None = None,
    time_label: str | None = None,
    run_id: str | None = None,
) -> pd.DataFrame:
    """Return filtered, canonical cohort metadata without selecting on outcomes."""
    guide_col = _resolve_column(obs, "guide_id")
    type_col = _resolve_column(obs, "guide_type")
    target_col = _resolve_column(obs, "target_gene")
    donor_col = _resolve_column(obs, "donor_id", required=sample_id is None)
    time_col = _resolve_column(obs, "time_label", required=time_label is None)
    low_quality_col = _resolve_column(obs, "low_quality")

    frame = pd.DataFrame(index=obs.index.copy())
    frame["_source_position"] = np.arange(len(obs), dtype=np.int64)
    frame["guide_id"] = obs[guide_col].astype("string")
    guide_type = obs[type_col].astype("string").str.strip().str.lower().str.replace("_", "-")
    guide_type = guide_type.replace(
        {
            "target": "targeting",
            "non-target": "non-targeting",
            "nontargeting": "non-targeting",
            "ntc": "non-targeting",
            "control": "non-targeting",
        }
    )
    frame["guide_type"] = guide_type
    frame["target_gene"] = obs[target_col].astype("string").str.strip()
    target_id_col = _resolve_column(obs, "target_gene_id", required=False)
    frame["target_gene_id"] = (
        obs[target_id_col].astype("string").str.strip()
        if target_id_col is not None
        else pd.Series(pd.NA, index=obs.index, dtype="string")
    )

    if sample_id is None:
        assert donor_col is not None
        frame["original_donor_id"] = obs[donor_col].astype(str)
        aliases = dict(donor_aliases or {})
        unknown_donors = sorted(set(frame["original_donor_id"]) - set(aliases))
        if unknown_donors:
            raise KeyError(f"Missing stable donor aliases for: {unknown_donors}")
        frame["sample_id"] = frame["original_donor_id"].map(aliases)
    else:
        frame["sample_id"] = str(sample_id)
        frame["original_donor_id"] = str(original_donor_id or sample_id)

    if time_label is None:
        assert time_col is not None
        frame["time_label"] = obs[time_col].map(normalize_time_label)
    else:
        frame["time_label"] = normalize_time_label(time_label)
    physical_times = dict(DEFAULT_PHYSICAL_TIMES)
    physical_times["Stim48hr"] = late_time.physical_time_hours
    frame["physical_time"] = frame["time_label"].map(physical_times).astype(float)
    frame["low_quality"] = _as_bool(obs[low_quality_col]).to_numpy()

    guide_group_col = _resolve_column(obs, "guide_group", required=False)
    if guide_group_col is not None:
        guide_group = obs[guide_group_col].astype("string").str.strip()
        normalized_group = (
            guide_group.str.lower()
            .str.replace("_", " ", regex=False)
            .str.replace("-", " ", regex=False)
            .str.replace(r"\s+", " ", regex=True)
        )
        single_guide = normalized_group.str.contains("single", na=False) & ~normalized_group.str.contains(
            "multi|no sgrna|unassigned", regex=True, na=False
        )
        frame["guide_group"] = guide_group
    else:
        single_guide = pd.Series(True, index=frame.index, dtype=bool)

    normalized_guide_id = (
        frame["guide_id"].astype("string").str.strip().str.lower().str.replace("-", "_")
    )
    multi_guide = normalized_guide_id.isin(
        {"multi_guide", "multi_sgrna", "multiple_sgrna", "multiple_guides"}
    )

    keep = (
        ~frame["low_quality"]
        & single_guide
        & frame["guide_type"].isin(["targeting", "non-targeting"])
        & frame["guide_id"].notna()
        & frame["guide_id"].astype(str).str.strip().ne("")
        & ~multi_guide
    )
    frame = frame.loc[keep].copy()
    frame["guide_id"] = frame["guide_id"].astype(str)
    frame["is_control"] = frame["guide_type"].eq("non-targeting")
    missing_target = (
        ~frame["is_control"]
        & (
            frame["target_gene"].isna()
            | frame["target_gene"].astype(str).str.strip().isin({"", "nan", "None"})
        )
    )
    if missing_target.any():
        examples = frame.loc[missing_target, "guide_id"].head().tolist()
        raise ValueError(f"Targeting guides lack curated target_gene_name: {examples!r}")
    frame.loc[frame["is_control"], "target_gene"] = NTC_EMBEDDING_ID
    frame.loc[frame["is_control"], "target_gene_id"] = NTC_EMBEDDING_ID
    frame["embedding_id"] = frame["target_gene"].astype(str)
    frame["perturbation_id"] = frame["guide_id"]
    frame["context_group_id"] = frame["sample_id"]
    frame["view_id"] = frame["sample_id"] + "::" + frame["guide_id"]
    frame["source_file"] = str(source_name)

    optional_defaults = {
        "lane_id": "",
        "run_id": str(run_id or ""),
        "guide_umi_count": np.nan,
        "offtarget_flag": False,
        "guide_effective_rest": False,
        "guide_effective_8h": False,
        "guide_effective_48h": False,
    }
    for canonical, default in optional_defaults.items():
        column = _resolve_column(obs, canonical, required=False)
        values = obs.loc[frame.index, column] if column is not None else pd.Series(default, index=frame.index)
        if canonical in {
            "offtarget_flag",
            "guide_effective_rest",
            "guide_effective_8h",
            "guide_effective_48h",
        }:
            values = _as_bool(values)
        frame[canonical] = values.to_numpy()
    return frame


def stable_donor_aliases(donor_ids: Iterable[object]) -> dict[str, str]:
    donors = sorted({str(value) for value in donor_ids})
    if len(donors) != 4:
        raise ValueError(f"GSE314342 expects four donors, found {donors!r}.")
    return {donor: f"D{idx + 1}" for idx, donor in enumerate(donors)}


def build_mass_and_count_tables(
    full_obs: pd.DataFrame,
    *,
    alpha: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct within-donor guide masses and source-exposure count blocks."""
    if alpha <= 0:
        raise ValueError("Mass pseudocount alpha must be positive.")
    group_cols = ["sample_id", "guide_id", "time_label"]
    if "count" in full_obs.columns:
        counts = (
            full_obs.groupby(group_cols, observed=True)["count"]
            .sum()
            .astype(int)
            .reset_index()
        )
    else:
        counts = full_obs.groupby(group_cols, observed=True).size().rename("count").reset_index()
    guide_totals = counts.groupby(["sample_id", "time_label"], observed=True)["guide_id"].transform("nunique")
    captured_totals = counts.groupby(["sample_id", "time_label"], observed=True)["count"].transform("sum")
    counts["mass_value"] = (counts["count"] + alpha) / (captured_totals + alpha * guide_totals)
    sums = counts.groupby(["sample_id", "time_label"], observed=True)["mass_value"].sum()
    if not np.allclose(sums.to_numpy(), 1.0, rtol=1e-8, atol=1e-10):
        raise AssertionError("Within-donor guide masses do not sum to one.")

    source = counts[counts["time_label"].eq("Rest")].copy()
    source = source.rename(columns={"mass_value": "exposure", "count": "rest_count"})
    rows: list[dict[str, object]] = []
    count_lookup = counts.set_index(["sample_id", "time_label", "guide_id"])["count"]
    for donor, donor_source in source.groupby("sample_id", observed=True):
        donor_source = donor_source.sort_values("guide_id")
        for time_label in TIME_LABELS[1:]:
            target_counts = np.array(
                [float(count_lookup.get((donor, time_label, guide_id), 0.0)) for guide_id in donor_source["guide_id"]],
                dtype=float,
            )
            for guide_id, exposure, count in zip(
                donor_source["guide_id"], donor_source["exposure"], target_counts
            ):
                rows.append(
                    {
                        "context_group_id": str(donor),
                        "time_label": time_label,
                        "guide_id": str(guide_id),
                        "exposure": float(exposure),
                        "count": int(count),
                        "n_total": int(target_counts.sum()),
                    }
                )
    return counts, pd.DataFrame(rows)


def build_measure_manifest(full_obs: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    annotation_cols = [
        "sample_id",
        "original_donor_id",
        "guide_id",
        "embedding_id",
        "target_gene",
        "target_gene_id",
        "guide_type",
        "is_control",
        "context_group_id",
        "guide_effective_rest",
        "guide_effective_8h",
        "guide_effective_48h",
        "offtarget_flag",
    ]
    annotation_cols.extend(
        column
        for column in ("library_flag", "putative_bidirectional_promoter")
        if column in full_obs.columns
    )
    manifest = full_obs[annotation_cols].drop_duplicates()
    if manifest.duplicated(["sample_id", "guide_id"]).any():
        raise ValueError("Guide annotations vary within a donor finite-measure key.")
    availability = counts.pivot_table(
        index=["sample_id", "guide_id"],
        columns="time_label",
        values="count",
        fill_value=0,
        aggfunc="sum",
    ).reset_index()
    for label in TIME_LABELS:
        if label not in availability:
            availability[label] = 0
        availability[f"has_{label}"] = availability[label].gt(0)
    return manifest.merge(availability, on=["sample_id", "guide_id"], how="left", validate="one_to_one")


__all__ = [
    "ACCESSION",
    "DEFAULT_PHYSICAL_TIMES",
    "LateTimeResolution",
    "NTC_EMBEDDING_ID",
    "TIME_LABELS",
    "build_mass_and_count_tables",
    "build_measure_manifest",
    "canonicalize_obs",
    "normalize_time_label",
    "stable_donor_aliases",
]
