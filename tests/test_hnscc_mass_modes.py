from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credo.data.hnscc import build_study_from_split


def _obs_with_mass(values: list[float] | None = None) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    rows = []
    latent = []
    i = 0
    for pid, is_control in [("ctrl", True), ("GeneA_sg1", False)]:
        for time_label in ["P4", "P60"]:
            for cell_i in range(2):
                row = {
                    "cell_id": f"{pid}_{time_label}_{cell_i}",
                    "perturbation_id": pid,
                    "time_label": time_label,
                    "sample_id": "D1",
                    "is_control": is_control,
                }
                if values is not None:
                    row["mass_value"] = float(values[i])
                rows.append(row)
                latent.append([float(i), 0.0])
                i += 1
    obs = pd.DataFrame(rows)
    latent_arr = np.asarray(latent, dtype=np.float32)
    split = pd.Series(["test"] * len(obs), index=obs.index)
    return obs, latent_arr, split


def test_hnscc_group_total_mass_not_summed_over_cells() -> None:
    obs, latent, split = _obs_with_mass([10.0] * 8)

    study = build_study_from_split(
        obs,
        latent,
        split=split,
        split_name="test",
        mass_value_col="mass_value",
        mass_mode="group_total",
    )

    assert study.mass_table.get("ctrl", "P4", "D1") == 10.0
    assert study.mass_table.df.attrs["mass_mode"] == "subset_only:mass_value:group_total"
    assert study.mass_table.df.attrs["requested_mass_mode"] == "group_total"
    assert study.mass_table.df.attrs["mass_mode_resolution_reason"] == "explicit_group_total"


def test_hnscc_per_cell_contribution_mass_sums() -> None:
    obs, latent, split = _obs_with_mass([0.5] * 8)

    study = build_study_from_split(
        obs,
        latent,
        split=split,
        split_name="test",
        mass_value_col="mass_value",
        mass_mode="per_cell_contribution",
    )

    assert study.mass_table.get("ctrl", "P4", "D1") == 1.0
    assert study.mass_table.df.attrs["mass_mode"] == "subset_only:mass_value:per_cell_contribution"
    assert study.mass_table.df.attrs["requested_mass_mode"] == "per_cell_contribution"


def test_hnscc_explicit_mass_mode_missing_column_raises() -> None:
    obs, latent, split = _obs_with_mass(None)

    with pytest.raises(KeyError, match="mass_value_col"):
        build_study_from_split(
            obs,
            latent,
            split=split,
            split_name="test",
            mass_value_col="missing",
            mass_mode="group_total",
        )


def test_hnscc_auto_mass_mode_rejects_ambiguous_repeated_group_values() -> None:
    obs, latent, split = _obs_with_mass([10.0] * 8)

    with pytest.raises(ValueError, match="constant within at least one multi-cell group"):
        build_study_from_split(
            obs,
            latent,
            split=split,
            split_name="test",
            mass_value_col="mass_value",
            mass_mode="auto",
        )


def test_hnscc_count_mass_mode_ignores_mass_column() -> None:
    obs, latent, split = _obs_with_mass([100.0] * 8)

    study = build_study_from_split(
        obs,
        latent,
        split=split,
        split_name="test",
        mass_value_col="mass_value",
        mass_mode="count",
    )

    assert study.mass_table.get("ctrl", "P4", "D1") == 2.0
    assert study.mass_table.df.attrs["mass_mode"] == "subset_only:count"
    assert study.mass_table.df.attrs["requested_mass_mode"] == "count"
