from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credo.data.hnscc import build_study_from_split


pytestmark = pytest.mark.unit


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


def test_full_obs_mass_scope_drops_perturbations_outside_split_catalog() -> None:
    # A perturbation whose cells live only in the train split must not leak into the mass
    # table (built from all obs under mass_scope="full_obs") for a test-split study --
    # PerturbSeqDynamicsData.validate() would otherwise reject it as outside the catalog.
    rows: list[dict] = []
    latent: list[list[float]] = []
    i = 0
    for pid, is_control, spl in [
        ("ctrl", True, "test"),
        ("GeneA_sg1", False, "test"),
        ("GeneB_sg1", False, "train"),
    ]:
        for time_label in ["P4", "P60"]:
            for _ in range(2):
                rows.append(
                    {
                        "cell_id": f"{pid}_{time_label}_{i}",
                        "perturbation_id": pid,
                        "time_label": time_label,
                        "sample_id": "D1",
                        "is_control": is_control,
                        "mass_value": 10.0,
                        "_split": spl,
                    }
                )
                latent.append([float(i), 0.0])
                i += 1
    obs = pd.DataFrame(rows)
    latent_arr = np.asarray(latent, dtype=np.float32)

    study = build_study_from_split(
        obs,
        latent_arr,
        split=obs["_split"],
        split_name="test",
        mass_value_col="mass_value",
        mass_scope="full_obs",
        mass_mode="group_total",
    )

    catalog = set(study.catalog.perturbation_ids)
    assert catalog == {"ctrl", "GeneA_sg1"}
    mass_pids = set(study.mass_table.df["perturbation_id"].astype(str))
    assert "GeneB_sg1" not in mass_pids
    # global-abundance intent preserved for in-catalog perturbations
    assert study.mass_table.df.attrs["mass_mode"].startswith("full_obs:")
