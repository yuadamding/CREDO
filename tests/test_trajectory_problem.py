from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from credo.data.core import (
    CellStateTable,
    ExposureTable,
    FiniteMeasure,
    MassTable,
    POOLED_SAMPLE_ID,
    PerturbSeqDynamicsData,
    PerturbationCatalog,
    ReplicateCountTable,
    SparseTrajectoryProblem,
    TimeAxis,
    TrajectoryProblem,
)
from credo.models.simulator import initialise_particles_from_trajectory
from credo.models.simulator import initialise_particles


def _three_time_data() -> PerturbSeqDynamicsData:
    labels = ["90m", "6h", "10h"]
    rows = []
    latent = []
    for pid_i, pid in enumerate(["LPS__mono", "ctrl"]):
        for time_i, label in enumerate(labels):
            for cell_i in range(4):
                rows.append(
                    {
                        "cell_id": f"{pid}_{label}_{cell_i}",
                        "perturbation_id": pid,
                        "time_label": label,
                        "sample_id": "D1",
                    }
                )
                latent.append([float(pid_i), float(time_i), float(cell_i)])
    cell_df = pd.DataFrame(rows)
    latent_arr = np.asarray(latent, dtype=np.float32)
    mass_df = (
        cell_df.groupby(["perturbation_id", "time_label", "sample_id"], observed=True)
        .size()
        .rename("mass")
        .reset_index()
    )
    return PerturbSeqDynamicsData(
        time_axis=TimeAxis(labels=labels, physical_times=[1.5, 6.0, 10.0]),
        catalog=PerturbationCatalog(["LPS__mono", "ctrl"], ["ctrl"]),
        cell_state=CellStateTable(cell_df, latent_arr),
        mass_table=MassTable(mass_df),
    )


def test_trajectory_problem_three_times() -> None:
    data = _three_time_data()
    trajectory = data.to_trajectory_problem()

    assert trajectory.time_labels == ["90m", "6h", "10h"]
    assert trajectory.observed_taus == [0.0, (6.0 - 1.5) / (10.0 - 1.5), 1.0]
    assert trajectory.interval_pairs() == [("90m", "6h"), ("6h", "10h")]
    assert set(trajectory.keys) == {"LPS__mono", "ctrl"}
    assert trajectory.get("6h", "LPS__mono").n_atoms == 4


def test_time_axis_rejects_duplicate_labels() -> None:
    with pytest.raises(ValueError, match="labels must be unique"):
        TimeAxis(labels=["90m", "90m"], physical_times=[1.5, 6.0])


def test_finite_measure_rejects_nonfinite_and_negative_values() -> None:
    with pytest.raises(ValueError, match="support contains"):
        FiniteMeasure(
            support=np.asarray([[0.0], [np.nan]], dtype=np.float32),
            weights=np.ones(2, dtype=np.float32),
            total_mass=2.0,
        )
    with pytest.raises(ValueError, match="weights must be nonnegative"):
        FiniteMeasure(
            support=np.zeros((2, 1), dtype=np.float32),
            weights=np.asarray([3.0, -1.0], dtype=np.float32),
            total_mass=2.0,
        )


def test_mass_table_get_pooled_requires_rows() -> None:
    table = MassTable(
        pd.DataFrame(
            [{"perturbation_id": "ctrl", "time_label": "t0", "sample_id": "D1", "mass": 1.0}]
        )
    )

    with pytest.raises(KeyError, match="No mass rows"):
        table.get_pooled("missing", "t0")


def test_mass_table_rejects_mixed_pooled_and_sample_specific_rows() -> None:
    with pytest.raises(ValueError, match="mixes pooled and sample-specific rows"):
        MassTable(
            pd.DataFrame(
                [
                    {"perturbation_id": "ctrl", "time_label": "t0", "sample_id": "pooled", "mass": 1.0},
                    {"perturbation_id": "ctrl", "time_label": "t0", "sample_id": "D1", "mass": 1.0},
                ]
            )
        )


def test_mass_table_get_pooled_prefers_explicit_pooled_row() -> None:
    table = MassTable(
        pd.DataFrame(
            [
                {"perturbation_id": "ctrl", "time_label": "t0", "sample_id": POOLED_SAMPLE_ID, "mass": 3.0},
                {"perturbation_id": "ctrl", "time_label": "t1", "sample_id": "D1", "mass": 1.0},
                {"perturbation_id": "ctrl", "time_label": "t1", "sample_id": "D2", "mass": 2.0},
            ]
        )
    )

    assert table.get_pooled("ctrl", "t0") == 3.0
    assert table.get_pooled("ctrl", "t1") == 3.0


def test_mass_table_keys_are_string_canonicalized() -> None:
    table = MassTable(
        pd.DataFrame(
            [
                {"perturbation_id": 1, "time_label": 0, "sample_id": 7, "mass": 2.0},
            ]
        )
    )

    assert table.get("1", "0", "7") == 2.0
    assert table.get_pooled("1", "0") == 2.0


def test_mass_table_rejects_string_equivalent_duplicate_keys() -> None:
    with pytest.raises(ValueError, match="Duplicate MassTable row"):
        MassTable(
            pd.DataFrame(
                [
                    {"perturbation_id": "ctrl", "time_label": "t0", "sample_id": 1, "mass": 1.0},
                    {"perturbation_id": "ctrl", "time_label": "t0", "sample_id": "1", "mass": 1.0},
                ]
            )
        )


def test_mass_table_rejects_string_equivalent_mixed_mass_modes() -> None:
    with pytest.raises(ValueError, match="mixes pooled and sample-specific rows"):
        MassTable(
            pd.DataFrame(
                [
                    {"perturbation_id": 1, "time_label": 0, "sample_id": POOLED_SAMPLE_ID, "mass": 1.0},
                    {"perturbation_id": "1", "time_label": "0", "sample_id": "D1", "mass": 1.0},
                ]
            )
        )


def test_exposure_and_count_tables_string_canonicalize_keys() -> None:
    exposure = ExposureTable(
        pd.DataFrame(
            [
                {"perturbation_id": 1, "library_batch": 7, "exposure": 0.25},
            ]
        )
    )
    counts = ReplicateCountTable(
        pd.DataFrame(
            [
                {
                    "sample_id": 3,
                    "time_label": 0,
                    "library_batch": 7,
                    "perturbation_id": 1,
                    "count": 5,
                    "n_total_sample": 5,
                }
            ]
        )
    )

    assert exposure.get("1", "7") == 0.25
    matrix, sample_ids, totals = counts.get_count_matrix("0", ["1"])
    assert sample_ids == ["3"]
    assert matrix.tolist() == [[5.0]]
    assert totals.tolist() == [5.0]


def test_replicate_count_table_rejects_fractional_counts() -> None:
    with pytest.raises(ValueError, match="integer-like"):
        ReplicateCountTable(
            pd.DataFrame(
                [
                    {
                        "sample_id": "D1",
                        "time_label": "t0",
                        "library_batch": "b1",
                        "perturbation_id": "ctrl",
                        "count": 1.5,
                        "n_total_sample": 2,
                    }
                ]
            )
        )


def test_mass_table_rejects_multiple_pooled_sentinels() -> None:
    with pytest.raises(ValueError, match="multiple pooled sentinel"):
        MassTable(
            pd.DataFrame(
                [
                    {"perturbation_id": "ctrl", "time_label": "t0", "sample_id": POOLED_SAMPLE_ID, "mass": 1.0},
                    {"perturbation_id": "ctrl", "time_label": "t0", "sample_id": "pooled", "mass": 1.0},
                ]
            )
        )


def test_pooled_measure_uses_sample_specific_mass_weights() -> None:
    cell_df = pd.DataFrame(
        [
            {"cell_id": "d1_a", "perturbation_id": "ctrl", "time_label": "t0", "sample_id": "D1"},
            {"cell_id": "d1_b", "perturbation_id": "ctrl", "time_label": "t0", "sample_id": "D1"},
            {"cell_id": "d2_a", "perturbation_id": "ctrl", "time_label": "t0", "sample_id": "D2"},
            {"cell_id": "d1_c", "perturbation_id": "ctrl", "time_label": "t1", "sample_id": "D1"},
            {"cell_id": "d2_b", "perturbation_id": "ctrl", "time_label": "t1", "sample_id": "D2"},
        ]
    )
    latent = np.arange(len(cell_df), dtype=np.float32).reshape(-1, 1)
    mass_df = pd.DataFrame(
        [
            {"perturbation_id": "ctrl", "time_label": "t0", "sample_id": "D1", "mass": 10.0},
            {"perturbation_id": "ctrl", "time_label": "t0", "sample_id": "D2", "mass": 1.0},
            {"perturbation_id": "ctrl", "time_label": "t1", "sample_id": "D1", "mass": 2.0},
            {"perturbation_id": "ctrl", "time_label": "t1", "sample_id": "D2", "mass": 3.0},
        ]
    )
    data = PerturbSeqDynamicsData(
        time_axis=TimeAxis(labels=["t0", "t1"], physical_times=[0.0, 1.0]),
        catalog=PerturbationCatalog(["ctrl"], ["ctrl"]),
        cell_state=CellStateTable(cell_df, latent),
        mass_table=MassTable(mass_df),
    )

    measure = data.build_measure("ctrl", "t0")

    assert np.allclose(measure.weights, np.asarray([5.0, 5.0, 1.0]))
    assert measure.total_mass == 11.0


def test_perturbseq_data_validation_rejects_missing_mass_rows() -> None:
    cell_df = pd.DataFrame(
        [
            {"cell_id": "c1", "perturbation_id": "ctrl", "time_label": "t0", "sample_id": "D1"},
            {"cell_id": "c2", "perturbation_id": "ctrl", "time_label": "t1", "sample_id": "D1"},
        ]
    )
    mass_df = pd.DataFrame(
        [{"perturbation_id": "ctrl", "time_label": "t0", "sample_id": "D1", "mass": 1.0}]
    )

    with pytest.raises(ValueError, match="Missing MassTable rows"):
        PerturbSeqDynamicsData(
            time_axis=TimeAxis(labels=["t0", "t1"], physical_times=[0.0, 1.0]),
            catalog=PerturbationCatalog(["ctrl"], ["ctrl"]),
            cell_state=CellStateTable(cell_df, np.zeros((2, 1), dtype=np.float32)),
            mass_table=MassTable(mass_df),
        )


def test_perturbseq_data_validation_allows_pooled_mass_rows() -> None:
    cell_df = pd.DataFrame(
        [
            {"cell_id": "c1", "perturbation_id": "ctrl", "time_label": "t0", "sample_id": "D1"},
            {"cell_id": "c2", "perturbation_id": "ctrl", "time_label": "t1", "sample_id": "D1"},
        ]
    )
    mass_df = pd.DataFrame(
        [
            {"perturbation_id": "ctrl", "time_label": "t0", "sample_id": "pooled", "mass": 1.0},
            {"perturbation_id": "ctrl", "time_label": "t1", "sample_id": "pooled", "mass": 1.0},
        ]
    )

    data = PerturbSeqDynamicsData(
        time_axis=TimeAxis(labels=["t0", "t1"], physical_times=[0.0, 1.0]),
        catalog=PerturbationCatalog(["ctrl"], ["ctrl"]),
        cell_state=CellStateTable(cell_df, np.zeros((2, 1), dtype=np.float32)),
        mass_table=MassTable(mass_df),
    )

    assert data.build_measure("ctrl", "t0").total_mass == 1.0
    assert data.build_measure("ctrl", "t0", sample_id="D1").total_mass == 1.0


def test_perturbseq_data_rejects_multi_sample_pooled_mass_fallback() -> None:
    cell_df = pd.DataFrame(
        [
            {"cell_id": "c1", "perturbation_id": "ctrl", "time_label": "t0", "sample_id": "D1"},
            {"cell_id": "c2", "perturbation_id": "ctrl", "time_label": "t0", "sample_id": "D2"},
            {"cell_id": "c3", "perturbation_id": "ctrl", "time_label": "t1", "sample_id": "D1"},
            {"cell_id": "c4", "perturbation_id": "ctrl", "time_label": "t1", "sample_id": "D2"},
        ]
    )
    mass_df = pd.DataFrame(
        [
            {"perturbation_id": "ctrl", "time_label": "t0", "sample_id": POOLED_SAMPLE_ID, "mass": 2.0},
            {"perturbation_id": "ctrl", "time_label": "t1", "sample_id": POOLED_SAMPLE_ID, "mass": 2.0},
        ]
    )

    with pytest.raises(ValueError, match="Sample-specific MassTable rows are required"):
        PerturbSeqDynamicsData(
            time_axis=TimeAxis(labels=["t0", "t1"], physical_times=[0.0, 1.0]),
            catalog=PerturbationCatalog(["ctrl"], ["ctrl"]),
            cell_state=CellStateTable(cell_df, np.zeros((4, 1), dtype=np.float32)),
            mass_table=MassTable(mass_df),
        )


def test_trajectory_to_endpoint_backward_compatibility() -> None:
    data = _three_time_data()
    trajectory = data.to_trajectory_problem()

    from_trajectory = trajectory.to_endpoint_problem("90m", "10h")
    direct = data.to_endpoint_problem(
        perturbation_ids=["LPS__mono", "ctrl"],
        initial_label="90m",
        terminal_label="10h",
    )

    for pid in ["LPS__mono", "ctrl"]:
        assert np.allclose(from_trajectory.initial[pid].support, direct.initial[pid].support)
        assert np.allclose(from_trajectory.initial[pid].weights, direct.initial[pid].weights)
        assert np.allclose(from_trajectory.terminal[pid].support, direct.terminal[pid].support)
        assert np.allclose(from_trajectory.terminal[pid].weights, direct.terminal[pid].weights)


def test_trajectory_problem_requires_common_keys() -> None:
    data = _three_time_data()

    sparse = data.to_trajectory_problem(require_all_times=False)

    assert isinstance(sparse, SparseTrajectoryProblem)
    assert sparse.target_keys("90m", "10h") == {"LPS__mono", "ctrl"}


def test_to_trajectory_problem_rejects_empty_time_labels() -> None:
    data = _three_time_data()

    with pytest.raises(ValueError, match="Need at least two time labels"):
        data.to_trajectory_problem(time_labels=[])


def test_trajectory_problem_rejects_mixed_key_types() -> None:
    measure = FiniteMeasure(
        support=np.zeros((2, 1), dtype=np.float32),
        weights=np.ones(2, dtype=np.float32),
        total_mass=2.0,
    )

    with pytest.raises(ValueError, match="all pooled ids or all sample-aware tuples"):
        TrajectoryProblem(
            measures={
                "t0": {"pert": measure, ("D1", "pert"): measure},
                "t1": {"pert": measure, ("D1", "pert"): measure},
            },
            catalog=PerturbationCatalog(["pert"], ["pert"]),
            time_axis=TimeAxis(["t0", "t1"], [0.0, 1.0]),
            time_labels=["t0", "t1"],
        )


def test_sparse_trajectory_preserves_incomplete_donor_time_keys() -> None:
    labels = ["90m", "6h", "10h"]
    rows = []
    latent = []
    for label in labels:
        donors = ["D1", "D2"] if label != "6h" else ["D1"]
        for donor in donors:
            for cell_i in range(2):
                rows.append(
                    {
                        "cell_id": f"{label}_{donor}_{cell_i}",
                        "perturbation_id": "LPS__mono",
                        "time_label": label,
                        "sample_id": donor,
                    }
                )
                latent.append([float(len(latent)), 0.0])
    cell_df = pd.DataFrame(rows)
    data = PerturbSeqDynamicsData(
        time_axis=TimeAxis(labels=labels, physical_times=[1.5, 6.0, 10.0]),
        catalog=PerturbationCatalog(["LPS__mono"], ["LPS__mono"]),
        cell_state=CellStateTable(cell_df, np.asarray(latent, dtype=np.float32)),
        mass_table=MassTable(
            cell_df.groupby(["perturbation_id", "time_label", "sample_id"], observed=True)
            .size()
            .rename("mass")
            .reset_index()
        ),
    )

    sparse = data.to_sparse_trajectory_problem(by_sample=True)

    assert sparse.available_keys("90m") == {("D1", "LPS__mono"), ("D2", "LPS__mono")}
    assert sparse.available_keys("6h") == {("D1", "LPS__mono")}
    assert sparse.target_keys("90m", "6h") == {("D1", "LPS__mono")}


def test_same_seed_trajectory_initialization_starts_identically() -> None:
    data = _three_time_data()
    trajectory = data.to_trajectory_problem()

    z0_a, logw0_a, log_m0_a = initialise_particles_from_trajectory(
        trajectory,
        source_label="90m",
        perturbation_ids=["LPS__mono"],
        n_particles=8,
        seed=17,
    )
    z0_b, logw0_b, log_m0_b = initialise_particles_from_trajectory(
        trajectory,
        source_label="90m",
        perturbation_ids=["LPS__mono"],
        n_particles=8,
        seed=17,
    )

    assert torch.equal(z0_a, z0_b)
    assert torch.equal(logw0_a, logw0_b)
    assert torch.equal(log_m0_a, log_m0_b)


def test_endpoint_initializer_matches_legacy_uniform_sampling() -> None:
    data = _three_time_data()
    endpoint = data.to_endpoint_problem(
        perturbation_ids=["LPS__mono", "ctrl"],
        initial_label="90m",
        terminal_label="10h",
    )

    z0, logw0, log_m0 = initialise_particles(
        endpoint,
        perturbation_ids=["LPS__mono", "ctrl"],
        n_particles=8,
        seed=23,
    )

    torch.manual_seed(23)
    expected_z = torch.zeros_like(z0)
    expected_logw = torch.zeros_like(logw0)
    expected_log_m = torch.zeros_like(log_m0)
    for g, pid in enumerate(["LPS__mono", "ctrl"]):
        mu = endpoint.initial[pid]
        support = torch.tensor(mu.support, dtype=torch.float32)
        idx = torch.randint(0, len(support), (8,))
        expected_z[g] = support[idx]
        expected_logw[g] = torch.full((8,), -np.log(8.0))
        expected_log_m[g] = torch.tensor(np.log(mu.total_mass), dtype=torch.float32)

    assert torch.equal(z0, expected_z)
    assert torch.equal(logw0, expected_logw)
    assert torch.equal(log_m0, expected_log_m)
