from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from credo.data.core import (
    CellStateTable,
    FiniteMeasure,
    MassTable,
    PerturbSeqDynamicsData,
    PerturbationCatalog,
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

    with pytest.raises(NotImplementedError, match="requires common measure keys"):
        data.to_trajectory_problem(require_all_times=False)


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
