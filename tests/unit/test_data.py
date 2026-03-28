"""Unit tests for data layer: validation, pooling, filters, measure construction."""
import numpy as np
import pandas as pd
import pytest

from cape.data.core import (
    TimeAxis, PerturbationCatalog, CellStateTable, MassTable,
    FiniteMeasure, EndpointProblem, PerturbSeqDynamicsData, ExposureTable,
)
from cape.data.filters import filter_state_supported_perturbations


# ---------------------------------------------------------------------------
# TimeAxis
# ---------------------------------------------------------------------------

def test_time_axis_tau():
    ta = TimeAxis(labels=["P4", "P60"], physical_times=[4.0, 60.0])
    assert ta.tau("P4") == pytest.approx(0.0)
    assert ta.tau("P60") == pytest.approx(1.0)


def test_time_axis_rejects_non_monotone():
    with pytest.raises(AssertionError):
        TimeAxis(labels=["A", "B"], physical_times=[10.0, 5.0])


# ---------------------------------------------------------------------------
# PerturbationCatalog
# ---------------------------------------------------------------------------

def test_catalog_is_control():
    cat = PerturbationCatalog(
        perturbation_ids=["ctrl", "gene1", "gene2"],
        control_ids=["ctrl"],
    )
    assert cat.is_control("ctrl")
    assert not cat.is_control("gene1")


def test_catalog_rejects_missing_control():
    with pytest.raises(AssertionError):
        PerturbationCatalog(
            perturbation_ids=["gene1"],
            control_ids=["ctrl_missing"],
        )


# ---------------------------------------------------------------------------
# CellStateTable
# ---------------------------------------------------------------------------

def _make_cell_table(n=30, d=4, n_perts=3, time_labels=("P4", "P60")):
    rows, latent = [], []
    cid = 0
    for tl in time_labels:
        for i in range(n_perts):
            pid = "ctrl" if i == 0 else f"gene_{i}"
            for _ in range(n):
                rows.append({"cell_id": f"c{cid}", "perturbation_id": pid,
                             "time_label": tl, "sample_id": "pooled"})
                latent.append(np.random.randn(d))
                cid += 1
    return CellStateTable(df=pd.DataFrame(rows), latent=np.array(latent))


def test_cell_state_shape():
    ct = _make_cell_table(n=20, d=4, n_perts=3)
    assert ct.n_cells == 20 * 3 * 2
    assert ct.latent_dim == 4


def test_cell_state_select():
    ct = _make_cell_table(n=20, d=4, n_perts=3)
    p4 = ct.select_time("P4")
    assert p4.n_cells == 20 * 3


def test_cell_state_rejects_bad_columns():
    df = pd.DataFrame({"cell_id": ["c1"], "perturbation_id": ["g1"]})
    with pytest.raises(AssertionError):
        CellStateTable(df=df, latent=np.zeros((1, 4)))


# ---------------------------------------------------------------------------
# FiniteMeasure
# ---------------------------------------------------------------------------

def test_finite_measure_weights_sum():
    support = np.random.randn(10, 4)
    weights = np.ones(10) * 2.0
    mu = FiniteMeasure(support=support, weights=weights, total_mass=20.0)
    assert mu.n_atoms == 10
    assert mu.latent_dim == 4
    np.testing.assert_allclose(mu.normalized_weights.sum(), 1.0)


def test_finite_measure_rejects_wrong_mass():
    support = np.random.randn(5, 4)
    with pytest.raises(AssertionError):
        FiniteMeasure(support=support, weights=np.ones(5), total_mass=99.0)


# ---------------------------------------------------------------------------
# MassTable
# ---------------------------------------------------------------------------

def _make_mass_table(pids, time_labels, mass=1000.0):
    rows = [{"perturbation_id": p, "time_label": t, "sample_id": "pooled", "mass": mass}
            for p in pids for t in time_labels]
    return MassTable(df=pd.DataFrame(rows))


def test_mass_table_get_pooled():
    mt = _make_mass_table(["ctrl", "gene1"], ["P4", "P60"])
    assert mt.get_pooled("ctrl", "P4") == 1000.0


def test_mass_table_rejects_negative():
    with pytest.raises(AssertionError):
        MassTable(df=pd.DataFrame([
            {"perturbation_id": "g", "time_label": "P4", "sample_id": "s", "mass": -1.0}
        ]))


# ---------------------------------------------------------------------------
# Sparse support filter
# ---------------------------------------------------------------------------

def _make_study_data(n_cells=50):
    np.random.seed(0)
    pids = ["ctrl", "gene1", "gene2_sparse"]
    ctrl_ids = ["ctrl"]
    rows, latent = [], []
    cid = 0
    mass_rows = []
    # gene2_sparse has only 5 cells at P60
    n_by_pid_time = {
        ("ctrl", "P4"): n_cells, ("ctrl", "P60"): n_cells,
        ("gene1", "P4"): n_cells, ("gene1", "P60"): n_cells,
        ("gene2_sparse", "P4"): n_cells, ("gene2_sparse", "P60"): 5,
    }
    for (pid, tl), n in n_by_pid_time.items():
        for _ in range(n):
            rows.append({"cell_id": f"c{cid}", "perturbation_id": pid,
                         "time_label": tl, "sample_id": "pooled"})
            latent.append(np.random.randn(4))
            cid += 1
        mass_rows.append({"perturbation_id": pid, "time_label": tl,
                          "sample_id": "pooled", "mass": float(n)})

    ta = TimeAxis(labels=["P4", "P60"], physical_times=[4.0, 60.0])
    cat = PerturbationCatalog(perturbation_ids=pids, control_ids=ctrl_ids)
    ct = CellStateTable(df=pd.DataFrame(rows), latent=np.array(latent))
    mt = MassTable(df=pd.DataFrame(mass_rows))
    return PerturbSeqDynamicsData(time_axis=ta, catalog=cat, cell_state=ct, mass_table=mt)


def test_filter_removes_sparse():
    data = _make_study_data(n_cells=50)
    supported = filter_state_supported_perturbations(data, min_cells_p4=20, min_cells_p60=20)
    assert "gene2_sparse" not in supported
    assert "ctrl" in supported
    assert "gene1" in supported


def test_filter_all_supported_with_low_threshold():
    data = _make_study_data(n_cells=50)
    supported = filter_state_supported_perturbations(data, min_cells_p4=1, min_cells_p60=1)
    assert len(supported) == 3


# ---------------------------------------------------------------------------
# PerturbSeqDynamicsData
# ---------------------------------------------------------------------------

def test_to_endpoint_problem():
    data = _make_study_data(n_cells=50)
    supported = filter_state_supported_perturbations(data, min_cells_p4=20, min_cells_p60=20)
    ep = data.to_endpoint_problem(perturbation_ids=supported)
    assert len(ep.perturbation_ids) == len(supported)
    for pid in supported:
        assert pid in ep.initial
        assert pid in ep.terminal
        # Total mass preserved
        assert ep.initial[pid].total_mass > 0
        assert ep.terminal[pid].total_mass > 0


def test_summary_shape():
    data = _make_study_data(n_cells=50)
    s = data.summary()
    assert "n_cells" in s.columns
    assert len(s) == 3 * 2  # 3 pids x 2 time labels
