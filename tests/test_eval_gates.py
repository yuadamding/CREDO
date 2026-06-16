from __future__ import annotations

import pandas as pd
import pytest
import numpy as np

from credo.data.core import FiniteMeasure
from credo.eval.hnscc import cap_measure_atoms, summarize_eval
from credo.eval.gates import ESS_GATE_COLUMNS, append_ess_claim_gate, ess_claim_gate, ess_gate_status


pytestmark = pytest.mark.unit


def test_ess_gate_status_matches_claim_grade_thresholds() -> None:
    kwargs = {
        "ess_warn_frac": 0.2,
        "ess_fail_frac": 0.05,
        "ess_claim_grade_min_frac": 0.1,
        "ess_max_weight_frac_fail": 0.5,
    }

    assert ess_gate_status({"terminal_ess_frac_min": 0.3, "max_weight_frac_mean": 0.2}, **kwargs) == "pass"
    assert ess_gate_status({"terminal_ess_frac_min": 0.15, "max_weight_frac_mean": 0.2}, **kwargs) == "warn"
    assert (
        ess_gate_status({"terminal_ess_frac_min": 0.08, "max_weight_frac_mean": 0.2}, **kwargs)
        == "claim_grade_blocked"
    )
    assert ess_gate_status({"terminal_ess_frac_min": 0.03, "max_weight_frac_mean": 0.2}, **kwargs) == "fail"
    assert ess_gate_status({"terminal_ess_frac_min": 0.3, "max_weight_frac_mean": 0.8}, **kwargs) == "fail"
    assert (
        ess_gate_status(
            {
                "terminal_ess_frac": 0.3,
                "min_ess_frac_over_time": 0.03,
                "max_weight_frac_over_time": 0.2,
            },
            **kwargs,
        )
        == "fail"
    )


def test_ess_gate_blocks_intra_trajectory_collapse_below_claim_floor() -> None:
    """A terminal-recovered run that collapsed mid-trajectory must be blocked.

    terminal ESS (0.15) is above the claim-grade floor (0.10) but the
    intra-trajectory minimum (0.08) is below it; the run must not be reported as
    claim-grade-allowed. Previously this returned 'warn' (lenient-allowed).
    """
    kwargs = {
        "ess_warn_frac": 0.2,
        "ess_fail_frac": 0.05,
        "ess_claim_grade_min_frac": 0.1,
        "ess_max_weight_frac_fail": 0.5,
    }
    metrics = {
        "terminal_ess_frac_min": 0.15,
        "min_ess_frac_over_time": 0.08,
        "max_weight_frac_over_time": 0.2,
    }

    assert ess_gate_status(metrics, **kwargs) == "claim_grade_blocked"

    gate = ess_claim_gate(metrics, **kwargs)
    assert gate["ess_gate_status"] == "claim_grade_blocked"
    assert gate["ess_claim_grade_allowed"] is False
    assert gate["ess_claim_grade_allowed_lenient"] is False
    assert gate["ess_claim_grade_allowed_strict"] is False
    assert "min_ess_frac_over_time" in gate["ess_failed_gates"]


def test_ess_claim_gate_payload_blocks_low_ess_rows() -> None:
    gate = ess_claim_gate({"terminal_ess_frac_min": 0.08, "max_weight_frac_mean": 0.2})

    assert gate["ess_gate_status"] == "claim_grade_blocked"
    assert gate["ess_claim_grade_allowed"] is False
    assert gate["ess_claim_grade_allowed_strict"] is False
    assert gate["ess_claim_grade_allowed_lenient"] is False
    assert gate["ess_failed_gates"] == "terminal_ess_frac_min"
    assert gate["ess_claim_grade_min_frac_used"] == pytest.approx(0.1)


def test_ess_claim_gate_distinguishes_strict_and_lenient_warn_rows() -> None:
    gate = ess_claim_gate({"terminal_ess_frac": 0.15, "max_weight_frac_over_time": 0.2})

    assert gate["ess_gate_status"] == "warn"
    assert gate["ess_claim_grade_allowed"] is True
    assert gate["ess_claim_grade_allowed_strict"] is False
    assert gate["ess_claim_grade_allowed_lenient"] is True


def test_append_ess_claim_gate_adds_stable_columns() -> None:
    frame = pd.DataFrame(
        {
            "perturbation_id": ["a", "b"],
            "terminal_ess_frac_min": [0.25, 0.04],
            "max_weight_frac_mean": [0.2, 0.2],
        }
    )

    out = append_ess_claim_gate(frame)

    assert list(out["ess_gate_status"]) == ["pass", "fail"]
    assert list(out["ess_claim_grade_allowed"]) == [True, False]
    assert list(out["ess_claim_grade_allowed_strict"]) == [True, False]
    assert out.loc[1, "ess_failed_gates"] == "terminal_ess_frac_min"
    assert out["ess_warn_frac_used"].tolist() == [pytest.approx(0.2), pytest.approx(0.2)]


def test_append_ess_claim_gate_is_idempotent() -> None:
    frame = pd.DataFrame(
        {
            "perturbation_id": ["a"],
            "terminal_ess_frac": [0.25],
            "max_weight_frac_over_time": [0.2],
            "ess_gate_status": ["stale"],
        }
    )

    out = append_ess_claim_gate(append_ess_claim_gate(frame))

    assert out.columns.tolist().count("ess_gate_status") == 1
    assert all(column in out.columns for column in ESS_GATE_COLUMNS)
    assert out.loc[0, "ess_gate_status"] == "pass"


def test_endpoint_eval_summary_prefers_endpoint_geom_mass_with_uot_aliases() -> None:
    frame = pd.DataFrame(
        {
            "perturbation_id": ["ctrl", "gene_a"],
            "endpoint_geom_mass": [0.2, 0.6],
            "uot": [99.0, 99.0],
            "mass_rel_error": [0.1, 0.3],
            "is_control": [True, False],
            "ess_claim_grade_allowed": [True, False],
        }
    )

    summary = summarize_eval(frame)

    assert summary["mean_endpoint_geom_mass"] == pytest.approx(0.4)
    assert summary["median_endpoint_geom_mass"] == pytest.approx(0.4)
    assert summary["mean_uot"] == pytest.approx(summary["mean_endpoint_geom_mass"])
    assert summary["control_mean_endpoint_geom_mass"] == pytest.approx(0.2)
    assert summary["non_control_mean_endpoint_geom_mass"] == pytest.approx(0.6)
    assert summary["n_ess_claim_grade_allowed"] == 1
    assert summary["n_ess_claim_grade_blocked"] == 1


def test_cap_measure_atoms_preserves_selected_finite_measure_weights() -> None:
    measure = FiniteMeasure(
        support=np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32),
        weights=np.asarray([1.0, 3.0, 6.0], dtype=np.float32),
        total_mass=10.0,
    )

    capped = cap_measure_atoms(measure, max_atoms=2, seed=4, mode="preserve_weights")

    assert capped.n_atoms == 2
    assert capped.total_mass == pytest.approx(10.0)
    assert float(capped.weights.sum()) == pytest.approx(10.0)
    assert len(set(np.round(capped.weights, 6))) > 1


def test_endpoint_eval_summary_handles_csv_loaded_boolean_strings() -> None:
    frame = pd.DataFrame(
        {
            "perturbation_id": ["ctrl", "gene_a"],
            "endpoint_geom_mass": [0.2, 0.6],
            "mass_rel_error": [0.1, 0.3],
            "is_control": ["True", "False"],
            "ess_claim_grade_allowed": ["False", "True"],
        }
    )

    summary = summarize_eval(frame)

    assert summary["n_ess_claim_grade_allowed"] == 1
    assert summary["n_ess_claim_grade_blocked"] == 1
    assert summary["n_controls"] == 1
    assert summary["control_mean_endpoint_geom_mass"] == pytest.approx(0.2)
    assert summary["non_control_mean_endpoint_geom_mass"] == pytest.approx(0.6)
