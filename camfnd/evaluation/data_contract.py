from __future__ import annotations

"""Acceptance checks for the endpoint data contract."""

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from camfnd.data.contract import EndpointProblem, FiniteMeasure, PerturbSeqDynamicsData


@dataclass(slots=True)
class Step1Evaluation:
    """Structured result of the Step-1 acceptance checks."""

    dataset_valid: bool
    endpoint_valid: bool
    fixed_cell_counts_ok: bool
    separate_mass_scale_ok: bool
    qualitative_signatures_ok: bool
    empirical_terminal_summary: pd.DataFrame
    count_summary: pd.DataFrame
    analytic_comparison: pd.DataFrame
    signature_checks: Dict[str, bool]

    @property
    def ok(self) -> bool:
        return all(
            [
                self.dataset_valid,
                self.endpoint_valid,
                self.fixed_cell_counts_ok,
                self.separate_mass_scale_ok,
                self.qualitative_signatures_ok,
            ]
        )

    def to_dict(self) -> dict:
        return {
            "dataset_valid": self.dataset_valid,
            "endpoint_valid": self.endpoint_valid,
            "fixed_cell_counts_ok": self.fixed_cell_counts_ok,
            "separate_mass_scale_ok": self.separate_mass_scale_ok,
            "qualitative_signatures_ok": self.qualitative_signatures_ok,
            "ok": self.ok,
            "signature_checks": dict(self.signature_checks),
        }


def _measure_stats(measure: FiniteMeasure) -> tuple[float, float]:
    weights = measure.normalized_weights
    x = measure.support
    mean = np.sum(weights[:, None] * x, axis=0)
    centered = x - mean
    covariance = centered.T @ (centered * weights[:, None])
    return float(mean[0]), float(np.trace(covariance))


def terminal_summary_table(problem: EndpointProblem) -> pd.DataFrame:
    """Per-key terminal summary table used in Step-1 evaluation."""

    rows = []
    for key, measure in problem.terminal.items():
        terminal_mean, terminal_var = _measure_stats(measure)
        rows.append(
            {
                "sample_id": measure.sample_id,
                "perturbation_id": measure.perturbation_id,
                "terminal_mean": terminal_mean,
                "terminal_variance": terminal_var,
                "terminal_mass": measure.total_mass,
                "n_terminal_atoms": measure.n_atoms,
                "key": key,
            }
        )
    return pd.DataFrame(rows).sort_values(["sample_id", "perturbation_id"]).reset_index(drop=True)


def _check_fixed_cell_counts(count_summary: pd.DataFrame) -> tuple[bool, bool]:
    counts_by_time = count_summary.groupby("time_label")["n_cells"]
    fixed_counts_ok = bool(all(group.nunique() == 1 for _, group in counts_by_time))

    term = count_summary[count_summary["time_label"] == "P60"].copy()
    if term.empty:
        return fixed_counts_ok, False
    masses_vary = term["mass"].nunique() > 1
    counts_constant = term["n_cells"].nunique() == 1
    separate_mass_scale_ok = bool(masses_vary and counts_constant)
    return fixed_counts_ok, separate_mass_scale_ok


def _signature_checks(empirical_terminal_summary: pd.DataFrame) -> Dict[str, bool]:
    required = {"ctrl", "drift", "diff", "react"}
    rows = empirical_terminal_summary
    checks: Dict[str, bool] = {}

    for sample_id, sub in rows.groupby("sample_id"):
        observed = set(sub["perturbation_id"])
        if required - observed:
            missing = sorted(required - observed)
            raise ValueError(f"Sample {sample_id!r} is missing required perturbations for Step-1 checks: {missing}")

        ctrl = sub.loc[sub["perturbation_id"] == "ctrl"].iloc[0]
        drift = sub.loc[sub["perturbation_id"] == "drift"].iloc[0]
        diff = sub.loc[sub["perturbation_id"] == "diff"].iloc[0]
        react = sub.loc[sub["perturbation_id"] == "react"].iloc[0]

        checks[f"{sample_id}:drift_mean_gt_ctrl"] = bool(drift["terminal_mean"] > ctrl["terminal_mean"])
        checks[f"{sample_id}:diff_var_gt_ctrl"] = bool(diff["terminal_variance"] > ctrl["terminal_variance"])
        checks[f"{sample_id}:react_mass_lt_ctrl"] = bool(react["terminal_mass"] < ctrl["terminal_mass"])

    return checks


def _analytic_comparison_table(
    empirical_terminal_summary: pd.DataFrame,
    dataset: PerturbSeqDynamicsData,
) -> pd.DataFrame:
    if dataset.truth is None or dataset.truth.analytic_summary is None:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "perturbation_id",
                "empirical_terminal_mean",
                "truth_terminal_mean",
                "abs_error_mean",
                "empirical_terminal_variance",
                "truth_terminal_variance",
                "abs_error_variance",
                "empirical_terminal_mass",
                "truth_terminal_mass",
                "abs_error_mass",
            ]
        )

    analytic = dataset.truth.analytic_summary.rename(
        columns={
            "terminal_mean": "truth_terminal_mean",
            "terminal_variance": "truth_terminal_variance",
            "terminal_mass": "truth_terminal_mass",
        }
    )
    merged = empirical_terminal_summary.merge(
        analytic[[
            "sample_id",
            "perturbation_id",
            "truth_terminal_mean",
            "truth_terminal_variance",
            "truth_terminal_mass",
        ]],
        on=["sample_id", "perturbation_id"],
        how="left",
    )
    merged = merged.rename(
        columns={
            "terminal_mean": "empirical_terminal_mean",
            "terminal_variance": "empirical_terminal_variance",
            "terminal_mass": "empirical_terminal_mass",
        }
    )
    merged["abs_error_mean"] = (merged["empirical_terminal_mean"] - merged["truth_terminal_mean"]).abs()
    merged["abs_error_variance"] = (
        merged["empirical_terminal_variance"] - merged["truth_terminal_variance"]
    ).abs()
    merged["abs_error_mass"] = (merged["empirical_terminal_mass"] - merged["truth_terminal_mass"]).abs()
    return merged.sort_values(["sample_id", "perturbation_id"]).reset_index(drop=True)


def evaluate_step1(dataset: PerturbSeqDynamicsData) -> Step1Evaluation:
    """Run the Step-1 structural and benchmark acceptance checks."""

    dataset.validate()
    problem = dataset.to_endpoint_problem(by_sample=True)
    problem.validate()

    count_summary = dataset.summary()
    fixed_cell_counts_ok, separate_mass_scale_ok = _check_fixed_cell_counts(count_summary)

    empirical_terminal_summary = terminal_summary_table(problem)
    signature_checks = _signature_checks(empirical_terminal_summary)
    qualitative_signatures_ok = bool(all(signature_checks.values()))

    analytic_comparison = _analytic_comparison_table(empirical_terminal_summary, dataset)

    return Step1Evaluation(
        dataset_valid=True,
        endpoint_valid=True,
        fixed_cell_counts_ok=fixed_cell_counts_ok,
        separate_mass_scale_ok=separate_mass_scale_ok,
        qualitative_signatures_ok=qualitative_signatures_ok,
        empirical_terminal_summary=empirical_terminal_summary,
        count_summary=count_summary,
        analytic_comparison=analytic_comparison,
        signature_checks=signature_checks,
    )


# Semantic aliases for the clearer software-facing API.
DataContractEvaluation = Step1Evaluation
evaluate_data_contract = evaluate_step1
