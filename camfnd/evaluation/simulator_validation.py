from __future__ import annotations

"""Acceptance checks for the trusted simulator implementation."""

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

from camfnd.data.contract import EndpointProblem, FiniteMeasure, PerturbSeqDynamicsData
from camfnd.numerics.particles_np import ParticleState, compare_measures_exact
from camfnd.numerics.euler_maruyama import EulerMaruyamaConfig, Stage1EulerMaruyamaSimulator
from camfnd.numerics.truth_coeffs import Stage1SDECoefficients, build_truth_coefficients


@dataclass(slots=True)
class Step2Evaluation:
    """Structured result of the Step-2 acceptance checks."""

    initialization_exact: bool
    stability_ok: bool
    accuracy_ok: bool
    convergence_ok: bool
    default_run_summary: pd.DataFrame
    default_analytic_comparison: pd.DataFrame
    convergence_table: pd.DataFrame
    initialization_table: pd.DataFrame
    thresholds: Dict[str, float]

    @property
    def ok(self) -> bool:
        return bool(self.initialization_exact and self.stability_ok and self.accuracy_ok and self.convergence_ok)

    def to_dict(self) -> dict:
        return {
            "initialization_exact": self.initialization_exact,
            "stability_ok": self.stability_ok,
            "accuracy_ok": self.accuracy_ok,
            "convergence_ok": self.convergence_ok,
            "ok": self.ok,
            "thresholds": dict(self.thresholds),
        }


@dataclass(frozen=True, slots=True)
class ConditionalAnalyticMoments:
    sample_id: str
    perturbation_id: str
    terminal_mean: float
    terminal_variance: float
    terminal_mass: float


def _conditional_analytic_moments(
    measure: FiniteMeasure,
    coeffs: Stage1SDECoefficients,
    *,
    T: float,
) -> ConditionalAnalyticMoments:
    m0 = float(measure.mean()[0])
    v0 = float(measure.variance_trace())
    if coeffs.kappa == 0.0:
        mean_T = m0
        var_T = v0 + coeffs.sigma ** 2 * T
    else:
        exp_term = np.exp(-coeffs.kappa * T)
        mean_T = coeffs.theta + (m0 - coeffs.theta) * exp_term
        var_T = v0 * np.exp(-2.0 * coeffs.kappa * T) + (coeffs.sigma ** 2 / (2.0 * coeffs.kappa)) * (
            1.0 - np.exp(-2.0 * coeffs.kappa * T)
        )
    mass_T = float(measure.total_mass * np.exp(coeffs.rho * T))
    return ConditionalAnalyticMoments(
        sample_id=measure.sample_id,
        perturbation_id=measure.perturbation_id,
        terminal_mean=float(mean_T),
        terminal_variance=float(var_T),
        terminal_mass=mass_T,
    )


def conditional_analytic_summary(problem: EndpointProblem, coeffs: Dict[str, Stage1SDECoefficients]) -> pd.DataFrame:
    t0 = problem.time_axis.t(problem.time_axis.initial_label)
    t1 = problem.time_axis.t(problem.time_axis.terminal_label)
    T = float(t1 - t0)
    rows = []
    for key, measure in problem.initial.items():
        perturbation_id = key[1] if isinstance(key, tuple) else str(key)
        stats = _conditional_analytic_moments(measure, coeffs[perturbation_id], T=T)
        rows.append({
            "key": key,
            "sample_id": stats.sample_id,
            "perturbation_id": stats.perturbation_id,
            "truth_terminal_mean": stats.terminal_mean,
            "truth_terminal_variance": stats.terminal_variance,
            "truth_terminal_mass": stats.terminal_mass,
        })
    return pd.DataFrame(rows).sort_values(["sample_id", "perturbation_id"]).reset_index(drop=True)


def initialization_exactness_table(problem: EndpointProblem) -> pd.DataFrame:
    rows = []
    for key, measure in problem.initial.items():
        particles = ParticleState.from_measure(measure, particles_per_atom=1)
        reconstructed = particles.to_measure(time_label=problem.time_axis.initial_label)
        exact, reason = compare_measures_exact(reconstructed, measure)
        rows.append(
            {
                "key": key,
                "sample_id": measure.sample_id,
                "perturbation_id": measure.perturbation_id,
                "exact": bool(exact),
                "reason": reason,
                "n_atoms": measure.n_atoms,
            }
        )
    return pd.DataFrame(rows).sort_values(["sample_id", "perturbation_id"]).reset_index(drop=True)


def _comparison_for_run(
    problem: EndpointProblem,
    run_summary: pd.DataFrame,
    coeffs: Dict[str, Stage1SDECoefficients],
    *,
    seed: int,
    n_steps: int,
) -> pd.DataFrame:
    truth = conditional_analytic_summary(problem, coeffs)
    merged = run_summary.merge(
        truth,
        on=["key", "sample_id", "perturbation_id"],
        how="left",
    )
    merged["seed"] = int(seed)
    merged["n_steps"] = int(n_steps)
    merged["abs_error_mean"] = (merged["terminal_mean_0"] - merged["truth_terminal_mean"]).abs()
    merged["abs_error_variance"] = (merged["terminal_var_trace"] - merged["truth_terminal_variance"]).abs()
    merged["abs_error_mass"] = (merged["terminal_mass"] - merged["truth_terminal_mass"]).abs()
    return merged.sort_values(["sample_id", "perturbation_id"]).reset_index(drop=True)


def _aggregate_convergence(comparison_rows: pd.DataFrame) -> pd.DataFrame:
    grouped = comparison_rows.groupby("n_steps", as_index=False).agg(
        mean_abs_error_mean=("abs_error_mean", "mean"),
        mean_abs_error_variance=("abs_error_variance", "mean"),
        mean_abs_error_mass=("abs_error_mass", "mean"),
        max_abs_error_mean=("abs_error_mean", "max"),
        max_abs_error_variance=("abs_error_variance", "max"),
        max_abs_error_mass=("abs_error_mass", "max"),
    )
    return grouped.sort_values("n_steps").reset_index(drop=True)


def evaluate_step2(
    dataset: PerturbSeqDynamicsData,
    *,
    default_n_steps: int = 128,
    n_steps_grid: Sequence[int] = (16, 32, 64, 128),
    seeds: Sequence[int] = (11, 19, 29, 37),
    particles_per_atom: int = 32,
    accuracy_threshold_mean: float = 0.015,
    accuracy_threshold_variance: float = 0.0015,
    accuracy_threshold_mass: float = 1e-10,
) -> Step2Evaluation:
    """Run the Step-2 structural, stability, accuracy, and convergence checks."""

    dataset.validate()
    problem = dataset.to_endpoint_problem(by_sample=True)
    problem.validate()
    coeffs = build_truth_coefficients(dataset)

    init_table = initialization_exactness_table(problem)
    initialization_exact = bool(init_table["exact"].all())

    comparison_frames = []
    default_run_summary = None
    default_analytic_comparison = None
    stability_flags = []

    for n_steps in n_steps_grid:
        for seed in seeds:
            config = EulerMaruyamaConfig(
                n_steps=int(n_steps),
                seed=int(seed),
                particles_per_atom=int(particles_per_atom),
                store_history=False,
            )
            simulator = Stage1EulerMaruyamaSimulator(
                endpoint_problem=problem,
                coefficients=coeffs,
                config=config,
            )
            result = simulator.run()
            stability_flags.append(result.stable)
            comparison = _comparison_for_run(problem, result.summary, coeffs, seed=seed, n_steps=n_steps)
            comparison_frames.append(comparison)
            if n_steps == default_n_steps and seed == seeds[0]:
                default_run_summary = result.summary.copy()
                default_analytic_comparison = comparison.copy()

    if default_run_summary is None or default_analytic_comparison is None:
        raise RuntimeError("Could not construct the default Step-2 evaluation outputs.")

    all_comparisons = pd.concat(comparison_frames, axis=0, ignore_index=True)
    convergence = _aggregate_convergence(all_comparisons)

    default_mean_error = float(
        default_analytic_comparison.loc[:, "abs_error_mean"].mean()
    )
    default_variance_error = float(
        default_analytic_comparison.loc[:, "abs_error_variance"].mean()
    )
    default_mass_error = float(
        default_analytic_comparison.loc[:, "abs_error_mass"].mean()
    )

    accuracy_ok = bool(
        default_mean_error <= accuracy_threshold_mean
        and default_variance_error <= accuracy_threshold_variance
        and default_mass_error <= accuracy_threshold_mass
    )

    stability_ok = bool(all(stability_flags))

    convergence_ok = bool(
        np.all(np.diff(convergence["mean_abs_error_mean"].to_numpy()) < 0)
        and np.all(np.diff(convergence["mean_abs_error_variance"].to_numpy()) < 0)
    )

    thresholds = {
        "accuracy_threshold_mean": float(accuracy_threshold_mean),
        "accuracy_threshold_variance": float(accuracy_threshold_variance),
        "accuracy_threshold_mass": float(accuracy_threshold_mass),
        "default_mean_error": default_mean_error,
        "default_variance_error": default_variance_error,
        "default_mass_error": default_mass_error,
        "particles_per_atom": float(particles_per_atom),
    }

    return Step2Evaluation(
        initialization_exact=initialization_exact,
        stability_ok=stability_ok,
        accuracy_ok=accuracy_ok,
        convergence_ok=convergence_ok,
        default_run_summary=default_run_summary,
        default_analytic_comparison=default_analytic_comparison,
        convergence_table=convergence,
        initialization_table=init_table,
        thresholds=thresholds,
    )


# Semantic aliases for the clearer software-facing API.
SimulatorValidationEvaluation = Step2Evaluation
evaluate_simulator_validation = evaluate_step2
