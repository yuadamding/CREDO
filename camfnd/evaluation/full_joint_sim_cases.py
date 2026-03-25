from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import math

import numpy as np
import pandas as pd
import torch

from camfnd.data.contract import EndpointProblem, FiniteMeasure, PerturbationCatalog, TimeAxis
from camfnd.simulation.full_joint_sim import FullJointSimulator
from camfnd.simulation.single_screen_sim import LearnedSimulatorConfig


@dataclass(slots=True)
class FullJointSimCaseResult:
    case_name: str
    description: str
    passed: bool
    metrics: Dict[str, float | bool | int]


@dataclass(slots=True)
class FullJointSimCaseSuite:
    results: List[FullJointSimCaseResult]
    summary_table: pd.DataFrame

    @property
    def ok(self) -> bool:
        return bool(all(result.passed for result in self.results))

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "n_cases": len(self.results),
            "passed_cases": int(sum(result.passed for result in self.results)),
            "results": [
                {
                    "case_name": result.case_name,
                    "description": result.description,
                    "passed": result.passed,
                    "metrics": dict(result.metrics),
                }
                for result in self.results
            ],
        }


def _catalog(perturbation_ids: Iterable[str]) -> PerturbationCatalog:
    table = pd.DataFrame(
        [
            {"perturbation_id": str(pid), "is_control": bool(str(pid) == "ctrl")}
            for pid in perturbation_ids
        ]
    )
    catalog = PerturbationCatalog(table)
    catalog.validate()
    return catalog


def _measure(
    support: np.ndarray,
    *,
    mass: float,
    perturbation_id: str,
    time_label: str,
    sample_id: str,
) -> FiniteMeasure:
    support = np.asarray(support, dtype=float)
    if support.ndim != 2:
        raise ValueError("support must have shape [n, d].")
    n_atoms = support.shape[0]
    weights = np.full(n_atoms, float(mass) / n_atoms, dtype=float)
    measure = FiniteMeasure(
        support=support,
        weights=weights,
        total_mass=float(weights.sum()),
        perturbation_id=str(perturbation_id),
        time_label=str(time_label),
        sample_id=str(sample_id),
    )
    measure.validate()
    return measure


def _endpoint_problem(initial: Dict[object, FiniteMeasure], *, latent_dim: int) -> EndpointProblem:
    time_axis = TimeAxis()
    perturbation_ids = sorted({measure.perturbation_id for measure in initial.values()})
    terminal = {
        key: FiniteMeasure(
            support=measure.support.copy(),
            weights=measure.weights.copy(),
            total_mass=float(measure.total_mass),
            perturbation_id=measure.perturbation_id,
            time_label=time_axis.terminal_label,
            sample_id=measure.sample_id,
        )
        for key, measure in initial.items()
    }
    problem = EndpointProblem(
        initial=initial,
        terminal=terminal,
        catalog=_catalog(perturbation_ids),
        time_axis=time_axis,
        metadata={"latent_dim": int(latent_dim)},
    )
    problem.validate()
    return problem


class _BaseTestModel:
    def eval(self):
        return self


class _FixedFieldModel(_BaseTestModel):
    def __init__(
        self,
        *,
        latent_dim: int,
        context_dim: int,
        drift_by_perturbation: Dict[str, np.ndarray],
        diffusion_by_perturbation: Dict[str, np.ndarray] | None = None,
        growth_by_perturbation: Dict[str, float],
        context_by_sample: Dict[str, np.ndarray] | None = None,
    ) -> None:
        self.latent_dim = int(latent_dim)
        self.context_dim = int(context_dim)
        self.drift_by_perturbation = {
            str(pid): np.asarray(value, dtype=float).reshape(self.latent_dim)
            for pid, value in drift_by_perturbation.items()
        }
        self.diffusion_by_perturbation = {
            str(pid): np.asarray(value, dtype=float).reshape(self.latent_dim)
            for pid, value in (diffusion_by_perturbation or {}).items()
        }
        self.growth_by_perturbation = {str(pid): float(value) for pid, value in growth_by_perturbation.items()}
        self.context_by_sample = {
            str(sample_id): np.asarray(value, dtype=float).reshape(self.context_dim)
            for sample_id, value in (context_by_sample or {}).items()
        }

    def context_values(self, particles) -> Dict[str, torch.Tensor]:
        out = {}
        for sample_id in sorted({state.sample_id for state in particles.values()}):
            state = next(st for st in particles.values() if st.sample_id == sample_id)
            value = self.context_by_sample.get(sample_id, np.zeros(self.context_dim, dtype=float))
            out[sample_id] = torch.as_tensor(value, dtype=state.z.dtype, device=state.z.device)
        return out

    def coefficients(self, z: torch.Tensor, t: torch.Tensor, perturbation_id: str, context=None) -> Dict[str, torch.Tensor]:
        drift = torch.as_tensor(
            self.drift_by_perturbation[str(perturbation_id)],
            dtype=z.dtype,
            device=z.device,
        ).reshape(1, -1).repeat(z.shape[0], 1)
        diffusion = torch.as_tensor(
            self.diffusion_by_perturbation.get(str(perturbation_id), np.zeros(self.latent_dim, dtype=float)),
            dtype=z.dtype,
            device=z.device,
        ).reshape(1, -1).repeat(z.shape[0], 1)
        growth_value = torch.tensor(
            self.growth_by_perturbation[str(perturbation_id)],
            dtype=z.dtype,
            device=z.device,
        )
        growth = growth_value.reshape(1, 1).repeat(z.shape[0], 1)
        return {"drift": drift, "diffusion": diffusion, "growth": growth}


class _TotalMassContextModel(_BaseTestModel):
    def __init__(self, *, eta: float, base_drift_by_perturbation: Dict[str, float] | None = None) -> None:
        self.eta = float(eta)
        self.base_drift_by_perturbation = {
            str(pid): float(value) for pid, value in (base_drift_by_perturbation or {}).items()
        }

    def context_values(self, particles) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for sample_id in sorted({state.sample_id for state in particles.values()}):
            sample_states = [state for state in particles.values() if state.sample_id == sample_id]
            total_mass = sum(state.total_mass() for state in sample_states)
            out[sample_id] = total_mass.reshape(1)
        return out

    def coefficients(self, z: torch.Tensor, t: torch.Tensor, perturbation_id: str, context=None) -> Dict[str, torch.Tensor]:
        context_scalar = float(context.reshape(-1)[0].detach().cpu()) if context is not None else 0.0
        drift_value = self.base_drift_by_perturbation.get(str(perturbation_id), 0.0) + self.eta * context_scalar
        drift = torch.full((z.shape[0], 1), drift_value, dtype=z.dtype, device=z.device)
        diffusion = torch.zeros_like(drift)
        growth = torch.zeros((z.shape[0], 1), dtype=z.dtype, device=z.device)
        return {"drift": drift, "diffusion": diffusion, "growth": growth}


class _LinearDriftModel(_BaseTestModel):
    def __init__(self, *, alpha: float) -> None:
        self.alpha = float(alpha)

    def context_values(self, particles) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for sample_id in sorted({state.sample_id for state in particles.values()}):
            state = next(st for st in particles.values() if st.sample_id == sample_id)
            out[sample_id] = torch.zeros(1, dtype=state.z.dtype, device=state.z.device)
        return out

    def coefficients(self, z: torch.Tensor, t: torch.Tensor, perturbation_id: str, context=None) -> Dict[str, torch.Tensor]:
        drift = self.alpha * z
        diffusion = torch.zeros_like(z)
        growth = torch.zeros((z.shape[0], 1), dtype=z.dtype, device=z.device)
        return {"drift": drift, "diffusion": diffusion, "growth": growth}


class _TimeRampDriftModel(_BaseTestModel):
    def __init__(self, *, slope: float, intercept: float = 0.0) -> None:
        self.slope = float(slope)
        self.intercept = float(intercept)

    def context_values(self, particles) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for sample_id in sorted({state.sample_id for state in particles.values()}):
            state = next(st for st in particles.values() if st.sample_id == sample_id)
            out[sample_id] = torch.zeros(1, dtype=state.z.dtype, device=state.z.device)
        return out

    def coefficients(self, z: torch.Tensor, t: torch.Tensor, perturbation_id: str, context=None) -> Dict[str, torch.Tensor]:
        t_value = float(t.detach().cpu())
        drift_value = self.intercept + self.slope * t_value
        drift = torch.full((z.shape[0], 1), drift_value, dtype=z.dtype, device=z.device)
        diffusion = torch.zeros_like(drift)
        growth = torch.zeros((z.shape[0], 1), dtype=z.dtype, device=z.device)
        return {"drift": drift, "diffusion": diffusion, "growth": growth}


class _LinearContextVectorModel(_BaseTestModel):
    def __init__(self, *, latent_dim: int, context_weight: np.ndarray) -> None:
        weight = np.asarray(context_weight, dtype=float)
        if weight.ndim != 2:
            raise ValueError("context_weight must have shape [latent_dim, context_dim].")
        self.latent_dim = int(latent_dim)
        if weight.shape[0] != self.latent_dim:
            raise ValueError("context_weight first dimension must equal latent_dim.")
        self.context_weight = weight
        self.context_dim = int(weight.shape[1])

    def context_values(self, particles) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for sample_id in sorted({state.sample_id for state in particles.values()}):
            sample_states = [state for state in particles.values() if state.sample_id == sample_id]
            total_mass = sum(state.total_mass() for state in sample_states)
            features = torch.stack([total_mass, total_mass.pow(2)]).reshape(self.context_dim)
            out[sample_id] = features
        return out

    def coefficients(self, z: torch.Tensor, t: torch.Tensor, perturbation_id: str, context=None) -> Dict[str, torch.Tensor]:
        if context is None:
            context = torch.zeros(self.context_dim, dtype=z.dtype, device=z.device)
        drift_vector = torch.as_tensor(self.context_weight, dtype=z.dtype, device=z.device) @ context.reshape(-1, 1)
        drift = drift_vector.reshape(1, self.latent_dim).repeat(z.shape[0], 1)
        diffusion = torch.zeros_like(drift)
        growth = torch.zeros((z.shape[0], 1), dtype=z.dtype, device=z.device)
        return {"drift": drift, "diffusion": diffusion, "growth": growth}


class _TotalMassFeedbackGrowthModel(_BaseTestModel):
    def __init__(
        self,
        *,
        eta: float,
        growth_by_perturbation: Dict[str, float],
        base_drift_by_perturbation: Dict[str, float] | None = None,
    ) -> None:
        self.eta = float(eta)
        self.growth_by_perturbation = {str(pid): float(value) for pid, value in growth_by_perturbation.items()}
        self.base_drift_by_perturbation = {
            str(pid): float(value) for pid, value in (base_drift_by_perturbation or {}).items()
        }

    def context_values(self, particles) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for sample_id in sorted({state.sample_id for state in particles.values()}):
            sample_states = [state for state in particles.values() if state.sample_id == sample_id]
            total_mass = sum(state.total_mass() for state in sample_states)
            out[sample_id] = total_mass.reshape(1)
        return out

    def coefficients(self, z: torch.Tensor, t: torch.Tensor, perturbation_id: str, context=None) -> Dict[str, torch.Tensor]:
        context_scalar = float(context.reshape(-1)[0].detach().cpu()) if context is not None else 0.0
        drift_value = self.base_drift_by_perturbation.get(str(perturbation_id), 0.0) + self.eta * context_scalar
        drift = torch.full((z.shape[0], 1), drift_value, dtype=z.dtype, device=z.device)
        diffusion = torch.zeros_like(drift)
        growth_value = self.growth_by_perturbation.get(str(perturbation_id), 0.0)
        growth = torch.full((z.shape[0], 1), growth_value, dtype=z.dtype, device=z.device)
        return {"drift": drift, "diffusion": diffusion, "growth": growth}


class _StateDependentGrowthModel(_BaseTestModel):
    def __init__(self, *, drift_velocity: float, growth_intercept: float, growth_slope: float) -> None:
        self.drift_velocity = float(drift_velocity)
        self.growth_intercept = float(growth_intercept)
        self.growth_slope = float(growth_slope)

    def context_values(self, particles) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for sample_id in sorted({state.sample_id for state in particles.values()}):
            state = next(st for st in particles.values() if st.sample_id == sample_id)
            out[sample_id] = torch.zeros(1, dtype=state.z.dtype, device=state.z.device)
        return out

    def coefficients(self, z: torch.Tensor, t: torch.Tensor, perturbation_id: str, context=None) -> Dict[str, torch.Tensor]:
        drift = torch.full_like(z, self.drift_velocity)
        diffusion = torch.zeros_like(z)
        growth = self.growth_intercept + self.growth_slope * z[:, :1]
        return {"drift": drift, "diffusion": diffusion, "growth": growth}


class _SampleMeanContextFeedbackModel(_BaseTestModel):
    def __init__(self, *, eta: float, driver_velocity: float, driver_perturbation: str = "driver") -> None:
        self.eta = float(eta)
        self.driver_velocity = float(driver_velocity)
        self.driver_perturbation = str(driver_perturbation)

    def context_values(self, particles) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for sample_id in sorted({state.sample_id for state in particles.values()}):
            sample_states = [state for state in particles.values() if state.sample_id == sample_id]
            total_weight = None
            weighted_sum = None
            for state in sample_states:
                atom_weights = state.atom_weights()
                state_weight = atom_weights.sum()
                state_sum = (atom_weights[:, None] * state.z).sum(dim=0)
                total_weight = state_weight if total_weight is None else total_weight + state_weight
                weighted_sum = state_sum if weighted_sum is None else weighted_sum + state_sum
            if total_weight is None or weighted_sum is None:
                raise ValueError("sample must contain at least one particle state.")
            out[sample_id] = (weighted_sum / total_weight).reshape(-1)
        return out

    def coefficients(self, z: torch.Tensor, t: torch.Tensor, perturbation_id: str, context=None) -> Dict[str, torch.Tensor]:
        context_scalar = float(context.reshape(-1)[0].detach().cpu()) if context is not None else 0.0
        if str(perturbation_id) == self.driver_perturbation:
            drift_value = self.driver_velocity
        else:
            drift_value = self.eta * context_scalar
        drift = torch.full((z.shape[0], 1), drift_value, dtype=z.dtype, device=z.device)
        diffusion = torch.zeros_like(drift)
        growth = torch.zeros((z.shape[0], 1), dtype=z.dtype, device=z.device)
        return {"drift": drift, "diffusion": diffusion, "growth": growth}


def _summary_mean_vector(row: pd.Series) -> np.ndarray:
    columns = sorted(
        [column for column in row.index if str(column).startswith("terminal_mean_")],
        key=lambda value: int(str(value).split("_")[-1]),
    )
    return np.array([float(row[column]) for column in columns], dtype=float)


def _run(problem: EndpointProblem, *, config: LearnedSimulatorConfig, model) -> object:
    simulator = FullJointSimulator(problem, config)
    return simulator.run(model)


def _case_multidimensional_constant_fields_exact() -> FullJointSimCaseResult:
    latent_dim = 3
    sample_id = "screenA"
    initial = {
        (sample_id, "ctrl"): _measure(
            np.array([[0.0, 1.0, -1.0], [2.0, -1.0, 0.5]], dtype=float),
            mass=1.2,
            perturbation_id="ctrl",
            time_label="P4",
            sample_id=sample_id,
        ),
        (sample_id, "drift"): _measure(
            np.array([[-1.0, 0.0, 0.5], [1.0, 1.0, 1.5], [2.0, -2.0, -0.5]], dtype=float),
            mass=0.8,
            perturbation_id="drift",
            time_label="P4",
            sample_id=sample_id,
        ),
        (sample_id, "react"): _measure(
            np.array([[0.5, -0.5, 2.0], [1.5, 0.5, 0.0]], dtype=float),
            mass=1.5,
            perturbation_id="react",
            time_label="P4",
            sample_id=sample_id,
        ),
    }
    problem = _endpoint_problem(initial, latent_dim=latent_dim)
    model = _FixedFieldModel(
        latent_dim=latent_dim,
        context_dim=3,
        drift_by_perturbation={
            "ctrl": np.array([0.1, -0.2, 0.05], dtype=float),
            "drift": np.array([0.3, 0.0, -0.1], dtype=float),
            "react": np.array([-0.2, 0.2, 0.1], dtype=float),
        },
        diffusion_by_perturbation=None,
        growth_by_perturbation={"ctrl": 0.0, "drift": 0.2, "react": -0.4},
        context_by_sample={sample_id: np.array([0.0, 0.0, 0.0], dtype=float)},
    )
    result = _run(
        problem,
        config=LearnedSimulatorConfig(n_steps=10, seed=5, device="cpu", dtype="float64"),
        model=model,
    )
    T = problem.time_axis.t(problem.time_axis.terminal_label) - problem.time_axis.t(problem.time_axis.initial_label)

    max_mean_error = 0.0
    max_mass_error = 0.0
    max_var_error = 0.0
    for _, row in result.summary.iterrows():
        key = (str(row["sample_id"]), str(row["perturbation_id"]))
        init_measure = problem.initial[key]
        expected_mean = init_measure.mean() + model.drift_by_perturbation[key[1]] * T
        expected_mass = init_measure.total_mass * math.exp(model.growth_by_perturbation[key[1]] * T)
        expected_var = init_measure.variance_trace()
        max_mean_error = max(max_mean_error, float(np.max(np.abs(_summary_mean_vector(row) - expected_mean))))
        max_mass_error = max(max_mass_error, abs(float(row["terminal_mass"]) - expected_mass))
        max_var_error = max(max_var_error, abs(float(row["terminal_var_trace"]) - expected_var))

    context_columns = [column for column in result.context_summary.columns if column.startswith("context_")]
    passed = bool(
        result.stable
        and max_mean_error <= 1e-12
        and max_mass_error <= 1e-12
        and max_var_error <= 1e-12
        and context_columns == ["context_0", "context_1", "context_2"]
    )
    return FullJointSimCaseResult(
        case_name="multidimensional_constant_fields_exact",
        description="Exact deterministic 3D translation-plus-growth case with vector context logging.",
        passed=passed,
        metrics={
            "stable": bool(result.stable),
            "max_mean_error": max_mean_error,
            "max_mass_error": max_mass_error,
            "max_var_error": max_var_error,
            "n_context_columns": len(context_columns),
        },
    )


def _case_context_coupled_screen_shift_exact() -> FullJointSimCaseResult:
    ctrl_mass = 1.0
    driver_masses = {"screen_low": 0.5, "screen_high": 2.0}
    initial: Dict[object, FiniteMeasure] = {}
    for sample_id, driver_mass in driver_masses.items():
        initial[(sample_id, "ctrl")] = _measure(
            np.array([[0.0], [0.2]], dtype=float),
            mass=ctrl_mass,
            perturbation_id="ctrl",
            time_label="P4",
            sample_id=sample_id,
        )
        initial[(sample_id, "driver")] = _measure(
            np.array([[0.1], [0.3]], dtype=float),
            mass=driver_mass,
            perturbation_id="driver",
            time_label="P4",
            sample_id=sample_id,
        )
    problem = _endpoint_problem(initial, latent_dim=1)
    eta = 0.15
    model = _TotalMassContextModel(eta=eta)
    result = _run(
        problem,
        config=LearnedSimulatorConfig(n_steps=6, seed=13, device="cpu", dtype="float64"),
        model=model,
    )
    T = problem.time_axis.t(problem.time_axis.terminal_label) - problem.time_axis.t(problem.time_axis.initial_label)
    total_masses = {sample_id: ctrl_mass + driver_mass for sample_id, driver_mass in driver_masses.items()}
    expected_delta = eta * (total_masses["screen_high"] - total_masses["screen_low"]) * T

    summary = result.summary.set_index(["sample_id", "perturbation_id"])
    pred_delta = float(
        summary.loc[("screen_high", "ctrl"), "terminal_mean_0"]
        - summary.loc[("screen_low", "ctrl"), "terminal_mean_0"]
    )
    final_context = (
        result.context_summary[result.context_summary["step"] == result.config.n_steps]
        .set_index("sample_id")["context_0"]
        .to_dict()
    )
    context_error = max(
        abs(float(final_context["screen_low"]) - total_masses["screen_low"]),
        abs(float(final_context["screen_high"]) - total_masses["screen_high"]),
    )
    passed = bool(result.stable and abs(pred_delta - expected_delta) <= 1e-12 and context_error <= 1e-12)
    return FullJointSimCaseResult(
        case_name="context_coupled_screen_shift_exact",
        description="Two-screen exact context-coupled drift where screen shift is determined by total sample mass.",
        passed=passed,
        metrics={
            "stable": bool(result.stable),
            "pred_control_delta": pred_delta,
            "expected_control_delta": expected_delta,
            "control_delta_error": abs(pred_delta - expected_delta),
            "max_context_error": context_error,
        },
    )


def _case_driver_mass_monotonicity() -> FullJointSimCaseResult:
    ctrl_mass = 1.0
    driver_masses = {"screen_low": 0.5, "screen_mid": 1.5, "screen_high": 2.5}
    initial: Dict[object, FiniteMeasure] = {}
    for sample_id, driver_mass in driver_masses.items():
        initial[(sample_id, "ctrl")] = _measure(
            np.array([[0.0], [0.1]], dtype=float),
            mass=ctrl_mass,
            perturbation_id="ctrl",
            time_label="P4",
            sample_id=sample_id,
        )
        initial[(sample_id, "driver")] = _measure(
            np.array([[0.2], [0.4]], dtype=float),
            mass=driver_mass,
            perturbation_id="driver",
            time_label="P4",
            sample_id=sample_id,
        )
    problem = _endpoint_problem(initial, latent_dim=1)
    eta = 0.1
    model = _TotalMassContextModel(eta=eta)
    result = _run(
        problem,
        config=LearnedSimulatorConfig(n_steps=8, seed=17, device="cpu", dtype="float64"),
        model=model,
    )
    summary = result.summary[result.summary["perturbation_id"] == "ctrl"].set_index("sample_id")
    ctrl_means = {sample_id: float(summary.loc[sample_id, "terminal_mean_0"]) for sample_id in driver_masses}
    ordered_samples = sorted(driver_masses.keys(), key=lambda sample_id: driver_masses[sample_id])
    ordered_means = [ctrl_means[sample_id] for sample_id in ordered_samples]
    monotone = bool(all(left < right for left, right in zip(ordered_means[:-1], ordered_means[1:])))
    ctrl_initial_mean = float(problem.initial[("screen_low", "ctrl")].mean()[0])
    expected_means = {
        sample_id: ctrl_initial_mean + eta * (ctrl_mass + driver_mass)
        for sample_id, driver_mass in driver_masses.items()
    }
    max_exact_error = max(abs(ctrl_means[sample_id] - expected_means[sample_id]) for sample_id in driver_masses)
    passed = bool(result.stable and monotone and max_exact_error <= 1e-12)
    return FullJointSimCaseResult(
        case_name="driver_mass_monotonicity",
        description="Three-screen context response should increase monotonically with driver initial mass.",
        passed=passed,
        metrics={
            "stable": bool(result.stable),
            "ctrl_mean_low": ordered_means[0],
            "ctrl_mean_mid": ordered_means[1],
            "ctrl_mean_high": ordered_means[2],
            "monotone": monotone,
            "max_exact_error": max_exact_error,
        },
    )


def _case_euler_convergence_linear_drift() -> FullJointSimCaseResult:
    sample_id = "screenA"
    initial = {
        (sample_id, "ctrl"): _measure(
            np.array([[-1.0], [0.5], [2.0]], dtype=float),
            mass=1.0,
            perturbation_id="ctrl",
            time_label="P4",
            sample_id=sample_id,
        )
    }
    problem = _endpoint_problem(initial, latent_dim=1)
    alpha = -0.8
    mean0 = float(problem.initial[(sample_id, "ctrl")].mean()[0])
    exact_continuous = mean0 * math.exp(alpha)

    step_errors = []
    for n_steps in (4, 8, 16, 32, 64):
        result = _run(
            problem,
            config=LearnedSimulatorConfig(n_steps=n_steps, seed=23, device="cpu", dtype="float64"),
            model=_LinearDriftModel(alpha=alpha),
        )
        mean_pred = float(result.summary.iloc[0]["terminal_mean_0"])
        step_errors.append({"n_steps": n_steps, "error": abs(mean_pred - exact_continuous)})

    monotone = bool(
        all(
            step_errors[idx]["error"] >= step_errors[idx + 1]["error"] - 1e-12
            for idx in range(len(step_errors) - 1)
        )
    )
    reduction_ratio = step_errors[-1]["error"] / max(step_errors[0]["error"], 1e-12)
    passed = bool(monotone and reduction_ratio <= 0.2)
    return FullJointSimCaseResult(
        case_name="euler_convergence_linear_drift",
        description="Linear deterministic drift should converge toward the continuous-time solution as n_steps increases.",
        passed=passed,
        metrics={
            "error_n4": step_errors[0]["error"],
            "error_n8": step_errors[1]["error"],
            "error_n16": step_errors[2]["error"],
            "error_n32": step_errors[3]["error"],
            "error_n64": step_errors[4]["error"],
            "monotone": monotone,
            "reduction_ratio": reduction_ratio,
        },
    )


def _case_particles_per_atom_invariance() -> FullJointSimCaseResult:
    latent_dim = 2
    sample_id = "screenA"
    initial = {
        (sample_id, "ctrl"): _measure(
            np.array([[0.0, 0.0], [1.0, -1.0]], dtype=float),
            mass=1.0,
            perturbation_id="ctrl",
            time_label="P4",
            sample_id=sample_id,
        ),
        (sample_id, "drift"): _measure(
            np.array([[-0.5, 1.5], [0.5, 0.5], [1.5, -0.5]], dtype=float),
            mass=0.9,
            perturbation_id="drift",
            time_label="P4",
            sample_id=sample_id,
        ),
    }
    problem = _endpoint_problem(initial, latent_dim=latent_dim)
    model = _FixedFieldModel(
        latent_dim=latent_dim,
        context_dim=2,
        drift_by_perturbation={"ctrl": np.array([0.2, -0.1]), "drift": np.array([-0.1, 0.2])},
        diffusion_by_perturbation=None,
        growth_by_perturbation={"ctrl": 0.15, "drift": -0.25},
        context_by_sample={sample_id: np.array([0.0, 0.0])},
    )
    result_1 = _run(
        problem,
        config=LearnedSimulatorConfig(n_steps=9, seed=31, particles_per_atom=1, device="cpu", dtype="float64"),
        model=model,
    )
    result_4 = _run(
        problem,
        config=LearnedSimulatorConfig(n_steps=9, seed=31, particles_per_atom=4, device="cpu", dtype="float64"),
        model=model,
    )
    merged = result_1.summary.merge(
        result_4.summary,
        on=["sample_id", "perturbation_id"],
        suffixes=("_ppa1", "_ppa4"),
    )
    max_mass_diff = float(np.max(np.abs(merged["terminal_mass_ppa1"] - merged["terminal_mass_ppa4"])))
    max_var_diff = float(np.max(np.abs(merged["terminal_var_trace_ppa1"] - merged["terminal_var_trace_ppa4"])))
    mean_columns = [column for column in merged.columns if column.startswith("terminal_mean_") and column.endswith("_ppa1")]
    max_mean_diff = 0.0
    for column in mean_columns:
        paired = column.replace("_ppa1", "_ppa4")
        max_mean_diff = max(max_mean_diff, float(np.max(np.abs(merged[column] - merged[paired]))))
    passed = bool(result_1.stable and result_4.stable and max_mass_diff <= 1e-12 and max_var_diff <= 1e-12 and max_mean_diff <= 1e-12)
    return FullJointSimCaseResult(
        case_name="particles_per_atom_invariance",
        description="Deterministic dynamics should be invariant to particle replication when particles_per_atom changes.",
        passed=passed,
        metrics={
            "stable_ppa1": bool(result_1.stable),
            "stable_ppa4": bool(result_4.stable),
            "max_mass_diff": max_mass_diff,
            "max_mean_diff": max_mean_diff,
            "max_var_diff": max_var_diff,
        },
    )


def _case_time_dependent_drift_schedule_exact() -> FullJointSimCaseResult:
    sample_id = "screenA"
    initial = {
        (sample_id, "ctrl"): _measure(
            np.array([[-1.0], [0.0], [1.0]], dtype=float),
            mass=1.0,
            perturbation_id="ctrl",
            time_label="P4",
            sample_id=sample_id,
        )
    }
    problem = _endpoint_problem(initial, latent_dim=1)
    n_steps = 10
    slope = 0.7
    intercept = -0.1
    result = _run(
        problem,
        config=LearnedSimulatorConfig(n_steps=n_steps, seed=37, device="cpu", dtype="float64"),
        model=_TimeRampDriftModel(slope=slope, intercept=intercept),
    )
    dt = 1.0 / n_steps
    expected_shift = sum((intercept + slope * (step_idx * dt)) * dt for step_idx in range(n_steps))
    initial_mean = float(problem.initial[(sample_id, "ctrl")].mean()[0])
    expected_mean = initial_mean + expected_shift
    pred_mean = float(result.summary.iloc[0]["terminal_mean_0"])
    expected_var = float(problem.initial[(sample_id, "ctrl")].variance_trace())
    pred_var = float(result.summary.iloc[0]["terminal_var_trace"])
    passed = bool(result.stable and abs(pred_mean - expected_mean) <= 1e-12 and abs(pred_var - expected_var) <= 1e-12)
    return FullJointSimCaseResult(
        case_name="time_dependent_drift_schedule_exact",
        description="Time-dependent deterministic drift should match the exact left-Riemann schedule used by the simulator.",
        passed=passed,
        metrics={
            "stable": bool(result.stable),
            "pred_mean": pred_mean,
            "expected_mean": expected_mean,
            "mean_error": abs(pred_mean - expected_mean),
            "var_error": abs(pred_var - expected_var),
            "expected_shift": expected_shift,
        },
    )


def _case_stochastic_diffusion_moment_recovery() -> FullJointSimCaseResult:
    sample_id = "screenA"
    n_particles = 2048
    initial_support = np.repeat(np.array([[0.3, -0.2]], dtype=float), n_particles, axis=0)
    initial = {
        (sample_id, "ctrl"): _measure(
            initial_support,
            mass=1.0,
            perturbation_id="ctrl",
            time_label="P4",
            sample_id=sample_id,
        )
    }
    problem = _endpoint_problem(initial, latent_dim=2)
    sigma = np.array([0.2, 0.4], dtype=float)
    result = _run(
        problem,
        config=LearnedSimulatorConfig(n_steps=24, seed=41, device="cpu", dtype="float64"),
        model=_FixedFieldModel(
            latent_dim=2,
            context_dim=2,
            drift_by_perturbation={"ctrl": np.array([0.0, 0.0], dtype=float)},
            diffusion_by_perturbation={"ctrl": sigma},
            growth_by_perturbation={"ctrl": 0.0},
            context_by_sample={sample_id: np.array([0.0, 0.0], dtype=float)},
        ),
    )
    row = result.summary.iloc[0]
    pred_mean = _summary_mean_vector(row)
    expected_mean = np.array([0.3, -0.2], dtype=float)
    pred_trace = float(row["terminal_var_trace"])
    expected_trace = float(np.sum(sigma**2))
    terminal_cov = result.terminal_measures[(sample_id, "ctrl")].covariance()
    offdiag_abs = abs(float(terminal_cov[0, 1]))
    max_mean_error = float(np.max(np.abs(pred_mean - expected_mean)))
    trace_error = abs(pred_trace - expected_trace)
    passed = bool(result.stable and max_mean_error <= 0.03 and trace_error <= 0.03 and offdiag_abs <= 0.02)
    return FullJointSimCaseResult(
        case_name="stochastic_diffusion_moment_recovery",
        description="With many particles and constant diagonal diffusion, empirical terminal moments should recover the analytic Brownian moments.",
        passed=passed,
        metrics={
            "stable": bool(result.stable),
            "n_particles": n_particles,
            "max_mean_error": max_mean_error,
            "trace_error": trace_error,
            "offdiag_abs": offdiag_abs,
            "pred_trace": pred_trace,
            "expected_trace": expected_trace,
        },
    )


def _case_multichannel_context_routing_exact() -> FullJointSimCaseResult:
    latent_dim = 2
    initial: Dict[object, FiniteMeasure] = {}
    sample_total_masses = {"screen_low": 1.5, "screen_high": 3.0}
    for sample_id, total_mass in sample_total_masses.items():
        ctrl_mass = 1.0
        driver_mass = total_mass - ctrl_mass
        initial[(sample_id, "ctrl")] = _measure(
            np.array([[0.0, 0.5], [1.0, -0.5]], dtype=float),
            mass=ctrl_mass,
            perturbation_id="ctrl",
            time_label="P4",
            sample_id=sample_id,
        )
        initial[(sample_id, "driver")] = _measure(
            np.array([[0.2, 0.0], [0.6, 0.4]], dtype=float),
            mass=driver_mass,
            perturbation_id="driver",
            time_label="P4",
            sample_id=sample_id,
        )
    context_weight = np.array([[0.10, -0.02], [0.05, 0.03]], dtype=float)
    problem = _endpoint_problem(initial, latent_dim=latent_dim)
    result = _run(
        problem,
        config=LearnedSimulatorConfig(n_steps=5, seed=43, device="cpu", dtype="float64"),
        model=_LinearContextVectorModel(latent_dim=latent_dim, context_weight=context_weight),
    )
    summary = result.summary[result.summary["perturbation_id"] == "ctrl"].set_index("sample_id")
    max_mean_error = 0.0
    for sample_id, total_mass in sample_total_masses.items():
        context = np.array([total_mass, total_mass**2], dtype=float)
        expected_shift = context_weight @ context
        expected_mean = problem.initial[(sample_id, "ctrl")].mean() + expected_shift
        pred_mean = np.array(summary.loc[sample_id, ["terminal_mean_0", "terminal_mean_1"]], dtype=float)
        max_mean_error = max(max_mean_error, float(np.max(np.abs(pred_mean - expected_mean))))
    context_columns = [column for column in result.context_summary.columns if column.startswith("context_")]
    passed = bool(result.stable and max_mean_error <= 1e-12 and context_columns == ["context_0", "context_1"])
    return FullJointSimCaseResult(
        case_name="multichannel_context_routing_exact",
        description="A 2D drift driven by a 2-channel sample context should reproduce the exact analytic screen-specific shifts.",
        passed=passed,
        metrics={
            "stable": bool(result.stable),
            "max_mean_error": max_mean_error,
            "n_context_columns": len(context_columns),
            "screen_low_context_0": float(
                result.context_summary[result.context_summary["sample_id"] == "screen_low"]["context_0"].iloc[-1]
            ),
            "screen_high_context_1": float(
                result.context_summary[result.context_summary["sample_id"] == "screen_high"]["context_1"].iloc[-1]
            ),
        },
    )


def _case_mass_feedback_context_growth_exact() -> FullJointSimCaseResult:
    ctrl_mass = 1.0
    driver_masses = {"screen_low": 0.5, "screen_high": 2.0}
    initial: Dict[object, FiniteMeasure] = {}
    for sample_id, driver_mass in driver_masses.items():
        initial[(sample_id, "ctrl")] = _measure(
            np.array([[0.0], [0.2]], dtype=float),
            mass=ctrl_mass,
            perturbation_id="ctrl",
            time_label="P4",
            sample_id=sample_id,
        )
        initial[(sample_id, "driver")] = _measure(
            np.array([[0.1], [0.3]], dtype=float),
            mass=driver_mass,
            perturbation_id="driver",
            time_label="P4",
            sample_id=sample_id,
        )
    problem = _endpoint_problem(initial, latent_dim=1)
    eta = 0.12
    driver_growth = 0.5
    n_steps = 12
    dt = 1.0 / n_steps
    result = _run(
        problem,
        config=LearnedSimulatorConfig(n_steps=n_steps, seed=47, device="cpu", dtype="float64"),
        model=_TotalMassFeedbackGrowthModel(
            eta=eta,
            growth_by_perturbation={"ctrl": 0.0, "driver": driver_growth},
        ),
    )
    summary = result.summary.set_index(["sample_id", "perturbation_id"])
    expected_ctrl_means = {}
    expected_driver_masses = {}
    expected_final_context = {}
    for sample_id, driver_mass in driver_masses.items():
        initial_ctrl_mean = float(problem.initial[(sample_id, "ctrl")].mean()[0])
        mass_path = [ctrl_mass + driver_mass * math.exp(driver_growth * step_idx * dt) for step_idx in range(n_steps)]
        expected_ctrl_means[sample_id] = initial_ctrl_mean + eta * dt * sum(mass_path)
        expected_driver_masses[sample_id] = driver_mass * math.exp(driver_growth)
        expected_final_context[sample_id] = ctrl_mass + driver_mass * math.exp(driver_growth)
    max_ctrl_mean_error = max(
        abs(float(summary.loc[(sample_id, "ctrl"), "terminal_mean_0"]) - expected_ctrl_means[sample_id])
        for sample_id in driver_masses
    )
    max_driver_mass_error = max(
        abs(float(summary.loc[(sample_id, "driver"), "terminal_mass"]) - expected_driver_masses[sample_id])
        for sample_id in driver_masses
    )
    final_context = (
        result.context_summary[result.context_summary["step"] == n_steps].set_index("sample_id")["context_0"].to_dict()
    )
    max_context_error = max(
        abs(float(final_context[sample_id]) - expected_final_context[sample_id]) for sample_id in driver_masses
    )
    ctrl_delta_pred = float(
        summary.loc[("screen_high", "ctrl"), "terminal_mean_0"] - summary.loc[("screen_low", "ctrl"), "terminal_mean_0"]
    )
    ctrl_delta_expected = expected_ctrl_means["screen_high"] - expected_ctrl_means["screen_low"]
    passed = bool(
        result.stable
        and max_ctrl_mean_error <= 1e-12
        and max_driver_mass_error <= 1e-12
        and max_context_error <= 1e-12
        and abs(ctrl_delta_pred - ctrl_delta_expected) <= 1e-12
    )
    return FullJointSimCaseResult(
        case_name="mass_feedback_context_growth_exact",
        description="Sample total mass should evolve through growth and feed back exactly into context-driven control drift.",
        passed=passed,
        metrics={
            "stable": bool(result.stable),
            "max_ctrl_mean_error": max_ctrl_mean_error,
            "max_driver_mass_error": max_driver_mass_error,
            "max_context_error": max_context_error,
            "ctrl_delta_pred": ctrl_delta_pred,
            "ctrl_delta_expected": ctrl_delta_expected,
            "ctrl_delta_error": abs(ctrl_delta_pred - ctrl_delta_expected),
        },
    )


def _case_state_dependent_growth_reweighting_exact() -> FullJointSimCaseResult:
    sample_id = "screenA"
    support = np.array([[-1.0], [0.0], [1.0], [2.0]], dtype=float)
    mass0 = 1.2
    initial = {
        (sample_id, "ctrl"): _measure(
            support,
            mass=mass0,
            perturbation_id="ctrl",
            time_label="P4",
            sample_id=sample_id,
        )
    }
    problem = _endpoint_problem(initial, latent_dim=1)
    drift_velocity = 0.3
    growth_intercept = -0.2
    growth_slope = 0.4
    n_steps = 20
    dt = 1.0 / n_steps
    result = _run(
        problem,
        config=LearnedSimulatorConfig(n_steps=n_steps, seed=53, device="cpu", dtype="float64"),
        model=_StateDependentGrowthModel(
            drift_velocity=drift_velocity,
            growth_intercept=growth_intercept,
            growth_slope=growth_slope,
        ),
    )
    support_terminal = support.reshape(-1) + drift_velocity
    discrete_growth = dt * (
        n_steps * growth_intercept
        + growth_slope * (n_steps * support.reshape(-1) + drift_velocity * dt * sum(range(n_steps)))
    )
    expected_weights = (mass0 / support.shape[0]) * np.exp(discrete_growth)
    expected_mass = float(expected_weights.sum())
    normalized = expected_weights / expected_mass
    expected_mean = float(np.sum(normalized * support_terminal))
    expected_var = float(np.sum(normalized * (support_terminal - expected_mean) ** 2))
    expected_weight_vector = expected_weights
    row = result.summary.iloc[0]
    pred_measure = result.terminal_measures[(sample_id, "ctrl")]
    pred_weights = pred_measure.weights.reshape(-1)
    max_weight_error = float(np.max(np.abs(pred_weights - expected_weight_vector)))
    mass_error = abs(float(row["terminal_mass"]) - expected_mass)
    mean_error = abs(float(row["terminal_mean_0"]) - expected_mean)
    var_error = abs(float(row["terminal_var_trace"]) - expected_var)
    passed = bool(
        result.stable
        and mass_error <= 1e-12
        and mean_error <= 1e-12
        and var_error <= 1e-12
        and max_weight_error <= 1e-12
    )
    return FullJointSimCaseResult(
        case_name="state_dependent_growth_reweighting_exact",
        description="State-dependent growth should induce the exact discrete terminal reweighting, mass, mean, and variance.",
        passed=passed,
        metrics={
            "stable": bool(result.stable),
            "mass_error": mass_error,
            "mean_error": mean_error,
            "var_error": var_error,
            "max_weight_error": max_weight_error,
            "expected_mass": expected_mass,
            "expected_mean": expected_mean,
        },
    )


def _case_sample_mean_context_feedback_exact() -> FullJointSimCaseResult:
    ctrl_mass = 1.0
    driver_mass = 1.5
    initial_driver_means = {"screen_low": -1.0, "screen_high": 2.0}
    initial: Dict[object, FiniteMeasure] = {}
    for sample_id, driver_mean in initial_driver_means.items():
        initial[(sample_id, "ctrl")] = _measure(
            np.array([[0.0], [0.2]], dtype=float),
            mass=ctrl_mass,
            perturbation_id="ctrl",
            time_label="P4",
            sample_id=sample_id,
        )
        initial[(sample_id, "driver")] = _measure(
            np.array([[driver_mean - 0.1], [driver_mean + 0.1]], dtype=float),
            mass=driver_mass,
            perturbation_id="driver",
            time_label="P4",
            sample_id=sample_id,
        )
    problem = _endpoint_problem(initial, latent_dim=1)
    eta = 0.3
    driver_velocity = 0.25
    n_steps = 16
    dt = 1.0 / n_steps
    result = _run(
        problem,
        config=LearnedSimulatorConfig(n_steps=n_steps, seed=59, device="cpu", dtype="float64"),
        model=_SampleMeanContextFeedbackModel(eta=eta, driver_velocity=driver_velocity),
    )
    summary = result.summary.set_index(["sample_id", "perturbation_id"])
    expected_ctrl_means: Dict[str, float] = {}
    expected_final_context: Dict[str, float] = {}
    max_traj_context_error = 0.0
    for sample_id in initial_driver_means:
        ctrl_mean = float(problem.initial[(sample_id, "ctrl")].mean()[0])
        driver_mean = float(problem.initial[(sample_id, "driver")].mean()[0])
        expected_context_path = []
        for _ in range(n_steps):
            context_value = (ctrl_mass * ctrl_mean + driver_mass * driver_mean) / (ctrl_mass + driver_mass)
            expected_context_path.append(context_value)
            ctrl_mean = ctrl_mean + eta * context_value * dt
            driver_mean = driver_mean + driver_velocity * dt
        expected_ctrl_means[sample_id] = ctrl_mean
        expected_final_context[sample_id] = (ctrl_mass * ctrl_mean + driver_mass * driver_mean) / (ctrl_mass + driver_mass)
        pred_context = (
            result.context_summary[result.context_summary["sample_id"] == sample_id]
            .sort_values("step")["context_0"]
            .to_numpy(dtype=float)
        )
        expected_context = np.array(expected_context_path + [expected_final_context[sample_id]], dtype=float)
        max_traj_context_error = max(max_traj_context_error, float(np.max(np.abs(pred_context - expected_context))))
    max_ctrl_mean_error = max(
        abs(float(summary.loc[(sample_id, "ctrl"), "terminal_mean_0"]) - expected_ctrl_means[sample_id])
        for sample_id in initial_driver_means
    )
    ctrl_delta_pred = float(
        summary.loc[("screen_high", "ctrl"), "terminal_mean_0"] - summary.loc[("screen_low", "ctrl"), "terminal_mean_0"]
    )
    ctrl_delta_expected = expected_ctrl_means["screen_high"] - expected_ctrl_means["screen_low"]
    passed = bool(
        result.stable
        and max_ctrl_mean_error <= 1e-12
        and max_traj_context_error <= 1e-12
        and abs(ctrl_delta_pred - ctrl_delta_expected) <= 1e-12
    )
    return FullJointSimCaseResult(
        case_name="sample_mean_context_feedback_exact",
        description="Context computed from the evolving sample-wide weighted mean should feed back exactly into control drift at every step.",
        passed=passed,
        metrics={
            "stable": bool(result.stable),
            "max_ctrl_mean_error": max_ctrl_mean_error,
            "max_traj_context_error": max_traj_context_error,
            "ctrl_delta_pred": ctrl_delta_pred,
            "ctrl_delta_expected": ctrl_delta_expected,
            "ctrl_delta_error": abs(ctrl_delta_pred - ctrl_delta_expected),
        },
    )


def evaluate_full_joint_sim_cases() -> FullJointSimCaseSuite:
    """Run advanced direct simulation tests for the full joint simulator."""

    results = [
        _case_multidimensional_constant_fields_exact(),
        _case_context_coupled_screen_shift_exact(),
        _case_driver_mass_monotonicity(),
        _case_euler_convergence_linear_drift(),
        _case_time_dependent_drift_schedule_exact(),
        _case_stochastic_diffusion_moment_recovery(),
        _case_multichannel_context_routing_exact(),
        _case_mass_feedback_context_growth_exact(),
        _case_state_dependent_growth_reweighting_exact(),
        _case_sample_mean_context_feedback_exact(),
        _case_particles_per_atom_invariance(),
    ]
    rows = []
    for result in results:
        row = {"case_name": result.case_name, "passed": result.passed, "description": result.description}
        row.update(result.metrics)
        rows.append(row)
    summary_table = pd.DataFrame(rows).sort_values("case_name").reset_index(drop=True)
    return FullJointSimCaseSuite(results=results, summary_table=summary_table)
