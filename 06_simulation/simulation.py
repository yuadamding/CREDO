#!/usr/bin/env python3
"""
simulate_minimum_useful_benchmarks_reviewed.py

Reviewed and tightened reference implementation of the two-stage simulation plan
for a control-anchored mean-field neural SDE benchmark.

Key improvements relative to the first draft
-------------------------------------------
1) Adds argument validation.
2) Uses numerically stable sigmoid / exp helpers.
3) Uses separate spawned seeds for core and mean-field stages.
4) In the mean-field benchmark, the hidden simulator can start from the
   observed P4 samples (default), which more faithfully matches the benchmark
   specification.
5) Uses the exact OU transition under piecewise-constant context in the
   mean-field benchmark, reducing avoidable Euler discretization bias.
6) Exports exact terminal masses for the current constant-reaction benchmark.
7) Optionally exports hidden terminal particles for debugging.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Configuration dataclasses
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PerturbationParams:
    theta: float
    sigma: float
    rho: float


@dataclass
class CoreBenchmarkConfig:
    seed: int = 0
    n_obs_p4: int = 256
    n_obs_p60: int = 256
    T: float = 1.0
    m0: float = -1.0
    sd0: float = 0.15
    kappa: float = 2.0
    store_paths: bool = False
    n_path_particles: int = 4096
    n_path_steps: int = 200


@dataclass
class MeanFieldBenchmarkConfig:
    seed: int = 0
    n_obs_p4: int = 256
    n_obs_p60: int = 256
    T: float = 1.0
    m0: float = -1.0
    sd0: float = 0.15
    kappa: float = 2.0
    eta: float = 0.8
    n_sim_particles: int = 256
    n_sim_steps: int = 200
    driver_m0_screen1: float = 0.5
    driver_m0_screen2: float = 2.0
    use_observed_p4_as_particles: bool = True
    export_hidden_terminal_particles: bool = False


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")


def validate_core_config(cfg: CoreBenchmarkConfig) -> None:
    _validate_positive_int("n_obs_p4", cfg.n_obs_p4)
    _validate_positive_int("n_obs_p60", cfg.n_obs_p60)
    _validate_positive_int("n_path_particles", cfg.n_path_particles)
    _validate_positive_int("n_path_steps", cfg.n_path_steps)
    if cfg.T <= 0:
        raise ValueError(f"T must be positive, got {cfg.T}.")
    if cfg.sd0 <= 0:
        raise ValueError(f"sd0 must be positive, got {cfg.sd0}.")
    if cfg.kappa < 0:
        raise ValueError(f"kappa must be nonnegative, got {cfg.kappa}.")


def validate_mean_field_config(cfg: MeanFieldBenchmarkConfig) -> None:
    _validate_positive_int("n_obs_p4", cfg.n_obs_p4)
    _validate_positive_int("n_obs_p60", cfg.n_obs_p60)
    _validate_positive_int("n_sim_particles", cfg.n_sim_particles)
    _validate_positive_int("n_sim_steps", cfg.n_sim_steps)
    if cfg.T <= 0:
        raise ValueError(f"T must be positive, got {cfg.T}.")
    if cfg.sd0 <= 0:
        raise ValueError(f"sd0 must be positive, got {cfg.sd0}.")
    if cfg.kappa < 0:
        raise ValueError(f"kappa must be nonnegative, got {cfg.kappa}.")
    if cfg.driver_m0_screen1 <= 0 or cfg.driver_m0_screen2 <= 0:
        raise ValueError("Driver initial masses must be positive.")


# -----------------------------------------------------------------------------
# Ground-truth parameter builders
# -----------------------------------------------------------------------------


def build_core_truth_params() -> Dict[str, PerturbationParams]:
    return {
        "ctrl": PerturbationParams(theta=0.0, sigma=0.15, rho=0.0),
        "drift": PerturbationParams(theta=0.6, sigma=0.15, rho=0.0),
        "diff": PerturbationParams(theta=0.0, sigma=0.35, rho=0.0),
        "react": PerturbationParams(theta=0.0, sigma=0.15, rho=-0.7),
    }


def build_mean_field_truth_params() -> Dict[str, PerturbationParams]:
    params = build_core_truth_params()
    params["driver"] = PerturbationParams(theta=1.0, sigma=0.15, rho=float(np.log(2.0)))
    return params


# -----------------------------------------------------------------------------
# Numeric helpers
# -----------------------------------------------------------------------------


def stable_sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(x, dtype=float)
    out = np.empty_like(arr, dtype=float)
    pos = arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-arr[pos]))
    expx = np.exp(arr[~pos])
    out[~pos] = expx / (1.0 + expx)
    return float(out) if np.ndim(arr) == 0 else out


def safe_exp(x: np.ndarray | float) -> np.ndarray | float:
    out = np.exp(np.clip(x, -700.0, 700.0))
    return float(out) if np.ndim(np.asarray(x)) == 0 else out


def occupancy(z: np.ndarray) -> np.ndarray:
    return stable_sigmoid(6.0 * z)


def ou_terminal_moments(
    m0: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float = 1.0,
) -> Tuple[float, float]:
    if kappa == 0.0:
        mean_T = m0
        var_T = v0 + sigma**2 * T
        return mean_T, var_T
    exp_term = np.exp(-kappa * T)
    mean_T = theta + (m0 - theta) * exp_term
    var_T = v0 * np.exp(-2.0 * kappa * T) + (sigma**2 / (2.0 * kappa)) * (1.0 - np.exp(-2.0 * kappa * T))
    return mean_T, var_T


def exact_ou_step(
    z: np.ndarray,
    kappa: float,
    theta: float,
    sigma: float,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if kappa == 0.0:
        return z + sigma * np.sqrt(dt) * rng.normal(size=z.shape)
    exp_term = np.exp(-kappa * dt)
    sd = sigma * np.sqrt((1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa))
    return theta + exp_term * (z - theta) + sd * rng.normal(size=z.shape)


def exact_affine_ou_step_frozen_context(
    z: np.ndarray,
    kappa: float,
    theta: float,
    eta_c: float,
    sigma: float,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Exact transition for dz = [kappa(theta-z) + eta_c] dt + sigma dW
    when eta_c is held fixed over the step.
    """
    if kappa == 0.0:
        mean = z + eta_c * dt
        sd = sigma * np.sqrt(dt)
        return mean + sd * rng.normal(size=z.shape)
    exp_term = np.exp(-kappa * dt)
    center = theta + eta_c / kappa
    sd = sigma * np.sqrt((1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa))
    return center + exp_term * (z - center) + sd * rng.normal(size=z.shape)


def weighted_mean_and_var(x: np.ndarray, w: np.ndarray) -> Tuple[float, float]:
    w = np.asarray(w, dtype=float)
    x = np.asarray(x, dtype=float)
    w_sum = w.sum()
    if w_sum <= 0:
        raise ValueError("Weights must sum to a positive number.")
    p = w / w_sum
    mean = float(np.sum(p * x))
    var = float(np.sum(p * (x - mean) ** 2))
    return mean, var


def sample_from_weighted_particles(
    z: np.ndarray,
    logw: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    a = np.asarray(logw, dtype=float)
    if a.ndim != 1 or len(a) != len(z):
        raise ValueError("logw and z must be one-dimensional arrays of equal length.")
    a = a - np.max(a)
    p = np.exp(a)
    p_sum = p.sum()
    if not np.isfinite(p_sum) or p_sum <= 0:
        raise ValueError("Could not normalize particle weights for sampling.")
    p = p / p_sum
    idx = rng.choice(len(z), size=n, replace=True, p=p)
    return z[idx]


# -----------------------------------------------------------------------------
# Stage I: core benchmark
# -----------------------------------------------------------------------------


def simulate_core_benchmark(cfg: CoreBenchmarkConfig) -> Dict[str, pd.DataFrame | Dict[str, pd.DataFrame]]:
    validate_core_config(cfg)
    rng = np.random.default_rng(cfg.seed)
    params = build_core_truth_params()

    p4_rows: List[dict] = []
    p60_rows: List[dict] = []
    mass_rows: List[dict] = []
    param_rows: List[dict] = []
    summary_rows: List[dict] = []

    for g, p in params.items():
        x = rng.normal(loc=cfg.m0, scale=cfg.sd0, size=cfg.n_obs_p4)

        mean1, var1 = ou_terminal_moments(
            m0=cfg.m0,
            v0=cfg.sd0**2,
            kappa=cfg.kappa,
            theta=p.theta,
            sigma=p.sigma,
            T=cfg.T,
        )
        y = rng.normal(loc=mean1, scale=np.sqrt(var1), size=cfg.n_obs_p60)

        M0 = 1.0
        M1 = float(np.exp(p.rho * cfg.T) * M0)

        for i, zi in enumerate(x):
            p4_rows.append(
                {
                    "stage": "core",
                    "screen": "screen1",
                    "time_label": "P4",
                    "t": 0.0,
                    "perturbation": g,
                    "cell_id": f"core_P4_{g}_{i}",
                    "z": float(zi),
                }
            )

        for j, zj in enumerate(y):
            p60_rows.append(
                {
                    "stage": "core",
                    "screen": "screen1",
                    "time_label": "P60",
                    "t": 1.0,
                    "perturbation": g,
                    "cell_id": f"core_P60_{g}_{j}",
                    "z": float(zj),
                }
            )

        mass_rows.append(
            {
                "stage": "core",
                "screen": "screen1",
                "perturbation": g,
                "M0": M0,
                "M1": M1,
                "n_obs_p4": cfg.n_obs_p4,
                "n_obs_p60": cfg.n_obs_p60,
            }
        )

        param_rows.append(
            {
                "stage": "core",
                "screen": "screen1",
                "perturbation": g,
                "theta": p.theta,
                "sigma": p.sigma,
                "rho": p.rho,
                "kappa": cfg.kappa,
            }
        )

        summary_rows.append(
            {
                "stage": "core",
                "screen": "screen1",
                "perturbation": g,
                "truth_terminal_mean": mean1,
                "truth_terminal_var": var1,
                "truth_terminal_mass": M1,
                "summary_kind": "analytic",
            }
        )

    outputs: Dict[str, pd.DataFrame | Dict[str, pd.DataFrame]] = {
        "p4_cells": pd.DataFrame(p4_rows),
        "p60_cells": pd.DataFrame(p60_rows),
        "masses": pd.DataFrame(mass_rows),
        "truth_params": pd.DataFrame(param_rows),
        "truth_summary": pd.DataFrame(summary_rows),
    }

    if cfg.store_paths:
        outputs["paths"] = simulate_core_reference_paths(cfg, params)

    return outputs


def simulate_core_reference_paths(
    cfg: CoreBenchmarkConfig,
    params: Dict[str, PerturbationParams],
) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed + 10_000)
    dt = cfg.T / cfg.n_path_steps
    t_grid = np.linspace(0.0, cfg.T, cfg.n_path_steps + 1)

    path_rows: List[pd.DataFrame] = []
    for g, p in params.items():
        z = rng.normal(loc=cfg.m0, scale=cfg.sd0, size=cfg.n_path_particles)
        logw = np.zeros(cfg.n_path_particles, dtype=float)
        rows = []
        for step, t in enumerate(t_grid):
            if step > 0:
                z = exact_ou_step(z, cfg.kappa, p.theta, p.sigma, dt, rng)
                logw += p.rho * dt
            rows.append(
                pd.DataFrame(
                    {
                        "perturbation": g,
                        "step": step,
                        "t": t,
                        "particle_id": np.arange(cfg.n_path_particles),
                        "z": z.copy(),
                        "logw": logw.copy(),
                        "w": safe_exp(logw),
                    }
                )
            )
        path_rows.append(pd.concat(rows, ignore_index=True))
    return {"path_particles": pd.concat(path_rows, ignore_index=True)}


# -----------------------------------------------------------------------------
# Stage II: mean-field benchmark
# -----------------------------------------------------------------------------


def _spawn_rngs(seed: int, n_children: int) -> List[np.random.Generator]:
    children = np.random.SeedSequence(seed).spawn(n_children)
    return [np.random.default_rng(child) for child in children]


def _resample_to_size(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(x) == n:
        return x.copy()
    idx = rng.choice(len(x), size=n, replace=True)
    return x[idx].copy()


def simulate_mean_field_benchmark(cfg: MeanFieldBenchmarkConfig) -> Dict[str, pd.DataFrame]:
    validate_mean_field_config(cfg)
    rng_obs, rng_hidden, rng_term = _spawn_rngs(cfg.seed, 3)

    params = build_mean_field_truth_params()
    perturbations = list(params.keys())
    screens = ["screen1", "screen2"]

    M0_by_screen = {
        "screen1": {g: 1.0 for g in perturbations},
        "screen2": {g: 1.0 for g in perturbations},
    }
    M0_by_screen["screen1"]["driver"] = float(cfg.driver_m0_screen1)
    M0_by_screen["screen2"]["driver"] = float(cfg.driver_m0_screen2)

    # Observed P4 snapshots and a cached copy for optional hidden-particle initialization.
    p4_rows: List[dict] = []
    observed_p4: Dict[str, Dict[str, np.ndarray]] = {s: {} for s in screens}
    for s in screens:
        for g in perturbations:
            x = rng_obs.normal(loc=cfg.m0, scale=cfg.sd0, size=cfg.n_obs_p4)
            observed_p4[s][g] = x
            for i, zi in enumerate(x):
                p4_rows.append(
                    {
                        "stage": "mean_field",
                        "screen": s,
                        "time_label": "P4",
                        "t": 0.0,
                        "perturbation": g,
                        "cell_id": f"mf_P4_{s}_{g}_{i}",
                        "z": float(zi),
                    }
                )

    # Hidden particle system used to define screen-specific context and terminal law.
    states: Dict[str, Dict[str, np.ndarray]] = {s: {} for s in screens}
    logw: Dict[str, Dict[str, np.ndarray]] = {s: {} for s in screens}
    for s in screens:
        for g in perturbations:
            if cfg.use_observed_p4_as_particles:
                states[s][g] = _resample_to_size(observed_p4[s][g], cfg.n_sim_particles, rng_hidden)
            else:
                states[s][g] = rng_hidden.normal(loc=cfg.m0, scale=cfg.sd0, size=cfg.n_sim_particles)
            logw[s][g] = np.zeros(cfg.n_sim_particles, dtype=float)

    dt = cfg.T / cfg.n_sim_steps
    t_grid = np.linspace(0.0, cfg.T, cfg.n_sim_steps + 1)

    context_rows: List[dict] = []

    for step, t in enumerate(t_grid):
        c_map: Dict[str, float] = {}

        for s in screens:
            total_mass = 0.0
            total_occ = 0.0
            per_g_rows = []
            for g in perturbations:
                M0 = M0_by_screen[s][g]
                w = safe_exp(logw[s][g])
                z = states[s][g]
                mass_g = M0 * float(np.mean(w))
                occ_g = M0 * float(np.mean(w * occupancy(z)))
                total_mass += mass_g
                total_occ += occ_g
                per_g_rows.append((g, mass_g, occ_g))

            if total_mass <= 0:
                raise ValueError(f"Nonpositive total mass encountered in screen {s} at step {step}.")

            c_s = total_occ / total_mass
            c_map[s] = float(c_s)
            context_rows.append(
                {
                    "stage": "mean_field",
                    "screen": s,
                    "step": step,
                    "t": float(t),
                    "context": float(c_s),
                    "total_mass": float(total_mass),
                    **{f"mass_{g}": mass_g for g, mass_g, _ in per_g_rows},
                    **{f"occ_{g}": occ_g for g, _, occ_g in per_g_rows},
                }
            )

        if step == cfg.n_sim_steps:
            break

        # Exact affine OU transition with piecewise-constant context over each step.
        for s in screens:
            c_s = c_map[s]
            for g, p in params.items():
                z = states[s][g]
                states[s][g] = exact_affine_ou_step_frozen_context(
                    z=z,
                    kappa=cfg.kappa,
                    theta=p.theta,
                    eta_c=cfg.eta * c_s,
                    sigma=p.sigma,
                    dt=dt,
                    rng=rng_hidden,
                )
                logw[s][g] = logw[s][g] + p.rho * dt

    p60_rows: List[dict] = []
    mass_rows: List[dict] = []
    param_rows: List[dict] = []
    summary_rows: List[dict] = []
    hidden_terminal_rows: List[dict] = []

    for s in screens:
        for g, p in params.items():
            zT = states[s][g]
            logwT = logw[s][g]
            M0 = M0_by_screen[s][g]
            wT = safe_exp(logwT)
            M1_exact = float(M0 * np.exp(p.rho * cfg.T))

            y = sample_from_weighted_particles(zT, logwT, cfg.n_obs_p60, rng_term)
            for j, zj in enumerate(y):
                p60_rows.append(
                    {
                        "stage": "mean_field",
                        "screen": s,
                        "time_label": "P60",
                        "t": 1.0,
                        "perturbation": g,
                        "cell_id": f"mf_P60_{s}_{g}_{j}",
                        "z": float(zj),
                    }
                )

            truth_mean, truth_var = weighted_mean_and_var(zT, wT)

            mass_rows.append(
                {
                    "stage": "mean_field",
                    "screen": s,
                    "perturbation": g,
                    "M0": float(M0),
                    "M1": float(M1_exact),
                    "n_obs_p4": cfg.n_obs_p4,
                    "n_obs_p60": cfg.n_obs_p60,
                }
            )
            param_rows.append(
                {
                    "stage": "mean_field",
                    "screen": s,
                    "perturbation": g,
                    "theta": p.theta,
                    "sigma": p.sigma,
                    "rho": p.rho,
                    "kappa": cfg.kappa,
                    "eta": cfg.eta,
                }
            )
            summary_rows.append(
                {
                    "stage": "mean_field",
                    "screen": s,
                    "perturbation": g,
                    "truth_terminal_mean": float(truth_mean),
                    "truth_terminal_var": float(truth_var),
                    "truth_terminal_mass": float(M1_exact),
                    "summary_kind": "particle_monte_carlo_for_state_exact_for_mass",
                }
            )

            if cfg.export_hidden_terminal_particles:
                for i, (zi, lwi) in enumerate(zip(zT, logwT)):
                    hidden_terminal_rows.append(
                        {
                            "stage": "mean_field",
                            "screen": s,
                            "perturbation": g,
                            "particle_id": i,
                            "z": float(zi),
                            "logw": float(lwi),
                            "w": float(safe_exp(lwi)),
                        }
                    )

    outputs = {
        "p4_cells": pd.DataFrame(p4_rows),
        "p60_cells": pd.DataFrame(p60_rows),
        "masses": pd.DataFrame(mass_rows),
        "truth_params": pd.DataFrame(param_rows),
        "truth_summary": pd.DataFrame(summary_rows),
        "context_trajectory": pd.DataFrame(context_rows),
    }
    if cfg.export_hidden_terminal_particles:
        outputs["hidden_terminal_particles"] = pd.DataFrame(hidden_terminal_rows)
    return outputs


# -----------------------------------------------------------------------------
# Writing helpers
# -----------------------------------------------------------------------------


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_outputs(
    outdir: str | Path,
    outputs: Dict[str, pd.DataFrame | Dict[str, pd.DataFrame]],
    config_dict: dict,
) -> None:
    outdir = ensure_dir(outdir)

    for key, value in outputs.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(outdir / f"{key}.csv", index=False)
        elif isinstance(value, dict):
            subdir = ensure_dir(outdir / key)
            for subkey, subval in value.items():
                if isinstance(subval, pd.DataFrame):
                    subval.to_csv(subdir / f"{subkey}.csv", index=False)
                else:
                    raise TypeError(f"Unsupported nested output type for key={subkey}: {type(subval)}")
        else:
            raise TypeError(f"Unsupported output type for key={key}: {type(value)}")

    with open(outdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the minimum useful simulation benchmarks.")
    parser.add_argument("--outdir", type=str, default="out/minimum_useful_benchmarks", help="Output directory.")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "core", "mean_field"],
        help="Which benchmark(s) to generate.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Master random seed.")
    parser.add_argument(
        "--store-core-paths",
        action="store_true",
        help="Also write reference path particles for the core OU benchmark.",
    )
    parser.add_argument("--n-obs-p4", type=int, default=256, help="Observed P4 cells per perturbation.")
    parser.add_argument("--n-obs-p60", type=int, default=256, help="Observed P60 cells per perturbation.")
    parser.add_argument(
        "--n-sim-particles",
        type=int,
        default=256,
        help="Hidden simulation particles per screen and perturbation for the mean-field benchmark.",
    )
    parser.add_argument(
        "--n-sim-steps",
        type=int,
        default=200,
        help="Time steps for the mean-field simulation.",
    )
    parser.add_argument(
        "--independent-hidden-particles",
        action="store_true",
        help="Initialize mean-field hidden particles independently from the observed P4 samples instead of from them.",
    )
    parser.add_argument(
        "--export-hidden-terminal-particles",
        action="store_true",
        help="Export hidden terminal particles for the mean-field benchmark.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)

    child_seeds = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(args.seed).spawn(2)]
    core_seed, mf_seed = child_seeds

    if args.stage in {"all", "core"}:
        core_cfg = CoreBenchmarkConfig(
            seed=core_seed,
            n_obs_p4=args.n_obs_p4,
            n_obs_p60=args.n_obs_p60,
            store_paths=args.store_core_paths,
        )
        core_outputs = simulate_core_benchmark(core_cfg)
        write_outputs(outdir / "core", core_outputs, asdict(core_cfg))

    if args.stage in {"all", "mean_field"}:
        mf_cfg = MeanFieldBenchmarkConfig(
            seed=mf_seed,
            n_obs_p4=args.n_obs_p4,
            n_obs_p60=args.n_obs_p60,
            n_sim_particles=args.n_sim_particles,
            n_sim_steps=args.n_sim_steps,
            use_observed_p4_as_particles=not args.independent_hidden_particles,
            export_hidden_terminal_particles=args.export_hidden_terminal_particles,
        )
        mf_outputs = simulate_mean_field_benchmark(mf_cfg)
        write_outputs(outdir / "mean_field", mf_outputs, asdict(mf_cfg))

    print(f"Done. Wrote outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
