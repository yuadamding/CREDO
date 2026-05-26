#!/usr/bin/env python
"""Heavy randomized stress checks for trajectory-core compatibility.

This script intentionally avoids pytest so it can run many more cases on demand
without slowing the regular test suite.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "package" / "src"))

from credo.data.core import (  # noqa: E402
    CellStateTable,
    MassTable,
    PerturbSeqDynamicsData,
    PerturbationCatalog,
    TimeAxis,
)
from credo.losses.counts import count_fractions_from_zeta, integrated_fitness_curve  # noqa: E402
from credo.losses.multitime import make_observed_tau_grid  # noqa: E402
from credo.models.simulator import initialise_particles  # noqa: E402
from credo.models.weighted_sde import WeightedParticleSimulator  # noqa: E402


def synthetic_data(seed: int, n_pids: int, n_times: int, latent_dim: int) -> PerturbSeqDynamicsData:
    rng = np.random.default_rng(seed)
    labels = [f"t{idx}" for idx in range(n_times)]
    physical_times = np.cumsum(rng.uniform(0.2, 3.0, size=n_times)).tolist()
    pids = [f"pert_{idx}" for idx in range(n_pids)]
    rows = []
    latent = []
    mass_rows = []
    for pid in pids:
        for time_i, label in enumerate(labels):
            for sample_id in ["D0", "D1", "D2"]:
                n_cells = int(rng.integers(1, 12))
                mass_rows.append(
                    {
                        "perturbation_id": pid,
                        "time_label": label,
                        "sample_id": sample_id,
                        "mass": float(rng.uniform(0.01, 5.0)),
                    }
                )
                for cell_i in range(n_cells):
                    rows.append(
                        {
                            "cell_id": f"{pid}_{label}_{sample_id}_{cell_i}",
                            "perturbation_id": pid,
                            "time_label": label,
                            "sample_id": sample_id,
                        }
                    )
                    latent.append(rng.normal(time_i, 0.5, size=latent_dim).astype(np.float32))
    return PerturbSeqDynamicsData(
        time_axis=TimeAxis(labels=labels, physical_times=physical_times),
        catalog=PerturbationCatalog(pids, [pids[0]]),
        cell_state=CellStateTable(pd.DataFrame(rows), np.asarray(latent, dtype=np.float32)),
        mass_table=MassTable(pd.DataFrame(mass_rows)),
    )


class StableDynamics:
    def __init__(self, latent_dim: int, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.A = torch.randn(latent_dim, latent_dim, generator=generator) * 0.02

    def step(self, z, tau, logw, log_m0, perturbation_ids=None):
        A = self.A.to(device=z.device, dtype=z.dtype)
        drift = torch.einsum("gnd,df->gnf", z, A) + tau.reshape(1, 1, 1) * 0.01
        sigma = torch.full_like(z, 0.02)
        growth = 0.01 * z.mean(-1) + 0.02 * logw
        coeffs = SimpleNamespace(drift=drift, sigma_diag=sigma, growth=growth)
        ctx = SimpleNamespace(context=torch.zeros(1, dtype=z.dtype, device=z.device))
        return coeffs, ctx


def assert_close(actual: torch.Tensor, expected: torch.Tensor, message: str) -> None:
    if not torch.allclose(actual, expected, rtol=1e-5, atol=1e-6):
        max_err = float((actual - expected).abs().max())
        raise AssertionError(f"{message}; max_err={max_err:.3g}")


def run_case(seed: int) -> None:
    n_pids = 2 + seed % 6
    n_times = 2 + seed % 5
    latent_dim = 1 + seed % 8
    data = synthetic_data(seed, n_pids=n_pids, n_times=n_times, latent_dim=latent_dim)

    trajectory = data.to_trajectory_problem()
    endpoint_from_trajectory = trajectory.to_endpoint_problem()
    endpoint_direct = data.to_endpoint_problem()
    for pid in endpoint_direct.perturbation_ids:
        if not np.array_equal(endpoint_from_trajectory.initial[pid].support, endpoint_direct.initial[pid].support):
            raise AssertionError(f"initial support mismatch for seed={seed}, pid={pid}")
        if not np.array_equal(endpoint_from_trajectory.terminal[pid].weights, endpoint_direct.terminal[pid].weights):
            raise AssertionError(f"terminal weights mismatch for seed={seed}, pid={pid}")

    n_particles = 2 + seed % 17
    z0, logw0, log_m0 = initialise_particles(
        endpoint_direct,
        endpoint_direct.perturbation_ids,
        n_particles=n_particles,
        seed=seed,
    )
    torch.manual_seed(seed)
    for g, pid in enumerate(endpoint_direct.perturbation_ids):
        support = torch.tensor(endpoint_direct.initial[pid].support, dtype=torch.float32)
        idx = torch.randint(0, len(support), (n_particles,))
        if not torch.equal(z0[g], support[idx]):
            raise AssertionError(f"legacy initializer mismatch for seed={seed}, pid={pid}")

    model = StableDynamics(latent_dim, seed=100_000 + seed)
    n_steps = 1 + seed % 12
    simulator = WeightedParticleSimulator(n_steps=n_steps)
    z_small = z0[: min(3, z0.shape[0]), : min(9, z0.shape[1])].clone()
    logw_small = logw0[: z_small.shape[0], : z_small.shape[1]].clone()
    log_m_small = log_m0[: z_small.shape[0]].clone()
    uniform_grid = torch.linspace(0.0, 1.0, n_steps + 1)
    torch.manual_seed(50_000 + seed)
    default = simulator.rollout(z_small, logw_small, model, log_m_small)
    torch.manual_seed(50_000 + seed)
    explicit = simulator.rollout(z_small, logw_small, model, log_m_small, tau_grid=uniform_grid)
    assert_close(explicit.z_steps, default.z_steps, f"uniform tau_grid z mismatch seed={seed}")
    assert_close(explicit.logw_steps, default.logw_steps, f"uniform tau_grid logw mismatch seed={seed}")

    observed_grid = make_observed_tau_grid(trajectory.observed_taus, steps_per_interval=2)
    if len(observed_grid) != (len(trajectory.observed_taus) - 1) * 2 + 1:
        raise AssertionError(f"observed grid length mismatch seed={seed}")

    generator = torch.Generator().manual_seed(200_000 + seed)
    K = len(observed_grid) - 1
    G = 2 + seed % 4
    N = 2 + seed % 5
    growth = torch.randn(K, G, N, generator=generator)
    logw = torch.randn(K + 1, G, N, generator=generator) * 0.1
    curve = integrated_fitness_curve(growth, logw, observed_grid)
    if curve.shape != (K + 1, G):
        raise AssertionError(f"fitness curve shape mismatch seed={seed}")

    exposures = torch.rand(3, G, generator=generator) + 0.01
    counts = torch.ones(3, G)
    zeta = curve[-1]
    pi = count_fractions_from_zeta(zeta, exposures, counts)
    row_sums = pi.sum(dim=1)
    assert_close(row_sums, torch.ones_like(row_sums), f"count fractions not normalized seed={seed}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=1000)
    args = parser.parse_args()

    for seed in range(args.cases):
        run_case(seed)
        if (seed + 1) % 100 == 0:
            print(f"passed {seed + 1} cases", flush=True)
    print(f"all {args.cases} randomized trajectory-core stress cases passed")


if __name__ == "__main__":
    main()
