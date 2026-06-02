from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from credo.data.core import (
    CellStateTable,
    MassTable,
    PerturbSeqDynamicsData,
    PerturbationCatalog,
    TimeAxis,
)
from credo.losses.counts import count_fractions_from_zeta, integrated_fitness_curve
from credo.losses.multitime import MultiTimeEndpointLoss
from credo.losses.endpoint import EndpointGeometryMassLoss
from credo.models.simulator import initialise_particles
from credo.models.weighted_sde import ParticleRollout, WeightedParticleSimulator


pytestmark = pytest.mark.randomized


def _synthetic_data(seed: int, *, n_pids: int, n_times: int, latent_dim: int) -> PerturbSeqDynamicsData:
    rng = np.random.default_rng(seed)
    labels = [f"t{idx}" for idx in range(n_times)]
    physical_times = np.cumsum(rng.uniform(0.5, 2.0, size=n_times)).tolist()
    pids = [f"pert_{idx}" for idx in range(n_pids)]
    rows = []
    latent = []
    mass_rows = []
    for pid in pids:
        for time_i, label in enumerate(labels):
            for sample_id in ["D0", "D1"]:
                n_cells = int(rng.integers(2, 8))
                mass = float(rng.uniform(0.1, 3.0))
                mass_rows.append(
                    {
                        "perturbation_id": pid,
                        "time_label": label,
                        "sample_id": sample_id,
                        "mass": mass,
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
                    center = np.full(latent_dim, time_i, dtype=np.float32)
                    latent.append(center + rng.normal(0.0, 0.2, size=latent_dim).astype(np.float32))

    return PerturbSeqDynamicsData(
        time_axis=TimeAxis(labels=labels, physical_times=physical_times),
        catalog=PerturbationCatalog(pids, [pids[0]]),
        cell_state=CellStateTable(pd.DataFrame(rows), np.asarray(latent, dtype=np.float32)),
        mass_table=MassTable(pd.DataFrame(mass_rows)),
    )


def _legacy_uniform_rollout(z0, logw0, model, log_m0, *, n_steps: int):
    dtau = 1.0 / n_steps
    tau_steps = torch.linspace(0.0, 1.0, n_steps + 1, device=z0.device, dtype=z0.dtype)
    z_list = [z0]
    logw_list = [logw0]
    z = z0.clone()
    logw = logw0.clone()
    for k in range(n_steps):
        coeffs, _ = model.step(z=z, tau=tau_steps[k], logw=logw, log_m0=log_m0)
        noise = torch.randn_like(z)
        z = z + coeffs.drift * dtau + coeffs.sigma_diag * (dtau**0.5) * noise
        logw = logw + coeffs.growth * dtau
        z_list.append(z)
        logw_list.append(logw)
    return torch.stack(z_list, dim=0), torch.stack(logw_list, dim=0), tau_steps


class _RandomStableDynamics:
    def __init__(self, latent_dim: int, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.A = torch.randn(latent_dim, latent_dim, generator=generator) * 0.03
        self.b = torch.randn(latent_dim, generator=generator) * 0.02

    def step(self, z, tau, logw, log_m0, perturbation_ids=None):
        A = self.A.to(device=z.device, dtype=z.dtype)
        b = self.b.to(device=z.device, dtype=z.dtype)
        drift = torch.einsum("gnd,df->gnf", z, A) + b + tau.reshape(1, 1, 1) * 0.01
        sigma = torch.full_like(z, 0.03) + z.abs().clamp(max=1.0) * 0.002
        growth = 0.05 + 0.02 * logw + 0.01 * z.mean(-1)
        coeffs = SimpleNamespace(drift=drift, sigma_diag=sigma, growth=growth)
        ctx = SimpleNamespace(context=torch.zeros(1, dtype=z.dtype, device=z.device))
        return coeffs, ctx


def test_random_trajectory_endpoint_views_match_direct_endpoint() -> None:
    for seed in range(60):
        data = _synthetic_data(
            seed,
            n_pids=2 + seed % 4,
            n_times=2 + seed % 4,
            latent_dim=2 + seed % 5,
        )
        trajectory = data.to_trajectory_problem()
        endpoint_from_trajectory = trajectory.to_endpoint_problem()
        endpoint_direct = data.to_endpoint_problem()

        assert endpoint_from_trajectory.perturbation_ids == endpoint_direct.perturbation_ids
        for pid in data.catalog.perturbation_ids:
            assert np.array_equal(endpoint_from_trajectory.initial[pid].support, endpoint_direct.initial[pid].support)
            assert np.array_equal(endpoint_from_trajectory.initial[pid].weights, endpoint_direct.initial[pid].weights)
            assert np.array_equal(endpoint_from_trajectory.terminal[pid].support, endpoint_direct.terminal[pid].support)
            assert np.array_equal(endpoint_from_trajectory.terminal[pid].weights, endpoint_direct.terminal[pid].weights)


def test_random_endpoint_initializer_matches_legacy_sampling() -> None:
    for seed in range(80):
        data = _synthetic_data(seed + 1000, n_pids=2 + seed % 3, n_times=2, latent_dim=2 + seed % 4)
        endpoint = data.to_endpoint_problem()
        n_particles = 3 + seed % 13

        z0, logw0, log_m0 = initialise_particles(
            endpoint,
            endpoint.perturbation_ids,
            n_particles=n_particles,
            seed=seed,
        )

        torch.manual_seed(seed)
        expected_z = torch.zeros_like(z0)
        expected_logw = torch.zeros_like(logw0)
        expected_log_m = torch.zeros_like(log_m0)
        for g, pid in enumerate(endpoint.perturbation_ids):
            mu = endpoint.initial[pid]
            support = torch.tensor(mu.support, dtype=torch.float32)
            idx = torch.randint(0, len(support), (n_particles,))
            expected_z[g] = support[idx]
            expected_logw[g] = torch.full((n_particles,), -np.log(n_particles))
            expected_log_m[g] = torch.tensor(np.log(mu.total_mass), dtype=torch.float32)

        assert torch.equal(z0, expected_z)
        assert torch.equal(logw0, expected_logw)
        assert torch.equal(log_m0, expected_log_m)


def test_random_default_rollout_matches_legacy_reference() -> None:
    for seed in range(70):
        latent_dim = 1 + seed % 6
        n_steps = 1 + seed % 9
        z0 = torch.randn(1 + seed % 4, 3 + seed % 8, latent_dim)
        logw0 = torch.randn(z0.shape[:2]) * 0.05
        log_m0 = torch.randn(z0.shape[0]) * 0.1
        model = _RandomStableDynamics(latent_dim, seed=10_000 + seed)
        simulator = WeightedParticleSimulator(n_steps=n_steps)

        torch.manual_seed(seed)
        rollout = simulator.rollout(z0=z0, logw0=logw0, model=model, log_m0=log_m0)
        torch.manual_seed(seed)
        expected_z, expected_logw, expected_tau = _legacy_uniform_rollout(
            z0,
            logw0,
            model,
            log_m0,
            n_steps=n_steps,
        )

        assert torch.equal(rollout.tau_steps, expected_tau)
        assert torch.equal(rollout.z_steps, expected_z)
        assert torch.equal(rollout.logw_steps, expected_logw)


def test_random_explicit_uniform_tau_grid_matches_default_close() -> None:
    for seed in range(40):
        latent_dim = 2 + seed % 5
        n_steps = 2 + seed % 8
        z0 = torch.randn(1 + seed % 3, 4 + seed % 5, latent_dim)
        logw0 = torch.randn(z0.shape[:2]) * 0.05
        log_m0 = torch.randn(z0.shape[0]) * 0.1
        model = _RandomStableDynamics(latent_dim, seed=20_000 + seed)
        simulator = WeightedParticleSimulator(n_steps=n_steps)
        tau_grid = torch.linspace(0.0, 1.0, n_steps + 1)

        torch.manual_seed(seed)
        default = simulator.rollout(z0=z0, logw0=logw0, model=model, log_m0=log_m0)
        torch.manual_seed(seed)
        explicit = simulator.rollout(z0=z0, logw0=logw0, model=model, log_m0=log_m0, tau_grid=tau_grid)

        assert torch.allclose(explicit.tau_steps, default.tau_steps)
        assert torch.allclose(explicit.z_steps, default.z_steps, rtol=1e-6, atol=1e-7)
        assert torch.allclose(explicit.logw_steps, default.logw_steps, rtol=1e-6, atol=1e-7)


def test_random_count_helpers_match_manual_formula() -> None:
    for seed in range(120):
        generator = torch.Generator().manual_seed(seed)
        n_steps = 1 + seed % 9
        G = 2 + seed % 6
        N = 3 + seed % 7
        S = 1 + seed % 5
        growth_steps = torch.randn(n_steps, G, N, generator=generator)
        logw_steps = torch.randn(n_steps + 1, G, N, generator=generator) * 0.2
        tau_raw = torch.rand(n_steps + 1, generator=generator).sort().values
        tau_steps = (tau_raw - tau_raw[0]) / (tau_raw[-1] - tau_raw[0])

        curve = integrated_fitness_curve(growth_steps, logw_steps, tau_steps)

        logw_norm = logw_steps[:-1] - torch.logsumexp(logw_steps[:-1], dim=-1, keepdim=True)
        r_bar = (logw_norm.exp() * growth_steps).sum(-1)
        increments = r_bar * (tau_steps[1:] - tau_steps[:-1])[:, None]
        expected_curve = torch.cat(
            [torch.zeros(1, G), torch.cumsum(increments, dim=0)],
            dim=0,
        )
        assert torch.allclose(curve, expected_curve)

        zeta = torch.randn(G, generator=generator)
        count_matrix = torch.ones(S, G)
        if seed % 2 == 0:
            exposures = torch.rand(G, generator=generator) + 0.05
            log_l = torch.log(exposures + 1e-30).unsqueeze(0)
        else:
            exposures = torch.rand(S, G, generator=generator) + 0.05
            log_l = torch.log(exposures + 1e-30)
        pi = count_fractions_from_zeta(zeta, exposures, count_matrix)
        log_unnorm = log_l + zeta.unsqueeze(0)
        expected = (log_unnorm - torch.logsumexp(log_unnorm, dim=1, keepdim=True)).exp()
        if exposures.ndim == 1:
            expected = expected.expand(S, -1)
        assert torch.allclose(pi, expected)


def test_random_multitime_mass_loss_matches_closed_form() -> None:
    for seed in range(35):
        generator = torch.Generator().manual_seed(seed)
        n_atoms = 2 + seed % 6
        latent_dim = 1 + seed % 4
        support = torch.randn(1, n_atoms, latent_dim, generator=generator)
        relative_logw = torch.full((1, n_atoms), -np.log(n_atoms))
        pred_mass = float(0.5 + torch.rand((), generator=generator).item() * 2.0)
        target_mass = float(0.5 + torch.rand((), generator=generator).item() * 2.0)

        rollout = ParticleRollout(
            z_steps=torch.stack([support, support.clone()], dim=0),
            logw_steps=torch.stack([relative_logw, relative_logw.clone()], dim=0),
            tau_steps=torch.tensor([0.0, 1.0]),
            log_m0=torch.tensor([np.log(pred_mass)], dtype=torch.float32),
        )
        target_support = {"terminal": {"pert": support.squeeze(0)}}
        target_logw = {
            "terminal": {
                "pert": torch.full((n_atoms,), np.log(target_mass / n_atoms), dtype=torch.float32)
            }
        }
        loss_fn = MultiTimeEndpointLoss(
            EndpointGeometryMassLoss(eps=0.1, tau=1.0, max_iter=80, use_geomloss=False),
            time_weights={"terminal": 0.7},
        )

        loss, _ = loss_fn(
            rollout,
            checkpoint_indices={"terminal": 1},
            target_support_by_time=target_support,
            target_logw_by_time=target_logw,
            perturbation_ids=["pert"],
        )

        expected = 0.7 * (np.log(pred_mass) - np.log(target_mass)) ** 2
        assert np.isclose(float(loss), expected, rtol=1e-4, atol=1e-5)
