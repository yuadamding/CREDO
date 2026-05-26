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
)
from credo.losses.counts import (
    DirichletMultinomialLikelihood,
    count_fractions_from_zeta,
    integrated_fitness,
    integrated_fitness_curve,
)
from credo.losses.multitime import (
    MultiTimeEndpointLoss,
    build_target_tensors_by_time,
    checkpoint_indices_for_taus,
    make_observed_tau_grid,
)
from credo.losses.uot import UOTLoss
from credo.models.weighted_sde import ParticleRollout


def _trajectory_problem():
    labels = ["90m", "6h", "10h"]
    rows = []
    latent = []
    for time_i, label in enumerate(labels):
        for cell_i in range(4):
            rows.append(
                {
                    "cell_id": f"cell_{label}_{cell_i}",
                    "perturbation_id": "LPS__mono",
                    "time_label": label,
                    "sample_id": "D1",
                }
            )
            latent.append([float(time_i), float(cell_i), 0.0])
    cell_df = pd.DataFrame(rows)
    mass_df = (
        cell_df.groupby(["perturbation_id", "time_label", "sample_id"], observed=True)
        .size()
        .rename("mass")
        .reset_index()
    )
    data = PerturbSeqDynamicsData(
        time_axis=TimeAxis(labels=labels, physical_times=[1.5, 6.0, 10.0]),
        catalog=PerturbationCatalog(["LPS__mono"], ["LPS__mono"]),
        cell_state=CellStateTable(cell_df, np.asarray(latent, dtype=np.float32)),
        mass_table=MassTable(mass_df),
    )
    return data.to_trajectory_problem()


def test_tau_grid_contains_observed_times() -> None:
    observed = [0.0, (6.0 - 1.5) / (10.0 - 1.5), 1.0]
    tau_grid = make_observed_tau_grid(observed, steps_per_interval=4)
    indices = checkpoint_indices_for_taus(tau_grid, ["90m", "6h", "10h"], observed)

    assert indices == {"90m": 0, "6h": 4, "10h": 8}
    assert torch.isclose(tau_grid[indices["6h"]], torch.tensor(observed[1]))


def test_checkpoint_indices_require_1d_tau_steps() -> None:
    with pytest.raises(ValueError, match="tau_steps must be a 1D tensor"):
        checkpoint_indices_for_taus(torch.zeros(2, 2), ["a"], [0.0])


def test_multitime_endpoint_loss_zero_when_targets_equal_predictions() -> None:
    support = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    logw = torch.full((1, 3), -np.log(3.0))
    rollout = ParticleRollout(
        z_steps=torch.stack([support, support.clone(), support.clone()], dim=0),
        logw_steps=torch.stack([logw, logw.clone(), logw.clone()], dim=0),
        tau_steps=torch.tensor([0.0, 0.5, 1.0]),
        log_m0=torch.tensor([np.log(3.0)], dtype=torch.float32),
    )
    target = FiniteMeasure(
        support=support.squeeze(0).numpy(),
        weights=np.ones(3, dtype=np.float32),
        total_mass=3.0,
    )
    target_support = {
        "6h": {"LPS__mono": torch.tensor(target.support)},
        "10h": {"LPS__mono": torch.tensor(target.support)},
    }
    target_logw = {
        "6h": {"LPS__mono": torch.log(torch.tensor(target.weights) + 1e-30)},
        "10h": {"LPS__mono": torch.log(torch.tensor(target.weights) + 1e-30)},
    }
    loss_fn = MultiTimeEndpointLoss(UOTLoss(eps=0.1, max_iter=80, use_geomloss=False))

    loss, logs = loss_fn(
        rollout,
        checkpoint_indices={"6h": 1, "10h": 2},
        target_support_by_time=target_support,
        target_logw_by_time=target_logw,
        perturbation_ids=["LPS__mono"],
    )

    assert float(loss) < 1e-5
    assert set(logs) == {"endpoint/6h", "endpoint/10h"}


def test_build_target_tensors_by_time_accepts_trajectory_problem() -> None:
    trajectory = _trajectory_problem()
    target_support, target_logw = build_target_tensors_by_time(
        trajectory,
        time_labels=["6h", "10h"],
        perturbation_ids=["LPS__mono"],
    )

    assert set(target_support) == {"6h", "10h"}
    assert target_support["6h"]["LPS__mono"].shape == (4, 3)
    assert target_logw["10h"]["LPS__mono"].shape == (4,)


def test_build_target_tensors_accepts_sample_aware_keys() -> None:
    labels = ["90m", "6h"]
    cell_df = pd.DataFrame(
        [
            {
                "cell_id": f"cell_{label}_{idx}",
                "perturbation_id": "LPS__mono",
                "time_label": label,
                "sample_id": "D1",
            }
            for label in labels
            for idx in range(2)
        ]
    )
    data = PerturbSeqDynamicsData(
        time_axis=TimeAxis(labels=labels, physical_times=[1.5, 6.0]),
        catalog=PerturbationCatalog(["LPS__mono"], ["LPS__mono"]),
        cell_state=CellStateTable(cell_df, np.zeros((4, 2), dtype=np.float32)),
        mass_table=MassTable(
            cell_df.groupby(["perturbation_id", "time_label", "sample_id"], observed=True)
            .size()
            .rename("mass")
            .reset_index()
        ),
    )
    trajectory = data.to_trajectory_problem(by_sample=True)

    target_support, target_logw = build_target_tensors_by_time(trajectory)

    key = ("D1", "LPS__mono")
    assert target_support["90m"][key].shape == (2, 2)
    assert target_logw["6h"][key].shape == (2,)


def test_build_target_tensors_rejects_empty_and_unknown_labels() -> None:
    trajectory = _trajectory_problem()

    with pytest.raises(ValueError, match="at least one label"):
        build_target_tensors_by_time(trajectory, time_labels=[])
    with pytest.raises(KeyError, match="Unknown trajectory time labels"):
        build_target_tensors_by_time(trajectory, time_labels=["missing"])


def test_integrated_fitness_curve_matches_final_old_count_likelihood() -> None:
    torch.manual_seed(1)
    growth_steps = torch.randn(5, 2, 4)
    logw_steps = torch.randn(6, 2, 4)
    tau_steps = torch.linspace(0.0, 1.0, 6)

    curve = integrated_fitness_curve(growth_steps, logw_steps, tau_steps)
    final = integrated_fitness(growth_steps, logw_steps, tau_steps)

    assert curve.shape == (6, 2)
    assert torch.allclose(curve[-1], final)


def test_integrated_fitness_uses_variable_dtau() -> None:
    growth_steps = torch.tensor([[[1.0, 3.0]], [[2.0, 4.0]], [[5.0, 7.0]]])
    logw_steps = torch.zeros(4, 1, 2)
    tau_steps = torch.tensor([0.0, 0.2, 0.7, 1.0])

    zeta = integrated_fitness(growth_steps, logw_steps, tau_steps)

    r_bar = growth_steps.mean(-1)
    expected = (r_bar * (tau_steps[1:] - tau_steps[:-1])[:, None]).sum(0)
    assert torch.equal(zeta, expected)


def test_count_fractions_match_legacy_formula() -> None:
    zeta = torch.tensor([0.2, -0.1, 0.4])
    exposures = torch.tensor([[0.2, 0.3, 0.5], [0.5, 0.25, 0.25]])
    count_matrix = torch.ones(2, 3)

    pi = count_fractions_from_zeta(zeta, exposures, count_matrix)

    log_l = torch.log(exposures + 1e-30)
    log_unnorm = log_l + zeta.unsqueeze(0)
    log_pi = log_unnorm - torch.logsumexp(log_unnorm, dim=1, keepdim=True)
    expected = log_pi.exp()
    assert torch.equal(pi, expected)


def test_integrated_fitness_curve_uses_variable_dtau() -> None:
    growth_steps = torch.full((3, 1, 2), 2.0)
    logw_steps = torch.zeros(4, 1, 2)
    tau_steps = torch.tensor([0.0, 0.2, 0.7, 1.0])

    curve = integrated_fitness_curve(growth_steps, logw_steps, tau_steps)

    assert torch.allclose(curve.squeeze(1), torch.tensor([0.0, 0.4, 1.4, 2.0]))


def test_dirichlet_multinomial_includes_count_constant() -> None:
    lik = DirichletMultinomialLikelihood(log_phi=float(np.log(5.0)))
    counts = torch.tensor([[2.0, 1.0]])
    pi = torch.tensor([[0.25, 0.75]])
    n_total = torch.tensor([3.0])

    loss = lik(counts, pi, n_total)

    phi = torch.tensor(5.0)
    alpha = phi * pi
    expected_ll = (
        torch.lgamma(n_total + 1.0)
        - torch.lgamma(counts + 1.0).sum(-1)
        + torch.lgamma(phi)
        - torch.lgamma(phi + n_total)
        + (torch.lgamma(alpha + counts) - torch.lgamma(alpha)).sum(-1)
    )
    assert torch.allclose(loss, -expected_ll.sum())


def test_dirichlet_multinomial_validates_totals() -> None:
    lik = DirichletMultinomialLikelihood()
    with pytest.raises(ValueError, match="n_total must equal"):
        lik(
            counts=torch.tensor([[2.0, 1.0]]),
            pi=torch.tensor([[0.5, 0.5]]),
            n_total=torch.tensor([4.0]),
        )
