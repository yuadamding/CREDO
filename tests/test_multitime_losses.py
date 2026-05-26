from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

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
    MultiTimeCountLikelihood,
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


class _ConstantComponentLoss(nn.Module):
    def component_dict(self, pred_z, pred_logw_abs, target_support, target_logw, perturbation_ids):
        active = [pid for pid in perturbation_ids if pid in target_support]
        loss = pred_z.new_tensor(float(len(active)))
        components = {
            pid: {
                "geom": pred_z.new_tensor(0.25),
                "mass": pred_z.new_tensor(0.75),
                "total": pred_z.new_tensor(1.0),
            }
            for pid in active
        }
        return loss, components


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
    assert "endpoint/6h" in logs
    assert "endpoint/10h" in logs
    assert int(logs["endpoint/6h/n_active_keys"]) == 1
    assert int(logs["endpoint/10h/n_missing_keys"]) == 0
    assert float(logs["endpoint/6h/geom_mean"]) < 1e-5
    assert float(logs["endpoint/10h/mass_mean"]) < 1e-5


def test_multitime_endpoint_loss_reports_sparse_active_and_missing_keys() -> None:
    support = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[2.0, 0.0], [3.0, 0.0]],
        ]
    )
    logw = torch.full((2, 2), -np.log(2.0))
    rollout = ParticleRollout(
        z_steps=torch.stack([support, support.clone()], dim=0),
        logw_steps=torch.stack([logw, logw.clone()], dim=0),
        tau_steps=torch.tensor([0.0, 1.0]),
        log_m0=torch.tensor([np.log(2.0), np.log(2.0)], dtype=torch.float32),
    )
    target_support = {"10h": {"a": support[0].clone()}}
    target_logw = {"10h": {"a": torch.zeros(2)}}
    loss_fn = MultiTimeEndpointLoss(
        UOTLoss(eps=0.1, max_iter=80, use_geomloss=False),
        reduction="mean",
    )

    loss, logs = loss_fn(
        rollout,
        checkpoint_indices={"10h": 1},
        target_support_by_time=target_support,
        target_logw_by_time=target_logw,
        perturbation_ids=["a", "b"],
    )

    assert float(loss) < 1e-5
    assert int(logs["endpoint/10h/n_active_keys"]) == 1
    assert int(logs["endpoint/10h/n_missing_keys"]) == 1


def test_multitime_endpoint_loss_raises_when_checkpoint_has_no_active_keys() -> None:
    support = torch.zeros(1, 2, 2)
    logw = torch.full((1, 2), -np.log(2.0))
    rollout = ParticleRollout(
        z_steps=torch.stack([support, support.clone()], dim=0),
        logw_steps=torch.stack([logw, logw.clone()], dim=0),
        tau_steps=torch.tensor([0.0, 1.0]),
        log_m0=torch.tensor([np.log(2.0)], dtype=torch.float32),
    )
    loss_fn = MultiTimeEndpointLoss(UOTLoss(eps=0.1, max_iter=80, use_geomloss=False))

    with pytest.raises(ValueError, match="No active target keys"):
        loss_fn(
            rollout,
            checkpoint_indices={"10h": 1},
            target_support_by_time={"10h": {}},
            target_logw_by_time={"10h": {}},
            perturbation_ids=["a"],
        )


def test_multitime_endpoint_loss_can_normalize_time_weights() -> None:
    support = torch.zeros(1, 2, 2)
    logw = torch.full((1, 2), -np.log(2.0))
    rollout = ParticleRollout(
        z_steps=torch.stack([support, support.clone(), support.clone()], dim=0),
        logw_steps=torch.stack([logw, logw.clone(), logw.clone()], dim=0),
        tau_steps=torch.tensor([0.0, 0.5, 1.0]),
        log_m0=torch.tensor([0.0]),
    )
    target_support = {"6h": {"a": support[0]}, "10h": {"a": support[0]}}
    target_logw = {"6h": {"a": torch.zeros(2)}, "10h": {"a": torch.zeros(2)}}
    loss_fn = MultiTimeEndpointLoss(
        _ConstantComponentLoss(),
        time_weights={"6h": 0.5, "10h": 1.0},
        normalize_time_weights=True,
    )

    loss, logs = loss_fn(
        rollout,
        checkpoint_indices={"6h": 1, "10h": 2},
        target_support_by_time=target_support,
        target_logw_by_time=target_logw,
        perturbation_ids=["a"],
    )

    assert torch.allclose(loss, torch.tensor(1.0))
    assert torch.allclose(logs["endpoint/time_weight_normalizer"], torch.tensor(1.5))


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


def test_count_fractions_reject_bad_exposures() -> None:
    with pytest.raises(ValueError, match="exposures must be positive and finite"):
        count_fractions_from_zeta(
            torch.tensor([0.0, 0.0]),
            torch.tensor([1.0, 0.0]),
            torch.ones(1, 2),
        )


def test_count_fractions_reject_bad_count_matrix() -> None:
    with pytest.raises(ValueError, match="count_matrix must be 2D"):
        count_fractions_from_zeta(torch.zeros(2), torch.ones(2), torch.ones(2))
    with pytest.raises(ValueError, match="count_matrix must be nonnegative and finite"):
        count_fractions_from_zeta(torch.zeros(2), torch.ones(2), torch.tensor([[1.0, -1.0]]))
    with pytest.raises(ValueError, match="perturbation dimension must match zeta"):
        count_fractions_from_zeta(torch.zeros(2), torch.ones(2), torch.ones(1, 3))


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


def test_dirichlet_multinomial_validates_pi_rows() -> None:
    lik = DirichletMultinomialLikelihood()
    with pytest.raises(ValueError, match="pi rows must sum to 1"):
        lik(
            counts=torch.tensor([[2.0, 1.0]]),
            pi=torch.tensor([[0.4, 0.4]]),
            n_total=torch.tensor([3.0]),
        )


def test_dirichlet_multinomial_rejects_fractional_counts_and_zero_pi() -> None:
    lik = DirichletMultinomialLikelihood()
    with pytest.raises(ValueError, match="integer-like counts"):
        lik(
            counts=torch.tensor([[2.5, 0.5]]),
            pi=torch.tensor([[0.5, 0.5]]),
            n_total=torch.tensor([3.0]),
        )
    with pytest.raises(ValueError, match="strictly positive"):
        lik(
            counts=torch.tensor([[2.0, 1.0]]),
            pi=torch.tensor([[1.0, 0.0]]),
            n_total=torch.tensor([3.0]),
        )


def test_dirichlet_multinomial_accepts_integer_counts_and_totals() -> None:
    lik = DirichletMultinomialLikelihood()
    loss = lik(
        counts=torch.tensor([[2, 1]], dtype=torch.int64),
        pi=torch.tensor([[0.5, 0.5]], dtype=torch.float64),
        n_total=torch.tensor([3], dtype=torch.int64),
    )
    assert torch.isfinite(loss)


def test_multitime_count_likelihood_raises_on_missing_checkpoint() -> None:
    likelihood = MultiTimeCountLikelihood()
    growth_steps = torch.zeros(1, 2, 3)
    logw_steps = torch.zeros(2, 2, 3)
    tau_steps = torch.tensor([0.0, 1.0])

    with pytest.raises(KeyError, match="Missing checkpoint index"):
        likelihood(
            growth_steps=growth_steps,
            logw_steps=logw_steps,
            tau_steps=tau_steps,
            exposures=torch.ones(2),
            count_matrices={"6h": torch.ones(1, 2)},
            n_totals={"6h": torch.tensor([2.0])},
            checkpoint_indices={"10h": 1},
        )


def test_multitime_count_likelihood_returns_per_time_logs() -> None:
    likelihood = MultiTimeCountLikelihood()
    growth_steps = torch.zeros(2, 2, 3)
    logw_steps = torch.zeros(3, 2, 3)
    tau_steps = torch.tensor([0.0, 0.5, 1.0])

    loss, logs = likelihood.forward_with_logs(
        growth_steps=growth_steps,
        logw_steps=logw_steps,
        tau_steps=tau_steps,
        exposures=torch.ones(2),
        count_matrices={
            "6h": torch.tensor([[1.0, 1.0]]),
            "10h": torch.tensor([[2.0, 2.0]]),
        },
        n_totals={
            "6h": torch.tensor([2.0]),
            "10h": torch.tensor([4.0]),
        },
        checkpoint_indices={"6h": 1, "10h": 2},
    )

    assert torch.isfinite(loss)
    assert set(logs) >= {
        "counts/6h",
        "counts/10h",
        "counts/6h/n_samples",
        "counts/10h/n_perturbations",
        "counts/6h/n_total_sum",
    }


def test_multitime_count_likelihood_rejects_fractional_counts() -> None:
    likelihood = MultiTimeCountLikelihood()
    growth_steps = torch.zeros(1, 2, 3)
    logw_steps = torch.zeros(2, 2, 3)
    tau_steps = torch.tensor([0.0, 1.0])

    with pytest.raises(ValueError, match="integer-like counts"):
        likelihood(
            growth_steps=growth_steps,
            logw_steps=logw_steps,
            tau_steps=tau_steps,
            exposures=torch.ones(2),
            count_matrices={"10h": torch.tensor([[1.5, 0.5]])},
            n_totals={"10h": torch.tensor([2.0])},
            checkpoint_indices={"10h": 1},
        )
