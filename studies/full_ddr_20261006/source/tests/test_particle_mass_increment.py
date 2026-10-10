"""Finite-step state/mass coherence; all fixtures are tiny synthetic tensors."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from credo.contracts import Axis, FiniteMeasure
from credo.recipes.compact_sde_v3.model import CREDOModel
from credo.recipes.compact_sde_v3.objective import (
    CountBlock,
    catalog_count_block_loss,
    checkpoint_geometry_mass_loss,
    count_block_loss,
    integrated_fitness_curve,
)
from credo.recipes.compact_sde_v3.particles import (
    DynamicsStep,
    ParticleState,
    euler_maruyama_rollout,
)
from credo.recipes.compact_sde_v3.recipe import CompactV3Config, recipe
from credo.recipes.compact_sde_v3.training import CatalogBank


def make_rollout(rates: torch.Tensor, *, shift: float = 0.0, steps: int = 1):
    groups, particles = rates.shape
    state = ParticleState(
        z=rates.new_zeros(groups, particles, 1),
        logw=rates.new_full((groups, particles), -np.log(particles) + shift),
        log_m0=rates.new_tensor([2.0, 3.0][:groups]).log(),
        measure_ids=tuple(f"guide-{i}" for i in range(groups)),
        embedding_ids=tuple("control" if i == 0 else "target" for i in range(groups)),
        context_group_ids=("synthetic-capture",) * groups,
        measure_indices=torch.arange(groups),
        residual_scale=rates.new_ones(groups),
    )

    class Kernel:
        def step(self, **kwargs):
            z = kwargs["z"]
            return DynamicsStep(
                drift=torch.zeros_like(z),
                sigma_diag=torch.zeros_like(z),
                growth=rates,
                context=rates.new_zeros(groups, 2),
            )

    grid = torch.linspace(0.0, 1.0, steps + 1, dtype=rates.dtype)
    return euler_maruyama_rollout(
        Kernel(), state, grid, noise=rates.new_zeros(steps, groups, particles, 1)
    )


def make_data():
    ids = ("guide-0", "guide-1")
    block = CountBlock(
        "synthetic-capture", "late", np.arange(2), np.array([2.0, 3.0]), np.array([4, 6])
    )
    return SimpleNamespace(
        axis=Axis(kind="physical", labels=("early", "late"), values=(8.0, 48.0), source="early"),
        measure_ids=ids,
        measure_meta=pd.DataFrame(
            {"measure_id": ids, "context_group_id": ["synthetic-capture"] * 2}
        ),
        context_backgrounds=(),
        count_blocks=(block,),
        measures={
            "late": {
                key: FiniteMeasure(np.zeros((2, 1)), np.ones(2) * mass / 2, mass)
                for key, mass in zip(ids, (2.0, 3.0), strict=True)
            }
        },
    )


@pytest.mark.parametrize("steps", [1, 4])
def test_constant_growth_matches_legacy_and_analytic_mass(steps):
    rates = torch.tensor([[0.7, 0.7], [-0.2, -0.2]], dtype=torch.float64)
    roll = make_rollout(rates, steps=steps)
    exact = integrated_fitness_curve(roll, mass_increment="particle_log_mass")
    legacy = integrated_fitness_curve(roll)
    torch.testing.assert_close(exact, legacy, rtol=1e-7, atol=1e-7)
    torch.testing.assert_close(exact[-1], rates[:, 0])
    torch.testing.assert_close(exact[0], torch.zeros(2, dtype=rates.dtype))


def test_state_dependent_finite_step_has_expected_jensen_gap():
    rates = torch.tensor([[-1.0, 1.0]], dtype=torch.float64)
    roll = make_rollout(rates)
    exact = integrated_fitness_curve(roll, mass_increment="particle_log_mass")[-1]
    legacy = integrated_fitness_curve(roll)[-1]
    torch.testing.assert_close(exact, torch.cosh(torch.ones(1, dtype=rates.dtype)).log())
    torch.testing.assert_close(legacy, torch.zeros_like(legacy))
    assert exact.item() > legacy.item()


def test_exact_mass_is_constant_log_weight_shift_invariant():
    rates = torch.tensor([[-0.7, 0.9], [0.4, 1.3]], dtype=torch.float64)
    roll = make_rollout(rates, steps=3)
    shifted = replace(roll, logw_steps=roll.logw_steps + 731.25)
    first = integrated_fitness_curve(roll, mass_increment="particle_log_mass")
    second = integrated_fitness_curve(shifted, mass_increment="particle_log_mass")
    torch.testing.assert_close(first, second, rtol=1e-11, atol=1e-11)


def test_exact_mass_retains_double_precision_growth_gradients():
    rates = torch.tensor([[-0.5, 0.8], [0.3, -0.2]], dtype=torch.float64, requires_grad=True)

    def objective(value):
        return integrated_fitness_curve(
            make_rollout(value, steps=2), mass_increment="particle_log_mass"
        )[-1]

    assert torch.autograd.gradcheck(objective, (rates,))
    output = objective(rates)
    assert output.dtype is torch.float64
    output.sum().backward()
    torch.testing.assert_close(rates.grad, torch.softmax(rates.detach(), dim=-1))


def test_unknown_mass_semantics_fail_closed():
    roll = make_rollout(torch.zeros(1, 2))
    with pytest.raises(ValueError, match="Unknown mass_increment"):
        integrated_fitness_curve(roll, mass_increment="silent_new_method")


def test_config_preserves_default_and_records_exact_semantics():
    assert CompactV3Config().loss.mass_increment == "integrated_mean_growth"
    configured = CompactV3Config.model_validate({"loss": {"mass_increment": "particle_log_mass"}})
    objectives = recipe.build_objectives(make_data(), configured)
    count = next(value for value in objectives if value.name == "grouped_count_likelihood")
    assert count.config["mass_increment"] == "particle_log_mass"
    assert configured.model_dump()["loss"]["mass_increment"] == "particle_log_mass"


def test_count_loss_bank_and_published_mass_use_same_particles():
    data = make_data()
    rates = torch.tensor([[-1.0, 1.0], [0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    roll = make_rollout(rates)
    # The existing publication path obtains predicted_log_mass from actual
    # rollout weights.  The new count path must agree with that same measure.
    endpoint = checkpoint_geometry_mass_loss(
        roll, data, mass_weight=1.0, include_mass=True,
        validation_source="synthetic", compute_geometry=False,
    )
    published = torch.tensor([row["predicted_log_mass"] for row in endpoint.rows])
    probability = torch.softmax(published, dim=0)
    expected = -(torch.tensor([0.4, 0.6]) * probability.log()).sum()
    loss = count_block_loss(
        roll, data, log_concentration=torch.tensor(0.0),
        count_likelihood="categorical_ce", mass_increment="particle_log_mass",
    )
    torch.testing.assert_close(loss, expected.to(loss), rtol=1e-6, atol=1e-6)
    loss.backward()
    assert rates.grad is not None and torch.isfinite(rates.grad).all()
    assert rates.grad.abs().sum() > 0

    model = CREDOModel(
        embedding_ids=("control", "target"), control_embedding_ids=("control",),
        latent_dim=1, embedding_dim=2, n_programs=2, hidden_dim=8, context_mode="none",
    ).double()
    bank = CatalogBank.empty(
        data, model, 1, device=torch.device("cpu"), dtype=torch.float64,
        mass_increment="particle_log_mass",
    )
    bank.update_from_rollout(roll, model, data, full_refresh=True)
    torch.testing.assert_close(
        bank.full_fitness(time_label="late"),
        integrated_fitness_curve(roll, mass_increment="particle_log_mass")[-1].detach(),
    )
    bank_loss, block_count = catalog_count_block_loss(
        data, log_concentration=torch.tensor(0.0), fitness_bank=bank,
        count_likelihood="categorical_ce", mass_increment="particle_log_mass",
    )
    assert block_count == 1
    torch.testing.assert_close(bank_loss, loss.detach())
    with pytest.raises(ValueError, match="mass increments"):
        catalog_count_block_loss(
            data, log_concentration=torch.tensor(0.0), fitness_bank=bank,
            count_likelihood="categorical_ce", mass_increment="integrated_mean_growth",
        )
