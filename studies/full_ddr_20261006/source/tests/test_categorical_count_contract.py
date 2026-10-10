"""Observed-scale categorical counts: full denominator, not replicate inference."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from credo.data import open_study
from credo.recipes.compact_sde_v3.objective import (
    CountBlock,
    _count_block_nll,
    catalog_count_block_loss,
)
from credo.recipes.compact_sde_v3.recipe import CompactLossConfig
from credo.recipes.compact_sde_v3.training import Trainer
from credo.registry import get_recipe
from credo.runtime import TrainingEngine


def _block(multiplier: int = 1) -> CountBlock:
    return CountBlock(
        context_group_id="D",
        time_label="late",
        measure_indices=[0, 1],
        exposure=[2.0, 1.0],
        counts=[2 * multiplier, 0],
        background_series_ids=("missing_state",),
        background_fitness=[0.0],
        background_exposure=[7.0],
        background_counts=[8 * multiplier],
        conditioning_policy="preserve_background",
    )


def test_ce_uses_complete_catalog_and_preserves_observed_zeros() -> None:
    block = _block()
    value = _count_block_nll(
        block, torch.zeros(2), torch.tensor(5.0), count_likelihood="categorical_ce"
    )
    expected = -0.2 * torch.log(torch.tensor(0.2)) - 0.8 * torch.log(torch.tensor(0.7))
    torch.testing.assert_close(value, expected)
    # Dropping the background would give -log(2/3), a different estimand.
    assert not torch.isclose(value, -torch.log(torch.tensor(2 / 3)))
    assert block.counts.tolist() == [2.0, 0.0]
    assert block.background_counts.tolist() == [8.0]


def test_ce_count_replication_does_not_change_objective_scale() -> None:
    fitness = torch.tensor([0.3, -0.8], requires_grad=True)
    values = [
        _count_block_nll(
            _block(k), fitness, torch.tensor(5.0), count_likelihood="categorical_ce"
        )
        for k in (1, 10_000)
    ]
    torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)


def test_ce_has_weight_field_gradient_but_no_concentration_gradient() -> None:
    fitness = torch.tensor([0.3, -0.8], requires_grad=True)
    concentration = torch.tensor(5.0, requires_grad=True)
    value = _count_block_nll(
        _block(), fitness, concentration, count_likelihood="categorical_ce"
    )
    value.backward()
    assert fitness.grad is not None and torch.isfinite(fitness.grad).all()
    assert torch.any(fitness.grad != 0)
    assert concentration.grad is None


def test_ce_stays_finite_when_a_predicted_probability_underflows() -> None:
    fitness = torch.tensor([-1000.0, 1000.0], requires_grad=True)
    value = _count_block_nll(
        _block(), fitness, torch.tensor(5.0), count_likelihood="categorical_ce"
    )
    value.backward()
    assert torch.isfinite(value)
    assert fitness.grad is not None and torch.isfinite(fitness.grad).all()


def test_count_contract_rejects_unobserved_aggregate_and_unknown_likelihood() -> None:
    with pytest.raises(ValueError, match="at least one observed count"):
        CountBlock("D", "late", [0], [1.0], [0])
    with pytest.raises(ValueError, match="Unknown count likelihood"):
        _count_block_nll(_block(), torch.zeros(2), torch.tensor(5.0), count_likelihood="x")


def test_catalog_ce_is_equal_mean_of_complete_block_scores(tiny_data) -> None:
    class Bank:
        mass_increment = "particle_log_mass"

        def full_fitness(self, *, time_label):
            return torch.linspace(-0.2, 0.2, len(tiny_data.measure_ids))

    bank = Bank()
    concentration = torch.tensor(5.0, requires_grad=True)
    actual, count = catalog_count_block_loss(
        tiny_data,
        log_concentration=concentration,
        fitness_bank=bank,
        count_likelihood="categorical_ce",
        mass_increment="particle_log_mass",
    )
    expected = torch.stack(
        [
            _count_block_nll(
                block,
                bank.full_fitness(time_label=block.time_label)[block.measure_indices],
                concentration,
                count_likelihood="categorical_ce",
            )
            for block in tiny_data.count_blocks
        ]
    ).mean()
    assert count == len(tiny_data.count_blocks)
    torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError, match="different mass increments"):
        catalog_count_block_loss(
            tiny_data, log_concentration=concentration, fitness_bank=bank
        )
    # Subsetting contexts is allowed; it must not subset a catalog denominator.
    context = tiny_data.count_blocks[0].context_group_id
    selected = replace(
        tiny_data,
        count_blocks=tuple(b for b in tiny_data.count_blocks if b.context_group_id == context),
    )
    restricted, n_restricted = catalog_count_block_loss(
        selected,
        log_concentration=concentration,
        fitness_bank=bank,
        count_likelihood="categorical_ce",
        mass_increment="particle_log_mass",
    )
    filtered, n_filtered = catalog_count_block_loss(
        tiny_data,
        log_concentration=concentration,
        fitness_bank=bank,
        context_group_ids=[context],
        count_likelihood="categorical_ce",
        mass_increment="particle_log_mass",
    )
    assert n_restricted == n_filtered
    torch.testing.assert_close(restricted, filtered)


def test_default_count_model_remains_dirichlet_multinomial() -> None:
    settings = CompactLossConfig()
    assert settings.count_likelihood == "dirichlet_multinomial"
    assert settings.mass_increment == "integrated_mean_growth"
    concentration = torch.tensor(5.0, requires_grad=True)
    value = _count_block_nll(_block(), torch.zeros(2), concentration)
    value.backward()
    assert concentration.grad is not None and torch.isfinite(concentration.grad)


def test_ce_training_excludes_concentration_from_optimizer(tiny_config, monkeypatch) -> None:
    settings = tiny_config.recipe_config
    loss = settings.loss.model_copy(
        update={"count_likelihood": "categorical_ce", "mass_increment": "particle_log_mass"}
    )
    epochs = settings.training.epochs.model_copy(update={"state": 0, "mass": 1, "context": 0})
    training = settings.training.model_copy(update={"epochs": epochs})
    model = settings.model.model_copy(update={"context": "none"})
    settings = settings.model_copy(update={"loss": loss, "training": training, "model": model})
    config = tiny_config.model_copy(update={"recipe_config": settings})
    original_optimizer = Trainer._optimizer
    seen = []

    def inspect_optimizer(self, spec, parameters):
        seen.append(True)
        assert not self.log_count_concentration.requires_grad
        assert all(parameter is not self.log_count_concentration for parameter in parameters)
        return original_optimizer(self, spec, parameters)

    monkeypatch.setattr(Trainer, "_optimizer", inspect_optimizer)
    study = open_study(config)
    try:
        trainer = TrainingEngine().fit(
            get_recipe(config.recipe), study.view(), config, device="cpu"
        )
        assert seen
        assert not trainer.log_count_concentration.requires_grad
        torch.testing.assert_close(trainer.log_count_concentration, torch.tensor(100.0).log())
        descriptor = trainer.objective_map["grouped_count_likelihood"]
        assert descriptor.config["count_likelihood"] == "categorical_ce"
        assert descriptor.config["mass_increment"] == "particle_log_mass"
    finally:
        study.close()
