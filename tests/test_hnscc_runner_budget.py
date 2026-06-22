from __future__ import annotations

import argparse

import pytest

from runners.run_credo_hnscc_full import calibrate_train_budget


pytestmark = pytest.mark.unit


def _budget_args(**overrides):
    values = {
        "budget_headroom": 0.70,
        "n_particles": 96,
        "n_test_functions": 12,
        "n_steps": 24,
        "max_active_perturbations": 16,
        "max_train_target_atoms": 768,
        "auto_scale_budget": True,
        "latent_source": "expression",
        "context_kind": "causal_attention",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_raw_expression_causal_budget_scales_stage_d_oom_settings() -> None:
    budget = calibrate_train_budget(_budget_args(), 150, latent_dim=1024)

    assert budget["budget_scaled"] is True
    assert budget["effective_n_particles"] < 96
    assert budget["effective_n_steps"] < 24
    assert budget["effective_max_active_perturbations"] < 16
    assert budget["effective_max_train_target_atoms"] < 768
    assert budget["requested_graph_units"] > budget["target_graph_units"]


def test_raw_expression_causal_budget_keeps_h100_safe_defaults() -> None:
    budget = calibrate_train_budget(
        _budget_args(
            n_particles=64,
            n_steps=16,
            max_active_perturbations=8,
            max_train_target_atoms=384,
        ),
        150,
        latent_dim=1024,
    )

    assert budget["budget_scaled"] is False
    assert budget["effective_n_particles"] == 64
    assert budget["effective_n_steps"] == 16
    assert budget["effective_max_active_perturbations"] == 8
    assert budget["effective_max_train_target_atoms"] == 384
