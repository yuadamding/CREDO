"""Unit tests for synthetic benchmark simulations."""
import numpy as np
import pytest

from cape.benchmarks.simulation import (
    build_drift_diffusion_reaction_benchmark, DriftDiffusionReactionConfig,
    build_meanfield_ecology_benchmark, MeanFieldEcologyConfig,
)
from cape.data.filters import filter_state_supported_perturbations


def test_ddr_benchmark_builds():
    cfg = DriftDiffusionReactionConfig(
        n_gene_perturbations=4, n_controls=2, latent_dim=4,
        n_particles_gt=64, n_cells_per_group=30, n_steps_gt=10, seed=0,
    )
    data, truth = build_drift_diffusion_reaction_benchmark(cfg)
    assert data.latent_dim == 4
    assert len(data.catalog.perturbation_ids) == 6
    assert len(data.catalog.control_ids) == 2


def test_ddr_mass_varies():
    """Control growth=0 keeps mass at 1000; gene perturbations should vary."""
    cfg = DriftDiffusionReactionConfig(
        n_gene_perturbations=6, n_controls=2, latent_dim=4,
        n_particles_gt=256, n_cells_per_group=50, n_steps_gt=50, seed=42,
    )
    data, truth = build_drift_diffusion_reaction_benchmark(cfg)
    masses_p60 = {
        pid: data.mass_table.get_pooled(pid, "P60")
        for pid in data.catalog.perturbation_ids
    }
    ctrl_mass = np.mean([masses_p60[c] for c in data.catalog.control_ids])
    gene_masses = [masses_p60[p] for p in data.catalog.non_control_ids]
    # Some genes should have mass different from controls
    deviations = [abs(m - ctrl_mass) / ctrl_mass for m in gene_masses]
    assert max(deviations) > 0.05, "Some perturbations should alter mass"


def test_ddr_filter():
    cfg = DriftDiffusionReactionConfig(
        n_gene_perturbations=4, n_controls=2, latent_dim=4,
        n_particles_gt=64, n_cells_per_group=30, n_steps_gt=10,
    )
    data, _ = build_drift_diffusion_reaction_benchmark(cfg)
    supported = filter_state_supported_perturbations(data, min_cells_p4=20, min_cells_p60=20)
    assert len(supported) == 6  # all have 30 cells


def test_ddr_endpoint_problem():
    cfg = DriftDiffusionReactionConfig(
        n_gene_perturbations=4, n_controls=2, latent_dim=4,
        n_particles_gt=64, n_cells_per_group=30, n_steps_gt=10,
    )
    data, _ = build_drift_diffusion_reaction_benchmark(cfg)
    ep = data.to_endpoint_problem()
    for pid in data.catalog.perturbation_ids:
        assert pid in ep.initial
        assert ep.initial[pid].total_mass > 0
        assert ep.terminal[pid].total_mass > 0


def test_meanfield_ecology_builds():
    cfg = MeanFieldEcologyConfig(
        n_gene_perturbations=4, n_controls=2, latent_dim=4, n_programs=4,
        n_particles_gt=64, n_cells_per_group=30, n_steps_gt=10, seed=0,
    )
    data, truth = build_meanfield_ecology_benchmark(cfg)
    assert data.latent_dim == 4
    assert "context_trajectory" in truth
    assert truth["context_trajectory"] is not None


def test_simulation_reproducible():
    """Same seed must give identical results."""
    cfg = DriftDiffusionReactionConfig(
        n_gene_perturbations=3, n_controls=1, latent_dim=4,
        n_particles_gt=64, n_cells_per_group=20, n_steps_gt=10, seed=7,
    )
    d1, _ = build_drift_diffusion_reaction_benchmark(cfg)
    d2, _ = build_drift_diffusion_reaction_benchmark(cfg)
    mass1 = d1.mass_table.df.sort_values(["perturbation_id", "time_label"])["mass"].values
    mass2 = d2.mass_table.df.sort_values(["perturbation_id", "time_label"])["mass"].values
    np.testing.assert_array_equal(mass1, mass2)
