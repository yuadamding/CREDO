"""Fixed-truth positive and null count simulations for Dev40 qualification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import ContractError


@dataclass(frozen=True)
class SimulatedProgramData:
    """Raw counts, design labels, and exact latent truth."""

    counts: np.ndarray
    state: np.ndarray
    donor_index: np.ndarray
    sample_index: np.ndarray
    checkpoint_index: np.ndarray
    target_index: np.ndarray
    guide_index: np.ndarray
    guide_to_target: np.ndarray
    control_guide_indices: tuple[int, ...]
    target_descriptors: np.ndarray
    true_loadings: np.ndarray
    true_target_activity: np.ndarray
    true_gene_effect: np.ndarray


def simulate_program_counts(
    *,
    cells: int = 2400,
    genes: int = 120,
    programs: int = 5,
    donors: int = 4,
    checkpoints: int = 3,
    perturbation_targets: int = 5,
    guides_per_target: int = 2,
    controls: int = 2,
    state_dimension: int = 4,
    null: bool = False,
    seed: int = 0,
) -> SimulatedProgramData:
    """Simulate NB2 raw counts from the frozen hierarchical equation."""

    if (
        cells < 100
        or genes < 20
        or programs < 2
        or donors < 3
        or checkpoints < 2
        or perturbation_targets < 3
        or guides_per_target < 2
        or controls < 2
    ):
        raise ContractError("Program simulation dimensions are below qualification minima.")
    rng = np.random.default_rng(seed)
    targets = perturbation_targets + 1  # target zero is the shared control reference.
    guides = controls + perturbation_targets * guides_per_target
    guide_to_target = np.zeros(guides, dtype=np.int64)
    for target in range(1, targets):
        start = controls + (target - 1) * guides_per_target
        guide_to_target[start : start + guides_per_target] = target
    control_guides = tuple(range(controls))

    donor_index = rng.integers(0, donors, size=cells, dtype=np.int64)
    sample_index = donor_index.copy()
    checkpoint_index = rng.integers(0, checkpoints, size=cells, dtype=np.int64)
    guide_index = rng.integers(0, guides, size=cells, dtype=np.int64)
    target_index = guide_to_target[guide_index]
    state = rng.normal(size=(cells, state_dimension)).astype(np.float32)

    descriptors = rng.normal(size=(targets, 4))
    descriptors[0] = 0.0
    time_basis = np.column_stack((np.ones(checkpoints), np.linspace(0.0, 1.0, checkpoints)))
    descriptor_weight = rng.normal(scale=0.45, size=(4, 2, programs))
    target_basis = np.einsum("td,dbk->tbk", descriptors, descriptor_weight)
    target_activity = np.einsum("cb,tbk->tck", time_basis, target_basis)
    target_activity[0] = 0.0
    if null:
        target_activity[:] = 0.0
    guide_deviation_basis = rng.normal(scale=0.08, size=(guides, 2, programs))
    guide_deviation_basis[:controls] = 0.0
    guide_deviation = np.einsum("cb,gbk->gck", time_basis, guide_deviation_basis)
    efficiency = rng.uniform(0.55, 0.95, size=guides)
    efficiency[:controls] = 0.0

    loadings = np.zeros((genes, programs), dtype=np.float64)
    nonzero = max(5, genes // (programs * 3))
    for program in range(programs):
        selected = rng.choice(genes, size=nonzero, replace=False)
        loadings[selected, program] = rng.normal(scale=1.0, size=nonzero)
        loadings[:, program] /= np.linalg.norm(loadings[:, program]) + 1e-12
    if null:
        loadings *= 0.0

    reference_basis = rng.normal(scale=0.12, size=(2, programs))
    reference = time_basis @ reference_basis
    state_program = rng.normal(scale=0.15, size=(state_dimension, programs))
    gate = np.tanh(state @ state_program)
    activity = reference[checkpoint_index] + gate
    is_perturbation = (target_index != 0)[:, None]
    activity += is_perturbation * (
        target_activity[target_index, checkpoint_index] * (1.0 + 0.25 * gate)
        + efficiency[guide_index, None]
        * guide_deviation[guide_index, checkpoint_index]
        * (1.0 + 0.25 * gate)
    )

    base_frequency = rng.dirichlet(np.full(genes, 1.5))
    donor_gene = rng.normal(scale=0.06, size=(donors, genes))
    state_gene = rng.normal(scale=0.04, size=(state_dimension, genes))
    library = rng.integers(800, 3500, size=cells)
    gene_effect = activity @ loadings.T
    log_rate = (
        np.log(base_frequency)[None] + donor_gene[donor_index] + state @ state_gene + gene_effect
    )
    rate = np.exp(np.clip(log_rate, -12.0, 5.0))
    rate /= rate.sum(axis=1, keepdims=True)
    mean = library[:, None] * rate
    dispersion = rng.uniform(4.0, 20.0, size=genes)
    gamma_rate = rng.gamma(shape=dispersion, scale=mean / dispersion)
    counts = rng.poisson(gamma_rate).astype(np.int64)
    # Rare all-zero draws are made valid raw-count observations without changing labels.
    zero = np.where(counts.sum(axis=1) == 0)[0]
    counts[zero, np.argmax(base_frequency)] = 1
    return SimulatedProgramData(
        counts=counts,
        state=state,
        donor_index=donor_index,
        sample_index=sample_index,
        checkpoint_index=checkpoint_index,
        target_index=target_index,
        guide_index=guide_index,
        guide_to_target=guide_to_target,
        control_guide_indices=control_guides,
        target_descriptors=descriptors.astype(np.float32),
        true_loadings=loadings.astype(np.float32),
        true_target_activity=target_activity.astype(np.float32),
        true_gene_effect=gene_effect.astype(np.float32),
    )
