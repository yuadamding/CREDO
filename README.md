# psqdynamics

**Control-anchored, finite-measure dynamics for longitudinal Perturb-seq**

`psqdynamics` is a research software package for modeling **P4 → P60 perturb-seq dynamics** from destructive single-cell snapshots. The package is built for settings where each perturbation is observed as:

- a cell-state distribution at an initial time point,
- a cell-state distribution at a later time point,
- and a perturbation-specific abundance signal that should be modeled jointly with cell-state transport.

The software treats the primary object of interest as a **perturbation-indexed finite measure**, not a matched trajectory dataset. In practice, this means the model learns how a perturbation changes:

1. **where cells move** in latent state space,
2. **how much stochastic dispersion** they exhibit,
3. **how much mass expands or depletes**, and
4. **how population context feeds back** on those dynamics.

The package is designed around a **control-anchored mean-field neural differential equation** formulation with a **single unbalanced optimal transport endpoint loss**.

---

## Why this package exists

Single-cell RNA-seq is destructive. In longitudinal perturbation screens, the same physical cell is not measured twice. As a result, the software does **not** assume tracked single-cell trajectories. Instead, it models the evolution of perturbation-specific populations between observed endpoints.

For a perturbation `g`, the package represents:

- the observed initial population as `mu0^g = M0^g p0^g`,
- the observed terminal population as `nu1^g = M1^g p1^g`,

where `p0^g` and `p1^g` are empirical state distributions in a shared latent space, and `M0^g`, `M1^g` are abundance-scale masses derived from guide-level measurements.

This design is appropriate when terminal geometry and perturbation abundance are two aspects of the **same endpoint object** and should therefore be fitted jointly.

---

## Core modeling ideas

### 1. Finite-measure endpoints
The primary supervised objects are perturbation-indexed finite measures. The package compares predicted and observed terminal populations directly, rather than fitting a normalized distribution and total abundance with separate unrelated losses.

### 2. Control anchoring
Controls are encoded as the exact zero perturbation embedding. This means the same coefficient networks define both the baseline control dynamics and perturbation-induced deviations.

### 3. Three separate biological mechanisms
The dynamical system has three distinct learned components:

- **drift** for directional movement,
- **diffusion** for stochastic spread,
- **growth** for multiplicative mass expansion or depletion.

These mechanisms are parameterized separately because they are biologically distinct and have different identifiability properties.

### 4. Mean-field tissue context
The package optionally conditions dynamics on a low-dimensional summary of the current evolving population. This supports intrinsic-versus-context-mediated analysis and sample-pooled or sample-specific context protocols.

### 5. Backend separation
The software distinguishes between:

- the **law-level model**,
- the **numerical approximation**, and
- the **observation/loss model**.

Particle simulation is one backend for approximating the law-level measure. It is not the scientific definition of the model.

### 6. Unbalanced OT endpoint fitting
The default endpoint discrepancy is an entropic **unbalanced optimal transport** divergence on finite measures. This lets the model jointly fit latent geometry and total mass while tolerating imperfect abundance calibration.

---

## Design principles

`psqdynamics` is built around the following software principles.

### Measure-centric
The central domain object is a finite measure, not an `AnnData` object and not a trajectory tensor.

### Backend-agnostic
Simulation backends can be swapped without changing the scientific interface. The first backend is particle-based; future mean-flow or other one-step backends can share the same output contract.

### Control-anchored by construction
Control behavior is encoded structurally, not only encouraged by regularization.

### Perturbation-first batching
Training batches are organized around perturbations and endpoint measures, not around flat cell minibatches.

### Strict module boundaries
Data loading, dynamics, simulation, OT, training, evaluation, and plotting are separate subsystems.

### Frozen latent spaces first
The initial implementation assumes a shared latent representation has already been computed. Joint representation learning can be added later, but it is not mixed into the first dynamics layer.

---

## Repository layout

```text
src/psqdynamics/
  api/
    model.py
    fit.py
    simulate.py
    counterfactual.py

  contracts/
    measures.py
    perturbations.py
    batches.py
    outputs.py
    schemas.py

  data/
    anndata_adapter.py
    metadata.py
    latent_space.py
    endpoint_builder.py
    mass_builder.py
    perturbation_index.py
    splits.py
    filtering.py

  embeddings/
    control_anchor.py
    guide_embedding.py
    gene_embedding.py
    sgrna_residual.py

  context/
    base.py
    none.py
    pooled.py
    samplewise.py
    interpretable.py

  fields/
    common_input.py
    drift.py
    diffusion.py
    growth.py
    anchored_system.py

  law/
    generator.py
    weak_form.py
    measure_system.py

  simulate/
    backend.py
    particle_backend.py
    integrators.py
    context_update.py
    resampling.py
    meanflow_backend.py

  ot/
    costs.py
    finite_measure.py
    sinkhorn_unbalanced.py
    lowrank.py
    diagnostics.py

  losses/
    endpoint.py
    regularization.py
    objective.py

  training/
    trainer.py
    optimization.py
    checkpointing.py
    callbacks.py

  evaluation/
    metrics.py
    calibration.py
    ablations.py
    attribution.py

  plotting/
    endpoint.py
    masses.py
    context.py
    trajectories.py

  io/
    save.py
    load.py
    export.py

tests/
configs/
examples/
docs/
```

This structure is intentionally centered on **mathematical responsibility** rather than framework-specific conventions.

---

## Main abstractions

### `FiniteMeasure`
A finite nonnegative measure represented by support points and weights in latent space.

### `ObservedEndpoints`
A perturbation-indexed container holding `mu0^g` and `nu1^g` for all modeled perturbations.

### `PerturbationSpec`
Metadata for a perturbation, including control status, target gene, sgRNA identity, and optional sample membership.

### `ParticleCloud`
A numerical approximation object containing particle states and log-weights at one time point.

### `ContextState`
A low-dimensional tissue-context summary used by context-aware dynamics.

### `RolloutResult`
The output of a simulation backend, including terminal predicted measure and optional intermediate diagnostics.

### `SimulationBackend`
A protocol that maps initial perturbation-indexed measures to predicted terminal measures. The default implementation is particle-based.

---

## Input data assumptions

The current package assumes the user provides a shared latent state space, typically derived from pooled P4 and P60 transcriptomes.

### Minimum required inputs

- single-cell latent coordinates for P4 and P60 cells,
- perturbation labels,
- time labels identifying the initial and terminal states,
- perturbation-level abundance masses for both endpoints,
- a control annotation.

### Recommended cell metadata

- `cell_id`
- `sample_id`
- `timepoint`
- `perturbation`
- `target_gene`
- `is_control`
- `replicate`
- `coarse_cell_type` or other interpretable state labels

### Recommended perturbation metadata

- perturbation key
- gene target
- sgRNA identifier
- control/non-control status
- initial mass `M0`
- terminal mass `M1`

### AnnData compatibility

The package is expected to work with `AnnData`, but `AnnData` is treated as an **input adapter format**, not as the internal scientific object. Internally, the model operates on typed endpoint and measure contracts.

---

## Quick start

```python
from psqdynamics.api.model import PerturbSeqDynamicsModel
from psqdynamics.data.anndata_adapter import build_observed_endpoints
from psqdynamics.config.schema import (
    LatentConfig,
    DynamicsConfig,
    ContextConfig,
    OTConfig,
    TrainConfig,
)

# 1. Build perturbation-indexed endpoint measures from AnnData + guide metadata
endpoints = build_observed_endpoints(
    adata=adata,
    latent_key="X_latent",
    perturbation_key="perturbation",
    time_key="timepoint",
    initial_time="P4",
    terminal_time="P60",
    control_key="is_control",
    mass_table=mass_df,
)

# 2. Configure the model
model = PerturbSeqDynamicsModel(
    latent=LatentConfig(dim=32),
    dynamics=DynamicsConfig(
        perturbation_embed_dim=16,
        diffusion="diagonal",
        sigma_min=1e-3,
        r_max=2.0,
    ),
    context=ContextConfig(mode="none"),
    ot=OTConfig(
        divergence="sinkhorn_unbalanced",
        epsilon=0.05,
        tau=1.0,
        cost="sqeuclidean",
    ),
    train=TrainConfig(
        backend="particle",
        n_steps=32,
        lr=1e-3,
        batch_size=32,
        max_epochs=200,
    ),
)

# 3. Fit the model
model.fit(endpoints)

# 4. Predict terminal finite measures
pred = model.predict_terminal()

# 5. Run control-referenced or context-clamped counterfactuals
cf = model.counterfactual(
    perturbation="Notch1",
    mode="control_anchor",
)
```

---

## Typical workflow

### 1. Build a latent state space
Create a shared latent embedding from pooled P4 and P60 transcriptomes using PCA, scVI, or another method.

### 2. Construct finite-measure endpoints
For each perturbation, convert observed cells and abundance masses into:

- `mu0^g = M0^g p0^g`
- `nu1^g = M1^g p1^g`

### 3. Define perturbation specifications
Register control perturbations, gene targets, sgRNA-to-gene mappings, and optional sample identifiers.

### 4. Train a control-anchored dynamical system
Fit drift, diffusion, and growth fields using unbalanced OT endpoint supervision and chosen regularizers.

### 5. Evaluate endpoint fidelity
Assess terminal measure agreement, abundance calibration, normalized state-distribution accuracy, and biologically meaningful summaries.

### 6. Perform counterfactual analysis
Run simulations with:

- a perturbation embedding versus zero embedding,
- self-consistent versus clamped context,
- full versus ablated context modules.

### 7. Build perturbation atlases
Aggregate predicted perturbation effects into endpoint maps, abundance summaries, and context-mediated effect decompositions.

---

## Outputs

The package is designed to produce the following primary outputs.

### Predicted terminal finite measure
The main output is a perturbation-indexed terminal measure in latent space.

### Predicted terminal mass
A perturbation-specific estimate of endpoint expansion or depletion.

### Predicted normalized endpoint state distribution
The normalized terminal state allocation derived from the finite measure.

### Optional context trajectories
Low-dimensional summaries describing how the evolving tissue context changes during the interval.

### Counterfactual comparisons
Direct perturbation-versus-control and intrinsic-versus-context-mediated comparisons.

### Diagnostics
Transport cost, mass-relaxation penalties, abundance mismatch, diffusion magnitude, and regularization terms.

---

## Configuration philosophy

The package uses typed configuration objects instead of a single constructor with many loosely constrained keyword arguments.

Recommended config families:

- `LatentConfig`
- `DynamicsConfig`
- `ContextConfig`
- `OTConfig`
- `TrainConfig`
- `EvalConfig`

This keeps model specification explicit, reproducible, and easier to validate.

---

## Current implementation scope

### Version 1
The initial stable research implementation should prioritize:

- frozen latent input,
- exact control anchoring,
- drift + bounded growth,
- simple diagonal diffusion,
- no context by default,
- particle simulation backend,
- unbalanced OT endpoint loss,
- strong diagnostics and regularization.

### Later extensions
Planned future extensions include:

- pooled and sample-specific context protocols,
- sgRNA hierarchy with gene-level embeddings and sgRNA residuals,
- fast mean-flow inference backends,
- uncertainty models for abundance proxies,
- multimodal endpoint costs,
- large perturbation atlas generation.

---

## Reproducibility

We recommend the following defaults for all experiments:

- fixed random seeds,
- explicit config serialization,
- dataset versioning,
- checkpoint schema versioning,
- logged OT diagnostics and mass summaries,
- recorded latent-space construction metadata.

The package should save both model parameters and the full data/model configuration required to reproduce a training run.

---

## Development status

`psqdynamics` is intended as a **research software package**, not a generic black-box production library.

The initial public goal is to provide:

- a clean and testable implementation of the formulation,
- a stable measure-centric API,
- reference experiments for P4 → P60 perturb-seq,
- a foundation for future fast inference and perturbation-atlas backends.

---

## Testing priorities

The most important tests for early development are:

- control embeddings are exactly zero,
- `NoContext` reproduces the no-context model,
- zero growth preserves weights,
- particle terminal mass matches the weighted empirical definition,
- endpoint OT loss decreases on simple synthetic tasks,
- perturbation hierarchy collapses correctly when sgRNA residuals are zero,
- counterfactual control-anchor simulation matches baseline dynamics.

---

## When to use this package

Use `psqdynamics` when you have:

- destructive longitudinal single-cell snapshots,
- perturbation labels,
- perturbation-level abundance information,
- and a need to model **state transport and abundance change jointly**.

It is especially appropriate when your scientific question is not only *where cells go*, but also *which perturbations expand, deplete, or reshape endpoint composition relative to controls*.

---

## When not to use this package

This package is not the right starting point when:

- you have true lineage-resolved single-cell trajectories and want direct supervised path fitting,
- endpoint abundance information is absent and mass modeling is not needed,
- the latent representation is unreliable or not shared across time points,
- or you need a fully causal intervention model beyond the support of observed perturbation-conditioned data.

---

## Citation

If you use this software in academic work, cite:

1. the software repository,
2. the associated methods manuscript,
3. the underlying Perturb-seq dataset or study,
4. the OT / unbalanced OT / neural differential equation methods that the implementation builds on.

---

## License

Add the license that matches your intended release model, for example:

- MIT for maximal reuse,
- BSD-3-Clause for permissive academic/industry reuse,
- GPL/AGPL if copyleft is desired.

Make sure the repository license, package metadata, and documentation all match exactly.

---

## Contact

For bug reports, feature requests, or collaboration inquiries, please open an issue in the repository or contact the maintainers listed in `pyproject.toml` and `docs/`.

---

## Summary

`psqdynamics` is a measure-centric research framework for learning **control-anchored, mean-field, perturbation-conditioned dynamics** from destructive Perturb-seq snapshots. It separates the scientific law, the numerical backend, and the endpoint observation model, and it is built to support rigorous modeling of **transport, stochasticity, growth, and context** in a single coherent system.
