# CREDO

**CREDO is a framework for preparing, fitting, evaluating, and comparing
perturbation-conditioned population-dynamics models from longitudinal
Perturb-seq experiments with destructive single-cell observations.**

Its biological invariant is:

```text
perturbation-indexed biological populations
observed through destructive single-cell snapshots
at ordered experimental checkpoints
with explicit controls, replicates, representations, and abundance semantics
```

CREDO does not imply that the same cells were followed across checkpoints. A
population series records its continuity as same-unit, matched parallel,
cross-sectional, independent replicate, lineage-linked, exactly traced, or
unknown. Models estimate regularized population transitions; they do not infer
cell genealogies or turn model counterfactuals into experimental causal effects.

## Scope

CREDO covers endpoint and multi-checkpoint Perturb-seq screens along physical
time, ordered stage, development, disease progression, or pseudotime. It
supports CRISPR perturbations, combinations, chemicals, cytokines, controls,
donors, experimental units, biological and technical replicates, missing
geometry, zero counts, irregular coverage, alternative representations, and
absolute, relative, captured, censored, or absent abundance.

Static unperturbed scRNA-seq, spatial registration, foundation-model pretraining,
imaging, bulk omics, and unrelated multimodal prediction are outside the core.
Such tools may provide a frozen representation without changing the biological
contract.

## Core Contract

```text
PerturbSeqStudy
    -> PerturbSeqView
    -> SplitPlan
    -> CompiledLPSProblem
    -> versioned recipe
    -> run.json
```

`PerturbSeqStudy` separates identities that must not be conflated:

| Identity | Meaning |
| --- | --- |
| `perturbation_id` | Observed experimental condition |
| `construct_id` | Physical guide or reagent |
| `target_id` | Intended biological target |
| `effect_id` | Model parameter-sharing identity |
| `context_id` | Biological or technical covariates |
| `population_pool_id` | Physical or computational grouping with evidence |
| `series_id` | Matched population-level longitudinal unit |
| `observation_id` | One destructive snapshot at one checkpoint |

Intervention events state when perturbations and background stimuli occurred
relative to the source. The source checkpoint is therefore never silently
treated as a pre-perturbation baseline.

The public root names `Study` and `StudyView` are aliases for
`PerturbSeqStudy` and `PerturbSeqView`. Schema-v3 `Study` remains available from
`credo.data` only for explicit compatibility conversion. `TrajectoryData` and
`CREDOStudy` are deprecated recipe-facing numerical payloads, not study models.

## Install

```bash
python -m pip install -e ".[dev]"
credo --help
```

Python 3.11 through 3.13 are supported. GPU execution follows the installed
PyTorch build.

## Native Schema V4

Native v4 stores the full biological contract in typed Parquet tables and
stores empirical laws in packed HDF5 arrays:

```text
study/
|-- perturbations.parquet
|-- perturbation_components.parquet
|-- intervention_events.parquet
|-- contexts.parquet
|-- series.parquet
|-- observations.parquet
|-- support_index.parquet
|-- abundance.parquet
|-- compositions.parquet
|-- population_pools.parquet
|-- effect_bindings.parquet
|-- reference_bindings.parquet
|-- stores/store-0000.h5
|-- stores/store-0000.parquet
|-- provenance.json
`-- study.json
```

The packed store uses `coordinates`, `probabilities`, and `indptr` arrays per
representation, so writing does not create one HDF5 group per observation.
`write_study()` writes transactionally, hashes every artifact, writes the
manifest last, and atomically renames the completed directory.

```python
from credo import open_study, write_study

study = open_study("inputs/cohort/study.json", verify="semantic")
try:
    manifest = write_study(study, "inputs/cohort/native-v4")
finally:
    study.close()
```

Verification levels are `none`, `schema`, `manifest`, `semantic`, and `full`.
Full verification scans every packed support and recomputes semantic digests.
Schema-v3 native and five-file studies remain readable and are converted to v4
without silently inventing intervention timing or continuity.

## Selection And Splits

Selections are typed over subjects, experimental units, perturbations,
constructs, targets, controls, contexts, checkpoints, observations, QC tiers,
representations, abundance channels, and effect/reference bindings.

`SplitPlan` records exact train and validation observations plus held-out
subjects, units, perturbations, constructs, targets, contexts, checkpoints, and
series. Supported task labels include subject, unit, guide-within-target,
target, perturbation, context, checkpoint interpolation/extrapolation,
combination, series, and self-evaluation.

Representation provenance uses explicit protocols:

```text
external_frozen
shared_all_observations
shared_source_only
nested_by_subject
nested_by_perturbation
nested_by_checkpoint
fully_nested
```

The split ID is derived from biological selection and support availability, not
from downstream encoder artifacts. A nested representation can therefore be
fit after split planning and bind back to the exact split without circular
identity. CREDO checks fit subjects, perturbations, checkpoints, observations,
selection hash, and split ID against the task.

Compilation is split-specific. Training and validation target laws are separate
objects, and held-out outcomes never enter fitting through a composition
background. A partial held-out denominator is explicitly conditioned on its
training selection.

## Geometry, Abundance, And Pools

Geometry and abundance are independent. An empirical support law always has
normalized atom probabilities; a named abundance channel separately declares
semantics, unit, denominator scope, zero policy, and transform provenance.
Geometry-only recipes select no channel. Absolute-growth claims require
absolute abundance; relative channels support only within-denominator claims.

Composition blocks declare whether a denominator is a sequencing library,
competition pool, culture pool, capture stratum, or sampling stratum. Population
ecology recipes accept only evidence from shared living culture, shared tissue,
or competition pools. Sequencing and capture groupings cannot be promoted to
ecological interaction.

## Recipes

Released models consume the same study, selection, split, prediction, metric,
counterfactual, and run contracts:

- `credo.compact_sde_v3@3.0`: fresh endpoint or multitime weighted-SDE fitting,
  optional abundance/composition likelihood, one soft reference, and optional
  catalog population context.
- `credo.transformer_sde_v2@2.0`: strict inference from archived transformer-SDE
  checkpoints over the same compiled finite-measure problem; fresh training is
  not released.

The compact model, solver, objectives, and trainer live under
`credo.recipes.compact_sde_v3`. Thin old import modules remain for this alpha
cycle. Additional recipes may compile unbalanced-flow, state-sequence, or
coupling problems without defining another cohort data model.

## Run

```yaml
recipe: credo.compact_sde_v3@3.0
study: inputs/cohort/native-v4/study.json

selection:
  representation_id: latent32_all
  abundance_channel: modeled_frequency
  effect_binding_id: target_gene_shared
  reference_binding_id: donor_matched_ntc
  composition_policy: require_complete

recipe_config:
  validation:
    strategy: context_group
    values: [donor-1]
    fraction: 0.0
    representation_scope: shared

output: runs/cohort/donor-1
```

```bash
credo validate run.yaml
credo train run.yaml --device cuda
credo evaluate runs/cohort/donor-1/run.json --output evaluation.parquet
credo counterfactual runs/cohort/donor-1/run.json --series-id donor-1::guide-1
```

`run.json` binds recipe, config, study, selection, split, compiled problem,
checkpoint codec, and artifact hashes. Predictions are keyed by stable series,
observation, checkpoint, and representation IDs. Same-start counterfactuals
reuse the factual source population and noise while replacing only the modeled
perturbation residual according to the selected control binding.

## Repository Boundary

Cohort downloads, preprocessing, adapters, configs, and replay scripts belong
in the analysis workspace or a separate adapter package. Core code never parses
semantics from IDs and never branches on a cohort name.

See [the study contract](docs/study_model.md),
[runtime and recipes](docs/runtime_and_recipes.md), and
[scientific validation](docs/scientific_validation.md).

## Verify

```bash
ruff check src tests analysis
ruff format --check src tests analysis
pytest -q
python -m build
```
