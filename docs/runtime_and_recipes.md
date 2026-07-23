# Runtime and recipes

CREDO keeps semantic selection, split planning, compilation, and numerical
execution as separate contracts.

## Recipe boundary

Each recipe publishes:

- a stable ID and version;
- `RecipeRequirements` for design, representations, abundance, references,
  context, compositions, and replicates;
- a split planner and Study compiler;
- model, objective, training-plan, and checkpoint contracts;
- explicit capabilities for training, inference, evaluation, resume, and
  counterfactuals.

`train()` and `TrainingEngine.fit()` accept a `Study` or `StudyView`, plan and
verify the split, check representation leakage, validate recipe requirements,
compile through the recipe, and only then construct the model. Unsupported
semantics fail before tensorization.

The old `TrajectoryData` numerical object remains an internal compiler target
and a deprecated compatibility input. It is not a competing semantic API.

## Split contract

`SplitPlan` is created before compilation and contains:

- exact train and validation series, checkpoints, and observations;
- train and validation `SelectionSpec` values;
- held-out series, checkpoints, and observations;
- split strategy and evaluation source;
- shared or nested representation scope;
- transductive or inductive evaluation label;
- a SHA-256 ID bound to the selected Study content.

Saved plans are structurally revalidated when a run is opened. Generic loading
uses the saved plan rather than re-inferring a default split, so caller-supplied
explicit splits remain reproducible.

For nested checkpoint validation, held-out checkpoints must be absent from the
representation fit scope. For nested series or context validation, held-out
series must be absent. Shared representations are accepted but explicitly
labeled transductive.

## Shared numerical contract

Weighted-SDE recipes use `ParticleState` and the common Euler-Maruyama driver.
Absolute particle weight is always `log_m0 + logw`; conditional stabilization
never replaces absolute mass. Recipe kernels own coefficient evaluation while
the driver owns updates, noise consumption, and checkpoint capture.

Controls have no residual parameter. A reference counterfactual reuses source
particles, source masses, and Brownian noise and removes only the selected
residual.

Composition compilation carries both source and modeled denominator IDs.
Background-preserving selections retain detached background fitness, exposure,
and counts. Selection-conditioned blocks receive a new content-derived
denominator identity.

## Released recipes

### `credo.compact_sde_v3@3.0`

Compact-v3 supports fresh training on chain designs, one global soft reference,
optional catalog context, geometry/mass/count objectives, FP32 execution, and a
state-to-mass-to-context schedule. Context and sample identities must remain
static within a series. Replicates may be selected or pooled before numerical
execution.

### `credo.transformer_sde_v2@2.0`

Transformer-v2 preserves the historical 146-tensor dynamics architecture and
14-tensor expression VAE. Its importer strict-loads raw or embedded EMA state,
verifies source and portable-artifact hashes, and records the archived training
plan. Imported bundles are inference-only because optimizer, scheduler, RNG,
and terminal training state were not preserved.

Raw cohort interpretation and fold orchestration remain external adapter work.
Core CI uses generated complete-shape fixtures and contains no private cohort
paths.

## Generic run bundle

Every released runtime is opened through `open_run("run/run.json")`.
`run.json` records:

- recipe and checkpoint codec;
- resolved config and capabilities;
- Study, selection, split, and compiled-problem identities;
- state and output paths;
- size and SHA-256 for every artifact;
- a run contract hash.

The loader verifies artifacts, resolves the recipe registry entry, reopens the
bound Study, verifies content and selection hashes, validates the persisted
split, recompiles the problem, and delegates state reconstruction to
`recipe.load_checkpoint()`.

Imported transformer runs may be created unbound. `bind_run_study()` validates
a config and Study against the imported representation contract and rewrites
the run manifest with semantic identities. After binding, compact and
transformer runs use the same evaluate and counterfactual dispatch.

## Output contract

Outputs are separated by meaning:

```text
predictions: run, recipe, representation, series, observation, checkpoint,
             predicted and observed log abundance

metrics:     run, recipe, series, observation, checkpoint,
             metric_name, value, unit

diagnostics: run, recipe, series, observation, checkpoint,
             diagnostic_name, value, unit
```

Particle ESS, maximum weight fraction, integration steps, particle counts, and
seeds are diagnostics rather than required common metrics. Deterministic ODE,
OT, or decoder recipes can therefore emit the same common tables without
inventing particle values.

## Checkpoint modes

- `inference_only`: evaluation is supported but continuation is not.
- `resume_capable`: model, optimizer, scheduler, and RNG state are complete.
- `training_recipe_only`: a typed design exists without inference state.

Current compact and imported transformer checkpoints are honestly marked
`inference_only`. Compact supports deterministic fresh fitting, but its saved
checkpoint is not resumable.

## Claim boundary

Compact-v3 supports fresh fitting and same-study evaluation. Transformer-v2
supports strict imported inference after binding a compatible same-study
selection. Neither recipe claims arbitrary cross-dataset evaluation,
checkpoint resume, or cross-hardware bitwise retraining.

Cross-recipe comparisons may share splits, endpoint metrics, rankings,
uncertainty, runtime, and memory. Raw objectives, tensors, and coordinates from
different representations are not directly comparable.
