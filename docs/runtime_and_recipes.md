# Runtime and recipes

CREDO keeps semantic selection, recipe compilation, and numerical execution as
separate contracts.

## Recipe boundary

Each recipe publishes:

- a stable ID and version;
- `RecipeRequirements` for axes, topology, representations, abundance,
  references, compositions, and replicates;
- a `StudyView` compiler;
- model, objective, training-plan, and checkpoint contracts;
- explicit capabilities for training, inference, evaluation, and
  counterfactuals.

`TrainingEngine.fit()` accepts a `Study` or `StudyView`, validates requirements,
compiles it through the recipe, and then invokes the recipe executor. Passing a
`TrajectoryData` object remains a compatibility API. Unsupported semantics fail
before model construction.

Both released recipes currently compile to the established finite-measure
runtime: an ordered physical or effect axis, opaque measure metadata,
checkpoint laws, an abundance semantics, optional complete count blocks, and
one selected representation. Guide and target fields are optional. The current
alpha keeps an explicit compatibility `embedding_id` and reference assignment
in its condition and series tables; separate named binding catalogs are not yet
released.

## Shared numerical contract

SDE recipes use `ParticleState` and the common Euler-Maruyama driver. Absolute
particle weight is always `log_m0 + logw`; conditional stabilization never
replaces absolute mass. Recipe kernels own coefficient evaluation while the
driver owns state updates, noise consumption, and checkpoint capture.

Controls have no residual parameter. A reference counterfactual reuses source
particles, source masses, and Brownian noise and removes only the selected
residual.

## Released recipes

### `credo.compact_sde_v3@3.0`

Compact-v3 supports fresh training on physical-time chains and effect chains.
It preserves the released latent SDE, exact shared reference, optional catalog
context, geometry/mass/count objectives, FP32 execution, and
state-to-mass-to-context schedule. The Study compiler preserves legacy metadata
when reading a five-file dataset, so golden model and metric hashes are
unchanged.

### `credo.transformer_sde_v2@2.0`

Transformer-v2 preserves the historical 146-tensor dynamics architecture and
14-tensor expression VAE. Its importer strict-loads raw or embedded EMA state,
verifies source and portable-artifact hashes, and records the typed archived
training plan. Imported bundles are inference-only because optimizer, scheduler,
RNG, and terminal training state were not preserved.

Generic evaluation and same-start/same-noise counterfactual kernels remain in
core. Raw LPS AnnData interpretation, held-out donor reconstruction, fold
orchestration, and archived-output comparison are owned by the external
analysis workspace. Core CI uses a generated complete-shape transformer fixture
and contains no private cohort paths.

## Checkpoint boundary

Checkpoint envelopes record recipe, study, representation, split, state,
training, capability, and import provenance. Modes are:

- `inference_only`: evaluation is supported but continuation is not;
- `resume_capable`: model, optimizer, scheduler, and RNG state are complete;
- `training_recipe_only`: a design exists without inference state.

Current compact and imported transformer checkpoints are honestly marked
`inference_only`. Compact supports deterministic fresh fitting, but its saved
checkpoint is not resumable.

The current core CLI evaluates native compact checkpoints with an explicit
config. Generic `run.json` loading for all recipes has not yet been released;
the CLI therefore does not infer run type from file-versus-directory paths.

## Claim boundary

Compact-v3 supports fresh fitting and same-study evaluation. Transformer-v2
supports strict imported inference and same-study evaluation when supplied a
compatible compiled study through the Python API. Neither recipe claims
checkpoint resume, arbitrary cross-dataset evaluation, or cross-hardware
bitwise retraining.

Cross-recipe comparisons may share splits, endpoint metrics, particle grids,
rankings, uncertainty, runtime, and memory. Raw objectives, tensors, and
coordinates from different representations are not directly comparable.
