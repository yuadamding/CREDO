# Stable runtime and immutable recipes

CREDO separates scientific semantics from model architecture. Generality lives
in the runtime contracts; tested architecture, objective, precision, batching,
and schedule combinations live in immutable recipes.

## Runtime contract

`CREDOStudy` is the canonical in-memory study. It contains:

- an ordered physical or effect `Axis`;
- opaque `measure_id` metadata, including separate perturbation, embedding,
  guide, sample, context-group, and control identities;
- finite measures that jointly expose support and mass without parsing IDs;
- complete optional count denominators;
- one `RepresentationArtifact`;
- input and biological provenance.

The compatibility name `TrajectoryData` denotes the same object. Its `support`,
`masses`, and `counts` views expose the canonical components without creating a
second data representation.

Every SDE recipe uses `ParticleState`. Absolute particle weight is always
`log_m0 + logw`; context code must not use stabilized conditional weights as
absolute mass. Every rollout produces the common `ParticleRollout` /
`RolloutResult` fields, including state, conditional and absolute weights,
coefficients, context, IDs, grid, and consumed noise.

Controls have no residual parameter. Their effective embedding is exactly the
single learned reference. A reference counterfactual reuses source particles,
source masses, and Brownian noise, and removes only the selected residual.

## Recipe interface

Recipes publish:

- a stable ID and version;
- a machine-readable `CapabilitySet`;
- representation construction or validation;
- model construction;
- objective declarations;
- a `TrainingPlan` of typed stages;
- a checkpoint codec.

The registry rejects replacement of a loaded recipe ID/version. Unknown recipes
produce an installation-oriented error. Dataset adapters remain outside the
runtime and cannot introduce model combinations.

Third-party distributions expose `credo.recipes` entry points using the
packaging-safe name `recipe_id__version`; the registry decodes this to the
public `recipe_id@version` form and verifies that the loaded object agrees.

## Released recipes

### `credo.compact_sde_v3@3.0`

The default recipe preserves the current compact implementation: external
latent coordinates, an exact soft reference, compact weighted SDE, optional
catalog-bank growth context, geometry/mass/count/action objectives, FP32, and
state-to-mass-to-context stages. The recipe adapter delegates to the existing
trainer, and deterministic golden state and metric hashes guard behavior.

### `credo.transformer_sde_v2@2.0`

The compatibility recipe preserves historical parameter names and operations:

| Component | Archived value |
| --- | --- |
| VAE | 2,500 genes, latent 50, hidden 512, depth 2 |
| Dynamics state | 146 tensors, 5,634,421 state elements |
| Embedding / programs / mediator | 48 / 16 / 48 |
| Coefficient MLP | hidden 384, depth 4 |
| Context | token 128, 4 heads, 1 within, 1 cross, 32 inducing |
| Context effect | Growth only |
| Objective | Endpoint geometry/mass plus weak form at weight 0.12 |
| Precision / optimizer | BF16 / AdamW |
| Particles / integration | 128 train, 640 evaluation, 24 steps per interval |

All four archived 146-tensor dynamics states and all four 14-tensor VAE states
strict-load. Embedded EMA selection merges EMA parameters with preserved raw
buffers and also strict-loads. The normalized-law weak-form residual is kept in
the recipe package.

## Representation and split provenance

`RepresentationArtifact` records the latent cache, VAE or encoder state, gene
order and mask, normalization, fit scope, included samples and times, and
producer metadata. The LPS artifact is explicitly
`fit_scope=all_source_samples`: its donor-held-out dynamics result is therefore
a transductive-source evaluation, not a nested representation evaluation.

`SplitSpec` records the split strategy, train and validation values, fold, fold
count, and whether representation fitting was shared or nested. The imported
LPS folds preserve held-out donors `02`, `06`, `03/12`, and `01/04`; held-out
samples are asserted absent from dynamics training measures.

## Checkpoint modes

The schema-v2 envelope records recipe, study, representation, split, state,
training, capability, and import provenance contracts. Modes are:

- `inference_only`: evaluation and supported counterfactuals, no continuation;
- `resume_capable`: model, optimizer, scheduler, and RNG state are complete;
- `training_recipe_only`: a training design exists without an inference state.

Historical v2 and current compact checkpoints are honestly marked
`inference_only`. Both have released fresh-training plans; neither claims an
exact continuation trajectory from the saved checkpoint.

The compact plan is backed by the released shared `TrainingEngine` executor.
The v2 plan is presently a typed reconstruction of the archived design, not an
executable promise: this release supports strict import, raw/EMA inference,
weak-form diagnostics, counterfactuals, and replay. A fresh v2 optimizer loop,
fold-nested VAE experiment, and controlled v2/v3 retraining comparison remain
new experiments and must receive new manifests rather than being inferred from
the archived checkpoints.

## Archived LPS replay

The four raw selected checkpoints were replayed with 640 particles, 24 steps per
interval, BF16 CUDA execution, preserved latent rows, selected-epoch initial
sampling seeds, and deterministic noise seed 0.

| Fold | Held out | Rows | Max abs log-mass difference | Rank correlation | Agreement |
| --- | --- | ---: | ---: | ---: | --- |
| 00 | 02 | 49 | 0.05464 | 0.99980 | Tolerance-level |
| 01 | 06 | 52 | 0.04403 | 1.00000 | Tolerance-level |
| 02 | 03, 12 | 83 | 0.04334 | 0.99983 | Tolerance-level |
| 03 | 01, 04 | 84 | 0.03063 | 0.99990 | Tolerance-level |

All 268 OOF rows and per-fold measure order match the archive. Replayed mean
geometry differs from archived mean geometry by 0.009-0.061 across fold/time
blocks. Agreement is classified as tolerance-level because the original global
noise state and exact producer wrapper were not preserved. This is inference
replay, not byte-for-byte retraining.

## Comparison policy

Cross-recipe comparisons may use identical splits, endpoint metrics, particle
grids, donor summaries, perturbation rankings, uncertainty, runtime, and memory.
Raw objectives, parameter tensors, weak-form versus action penalties,
checkpoint hashes, and coordinates from different representations are not
directly comparable. A controlled architecture comparison must freeze one
representation for both recipes.
