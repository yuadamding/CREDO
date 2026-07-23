# Runtime and recipes

CREDO recipes are model implementations over one longitudinal Perturb-seq
contract. They do not read cohorts or define competing study schemas.

## Recipe protocol

Every `LongitudinalPerturbSeqRecipe` provides:

```text
config_schema
requirements
plan_split
compile
validate_compiled
fit
load
predict
counterfactual
```

`PredictionQuery` selects stable series and checkpoints. `CounterfactualQuery`
selects one series and enforces same source, same noise, and an explicit context
policy. Results use recipe-neutral prediction, metric, diagnostic, and
counterfactual tables.

The compatibility methods for model construction, objective declaration,
training plans, and checkpoint codecs remain while alpha checkpoints are still
supported. The engine validates both biological requirements and compiled
problem kind before constructing tensors or a model.

## Requirements

`LongitudinalPerturbSeqRequirements` declares checkpoint mode, topologies, axis
kinds, intervention timing, perturbation kinds, combination and unseen-target
support, representation kinds and protocols, abundance semantics, composition
support, control/reference limits, missing geometry, replicate modes,
continuity, sample scope, and context mode.

Context mode is one of none, series-static, observation-varying, or population
ecology. Population ecology is validated against physical pool evidence; a
sequencing or capture group is rejected.

Unsupported semantics fail before compilation. Examples include a target-held
out task for a recipe without unseen-target support, multiple reference pools
for compact-v3, an all-checkpoint representation presented as inductive, or an
absolute-growth claim from relative abundance.

## Split planning

`SplitPlan` is content-addressed and stores exact train/validation selections,
series, checkpoints, observations, task kind, representation protocol, and all
held-out biological identities. The same immutable plan is used by compilation,
training, evaluation, checkpoints, and run reload.

Tasks include subject, experimental unit, guide within target, target,
perturbation, context, checkpoint interpolation/extrapolation, combination,
series, and train-self evaluation. Legacy `SplitSpec` names remain accepted as
typed requests and are resolved into the richer plan.

Representation checks are task-specific. Nested subject evaluation checks fit
subjects; checkpoint evaluation checks fit checkpoints; perturbation and target
tasks check fit perturbations. Fully nested representations must bind the exact
split and optional training-selection hash.

## Compiled problems

The framework exposes a small problem family:

- `FiniteMeasureDynamicsProblem` for weighted ODE/SDE and reaction-drift-
  diffusion recipes.
- `UnbalancedFlowProblem` for endpoint or adjacent-checkpoint transport with
  mass variation.
- `StateSequencePredictionProblem` for source/context/target generators.
- `CouplingProblem` for probabilistic adjacent-checkpoint maps.

`FiniteMeasureDynamicsProblem` contains independent training and validation
numerical payloads plus a `CompiledLPSSplit` with permitted source observations,
training targets, validation targets, and optional denominator background.
Held-out outcomes cannot be reached through the training payload. When a
perturbation holdout cuts a composition block, training receives a new
selection-conditioned denominator rather than validation counts.

`TrajectoryData` is retained only as the compact/transformer numerical payload
and direct compatibility input. It is not a public biological study.

## Released recipes

### Compact weighted-SDE v3

`credo.compact_sde_v3@3.0` supports fresh endpoint and multitime chain fitting,
latent supports, optional abundance and composition likelihood, flat guide- or
target-level effects, one global soft reference, and optional population catalog
context. Its model, particle solver, objectives, and trainer are isolated in
`credo.recipes.compact_sde_v3`.

Training uses the training finite-measure payload and training catalog bank.
Evaluation uses a separately compiled validation payload and validation bank.
Validation source states are available only as task inputs; target outcomes do
not enter fitting.

### Archived transformer-SDE v2

`credo.transformer_sde_v2@2.0` consumes the same finite-measure problem for
strict imported inference. It preserves the historical dynamics and expression
VAE contracts. Fresh training and resume remain unavailable because optimizer,
scheduler, RNG, and terminal training state were not archived.

The two recipes share biological contracts and output tables, not architecture,
objective tensors, or representation assumptions.

## Same-start counterfactuals

A reference counterfactual starts from the factual empirical source population,
retains source abundance, uses the same Brownian noise, and removes only the
selected perturbation residual while retaining the chosen reference effect.
Context can be recomputed self-consistently or clamped when the recipe supports
it. This is a model contrast, not a lineage or causal-effect claim.

## Run bundles

```text
run/
|-- run.json
|-- state/checkpoint.pt
`-- tables/
    |-- history.parquet
    |-- predictions.parquet
    |-- metrics.parquet
    |-- diagnostics.parquet
    `-- counterfactuals.parquet
```

`run.json` records recipe and codec IDs, resolved config, capabilities, study
hash, selection hash, exact split, problem hash, state contract, and every
artifact hash. `open_run()` verifies these identities, reopens the bound study,
recompiles the saved split, and delegates state loading to the recipe.

Predictions are keyed by series, observation, checkpoint, and representation.
Metrics are long-form; ESS, weight concentration, integration steps, particle
counts, and seeds are diagnostics. Deterministic future recipes can use the same
tables without fabricating particle fields.

## Repository boundary

Raw cohort interpretation, preprocessing, fold orchestration, and replay
commands remain outside CREDO core. An adapter constructs one native v4 study;
all recipes then use that study without cohort-name branches.
