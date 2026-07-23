# Longitudinal Perturb-seq study contract

`PerturbSeqStudy` is the canonical biological object. It represents
perturbation-indexed populations sampled destructively at ordered checkpoints;
it is independent of storage and model architecture.

```text
PerturbSeqStudy -> PerturbSeqView -> SplitPlan -> CompiledLPSProblem
```

## Longitudinal design

`LongitudinalDesign` has one `ProgressionAxis`, at least two checkpoints,
allowed transitions, and a chain, star, or DAG topology. Supported progression
kinds are physical time, ordered stage, developmental stage, disease stage,
pseudotime, and the compatibility effect axis. Donor, dose, cell type, and
stimulation are contexts or intervention events, not extra time axes.

Every `PopulationSeriesTable` row includes subject, experimental unit,
perturbation, context trajectory, biological replicate, and mandatory
`continuity_kind`. Continuity is one of:

```text
same_experimental_unit
matched_subject_parallel
cross_sectional_population
independent_replicate
lineage_linked
exact_lineage_traced
unknown
```

This field limits permissible claims. `series_id` never implies that individual
cells survived from one checkpoint to the next.

## Perturbations and events

`PerturbationTable` records an experimentally assigned condition and whether it
is a control. `PerturbationComponentTable` separately normalizes constructs,
targets, doses, ordering, and combinations. Therefore a guide, perturbation,
target, and model effect may have different identities.

`InterventionEventTable` records every series-level primary perturbation and
optional background events. Coordinates are explicit nullable columns;
`start_relation` states whether an event occurred before source, at source,
between checkpoints, after the last observation, or at an unknown time. Every
series requires exactly one primary perturbation event whose agent matches its
perturbation.

## Destructive observations

`SnapshotObservationTable` separates observation existence from observed
modalities. `geometry_observed` and `abundance_observed` are independent. The
support index declares every `(observation_id, representation_id)` pair as
available or unavailable, allowing missing geometry in one representation and
available geometry in another. Replicate snapshots retain distinct IDs and
require distinct technical-replicate identities.

`sample_id` and `context_id` name the actual destructive sample and its
checkpoint-specific context, so either may vary across a population series.
Recipes that need a stable numerical group use the series'
`context_trajectory_id`; they do not reinterpret a sequencing sample as a
continuous culture.

No core logic parses biological semantics from `series_id`, `observation_id`,
or `perturbation_id`; all joins use declared foreign keys.

## Representations

Each `RepresentationSpec` binds a coordinate space to one support store and
records feature selection, normalization, encoder, decoder, and support
artifacts. Fit provenance includes:

```text
scope_mode
fit_split_id
fit_selection_hash
fit_subject_ids
fit_perturbation_ids
fit_checkpoint_ids
fit_observation_scope
```

Encoded supports may cover observations outside the fitting set. The fit fields
describe representation training, while support availability describes encoded
output. This distinction lets CREDO label shared representations transductive
and exact nested representations inductive for a particular task.

## Geometry and abundance

An `EmpiricalLaw` contains finite coordinates and nonnegative probabilities
summing to one. Abundance is a separate table of named channels. Each
`AbundanceChannelSpec` declares absolute, relative, capture-count, unit, or
unknown semantics; denominator scope; zero policy; and optional transform input,
identity, and immutable parameters.

The study may retain raw zero counts while exposing a positive transformed
modeling channel. A transformed row must have its observed input row. Relative
and captured channels require denominator IDs whose declared scope agrees with
observation metadata.

`LPSCompositionTable` records count denominators and a mandatory `block_kind`:

```text
sequencing_library
competition_pool
culture_pool
capture_stratum
sampling_stratum
```

`PopulationPoolTable` independently records physical or computational grouping
and evidence. Only shared living culture, shared tissue, and competition pools
constitute ecological evidence. A sequencing library can be a valid count
denominator without being a living interaction pool.

## Effects and controls

`PerturbationEffectBindingTable` maps perturbations to run-selectable effect IDs.
Separate bindings can express guide-specific, target-shared, residual, pathway,
or compositional parameterizations over one immutable study.

`PerturbationReferenceBindingTable` maps every perturbation to an observed
control pool and model counterfactual effect. Reference scopes are global,
subject, experimental unit, context, checkpoint, or processing batch.
`match_keys` is parsed as a JSON string array; non-global scopes must include
their identifying key. Every reference pool must contain an observed control,
and counterfactual effects must resolve in the effect catalog.

## Typed selection

`PerturbSeqSelection` selects stable subjects, units, perturbations, constructs,
targets, control kinds, contexts, checkpoints, observations, QC tiers,
representation, abundance channel, effect binding, and reference binding.
Unknown IDs fail immediately.

Composition policies are:

- `require_complete`: reject a partial denominator.
- `preserve_background`: retain unselected non-validation denominator members.
- `condition_on_selection`: mint a denominator bound to the selected subset.
- `drop`: omit count likelihoods.

Replicate policies are reject, select, pool, keep-separate, or hierarchical.
The released compact recipe executes reject, select, and explicit pooling.

## Content identity

```text
semantic tables + design + representation/support identities
    -> study_content_hash
biological view + selected contracts
    -> selection_hash
biological split basis + task + representation protocol
    -> split_id
split-specific compilation
    -> problem_hash
recipe + config + artifacts
    -> run contract hash
```

Artifact locations are excluded from semantic identity; artifact content hashes
are included. Split identity deliberately excludes fitted encoder artifacts and
`fit_split_id`, preventing a circular hash while retaining all biological
partition inputs and support availability.

## Native schema v4

Schema v4 persists perturbations, components, events, contexts, population
series, observations, support coverage, abundance, compositions, pools,
bindings, representations, and provenance. Supports use packed HDF5 arrays with
a Parquet law index. Writes are transactional and existing destinations are
never overwritten.

Verification levels are:

| Level | Work |
| --- | --- |
| `none` | Parse and construct lazy handles |
| `schema` | Construct typed tables and stores |
| `manifest` | Verify artifact sizes and SHA-256 hashes |
| `semantic` | Verify cross-table biological contracts |
| `full` | Scan every support and recompute semantic digests |

Schema-v3 and five-file codecs are read compatibility layers. Conversion emits
schema v4, records unknown continuity and intervention timing explicitly, and
labels legacy composition groups as sampling strata. It never silently
reinterprets old manifests. Lazy five-file conversion streams categorical H5AD
observation codes into a compact CSR-style law index and reads support
coordinates and atom weights on demand.
