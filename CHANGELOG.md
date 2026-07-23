# Changelog

## 3.0.0a5 - 2026-07-22

- Replaced the generic public study vocabulary with schema-v4
  `PerturbSeqStudy`: perturbations, components, intervention timing, contexts,
  population series, destructive snapshots, population pools, and mandatory
  continuity semantics are now explicit.
- Added typed Perturb-seq selection and task-aware split identities for
  subjects, units, guides, targets, perturbations, contexts, checkpoints,
  combinations, and series, with exact representation-fit leakage checks.
- Added split-scoped `FiniteMeasureDynamicsProblem` compilation with separate
  training and validation outcomes and leakage-safe composition conditioning,
  plus common unbalanced-flow, state-sequence, and coupling problem contracts.
- Added native schema-v4 transactional persistence with packed HDF5 support
  arrays, bounded batch writes, and buffer-based table hashing; conservative
  schema-v3 conversion records unknown biological semantics rather than
  inventing them.
- Made lazy five-file conversion stream categorical H5AD observation codes
  into a compact CSR-style support index and read atom weights per law,
  avoiding a full cell-level pandas table during large native conversions.
- Made compact-v3 and transformer-v2 consume the same study and compiled
  problem contracts; added typed prediction and same-start counterfactual
  queries and compiled-problem validation to the recipe protocol.
- Made compact compilation preserve checkpoint-specific destructive sample
  identities while using declared context trajectories for numerical and
  composition grouping.
- Moved compact model, solver, objectives, and trainer implementations under
  `credo.recipes.compact_sde_v3`, retaining thin alpha compatibility imports.
- Tightened cross-table validation for controls, effects, references,
  representation provenance, abundance denominators, compositions, contexts,
  pools, event coordinates, and replicate identities.

## 3.0.0a4 - 2026-07-22

- Added one content-addressed `SplitPlan` used by semantic compilation,
  training, evaluation, checkpoints, and run reloads, with machine-checked
  shared/transductive and nested/inductive representation scope.
- Added transactional native schema-v3 Study persistence with Parquet semantic
  tables, lazy HDF5 support stores, artifact verification, support semantic
  digests, and third-party codec discovery.
- Moved effect and reference parameter sharing into run-selectable binding
  catalogs and made compact-v3 reject multiple biological reference pools.
- Added explicit no-abundance unit semantics, static context/sample checks,
  replicate selection and pooling, selection-conditioned denominators, and
  detached background-preserving composition blocks.
- Replaced weak identifier hashes with Study, selection, split, compiled
  problem, and run contract hashes that bind scientific values while excluding
  artifact locations.
- Added the generic `run.json` loader and recipe-owned checkpoint reconstruction
  for compact and imported transformer bundles, including post-import Study
  binding.
- Split persisted results into predictions, long-form common metrics, and
  recipe diagnostics, and changed compact output to the generic seven-file run
  layout.
- Added the recipe-neutral `train()` facade, reduced root exports to the
  semantic workflow, and retained legacy numerical objects as deprecated
  compatibility shims.
- Documented the MIT to AGPL-3.0-only transition introduced in `39dcb615`.

## 3.0.0a3 - 2026-07-21

- Added the storage-independent `Study` model and a lazy five-file compatibility
  codec with explicit conditions, series, observations, abundance, composition,
  representation, and support-store contracts.
- Added real loader verification levels, immutable semantic tables,
  representation-specific support coverage, multiple support stores, replicate
  observations, and topology-aware design validation.
- Added recipe requirements and StudyView compilers; `credo train` now enters
  compact-v3 through the semantic Study layer while preserving golden numerics.
- Introduced one stable study, particle, evaluation, counterfactual, training-plan,
  and checkpoint-envelope runtime with immutable recipe identifiers.
- Preserved compact v3 behind `credo.compact_sde_v3@3.0` with golden state,
  metric, and serialization checks.
- Added `credo.transformer_sde_v2@2.0`, strict raw and EMA legacy import,
  preserved VAE loading, weak-form diagnostics, and inference-only provenance.
- Replayed all four archived LPS donor folds into the common metric contract;
  all 268 rows and measure ordering match with tolerance-level numerical agreement.
- Moved LPS raw-data interpretation and replay orchestration to the external
  workspace; core transformer tests now use a generated compatibility fixture.
- Made representation scope, validation split, recipe capability, implementation
  hash, and checkpoint continuation limits explicit and machine-readable.

## 3.0.0a2 - 2026-07-21

- Exposed growth and Sinkhorn scales and bound them into checkpoint contracts.
- Split evaluation resolution from training, added explicit donor/checkpoint
  validation and target-balanced training batches.
- Made intrinsic counterfactuals focal-only, admitted control nulls, and added
  particle, seed, checkpoint, package, and Git provenance to persisted rows.
- Advanced the checkpoint schema for the new validation and evaluation
  contract while preserving migration of earlier counterfactual tables.

## 3.0.0a1 - 2026-07-21

- Replaced endpoint, trajectory, and single-time problem classes with one
  typed `Axis` and one opaque-identifier `TrajectoryData` contract.
- Fixed the installed architecture to one soft reference, diagonal diffusion,
  observation-derived program context, and ecological growth.
- Consolidated simulation, checkpoint geometry and mass, complete count
  blocks, training, evaluation, persistence, and counterfactual execution.
- Added a fixed state to mass to context schedule with complete catalog-bank
  initialization and observation-weighted held-out evaluation.
- Moved GSE314342, HNSCC, and LPS preparation to workspace-owned adapters and
  retained only the synthetic contract example in CREDO.
- Replaced mode-specific runners and compatibility aliases with `credo
  validate`, `credo run`, and `credo summarize`.
- Reduced run output to one manifest, checkpoint, history table, metrics table,
  and counterfactual table.
- Enforced explicit mass denominators, diagnostic-only captured-count claims,
  and complete-context holdouts for compositional count validation.
- Bound checkpoint reloads to canonical metadata, input hashes, axis and mass
  semantics, and the non-operational run configuration; counterfactual rows
  now remain intact across separate reload-and-evaluate processes.
- Required self-describing dataset manifests, made adapters persist their
  validated support and count representations, rejected stale output files,
  and required mass training before ecological context.
- Removed experimental attention, search, expression-preprocessing, weak-form,
  cohort-loading, and legacy compatibility paths from the installed package.

The pre-compaction 2.0.1 implementation remains available at the
`pre-compaction-v2.0.1` Git tag.
