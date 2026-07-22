# Changelog

## 3.0.0a3 - 2026-07-21

- Introduced one stable study, particle, evaluation, counterfactual, training-plan,
  and checkpoint-envelope runtime with immutable recipe identifiers.
- Preserved compact v3 behind `credo.compact_sde_v3@3.0` with golden state,
  metric, and serialization checks.
- Added `credo.transformer_sde_v2@2.0`, strict raw and EMA legacy import,
  preserved VAE loading, weak-form diagnostics, and inference-only provenance.
- Replayed all four archived LPS donor folds into the common metric contract;
  all 268 rows and measure ordering match with tolerance-level numerical agreement.
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
- Moved cohort preparation to GSE314342, HNSCC, LPS, and synthetic adapters.
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
