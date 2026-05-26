# Changelog

## 2.0.5

- Marked the package release as `credo==2.0.5`.
- Canonicalized `MassTable` perturbation, time, and sample keys as strings
  during duplicate detection, pooled/sample mode checks, and lookup,
  preventing validation/measure-build mismatches for numeric or categorical
  metadata.
- Added regression coverage for string-equivalent mass keys.

## 2.0.4

- Added explicit SDE noise tensors/generators for same-noise counterfactuals
  without mutating global RNG state.
- Rejected mixed pooled/sample-specific mass rows and malformed
  clamped-context tensors before they can produce misleading mass or context
  calculations.
- Strengthened count-matrix and probability validation and added per-time count
  likelihood logs.
- Centralized the pooled-sample sentinel with legacy compatibility, made
  sample-aware pooled-mass fallback conservative, added optional multi-time
  endpoint time-weight normalization, and rejected fractional count matrices in
  count-likelihood paths.

## 2.0.3

- Hardened core data validation for time axes, finite measures, cell-state
  tables, mass tables, and cross-table consistency.
- Added explicit sparse multi-time endpoint diagnostics for active/missing
  target keys, mean reduction, and geometry/mass loss components.
- Strengthened count-likelihood input validation and missing-checkpoint errors.
- Made counterfactual branches use shared stochastic noise by default and made
  clamped-context rollouts preserve the reference tau grid.
- Removed site-specific absolute setup paths from the portable package.

## 2.0.2

- Renamed and consolidated the public package under the `credo` namespace.
- Added multi-time trajectory primitives: `TrajectoryProblem`, sparse trajectory
  support, observed-time tau grids, checkpoint endpoint losses, and cumulative
  count fitness.
- Preserved the endpoint HNSCC training path while exposing backward-compatible
  two-timepoint views from trajectory problems.
- Fixed non-uniform time integration for count fitness and made package-level
  clamped-context counterfactuals operational.

This release provides the compact CREDO package and multi-time building blocks.
A dedicated production `TrajectoryTrainer` and LPS three-time runner remain a
separate integration step.
