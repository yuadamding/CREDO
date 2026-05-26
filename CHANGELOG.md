# Changelog

## 2.0.9

- Marked the package release as `credo==2.0.9`.
- Preserved the 2.0.8 correctness-hardening state with soft-reference,
  explicit mass-mode, semantic-invariant, and trajectory provenance fixes.
- Hardened trajectory runner mass semantics so explicit `group_total` and
  `per_cell_contribution` modes fail when `--mass-col` is absent, while
  `auto` rejects any constant multi-cell group instead of silently guessing.
- Made fallback VAE gene selection use the VAE fitting subset when source-only
  fitting is requested, preventing target-time expression from choosing genes.
- Added provenance fields for resolved mass mode, AnnData shape/schema,
  dependency versions, CUDA details, git dirty status, and var-name hashes.
- Added semantic regressions for soft-reference counterfactual embedding
  context, extreme mass-faithful context values, strict mass-mode behavior,
  and VAE source-only fallback gene selection.
- Added `weighted_mean_shift_l2_fact_vs_ref` as the explicit name for the
  current counterfactual mean-shift metric while retaining the legacy
  `geom_shift_fact_vs_ref` alias for compatibility.
- Added biological interpretation gates to the HNSCC effect summary: fold
  stability, same-gene guide concordance, negative-control null gap,
  counterfactual replicate support, plasticity/diffusion stability, and
  context-ablation evidence are now emitted alongside priority classes.
- Added counterfactual provenance fields and a `counterfactual_manifest.json`
  that records same-start/same-noise, reference-consistent control rollout,
  context-clamping, and the weighted-mean geometry metric convention.

## 2.0.8

- Marked the package release as `credo==2.0.8`.
- Preserved the stabilized trajectory-trainer package state from the 2.0.6
  hardening pass.
- Fixed soft-reference effective embeddings so controls receive exactly the
  learned reference embedding and non-controls receive reference plus residual.
- Added explicit trajectory runner mass modes (`count`,
  `per_cell_contribution`, `group_total`) and made ambiguous constant
  mass-column usage fail until the user declares semantics.
- Added semantic invariant tests for soft-reference controls, mass-faithful
  ecological context, stabilized log-weight invariance, and non-uniform
  finite-measure particle initialization.
- Added rank/score-aware VAE gene selection, fp32 weak/count loss evaluation
  under mixed precision, physical-time columns in trajectory prediction tables,
  and a lightweight `run_manifest.json`.

## 2.0.6

- Marked the package release as `credo==2.0.6`.
- Added the first production trajectory training stack: `TrajectoryView`,
  sample-aware trajectory particle initialization, `TrajectoryTrainer`,
  per-key/time trajectory evaluation tables, and generic/LPS trajectory
  runners.
- Added a reproducible LPS 90m/6h/10h trajectory input builder and generated
  `../inputs/LPS/credo_lps_90m_6h_10h_celltype.h5ad` from the local private
  LPS AnnData.
- Added same-start, same-noise trajectory counterfactuals with time-indexed
  checkpoint metrics.
- Extended multi-time endpoint loss plumbing to separate prediction
  `measure_key`s from perturbation `embedding_id`s while preserving the
  legacy positional endpoint-loss API.
- Added VAE latent support to the generic trajectory runner for count-only
  trajectory inputs.
- Canonicalized exposure/count table keys and made pooled finite-measure
  geometry sample-mass-weighted when sample-specific masses are present.
- Canonicalized `MassTable` perturbation, time, and sample keys as strings
  during duplicate detection, pooled/sample mode checks, and lookup,
  preventing validation/measure-build mismatches for numeric or categorical
  metadata.
- Added regression coverage for sparse donor-aware trajectory training, LPS
  runner smoke tests, trajectory counterfactuals, and string-equivalent mass
  keys.
- Added a production-layer randomized stress harness covering sparse
  donor-aware trajectory views, measure-key/embedding-id separation,
  checkpoint endpoint diagnostics, trajectory counterfactuals, and one-epoch
  trainer smoke cases.

## 2.0.5

- Hardened mass-key canonicalization, pooled/sample mass semantics, count
  validation, multi-time endpoint time-weight normalization, and same-noise
  SDE utilities.

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

In 2.0.2, this provided the compact CREDO package and multi-time building
blocks. The dedicated production `TrajectoryTrainer` and LPS three-time runner
were added later in 2.0.6.
