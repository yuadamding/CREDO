# Changelog

## 2.0.1

- Added schema-profile support to `credo-validate-data`, including stricter
  endpoint/trajectory checks, richer AnnData shape/count reports, required
  column null/empty validation, latent row-count validation, and finite latent
  checks for compact embeddings.
- Renamed endpoint evaluation output toward `endpoint_geom_mass` while keeping
  `uot` compatibility aliases for this patch release.
- Added ESS claim-gate payloads to endpoint evaluation with strict and lenient
  claim-readiness flags, threshold provenance, and clearer row-level ESS
  diagnostics.
- Upgraded run manifests to schema v2 with command, working directory, output
  directory, config hash, and explicit git availability/dirty-state reporting.
- Routed setup verification scripts through the package AnnData schema
  validator so install/data smoke checks match `credo-validate-data`.
- Hardened single-time CREDO with explicit guide-level view keys, fixed-particle
  context recomputation policy, vector-valued effect regularizers, and a compact
  single-time runner.
- Added `single_time` setup-verification schema support, single-time runner
  strict schema validation, guide-view pooling warnings, and post-training
  single-time biological effect artifacts for effect, endpoint, guide
  concordance, and control-null diagnostics.
- Added a compact `single_time.effect_vector_components` policy so single-time
  control-null and guide-concordance regularizers can include mass, latent mean
  shift, and optional latent variance shift components.
- Made single-time reporting more claim-aware with training/report view-level
  labels, context-policy metadata, diagnostic versus claim-grade mass aliases,
  guide-concordance evaluability flags, signed latent-shift tables, control-null
  summaries, particle-weight diagnostics, and run provenance files.
- Extended single-time effect artifacts with explicit deprecated-alias semantics
  for `delta_log_mass`/`delta_mass`, factual/reference endpoint fit deltas,
  source/factual/reference weight diagnostics, guide-concordance claimability
  flags, and control-null z-score/p95 annotations.
- Made strict single-time runner schema validation column-map-aware so custom
  control, guide, target-gene, sample, and batch columns can be validated before
  problem construction.
- Kept compatibility shims and tests for existing runner manifests, package
  imports, endpoint summaries, and CREDO semantic invariants.

## 2.0

- Marked the compact package release as `credo==2.0`.
- Consolidated the endpoint and production trajectory stacks under the `credo`
  package while preserving the legacy HNSCC endpoint runner.
- Included the full-start multi-time trajectory trainer, donor-aware
  `measure_key` versus perturbation `embedding_id` separation, observed-time
  rollout grids, checkpointed finite-measure losses, cumulative count
  likelihoods, and generic/LPS trajectory runners.
- Hardened finite-measure data handling: string-canonicalized identity keys,
  explicit pooled/sample mass semantics, sample-mass-weighted pooled geometry,
  strict mass modes, and source-only VAE gene-selection provenance.
- Preserved soft-reference control semantics and same-start/same-noise
  counterfactuals with simulator-consumed noise provenance, context-clamped
  branches, and time-indexed trajectory outputs.
- Added conservative biology reporting gates: explicit requested mass mode,
  metric-specific null floors/profiles, counterfactual replicate support,
  guide concordance, fold/run support, optional sample/patient support,
  positive signed program gates, expansion/depletion separation, and distinct
  priority-class versus axis-specific claim readiness.
- Kept randomized trajectory stress harnesses and focused semantic regressions
  for trajectory views, counterfactuals, mass semantics, mixed precision,
  package imports, LPS runner smoke tests, and biological reporting gates.
