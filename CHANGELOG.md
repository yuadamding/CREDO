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
