# Changelog

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
- Removed experimental attention, search, expression-preprocessing, weak-form,
  cohort-loading, and legacy compatibility paths from the installed package.

The pre-compaction 2.0.1 implementation remains available at the
`pre-compaction-v2.0.1` Git tag.
