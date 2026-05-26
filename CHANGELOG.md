# Changelog

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
