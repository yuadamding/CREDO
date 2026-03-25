# CAMFND Visualization Pipeline

This document describes a comprehensive visualization method for each benchmark phase in `camfnd` and points to the implementation that renders the figures automatically.

## Design Goals

The visual report is designed to answer four questions:

1. Is the benchmark dataset structurally correct?
2. Is the trusted simulator numerically correct and convergent?
3. Does the single-screen learnable model recover the intended perturbation effects, and do the ablations fail in the right places?
4. Does the multiscreen context model recover cross-screen coupling and outperform the no-context ablation?

## Phase 1: Data Contract

Figures:

- `counts_and_mass`
  - grouped bars for cell counts and guide-abundance masses by endpoint group
- `terminal_moments`
  - terminal mean, variance, and mass by perturbation group
- `analytic_parity`
  - empirical vs analytic parity for mean, variance, and mass
- `signature_checks`
  - pass/fail checklist for the benchmark semantics

## Phase 2: Simulator Validation

Figures:

- `initialization_exactness`
  - exact reconstruction flag for the initial particle state
- `default_run_parity`
  - simulated vs analytic parity for the default Euler-Maruyama run
- `convergence`
  - error vs `n_steps` for mean, variance, and mass
- `default_run_errors`
  - per-perturbation error profile for the default run

## Phase 3: Single-Screen Model

Figures:

- `training_dynamics`
  - total, endpoint, auxiliary, and regularization losses for the full model and ablations
- `full_model_parity`
  - full-model predicted vs truth terminal mean, variance, and mass
- `ablation_metrics`
  - compact scorecard over the summary metrics used in acceptance testing
- `signature_gaps`
  - effect-size gaps for drift, diffusion, and reaction signatures

## Phase 4: Multiscreen Context Model

Figures:

- `training_dynamics`
  - full vs `no_context` optimization traces
- `screen_delta_comparison`
  - predicted `screen2 - screen1` mean shifts against the truth
- `context_trajectories`
  - learned context trajectories with truth overlaid when available
- `terminal_parity`
  - full and no-context predictions against terminal truth moments

## Overview Figure

The renderer also creates `overview/benchmark_overview.png`, which summarizes:

- phase-level pass/fail state
- single-screen full-model errors against thresholds
- simulator convergence trend
- multiscreen screen-delta error comparison

## Implementation

The implementation lives in:

- [`visualization/pipeline.py`](/home/yding1995/opscc_sc/tumor_evo/camfnd/visualization/pipeline.py)
- [`visualization/__init__.py`](/home/yding1995/opscc_sc/tumor_evo/camfnd/visualization/__init__.py)

Main entry points:

- [`generate_pipeline_visualizations(...)`](/home/yding1995/opscc_sc/tumor_evo/camfnd/visualization/pipeline.py)
  - render figures from an existing [`PipelineResult`](/home/yding1995/opscc_sc/tumor_evo/camfnd/pipeline.py)
- [`run_pipeline_with_visualizations(...)`](/home/yding1995/opscc_sc/tumor_evo/camfnd/visualization/pipeline.py)
  - run the benchmark pipeline and render the report in one step

## Typical Usage

```python
from camfnd.visualization import run_pipeline_with_visualizations

viz = run_pipeline_with_visualizations(
    pipeline_output_dir="./outputs",
    visualization_output_dir="./visualizations",
)

print(viz.artifacts.report_path)
print(viz.artifacts.manifest_path)
```

If the pipeline has already been run:

```python
from camfnd.pipeline import run_full_pipeline
from camfnd.visualization import generate_pipeline_visualizations

result = run_full_pipeline(output_dir="./outputs", verbose=False)
artifacts = generate_pipeline_visualizations(result, "./visualizations")
```

## Output Layout

The renderer writes:

- `overview/benchmark_overview.png`
- `data_contract/*.png`
- `simulator_validation/*.png`
- `single_screen_model/*.png`
- `multiscreen_context_model/*.png`
- `VISUALIZATION_REPORT.md`
- `visualization_manifest.json`
