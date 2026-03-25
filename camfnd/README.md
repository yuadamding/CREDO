# CAMFND: Control-Anchored Mean-Field Neural Differential Equations

CAMFND is a compact Python package for learning perturbation dynamics from endpoint data using finite measures, particle simulators, and transport-based objectives.

The current codebase has three main layers:

1. a synthetic benchmark pipeline for validating the data contract, simulator, and benchmark-path models
2. a direct multidimensional full-path simulator and trainer for the newer full model
3. comparison and reporting utilities, including visualization and optional `scDiffEq` benchmarking helpers

## Core Workflows

| Workflow | Main entry point | Purpose |
| --- | --- | --- |
| Benchmark pipeline | `camfnd.pipeline.run_full_pipeline` | Run the compact four-phase synthetic benchmark |
| Visualization report | `camfnd.visualization.run_pipeline_with_visualizations` | Render a full figure report for the benchmark phases |
| Direct full-path evaluation | `camfnd.evaluation.evaluate_full_model` | Evaluate the multidimensional `full_joint_sim.py` path directly |
| Direct simulator case suite | `camfnd.evaluation.evaluate_full_joint_sim_cases` | Stress-test `FullJointSimulator` on analytic and invariance cases |
| External comparisons | `camfnd.evaluation.evaluate_camfnd_vs_scdiffeq_*` | Compare CAMFND against `scDiffEq` on adapted datasets |

## Installation

```bash
conda activate ml1
cd /home/yding1995/opscc_sc/tumor_evo
```

The package is importable directly from this tree; no separate install step is required in the current environment.

## Quick Start

### CLI

Run the full benchmark pipeline:

```bash
conda run -n ml1 python -m camfnd --output-dir ./outputs
```

Run a faster small-config version:

```bash
conda run -n ml1 python -m camfnd --fast --output-dir ./outputs
```

Run only selected benchmark phases:

```bash
conda run -n ml1 python -m camfnd --steps 1 2 --output-dir ./outputs
```

The CLI still accepts numeric `--steps` for backward compatibility, but outputs are saved under the semantic phase names:

- `data_contract`
- `simulator_validation`
- `single_screen_model`
- `multiscreen_context_model`

### Python API

```python
from camfnd.pipeline import run_full_pipeline
from camfnd.data import SingleScreenBenchmarkConfig, MultiscreenBenchmarkConfig
from camfnd.training import SingleScreenTrainConfig, MultiscreenContextTrainConfig

result = run_full_pipeline(
    single_screen_config=SingleScreenBenchmarkConfig(),
    multiscreen_config=MultiscreenBenchmarkConfig(),
    single_screen_train_config=SingleScreenTrainConfig(),
    multiscreen_train_config=MultiscreenContextTrainConfig(),
    output_dir="./outputs",
    verbose=True,
)

print(result.data_contract.ok)
print(result.simulator_validation.ok)
print(result.single_screen_model.ok)
print(result.multiscreen_context_model.ok)
print(result.all_pass)
```

### Visualization Pipeline

Run the benchmark and generate a visualization report in one call:

```python
from camfnd.visualization import run_pipeline_with_visualizations

viz = run_pipeline_with_visualizations(
    pipeline_output_dir="./outputs",
    visualization_output_dir="./visualizations",
)

print(viz.artifacts.report_path)
print(viz.artifacts.manifest_path)
```

Render figures from an existing `PipelineResult`:

```python
from camfnd.visualization import generate_pipeline_visualizations

artifacts = generate_pipeline_visualizations(result, "./visualizations")
print(artifacts.report_path)
```

The visualization design is documented in [`VISUALIZATION_PIPELINE.md`](VISUALIZATION_PIPELINE.md).

### Direct Full-Path Evaluation

The benchmark pipeline is intentionally compact and synthetic. The direct full-path evaluator targets the newer multidimensional simulator and coefficient model:

```python
from camfnd.evaluation import evaluate_full_model, evaluate_full_joint_sim_cases

full_eval = evaluate_full_model()
case_suite = evaluate_full_joint_sim_cases()

print(full_eval.ok)
print(case_suite.ok)
```

The detailed math-and-code walkthrough for the full simulator is in [`FULL_JOINT_SIMULATION_GUIDE.md`](FULL_JOINT_SIMULATION_GUIDE.md).

### Optional `scDiffEq` Comparisons

The repo also contains comparison utilities for adapted `scDiffEq` datasets:

```python
from camfnd.evaluation import (
    evaluate_camfnd_vs_scdiffeq_larry_4to6,
    evaluate_camfnd_vs_scdiffeq_additional_datasets,
)
```

These evaluators are optional benchmarking tools, not part of the core synthetic benchmark pipeline.

## Output Layout

When `run_full_pipeline(..., output_dir=...)` is used, CAMFND writes:

- `data_contract/`
- `simulator_validation/`
- `single_screen_model/`
- `multiscreen_context_model/`
- `pipeline_summary.json`

When `generate_pipeline_visualizations(...)` is used, CAMFND writes:

- `overview/`
- `data_contract/`
- `simulator_validation/`
- `single_screen_model/`
- `multiscreen_context_model/`
- `VISUALIZATION_REPORT.md`
- `visualization_manifest.json`

## Package Structure

```text
camfnd/
├── data/
│   ├── contract.py
│   ├── single_screen_benchmark.py
│   └── multiscreen_benchmark.py
├── numerics/
│   ├── euler_maruyama.py
│   ├── particles_np.py
│   ├── particles_torch.py
│   └── truth_coeffs.py
├── models/
│   ├── coeff_nets.py
│   ├── context_map.py
│   ├── embeddings.py
│   ├── full_coeff_nets.py
│   ├── full_context_map.py
│   ├── sinkhorn.py
│   └── time_embedding.py
├── simulation/
│   ├── single_screen_sim.py
│   ├── multiscreen_context_sim.py
│   └── full_joint_sim.py
├── training/
│   ├── single_screen_model.py
│   ├── multiscreen_context_model.py
│   └── full_model.py
├── evaluation/
│   ├── data_contract.py
│   ├── simulator_validation.py
│   ├── single_screen_model.py
│   ├── multiscreen_context_model.py
│   ├── full_model.py
│   ├── full_joint_sim_cases.py
│   ├── scdiffeq_larry.py
│   ├── scdiffeq_larry_4to6_compare.py
│   └── scdiffeq_additional_datasets.py
├── visualization/
│   ├── __init__.py
│   └── pipeline.py
├── pipeline.py
├── cli.py
└── scripts/
    └── fetch_scdiffeq_datasets.sh
```

## Conceptual Model

### Finite Measures

The fundamental data object is a finite measure

\[
\mu = \sum_i w_i \delta(z_i),
\]

where:

- `z_i` are latent-state support points
- `w_i > 0` are guide-abundance weights
- total mass can vary across perturbations and screens

This separates biological abundance from raw cell count and is the core data-contract idea in [`data/contract.py`](data/contract.py).

### Benchmark Path Dynamics

The compact synthetic benchmark uses a 1D Ornstein-Uhlenbeck-style reference process:

\[
dz = \kappa(\theta_g - z)\,dt + \sigma_g\,dW,
\]
\[
d(\log w) = \rho_g\,dt.
\]

The multiscreen benchmark adds a screen-level context term:

\[
dz = \kappa(\theta_g - z)\,dt + \eta\,c_{s,t}\,dt + \sigma_g\,dW.
\]

This path is implemented by:

- [`simulation/single_screen_sim.py`](simulation/single_screen_sim.py)
- [`simulation/multiscreen_context_sim.py`](simulation/multiscreen_context_sim.py)

### Full Path Dynamics

The direct full model generalizes to multidimensional latent states and a learned sample-level mean-field context:

- simulator: [`simulation/full_joint_sim.py`](simulation/full_joint_sim.py)
- coefficient model: [`models/full_coeff_nets.py`](models/full_coeff_nets.py)
- context map: [`models/full_context_map.py`](models/full_context_map.py)

This is the path evaluated by [`evaluation/full_model.py`](evaluation/full_model.py), not by the compact benchmark pipeline.

### Control-Anchored Coefficients

Controls are hard-anchored at zero in the perturbation embedding space, so learned perturbation effects remain interpretable relative to control.

### Losses

CAMFND uses transport-based endpoint losses together with moment penalties:

- unbalanced Sinkhorn divergence or normalized geometry loss
- auxiliary mass penalty
- auxiliary mean penalty
- auxiliary variance penalty
- for the multiscreen benchmark, an additional screen-delta mean penalty

## Built-In Evaluation Suites

### Benchmark Pipeline Phases

- `evaluate_data_contract`
  - checks structural validity and benchmark signatures
- `evaluate_simulator_validation`
  - checks initialization exactness, parity with analytic truth, and convergence
- `evaluate_single_screen_model`
  - trains the single-screen model and required ablations
- `evaluate_multiscreen_context_model`
  - trains the context-aware model and the no-context ablation

### Direct Full-Path Validation

- `evaluate_full_model`
  - direct training-and-evaluation harness for the full multidimensional path
- `evaluate_full_joint_sim_cases`
  - analytic, invariance, and feedback test suite for `FullJointSimulator`

### External Comparisons

- `evaluate_camfnd_on_scdiffeq_larry`
- `evaluate_camfnd_vs_scdiffeq_larry_4to6`
- `evaluate_camfnd_vs_scdiffeq_larry_4to6_cv`
- `evaluate_camfnd_vs_scdiffeq_additional_datasets`

These rely on additional external data and are intended for comparative benchmarking rather than core package validation.

## Configuration Guide

Preferred public names are the semantic aliases:

- `SingleScreenBenchmarkConfig`
- `MultiscreenBenchmarkConfig`
- `SingleScreenTrainConfig`
- `MultiscreenContextTrainConfig`

The historical `Stage1*` and `Stage2*` names are still exported for compatibility.

### Single-Screen Benchmark

`SingleScreenBenchmarkConfig` controls:

- seed and sample id
- number of observed cells at `P4` and `P60`
- OU initial mean and variance
- shared `kappa`
- whether to infer and store a latent transform

Default synthetic perturbations:

- `ctrl`
- `drift`
- `diff`
- `react`

### Multiscreen Benchmark

`MultiscreenBenchmarkConfig` adds:

- two screen ids
- truth-particle count
- Euler-Maruyama step count
- context coupling strength `eta`
- screen-specific initial masses for the `driver` perturbation
- context-map configuration

Default synthetic perturbations:

- `ctrl`
- `drift`
- `diff`
- `react`
- `driver`

### Training Configs

`SingleScreenTrainConfig` / `Stage1TrainConfig`

- benchmark-path single-screen training
- controls network width/depth, number of simulation steps, OT hyperparameters, penalties, and optimizer settings

`MultiscreenContextTrainConfig` / `Stage2TrainConfig`

- benchmark-path multiscreen training
- adds `use_context` and `aux_screen_delta_mean_weight`

`FullModelTrainConfig`

- multidimensional full-path training
- adds learned context-map dimensions and hidden sizes for the full model

All training configs support:

- `device="auto" | "cpu" | "cuda"`
- `dtype="float32" | "float64"`

## CUDA Notes

Training configs use `device="auto"` by default and will select CUDA when available.

Because the synthetic benchmarks use stochastic simulation and floating-point transport losses, CPU and CUDA runs can differ slightly even with the same seed. This is expected; the main effect is usually on tight threshold-based acceptance checks, not on overall model validity.

## Notes

- The benchmark pipeline is intentionally small and synthetic. It is meant to validate mechanics, not to replace the direct full-path evaluator.
- The direct full path is the combination of [`simulation/full_joint_sim.py`](simulation/full_joint_sim.py), [`training/full_model.py`](training/full_model.py), and [`evaluation/full_model.py`](evaluation/full_model.py).
- The visualization package is report-oriented and writes static images plus a generated markdown summary.
- `scDiffEq` comparison utilities assume the external `scDiffEq` environment and downloaded datasets are available.
