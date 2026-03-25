# CAMFND: Control-Anchored Mean-Field Neural Differential Equations

A four-step computational framework for analyzing tumor cell evolution from Perturb-seq data. CAMFND models cell populations as finite measures evolving under stochastic differential dynamics, learning perturbation-specific drift, diffusion, and growth coefficients from paired initial/terminal snapshots (P4 → P60 developmental stages).

## Overview

Perturb-seq experiments produce paired cell-state snapshots at two time points across multiple perturbation conditions. CAMFND treats each condition's cell population as a **finite measure** — a discrete approximation carrying both support atoms (cell positions in latent space) and weights (guide-abundance masses) — and learns the SDE coefficients that best transport the initial measure to the terminal one.

The pipeline is organized as four sequential steps:

| Step | Name | Description |
|------|------|-------------|
| **1** | Data Contract + Benchmark | Define the finite-measure data structure; generate the Stage-I benchmark dataset |
| **2** | Euler-Maruyama Simulator | Verify ground-truth SDE dynamics via particle simulation |
| **3** | Stage-I Learnable Model | Learn perturbation-specific coefficients with unbalanced OT loss (no screen context) |
| **4** | Stage-II Context-Aware Model | Extend Stage-I with screen-level occupancy context coupling between perturbations |

## Installation

```bash
conda activate ml1
cd /home/yding1995/opscc_sc/tumor_evo
```

The `camfnd` package is importable directly from this directory (no separate install required). All dependencies are available in the `ml1` conda environment.

## Quick Start

### Run the full pipeline

```bash
conda run -n ml1 python3 -m camfnd --output-dir ./outputs
```

For a fast iteration (smaller benchmark configs):

```bash
conda run -n ml1 python3 -m camfnd --fast --output-dir ./outputs
```

Run only specific steps:

```bash
conda run -n ml1 python3 -m camfnd --steps 1 2 --output-dir ./outputs
```

### Python API

```python
from camfnd.pipeline import run_full_pipeline

result = run_full_pipeline(output_dir="./outputs", verbose=True)
print(
    result.data_contract.ok,
    result.simulator_validation.ok,
    result.single_screen_model.ok,
    result.multiscreen_context_model.ok,
)
```

### Visualization API

```python
from camfnd.visualization import run_pipeline_with_visualizations

viz = run_pipeline_with_visualizations(
    pipeline_output_dir="./outputs",
    visualization_output_dir="./visualizations",
)
print(viz.artifacts.report_path)
```

To render figures from an existing `PipelineResult`:

```python
from camfnd.visualization import generate_pipeline_visualizations

artifacts = generate_pipeline_visualizations(result, "./visualizations")
print(artifacts.manifest_path)
```

The figure design is documented in [`VISUALIZATION_PIPELINE.md`](/home/yding1995/opscc_sc/tumor_evo/camfnd/VISUALIZATION_PIPELINE.md).

## Package Structure

```
camfnd/
├── data/
│   ├── contract.py          # Core data structures: FiniteMeasure, EndpointProblem, PerturbSeqDynamicsData
│   ├── single_screen_benchmark.py  # Single-screen benchmark generator (4 perturbations, closed-form OU)
│   └── multiscreen_benchmark.py    # Multi-screen benchmark generator (5 perturbations, joint dynamics)
├── numerics/
│   ├── particles_np.py      # NumPy ParticleState for deterministic simulation
│   ├── particles_torch.py   # PyTorch TorchParticleState for learnable simulation
│   ├── truth_coeffs.py      # Ground-truth SDE coefficient dataclass (Stage1SDECoefficients)
│   └── euler_maruyama.py    # Euler-Maruyama integrator (Step 2 reference simulator)
├── models/
│   ├── embeddings.py        # ControlAnchoredEmbeddingStore (exact zero anchor for controls)
│   ├── time_embedding.py    # Fourier time feature map
│   ├── sinkhorn.py          # Unbalanced Sinkhorn OT divergence loss
│   ├── context_map.py       # OccupancyContextMap (soft right-occupancy screen summary)
│   └── coeff_nets.py        # ControlAnchoredStage1Model / ControlAnchoredStage2Model
├── simulation/
│   ├── single_screen_sim.py         # LearnedStage1Simulator (no context)
│   └── multiscreen_context_sim.py   # LearnedStage2JointSimulator (with context)
├── training/
│   ├── single_screen_model.py       # Stage1TrainConfig, train_stage1_model
│   └── multiscreen_context_model.py # Stage2TrainConfig, train_stage2_model
├── evaluation/
│   ├── data_contract.py              # Data contract acceptance checks
│   ├── simulator_validation.py       # Simulator exactness, stability, convergence
│   ├── single_screen_model.py        # Single-screen model benchmark + ablations
│   └── multiscreen_context_model.py  # Multi-screen model vs no-context ablation
├── visualization/
│   ├── __init__.py                   # Visualization exports
│   └── pipeline.py                   # Figure builders + report pipeline
├── pipeline.py              # run_full_pipeline() end-to-end runner
└── cli.py                   # python -m camfnd CLI
```

## Core Concepts

### Finite Measures

The fundamental data unit is a **finite measure** `μ = Σᵢ wᵢ δ(zᵢ)` where `zᵢ` are latent-space cell positions and `wᵢ > 0` are guide-abundance weights (not cell counts). The total mass `M = Σᵢ wᵢ` can differ across perturbations, capturing guide-induced changes in cell proliferation or death.

### Stochastic Dynamics

Cells evolve under an Ornstein-Uhlenbeck SDE:

```
dz  = κ(θ_g − z) dt + σ_g dW          (position)
d(log w) = ρ_g dt                       (mass growth)
```

Parameters are perturbation-specific (`_g` subscript) with shared mean-reversion rate `κ`. Stage-II adds a screen-level context term:

```
dz = κ(θ_g − z) dt + η · c_{s,t} dt + σ_g dW
```

where `c_{s,t}` is the soft right-occupancy of the screen computed across all perturbations.

### Control-Anchored Embeddings

Each perturbation is represented by a learnable embedding `a_g ∈ Rᵈ`. All **control** perturbations are hard-fixed to `a_g = 0`, making coefficient differences interpretable relative to the unperturbed baseline. Non-controls receive a one-hot initialization when `d ≥ n_perturbations`.

### Loss Function

Training minimizes an **unbalanced Sinkhorn divergence** (UOT) between predicted terminal particles and observed terminal measures, plus auxiliary moment penalties on mass, mean, and variance:

```
L = UOT(μ̂_terminal, μ_terminal) + λ_mass·(M̂−M)² + λ_mean·(m̂−m)² + λ_var·(v̂−v)²
```

Stage-II adds a screen-delta loss penalizing errors in the cross-screen mean shift per perturbation.

## Benchmarks

### Stage-I Benchmark

Four perturbations on a single screen: `ctrl`, `drift` (shifted mean), `diff` (increased diffusion), `react` (reduced mass). Ground-truth uses closed-form OU terminal moments.

**Acceptance criteria (Step 1):**
- `drift` terminal mean > `ctrl` terminal mean
- `diff` terminal variance > `ctrl` terminal variance
- `react` terminal mass < `ctrl` terminal mass

**Ablations tested (Step 3):**
- `no_growth`: growth field disabled — fails on `react` mass recovery
- `shared_diffusion`: single diffusion across all perturbations — fails on `diff` variance
- `normalized_only`: normalized OT loss (ignores mass) — fails on `react` mass

### Stage-II Benchmark

Five perturbations across two screens. Screens differ only in the initial mass of the `driver` perturbation, which modulates the screen-level occupancy context. The context-aware model (`use_context=True`) recovers the cross-screen mean shift; the ablation (`use_context=False`) fails.

## CUDA Acceleration

CAMFND automatically uses CUDA when available. Both training configs default to `device="auto"`:

```python
from camfnd.training.single_screen_model import SingleScreenTrainConfig

# Auto-selects CUDA if available, falls back to CPU
config = SingleScreenTrainConfig()
print(config.resolved_device)  # "cuda" or "cpu"

# Force a specific device
config_cpu = SingleScreenTrainConfig(device="cpu")
config_gpu = SingleScreenTrainConfig(device="cuda")
```

Observed speedup on a single GPU (256-cell benchmark, float64): **~1.8× vs CPU**.
Speedup scales with dataset size and `particles_per_atom`.

> **Note:** CUDA float64 arithmetic introduces different rounding than CPU for the same seed. Step 4's stochastic threshold tests (designed for CPU) may differ on CUDA; the underlying model training is numerically correct on both devices.

## Evaluation Results

All 17 tests pass with CPU parity against the original separate packages:

```
Steps 1–2 (data + simulator):  9/9  PASS
Step 3 (Stage-I model):         4/4  PASS
Step 4 (Stage-II model):        4/4  PASS
```

## Training Configuration

### Stage-I (`Stage1TrainConfig`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 60 | Training epochs |
| `lr` | 0.05 | Adam learning rate |
| `embedding_dim` | 3 | Perturbation embedding dimension |
| `n_steps` | 16 | Euler-Maruyama steps during training |
| `epsilon` | 0.08 | Sinkhorn regularization |
| `tau` | 0.45 | UOT marginal penalty |
| `device` | `"auto"` | `"auto"` / `"cpu"` / `"cuda"` |

### Stage-II (`Stage2TrainConfig`)

Inherits all Stage-I parameters plus:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 40 | Training epochs |
| `use_context` | `True` | Enable screen-level context coupling |
| `aux_screen_delta_mean_weight` | 10.0 | Weight on cross-screen delta loss |
| `embedding_dim` | 4 | Larger embedding for 5-perturbation catalog |

## Project Background

This framework was developed to analyze OROPHARYNGEAL SQUAMOUS CELL CARCINOMA (OPSCC) tumor evolution through single-cell Perturb-seq experiments. The four-step design separates concerns:

1. **Data modeling** (Step 1) ensures the finite-measure contract is satisfied before any learning occurs.
2. **Ground-truth simulation** (Step 2) validates the SDE dynamics independently of the neural components.
3. **Single-screen learning** (Step 3) establishes that perturbation-specific coefficients can be recovered without cross-screen information.
4. **Multi-screen context learning** (Step 4) demonstrates that screen-level occupancy context is necessary and sufficient to explain cross-screen variation.
