# CREDO

Compact CREDO package for endpoint and multi-time immune trajectory modeling.
The repository contains the installable `credo` Python package, the legacy
HNSCC endpoint runner, the production full-start trajectory trainer, generic
and LPS trajectory runners, post-training analysis utilities, setup scripts,
and tests.
It intentionally does not contain `.h5ad` data, checkpoints, or run outputs.

## Layout

| Path | Purpose |
| --- | --- |
| `package/` | Installable Python package. Public imports live under `credo`. |
| `runners/` | Training and CV-summary Python entry points. |
| `analysis/` | Post-training biology and signature utilities. |
| `scripts/` | Install, setup verification, and randomized stress testing. |
| `tests/` | Unit, regression, and randomized simulation tests. |
| `env/` | Minimal conda environment definition. |

Generated outputs belong in `runs/`, `outputs/`, `results/`, or `models/`.
Those paths are ignored and should stay outside portable source snapshots.

## Inputs

Use paths relative to this repository:

```text
../inputs/hnscc/GSE235325_P4P60_allgenes_allcells_latest_states.h5ad
../inputs/LPS/credo_lps_90m_6h_10h_celltype.h5ad
../inputs/LPS/credo_lps_90m_to_6h_celltype.h5ad
../inputs/LPS/credo_lps_90m_to_10h_celltype.h5ad
../inputs/LPS/credo_lps_0h_to_90m_celltype.h5ad
```

## Install

```bash
bash scripts/install_bundle.sh credo-hnscc
```

For an existing environment:

```bash
python -m pip install --no-cache-dir -e package
python scripts/verify_setup.py
```

## Run

Minimal HNSCC smoke run:

```bash
python runners/run_credo_hnscc_full.py \
  --data-path ../inputs/hnscc/GSE235325_P4P60_allgenes_allcells_latest_states.h5ad \
  --output-dir runs/hnscc_smoke \
  --latent-source pca \
  --epochs 2 \
  --split-strategy random \
  --train-frac 0.8 \
  --n-particles 32 \
  --n-steps 4 \
  --eval-particles 64 \
  --eval-steps 6
```

Summarize completed CV runs:

```bash
python runners/summarize_hnscc_cv.py \
  --cv-root runs/<run_root> \
  --output-dir runs/<run_root>/summary
```

Minimal LPS three-time trajectory smoke run:

```bash
python runners/run_credo_lps_3time.py \
  --data-path ../inputs/LPS/credo_lps_90m_6h_10h_celltype.h5ad \
  --output-dir runs/lps_trajectory_smoke \
  --mass-mode per_cell_contribution \
  --latent-source vae \
  --vae-layer counts \
  --vae-latent-dim 8 \
  --vae-hidden-dim 64 \
  --vae-epochs 1 \
  --expression-top-genes 128 \
  --epochs 2 \
  --n-particles 32 \
  --steps-per-interval 2 \
  --ecology-off
```

For the generic trajectory runner, finite-measure mass semantics are explicit:

```text
--mass-mode count                 # use captured cell counts
--mass-mode group_total           # one group-level mass value repeated on cells
--mass-mode per_cell_contribution # per-cell mass contributions that should sum
```

The default `--mass-mode auto` refuses any constant multi-cell mass group and
asks you to choose `group_total` or `per_cell_contribution`. Explicit
`group_total` and `per_cell_contribution` modes require `--mass-col`; only
`count` intentionally ignores the mass column.

Run the focused test suite:

```bash
pytest -q
```

Run randomized trajectory stress tests:

```bash
python scripts/stress_test_trajectory_core.py --cases 1000
python scripts/stress_test_trajectory_production.py \
  --cases 1000 \
  --counterfactual-cases 300 \
  --trainer-cases 100
```

Biology summary tables include conservative interpretation gates. A ranked hit
is marked `claim-ready` only when evidence clears counterfactual replicate
support, fold-stability, same-gene guide-concordance, metric-specific
negative-control null gaps, explicit mass-mode metadata when available, and
claim-specific counterfactual checks. Single guide genes are reported as
`not_assessable` for guide concordance, not as a pass. Ecology-dependent calls
require replicated context-ablation evidence; plasticity/state-shift calls
require stable diffusion/action evidence and state-shift null support. The
table records the missing condition, such as
`needs-counterfactual-replicates`, `needs-fold-stability`,
`needs-guide-concordance`, `needs-explicit-mass-mode`, `missing-mass-null`,
`below-context-null-gap`, or `needs-context-ablation`, and it also emits axis-specific columns such as
`expansion_claim_ready`, `plasticity_claim_ready`, and `ecology_claim_ready`.

## Public Imports

```python
from credo.data.problems import EndpointProblem, TrajectoryProblem
from credo.data import TrajectoryView
from credo.losses.trajectory import MultiTimeEndpointLoss
from credo.models import FullDynamicsModel
from credo.models.particles import WeightedParticleSimulator
from credo.training import Trainer, TrajectoryTrainer
```

The two-endpoint API remains compatible with the HNSCC runner. The trajectory
stack is separate: it consumes `TrajectoryProblem` or `SparseTrajectoryProblem`,
keeps sample-aware `measure_key`s separate from string perturbation
`embedding_id`s, rolls out one global-time path from a source label, evaluates
checkpointed finite-measure losses at downstream target labels, and writes
per-key/time prediction tables.

## Semantic Guarantees

- In `soft_ref` mode, controls have effective embedding equal to the learned
  shared reference, and non-controls have reference plus residual.
- Counterfactuals use the same source finite measure and particle seed; the
  reference branch removes the perturbation residual rather than swapping in a
  control initial population. The HNSCC counterfactual manifest computes
  same-start and same-noise flags from tensor checksums.
- Ecological context is computed from absolute particle weights, including
  `log_m0`, so expansion and depletion affect global context.
- Endpoint fitting uses finite-measure geometry plus log-mass consistency; the
  endpoint quantity is a Sinkhorn-geometry-plus-log-mass proxy, not a full
  KL-relaxed unbalanced OT solver.
- Trajectory counterfactual output names the current distribution summary as
  `weighted_mean_shift_l2_fact_vs_ref`; the older `geom_shift_fact_vs_ref`
  column is retained as a compatibility alias.
