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
