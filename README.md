# CREDO

Compact CREDO package for endpoint and multi-time immune trajectory modeling.
The repository contains the installable `credo` Python package, one HNSCC
training runner, post-training analysis utilities, setup scripts, and tests.
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
../inputs/LPS/credo_lps_90m_to_6h_celltype.h5ad
../inputs/LPS/credo_lps_90m_to_10h_celltype.h5ad
../inputs/LPS/credo_lps_0h_to_90m_celltype.h5ad
```

## Install

```bash
CONDA_BIN=/rsrch8/home/bcb/$USER/miniforge3/bin/conda \
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

Run the focused test suite:

```bash
pytest -q
```

Run randomized trajectory stress tests:

```bash
python scripts/stress_test_trajectory_core.py --cases 1000
```

## Public Imports

```python
from credo.data.problems import EndpointProblem, TrajectoryProblem
from credo.losses.trajectory import MultiTimeEndpointLoss
from credo.models import FullDynamicsModel
from credo.models.particles import WeightedParticleSimulator
from credo.training import Trainer
```

The two-endpoint API remains compatible with the HNSCC runner. The multi-time
surface adds trajectory problems, observed-time rollout grids, checkpointed
endpoint losses, and cumulative count fitness without changing default
two-endpoint behavior.
