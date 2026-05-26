# CREDO

Compact CREDO `2.0` package for endpoint and multi-time finite-measure
Perturb-seq trajectory modeling. The source snapshot contains the installable
`credo` package, runners, post-training analysis utilities, setup scripts, and
tests. It intentionally excludes `.h5ad` data, checkpoints, and run outputs.

## Layout

| Path | Purpose |
| --- | --- |
| `package/` | Installable Python package under `credo`. |
| `runners/` | HNSCC endpoint, generic trajectory, LPS trajectory, and summary entry points. |
| `analysis/` | Counterfactual biology and signature utilities. |
| `scripts/` | Setup verification, LPS input building, and randomized stress tests. |
| `tests/` | Unit, regression, smoke, and randomized tests. |

Generated outputs belong in `runs/`, `outputs/`, `results/`, or `models/`.

## Install

```bash
bash scripts/install_bundle.sh credo-hnscc
python -m pip install --no-cache-dir -e package
python scripts/verify_setup.py
```

## Smoke Runs

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

python runners/run_credo_lps_3time.py \
  --data-path ../inputs/LPS/credo_lps_90m_6h_10h_celltype.h5ad \
  --output-dir runs/lps_trajectory_smoke \
  --mass-mode per_cell_contribution \
  --latent-source vae \
  --vae-layer counts \
  --vae-epochs 1 \
  --epochs 2 \
  --n-particles 32 \
  --steps-per-interval 2 \
  --ecology-off
```

Trajectory mass semantics are explicit:

```text
--mass-mode count                 # captured cell counts
--mass-mode group_total           # one repeated group-level mass value
--mass-mode per_cell_contribution # per-cell weights that should sum
```

`auto` refuses ambiguous constant multi-cell mass groups. Claim-grade biology
summaries should pass `--claim-grade` with either explicit floor JSON or
`--practical-null-floor-profile hnscc_claim_grade`; the applied profile is
written to `practical_null_floors_used.json`.

## Verify

```bash
pytest -q
python scripts/stress_test_trajectory_core.py --cases 1000
python scripts/stress_test_trajectory_production.py \
  --cases 1000 \
  --counterfactual-cases 300 \
  --trainer-cases 100
```

## Public Imports

```python
from credo.data import EndpointProblem, TrajectoryProblem, TrajectoryView
from credo.losses import MultiTimeEndpointLoss
from credo.models import FullDynamicsModel, WeightedParticleSimulator
from credo.training import Trainer, TrajectoryTrainer
```

Compatibility facades such as `credo.data.problems`,
`credo.losses.trajectory`, and `credo.models.particles` remain available.

## Semantic Guarantees

- `soft_ref` controls use the learned reference embedding; non-controls use
  reference plus residual.
- Counterfactuals are same-start and same-noise: the reference branch removes
  the perturbation residual instead of swapping in control initial cells.
- Ecological context is computed from absolute particle weights, including
  source mass offsets.
- Multi-time training consumes `TrajectoryProblem` or `SparseTrajectoryProblem`,
  keeps sample-aware `measure_key`s separate from perturbation `embedding_id`s,
  rolls out one continuous global-time trajectory, and evaluates downstream
  checkpoint finite-measure losses.
- Biology tables separate priority-class readiness from axis readiness and gate
  expansion, depletion, plasticity, ecology, TNF, CIS-like, and TSK/pEMT claims
  with explicit mass semantics, replicate/fold/sample support, guide
  concordance, metric-specific nulls, and signed program effects.
