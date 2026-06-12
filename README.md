# CREDO

Compact CREDO `2.0.1` package for endpoint and multi-time finite-measure
Perturb-seq trajectory modeling. The source snapshot contains the installable
`credo` package, runners, post-training analysis utilities, setup scripts, and
tests. It intentionally excludes `.h5ad` data, checkpoints, and run outputs.

## Layout

| Path | Purpose |
| --- | --- |
| `package/` | Installable Python package under `credo`. |
| `runners/` | HNSCC endpoint, generic trajectory, LPS trajectory, single-time, and summary entry points. |
| `analysis/` | Counterfactual biology and signature utilities. |
| `scripts/` | Setup verification, LPS input building, and randomized stress tests. |
| `tests/` | Unit, regression, smoke, and randomized tests. |

Generated outputs belong in `runs/`, `outputs/`, `results/`, or `models/`.

## Install

```bash
bash scripts/install_bundle.sh credo-hnscc
python -m pip install --no-cache-dir -e package
python scripts/verify_setup.py
python scripts/verify_setup.py --json
python scripts/verify_setup.py \
  --check-data \
  --data-path /path/to/input.h5ad \
  --data-schema trajectory \
  --strict-data-schema \
  --latent-key X_pca
credo-validate-data \
  --data-path /path/to/input.h5ad \
  --schema trajectory \
  --strict \
  --latent-key X_pca \
  --json
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

python runners/run_credo_single_time.py \
  --data-path ../inputs/single_time/example.h5ad \
  --output-dir runs/single_time_smoke \
  --latent-key X_pca \
  --perturbation-col guide_id \
  --guide-col guide_id \
  --target-gene-col target_gene \
  --control-col is_control \
  --sample-col sample_id \
  --embedding-level target_gene \
  --view-key-level sample_guide \
  --view-level view \
  --mass-mode unit_mass \
  --context-protocol observed_snapshot \
  --epochs 2 \
  --n-particles 32 \
  --n-steps 4
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

Single-time CREDO estimates control-referenced static effect paths on a
non-physical axis from constructed control reference to observed snapshot. It
does not infer physical temporal dynamics from one snapshot. Use
`--view-key-level sample_guide` to preserve sample-specific guide-level finite
measures while learning target-gene embeddings, or `guide` to pool guides
across samples with a global reference.

## Verify

```bash
pytest -q -m "not slow and not gpu"
python scripts/stress_test_trajectory_core.py --cases 1000
python scripts/stress_test_trajectory_production.py \
  --cases 1000 \
  --counterfactual-cases 300 \
  --trainer-cases 100
```

## Public Imports

```python
from credo.data import EndpointProblem, SingleTimeProblem, TrajectoryProblem, TrajectoryView
from credo.losses import MultiTimeEndpointLoss
from credo.models import FullDynamicsModel, SingleTimeCounterfactualEngine, WeightedParticleSimulator
from credo.training import SingleTimeTrainer, Trainer, TrajectoryTrainer
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
- Perturbation chunking for transformer or causal-attention ecology uses
  exact full-context caching by default; chunk-local ecology is not a supported
  claim-grade training mode.
- The default endpoint loss is a finite-measure geometry-plus-log-mass proxy:
  debiased Sinkhorn geometry on normalized measures plus a log-mass penalty,
  not a full dynamic unbalanced-OT path objective.
- Multi-time training consumes `TrajectoryProblem` or `SparseTrajectoryProblem`,
  keeps sample-aware `measure_key`s separate from perturbation `embedding_id`s,
  rolls out one continuous global-time trajectory, and evaluates downstream
  checkpoint finite-measure losses.
- Single-time training consumes `SingleTimeProblem`, keeps finite-measure
  `view_id`s separate from perturbation embeddings, labels outputs as
  non-physical effect-axis diagnostics, and caches sampled fixed-context
  particles while recomputing learned context features by default.
- Biology tables separate priority-class readiness from axis readiness and gate
  expansion, depletion, plasticity, ecology, TNF, CIS-like, and TSK/pEMT claims
  with explicit mass semantics, replicate/fold/sample support, guide
  concordance, metric-specific nulls, and signed program effects.
