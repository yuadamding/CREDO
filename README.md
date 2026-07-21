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
python scripts/verify_setup.py \
  --check-data \
  --data-path ../inputs/single_time/example.h5ad \
  --data-schema single_time \
  --strict-data-schema \
  --latent-key X_pca \
  --control-col is_control \
  --guide-col guide_id \
  --target-gene-col target_gene \
  --sample-col sample_id
credo-validate-data \
  --data-path /path/to/input.h5ad \
  --schema trajectory \
  --strict \
  --latent-key X_pca \
  --json
```

## When to Use Each Mode

| Data setting | Entry point | Claim boundary |
| --- | --- | --- |
| P4/P60 endpoint Perturb-seq | `runners/run_credo_hnscc_full.py` | Finite-measure endpoint transport, mass calibration, and same-start counterfactuals over the observed interval. |
| Multi-time trajectory | `runners/run_credo_trajectory.py` or `runners/run_credo_lps_3time.py` | Continuous global-time rollout through observed checkpoint finite measures. |
| GSE314342 T-cell trajectory | `runners/run_credo_gse314342.py` | Target-balanced guide/donor batches with shared target-gene embeddings and optional donor-grouped context. |
| True one-timepoint Perturb-seq | `runners/run_credo_single_time.py` | Control-referenced static effect paths on a non-physical effect axis. |
| Pseudotime-only snapshot | Diagnostic only | Not a first-class physical-time CREDO mode. Do not claim temporal drift or growth from pseudotime alone. |

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
  --effect-vector-components delta_log_mass,latent_mean_shift,latent_variance_shift \
  --strict-data-schema \
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

For guide-level single-time Perturb-seq, the recommended convention is:

```text
--perturbation-col guide_id
--guide-col guide_id
--target-gene-col target_gene
--embedding-level target_gene
--view-key-level sample_guide
--view-level view
```

`view_key_level` controls how finite-measure views are constructed; `view_level`
controls whether endpoint training preserves those views or pools them by
embedding. Setting `--view-key-level guide` or `sample_guide` together with
`--view-level embedding` is allowed, but it pools away guide-level endpoint
views and is reported with a warning.

Single-time context gradient modes have distinct claim semantics:

| Mode | Meaning |
| --- | --- |
| `recompute_no_grad` | Reuse fixed sampled context particles and recompute learned context features without gradients. This is the default static observed-context covariate mode. |
| `recompute_with_grad` | Reuse fixed sampled context particles while allowing gradients through learned context feature maps. |
| `detached_cache` | Freeze both sampled particles and computed context features. Use mainly for diagnostics. |

Single-time control-null and guide-concordance regularizers use
`--effect-vector-components`. The default is
`delta_log_mass,latent_mean_shift` for compatibility; add
`latent_variance_shift` when dispersion effects should enter the regularized
effect vector. Richer program/prototype components are intentionally left for a
future typed identity/effect-vector layer.

The single-time runner writes biological effect artifacts after training:

```text
single_time_effects.csv
single_time_endpoint_metrics.csv
single_time_guide_concordance.csv
single_time_control_null.csv
single_time_control_null_summary.csv
single_time_latent_mean_shift_by_dim.csv
single_time_latent_variance_shift_by_dim.csv
single_time_claim_report.json
single_time_problem_summary.json
single_time_resolved_config.json
single_time_command.txt
single_time_git_sha.txt
```

These files preserve the single-time claim boundary with columns such as
`context_protocol`, `context_gradient_mode`, `effect_axis_is_physical_time`,
`mass_claim_grade`, explicit diagnostic and claimable mass-effect columns,
latent mean/variance shift norms, factual/reference endpoint
geometry-plus-mass metrics, guide concordance summaries, and control-null
diagnostics.

Effect outputs distinguish training and reporting levels with
`training_view_level`, `report_view_level`, and
`report_is_posthoc_view_level`. The runner always emits per-view biological
diagnostics; if training used `--view-level embedding`, those rows are labeled
as post hoc disaggregated view diagnostics. Mass-effect values are always
reported as diagnostic finite-measure weight effects in
`diagnostic_delta_log_mass` and `diagnostic_delta_mass`. The legacy
`delta_log_mass` and `delta_mass` columns remain compatibility aliases and are
labeled with `*_semantics`; claim-grade abundance columns are populated only
when `abundance_claim_grade == claim_grade`. Endpoint output includes both
factual-vs-target and reference-vs-target metrics plus
`delta_*_ref_minus_fact` improvement columns. Guide-concordance summaries
include `n_views`, `n_guides`, `n_samples`, `guide_concordance_evaluable`, and
`guide_concordance_claimable` so post hoc or single-view summaries are not
mistaken for claim-grade concordance. Per-view particle diagnostics include
source, factual-terminal, and reference-terminal ESS fraction, max-weight
fraction, log-weight range, and a coarse `weight_diagnostic_status`. Control
null outputs also annotate every effect row with metric-specific control-null
z-scores and absolute-p95 exceedance flags.

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
- Trajectory key minibatching is valid with zero context or a grouped context
  bank. Global self-consistent context requires an all-key rollout.
- Single-time training consumes `SingleTimeProblem`, keeps finite-measure
  `view_id`s separate from perturbation embeddings, labels outputs as
  non-physical effect-axis diagnostics, and caches sampled fixed-context
  particles while recomputing learned context features by default.
- Biology tables separate priority-class readiness from axis readiness and gate
  expansion, depletion, plasticity, ecology, TNF, CIS-like, and TSK/pEMT claims
  with explicit mass semantics, replicate/fold/sample support, guide
  concordance, metric-specific nulls, and signed program effects.
