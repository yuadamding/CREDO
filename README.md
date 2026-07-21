# CREDO

CREDO is a compact research-alpha model for control-referenced finite-measure
dynamics in longitudinal Perturb-seq. Version `3.0.0a1` deliberately provides
one data representation, one soft-reference model, one particle rollout, one
checkpoint objective, one trainer, one counterfactual engine, and one CLI.

CREDO estimates regularized effective-generator contrasts from destructive
snapshots. It does not reconstruct cell genealogies or turn fitted-model
counterfactuals into experimental causal effects.

## Install

```bash
python -m pip install -e ".[dev]"
credo --help
```

Python 3.11 through 3.13 are supported.

## Canonical data

Every adapter writes the same files:

| File | Contract |
| --- | --- |
| `support.h5ad` | `obs.measure_id`, `obs.time_label`, optional `obs.atom_weight`, and one latent `obsm` matrix. |
| `measure_meta.parquet` | One row per opaque `measure_id` with sample, guide, embedding, target, context group, and control identity. |
| `masses.parquet` | One positive mass and denominator per observed measure/checkpoint, with one explicit semantics. |
| `counts.parquet` | Optional complete context-group/time compositional count blocks. |
| `dataset.json` | Axis, latent key, mass semantics, and source provenance. |

The model never parses `measure_id`. `perturbation_id` identifies the
experimental construct, while `embedding_id` identifies the learned residual;
multiple guides may share one target-gene embedding.

Mass semantics are one of `absolute`, `relative_within_group`,
`captured_count`, or `unit`. Only absolute mass permits an absolute-growth
interpretation. Relative mass supports within-denominator comparisons,
captured counts are capture-scale diagnostics, and unit mass carries no
abundance information.

## Run

The synthetic example exercises the complete contract:

```bash
python examples/synthetic/generate.py
credo validate examples/synthetic/config.yaml
credo run examples/synthetic/config.yaml --device cpu
credo summarize runs/synthetic
```

The YAML is authoritative. The CLI permits only operational overrides for
output directory, device, and seed. Unknown or duplicate YAML keys are errors.

```yaml
data:
  support: data/support.h5ad
  latent_key: X_credo
  measure_meta: data/measure_meta.parquet
  masses: data/masses.parquet
  counts: data/counts.parquet

axis:
  kind: physical
  source: Rest
  labels: [Rest, Stim8hr, Stim48hr]
  values: [0.0, 8.0, 48.0]

model:
  embedding_dim: 8
  n_programs: 8
  hidden_dim: 128
  context: catalog_bank

training:
  epochs: {state: 40, mass: 20, context: 20}
  particles: 64
  eval_particles: 256
  steps_per_interval: 4
  measures_per_batch: 256
  learning_rate: 0.001
  validation_fraction: 0.2
  patience: 10
  seed: 0

loss: {mass: 1.0, count: 0.1}
output: runs/example
```

Training follows a fixed continuation schedule: state geometry, finite mass
and optional counts, then ecological context. Before the latter two phases,
CREDO initializes every catalog-bank entry with detached complete-group
rollouts and refuses optimization unless coverage is complete.

When compositional counts are present, held-out validation uses complete
context groups so count denominators cannot leak held-out outcomes. If no such
split is possible, the manifest says `train_self_eval` instead of overstating
the validation evidence.

## Counterfactuals

```python
from credo import Trainer, counterfactual, load_config, load_data

config = load_config("examples/synthetic/config.yaml")
data = load_data(config)
trained_run = Trainer.load(config.output / "checkpoint.pt", data, config)
effects = counterfactual(
    trained_run,
    "D1::GENE1-1",
    context_policy="self_consistent",
)
```

Factual and reference branches use identical observed source particles and
identical noise. The reference removes only the selected perturbation residual;
all controls already have an exact zero residual around one shared learned
reference. Context can be recomputed or clamped to the reference rollout.

## Artifacts

Each run writes exactly five durable artifacts:

```text
manifest.json
checkpoint.pt
history.parquet
metrics.parquet
counterfactuals.parquet
```

The manifest records the resolved config, package and dependency versions,
Git state, command, input hashes, axis and mass contracts, validation split,
bank coverage, and particle-weight thresholds.

## Cohort adapters

- [`examples/gse314342`](examples/gse314342/README.md) converts the downloaded
  primary human CD4+ T-cell cohort and preserves donor-guide measures.
- [`examples/hnscc_p4_p60`](examples/hnscc_p4_p60/README.md) converts the P4/P60
  HNSCC endpoint cohort with captured-count mass.
- [`examples/lps`](examples/lps/README.md) converts the 90-minute to 10-hour LPS
  cohort and keeps expression preprocessing outside the installed package.
- [`examples/synthetic`](examples/synthetic/README.md) is the deterministic
  contract and smoke-test cohort.

Adding a dataset requires no change under `src/credo`.

## Verify

```bash
ruff check src examples analysis tests
ruff format --check src examples analysis tests
pytest -q
python -m build
```

See [scientific validation](docs/scientific_validation.md) before making
biological claims.
