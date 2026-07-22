# CREDO

CREDO is a compact research-alpha framework for control-referenced
finite-measure dynamics in longitudinal Perturb-seq. It provides one stable
study, particle, evaluation, counterfactual, and artifact runtime with
immutable model recipes. The default is `credo.compact_sde_v3@3.0`; archived
transformer checkpoints use `credo.transformer_sde_v2@2.0` without translating
their tensors into the compact architecture.

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
| `dataset.json` | Axis, latent key, mass semantics, representation contract, and source provenance. |

Dense H5AD latent supports are read lazily by default through a bounded
finite-measure cache. Set `data.lazy_support: false` only for small fixtures.

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
credo train examples/synthetic/config.yaml --device cpu
credo summarize runs/synthetic
```

The YAML is authoritative. The CLI permits only operational overrides for
output directory, device, and seed. Unknown or duplicate YAML keys are errors.

```yaml
recipe: credo.compact_sde_v3@3.0

data:
  support: data/support.h5ad
  latent_key: X_credo
  measure_meta: data/measure_meta.parquet
  masses: data/masses.parquet
  counts: data/counts.parquet
  dataset: data/dataset.json
  lazy_support: true
  support_cache_size: 256

axis:
  kind: physical
  source: Rest
  labels: [Rest, Stim8hr, Stim48hr]
  values: [0.0, 8.0, 48.0]

recipe_config:
  model:
    embedding_dim: 8
    n_programs: 8
    hidden_dim: 128
    context: catalog_bank
    growth_max: 3.0
  training:
    epochs: {state: 40, mass: 20, context: 20}
    particles: 64
    steps_per_interval: 4
    measures_per_batch: 256
    batching: target_round_robin
    learning_rate: 0.001
    patience: 10
    seed: 0
  evaluation: {particles: 256, measures_per_batch: 256}
  validation: {strategy: auto, fraction: 0.2, representation_scope: shared}
  loss: {mass: 1.0, count: 0.1, sinkhorn_epsilon: 0.1}
output: runs/example
```

`target_round_robin` interleaves targets; `target_blocked` keeps all views of a
target in one optimizer batch. `credo run` remains a compatibility alias for
`credo train`.

Training follows a fixed continuation schedule: state geometry, finite mass
and optional counts, then ecological context. Before the latter two phases,
CREDO initializes every catalog-bank entry with detached complete-group
rollouts and refuses optimization unless coverage is complete.

Validation may be automatic, an explicit `context_group` list, an explicit
downstream `checkpoint` list, or `train_self_eval`. Complete context groups are
held out whenever compositional counts are present, so their denominators do
not leak outcomes. Evaluation particle and batch counts are operational and may
be overridden when loading a checkpoint; model, training, validation, and loss
settings remain bound by the checkpoint contract.

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
    n_particles=512,
)
```

Factual and reference branches use identical observed source particles and
identical noise. The reference removes only the selected perturbation residual;
all controls already have an exact zero residual around one shared learned
reference. Controls are valid numerical-null counterfactuals. Intrinsic models
roll only the focal measure; contextual models use the complete context group.
Context can be recomputed or clamped to the reference rollout.

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
representation artifact, recipe capabilities, checkpoint mode, bank coverage,
and particle-weight thresholds. Native compact checkpoints currently declare
`inference_only`: fresh training is deterministic, but optimizer and RNG state
are not persisted for trajectory continuation.

## Recipes

| Recipe | Representation | Context | Training | Imported checkpoint mode |
| --- | --- | --- | --- | --- |
| `credo.compact_sde_v3@3.0` | Frozen external latent | None or catalog bank | Released state, mass, context executor | Native inference; fresh fit available |
| `credo.transformer_sde_v2@2.0` | Preserved 50-D expression VAE | Full-population inducing transformer | Typed archived-plan reconstruction | Historical inference only |

Both recipes use absolute particle weights `log_m0 + logw`, exact shared-control
reference semantics, same-start/same-noise counterfactuals, and the common
metric table returned by `credo.evaluate`. Recipe-specific objective values are
not comparable across model families.

The v2 importer strict-loads raw or embedded EMA dynamics and the preserved VAE,
records source hashes, and refuses resume when optimizer or RNG state is absent.
It writes a portable, hash-verified bundle containing canonical model and VAE
states, representation metadata, latent cache, and source/artifact manifests.
Its frozen compatibility modules are loaded only when the recipe is requested
and introduce no dependencies beyond the core runtime. Compact v3 has the
released training executor; transformer v2 preserves a typed, non-executable
reconstruction and the complete source run config, but supports imported
inference and replay only.
See [runtime and recipes](docs/runtime_and_recipes.md) and the
[LPS replay example](examples/lps_v2_replay/README.md).

## Cohort adapters

- [`examples/gse314342`](examples/gse314342/README.md) converts the downloaded
  primary human CD4+ T-cell cohort and preserves donor-guide measures.
- [`examples/hnscc_p4_p60`](examples/hnscc_p4_p60/README.md) converts the P4/P60
  HNSCC endpoint cohort with captured-count mass.
- [`examples/lps`](examples/lps/README.md) converts the 90-minute to 10-hour LPS
  cohort and keeps expression preprocessing outside the installed package.
- [`examples/lps_v2_replay`](examples/lps_v2_replay/README.md) imports and
  replays the four preserved transformer-v2 donor folds.
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
