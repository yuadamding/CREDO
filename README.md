# CREDO

CREDO is a research-alpha framework for control-referenced finite-measure
perturbation dynamics. A storage-independent `Study` holds experimental meaning;
a selected `StudyView` is validated, split, and compiled by a versioned recipe;
and a generic run bundle records the resulting model and outputs.

```text
Study -> StudyView -> SplitPlan -> recipe compiler -> recipe runtime -> run.json
```

The released recipes are `credo.compact_sde_v3@3.0` for fresh training and
`credo.transformer_sde_v2@2.0` for strict inference from imported historical
checkpoints. CREDO estimates regularized generator contrasts from destructive
snapshots. It does not reconstruct cell genealogies or turn model
counterfactuals into experimental causal effects.

## Install

```bash
python -m pip install -e ".[dev]"
credo --help
```

Python 3.11 through 3.13 are supported. GPU execution follows the installed
PyTorch build.

## Core workflow

```python
from credo import evaluate, load_config, open_run, open_study, train

config = load_config("run.yaml")
study = open_study(config.study, verify="semantic")
try:
    fitted = train(study, config, device="cuda")
    fitted.save()
finally:
    study.close()

run = open_run(config.output / "run.json", device="cuda")
try:
    metrics = evaluate(run)
finally:
    run.close()
```

The package root intentionally exposes the semantic workflow: `Study`,
`StudyView`, `SelectionSpec`, `SplitPlan`, `open_study`, `write_study`, `train`,
`open_run`, `evaluate`, and `counterfactual`. The old top-level `TrajectoryData`,
`CREDOStudy`, `CREDOModel`, `Trainer`, and `load_data` names remain deprecated
compatibility shims for this alpha cycle.

## Native studies

A native schema-v3 study is the single source of truth for design, conditions,
series, observations, representations, support stores, abundance channels,
compositions, run-selectable effect and reference bindings, and provenance.

```python
from credo import open_study, write_study

legacy = open_study("inputs/cohort/dataset.json", verify="semantic")
try:
    manifest = write_study(legacy, "inputs/cohort/native-study")
finally:
    legacy.close()
```

`write_study()` writes Parquet semantic tables, lazy HDF5 empirical-law stores,
artifact hashes, provenance, and `study.json` transactionally. The manifest is
written last and the temporary directory is atomically renamed. Native reads
support `none`, `schema`, `manifest`, `semantic`, and `full` verification;
`full` additionally scans every support law and checks its semantic digest.

The schema-v1/v2 five-file format remains a permanent read-only compatibility
codec:

```text
support.h5ad
measure_meta.parquet
masses.parquet
counts.parquet          # optional
dataset.json
```

Codecs are discovered through the `credo.study_codecs` entry-point group.

## Run configuration

New runs should reference one native manifest. Axis and data paths are not
duplicated in YAML.

```yaml
recipe: credo.compact_sde_v3@3.0
study: inputs/cohort/native-study/study.json

selection:
  representation_id: latent32_all
  abundance_channel: modeled_frequency
  effect_binding_id: target_gene_shared
  reference_binding_id: donor_matched_ntc
  composition_policy: require_complete
  replicate_policy: {mode: reject}

recipe_config:
  validation:
    strategy: context_group
    values: [donor-1]
    fraction: 0.0
    representation_scope: shared

output: runs/cohort/donor-1
```

Omitting `selection.abundance_channel` selects the study primary. Explicit
`null` disables abundance and gives compact-v3 unit mass; unit-mass runs may
train geometry only. Raw zero abundance is retained by the Study, but a recipe
that models finite mass requires an explicitly selected positive transformed
channel.

Effect and reference parameterizations are selected by binding ID at run time.
Compact-v3 supports one global soft reference and rejects selections resolving
to multiple reference pools. It also requires series-static context and sample
identity. Replicates may be selected or pooled by concatenating geometry and
using an explicit abundance pooling rule.

## Splits

The recipe plans the actual train and validation partition before compilation.
`SplitPlan` records stable series, checkpoint, and observation IDs, selection
objects, held-out identities, and a content-derived split ID. The same plan is
used by training, evaluation, checkpoints, and `run.json`.

A representation without `fit_split_id` is shared and results are labeled
`transductive`. A nested representation must record its fitted series and
checkpoints; CREDO rejects overlap with held-out donors or checkpoints and
labels the result `inductive`.

## Composition semantics

Geometry and abundance are separate: each empirical law contains normalized
atom probabilities, while named abundance channels carry mass semantics,
units, denominator scope, and transform provenance. Composition selection is
explicit:

- `require_complete` rejects partial denominators.
- `preserve_background` keeps unselected members as detached denominator rows.
- `condition_on_selection` mints a new denominator identity tied to the
  selection hash.
- `drop` removes composition likelihoods.

Only `absolute` mass permits absolute-growth claims. Relative mass supports
within-denominator comparisons, captured counts are capture-scale diagnostics,
and unit mass contains no abundance information.

## Run bundles

Fresh compact runs use one recipe-neutral layout:

```text
run/
|-- run.json
|-- state/checkpoint.pt
`-- tables/
    |-- history.parquet
    |-- predictions.parquet
    |-- metrics.parquet
    |-- diagnostics.parquet
    `-- counterfactuals.parquet
```

`run.json` binds the recipe, resolved config, Study content hash, selection
hash, compiled problem hash, exact split, checkpoint codec, and every artifact
hash. `open_run()` verifies the bundle, resolves the recipe, reopens and checks
the bound study, compiles the saved split, and delegates checkpoint loading to
the recipe.

Predictions use stable series, observation, checkpoint, and representation IDs.
Common metrics are long form (`metric_name`, `value`, `unit`); particle ESS,
weight concentration, solver steps, and seeds live in a separate diagnostics
table. Future deterministic recipes therefore do not need to fabricate
particle fields.

```bash
credo validate run.yaml
credo train run.yaml --device cuda
credo evaluate runs/example/run.json --output evaluation.parquet
credo counterfactual runs/example/run.json --series-id D1::GENE1-1
credo summarize runs/example
```

Imported transformer-v2 bundles initially have no study binding. Bind them at
import with `--bind-config`, or call `bind_run_study()` before generic
evaluation. Historical bundles remain inference-only because optimizer,
scheduler, RNG, and terminal training state were not preserved.

## Repository boundary

Cohort-specific preprocessing, configs, provenance, and replay commands belong
to an external analysis workspace or adapter package. CREDO core contains only
generic semantic, recipe, execution, and generated conformance fixtures. Adding
a cohort must not add dataset branches under `src/credo`.

See [the Study contract](docs/study_model.md), [runtime and recipes](docs/runtime_and_recipes.md),
[scientific validation](docs/scientific_validation.md), and the
[license transition](docs/license_migration.md).

## Verify

```bash
ruff check src examples analysis tests
ruff format --check src examples analysis tests
pytest -q
python -m build
```
