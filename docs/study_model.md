# Study model

CREDO separates biological semantics from support storage and numerical
execution:

```text
Study -> StudyView -> SplitPlan -> recipe compiler
```

`Study` is immutable at its public boundary. Table access returns copies, while
support laws remain lazy and are shared by zero-copy views.

## Identities and design

| Identity | Meaning |
| --- | --- |
| `condition_id` | Experimental intervention or control condition |
| `series_id` | Longitudinal unit advanced through checkpoints |
| `observation_id` | One assay or replicate of a series at one checkpoint |

Conditions do not require CRISPR fields. Guide, target, compound, dose,
genotype, and modality are optional metadata. Replicates retain distinct
observation IDs rather than becoming artificial longitudinal series.

`StudyDesign` validates chain, star, and DAG reachability, source and target
roles, transition direction, and monotone coordinates on ordered axes. Recipes
declare the axis kinds and topologies they can compile.

## Geometry and abundance

An `EmpiricalLaw` contains support coordinates and probabilities summing to
one. `SupportIndexTable` declares coverage for each
`(observation_id, representation_id)` pair and resolves available laws through
`(store_id, representation_id, support_key)`. A `SupportStoreRegistry` can route
different representations to different lazy backends.

Abundance is independent of geometry. Each channel declares semantics, unit,
denominator scope, zero policy, and optional transform identity, input channel,
and immutable transform parameters. The output channel is the channel's own ID. Calling
`study.view()` selects the primary channel; calling
`study.view(abundance_channel=None)` explicitly disables abundance. Raw zeros
remain valid semantic observations, while positive-mass recipes require an
explicit positive modeling channel.

## Bindings

Biological conditions are immutable, but model parameter sharing is a run
choice:

```text
EffectBindingTable
binding_id, condition_id, effect_id, parameterization_kind,
parent_effect_id?, shrinkage_group_id?

ReferenceBindingTable
binding_id, condition_id, reference_pool_id, scope_kind, scope_key?
```

A recipe that implements the corresponding parameterization can therefore
compare guide-specific, target-shared, hierarchical, or pathway-shared effects
without creating another Study. The released trajectory recipes currently use
flat bindings. The five-file codec synthesizes compatibility bindings from
legacy embeddings and controls. Legacy embedding, reference-group, and
reference-role columns may round-trip as metadata but are not compiler inputs.

## Selection transforms

`SelectionSpec` binds IDs, metadata filters, representation, abundance,
bindings, composition policy, and replicate policy. A recipe must advertise
support for the selected policies before compilation.

Composition policies are:

- `require_complete`: reject a selection cutting through a denominator.
- `preserve_background`: retain unselected members as detached background.
- `condition_on_selection`: retain selected rows and mint a denominator suffixed
  by the selection hash.
- `drop`: omit composition blocks.

Replicate modes are `reject`, `select`, `pool`, `keep_separate`, and
`hierarchical`. Compact-v3 currently executes `reject`, `select`, and `pool`.
Pooling produces a stable pooled observation ID and source-observation
provenance. Its released geometry rule is concatenation; abundance may use
sum, mean, or exposure-weighted pooling.

## Content identity

Scientific provenance is content-addressed in layers:

```text
semantic table hashes + representation artifact hashes + support identities
    -> study_content_hash
    -> selection_hash
    -> split_id
    -> compiled_problem_hash
    -> run_contract_hash
```

Artifact locations are excluded from semantic identity; artifact content
hashes remain included. In-memory stores are materialized into a deterministic
support digest when necessary. Native stores persist this digest and recompute
it during full verification.

## Native schema v3

`write_study()` creates a transactional directory:

```text
study/
|-- conditions.parquet
|-- series.parquet
|-- observations.parquet
|-- support_index.parquet
|-- abundance.parquet              # when present
|-- compositions.parquet           # when present
|-- effect_bindings.parquet        # when present
|-- reference_bindings.parquet     # when present
|-- stores/
|   |-- store-0000.h5
|   `-- store-0000.parquet
|-- representations/                 # embedded local artifacts, when present
|-- provenance.json
`-- study.json
```

All files except `study.json` are written into a sibling temporary directory.
Their size and SHA-256 identities are recorded, the manifest is written last,
and the directory is atomically renamed. Existing destinations are never
overwritten.

Verification costs are explicit:

| Level | Work |
| --- | --- |
| `none` | Parse metadata and construct lazy handles |
| `schema` | Construct typed local tables and stores |
| `manifest` | Recompute artifact sizes and SHA-256 hashes |
| `semantic` | Check foreign keys, coverage, bindings, denominators, and design |
| `full` | Scan every support law and recompute support semantic digests |

```python
from credo import open_study, write_study

study = open_study("study/study.json", verify="semantic")
try:
    report = study.validate(level="semantic")
    report.raise_for_errors()
    view = study.view(
        representation_id="latent-all",
        abundance_channel="modeled_frequency",
        effect_binding_id="target_gene_shared",
        reference_binding_id="donor_matched_ntc",
    )
finally:
    study.close()
```

## Codec extension

The built-in codecs are `credo.native_study` and the read-only
`credo.current_five_file`. Third-party distributions register a complete codec
through the `credo.study_codecs` entry-point group. Probes must inspect an
actual schema marker and must not claim arbitrary YAML or directories.
