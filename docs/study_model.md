# Study model

CREDO separates biological semantics from storage and numerical execution:

```text
Study -> StudyView -> recipe compiler -> recipe runtime
```

`Study` is the immutable semantic object. Compact-v3 and transformer-v2
compile a view into the existing `TrajectoryData` numerical shape internally,
which preserves released model behavior while removing that legacy shape from
the public training boundary.

## Identities and design

| Identity | Meaning |
| --- | --- |
| `condition_id` | Experimental intervention or condition |
| `series_id` | Longitudinal unit advanced through checkpoints |
| `observation_id` | One replicate of a series at one checkpoint |

Conditions do not require CRISPR fields. Guide, target, compound, dose,
genotype, and modality columns are optional. Replicates have distinct
observation IDs without pretending to be separate longitudinal series.

`StudyDesign` validates chain, star, and DAG reachability, source and target
roles, transition direction, and monotone coordinates on ordered axes. Recipes
declare the axis kinds and topologies they can compile.

## Geometry and abundance

An `EmpiricalLaw` contains support coordinates and probabilities summing to
one. A `SupportIndexTable` records coverage by
`(observation_id, representation_id)` and resolves each available law through
a fully qualified `(store_id, representation_id, support_key)` reference.
`SupportStoreRegistry` permits different representations to use different
backends.

Abundance is independent of support. Each channel declares semantics, unit,
denominator scope, zero policy, and an optional transform ID. Claim permissions
are derived from semantics. Raw zero abundance and missing geometry can be
retained; a recipe that needs positive model mass must select an explicit
positive transformed channel.

Composition rows reference observations. A `StudyView` selection states how a
partial composition block is handled: require completeness, preserve the full
background, condition on the selection, or drop compositions.

## Validation

Table constructors enforce local shape, type, and primary-key invariants.
Cross-table checks are explicit, so malformed studies remain inspectable:

```python
report = study.validate(level="semantic")
report.raise_for_errors()
```

Loader verification levels have distinct costs:

| Level | Work |
| --- | --- |
| `none` | Parse metadata and construct lazy handles |
| `schema` | Validate local table schemas |
| `manifest` | Check declared artifact existence, size, and hashes |
| `semantic` | Check foreign keys, coverage, references, denominators, and design |
| `full` | Scan support arrays and validate all numeric values |

## Compatibility codec

`open_study()` selects a registered codec. The released registry currently
contains the read-only schema-v1/v2 five-file codec, which accepts a run YAML,
`dataset.json`, its directory, a `RunConfig`, or `TrajectoryData`:

```python
from credo import open_study

study = open_study("examples/synthetic/config.yaml", verify="semantic")
view = study.view(
    representation_id=study.manifest.primary_representation,
    abundance_channel=study.manifest.primary_abundance_channel,
)
```

Legacy IDs map to explicit condition, series, observation, abundance,
composition, representation, and support-index tables. Lazy finite-measure
support is converted to conditional probabilities only when read; abundance
access does not materialize coordinates.

Native schema-v3 transactional writing, migration tooling, and generic run
bundles are separate unreleased migrations. The codec registry is the extension
point for those additions.
