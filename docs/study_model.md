# Study model

CREDO separates biological study semantics from storage and recipe execution:

```text
Study -> recipe-specific compiled view -> runtime and artifacts
```

`Study` is the storage-independent semantic object. The released compact-v3
executor and transformer-v2 replay continue to consume `TrajectoryData` during
the compatibility migration; their numerical behavior and checkpoint schemas
are unchanged.

## Identities

The study layer makes three identities explicit:

| Identity | Meaning |
| --- | --- |
| `condition_id` | Experimental intervention or condition |
| `series_id` | Longitudinal unit advanced through checkpoints |
| `observation_id` | One series at one checkpoint |

Condition metadata require no CRISPR-specific fields. Guide, target, compound,
dose, genotype, and modality columns are optional. Context and processing
metadata belong to observations and may vary across checkpoints.

## Geometry and abundance

An `EmpiricalLaw` contains coordinates and probabilities that sum to one.
Abundance is stored independently in named channels with typed semantics,
units, denominator requirements, claim permissions, and zero policy.

Consequently, a study can retain:

- geometry with no abundance;
- abundance or composition counts with no geometry;
- raw zero abundance;
- several abundance channels over one support;
- several representations of the same observations.

`Study.snapshot()` combines one selected representation and abundance channel
without changing either source table.

## Catalogs

Large semantic entities remain DataFrame-backed rather than becoming one
Python object per cell:

- `ConditionTable`
- `SeriesTable`
- `ObservationTable`
- `AbundanceTable`
- `CompositionTable`
- `RepresentationCatalog`

Construction validates primary keys and local types. `Study.validate()` checks
foreign keys, representation scope, support references, dimensions, and
composition alignment. `StudyView` binds a selection, representation, and
abundance channel while sharing the original support store.

`RepresentationSpec.included_series` and `included_checkpoints` record fitting
scope for leakage checks. They do not hide encoded evaluation observations;
geometry availability is determined by the observation and support store.

## Support stores

`SupportStore` is a backend-neutral protocol keyed by `SupportRef`. Its public
read result is always an `EmpiricalLaw`; abundance is never part of the store
interface. `InMemorySupportStore` supports tests and simulations.

`CurrentFiveFileStudyCodec` adapts the existing lazy finite-measure H5AD store.
The adapter converts absolute finite-measure weights to conditional
probabilities only when a snapshot is read. Reading abundance and validating
stable IDs does not materialize support arrays.

## Compatibility codec

Schema-v1/v2 datasets can be opened from a run YAML, `dataset.json`, its
containing directory, an existing `RunConfig`, or `TrajectoryData`:

```python
from credo import open_study

study = open_study("examples/synthetic/config.yaml", verify="semantic")
view = study.view(
    representation_id=study.manifest.primary_representation,
    abundance_channel=study.manifest.primary_abundance_channel,
)
```

The codec maps legacy fields as follows:

| Legacy | Study |
| --- | --- |
| `measure_id` | `series_id` |
| `perturbation_id` | `condition_id` |
| `(measure_id, time_label)` | `observation_id` |
| `context_group_id` | observation `context_id` |
| one mass table | `legacy_mass` abundance channel |
| positional `CountBlock` | stable-ID `CompositionTable` |
| one `RepresentationArtifact` | one-entry `RepresentationCatalog` |

The next migration stage can add schema-v3 transactional writing and more
support backends against these contracts without changing recipe numerics.
