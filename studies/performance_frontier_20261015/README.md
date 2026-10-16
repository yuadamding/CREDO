# Performance-frontier CREDO: source publication

This source-only snapshot was published by user request on October 16, 2026
(America/Chicago). It preserves the current implementation from the separate
`credo_performance_frontier_20261015_v1` workspace. It does not replace the v4
package, modify earlier source archives, or update a running worker.

## What is included

- `frontier/`: fold-specific count-native observation fitting; six direct/DDR
  candidate families; objective, sampling and checkpoint code; streamed
  forecasts, calibration, scoring, ensembles and selector contracts.
- `deployment/` and `operations/`: packaging, resource accounting, exact-job
  custody, explicit recovery, continuation acceptance and read-only progress.
- `tests/`: current regression sources, including tests that require excluded
  private runtime evidence and are therefore not standalone public tests.
- Root JSON/TSV files: exact archived protocol/configuration and authorization
  records. Their historical approval fields are **not transferable execution
  authority**. Resource and numerical budgets must not be reset by copying them.
- `SOURCE_MANIFEST.json`: per-file sizes, hashes and original relative paths,
  with separate identity checks against the four v4 observation-worker bundles.

Source and protocol bytes are preserved without reformatting. The working
implementation and executed worker closure are distinct: an unexecuted source
module is not evidence that its scientific workflow has completed. The new
progress helper is controller-side code, not a modification to those workers.

## Scientific and runtime boundaries

The registered search has 24 configurations and a maximum of four independent
H100 workers. Its larger-model comparison is the matched width-128 versus
width-256 pair within each family, not an unlimited capacity search. Observation
models use their separately declared architecture and training subsets.

At publication, observation fitting is in progress; the main predictor search
and its final assessments have not established a new performance result.
Publication does not assert that a larger model helps, that all four observation
fits finished, or that new direct/DDR predictors passed qualification.

All four GSE314342 donors are exposed developmental data. Nested fitting does
not make them independent confirmation. Rest and Stim8hr remain contemporaneous
sister branches; captured relative representation is not absolute growth.
State proximity, spread, abundance, molecular endpoints and contrasts retain
separate interpretations. No LARRY artifacts or access are included.

## Deliberate exclusions and dependencies

No biological arrays, cell/guide inventories, checkpoints, particle banks,
cache payloads, credentials, live Job manifests, admission journals, input
leases, or run-output receipts are included. Historical failed evidence remains
in the original workspace; it has not been erased or converted into success.

This is **not a standalone launch bundle**. Operational modules retain exact
institutional paths and depend on private identity-bound inputs, pinned reader
sources, lifecycle code, qualification evidence and execution receipts. See
`deployment/build.py` for the explicit reader/lifecycle hashes. Those dependencies
must not be substituted with whatever happens to be installed. Scientific modules
use NumPy, SciPy and PyTorch; operational tests also use pytest. No environment
or dependency versions were changed for this publication.

Do not execute launch, recovery, acquisition or training entry points merely to
test installation. Current authorization, capacity, input, code and ownership
checks remain necessary for any separately requested numerical work.

## Publication checks

From this directory:

```bash
python verify_source.py
PYTHONPATH=. python -m pytest -q -c pytest-publication.ini
```

The second command is deliberately restricted by `pytest-publication.ini` to
fabricated metadata tests. It does not train models, load real checkpoints,
read biological payloads, call the scheduler or claim H100 qualification.
It is not the entire archived regression suite. Tests requiring private launch
bundles and numerical suites are retained as source, not silently counted as
passing. See `PUBLICATION_VALIDATION.md` for the checks actually performed.

The root repository license applies. Nothing in this snapshot changes the
scientific conclusions or artifact identities of completed historical studies.
