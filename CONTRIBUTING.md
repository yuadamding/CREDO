# Contributing

CREDO is research-alpha software. Focused fixes, semantic tests, adapter
contracts, and validation baselines are welcome.

```bash
python -m pip install -e ".[dev]"
ruff check src examples tests
ruff format --check src examples tests
pytest -q
python -m build
```

The installed package must retain one semantic representation and one execution
path per scientific concept. Core keeps generated conformance fixtures only.
Real cohort adapters, preprocessing, configs, and claims belong in a versioned
external workspace or adapter distribution and must not add cohort branches to
`src/credo`.

Model changes require an invariant or end-to-end regression test and an
explicit claim boundary. Adapters must document source provenance, axis
meaning, controls, mass semantics, denominators, QC, and support sampling. Do
not commit patient-level data, downloaded expression matrices, checkpoints,
credentials, or generated runs.

Contributions are accepted under the GNU Affero General Public License v3.0.
Recipes and codecs distributed through entry points receive no plugin exception;
see [the license transition](docs/license_migration.md).
