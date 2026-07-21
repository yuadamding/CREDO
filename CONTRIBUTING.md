# Contributing

CREDO is research-alpha software. Focused fixes, semantic tests, cohort
adapters, and validation baselines are welcome.

```bash
python -m pip install -e ".[dev]"
ruff check src examples analysis tests
ruff format --check src examples analysis tests
pytest -q
python -m build
```

The installed package must retain one representation and one execution path
per scientific concept. New datasets belong under `examples/` and must emit
the canonical files without adding branches to `src/credo`.

Model changes require an invariant or end-to-end regression test and an
explicit claim boundary. Adapters must document source provenance, axis
meaning, controls, mass semantics, denominators, QC, and support sampling. Do
not commit patient-level data, downloaded expression matrices, checkpoints,
credentials, or generated runs.

Contributions are accepted under the MIT license.
