# Full drift–diffusion–reaction CREDO: run-source publication

This is the source used for the October 6, 2026 GSE314342 full-DDR study,
published by explicit user authorization on October 10. It does not replace
the count-SDE v4 project on `main`, modify the original frozen compatibility
archive, or change historical scientific results. It is not a new v4 model.

## Source identities and layout

- `source/src/credo/`: the modified v3-derived package, based on commit
  `6f4f57c114606476cb6e93c6d52dc34628173a4d`. Every Python file matches both
  retained phase bundle inventories and the study worktree byte for byte.
- `frozen/fitting_gpu_native_v6/`: exact non-package Python files in the
  successful GPU-native fitting-v6 bundle, including its worker and controller.
- `frozen/d2_forecast_v5/`: exact non-package Python files in the D2 forecast-v5
  bundle, including its worker, controller, and reuse/qualification paths.
- Root-level Python files and `tests/`: inspected working-tree support and
  regression tests. These are distinct from the frozen phase identities.
- `source/tests/`: four package regression modules and their synthetic fixture
  support. `source/pyproject.toml` records the separate package dependencies.
- `SOURCE_MANIFEST.json`: exact file hashes, sizes, original relative paths,
  authority classifications, and parent BUILD/plan/archive hashes.

The shared `source/` subtree was deduplicated between phase snapshots. Original
bundle files had that subtree at `source/` alongside `worker.py`; the published
phase directories alone therefore do not reproduce the original bundle layout.
No numerical source was reformatted or rewritten for publication. Repository
lint excludes this immutable snapshot; v4 lint and test rules remain in place.

The package changes add residual-trunk alternatives, categorical complete-catalog
count loss, particle-log-mass increments, and slice-local absolute-weight
calculations that avoid retaining unnecessary full-trajectory tensors. The
separate GPU-native code contains the actual fitting, forecast, qualification,
compiler, ownership, and recovery machinery. Default legacy package semantics
are retained unless the new options are selected.

## Scope and limitations

No biological arrays, count vectors, cell/guide inventories, checkpoints,
particle banks, credentials, live Job manifests, or run output receipts are
included. Operational scripts retain their institutional paths and expect
private, identity-bound inputs and runtime manifests that are intentionally
absent. They are provenance code, **not ready-to-launch public examples**.
Do not execute launch/operate/status scripts as part of installation or tests.

This publication neither retrains models nor replays H100 qualification. It
does not assert that D2 forecast sealing, selection, or D1 assessment completed.
The nine fitting runs and the later D2 forecast phase are separate stages; code
publication is not evidence of scientific acceptance. No LARRY artifacts are
included or admitted. Source-only text may retain exclusion guards.

All-four-donor authorization applies to this new study, not to historical
protected-role decisions. Rest and Stim8hr remain contemporaneous sister
branches. Captured relative representation is not absolute biological growth.

## Local source checks

From this directory, with the dependencies in `source/pyproject.toml` installed
in a separate environment (do not replace the v4 installation):

```bash
python verify_source.py
PYTHONPATH=source/src:tests:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q -c source/pyproject.toml \
  tests/test_native_study.py tests/test_gpu_native.py \
  tests/test_gpu_native_fitting.py tests/test_full_forecast.py \
  tests/test_full_evaluation.py tests/test_d2_forecast_precision.py \
  tests/test_selection.py
cd source
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/test_full_ddr_architectures.py tests/test_categorical_count_contract.py \
  tests/test_particle_mass_increment.py tests/test_absolute_weight_memory.py
```

These selected tests use synthetic fixtures, not cohort observations. Some
other retained operational/packaging tests require private historical receipts
and intentionally are not advertised as self-contained public tests. The
publication validation report records exactly what was run; no omitted test
is silently counted as passed.

The root project license continues to apply. The snapshot's original package
license is retained in `source/LICENSE`.
