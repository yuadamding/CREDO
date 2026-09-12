# Review of b45330b: CI and evaluation integration

Date: 2026-09-12. Base: `b45330be0d39c39b16309dd7e9d162cd121c9199`.
Review attachment SHA-256:
`f3a98b4cf70f16f2e1f4645116a26426fcfd40fc1071f10672b4b4091d109fe3`.

This follow-up accepts the review's four immediate corrections. It retains the
guide-catalog binding, row permissions, keyed metric definitions, separate NB
scores, checkpoint shapes, scientific thresholds, and 85% coverage requirement.
It does not reopen V4's overall design or implement the R48 forecasting run.
Original CREDO, historical receipts, and cohort input packages remain untouched.

## Reproduced failure boundaries

The exact-commit [GitHub Actions run 34700035365](https://github.com/yuadamding/CREDO/actions/runs/34700035365)
was inspected directly. Integrity, formatting, and lint passed; wheel passed.
Python 3.11 reached tests and reported 2 failed, 524 passed, 4 skipped and 84.92%
coverage. Python 3.12/3.13 stopped at the redundant CSR cast and dynamic NumPy
`savez` keyword expansion typing errors. This is evidence for the base commit,
not a CI result for this follow-up.

The source also confirmed two integration defects: held-out-guide/target masks
provided no controls to the corrected sign evaluator, and outer prediction
still moved the complete evaluation batch to the device.

## Corrections

### Workspace and numerical-library typing

The compatibility tests now resolve the workspace through
`verify_module._workspace_root()`, the same function as the CLI preflight.
An explicit `CREDO_V4_WORKSPACE_ROOT` disables the optional local skip: missing
checkout/archive evidence fails. Nested `work/CREDO/CREDO` tests exercise the
configured root with and without Git. Neither the frozen commit nor archive
hash or verification algorithm changed.

`SparseCountBatch.library_sizes` uses a typed local array rather than a
version-dependent cast. Compiled publication supplies eleven fixed keyword
members to `np.savez`. Regression tests compare historical expansion with the
new archive's ordered member names, array values, dtypes and semantic hash,
then exercise the unchanged loader. ZIP timestamps are not an archive-byte
identity guarantee.

### Independent observed evaluation references

The split harness explicitly returns fitting rows, evaluation target rows, and
evaluation reference rows. For guide/target splits, a seeded outcome-blind
25% reservation within donor/condition is made before inner splitting; rounding
is upward, with at least one fitting control retained. Zero/singleton-control
strata supply no reserved reference. Donor/time splits use their naturally
held-out controls. Invalid indices, overlaps and noncontrol references fail
before fitting.

Reference outcomes are evaluator-only within each outer split. Neither that
split's fitting nor baseline estimation nor model reference prediction sees
them. The later all-data reference/stability diagnostics remain separate;
they are not held-out predictive evidence. Likelihoods use target rows only;
sign truth uses matched reserved controls. Independent predicted controls still
come from the fitted head's masked reference. Missing support remains undefined.

Metric detail revision 3 records reservation settings, row counts, a physical
partition digest, target-only likelihood support and the output budget. The
keyed sign formula itself remains revision 2. Existing V1 protocol/split summary
objects remain legacy summaries: the hash-bound detailed artifact supplies
these additional execution semantics. They are not protected-access contracts.

Reserving cells does not make donors, cultures or libraries independent. The
static model still conditions on each evaluation target's observed library
offset and supplied state covariates. This is not source-only forecasting.

### Chunked evaluation

A shared iterator bounds inner-validation, outer-mean and independently
predicted-reference tensors by minibatch rows and output bytes. Outer scores
accumulate immediately. Equal-cell compositions are summed by
donor/condition/guide and control stratum, preserving cell counts and macro
aggregation. The six baselines are fitted once at unit depth for unique
target/checkpoint pairs and expanded only for the current scoring chunk.

The evaluation budget (default 64 MiB) accounts for retained numeric keyed
summaries, baseline tables, and a prediction/reference chunk before fitting;
the accumulator also checks new allocations. It is not peak RSS or peak VRAM.
Input arrays, temporary work, Python metadata, model activations and allocator
caches remain outside that byte accounting. The 2,048-gene/512-MiB host diagnostic
limits still apply. This is not a scalable full-cohort training implementation.

## Validation

The [local validation receipt](../receipts/review-b45330b-local-validation.json)
binds corrected package SHA-256
`cf5c76ccf031b05af0b38606d737ed2739b03a61a554c1734117d3906afd6965`
and the 162-file source/test/script/config inventory. It records evidence
collected before the follow-up commit, not a fabricated commit-bound CI result.

| Check | Measured result |
| --- | --- |
| One fresh complete CPU suite | **543 passed, 4 CUDA skips, 0 failures**, 859.46 seconds |
| Coverage | **85.1904%** combined statements/branches; unchanged 85% gate passed |
| Warning | One existing near-constant-correlation warning in a pooled evaluation test |
| Focused compatibility/program checks | 44 passed; 17 cases added across the final full test inventory |
| Archive/store regression checks | 2 passed; member names, arrays, dtypes and reload hash preserved |
| Formatting, lint, schemas, documentation | Passed; 159 Python files formatted |
| Strict mypy | Passed for 99 source files with each tested NumPy version |
| Build and installed wheel | Audit, 192-cell/four-update synthetic lifecycle and full reload passed |

The complete suite used a **new coverage database without append**, not a merger
of separate test runs. Branch-only coverage was 69.9263%; the configured gate
applies to the combined statement/branch total. The four skips are explicitly
CUDA-only checks, not compatibility skips. No fresh remote CI pass is claimed
before publication and completion of the Python matrix.

The installed wheel's package hash matched the checkout. Its SHA-256 is
`3cb04f4b0eed8fc247076567db45c843882e7eb385cb5c59362674db25925f6e`.
It reused existing dependency site-packages in a fresh wheel environment and is
not a claim of a fully locked or freshly resolved qualification environment.

Focused integration evidence includes known-correct sign accuracy of 1 and
swapped-effect accuracy of 0 in both guide and target splits; fitting/predictor
row tracing; reserved-outcome mutation with unchanged fitted state and
predictions; dense-versus-chunked likelihood and keyed-summary equivalence;
transfer bounds through `_evaluate_split()`; fail-before-fitting output limits;
and explicit missing-reference behavior. Scientific promotion stays blocked.

Strict typing passed for all 99 source files with NumPy 2.2.6, 2.4.6 and 2.5.3.
The two newer NumPy versions were installed into isolated overlays of the
existing CPython 3.13.2 environment; each passed 31 focused metric, split,
archive and CSR tests. These checks isolate the numerical-library typing
changes but do not reproduce GitHub's three Python interpreters or fully fresh
dependency resolution. The existing conda environment was not modified.

## What remains contained, not solved

- Null discovery calibration remains unimplemented and
  `null_inclusion_calibrated` stays false; no new scientific pass is possible.
- The guide multiplier remains an unidentified latent scale, not knockdown
  efficiency. No new centering or biological efficacy anchor is introduced.
- Row allowlists are application guards; physical shard decompression is not
  worker isolation. No protected-access claim is promoted.
- The existing CI matrix still resolves supported dependencies independently.
  A locked qualification environment alongside an unlocked compatibility job
  remains a separate follow-up; historical lock files alone do not prove such
  a qualified environment. This patch does not suppress newer-dependency errors.

## Next gate: the separate R48 baseline workflow

After a complete fresh CI pass, the next experiment remains:

```text
Task: Rest -> Stim48hr
Fit: D3 + D4
Inner validation: D2
Outer protected donor: D1
Seed: 0
Stim8hr input: none
Context: disabled
```

Generate immutable D2 predictions from permitted fitting/query inputs before
a separately scoped evaluator opens D2 Stim48hr. Report ordered guide/feature
alignment, coverage, explicit abstentions, baseline errors and invariance to
protected-truth changes. Keep D1 closed. Neural superiority is not required for
valid execution. No real-cohort expression or new training is needed to close
this correction's synthetic engineering tests.
