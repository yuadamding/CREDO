# Review of 0cb50da: linked population-baseline scoring correction

The workflow integration is retained. This revision fixes scoring and reporting;
it does not change the five baseline formulas, refit existing parameters, add
donors, change the RNA/guide universe, or qualify a neural SDE/full cell-state law.

## Reproduced defect and scoring version 2

The actual fit → float32 prediction → evaluation regression reproduced effect
MSE **0.24022650138707496** for an exact endpoint transfer with guide counts
`[0, 10]` and control counts `[5, 5]`, despite negligible composition MSE. A
shared-zero three-gene case also failed (MSE **0.16015100323520742**). The
unequal positive-composition case passed. This is synthetic regression evidence,
not a measured effect on GSE314342.

The predictor already publishes normalized positive mean and reference
compositions. Scoring v1 added another `1e-8` on that side, but only one on truth.
Scoring v2 computes `log(predicted_composition) - log(predicted_reference)`.
Observed guide/control compositions each receive the declared pseudocount once
and normalization before their log contrast. Float32 mean serialization is
renormalized; a float64-tiny floor handles numerical underflow, without another
biological pseudocount. Unsupported rows still abstain. Informative-gene and
sign scores use this same contrast definition; no claim that their real-cohort
values must remain unchanged is made.

Every new evaluation records `scoring_version = 2`, the effect definition,
zero policy and evaluator source hash. Historical v1 outputs are not rewritten.

## Reuse without changing frozen prediction authority

`rescore_baselines` is a separate **score-only** entry point. Its strict
`EvaluationCorrection` V1 binds the original specification and implementation,
new evaluator implementation, immutable prediction seal, existing endpoint
capability and summary, and previous v1 evaluation. It verifies the old artifact
contents and lineage; numerical environment and allowlisted process settings
must still match the original specification. It cannot mint endpoint access,
aggregate new raw counts, refit or publish predictions. Normal runtime checks on
fitting, prediction, access granting and ordinary evaluation remain unchanged.

The new evaluation retains the original prediction specification and adds
`correction.json`, plus `previous_evaluation` and `correction` parent identities.
The no-clobber publisher refuses an existing destination. Corrections use a new
external output directory; old predictions, evaluations, logs and completion
receipts remain authoritative historical artifacts under their original scoring
definition. No checkpoint or prediction schema is silently rewritten.

## Reporting and tests

- Coverage records now separate `query_source_support` (present/absent) from
  `abundance_prediction_basis`. Guide/target endpoint-transfer abundance is
  fitting-endpoint-based regardless of query support. The corrected evaluator
  derives these two columns from the original counts and family definition,
  including when reading a legacy coverage table; it does not modify that table.
- Family-specific conditional cross entropy remains descriptive. The new
  `expression_common_support_conditional_count_score` uses the same all-family
  scored guide intersection for every family, including controls, and reports
  `scored_guides`, `endpoint_RNA_UMIs`, and the UMI-weighted score. The targeting-
  only common-support section reports its own denominator. Empty support yields
  a missing score and zero contributors, never a zero-error claim.
- The deliberate prediction permutation recomputes its HDF5 numerical digest.
  Tests assert changed numerical predictions change that digest; the existing
  endpoint-authority invariance test asserts provenance can change while fitted
  and prediction numerical identities do not.
- Regression cases include sparse compositions, shared zeros, unequal controls,
  actual float32 publication, abstention-dependent common support, unchanged
  prediction/previous-evaluation hashes, no overwrite, and correction identity
  or runtime rejection. The original endpoint-access gates remain exercised.

## Evidence boundaries

The [fresh local receipt](../receipts/review-0cb50da-local-scoring-validation.json)
records **563 passed, four expected CUDA skips and 85.3358% combined coverage**
from a complete single suite, with the unchanged 85% gate. Source and installed-
wheel external canaries passed 11 tests each. Rescoring an actual archived old-code
canary produced the same corrected identity from the source checkout and wheel,
preserving old predictions/evaluation and unchanged non-effect outputs.

The [completed remote CI observation](../receipts/review-0cb50da-remote-ci.json)
records green wheel and Python 3.11/3.12/3.13 jobs for **0cb50da only**. It is not
remote validation of this subsequent correction.

The existing real R48 attempt keeps Rest → Stim48hr, fitting D3+D4, query D2
Rest, evaluation D2 Stim48hr, protected D1, seed 0, context disabled, and no
Stim8hr. Corrections were developed in a separate working copy while the original
process still depended on its frozen implementation hash. Real outputs and
resource measurements remain outside this generic repository, in the cohort
execution report. A valid mean forecast is not evidence of multimodality,
covariance, rare-state recovery, absolute growth, or identified dynamics.

## Real execution closed

The [metadata-only real publication receipt](../receipts/review-0cb50da-real-r48-publication.json)
records completed original execution and linked scoring-v2 correction, both with
exit code 0. It binds the actual resolved specification, adapter/runner identities,
unchanged fitted/prediction numerical digests, old/new evaluations and resource
measurements. Original scoring completed at 23:18:45 UTC and correction at
23:36:15 UTC on 2026-09-12. All 2,053,058 authorized endpoint cells were processed.

The corrected comparison uses 20,634 targeting guides / 11,741 targets, with
21,487 guides and 20,243,714,670 RNA UMIs when including controls in common-support
cross entropy. Abundance keeps all 26,504 guide categories. Numerical fits and
predictions are unchanged. The complete tables and interpretation remain in the
external report; source code is not a substitute for those result artifacts.

Effect magnitudes changed, but sign accuracy and all non-effect expression
columns did not. On endpoint-absent, unscored rows only, the reported informative-
gene count is a computational placeholder and must not be interpreted; always
use the explicit endpoint status and scoring mask. This does not alter any
scored-row informative count or metric. No neural/discovery gate is promoted.
