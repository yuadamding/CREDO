# Source-only paired-condition population baselines

See the [scoring-v2 correction](review-0cb50da-scoring-correction.md) before
interpreting effect magnitudes from historical v1 evaluations.

This is the separate R48 integration requested by the review of `21656d8`.
The four immediate static-program corrections are closed. The additional
[remote CI observation](../receipts/review-21656d8-remote-ci.json) records the
completed green Python 3.11/3.12/3.13 and wheel jobs for that exact base commit;
it does not validate these subsequent working-tree changes.

## Scope and interfaces

`forecast/` implements population-mean baseline fitting and independent endpoint
evaluation, not a count representation, neural SDE, cell-state distribution,
variance prediction, or biological discovery calibration. Cohort adapters and
results remain outside this repository. No existing checkpoint or model channel
is changed. The static evaluator is not used as a source-only forecasting runner.

The external first-split specification is Rest → Stim48hr, fit D3+D4, query D2
Rest, evaluate D2 Stim48hr, protect D1, seed 0, and context off. Stim8hr is excluded.
All canonical RNA features and every guide category are retained; no endpoint-
derived feature selection, source-support filtering of abundance, or cell cap
is applied. Local CPU execution is appropriate for these sparse sufficient-
statistic baselines; it is not H100 training qualification.

1. `ForecastSpec` freezes code/runtime, roles, prepared views, RNA positions and
   names, guide order, transforms, support rules, resource limits, and an opaque
   endpoint-view artifact identity. It contains no endpoint observations.
2. `aggregate_view` reads the exact authorized integer CSR rows and emits a
   complete-source summary. Fitting sees only fitting capabilities.
3. `fit_baselines` accepts fitting summaries and a guide catalog. Predictions
   accept only a fitted artifact and source-query summaries.
4. Predictions publish once, with keyed coverage/abstentions, compositions,
   independently predicted control references, and complete-catalog abundance.
5. Only a fully verified prediction publication permits the external adapter
   to mint a distinct `PreparedEvaluationAccess`. It is bound to that prediction
   seal. Endpoint aggregation and evaluation require this seal again.

This is an application-level information-flow guard. Shared local filesystem
access is **not physical storage isolation**; hostile code could bypass Python
interfaces. No storage-isolation or remote-deployment claim is made.

## Frozen baseline definitions

RNA expression is an **equal-cell mean composition** over positive-RNA cells,
not a pooled-UMI mean. Technical features never enter RNA depth. There is one
mean per guide, with no source/endpoint cell pairing. Source cells lacking RNA
remain in captured abundance but not the RNA-composition mean.

| Family | Expression prediction | Captured-abundance prediction |
|---|---|---|
| Source persistence | D2 source mean | D2 source frequency |
| Fitting control response | Source mean × fitting control log-response | Source frequency; common mass multiplier cancels |
| Guide endpoint transfer | Mean available fitting-donor endpoint means for the guide | Mean fitting-donor endpoint frequencies |
| Target endpoint transfer | Equal-guide target mean of the preceding guide means | Fitting target mass divided across its designed guides |
| Hierarchical source response | Source mean × paired guide response, shrunk toward its target response | Source frequency × shrunken paired fitting log-frequency response |

Expression responses use natural logs with pseudocount `1e-8`, followed by
normalization. Shrinkage is `n/(n+32)` where `n` is summed fitting paired support
(minimum source/endpoint cells per donor; positive-RNA cells for expression).
Target expression responses average available paired guide responses equally;
missing target response falls back to the fitting control response. The
hierarchical control-guide response is explicitly restricted to the fitting
control response. Fitting-donor means receive equal available-donor weight.
These are predeclared baseline rules, not validation-selected hyperparameters.

Controls are explicitly catalog-bound. Expression references pool positive-RNA
control cells within each donor/condition, then average fitting donors equally.
Persistence uses the source-query control mean; control-response and hierarchy
transform it using the fitting control response; endpoint transfers use the
fitting endpoint-control mean. No observed endpoint control enters prediction.
Missing required control reference fails the workflow rather than manufacturing
a reference from endpoint truth.

Source-absent or zero-RNA-source guides have explicit expression abstentions.
Guide/target endpoint transfers also abstain when their fitting support is absent.
The hierarchical family records its guide/target/control fallback. Abundance
always uses all bound categories with Jeffreys frequencies
`(n_g + 0.5)/(N + 0.5 G)`, including zeros. Source-absent mass is labeled a prior,
not source-conditioned expression evidence. Cohort-wide prevalence filtering is
not used to redefine `G`.

Coverage separates query source support from family-specific abundance provenance;
endpoint transfers use fitting-only frequencies even when query source cells exist.

## Evaluation and interpretation

The evaluator emits one row per guide/family for expression and abundance,
including unsupported rows and reasons. It reports equal-target macro averages
of within-target guide means, separates controls, and provides an all-family
common-support comparison. Expression MSE, squared Hellinger distance,
`log1p(10000 × composition)` MSE, endpoint/control log-effect MSE and informative
sign accuracy are distinct metrics. The informative threshold is absolute
observed natural-log effect ≥ 0.05. A missing metric is not zero error.

Observed composition receives the same `1e-8` pseudocount normalization for
composition scoring. Effect truth is the observed endpoint guide composition
relative to its observed endpoint control; predicted effects use the independently
predicted reference. This evaluates endpoint perturbation contrasts, not absolute
growth, molecular knockdown efficiency, or tracked-cell trajectories.

Scoring v2 takes logs of the already-positive published prediction/reference
without a second pseudocount. Both observed compositions are smoothed once and
normalized. Historical scoring-v1 outputs require a separately linked correction.

Observed raw RNA totals are read only after publication for conditional
multinomial cross entropy. Factorial constants are omitted and the score is
RNA-UMI weighted. It is **not** unconditional count likelihood or a cell-level
distribution/variance score. Abundance reports full-catalog log-frequency RMSE,
conditional cross entropy and Jeffreys KL, plus source-support strata without
renormalizing within strata. Interval-log-effect error uses the same fixed
source frequency and therefore equals endpoint log-frequency error.

Family-specific cross entropy is descriptive. Direct family comparisons use the
new common-support conditional score, with its identical guide population and
explicit contributing-guide and RNA-UMI denominators.

One inner-validation donor is not outer-donor generalization or biological
discovery qualification. No result here lifts discovery, dynamics, GPU, or
physical-isolation gates.

## Resources, immutability, recovery, and validation

Sufficient statistics remain in memory per source and publish once as HDF5.
This avoids a potentially enormous set of dense per-shard intermediates.
The external runner uses four CPU workers, 2,048-row batches, a 64 GiB sampled
process-tree RSS cap, and a 128 GiB output cap. It checks conservative numerical
estimates and available disk first; actual one-second samples and worker high-
water marks are retained. A sampled peak is not a continuous exact peak.
The actual process environment (NumPy huge-page advice, thread counts and visible
devices) is also frozen and checked at every stage. On the first local host,
disabling NumPy's huge-page advice removed measured allocation/compaction stalls;
the external runner requires `NUMPY_MADVISE_HUGEPAGE=0` at process launch. This
does not alter the scientific baseline formulas or configure the host kernel.

Completed sources are fully hash-verified and reused. An interrupted incomplete
source is replayed from its first shard, in canonical shard/row order. This is
source-level sufficient-statistic recovery, **not optimizer-resume equivalence**.
The publisher refuses overwrites and retains immutable specification/parent
identities. Operational logs and lock files are outside immutable stage bundles.

The connected synthetic tests cover fit → publication → reload → independent
evaluation, null responses, missing source/fitting/RNA support, complete guide
denominators, shuffled biological predictions, endpoint expression/control/mass
invariance, two-worker execution, interrupted-source recovery, changed transforms
and roles, corruption, resource-budget rejection, and no-clobber publication.
The external adapter canary exercises the same complete workflow and delayed
endpoint resolution against a synthetic prepared-package schema. Real outcomes
and final resource measurements belong in the external execution report, not
in a synthetic-test success claim.

The [fresh local validation receipt](../receipts/review-21656d8-local-r48-validation.json)
records 558 passing tests, four deliberate CUDA skips and 85.3288% combined
statement/branch coverage from one complete suite, plus the connected external
and installed-wheel canaries. It records the real workload as running at its
observation time, not as a completed biological or performance result.
