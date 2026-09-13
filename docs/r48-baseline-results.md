# Frozen R48 baseline result: review of 9ef72c6

This is the numerical result behind the accepted first-split baseline milestone,
not another baseline revision. The original fits, predictions, five definitions,
coverage and scoring-v2 evaluation are unchanged. The complete numerical summary
and paired comparisons are retained in the portable
[result receipt](../receipts/review-9ef72c6-baseline-results.json).

The experiment is Rest → Stim48hr; fit D3+D4, query D2 Rest, evaluate D2
Stim48hr, D1 protected, seed 0, context off, no Stim8hr. D2 has now been observed
for development; further selection on it must be labeled accordingly.

## Numerical comparison

Expression errors below use the same 20,634 targeting guides / 11,741 targets.
They average sister-guide errors within target, then average targets. Cross
entropy additionally includes 853 controls: the identical 21,487 guides and
20,243,714,670 RNA UMIs for every family. Abundance retains all 26,504 catalog
categories and all 2,053,058 captured endpoint cells, not just expression support.

| Family | Composition MSE | Hellinger² | Effect MSE (v2, natural-log contrast) | Common CE (nats/UMI) | Full-catalog abundance log-frequency RMSE |
|---|---:|---:|---:|---:|---:|
| Source persistence | 1.716805e-8 | 0.092211 | 4.796085 | 8.705653 | 0.727484 |
| Fitting control response | 8.128964e-9 | 0.061751 | 4.796232 | 8.516848 | 0.727484 |
| Guide endpoint transfer | 5.716263e-9 | 0.041468 | 4.275944 | 8.486671 | 1.848327 |
| Target endpoint transfer | 5.143902e-9 | 0.037514 | 4.181186 | 8.471490 | 2.006291 |
| Hierarchical source response | 2.731480e-7 | 0.190996 | 7.024234 | 8.835611 | 0.734491 |

There is no single winner across all measurement targets. Target transfer is
strongest on common-support mean composition and average effect magnitude.
Persistence/shared-control mass wins abundance. The tested hierarchical family
does not improve expression; its abundance is better than the fitting-endpoint
transfers but slightly worse than persistence. This is not a general proof that
source conditioning is useless: it is a negative result for this fixed family.

## Three questions answered with paired target differences

Differences are **candidate minus reference**; negative means lower error.
The following intervals are post-hoc descriptive percentile target bootstraps
(2,000 draws, PCG64DXSM seed 0), after averaging sister guides within each target.
They are **conditional on this one observed D2 validation donor**, not donor-level
replication, simultaneous intervals, or a multiplicity-controlled discovery test.

| Candidate − reference | Metric | Mean difference | Conditional 95% interval | Targets with lower error |
|---|---|---:|---:|---:|
| Control response − persistence | Composition MSE | −9.039085e-9 | [−9.132344e-9, −8.941221e-9] | 98.08% |
| Control response − persistence | Effect MSE | +0.000147 | [−0.000431, +0.000709] | 50.84% |
| Target transfer − guide transfer | Composition MSE | −5.723610e-10 | [−6.138043e-10, −5.345379e-10] | 80.37% |
| Target transfer − guide transfer | Effect MSE | −0.094758 | [−0.112770, −0.077811] | 47.18% |
| Hierarchy − target transfer | Composition MSE | +2.680041e-7 | [+2.505786e-7, +2.862645e-7] | 0.23% |
| Hierarchy − target transfer | Effect MSE | +2.843048 | [+2.728815, +2.955858] | 5.97% |

1. The shared stimulation response helps mean composition substantially, but
   provides essentially no average gain in control-referenced effect magnitude.
   Its abundance is exactly persistence by construction.
2. Target sharing is preferable to same-guide transfer on these average RNA
   scores. However, lower mean effect error occurs despite fewer than half of
   individual targets improving: gains are not uniform. This does not establish
   guide-specific effect recovery or justify discarding sister-guide comparisons.
3. The tested hierarchy adds no useful RNA increment over transfer here. Do not
   tune its shrinkage or support against D2 and relabel that as this frozen run.

For abundance, the separate paired analysis uses all 12,779 targeting targets,
including zero-capture guides, and compares target-macro **squared** log-frequency
error (not the complete-catalog guide-weighted RMSE in the first table). Target
transfer minus guide transfer is +0.593382, interval [+0.550201, +0.635335].
Thus, the direction of their abundance comparison is opposite to their mean RNA
comparison. Control response minus persistence is exactly zero.

## Coverage is a result, not a filter to hide

| Family | Expression predictions | Scored targeting guides | Scored controls | All scored guides |
|---|---:|---:|---:|---:|
| Persistence | 22,371 | 20,787 | 853 | 21,640 |
| Control response | 22,371 | 20,787 | 853 | 21,640 |
| Guide transfer | 22,109 | 20,634 | 853 | 21,487 |
| Target transfer | 22,269 | 20,727 | 853 | 21,580 |
| Hierarchy | 22,371 | 20,787 | 853 | 21,640 |

The common intersection is 81.0708% of the full catalog. Its 5,017 excluded
categories were not removed from the package or abundance denominator, and are
not necessarily unsupported by every family. Keep full support/abstention tables
when comparing learned models: report both common-support performance and any
additional supported predictions. Source absence is not demonstrated extinction.

## Preservation and interpretation

The corrected evaluation identity is
`0189bbf0b58c6579ad6c586c86b8a7f8766b70aaaa685dae2ccabed300e31026`.
The result review verified its immutable bundle and read only the existing keyed
scores, never raw counts or new donor data. The external paired-target table and
review script are hash-bound in the receipt. Original baseline artifacts were
neither overwritten nor rerun.

These are population-mean RNA compositions and relative captured abundance.
They do not establish heterogeneity, covariance, rare-state recovery, molecular
causality, absolute growth, or identified drift/diffusion/reaction. Rest is not
a measured physical-time-zero population. The next component is the separate
[fold-fitted count representation](fold-count-representation.md), not more tuning
of this baseline layer.
