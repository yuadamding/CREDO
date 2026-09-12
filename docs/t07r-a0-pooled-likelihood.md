# T07R-A0 pooled relative-guide likelihood

Last verified: 2026-08-16. Status: authoritative dev27 forensic record;
historically exposed development evidence. Scope: one Renz/GSE235325 guide-
within-target fold, CPU only.

## Final disposition

Two separately versioned records now exist:

| Version | Method | Result | Disposition |
| --- | --- | --- | --- |
| dev26/v1 | `fold_subcomposition_fixed_concentration_dm_v1` | update 0 selected; M1 superior | immutable `fail_retired` selector/protocol |
| dev27/v2 | `physical_pool_conditional_dm_likelihood_v2` | update 0 selected; M1 superior | forensic correction; no promotion |

V1 qualified numerical agreement between independent SciPy and production
estimators, but normalized each fold subset and reset its concentration to
1,000. It was therefore a fixed-concentration fold-subcomposition likelihood,
not a complete-physical-pool DM test. The external dev26 bytes are unchanged.
Its selector and protocol are retired; its numerical estimator and constant
target-reaction family are not broadly retired by that result.

V2 computes probabilities from the full 495-guide P4 physical denominator,
reads terminal counts only for the active fit/validation/evaluation mask, and
inherits the active concentration

\[
\kappa_A=\sum_{g\in A}\alpha_g
        =1000\sum_{g\in A}p_g.
\]

Fold 0 was already inspected, so v2 is explicitly
`forensic_estimand_correction` / `historically_exposed_development`. It is not
fresh validation, a biological claim, or authorization for more Renz folds.

## Frozen v2 contract

- parents: exact passed T00, T02A raw bundle, dev22 T02A amendment, and dev25
  T07S-A amendment, with contract IDs, file hashes, manifests and P4/P60
  checkpoint identities cross-linked;
- 495 retained guides in one physical `pooled_P4_to_P60` catalog;
- full P4 source denominator for every objective; terminal masks folds 2/3 for
  fitting, fold 1 for selection, fold 0 for exposed evaluation;
- updates `0, 25, 50, 100, 200`, with update 0 selectable;
- fresh zero-initialized refit on all non-outer guides after selection;
- controls exact zero, pool reference fixed zero, smoothing 0.5, full
  concentration 1,000, ridge 0.05;
- every inner targeting guide must have a fit-fold sister and every outer
  targeting guide a non-outer sister;
- no representation, state transport, diffusion, selection, ecology, decoder,
  particles, GPU, or additional fold;
- two 4,000-draw uncertainty surfaces: a conditional multinomial catalog
  bootstrap and a conditional DM sensitivity using the inherited
  concentration.

The primary predictive rule is independent of numerical parity: M2 is superior
only when the primary interval upper bound is below zero; M1 is superior only
when its lower bound is above zero; otherwise the result is inconclusive.

## Dev27 numerical result

The full-DM conditional factorization error is `1.36e-14` under a `1e-12`
tolerance. Independent SciPy and production estimators agree on a training-only
synthetic catalog:

| Numerical check | Result | Tolerance |
| --- | ---: | ---: |
| Maximum guide-probability error | `8.1784e-09` | `<1e-6` |
| Maximum target-effect error | `7.7646e-07` | `<5e-6` |

Update 0 again wins inner validation (`0.026048` versus approximately
`0.026106` for updates 25–200). M2 is therefore source-frequency persistence;
M1 is the fully fitted non-outer sister-target reference.

| Exposed outer-fold metric | Result |
| --- | ---: |
| M2 selected-policy NLL/count | `0.0205660602` |
| M1 sister-target NLL/count | `0.0204858983` |
| M2 − M1 | `+0.0000801619` |
| Conditional multinomial 95% interval | `[+0.0000762546,+0.0000843557]` |
| Conditional DM 95% sensitivity | `[+0.000115735,+0.000232373]` |

Both uncertainty surfaces favor M1. The physical-denominator correction reduces
the point gap by roughly eight-fold relative to v1 but does not change the
qualitative selector conclusion.

Sister support passes: inner guides have 1 sister for 74 guides and 2 for 38;
outer guides have 1 sister for 2 guides and 2 for 108. No scored targeting guide
has zero permitted sisters.

## Interpretation

The corrected result supports a narrow statement:

> In one historically exposed Renz fold, the nested selector again chose
> persistence, and that policy was worse than a sister-guide target reference
> under a physical-pool conditional DM estimand.

It does not establish that constant target reaction is numerically invalid.
The independent and production estimators agree when optimized. It establishes
that the tested nested selection policy does not promote a useful real Renz
policy. More LBFGS updates, GPU memory, particles, or H100 time cannot address
that failure.

## Evidence and reproduction

The dev27 evidence authority was recorded at workspace-relative path
`credo_v4_renz_t07r_a0_v2_20260816/`. The dev26 predecessor was recorded at
`credo_v4_renz_t07r_a0_20260816/`. These historical external artifacts are not
shipped in this repository; resolve them through the project evidence index.

```bash
credo-v4 correct-pooled-reaction \
  --pooled-bundle /path/to/T00_pooled_data_contract \
  --t02a-bundle /path/to/T02A_raw_count_mass_noise_v2 \
  --t02a-amendment /path/to/T02A_INTERPRETATION_AMENDMENT_dev22_r2 \
  --t07s-amendment /path/to/T07S_NULL_INTERVAL_AMENDMENT \
  --fold-assignment /path/to/guide_fold_assignment.parquet \
  --output /new/path/T07R_A0_PHYSICAL_POOL_CONDITIONAL_DM_V2_R2
```

The verifier checks every artifact and parent cross-link, recomputes the
selection, refit, parity, factorization and both bootstrap surfaces, and reloads
the selected tensor file.
