# Component-wise qualification program

Status: authoritative development order through `4.0.0.dev40`. This page defines
software promotion, not a biological claim.

CREDO V4 is qualified as three dependency tracks rather than one large training
run:

```text
Pooled Renz track:
T00 → T01 → T02B latent noise → T03R shared control/target hierarchy
     → T05R/T06R/T07R → T08R → T09R–T13R

Independent raw-count track:
T00 → T02A raw-count and mass noise floors

Software/synthetic track:
T04 particle engine → T05S drift ───────────────┐
                    └→ T07S reaction ───────────┼→ T08S finite measure
T04D state-dependent diffusion → T06S diffusion ┘
```

T04 is an independent numerical prerequisite for learned dynamics. A failed
component is disabled downstream. More updates, particles, decoder width, or
GPU memory cannot substitute for a failed scientific gate.

T12 is the claim-grade gene-decoding/program stage. T01 asks whether a latent
state preserves stable, count-supported geometry; it must not be made
impossible by requiring full-gene reconstruction to carry the entire
representation decision.

## Universal promotion rule

For lower-is-better loss `L`, a new component must satisfy all of:

1. genuine component-specific null refits control false selection;
2. the target-clustered 95% upper bound of `L(new)-L(previous)` is below the
   negative frozen margin;
3. channel activity exceeds a frozen nonzero floor;
4. every previously passed protected metric remains within tolerance;
5. update 0, the previous model, remains selectable;
6. selection is followed by a fresh all-training-data refit.

Cells estimate guide-level empirical laws. Guides are held-out prediction
units within targets. Targets are the primary uncertainty units. Cells,
guides from one target, and folds sharing targets are never treated as
independent biological replicates.

T04 is a fixed-truth numerical exception to the learned-component rule: it has
no optimizer or post-selection refit. It passes only by matching analytic
drift, OU, constant scalar reaction, fixed pool aggregation, and lifecycle
truths under frozen tolerances. It does not qualify trainable ecology or
trainable coefficient recovery.

## Channel isolation

`ComponentTestContract` enforces the frozen channel matrix:

| Stage | Drift | Diffusion | Reaction | Ecology | Decoder |
|---|---|---|---|---|---|
| T03 | off | off | off | off | off |
| T04 | fixed | fixed | fixed | off | off |
| T05 | trainable | fixed | off | off | off |
| T06 | fixed | trainable | off | off | off |
| T07 | fixed | fixed | trainable | off | off |
| T08 | trainable | trainable | trainable | off | off |
| T09 | fixed | fixed | fixed | trainable | off |
| T10 | fixed | fixed | fixed | off or fixed | off |
| T11 | fixed | fixed | fixed | fixed | off |
| T12 | fixed | fixed | fixed | fixed | trainable |
| T13 | fixed | fixed | fixed | fixed | fixed |

T00–T03 expose no drift, diffusion, reaction, ecology, or decoder channel.
For T04, `Ecology = off` means that trainable ecology is absent. Its separate
fixed-truth check qualifies only mean aggregation from stabilized absolute
series log masses.

## T00 pooled finite-measure contract

The public cohort-neutral API is:

```python
from credo_count_sde_v4.data import build_pooled_finite_measures
```

and the CLI surface is:

```bash
credo-v4 pool-data \
  --cells cells.parquet \
  --guide-catalog guide-catalog.parquet \
  --feature-hashes feature-hashes.json \
  --source-checkpoint P4 \
  --terminal-checkpoint P60 \
  --minimum-source-cells 20 \
  --output T00_pooled_data_contract
```

The input cell table contains `row_id`, `cell_id`, source `sample_id`,
`guide_id`, and `checkpoint`. The catalog contains one immutable
`guide_id → target_id, is_control` mapping. The adapter:

- emits the sole model-facing `sample_id="pooled"`;
- preserves source sample identity only as provenance;
- makes row order irrelevant through canonical sorting;
- freezes eligibility from source counts only;
- fails, rather than changing eligibility, when an eligible guide lacks a
  terminal empirical law;
- requires one exact feature-order hash at both checkpoints;
- assigns every retained cell to exactly one guide/checkpoint measure;
- uses `(n + 0.5) / sum_g(n_g + 0.5)` relative masses;
- verifies atom weights sum to their declared finite-measure mass;
- publishes through manifest-last atomic directory creation.

The generated directory includes `TEST_CONTRACT.json`, `INPUTS.sha256`,
`CONFIG.yaml`, null/bootstrap/channel/model status files, per-guide and
per-target metrics, `TEST_RECEIPT.json`, and `SHA256SUMS`. T00 has no learned
model; its null calibration and bootstrap entries are explicitly
`not_applicable` rather than silently omitted.

## Current status

| Stage | Package status | Pooled real-data status | Promotion |
|---|---|---|---|
| T00 | implemented; eight focused tests pass | external Renz receipt passes | T00 only |
| T01 | v1/v2 retired; v3 not implemented | both failed; dimension 0 in 4/4 folds | general test open; pooled path blocked |
| T02A | implemented and fully verified | Renz 100-repeat receipt passes | thresholds frozen; no model promoted |
| T02B | latent primitives only | not run | blocked by T01 |
| T03 | identifiability/null hardening exists as engineering code | not run under this ladder | blocked |
| T04 | implemented and fixed-truth qualified | cohort-independent | synthetic T05S/T07S eligible |
| T04G | specification only | not run | required before CUDA scientific use |
| T04D | not implemented | not run | required before full state-dependent T06S |
| T05S | specification only | not run | independently eligible after T04 |
| T07S-A | implemented; duration-correct R0/R1 passed | cohort-independent | T07R-A0 design eligible |
| T07S-B | not run | cohort-independent | state-dependent reaction remains unqualified |
| T07R-A0 | implemented; estimator parity passed, real fold failed | pooled Renz fold 0 | real target-reaction adapter retired; no scale-out |
| T05R–T13 | partial numerical/model primitives exist | not run under this ladder | dependency-specific |

The external T00 Renz receipt retains 495 source-eligible guides (445
targeting and 50 controls), 150 perturbation targets, and 277,200 cells. Both
checkpoint mass sums are exactly one. This establishes only data semantics.
It does not qualify representation, transport, mass reaction, ecology,
counterfactuals, decoding, or biological claims.

The first T01 candidate used multinomial/Hellinger low-rank factors with
dimensions 8, 16, 32, and 48. The global gene-frequency decoder (dimension 0)
was selectable and won every fold. The learned candidates were worse by
0.204–0.256 nats per count on P4-only inner validation. T01 v1 is therefore
`fail_retired`; it was not refit, P60 support was not inspected, and T02B remains
blocked. The failure receipt is retained outside the package in
`credo_v4_renz_t01_representation_20260815/`.

The separately frozen centered successor represented the global square-root
gene frequency explicitly and fit only residual Hellinger structure. It also
selected dimension 0 in every fold: learned dimensions 8, 16, 32, and 48 were
0.178–0.238 nats per count worse than the global gene-frequency decoder. T01
v2 is also `fail_retired`; no post-selection refit or P60-support diagnostic
was eligible. The two immutable failures block T02B and the pooled Renz
dynamics path, but not T02A or T04–T07S. Changing the loss, threshold, or
candidate family would be a new T01 protocol, not continuation of either
failed test. See the
[detailed T01 implementation and result record](t01-representation-qualification.md).

T04 now routes analytic fixed truths through the same streaming particle
kernel used by inference. Its committed qualification uses 50 seeds over the
64/256/1,024/4,096-particle and 8/16/32/64-step OU grid. Deterministic drift
was exact, largest-grid OU variance error was 0.004459, reaction relative
error was below `6e-16`, fixed absolute-weight pool aggregation matched exactly, the
normalized-within-guide negative control failed as required, and interrupted
resume was bitwise identical. See the
[detailed T04 record](t04-particle-engine-qualification.md).

T07S-A trains only target constant-reaction contrasts through the production
complete-denominator count likelihood while every state, ecology, decoder,
pool-intercept, and concentration channel remains fixed. Fifty-nine
zero-reaction refits froze the false-improvement margin; 60 independent null
audit refits produced zero false promotions (one-sided 95% upper bound
0.048703 under the frozen 0.05 gate). Nonzero checkpoints were internally
selected in 9/59 calibration and 4/60 audit refits, so this is a false-
promotion—not false-selection—guard. The nonzero R1 test selected update 100
and was freshly refit. Dev25 reuses that model without optimizer work and
correctly evaluates both R0 and R1 in duration-integrated endpoint units. Its
target-balanced RMSE is 0.033191 versus 0.706508 for zero reaction, a delta of
-0.673317 with target-bootstrap interval [-0.829965, -0.482187]. Raw reaction RMSE was
0.017233 and sign accuracy was 1.0. This qualifies learned constant scalar
reaction in synthetic complete catalogs only. See the
[dev23 training record](t07s-reaction-recovery.md) and authoritative
[dev24 R1 amendment](t07s-reaction-metric-amendment.md) and authoritative
[dev25 unified R0/R1 amendment](t07s-null-interval-amendment.md).

T07R-A0-v1 then tested only pooled relative-guide representation, without a gene
representation, drift, diffusion, within-guide selection, ecology, decoder, or
GPU. Its independent penalized-DM reference and production implementation
agreed to `7.82e-08` maximum probability error and `9.53e-07` maximum target-
effect error. On the predeclared real fold, however, update 0 won inner
validation. The resulting production/source-persistence loss was `0.0230634`
versus `0.0224217` for the sister-guide target reference. The paired delta was
`+0.000641766`, with interval `[+0.000621024,+0.000662385]`, far above the
training-only `1e-08` noninferiority margin. T07R-A0 is therefore
`fail_retired`; the stop rule forbids more updates or four-fold scale-out. See
the [exact dev26/v1 and dev27/v2 record](t07r-a0-pooled-likelihood.md).

Dev27 corrects v1's fold-subcomposition concentration without reopening any
fresh endpoint. The v2 physical-pool conditional DM computes every guide's
source probability in the complete 495-guide pool, then scores an active fold
with `alpha_active = 1000 * p_full[active]`. Estimator parity and DM
factorization pass. Update 0 still wins, and M1 remains superior under both the
conditional multinomial primary interval and conditional DM sensitivity. This
is historically exposed forensic evidence. It retires the T07R-A0 selector,
not the numerical reaction estimator or constant target reaction in general.

T02A consumed the exact passed T00 population and complete common-34,699 raw
CountStore. One hundred balanced cell split-halves at both checkpoints and 100
Jeffreys-smoothed multinomial catalog bootstraps completed. The frozen
target-balanced observed-endpoint sampling RMSE q95 is 0.132578, and the
guide absolute-error q95 target-median q95 is 0.289973 natural-log units.
Neither is a model-improvement margin or formal minimum detectable effect. T02A pass
means the measurement-noise calibration is complete; it does not promote a
model or authorize pooled state dynamics. See the
[detailed T02A record](t02a-raw-count-mass-noise.md).

Synthetic values reported outside a committed component directory are useful
design evidence but are not promotion evidence. Every subsequent stage must
publish the same typed receipt surface before downstream use.

## Resource and stop policy

T00–T04 run on CPU. The first real T03 fold is limited to one GPU and at most
250 updates. T05–T11 synthetic tests use CPU or one small GPU. T12 is sized
independently only after dynamics pass. Four-fold out-of-fold scale-out is
reserved for T13 after every required parent component passes.

There is no target VRAM requirement. Peak memory is an engineering consequence
of the qualified model, never a selection metric.
