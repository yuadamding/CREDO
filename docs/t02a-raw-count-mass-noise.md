# T02A raw-count and relative-mass noise qualification

Last verified: 2026-08-15. Status: authoritative dev22 interpretation amendment
over the immutable dev21 Renz calibration. This is a measurement-noise result,
not model performance or biology.

## Decision

T02A passes. The cohort-neutral implementation completed 100 cell split-half
repeats at each checkpoint and 100 relative-mass catalog bootstraps against
the exact passed T00 population. Every output is content-addressed,
environment-bound, published manifest-last, and fully reverified with the
complete CountStore parent.

The Renz noise ID is
`ca3b8c5c6575a399c6125047530d92a8d337cc0a54cc525dfb5daf9d4e0e5310`.
The derived dev22 amendment ID is
`9f710d3131eb2d9c41e03e311efae7746561a718dab2e2a5257febe344be54c0`;
its verification receipt is
`612487edf62982f970d43f6f5e2c4d3b24c03c5d301cb61ea0e7c7095ee314f4`.
The historical external report was recorded at workspace-relative path
`credo_v4_renz_t02a_noise_20260815/T02A_RUN_REPORT.md`; it is not shipped in this repository.

## Dependency and channel boundary

```text
T00 passed pooled population ──→ T02A raw-count/mass noise ──→ conditional floors

T01 representation ──╳── not read
learned model ────────╳── not read
external biological annotations ─╳── not read
```

T02A uses no representation, state field, diffusion, reaction, ecology,
decoder, optimizer, checkpoint, or particle. Its `pass` status means the
calibration is complete and internally valid. It makes no assertion that the
measured noise is small. The observed P4 and P60 pooled endpoint counts are
read by design to characterize their conditional sampling noise; learned-model
outputs and external biological annotations are not read.

## Raw-count estimand

For each retained guide and checkpoint, cells are randomly split into two
balanced halves. Raw counts are summed independently and compared by:

- Hellinger distance;
- Jensen–Shannon divergence;
- symmetric multinomial deviance per count;
- pseudobulk gene Spearman correlation; and
- top-variable-gene overlap.

The default protocol requires at least 100 deterministic repeats. Guide
metrics are averaged within perturbation target and then equally across
targets. Controls are reported separately, preventing the pooled control
category from receiving the weight of one ordinary perturbation target.

Pseudobulk Spearman and top-gene overlap use a checkpoint-specific frozen
2,000-gene universe selected by guide-pseudobulk `log1p(CPM)` variance. Top-50
overlap uses absolute smoothed log-enrichment relative to the checkpoint-wide
pool. The feature indices and selection scores are exported rather than
silently recomputed later.

## Relative-mass estimand

For checkpoint total `N_t` and Jeffreys-smoothed observed guide probabilities

\[
\widehat p_{g,t}=\frac{n_{g,t}+0.5}{\sum_h(n_{h,t}+0.5)},
\]

T02A samples independent catalogs

\[
n^{(b)}_{\cdot,t}\sim\operatorname{Multinomial}(N_t,\widehat p_{\cdot,t})
\]

and recomputes

\[
y_g^{(b)}=\log\widehat p^{(b)}_{g,60}-\log\widehat p^{(b)}_{g,4}.
\]

The exported floors cover target-balanced interval RMSE, sign stability,
guide and target ranks, and top/bottom-`k` stability.

## Renz calibration result

The exact parents are T00 ID
`7313a9d990900148618edc0fe6a72d5f41e2594641b10d48c278370240964f58`
and CountStore SHA-256
`cb7bd724b0adf3b74ec7141d6aa5605566c2b8f3f9b2f3d846482bc7b7efafd4`.
The population contains 495 guides, 150 perturbation targets, 277,200 cells,
and 34,699 assay-common features.

| Frozen quantity | Value |
| --- | ---: |
| Target-balanced Hellinger q95 | 0.154232 |
| Control-guide Hellinger q95 | 0.231343 |
| Target-balanced Jensen–Shannon q95 | 0.0231605 |
| Target-balanced deviance/count q95 | 0.0459571 |
| Target-balanced pseudobulk Spearman q05 | 0.881957 |
| Target-balanced top-50 overlap q05 | 0.330921 |
| Observed-endpoint sampling RMSE q95 | 0.132578 |
| Guide absolute-error q95, target-median q95 | 0.289973 |
| Sign accuracy q05 | 0.930112 |
| Guide-rank Spearman q05 | 0.978563 |
| Target-rank Spearman q05 | 0.980054 |
| Top-20 / bottom-20 overlap q05 | 0.90 / 0.80 |

The amendment separates the raw-count surface by checkpoint and population:

| Checkpoint/population | Hellinger q95 | JS q95 | Deviance/count q95 | Spearman q05 | Top-50 overlap q05 |
| --- | ---: | ---: | ---: | ---: | ---: |
| P4 targeting | 0.146991 | 0.019783 | 0.039457 | 0.889998 | 0.458207 |
| P4 controls | 0.212830 | 0.037570 | 0.074892 | 0.789110 | 0.220000 |
| P60 targeting | 0.154285 | 0.023201 | 0.046065 | 0.881828 | 0.329571 |
| P60 controls | 0.258018 | 0.053727 | 0.106807 | 0.757615 | 0.160000 |

These thresholds were frozen before learned-model inspection. The two renamed
mass quantities are conditional noise summaries: `0.132578` is not a direct
model-versus-baseline improvement margin, and `0.289973` is not a formal
minimum detectable effect. T07R must use a paired loss-difference bootstrap on
the same resampled catalogs with a separately preregistered comparison margin.
Formal detectability requires an explicit estimand, Type-I error, power, and
detection rule.

Checkpoint-specific consumers must use the derived amendment: T01B uses the P4
source rows and T12 uses the P60 terminal rows. Pooled raw thresholds remain
descriptive. Raw-count Hellinger is not a latent-space tolerance; T02B must
establish that floor after T01 passes. Changing the population, feature scope,
pseudocount, target weighting, or quantile definition creates a new T02A
protocol.

## Public API and CLI

```python
from credo_count_sde_v4 import api

api.qualify_raw_noise(
    output_directory,
    pooled_bundle=t00_directory,
    count_store=count_store_path,
)

api.verify_raw_noise(
    output_directory,
    pooled_bundle=t00_directory,
    count_store=count_store_path,
)

api.amend_raw_noise_interpretation(
    amendment_directory,
    t02a_bundle=output_directory,
    pooled_bundle=t00_directory,
    count_store=count_store_path,
)
```

```bash
credo-v4 qualify-raw-noise \
  --pooled-bundle T00_pooled_data_contract \
  --count-store counts.h5 \
  --split-repeats 100 \
  --mass-bootstrap-repeats 100 \
  --output T02A_raw_count_mass_noise

credo-v4 amend-raw-noise \
  --t02a-bundle T02A_raw_count_mass_noise_v2 \
  --pooled-bundle T00_pooled_data_contract \
  --count-store counts.h5 \
  --output T02A_INTERPRETATION_AMENDMENT
```

Implementation:
[`noise/qualification.py`](../src/credo_count_sde_v4/noise/qualification.py).

## Artifact surface

| Artifact | Purpose |
| --- | --- |
| `raw-count-mass-noise.json` | typed content-addressed bundle |
| `RAW_SPLIT_HALF_METRICS.parquet` | guide/checkpoint/repeat raw metrics |
| `RAW_REPEAT_SUMMARY.parquet` | target-balanced and control summaries |
| `RAW_TARGET_SUMMARY.parquet` | per-target raw stability |
| `VARIABLE_GENES.parquet` | frozen checkpoint-specific gene universes |
| `MASS_BOOTSTRAP_METRICS.parquet` | catalog-level mass/rank stability |
| `MASS_GUIDE_NOISE.parquet` | per-guide mass effect error |
| `MASS_TARGET_NOISE.parquet` | per-target mass effect error |
| `FROZEN_THRESHOLDS.json` | downstream calibration values |
| `TEST_RECEIPT.json` | typed detailed decision |
| `COMPONENT_RECEIPT.json` | common component surface |
| `INPUTS.sha256`, `IMPLEMENTATION.sha256` | parent and code identities |
| `artifacts.json`, `COMMITTED`, `SHA256SUMS` | transactional integrity |

The immutable v2 bundle is not overwritten. The derived amendment contains:

- `THRESHOLDS_BY_CHECKPOINT.parquet` with P4/P60 targeting/control rows;
- `RECOMPUTED_THRESHOLDS.json` with exact non-overstated estimand names;
- `TARGET_RANK_STABILITY.parquet` with the per-bootstrap target-rank Spearman distribution;
- `THRESHOLD_SEMANTICS.json` with allowed and forbidden downstream uses; and
- `IMPLEMENTATION.sha256`, `ENVIRONMENT.json`, `VERIFICATION_RECEIPT.json`, and
  complete hashes.

Full verification now recomputes every legacy frozen threshold from the row-level
tables and checks exact guide/repeat/checkpoint catalogs, target/control mappings,
seed sequences, cell-half reconciliation, mass-bootstrap coverage, and bundle
configuration. Seed ranges must be disjoint, not merely different at their starts.

## Limitations

Cell halves are technical sampling replicates, not mice. Multinomial catalogs
condition on observed pooled frequencies and omit systematic WTA-library,
batch, capture, and biological-replicate variation. Consequently these floors
are necessary protected tolerances but not a complete biological noise model.
T02A does not provide a model-comparison null or a powered detectable-effect
threshold. Its top/bottom overlap values are guide-level. Target-level top/bottom
overlap remains a separate T07R endpoint. T02A does not unblock T02B, T03R, or
pooled dynamics while T01 remains open.
