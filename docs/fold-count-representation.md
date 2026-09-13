# Streamed fold-fitted count representation

Status: successor implementation with synthetic integration tests; **not a
trained GSE314342 representation, production GPU qualification, or dynamics
promotion**. The accepted baseline/scoring milestone remains `9ef72c6` and its
[frozen numerical result](r48-baseline-results.md). No old checkpoint schema,
baseline formula, historical T01/E2 record, or existing `credo-v4` lifecycle
has been repurposed.

The [bd2a639 follow-up](review-bd2a639-count-qualification.md) advances the
representation-only specification and bundle to V2. V1 schemas and historical
results remain immutable; replay them with their archived source/wheel.

## Implementation and fixed starting configuration

The separate `credo_count_sde_v4.count_representation` package connects the
prepared integer CSR reader to a learned encoder/decoder:

| Surface | Implemented behavior |
|---|---|
| Input authority | Complete canonical feature identities and RNA mask must hash to the prepared access; technical features never enter RNA depth or the model |
| Encoder | Sparse `log1p(10000*A/RNA_depth(A))` first projection; default F → 512 → 128 → 48, GELU and LayerNorm; no metadata embeddings or dropout |
| Decoder | 48 → 128 → 512 → F; full-gene log-softmax; multinomial conditional on supplied RNA exposure |
| Objective | Symmetric A→B and B→A equal-scored-cell-direction conditional cross entropy; raw target counts remain sparse |
| Calibration | Fixed, source/guide-stratified audit cells within fitting donors; no query counts, endpoints or query-derived statistics |
| Comparators | Retained UMI-pooled composition; objective-matched mean training-half composition; separately trained rank-8 count factor; zero-latent diagnostic |
| Training schedule | Epoch-dependent shuffled source-interleaved shards with a rotating final source round; one contiguous visit per shard |
| Exposure selection | Prespecified full-epoch candidates (1, 2, 4, 8 by default), earliest epoch on exact score ties; candidate selection uses only fitting-donor audit cells |
| Fresh refit | Gate must pass, or an explicit diagnostic reason must be supplied; fresh initialization, all authorized rows per selected epoch |
| Geometry | Fitting-only streaming latent population mean/variance; frozen center and scale, no PCA or whitening |
| Query | Frozen encoder/decoder, eval mode, no optimizer or normalization updates |
| Latent export | Every captured cell retained; individual 48D states plus explicit positive-RNA geometry mask and complete source/guide support |
| Population access | Compact source/guide offsets and integer row index; chunked empirical laws, not repeated centroids or a full metadata scan per guide |

The initial observation family conditions the **expected counts** on library
size (`exposure * softmax(decoded_state)`). Exposure is not an extra conditioning
input to biological state or a fitted endpoint-depth predictor. A is normalized
using A alone; neither B counts nor full A+B library size enters that encoder
call. The split uses the existing identity-keyed PCG64DXSM binomial recipe, is
stable under row reordering/batching, and satisfies A+B=X exactly. This technical
diagnostic is not an independent biological replicate or a general guarantee
that complementary counts are independent.

The original UMI-pooled condition baseline remains reported. V2 additionally
fits an objective-matched mean of positive-depth training-half compositions;
that comparator enters the primary count gate. Its fixed numerical positivity
floor is not a biological pseudocount. Both equal-cell-direction CE and UMI-weighted
CE retain explicit contributors. Missing scored-half depth has no score. Zero-input
halves receive the model's input-independent response, with their frequency
reported, not a claim to cell-specific information.

## Information, fitting and evaluation boundaries

`CountRepresentationSpec` is a new, strict, versioned contract. It binds the
exact representation-fitting and query capabilities; donor/source roles;
complete canonical features; configuration/seed; executing source hash;
environment, allowlisted process settings, and PyTorch deterministic/TF32/cuDNN
controls. In the first R48 application this
means D3/D4 Rest and Stim48hr for fitting, D2 Rest for encoding after freezing,
and no D1, D2 Stim48hr, Stim8hr or context fitting.

Calibration selects the lowest deterministic cell-identity hashes within each
authorized source/guide stratum. Sources are explicitly mapped to donor and
condition, not parsed. It reserves `floor(0.05*n)` rows, with at least one audit
and one training row when n≥2. Singleton strata remain training-only. Thus 5% is
a requested fraction, not a promise of exactly 5% globally. Realized counts,
singleton coverage and audit coordinate identity are recorded. Selection takes
two metadata passes and retains only audit coordinates; no count payload opens
during this partition step. Row subsets already present in the fitting access
remain authoritative.

Only training rows enter calibration optimization, condition baselines and
normalization. Audit counts are read for fixed molecule-split scoring at candidate
epochs. They are cell-held-out relative to those evaluated calibration weights;
repeated training A/B splits are **not** called held-out evaluation. The same
audit cells select stopping exposure, so their selected score is development
evidence, not an independent confirmatory estimate. The selected calibration
autoencoder and factor weights are retained separately from the fresh refit.

The refit subsequently includes audit cells. Its manifest explicitly states
that the held-out score belongs to the calibration model, **not** to the all-cell
refit. Per-epoch and scaling-pass receipts record every consumed row count,
per-source counts, positive-RNA cells, RNA UMIs and ordered row-address digest.
Sampling/thinning do not use unrelated package/query/protected hashes, so changing
provenance outside authorized numerical inputs need not change fitted values.

The held-out count gate requires strictly lower selected error than the
equal-cell-direction condition baseline, separately selected factor, and zero-latent ablation,
by the prespecified margin. This is only a component count gate. It never sets
`representation_qualified` or scientific promotion true: real-cohort execution,
perturbation-preservation assessment and additional selection calibration remain
unqualified. One synthetic null does not substitute for independent null refits.

## Artifact and execution interface

The new schemas are
[specification V2](../schemas/fold-count-representation-spec.v2.json) and
[representation bundle V2](../schemas/fold-count-representation-bundle.v2.json).
Artifacts use no-pickle safe tensors, exact-content verification and no-clobber
publication. Output roots must remain outside the prepared input package.
The historical checkpoint schemas and baseline publications are unchanged.

```bash
python -m credo_count_sde_v4.count_representation calibrate \
  --input-root /absolute/prepared-package \
  --specification /absolute/run/representation-spec.json \
  --output /absolute/run/calibration

python -m credo_count_sde_v4.count_representation refit \
  --input-root /absolute/prepared-package \
  --calibration /absolute/run/calibration \
  --output /absolute/run/fitted-representation

python -m credo_count_sde_v4.count_representation encode \
  --input-root /absolute/prepared-package \
  --fitted /absolute/run/fitted-representation --role query \
  --output /absolute/run/query-latents
```

These are generic examples, not issued cluster jobs. The cohort adapter must
resolve the new specification from its already approved component views. It must
not merely relabel an endpoint capability as a fitting capability. Use the exact
installed wheel/source and frozen environment for every stage. The old baseline
rescoring wheel remains archived for its own historical artifacts.

`EmpiricalLatentStore(..., representation_sha256=...)` verifies the frozen
representation binding and returns `(states, weights, row_references)` in bounded
chunks per source/guide. Each available cell has weight 1/n within its guide law.
The row references address the immutable `cells.parquet` metadata; means and
variances cannot silently replace the individual support. Zero-RNA cells have
explicit unavailable geometry and zero coordinate placeholders; they remain in
captured-cell support and are not claimed to have identified latent states.
An empty population returns no samples, never a fabricated NTC population.

## Resource and qualification limitations

- This is a single-reader stage, not a worker-prefetch or distributed trainer.
  Sparse count batches and bounded full-gene decoder batches avoid cohort-sized
  dense matrices. The latent index uses approximately eight bytes per geometric
  cell plus offsets and bounded metadata, with a separate explicit byte cap.
- Decoder activation budgeting is a conservative allocation estimate, **not**
  measured process RSS, model/optimizer memory, or peak GPU VRAM. Full-shard hash
  and decompression costs recur across passes and require actual profiling.
- CPU and CUDA are explicit settings; unavailable CUDA fails rather than
  falling back. Only CPU synthetic execution is tested here. This does not
  qualify CUDA determinism, mixed precision, a cluster image/mount, or a 30-GB
  memory target. No new cohort GPU training was launched for this review.
- Current stages publish on completion; interrupted fitting is not yet a
  qualified optimizer/cursor restart. A completed calibration can be reused for
  a new fresh refit, but no interrupted update-exact continuation is claimed.
- The separate endpoint evaluator/outer-release workflow is not bypassed:
  this module accepts only exact bound fitting/query views for latent export.
  Protected endpoint latent compilation and state-model training remain later
  work. Software capabilities do not establish filesystem isolation.
- Geometry scaling excludes zero-RNA cells and uses population variance plus
  `max(1e-6, 0.01*median_positive_fitting_std)` as the default scale floor.
  Full-cohort degeneracy and perturbation-preservation diagnostics are still
  required before selecting this representation for scientific state modeling.

The historical bd2a639 [local validation receipt](../receipts/review-9ef72c6-count-representation-validation.json)
separates synthetic integration evidence from real-cohort baseline results.
That implementation's fresh complete suite passed **585 tests, four expected CUDA skips, and
85.5666% combined coverage** (unchanged 85% gate). The 22 focused component
tests reached 91.7735% combined component coverage; that coverage was not merged
with the full suite. The installed wheel passed the same 22 tests and all 14
external adapter/workflow tests, including a valid changed-protected-outcome
package with identical fitted tensor bytes and query states.

The [native-view preflight receipt](../receipts/review-9ef72c6-representation-input-resolution.json)
resolves 7,416,636 fitting rows / 607 shards and 1,983,457 query rows / 152 shards,
with all 18,129 RNA genes. It opened no raw count payload and launched no training.
Its CPU preflight binding is not authorization or qualification for a GPU job.
