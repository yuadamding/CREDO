# Implementation status

Last verified: 2026-08-21. Authority: package engineering status; biological
run receipts and cohort audits remain external to this repository.

Implemented in this repository:

- independent package/repository, licenses, locks, frozen dependency receipt;
- strict contracts and generated schemas;
- canonical hashes, sparse CSR store, correction gate, representation cache;
- compiled run identity, state/measure/context model, exact full-block count loss;
- update-based training, safe optimizer serialization, checkpoint/resume/fork;
- physical-grid streaming inference, latent-state export, four contrasts;
- one-shot evaluation, baseline audit, lifecycle ledger, sealed aggregate;
- restartable particle-state integration and stable absolute log-mass pools;
- CLI/API, full content/reload verification, correction/context audit helpers;
- CPU regression, leakage, corruption, fault, resume, compatibility, and wheel CI.

Dev29 retains the component-qualified boundary introduced in dev18: the T00
pooled finite-measure contract, public API/CLI, strict channel-isolation
contract, and focused invariants. The external pooled Renz adapter passed T00
with 495 guides and 277,200 cells. T04 independently passed its fixed-truth
numerical qualification. T02A now freezes the independent raw-count and
relative-mass sampling noise surface from 100 repeats without a learned model.
T07S-A separately passes learned constant target-average reaction recovery on
synthetic complete-denominator catalogs after the duration-correct dev25
unified R0/R1 amendment; no pooled Renz dynamics component is promoted. See
[component qualification](component-qualification.md).

T07R-A0-v1 was exercised on the predeclared pooled Renz fold 0. Its parity gate
passed and its selector failed, but its likelihood reset concentration after
subsetting folds. Dev27 preserves that record as
`fold_subcomposition_fixed_concentration_dm_v1` and adds the one permitted
CPU-only `physical_pool_conditional_dm_likelihood_v2` forensic correction.
V2 uses all 495 P4 guides for the physical denominator and inherits each active
fold's concentration from the full alpha vector. It again selects update 0;
M1 is superior with conditional-multinomial interval
`[+0.00007625,+0.00008436]` and conditional-DM sensitivity
`[+0.0001157,+0.0002324]`. No additional Renz update, GPU run, or fold scale-out
is eligible. See the
[T07R-A0 record](t07r-a0-pooled-likelihood.md).

T01 v1 and v2 are implemented fail-closed and have been exercised on the
pooled Renz population. Both correctly retained the global decoder in all four
folds because every low-rank candidate worsened source-only count likelihood.
The raw Hellinger candidates were 0.204–0.256 nats/count worse; centered
residual Hellinger candidates were 0.178–0.238 worse. These are retired
candidates, not a representation qualification; downstream tests remain
blocked only on the pooled real-data representation path. The independent
T02A raw-count/mass calibration passes with a target-balanced observed-endpoint
sampling RMSE q95 of 0.132578 and guide absolute-error q95 target-median q95 of
0.289973. These are conditional noise summaries, not a direct model-improvement
margin or formal minimum detectable effect. Synthetic T04–T07 remains independent. See the
[detailed T02A record](t02a-raw-count-mass-noise.md).

T04 routes non-trainable analytic truths through the production streaming
Euler–Maruyama engine. It qualifies deterministic drift, OU moments over a
16-point particle/step grid with 50 seeds, exact reaction mass, stabilized
absolute-weight pool feedback, normalized-context negative control,
deterministic replay, and interrupted/resumed parity. The largest-grid OU
variance relative error is 0.004459. This is numerical software evidence; it
does not qualify a learned channel or biological model. See the
[detailed T04 record](t04-particle-engine-qualification.md).

T07S-A isolates target constant reaction as the sole trainable channel. It uses
119 independent zero-truth refits split into calibration and audit sets,
retains update 0, selects on separate validation catalogs, performs a fresh
post-selection refit, and evaluates independent test catalogs once. Dev25
preserves that selected model byte-for-byte and corrects both R0 and R1 to
duration-integrated centered interval change. The R1 target-balanced RMSE
is 0.033191 versus 0.706508 for zero reaction; delta -0.673317 with target-
bootstrap interval [-0.829965, -0.482187]. Reaction RMSE is 0.017233 and all
12 nonzero signs are recovered. Fixed channels, the complete-denominator gauge, probability sums,
control mask, and streaming relative masses pass their tolerances. This is
software/synthetic qualification, not evidence about a cohort. See the
[dev23 T07S record](t07s-reaction-recovery.md) and
[dev24 R1 amendment](t07s-reaction-metric-amendment.md) and authoritative
[dev25 unified R0/R1 amendment](t07s-null-interval-amendment.md).

Local acceptance completed with 291 CPU tests passing and one CUDA-only test
skipped on the CPU validation host, with 85.00% branch-aware coverage. The exact
test and coverage results are in
`receipts/local-validation.json`. Ruff,
mypy, generated-schema, documentation-link, wheel
namespace/license, clean-wheel lifecycle, and frozen-CREDO preflight checks
passed. The wheel is reproducible under a fixed `SOURCE_DATE_EPOCH`; normalized
sdist publication is reproducible through `scripts/normalize_sdist.py`.

Dev28 added the G00/G04/G14 infrastructure required by the GSE314342 component
program: exact sharded CSR manifests and Merkle identities, persistent readers,
a bounded interruption-reconciling writer, the checkpoint-conditioned
dimension-zero multinomial null, and evidence-only claim, robustness,
multiplicity, and seal contracts. External G00 probes passed source, feature,
capacity, and sparse-read correctness checks but failed the frozen H100
throughput requirement by orders of magnitude. The full CountStore and every
real-data downstream component remain blocked.

Dev29 reclassifies that result correctly. G00A is the source/feature authority;
G00B is a metadata-only virtual raw-count plane for sequential statistics and
one-time extraction; G00C is a fold-native compact count view; G00D is an
integrated loader/compute qualification. The direct 2.75-million-raw-row/s
threshold is retired as a production gate because it implied approximately
91 GB/s of decoded CSR traffic. The dev28 failure remains authoritative for
direct raw streaming. Dev29 implements strict G00A–D contracts and the G00B
reader. The external dev29 execution froze all 12 full source hashes and now
passes G00A and G00B with authority `0bda9cf4…` and virtual store `cb452528…`;
it reconciles
21,996,842 eligible rows and 90,997,745,441 eligible nonzeros without copying
the raw count matrix. It does not claim G00C or G00D execution. No
outer-donor stimulated outcome is opened by package tests. See the
[two-tier data-plane record](g00-two-tier-data-plane.md).

Dev29 emits checkpoint-decoder and G14 schema v2 contracts. The public
validator retains explicit read-only v1 models so the immutable dev28 evidence
continues to validate; new construction and cross-contract sealing require v2.

Dev30-A extends the same compatibility policy to G00. Accepted Dev29 G00A/G00B
v1 bytes remain valid. New v2 construction distinguishes authority reads from
forbidden protected-expression use, binds numerical CSR and guide-target
crosswalk artifacts, enforces complete G00A-to-G00B parent recomputation,
freezes the training-only G00C feature/sample/sampler contract, and makes G00D
environment, parity, performance, and numerical memory gates fail-closed. This
is contract-only work: G00C and G00D remain not run and downstream real-data
components remain blocked. See the
[Dev30-A contract amendment](g00-dev30a-contract-amendment.md).

Dev30-A-r2 finishes the contract path without executing it. It adds the
immutable v1-to-v2 source-plane amendment, construction derivation receipt,
full locator/crosswalk reconciliation, derived G00C feature/scale/row/sampler
decisions, and exact G00D GPU/protocol/evidence wiring. G00A/G00B v1 remain the
only executed source-plane evidence; G00C, G00D, and downstream real-data
components remain blocked. See the
[Dev30-A-r2 finalization](g00-dev30a-r2-contract-finalization.md).

External review conditionally rejected that r2 draft as an execution authority.
Dev31 supersedes it by enforcing exact v1-to-v2 source-record projection,
parent/derived file `ArtifactRef` wiring, parsed amendment verification, and a
passed amendment-receipt parent chain for G00C. G00C verification now derives
fit/validation roles, selected feature and cell prefixes, literal compact CSR
counts, and uninterrupted/resumed sampler traces from their artifacts. G00D
verification now derives all performance, parity, and memory summaries from
the bound files and applies gate-specific exact versus numerical semantics.
This remains contract-only work: Dev31 has not constructed the amendment or run
G00C/G00D. See the
[Dev31 provenance finalization](g00-dev31-provenance-finalization.md).

Dev31-B0 A1 then failed correctly before source access: accepted Dev29 bytes
passed their historical `SHA256SUMS`, but the original publication has no
native `artifacts.json` or `COMMITTED` marker. Dev32 does not make those native
markers optional. It introduces a discriminated
`legacy_checksum_attested_v1` parent type whose separately published sibling
wrapper proves current checksum coverage and exact accepted G00A/G00B
semantics while explicitly denying historical atomic or manifest-last
publication. The wrapper builder never opens the twelve raw expression
matrices. See the
[Dev32 legacy-parent attestation boundary](g00-dev32-legacy-parent-attestation.md).

External Dev32 B0-A2 then passed the complete 12-source byte/CSR scan and
published accepted G00A-v2 `b12978ea…` and G00B-v2 `ee4dc94c…`. The exact
eligible population is 21,996,842 rows and includes 24,972 targeting plus 984
control guides. The raw source category named `targeting single sgRNA` is an
assignment-class label for both groups; the crosswalk supplies targeting versus
control identity. See the
[B0-A2 execution record](g00-dev32-b0-a2-source-plane-execution.md).

Dev33-A hardens the not-yet-run G00C boundary without touching those parents.
It requires a base-grid `extension_required` stop when no sub-million candidate
saturates, separate row-set and ordered-row hashes, exact physical block order,
bounded streaming CSR comparison, and executable replay of full selected and
reference refits plus a preregistered audit subset. New work uses fold-contract
schema v3 and execution/decision schema v2. No canary, feature ranking, sample
selection, compact materialization, G00D, GPU, model, or biological result was
run in this release. See the
[Dev33-A hardening record](g00-dev33-g00c-hardening.md).

Dev33-B then executed the authorized fixed 50,000-training-row/256-feature
CPU extraction canary on LODO fold 0 (D1 held out). The A3 attempt passed
byte-identical writer restart, bounded source equality, exact 32-microbatch
sampler replay, a 16-GiB RSS gate, and zero protected D1 stimulated reads.
Its status is `pass_engineering_canary`, not a claim-bearing G00C pass; it may
not parent G00D or G04. See the
[Dev33-B extraction-canary record](g00-dev33b-extraction-canary.md).

Dev34-A closes the canary and freezes the next claim-bearing G00C contract. It
expands all 59 paired-refit seed streams before result access, makes the feature
selection result a cryptographic parent of cell-budget selection, evaluates
every feature prefix on the same 4,096-feature weighted-count denominator, and
adds terminal `fail_no_saturation` when the 2M reference alone qualifies. The
fixed feature and cell margins are both `1e-4` nats per weighted validation
count. No cohort expression, G00C execution, G00D, GPU, model, or biological
result is part of Dev34-A. See the
[Dev34-A claim-contract record](g00-dev34a-claim-contract.md).

Dev35 preserves Dev34-A and makes the not-yet-run selection path decidable from
artifacts. V3 execution and decision verifiers recompute all 59 paired refits,
reference-zero checks, support eligibility, feature and row prefixes, and the
terminal publication state. A separate pre-access extension contract is the
only way to expose the 2M candidate. The smoothing prior is now exactly 0.5 per
gene on the shared 4,096-gene support. Portable review hashes and the exact
tested CPU environment replace attachment-path and stale-lock provenance. See
the [Dev35 execution-authority record](g00-dev35-execution-authority.md).

Dev36 preserves that milestone while replacing every opaque or self-attested
execution gate. It adds a distinct source-backed materialization verifier,
typed hierarchical sampler replay, sampler-derived support, executable
closed-form refit replay, exact status-dependent publication inventories, and
a metadata-only D1 A2 authority plus independently audited immutable archive.
The full-grid process-tree ceiling is explicitly 64 GiB. Dev36 still does not
open expression, run the G00C curve, authorize G00D, or provide model or
biological evidence. See the
[Dev36 execution-seal record](g00-dev36-execution-seal.md).

Dev7 replaced the invalid reconstructed-SVD softmax with a checkpoint-owned
multinomial count-composition decoder. Dev10 added a deterministic
training-only validation holdout and minimum-validation-cross-entropy
checkpoint selection. Dev12 caches the immutable sparse decoder row universe
once per training process, eliminating repeated whole-window HDF5 scans while
preserving row order, targets, validation rows, and checkpoint semantics.
Dev13 adds an analytically fitted adaptive target anchor around a global
training-terminal reference. It uses support-weighted leave-one-guide-out
sister-guide evidence, a bounded nonnegative gate, and exact-zero control or
discordant-target corrections. That state estimator is frozen before decoder
optimization; CUDA updates do not train it.
Decoder reconstruction validation does not establish perturbation prediction
or biological validity.

Dev14 adds a null-nested source-conditioned terminal-anchor residual and an
explicit joint state/decoder training switch. It replaces atomic CUDA decoder
reductions with deterministic CSR segment reductions and applies the frozen
primary-population definition consistently to both point metrics and target
bootstraps. Local acceptance establishes contracts and CPU behavior; the
completed dev14 H100 run below establishes the external development result.

Dev17 superseded dev16 before cohort execution. It estimates a bounded scalar
target-main shrinkage model from leave-one-guide-out sister-target predictions
without an interaction, freezes that model, and
tests the source × target term only through the full-versus-interaction-off
increment. The selector uses separate target and interaction margins from a
byte-verified row-level calibration with at least 59 genuine independent null
fits. It admits exactly update 0 and the calibrated checkpoint schedule, and
records all three scores and displacement magnitudes. A typed
selection manifest is embedded in inference and reported by evaluation; only a
deployed interaction family that beats both M1 and M0 can pass the interaction
advancement gate. The
gene decoder is structurally absent. Post-selection refit publishes through
`REFIT_PLANNED → REFIT_RUNNING → REFIT_COMMITTED`, resumes after interruption,
uses no false state-parent edge, and records empirical and complete regularized
objectives separately. Local
acceptance covers global-only, shrunk-target, full-target, and interaction truth
regimes plus a 59-fit evidence-bearing null-calibration lifecycle. The pilot
also forbids diffusion, selection, alternative drift, analytic fitting,
decoder training, support weighting, and checkpoint forks. No real-cohort
dev15, dev16, or dev17 performance result exists. Dev18 further target-centers
the interaction, enforces target-wise zero mean, separates target-main,
conditional-interaction, and joint nulls, freezes fold and seed identities,
and strengthens noninteraction comparators. That T03 code remains unqualified
until T01, T02, and the prescribed pooled pilot pass.

Dev37 supersedes only the Dev36 verifier surface. It now constructs and fully
verifies the accepted G00B store internally, recomputes the complete
checkpoint-conditioned feature ranking, derives checkpoint-indexed refit
statistics from source reads, freezes the sampler plan pre-access, verifies
typed process/access/restart evidence, integrates the sealed V5 extension, and
uses a decision-free inner publication plus outer final seal. The fresh D1
authority is finalized as A3 (`5ee82382…`) and independently archived as
`2f3085c9…`. It remains metadata-only; G00C expression execution is still blocked.
See the [Dev37 source-derived-seal record](g00-dev37-source-derived-seal.md).

The Dev38 engineering branch caches the five sampler prefixes, replaces
per-draw Pandas/dictionary construction with typed arrays, and adds three
explicit CUDA refit envelopes. One-H100 Kubernetes canaries proved exact
Dev37-reference/CUDA equality for the expanded-row path and a 273.20x warm
batched speedup for the CUDA-native resident path. Both compact paths consume
another RNG stream and are not Dev37 promotion evidence. See the
[Dev38 GPU qualification record](g00-dev38-gpu-qualification.md).

The following Dev39 Phase 1 branch adds no model channel and makes no biological
claim. It introduces content-addressed study-capability, abundance-gauge,
trajectory/lineage, evidence-tier, claim-adjudication, and perturbation-dossier
contracts. Positive, null, adversarial, and ablation tests enforce that missing
physical pools, absolute counts, barcodes, or live tracking disable the related
language without weakening other capabilities. See the
[Dev39 evidence-framework record](dev39-evidence-framework.md).

Dev39 and Dev40 add an orthogonal scientific-result boundary. Dev39 factorizes
evidence tier, channel, independence, exposure, replication, multiplicity,
semantic role, abundance axes, and section availability. Dev40 implements the
static count-linked program observation model and its full qualification
surface: four non-interchangeable outer splits, six frozen baselines, separate
seed and donor stability, sister-guide agreement, three null families, gene
sign calibration, and content-verified publication. Identifier-only unseen
targets are not eligible, two-checkpoint studies cannot claim held-out-time
qualification, and technical batches cannot substitute for donors. See the
[Dev39 framework](dev39-evidence-framework.md) and
[Dev40 qualification](dev40-biological-program-qualification.md).

Deliberately not asserted complete in `4.0.0.dev40`:

- stable CREDO entry-point discovery;
- in-repository real-cohort adapters, biological thresholds, or biological claims;
- claim-eligible context status without all external audit receipts;
- CUDA/BF16 hardware qualification or production numerical tolerance;
- stable legacy-family replay, SBOM attestation, and signed release archive.

Those are release gates, not silent assumptions. The current implementation is
a complete non-device engineering lifecycle and remains `engineering_only`.

Earlier external GSE235325 CUDA reruns failed their scientific baselines and
are retained as engineering diagnostics. The eight-worker dev12 H100
development run completed 2,000 decoder updates in all eight scope-fold jobs,
then finalized, CPU-evaluated, fully verified, and sealed every bundle. Sampled
peak execution memory was 19.29-20.44 decimal GB. Its state model was fixed to
the global training-terminal centroid; long optimization trained the decoder,
not an SDE trajectory. Final target-balanced state RMSE differed from that null
by -5.92e-09 (`limited2500`) and +4.99e-09 (`common34699`), so both scientific
advancement gates were false. The external final report and receipts—not this
package-status page—are the run authority.

The dev13/a10 successor completed all eight 2,000-update H100 workers and was
subsequently evaluated and sealed by a fresh Kubernetes recovery. It remained
non-promotable: `limited2500` improved by only 0.000922 RMSE and
`common34699` worsened by 0.001425 relative to the global-terminal comparator;
both conditional target-bootstrap intervals crossed zero. Those CUDA updates
trained only the decoder because the adaptive state estimator was analytic and
frozen. Exact run receipts remain the authority; these numbers are a package
status synopsis, not a biological result.

The dev14/a11 successor trained a zero-initialized source-conditioned residual
jointly with the decoder, so this state channel was no longer analytic or
detached. All eight projects completed 2,000 H100 updates and selected update
250 by training-only target-stratified state validation. The residual was
nonzero in every selected checkpoint, but outer performance was worse than the
global-terminal null: `limited2500` delta +0.012688, 95% conditional
target-bootstrap interval [+0.005258, +0.019539]; `common34699` delta
+0.009976, interval [+0.006061, +0.014019]. Both advancement gates failed.
This falsifies the tested source-state-only residual under the current
historically exposed development contract; additional epochs, particles, or
memory are not a justified retry. Exact external run receipts remain the
authority.

Exact future use depends on the branch commit, `REPOSITORY.sha256`, the
wheel/sdist bytes, generated schemas, and the local validation receipt. A clean
dev30 commit and release receipt are necessary—but not sufficient—before any
pilot deployment or stable promotion.
