# CREDO count-SDE v4

Current follow-up: [count-representation qualification changes](docs/review-bd2a639-count-qualification.md)
(cross-source scheduling, objective-matched gate, explicit failed-gate stop, and CUDA tests;
GPU execution and real-cohort calibration remain pending).

New component: [streamed fold-fitted count representation](docs/fold-count-representation.md)
(separate calibration/refit/frozen per-cell laws; no full-cohort representation trained yet).
Accepted baseline: [frozen R48 numerical results and paired target review](docs/r48-baseline-results.md).

Current review: [population-baseline scoring correction](docs/review-0cb50da-scoring-correction.md)
(real first-split execution and linked rescoring completed; frozen predictions unchanged).

New work: [source-only population baseline workflow](docs/source-only-population-baselines.md)
(separate fitting, immutable prediction and endpoint evaluation; neural successor training remains unrun).
Foundation: [full-cohort successor access](docs/full-cohort-successor-foundation.md).
Previous correction: [review of b45330b: CI and split integration](docs/review-b45330b-integration.md).
Previous foundation: [review of 789537f and promotion gates](docs/review-789537f-corrections.md).

Independent count-native finite-measure SDE recipe for longitudinal
perturbation screens. This repository implements three gated intents:

```text
count_state → count_measure → count_context
```

It is a sibling of, not a modification to, the frozen CREDO checkout. Version
`4.0.0.dev40` is engineering software; it is not a biological result and cannot
be relabeled as stable `4.0`.

Implemented surfaces include strict hash-bound contracts, sparse count storage,
source-only representation preparation, exact complete-denominator count
likelihood, separate state/fitness/context channels, immutable checkpoint and
resume, streaming inference, four-branch contrasts, one-shot evaluation, and a
sealed aggregate. The repository contains no cohort adapter or biological data.

Dev27 retains the two immutable, failed T01 Hellinger candidates, passed T04
fixed-pool particle-engine qualification, and completed T02A calibration. It
adds a no-retraining amendment that places both dev23 T07S null and nonzero
tests in duration-integrated endpoint units, separates nonzero checkpoint
selection from false promotion, and retains the tightened 0.05 false-promotion
limit. It adds the frozen T07R-A0 pooled relative-guide likelihood pilot. The
production likelihood matches an independent reference on training-only
synthetic counts, but its fixed subset concentration is now explicitly labeled
as a fold-subcomposition v1 method. Dev27 adds the physical-pool conditional-DM
v2 forensic correction. It again selects update 0 and the sister-guide target
reference is superior under both conditional multinomial and conditional-DM
uncertainty surfaces. The selector/protocol is retired; the numerical estimator
and constant-reaction family are not broadly retired. T01 blocks the
pooled real-data state-dynamics path; it does not block T02A raw-count/mass
noise qualification or the passed synthetic T04/T07S numerical path. T00 has an
explicit, immutable pooled finite-measure API and receipt. The T03 source ×
target hardening is engineering code, not a qualified real-cohort result.

Dev28 added a content-addressed sharded CSR store, persistent process-local
readers, and a bounded resumable shard writer whose typed checkpoint truncates
uncommitted payload after interruption. It also freezes the checkpoint-only
dimension-zero multinomial decoder required by G04 and evidence-only G14 claim,
robustness, multiplicity, and seal contracts. External GSE314342 G00 probes
verify exact reordered/duplicated reads and zero planned guide fragmentation,
but fail the frozen H100 loader-throughput gate. The full approximately 765 GB
CountStore was therefore not built, and no biological or outer-donor result is
promoted.

Dev29 replaces the impossible direct-raw-store H100 rate requirement with a
two-tier data plane: immutable source authority and virtual canonical access
feed fold-native compact views, which are qualified only by integrated wait,
utilization, parity, and bounded-memory gates. It also closes the shard
finalization crash window, binds appends to exact frozen chunk identities,
bounds reader handles, restores physical checkpoint chronology, and hardens the
G14 evidence graph. See the
[G00 two-tier data-plane contract](docs/g00-two-tier-data-plane.md).

Dev30-A preserves those accepted v1 results and adds v2 protected-access,
numeric-CSR, guide-target crosswalk, exact-parent, G00C selection/sampler, and
G00D parity/performance contracts. It runs no G00C/G00D workload and changes no
scientific gate. See the
[G00 Dev30-A amendment](docs/g00-dev30a-contract-amendment.md) and its
[superseded Dev30-A-r2 draft](docs/g00-dev30a-r2-contract-finalization.md).
Dev31 remains the frozen v2 provenance authority. Dev32 adds only the distinct
[legacy-parent attestation boundary](docs/g00-dev32-legacy-parent-attestation.md)
required after B0 A1 correctly rejected Dev29's historical checksum-only
publication.

The external Dev32 B0-A2 execution subsequently passed and promoted G00A-v2
and G00B-v2 as source-plane parents. Dev33-A adds only the separately versioned
[G00C hardening boundary](docs/g00-dev33-g00c-hardening.md): fail-closed
two-million-cell extension semantics, exact physical row ordering, streaming
compact verification, and executable refit replay. No extraction canary or
G00C workload is part of Dev33-A.

The separately frozen
[Dev33-B extraction canary](docs/g00-dev33b-extraction-canary.md) subsequently
passed on LODO fold 0 with 58,192 rows and 256 primary features. It is
non-promotable engineering evidence only; claim-bearing G00C, G00D, and G04
remain blocked.

Dev34-A closes Dev33-B and freezes the first claim-bearing G00C contract without
reading expression. The [Dev34-A claim-contract record](docs/g00-dev34a-claim-contract.md)
binds all 59 expanded seed tuples, feature-then-cell parentage, common 4,096-
feature scoring support, fixed `1e-4` margins, and terminal
`fail_no_saturation` semantics. Claim-bearing G00C remains unrun; no G00D, GPU,
model, or biological operation is authorized by this release.

Dev35 preserves that milestone and adds the artifact-derived V3 execution and
decision chain, a pre-access 2M extension freeze, stage-scoped support gates,
the common 0.5-per-gene prior, portable review provenance, and an exact tested
CPU environment authority. See the
[Dev35 execution-authority record](docs/g00-dev35-execution-authority.md) and
[Dev36 execution-seal hardening](docs/g00-dev36-execution-seal.md). Dev37 closes
the remaining source-substitution, supplied-statistics, publication-cycle,
extension-parent, monitoring/access, and durable-restart defects. See the
[Dev37 source-derived seal](docs/g00-dev37-source-derived-seal.md).
Its metadata-only A3 authority and deterministic archive are finalized; G00C
expression access and G00D remain blocked.
Dev38 adds a behavior-preserving sampler optimization and three explicitly
separated CUDA reduction envelopes. The H100 canaries are engineering evidence
only; the fast CUDA-native stream requires a new authority. See the
[Dev38 GPU qualification](docs/g00-dev38-gpu-qualification.md).
Dev39 begins an evidence-aware result layer without changing the
finite-measure core: observed study fields derive hard capability limits,
abundance and lineage semantics are explicit, and biological wording fails
closed. See the [Dev39 Phase 1 evidence framework](docs/dev39-evidence-framework.md).
Dev40 adds a static count-linked biological-program component, legacy
split/baseline/null diagnostics, and an immutable result bundle. It does
not couple the new head to SDE dynamics or alter checkpoint schemas. See the
[Dev40 program qualification](docs/dev40-biological-program-qualification.md).
The corrected evaluator does not certify biological programs: the old
coefficient-exceedance diagnostic is not discovery calibration.

## Quick start

```bash
python -m pip install -e '.[test]'
credo-v4 synthetic --output /tmp/credo-v4-demo --intent count_context
credo-v4 prepare /tmp/credo-v4-demo/config.yaml
credo-v4 compile /tmp/credo-v4-demo/config.yaml
credo-v4 train /tmp/credo-v4-demo/config.yaml
credo-v4 finalize /tmp/credo-v4-demo/config.yaml
credo-v4 evaluate /tmp/credo-v4-demo/config.yaml
credo-v4 seal /tmp/credo-v4-demo/config.yaml
credo-v4 verify /tmp/credo-v4-demo/work/sealed --level full

# Independent component test; no cohort or GPU is used.
credo-v4 qualify-particle-engine --output /tmp/T04_PARTICLE_ENGINE

# Learned constant-reaction recovery; synthetic CPU catalogs only.
credo-v4 qualify-reaction --output /tmp/T07S_REACTION_RECOVERY

# Correct a manifest-verified dev23 T07S bundle without rerunning its optimizer.
credo-v4 amend-reaction \
  --t07s-bundle /path/to/T07S_REACTION_RECOVERY \
  --output /tmp/T07S_REACTION_METRIC_AMENDMENT

# One-fold, CPU-only pooled relative-guide likelihood qualification.
credo-v4 qualify-pooled-reaction \
  --pooled-bundle /path/to/T00_pooled_data_contract \
  --t02a-amendment /path/to/T02A_INTERPRETATION_AMENDMENT \
  --t07s-amendment /path/to/T07S_NULL_INTERVAL_AMENDMENT \
  --fold-assignment /path/to/guide_fold_assignment.parquet \
  --output /tmp/T07R_A0_POOLED_LIKELIHOOD

# One permitted, historically exposed physical-pool denominator correction.
credo-v4 correct-pooled-reaction \
  --pooled-bundle /path/to/T00_pooled_data_contract \
  --t02a-bundle /path/to/T02A_raw_count_mass_noise_v2 \
  --t02a-amendment /path/to/T02A_INTERPRETATION_AMENDMENT \
  --t07s-amendment /path/to/T07S_NULL_INTERVAL_AMENDMENT \
  --fold-assignment /path/to/guide_fold_assignment.parquet \
  --output /tmp/T07R_A0_PHYSICAL_POOL_CONDITIONAL_DM_V2_R2

# Independent T02A calibration against a passed T00 bundle and raw CountStore.
credo-v4 qualify-raw-noise \
  --pooled-bundle /path/to/T00_pooled_data_contract \
  --count-store /path/to/counts.h5 \
  --output /tmp/T02A_RAW_COUNT_MASS_NOISE
```

See [architecture](docs/architecture.md), [contracts](docs/contracts.md),
[lifecycle](docs/lifecycle.md),
[the G00 two-tier data-plane contract](docs/g00-two-tier-data-plane.md),
[the G00 Dev30-A contract amendment](docs/g00-dev30a-contract-amendment.md),
[the superseded G00 Dev30-A-r2 draft](docs/g00-dev30a-r2-contract-finalization.md),
[the active G00 Dev31 provenance finalization](docs/g00-dev31-provenance-finalization.md),
[the Dev32 legacy-parent attestation boundary](docs/g00-dev32-legacy-parent-attestation.md),
[the accepted Dev32 B0-A2 execution](docs/g00-dev32-b0-a2-source-plane-execution.md),
[the Dev33-A G00C hardening boundary](docs/g00-dev33-g00c-hardening.md),
[the Dev33-B G00C extraction canary](docs/g00-dev33b-extraction-canary.md),
[the Dev34-A claim contract](docs/g00-dev34a-claim-contract.md),
[the Dev35 execution authority](docs/g00-dev35-execution-authority.md),
[the Dev36 execution seal](docs/g00-dev36-execution-seal.md),
[the Dev37 source-derived seal](docs/g00-dev37-source-derived-seal.md),
[component qualification](docs/component-qualification.md),
[the detailed T01 record](docs/t01-representation-qualification.md),
[the detailed T02A record](docs/t02a-raw-count-mass-noise.md),
[the detailed T04 record](docs/t04-particle-engine-qualification.md),
[the dev23 T07S record](docs/t07s-reaction-recovery.md),
[the dev24 R1 amendment](docs/t07s-reaction-metric-amendment.md),
[the authoritative dev25 unified R0/R1 amendment](docs/t07s-null-interval-amendment.md),
[the dev26/v1 and dev27/v2 T07R-A0 record](docs/t07r-a0-pooled-likelihood.md), and
[release policy](docs/release.md).

Current verification status is recorded in
[implementation-status.md](docs/implementation-status.md) and
`receipts/local-validation.json`.
