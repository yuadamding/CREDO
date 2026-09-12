# Dev40 static biological-program qualification

Original design: 2026-08-21. Interpretation corrected: 2026-09-12.

Status: **bounded component diagnostic, not qualified biological programs**.
See [the foundation correction](review-789537f-corrections.md) and
[split-integration follow-up](review-b45330b-integration.md). Historical receipts
remain unchanged; newly computed qualification details use revision 3. The keyed
effect formula retains revision-2 semantics. The legacy universal
gate remains named and separate, but its null-calibration gate now fails closed.

## Decision

Dev40 implements Phase 2 in two parts:

- Dev40-A freezes the raw-count observation model and typed program result
  contracts.
- Dev40-B evaluates the static/checkpoint-conditional head against diagnostic
  splits, baselines, stability tests, and negative controls. That implementation
  did not establish valid discovery calibration.

It does not couple programs to the V4 SDE, add a transition loss, or modify the
checkpoint schema. Those operations are reserved for Dev41 after a static
program result passes its own qualification.

## Observation model

For cell \(i\), gene \(g\), program \(k\), target \(t\), guide \(j\), sample
\(s\), state covariate \(z_i\), and physical time \(\tau_i\):

\[
y_{ig}\sim\operatorname{NB2}(\mu_{ig},\theta_g),
\]

\[
\log\mu_{ig}=\log L_i+b_{sg}+z_i^\top B_g+
\sum_k W_{gk}\left[a^{ref}_{ik}+a^{target}_{itk}
+q_j a^{guide}_{ijk}\right].
\]

The library term \(L_i\) is the exact observed total within the frozen modeled
gene universe. Dispersion is gene-specific. `W` has unit-norm columns and an
L1 sparsity penalty. Target and guide deviations use separate L2 penalties.
Control guides have an exact-zero target/guide contribution. Checkpoint
dependence uses a linear basis in continuous physical time so an unseen time
is representable without a new checkpoint embedding.

The target/guide hierarchy is frozen before fitting:

\[
a_{ij}=a_{ref}+a_{target(t_j)}+q_j a_{guide(j)}.
\]

`q_j` is an **unidentified latent guide scale**, not measured guide efficiency.
Its product with a free deviation is rescalable without changing predictions;
regularization does not identify biological efficacy. The historical checkpoint
key and artifact name `guide_efficiency` remain for compatibility, with this
restriction recorded in new metric artifacts. Within-target hard centering is
not implemented. Do not derive an efficacy anchor from protected endpoints.

With only target identifiers, unseen-target prediction is structurally
ineligible. It becomes eligible only when predeclared target descriptors exist
independently of protected expression and are bound as an artifact.

## Result contracts

The immutable result surface contains:

| Contract | Meaning |
| --- | --- |
| `ProgramDefinition` | Sparse signed gene loading, without an automatic pathway name |
| `PerturbationProgramEffect` | Reference, target, guide-deviation, and legacy-named latent guide scale |
| `GeneLevelEffect` | Program-reconstructed signed gene effect and uncertainty |
| `GuideTargetConsistency` | Sister-guide agreement and target/guide variance summary |
| `ProgramUncertainty` | Separate seed and biological-donor loading stability plus null inclusion |
| `ProgramQualificationReceipt` | Conjunctive promotion decision and every split result |
| `BiologicalProgramQualificationBundle` | Cross-wired-ID-resistant, scope-exact enclosing object |

Every top-level result is schema-versioned and content-addressed. Model state is
stored as non-pickle NPZ. Publication is atomic and no-clobber; verification
recomputes every artifact size and SHA-256.

## Four legacy split types, not four prepared outer donor folds

The split kinds are not interchangeable:

1. held-out biological donor;
2. held-out guide with its target still represented by a sister guide;
3. held-out target; and
4. held-out time.

All four must be eligible and pass for scientific promotion. A WTA library or
sequencing batch cannot substitute for donor identity. Held-out time needs at
least two fit checkpoints, one evaluation checkpoint, and a
`ProtectedExpressionAccessContract`. Held-out target needs predeclared target
descriptors; an identifier embedding cannot extrapolate to an unseen ID.

This component chooses the maximum donor index and a random-cell inner split.
It does **not** consume the prepared four-outer/12-inner donor plan and must not
be reported as that experiment. Stopping inspects only its inner subset.
Known-target donor forecasting needs a separately qualified nested-donor
profile, not mandatory unseen-target success under this universal diagnostic.

Evaluation targets and observed evaluation references are now separate row sets.
For held-out guide/target diagnostics, reserve `ceil(0.25 * n)` controls within
each donor/condition using the configured seed and ascending physical row order,
leaving at least one control for fitting. A stratum with fewer than two controls
provides no independent reserved reference; its sign metric stays undefined.
Reservation inspects metadata only and precedes the inner fitting/validation
split. Donor/time references are controls already inside the held-out stratum.
All three row sets must be disjoint. The reference outcomes enter only the
evaluator, not that split's head fitting, baseline fitting, or control prediction.
Later all-data reference/stability fits are separate diagnostics, not held-out
predictive evidence.

This is held-out **cell-observation** evidence; shared donor/culture effects do
not become independent biological replicates. Count likelihoods now score
perturbation evaluation targets only, not reserved control cells. Revision-3
scores therefore must not be pooled with historical scores under old supports.
The detailed artifact binds physical row partitions by digest. Frozen V1 split
summary IDs remain unit-label summaries, not row-level access authorities.

Strictly increasing physical checkpoint times remain required. GSE314342's
Rest/Stim8hr/Stim48hr `[8,8,48]` collection times cannot be supplied as a single
ordered trajectory. A condition/branch-aware successor remains necessary; do
not fabricate Rest time zero or merely relax the duplicate-time check.

## Six frozen baselines

Every eligible split reports held-out count likelihood for all comparators in
this exact order:

1. exact global gene frequency;
2. pseudobulk negative binomial;
3. per-gene differential expression;
4. GSFA-style sparse factors;
5. target average; and
6. control only.

Baseline fitting receives training counts, labels, and library totals only.
Evaluation counts never affect baseline parameters. The common-dispersion
mean-prediction score uses training-only method-of-moments dispersion for both
model and baselines. Improvement uses this shared score, not the likelihood
of the fitted NB distribution. Revision 2 also reports
`predictive_nb_log_likelihood` using the head's fitted dispersion. Inner
checkpoint selection uses that head's fitted-NB negative log likelihood.

Gene sign is evaluated per donor/condition/guide/gene, then macro-aggregated
over conditions, guides, targets and donors. Observed matched controls define
truth only; model control predictions define the predicted contrast. The
fixed informative threshold is absolute log-composition effect at least 0.05,
with log pseudocount `1e-8`. Missing matched controls or no informative effects
produce undefined metrics and explicit coverage losses. All reported units
must be supported to pass the diagnostic split gate. This is a panel-composition
metric, not absolute expression or causal knockdown validation.

## Bounded diagnostic execution

The head remains a dense-host small-panel component (default 2,048 genes and
512 MiB training/inner-validation input-tensor payload), not a full-cohort CSR
trainer. Inner validation, outer prediction, and independently predicted
references use a shared bounded batch routine. Outer likelihoods and keyed
effects accumulate per chunk; baseline tables are fitted once at unit depth for
unique target/checkpoint metadata pairs, then expanded only for the current
chunk. No complete cell-by-gene outer prediction panel is retained.

The default 64 MiB evaluation-output cap covers retained numeric guide/control
summaries, baseline tables, and one prediction/reference chunk; it is checked
before fitting. It does not measure process RSS, GPU allocator peaks, model
activations, temporary likelihood/SVD work, metadata objects, or the caller's
input matrix. Exceeding the cap fails, without subsampling or changing scores.

## Stability and null calibration

Seed stability uses at least three unique fits and Hungarian alignment of
absolute loading correlations. Donor stability is separate: each biological
donor is excluded in turn and its loading basis is aligned to the full-data
reference. When donor identity is unavailable, donor stability is unavailable
and promotion fails.

Sister-guide consistency intersects explicit donor/condition/gene keys for
each pair. Equal vector length is insufficient. Pair support and missing
comparisons are persisted; incomplete support fails the gate. Between-target
variance is not estimated by this keyed evaluator: the legacy scalar slot is
zero with an explicit unestimated status, not a variance-explained result.

At least 20 negative-control fits cycle through three families:

- control-label/count permutation within checkpoint;
- target/guide permutation within checkpoint; and
- NB no-program simulation preserving checkpoint-specific library structure.

These legacy runs measure raw coefficient exceedance, not false discoveries.
They retain shortened schedules and imperfectly stratified permutations for
historical diagnostic continuity only. Revision 2 always records
`null_inclusion_calibrated=false`; no threshold setting can promote this route.
An effect-scale inclusion rule, the actual factual selection/stopping procedure,
and defensible donor/condition exchangeability remain to be implemented under
a new qualification protocol. More null replicates alone do not solve this.

## Promotion rule

`pass_scientific` is conjunctive. It requires:

- every outer split to be eligible and pass;
- held-out target performance to pass explicitly;
- seed loading stability to pass;
- donor loading stability to be available and pass;
- sister-guide consistency to pass;
- null inclusion to calibrate; and
- gene-sign calibration to pass on every eligible split.

External pathway concordance is reported only when an independent pathway
resource is available. Its absence is explicit; it is never fabricated from
the same fitted genes. A failed qualification publishes no qualified program
IDs and cannot populate biological program or gene-effect dossier sections.

## Cohort adapter boundary

The package contains no cohort adapter or data path. Workspace-owned adapters
must supply raw nonnegative integer panel counts, exact panel-library totals,
physical times, frozen guide-to-target hierarchy, controls, stable row IDs,
and source-bound feature order.

Expression-derived PCA, HVG, DE, pathway, or endpoint QC artifacts may not be
silently reused in a strict time test. A metadata-only fixed audit panel and a
zero-dimensional state covariate are legal when their limitations are stated.

## Test surface

Synthetic positive and null simulations exercise the count model, exact
control reference, optimizer, six baselines, four splits, seed/donor
stability, sister guides, three null families, persistence, and corruption
detection. Adversarial tests reject noninteger counts, cross-wired hierarchy,
missing donor semantics, two-checkpoint time claims, identifier-only target
claims, unsafe artifact changes, and incomplete baseline sets.

The existence of a complete Dev40 bundle proves that the software ran. The
corrected route cannot currently emit a new `pass_scientific` result. Historical
numerical passes do not override the subsequently identified metric and null
defects or authorize biological interpretation. A GPU allocation, low training
loss, or visually sparse loading does not substitute for those gates.
