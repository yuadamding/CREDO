# Dev40 static biological-program qualification

Last verified: 2026-08-21 (America/Chicago)

Status: **implemented qualification program; cohort outcomes remain external
receipt-bound evidence**

## Decision

Dev40 implements Phase 2 in two parts:

- Dev40-A freezes the raw-count observation model and typed program result
  contracts.
- Dev40-B qualifies the static/checkpoint-conditional head against fixed
  splits, baselines, stability tests, and negative controls.

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

With only target identifiers, unseen-target prediction is structurally
ineligible. It becomes eligible only when predeclared target descriptors exist
independently of protected expression and are bound as an artifact.

## Result contracts

The immutable result surface contains:

| Contract | Meaning |
| --- | --- |
| `ProgramDefinition` | Sparse signed gene loading, without an automatic pathway name |
| `PerturbationProgramEffect` | Reference, target, guide-deviation, and guide-efficiency program activity |
| `GeneLevelEffect` | Program-reconstructed signed gene effect and uncertainty |
| `GuideTargetConsistency` | Sister-guide agreement and target/guide variance summary |
| `ProgramUncertainty` | Separate seed and biological-donor loading stability plus null inclusion |
| `ProgramQualificationReceipt` | Conjunctive promotion decision and every split result |
| `BiologicalProgramQualificationBundle` | Cross-wired-ID-resistant, scope-exact enclosing object |

Every top-level result is schema-versioned and content-addressed. Model state is
stored as non-pickle NPZ. Publication is atomic and no-clobber; verification
recomputes every artifact size and SHA-256.

## Four outer splits

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

Each outer training set is split again into fit and inner-validation rows.
Stopping and checkpoint selection inspect only that inner validation subset.

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
Evaluation counts never affect baseline parameters. The program head must
improve mean log likelihood per observed count over the strongest comparator
and satisfy the frozen gene-sign threshold.

## Stability and null calibration

Seed stability uses at least three unique fits and Hungarian alignment of
absolute loading correlations. Donor stability is separate: each biological
donor is excluded in turn and its loading basis is aligned to the full-data
reference. When donor identity is unavailable, donor stability is unavailable
and promotion fails.

Sister-guide consistency is computed only from guides actually observed in
the qualification dataset. Unobserved library members cannot create artificial
agreement.

At least 20 negative-control fits cycle through three families:

- control-label/count permutation within checkpoint;
- target/guide permutation within checkpoint; and
- NB no-program simulation preserving checkpoint-specific library structure.

The program inclusion threshold is frozen before those nulls. The aggregate
false inclusion rate must remain below its preregistered maximum.

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

The existence of a complete Dev40 bundle proves that the software ran. Only a
`pass_scientific` receipt with all gates satisfied authorizes program-level
biological interpretation. A GPU allocation, long optimization, low training
loss, or visually sparse loading does not substitute for those gates.
