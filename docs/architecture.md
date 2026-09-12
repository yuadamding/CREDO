# Architecture

Status: component-qualified engineering architecture for `4.0.0.dev40`.

The package is a sibling distribution. It owns the count-native numerical
recipe and never writes into the frozen CREDO checkout. Numerical modules can
be imported without CREDO; every supported lifecycle API and CLI command first
verifies the exact frozen checkout and vendored source archive.

```text
strict contracts → sparse CountStore → source-only preparation
→ content-addressed compilation → update-based training/checkpoints
→ inference bundle → separate one-shot evaluation → sealed aggregate
```

The evidence-aware result layer is a second, orthogonal graph:

```text
observed study fields → structural capability assessment
→ static count-linked program qualification
→ abundance/trajectory/lineage semantics → perturbation dossier
→ evidence-aware claim adjudication → G14 claim seal
```

It cannot promote a numerical capability. Missing physical pools, absolute
counts, clone observations, or live tracking disable the corresponding result
instead of selecting a weaker model under stronger biological wording. See the
[Dev39 Phase 1 evidence framework](dev39-evidence-framework.md).

The Dev40 head is static/checkpoint-conditional. It models raw panel counts
with a gene-dispersion NB2 likelihood, observed library offset, sparse signed
loadings, continuous physical time, and a control/target/guide hierarchy. Its
qualification graph is independent of the V4 state SDE and cannot modify an
existing checkpoint. Dynamic coupling remains a Dev41 question. See
[Dev40 program qualification](dev40-biological-program-qualification.md).

The three intents are monotone. `count_state` exposes state transport only.
`count_measure` adds a complete-denominator relative-fitness model and exact
Dirichlet–multinomial count objective. `count_context` adds separate
state-context and fitness-context channels plus four same-start contrasts.

The public loader is `credo-v4 open-run`. Development builds do not register a
`credo.recipes` entry point. Stable discovery is reserved for the independently
qualified `4.0.0` release.

No cohort adapter, biological claim, filesystem location, or real input is
embedded in this repository.

Scientific promotion follows the T00–T13 dependency graph in
[component-qualification.md](component-qualification.md). Existing numerical
surfaces are capabilities, not evidence that their scientific component has
passed. Only typed component receipts can unlock downstream channels.

The graph has independent real-data, raw-count, and synthetic tracks. Failed
T01-v1/v2 representations block pooled state dynamics but do not block T02A
raw-count/mass noise floors or T04 fixed-truth engine testing. Dev20 added a
restartable `ParticleState` and pool-local log-sum-exp aggregation; its T04
receipt qualifies the numerical kernel for subsequent isolated synthetic
drift and reaction tests. Dev21 freezes T02A measurement-noise thresholds
from T00 plus raw counts. Neither result promotes a learned SDE.

The optional, not-yet-qualified dev18 T03 pilot separates three nested
families:

```text
M0 = global training-terminal centroid
M1 = M0 + bounded, analytically fitted sister-guide target shrinkage
M2 = M1 + low-rank source × target interaction
```

M1 is selected from preregistered training-only target-main predictors and
frozen before M2 optimization. M2 receives the source residual relative to its
training-target source centroid, uses a target-balanced ridge whitener, and
subtracts its exact training-target interaction mean. It therefore cannot
silently act as another target-main predictor. Target-main, conditional
interaction, and joint nested null families use fixed split identities and
separate permutation, optimizer, and initialization seeds. Development
calibration requires at least 119 randomization fits per null; a locked audit
requires 199 fresh fits. Every checkpoint is scored against the complete
preregistered noninteraction family. Update 0 remains deployable, controls and
reference mode have exact-zero target effects, and transactional refit uses all
outer-training series after selection.
