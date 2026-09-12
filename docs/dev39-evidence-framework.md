# Dev39 evidence framework

Last verified: 2026-08-21 (America/Chicago)

Status: **sealed contract layer; no cohort result or biological claim**

## Scientific boundary

Dev39 makes four non-equivalences executable:

\[
\text{population transition} \ne \text{lineage},\qquad
\text{relative abundance} \ne \text{absolute change},
\]

\[
\text{model attribution} \ne \text{mechanism},\qquad
\text{physical association} \ne \text{ecological intervention}.
\]

It does not add a model channel. It defines how observed study design, typed
results, evidence, claim language, and report availability must relate before
a later component may expose a scientific statement.

## Evidence is factorized

An evidence tier is a reporting class, not a universal scalar ranking. Every
`EvidenceDescriptor` records independent axes:

- tier;
- measurement or validation channel;
- independence from model fitting and analysis choices;
- unit of replication;
- prospective versus exposed status; and
- multiplicity family.

The tier/channel pairing is exact. A held-out channel must carry the matching
held-out independence class and split identity. Independent-cohort evidence
must identify a different evidence study. Direct measurement does not
automatically validate an unrelated estimand.

`EvidenceLink` is semantic rather than membership-only. It binds one claim to
one result, artifact, study, evidence study, exact `ScientificScope`, split,
metric/statistic, descriptor, and semantic role. A hash that happens to occur
elsewhere in a dossier cannot support a claim with another scope or role.

## Typed estimands precede prose

`ScientificClaimRequest` contains one atomic `ScientificEstimand` with:

- exact entity, sample, time, population, and comparison scope;
- outcome, estimand, direction, and effect measure;
- predictive, attributional, or intervention-supported causal status;
- optional mechanism type;
- complete abundance scale/entity/process/boundary axes when relevant;
- optional trajectory semantics; and
- an uncertainty-result identity.

Requested free text is non-authoritative. `adjudicate_scientific_claim`
checks the capability and evidence conjunction and then renders permitted
wording deterministically. Blocked requests carry explicit reasons and no
publishable wording. Unicode or rhetorically stronger free text cannot bypass
the typed estimand.

## Structural capability is not validation

`StudyEvidenceContract` records observed fields, entity counts, physical-pool
identity, destructive-snapshot status, and optional design contracts. The
derived `DatasetCapabilityAssessment` is reproducible from that object and is
explicitly structural-only.

Important gates include:

| Capability | Required design property |
| --- | --- |
| Strict held-out time | At least three checkpoints plus a protected-expression access contract |
| Relative abundance | Physical pool/exposure alignment, a bound guide catalog, stable denominator, replication, and sampling uncertainty |
| Absolute abundance | Repeated calibrated total counts inside an exact system boundary |
| Physical-context association | Co-resident populations with physical-pool identity |
| Ecological counterfactual | Association requirements plus a qualified context model |
| Ecological intervention | Co-residence plus an observed mediator intervention |
| Stable clone fate | Verified continuity and bounded collision/dropout with adequate support, confidence, and replication |
| Ancestral lineage | Stable clone evidence plus an observed evolving barcode history and direct parent relation |
| Direct cell path | Non-destructive live or paired-cell tracking |

Technical batches do not become donors. Barcode proliferation does not become
ancestry. A fitted counterfactual does not become an intervention.

## Protected-expression firewall

`ProtectedExpressionAccessContract` separately records whether protected
checkpoint expression influenced representation fitting, feature selection,
normalization estimation, batch correction, model selection, stopping,
threshold selection, or program/pathway discovery. Every flag is fixed false
for a strict held-out-time result. The audit artifact and protected checkpoint
identities are content-bound.

This contract concerns information access, not merely whether a row appeared
in an optimizer. A transductive embedding, endpoint-derived HVG list, or
endpoint-informed stopping rule invalidates strict time evidence even if the
final head did not train directly on the held-out rows.

## Abundance has three independent biological axes

`AbundanceResult` declares:

1. scale: relative within pool, capture-calibrated sampled compartment, or
   absolute tissue;
2. entity: guide, target, perturbation, clone, state, or clone-state; and
3. process: relative selection, absolute net change, proliferation, death, or
   migration/compartment loss.

It also declares the observation model and exact system boundary. Relative
guide fractions identify only relative selection. They cannot authorize
absolute growth, clone size, survival, proliferation, death, or migration.

Absolute net change needs repeated boundary-matched calibration and absolute
cell-count evidence. A separated proliferation, death, or loss component
additionally needs its matching assay and an orthogonal intervention, rescue,
epistasis, or component-assay channel. Net change is never silently renamed as
one of its possible causes.

## Trajectory and lineage

Population transitions, stable-clone fate, ancestral lineage, and direct cell
paths remain distinct result semantics. Destructive snapshots can support a
population transition law. Stable barcodes can support clone fate only after
quality gates. An ancestry claim needs an observed evolving history and parent
relation. Simulated particles are not observations of individual cells.

Semigroup and strict time evidence require the necessary checkpoints and the
protected-expression firewall. Missing design information disables only the
affected semantic layer rather than erasing unrelated program capability.

## Dossier availability

Each `PerturbationDossier` section has one explicit status:

- available;
- structurally unavailable;
- not run;
- failed validation;
- not applicable; or
- withheld.

Availability is not evidence tier E0. An unavailable section cannot carry a
result, tier, or evidence artifact. A failed program qualification cannot
populate biological-program or gene-decomposition sections. Every available
claim/result link must match the study, scope, result identity, artifact, and
semantic role exactly.

## Qualification surface

Positive, null, adversarial, and ablation tests cover the contract graph.
They reject, among other cases:

- same-cohort evidence labeled independent;
- a held-out tier without a split;
- unrelated artifact roles used for a claim;
- single-time absolute counts labeled as net change;
- barcodes labeled ancestry without observed parent history;
- relative guide abundance labeled proliferation;
- cross-wired claim/result scopes or identities;
- protected checkpoint access hidden behind a downstream representation; and
- unavailable dossier sections carrying fabricated E0 evidence.

Dev39 is therefore a semantic authorization layer, not evidence that any
particular biological statement is true.
