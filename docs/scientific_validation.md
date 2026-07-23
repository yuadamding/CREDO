# Scientific validation

CREDO outputs are fitted effective-generator contrasts, not identified cell
lineages or experimental causal effects. A favorable training fit is not
sufficient biological evidence.

## Claim boundaries

| Contract | Permitted interpretation |
| --- | --- |
| Physical axis and absolute mass | Potential physical interpolation and absolute mass change, subject to external calibration. |
| Physical axis and relative-within-group mass | Relative expansion or depletion within the declared denominator. |
| Physical axis and captured-count mass | Capture-scale abundance diagnostics only. |
| Effect axis | Static control-referenced effect path; no physical interpolation, growth, or count likelihood. |
| Unit mass | State geometry only; no abundance interpretation. |
| Source after perturbation | Response of an already perturbed population; not the effect of initiating the perturbation. |
| Matched/cross-sectional continuity | Population-level transition only; no lineage reconstruction. |
| Sequencing-library composition | Relative representation in that library; no ecological interaction claim. |

The run manifest records the axis, mass semantics, validation source, input
hashes, and bank completeness needed to enforce these boundaries.

## Evidence program

Before a cohort-level claim, predeclare and record:

1. Simple baselines such as persistence, matched controls, and linear latent transitions.
2. Dynamic baselines under the same splits, support, and metrics.
3. Leakage-free guide, target, donor, unit, and checkpoint holdouts where applicable.
4. Control and ineffective-guide calibration by checkpoint.
5. Particle, time-step, seed, and support-subsampling sensitivity.
6. Donor- and guide-aware uncertainty intervals.
7. Independent phenotype, arrayed perturbation, bulk, or external-cohort validation.
8. Ablations for target sharing, soft reference, reaction, diffusion, context, payoff, and counts.
9. A preflight showing observed required growth rates do not saturate `growth_max`.
10. Latent gene coverage and decoded-pseudobulk checks at the configured Sinkhorn scale.

Use explicit `validation.strategy: context_group` for frozen donor folds and
`validation.strategy: checkpoint` for checkpoint masking. A strict held-out
checkpoint claim also requires a representation whose fitting excluded that
checkpoint; objective masking alone cannot remove encoder leakage.

Review intervention timing and `continuity_kind` before describing a source as
baseline or a series as longitudinal. Review `block_kind` and population-pool
evidence before interpreting count competition as biological ecology.

Evidence should point to immutable manifests and tables with acceptance
criteria. Biology-specific aggregation belongs under `analysis/`, outside the
trainer and installed package.
