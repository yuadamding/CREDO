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

The run manifest records the axis, mass semantics, validation source, input
hashes, and bank completeness needed to enforce these boundaries.

## Evidence program

Before a cohort-level claim, predeclare and record:

1. Simple baselines such as persistence, matched controls, and linear latent transitions.
2. Dynamic baselines under the same splits, support, and metrics.
3. Leakage-free guide, donor, and checkpoint holdouts where scientifically applicable.
4. Control and ineffective-guide calibration by checkpoint.
5. Particle, time-step, seed, and support-subsampling sensitivity.
6. Donor- and guide-aware uncertainty intervals.
7. Independent phenotype, arrayed perturbation, bulk, or external-cohort validation.
8. Ablations for target sharing, soft reference, reaction, diffusion, context, payoff, and counts.

Evidence should point to immutable manifests and tables with acceptance
criteria. Biology-specific aggregation belongs under `analysis/`, outside the
trainer and installed package.
