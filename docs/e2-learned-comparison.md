# E2 learned count-composition comparison

The reusable implementation is
`src/credo_count_sde_v4/representation/learned_state.py`. This is a bounded
snapshot representation comparison, not the complete dynamical CREDO model
and not a source-only perturbation forecast.

## Matched objective and nested references

For normalized hierarchical cell weights `w_i`, let `c_i = B_i / sum(B_i)`.
The constant composition minimizing the corresponding cross-entropy is
`sum_i w_i c_i`, not a molecule-pooled frequency. With fixed uniform-support
mixture `epsilon`, the fitted constant is
`(1 - epsilon) * sum_i w_i c_i + epsilon / number_of_RNA_genes`.
The learned loss must use that same mixture convention.

`hierarchical_cell_weights` implements equal donor, checkpoint, target, guide,
then cell weights. Controls are excluded from targeting training weights and
reported separately. `score_matched_composition` validates these weights and
computes the analytic constant from sparse count inputs.

`CountCompositionModel` provides three nested families:

| Family | Composition parameterization |
|---|---|
| M0 | Fixed analytic score-matched constant |
| M1 | Shared intercept plus rank-limited linear cell-dependent logits |
| M2 | Shared intercept plus small nonlinear encoder/decoder logits |

The count input uses full-RNA library normalization followed by log1p. No PCA,
donor/target/guide/time embedding, identity shortcut, or candidate-specific
dispersion parameter is added. Zero-initialized final residual decoding starts
each learned family at M0. Intercept offsets are not weight-decayed.
`frozen_latent_ablation` removes the cell-dependent term without refitting the
intercept or changing the observation treatment. Keep checkpoint formats for
the complete CREDO model separate from these dedicated component checkpoints.

## Measurement and failure boundaries

- `strict_seed_aggregate` requires all declared seeds and reports missing and
  undefined seed identities and coverage. An undefined seed must not disappear
  through a missing-value-skipping average.
- `smoothed_reference_delta_components` decomposes the historical fixed-prior
  loss difference into genes absent from A and genes detected in A. It is a
  diagnostic, not a prior optimizer or evidence against every learned model.
- Tensor counts must be finite, nonnegative and integral, with a true integer
  range check. A float32 maximum must not round the integer limit upward.
- Same-half decoded targeting and control populations must be combined
  coherently before evaluating their contrast against the other captured-cell
  half. Repeatability alone cannot qualify a constant representation.
- Decoded mean compositions and sampled predictive counts answer different
  questions. Apply the same full-RNA normalization/log transform to predictions
  and observations, and distinguish `q(E[X])` from `E[q(X)]`.
- Conditional-mean and conditional-observation variance components are under
  the fitted model. They do not identify biological versus technical variance.
- Marginal variance/occupancy preservation does not establish complete cell
  state geometry. Joint readouts and their own split reference are separate.

## Execution provenance

The first cohort-specific implementation and immutable scientific protocol
reside outside this repository at
`../credo_v4_e2_learned_20260910/`. Its README, source manifest, protocol,
preparation receipt, GPU worker and CPU evaluation coordinator are the exact
study authority. Four outer donor exclusions and two fixed learned families
are reported in full; selection is restricted to fitting donors. CPU scoring
code is bound before GPU outcomes, not chosen afterward.

Before packaging on 2026-09-10: 458 repository tests passed and four CUDA tests
were skipped. The external study added 105 passing synthetic tests. The full
repository Ruff check retained one existing import-order finding; new authored
files passed. These are engineering checks only. A trained application terminal
is not an E2 pass: the separately authenticated CPU fidelity and joint decision
must finish first. The original CREDO checkout and previous results are preserved.
