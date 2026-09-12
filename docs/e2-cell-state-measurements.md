# E2 cell-state measurement primitives

`representation/state_information.py` provides pure measurement functions for a
cell-linked count-state qualification harness. It does not fit a representation,
assign biological programs, predict a future snapshot, or authorize dynamics.
Existing T01, program-head, checkpoint, and artifact schemas are unchanged.

## Information and source boundaries

The caller owns the verified cell/feature linkage, donor partition, readout
selection, library definition, evaluation role, and strict hash-bound run record.
Training-donor later snapshots can be valid fitting information for a declared
representation task. Encoding a held-out future snapshot is an observed-snapshot
test, not a forecast. Future-expression calibration may not enter a source-only
forecast. These functions cannot discover such leakage inside an arbitrary
caller-supplied array.

No function learns gene selection, bins, correction, dispersion, thresholds, or
donor/guide/checkpoint embeddings. A caller must learn permitted parameters in
the fitting partition, freeze them, then evaluate. Biological data and named
cohort adapters remain outside the reusable repository.

## Count and identity contract

`validate_count_csr` requires canonical integer CSR: nonnegative int32-compatible
values, valid integer indices/pointers, sorted duplicate-free columns, and
optional aligned unique ordered cell/feature identities. Boolean and continuous
matrices are rejected. It does not silently sum duplicates or reorder features.
Validation may share input storage; it is neither an immutable copy nor a file
integrity proof.

`thin_counts_by_cell(counts, cell_ids, seed=..., probability=0.5)` returns integer
`A` and `B` with exact `A+B=X`. Each cell gets a SHA-256-derived PCG64DXSM stream
under `THINNING_SCHEME`. Results are invariant to row permutation/minibatching
when identities and canonical feature order are preserved; feature permutation
is not promised. Explicit zeros consume no random variates. The source and
global RNG are unchanged. Binomial thinning is a declared technical diagnostic,
not an automatic independence guarantee for overdispersed RNA counts.

## Four complementary measurement roles

| Role | Available primitive | What remains caller-owned |
| --- | --- | --- |
| A: held-back expression | `conditional_composition_cross_entropy` | Frozen A-derived predictions, count split identity, RNA universe, baseline construction and donor/population aggregation. |
| B: perturbation contrasts | `count_expression_readouts` | Grouped RNA/cell weighting, fixed control reference, snapshot contrast C and post-source change D, decoded-versus-observed comparison. |
| C: source-summary reproducibility | Common-space moments/occupancies | Disjoint cell subsets, declared seeds, small-population coverage and distinction from independent donor replication. |
| D: distribution sensitivity | `fixed_bin_occupancies`, `compare_readout_distributions` | Training-defined bins, shared readout axes, matched observation treatment and representation qualification criteria. |

Cross-entropy is in natural-log units per held-back UMI. It is not the full
multinomial log likelihood or an NB likelihood. B's total only normalizes the
score and must not feed prediction. Zero B depth returns NaN; positive counts at
zero probability return infinity. Coverage must remain explicit rather than
silently discarding such rows.

`count_expression_readouts` gives fixed-scale composition or log1p composition.
When passing only selected readout columns, supply `library_sizes` from the
**complete permitted RNA universe**. A selected-gene sum is not an equivalent
library denominator. The function rejects denominators below the selected sum;
zero depth returns an undefined all-NaN row. It densifies only the supplied
columns/batch. Preserve integer UMI counts separately from these continuous
readouts.

Fixed-bin occupancy requires explicit strictly increasing edges with infinite
tails. Equality at an internal edge enters the right bin. A caller wishing to
keep empirical quantile ties in the lower bin can freeze
`nextafter(quantile, +infinity)` boundaries. This is particularly relevant for
zero-inflated empirical readouts; bins are never fitted by this function.

Distribution comparisons report per-readout means, population variances and
occupancy total variation. They can detect a same-mean collapsed population.
They do not detect every joint-distribution discrepancy, distinguish technical
noise from biology, or identify biological diffusion. Compare candidates under
the same observation treatment and common readout space, not solely their own
rescalable latent distances.

`weighted_error_decomposition` separately gives MSE = squared signed bias +
centered error variance, with bias defined as predicted minus observed. This is
an arithmetic diagnostic, not a causal variance decomposition. The function
does not emit an outcome-corrected forecast.

## Qualification boundary and tests

Passing primitive tests or cell linkage is not passing a learned representation.
Full E2 still needs fixed small count-model candidates, training-only tolerance
calibration/selection, an explicit library/depth/dispersion observation contract,
held-out snapshot scoring, contrast preservation, source reproducibility and
distribution checks jointly. E3 later tests source-only state forecasting;
abundance and ecological qualifications remain separate.

`tests/unit/test_state_information.py` includes a frozen RNG-stream fixture,
positive/null/adversarial count and identity cases, batching/order invariance,
zero-depth behavior, complete-library normalization, weighted error identities,
bin boundaries, and same-mean/different-mixture or collapsed-population tests.
