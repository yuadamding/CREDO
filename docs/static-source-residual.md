# Static source-residual information diagnostic

This auxiliary model tests whether measured source information adds value beyond
a known-target average. It is **not** a count encoder, decoder, SDE, biological
program qualification, or checkpoint-schema revision. Cohort adapters, source
data, selection protocols and run evidence belong outside this repository.

## Model and fitting boundary

Import `fit_target_source_residual` and `TargetSourceResidual` from
`credo_count_sde_v4.static_effects`.

```python
model = fit_target_source_residual(
    source_features=x_train,
    observed_effects=y_train,
    donor_ids=train_donors,
    target_ids=train_targets,
    row_ids=train_observation_ids,
    feature_names=ordered_feature_names,
    ridge_alpha=100.0,
    residual_scale=0.25,
)
prediction = model.predict(x_test, test_targets, ordered_feature_names)
restored = TargetSourceResidual.from_payload(model.to_payload())
```

`ridge_alpha` and `residual_scale` above are API examples, not recommended or
qualified hyperparameters. The caller must select them without test outcomes.
Separate future-checkpoint effects normally require separate fits.

The prediction is the training target-average plus a shrunk, regularized source
residual. Target averages first average guides within each donor, then donors.
For each fitting row, its residual response subtracts a target average computed
from **other donors in that same fitting partition**. Every target therefore
needs at least two fitting donors. Inner and outer fitting partitions must
rebuild these residuals separately; cross-fitting once globally is insufficient.

Weights give equal mass to each target, equal mass to donors within a target,
and equal mass to rows within a target/donor. Mean row weight is one. Source
features and optional target indicators are standardized with fitting-only
weighted means and deviations; deviation at or below `1e-12` is replaced by one.
The intercept is unpenalized. The objective is weighted residual SSE plus
`ridge_alpha * squared_norm(coefficients)`. The solver uses the primal or dual
form according to design dimensions. A zero residual scale reproduces the
target-average prediction exactly.

## Caller-owned scientific obligations

- Define the response/reference and distinguish raw change from a centered
  perturbation effect. Relative abundance does not identify common absolute
  growth.
- Keep future expression/outcomes out of source features. Donor identity is a
  split/group key, not a donor embedding. The numeric API cannot detect hidden
  leakage or an encoded identifier inside a supplied feature matrix.
- Fit feature selection and every learned transform independently inside each
  training partition. Held-out source measurements may be available inputs;
  held-out future outcomes may not select preprocessing or hyperparameters.
- Include exact baseline fallback and a declared donor/target-balanced selection
  rule. Preserve thresholds and selection records before outer scoring.
- Distinguish known-target transfer from unseen-target extrapolation; unknown
  target identities are rejected. Previously exposed donors are not newly
  independent validation just because the split is nested.
- Do not treat target indicators, source-depth calibration or transductive
  representations as proof of an RNA-specific biological mechanism. Qualify
  these interpretations with explicit ablations and independent evidence.

## Persistence, identity and replay

The immutable model stores canonical JSON bytes under schema
`credo.target_source_residual`, version 1. `model_id` is content-derived. Schema
validation rejects missing/extra fields, nonfinite parameters, duplicate row or
feature identities, unknown target identities and prediction feature-order
mismatches. Fit rows are sorted jointly by identity for order-stable fitting.

The fit record binds training rows, targets, donors, responses, cross-fitted
baselines/residuals, weights, feature identities, policies and exact input-array
digests. Loading verifies the content identity and reconstructs the baseline,
residual and weighting contract. The model does **not** embed the source matrix;
checking source digests and replaying predictions still requires its external
source/preprocessing evidence. Content identity is not a signature or proof that
a caller supplied scientifically appropriate features.

Existing V4 artifact/checkpoint schemas, package exports and recipe execution
paths are unchanged. This standalone static model cannot authorize integrated
count-state/SDE training or a biological claim.

## Tests

`tests/unit/test_static_effects.py` covers positive source signal, exact null and
zero-residual fallback, donor-crossfit separation, donor/target weighting,
primal/dual numerical agreement, source ablation, deterministic row identities,
strict persistence/immutability and adversarial input/schema failures. Cohort
adapters additionally need tests of fold-specific source preprocessing and
outer-outcome exclusion.
