# HNSCC Guide vs Shared-Guide Comparison

This report compares the best local-log fold-0 candidate under two guide
representations:

- `with_guide`: each perturbation keeps its own guide embedding.
- `shared_guide`: the same guide-confident cells and perturbation groups are
  used, but every perturbation receives one shared guide embedding in the
  dynamics model.

The purpose is to test whether perturbation guide identity carries predictive
signal beyond the cell population and endpoint grouping.

## Executive Summary

The perturbation-specific guide model is better for perturbation-specific
state prediction and growth/mass behavior.

Across 4 random CV folds, `with_guide` improves dominant-state accuracy from
`0.3267` to `0.4117`, an absolute gain of `+0.0850` and a relative gain of
about `+26.0%` over the shared-guide ablation. It also reduces mean mass
relative error from `0.3443` to `0.1163` and mean absolute expansion-ratio gap
from `0.4417` to `0.1666`.

The shared-guide ablation has lower endpoint UOT and lower state TV. That is
not enough to call it better: it appears to smooth predictions toward a common
population response, improving aggregate endpoint/state-mixture distances while
losing perturbation-specific mass, expansion, and dominant-state identity.

## Reproducibility

Script:

```bash
scripts/run_hnscc_h100_heavy_f_best_ur01_guide_vs_shared_4cv.sh
```

Setting file:

```bash
scripts/settings_hnscc_heavy_f_best_ur01.txt
```

Recommended 4-H100 command:

```bash
CONDA_BIN=/rsrch8/home/bcb/yding4/miniforge3/bin/conda \
GPU_LIST=0,1,2,3 MAX_PARALLEL_JOBS=4 NPROC_TOTAL=224 \
bash scripts/run_hnscc_h100_heavy_f_best_ur01_guide_vs_shared_4cv.sh
```

Run root used for the reported results:

```text
runs/hnscc_random_h100_heavy_f_best_ur01_guide_vs_shared_4cv_20260505_231959
```

## Experimental Setting

Both conditions used the same data, folds, architecture, and training
hyperparameters. The only intended model-side difference was whether guide
identity was perturbation-specific or shared.

| field | value |
|---|---|
| Data path | `../GSE235325_P4P60_allgenes_allcells_latest_states.h5ad` |
| Split strategy | `random_kfold` |
| CV folds | `4` |
| Random stratify cols | `Time point, perturbation_id` |
| State key | `Cell type annotation` |
| Guide-confident only | `True` |
| Latent source | `vae` |
| Expression genes for VAE | `2000` |
| Control mode | `soft_ref` |
| Program basis | learned latent programs |
| Ecology | on |
| Growth intercept | off |
| Training schedule | staged |
| Stages | `C:150, D:150, E:1500` |
| Epochs | `1800` |
| Hidden dim / depth | `1344 / 7` |
| Programs | `42` |
| Embedding dim / mediator dim | `116 / 116` |
| Train particles / steps | `352 / 28` |
| Eval particles / steps | `1408 / 28` |
| Eval target particles | `2816` |
| Max active perturbations | `40` |
| Max train target atoms | `4096` |
| Weak loss weight | `0.20` |
| Control ref penalty | `0.002` |
| Growth-bias regularization | `0.0005` |
| Precision | `bf16` |
| Activation checkpointing | on |

The exact settings row is:

```text
heavy_f_best_ur01_h1344_d7_prog42_p352_s28_active40_lc2e3_lw20_gr5e4_nogint_e1800|116|116|42|1344|7|352|1408|2816|4096|40|2e-3|0.20|5e-4|28|28|0|learned|on|1800
```

## Model Formulation

Let `p` index perturbations, `z_t` be the latent state at time `t`, and `a_p`
be the guide embedding used by the dynamics model. At each simulator step the
model computes coefficients from the latent particles, time, ecological
context, and guide embedding:

```text
(drift, diffusion, growth) = F_theta(z_t, t, context_t, a_p)
```

The context is shared across perturbations in the current batch and is computed
from the particle population:

```text
context_t = C_theta({z_t, w_t, a_p, m_0}_p)
```

The guide embedding enters both the context aggregation and the coefficient
networks. Therefore changing `a_p` changes both the local dynamics for each
perturbation and the ecological context seen by the batch.

### With-Guide Model

In the normal model, perturbations have distinct guide embeddings:

```text
a_p = e_p
```

With `control_mode=soft_ref`, the implementation represents non-control
perturbations as residual embeddings around a learned control reference:

```text
a_p = r_ref + e_p,    p not in controls
a_p = r_ref,          p in controls
```

This lets the model learn perturbation-specific departures from a soft control
reference.

### Shared-Guide Ablation

In the shared-guide ablation, every perturbation receives the same trainable
embedding:

```text
a_p = e_shared    for all p
```

The original `perturbation_id` values are still used to build train/test target
groups and to evaluate each perturbation separately. The cell population is not
changed. This is not an `include-nonconfident` experiment; it is a guide-identity
ablation.

Because the selected setting has `growth_intercept=0`, there is no separate
perturbation-specific growth intercept in either condition. The key removed
signal is the perturbation-specific guide embedding itself.

## Metric Formulations

All summary metrics are averaged over supported perturbations in the evaluated
split.

### Endpoint UOT

For perturbation `p`, the evaluator rolls out predicted terminal weighted
particles and compares them to the observed terminal measure with the Sinkhorn
unbalanced OT divergence:

```text
UOT_p = SinkhornUOT(mu_pred_p, mu_true_p; epsilon, tau)
```

The reported `mean test UOT` is:

```text
mean_UOT = (1 / |P|) * sum_p UOT_p
```

Lower is better, but UOT alone can favor smoother aggregate endpoint fits that
do not preserve perturbation-specific mass or dominant state.

### Mass Relative Error

For perturbation `p`:

```text
mass_rel_error_p = |M_pred_p - M_true_p| / M_true_p
```

The reported mass error is:

```text
mean_mass_rel_error = (1 / |P|) * sum_p mass_rel_error_p
```

Lower is better.

### State TV

Predicted terminal particles are assigned to the nearest state centroid. This
gives a predicted state distribution `q_pred_p(s)` for perturbation `p`. The
observed terminal cells give `q_true_p(s)`.

The total variation distance is:

```text
state_TV_p = 0.5 * sum_s |q_pred_p(s) - q_true_p(s)|
```

The reported `mean test state TV` is the mean of `state_TV_p`. Lower is better.

### Dominant-State Accuracy

For perturbation `p`, define:

```text
s_pred_p = argmax_s q_pred_p(s)
s_true_p = argmax_s q_true_p(s)
```

The per-perturbation match is:

```text
match_p = 1[s_pred_p = s_true_p]
```

The reported test accuracy is:

```text
dominant_state_accuracy = (1 / |P_valid|) * sum_p match_p
```

Higher is better. This is the ranking metric used in the reported summaries.

### Expansion-Ratio Gap

For perturbation `p`, the evaluator computes expansion relative to the initial
mass:

```text
exp_pred_p = M_pred_p / M_init_p
exp_true_p = M_true_p / M_init_p
gap_p = exp_pred_p - exp_true_p
```

The summary reports:

```text
mean_abs_expansion_ratio_gap = (1 / |P|) * sum_p |gap_p|
```

Lower is better.

## Aggregate Results

| metric | with guide | shared guide | delta: with - shared | better condition |
|---|---:|---:|---:|---|
| Mean test acc | `0.4117` | `0.3267` | `+0.0850` | with guide |
| Mean test UOT | `31.9366` | `24.1509` | `+7.7857` | shared guide |
| Mean test mass err | `0.1163` | `0.3443` | `-0.2280` | with guide |
| Mean test state TV | `0.1955` | `0.1608` | `+0.0347` | shared guide |
| Mean expansion gap | `0.1666` | `0.4417` | `-0.2751` | with guide |
| Train peak GB | `49.2` | `49.2` | `0.0` | tie |
| Eval peak GB | `8.2` | `8.2` | `0.0` | tie |
| Mean train time (s) | `12960.3` | `12111.5` | `+848.8` | shared guide |

Relative to shared guide, the with-guide model gives:

- `+26.0%` relative dominant-state accuracy.
- `66.2%` lower mean mass relative error.
- `62.3%` lower mean absolute expansion-ratio gap.
- `32.2%` higher mean UOT.
- `21.6%` higher state TV.
- about `7.0%` longer mean training time.

## Per-Fold Results

### Dominant-State Accuracy

| fold | with guide | shared guide | delta |
|---|---:|---:|---:|
| fold 0 | `0.4333` | `0.2600` | `+0.1733` |
| fold 1 | `0.4467` | `0.3733` | `+0.0734` |
| fold 2 | `0.3733` | `0.3133` | `+0.0600` |
| fold 3 | `0.3933` | `0.3600` | `+0.0333` |

The with-guide model wins dominant-state accuracy on all 4 folds.

### Endpoint UOT

| fold | with guide | shared guide | delta |
|---|---:|---:|---:|
| fold 0 | `30.2506` | `23.5421` | `+6.7085` |
| fold 1 | `33.2420` | `25.1013` | `+8.1407` |
| fold 2 | `27.3333` | `19.6937` | `+7.6396` |
| fold 3 | `36.9206` | `28.2666` | `+8.6540` |

The shared-guide model wins UOT on all 4 folds.

### Mass Relative Error

| fold | with guide | shared guide | delta |
|---|---:|---:|---:|
| fold 0 | `0.0659` | `0.3359` | `-0.2700` |
| fold 1 | `0.1545` | `0.3479` | `-0.1934` |
| fold 2 | `0.0776` | `0.3518` | `-0.2742` |
| fold 3 | `0.1671` | `0.3414` | `-0.1743` |

The with-guide model wins mass error on all 4 folds.

### State TV

| fold | with guide | shared guide | delta |
|---|---:|---:|---:|
| fold 0 | `0.1943` | `0.1668` | `+0.0275` |
| fold 1 | `0.1820` | `0.1398` | `+0.0422` |
| fold 2 | `0.2382` | `0.1963` | `+0.0419` |
| fold 3 | `0.1674` | `0.1403` | `+0.0271` |

The shared-guide model wins state TV on all 4 folds.

### Expansion-Ratio Gap

| fold | with guide | shared guide | delta |
|---|---:|---:|---:|
| fold 0 | `0.1020` | `0.4404` | `-0.3384` |
| fold 1 | `0.2335` | `0.4464` | `-0.2129` |
| fold 2 | `0.1095` | `0.4392` | `-0.3297` |
| fold 3 | `0.2214` | `0.4407` | `-0.2193` |

The with-guide model wins expansion-ratio gap on all 4 folds.

## Interpretation

The results separate two kinds of performance:

1. Aggregate endpoint/state-mixture fit.
2. Perturbation-specific biological prediction.

The shared-guide model does better on UOT and state TV. Since it removes guide
identity, this suggests the common dynamics can learn a smoothed terminal
distribution that is closer to the average held-out population. This can reduce
distributional distances even if the model is less able to distinguish which
perturbation produced which response.

The with-guide model does better on dominant-state accuracy, mass relative
error, and expansion-ratio gap. These metrics require perturbation-specific
information:

- Dominant-state accuracy asks whether the model predicts the correct most
  abundant terminal state for each perturbation.
- Mass relative error asks whether terminal mass is right for each
  perturbation.
- Expansion-ratio gap asks whether growth/expansion from P4 to P60 is right for
  each perturbation.

The shared-guide model loses on all three across every fold for mass and
expansion, and across every fold for dominant-state accuracy. That is strong
evidence that the guide embedding is not merely overfitting; it carries useful
perturbation-specific signal.

## Practical Conclusion

Use the with-guide condition as the primary model for perturbation-specific
prediction:

```text
heavy_f_best_ur01_h1344_d7_prog42_p352_s28_active40_lc2e3_lw20_gr5e4_nogint_e1800
```

Primary 4-fold performance:

- Mean test dominant-state accuracy: `0.4117`
- Mean test UOT: `31.9366`
- Mean test mass relative error: `0.1163`
- Mean test state TV: `0.1955`
- Mean absolute expansion-ratio gap: `0.1666`
- Train peak GPU: `49.2 GB`

Use the shared-guide condition as a negative-control ablation. It answers a
different question: how well the model can fit endpoint/state distributions
when perturbation identity is removed from the dynamics. It should not replace
the with-guide model when the goal is perturbation-specific state or expansion
prediction.
