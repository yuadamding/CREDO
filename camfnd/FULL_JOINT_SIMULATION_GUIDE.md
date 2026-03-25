# Full Joint Simulation Guide

This document explains the full-path simulator in [`simulation/full_joint_sim.py`](simulation/full_joint_sim.py) from three angles:

1. the mathematical formulation
2. the code-to-math correspondence
3. the performance test used to validate the full path directly

The guide covers the simulator itself, the supporting coefficient/context modules, and the direct full-path training/evaluation harness added for this implementation.

## Scope

The relevant files are:

- [`simulation/full_joint_sim.py`](simulation/full_joint_sim.py)
- [`models/full_coeff_nets.py`](models/full_coeff_nets.py)
- [`models/full_context_map.py`](models/full_context_map.py)
- [`numerics/particles_torch.py`](numerics/particles_torch.py)
- [`training/full_model.py`](training/full_model.py)
- [`evaluation/full_model.py`](evaluation/full_model.py)

Important distinction:

- [`simulation/full_joint_sim.py`](simulation/full_joint_sim.py) is the general multidimensional full-model simulator.
- [`simulation/multiscreen_context_sim.py`](simulation/multiscreen_context_sim.py) is the older benchmark-path context simulator.
- [`evaluation/full_model.py`](evaluation/full_model.py) is the new direct acceptance harness for the full path.

## 1. Mathematical Formulation

### 1.1 Endpoint Data

For each key
\[
k=(s,g),
\]
where:

- \(s\) is the sample or screen
- \(g\) is the perturbation

we start from an initial finite measure
\[
\mu_k^0 = \sum_{i=1}^{N_k} w_{k,i}^0 \, \delta_{z_{k,i}^0},
\]
and want to simulate a predicted terminal measure
\[
\hat{\mu}_k^T = \sum_{i=1}^{N_k} w_{k,i}^T \, \delta_{z_{k,i}^T}.
\]

Each particle lives in latent space
\[
z_{k,i}(t) \in \mathbb{R}^d,
\]
where \(d=\) `latent_dim`.

### 1.2 Particle State

The simulator does not evolve mass directly. It evolves log-mass:
\[
\ell_{k,i}(t)=\log \tilde{w}_{k,i}(t),
\]
and the actual particle mass used in summaries is
\[
w_{k,i}(t)=\frac{M_k^0}{N_k}\exp(\ell_{k,i}(t)),
\]
where:

- \(M_k^0\) is the initial total mass for key \(k\)
- \(N_k\) is the number of particles for key \(k\)

This is exactly the representation implemented by [`TorchParticleState.atom_weights()`](numerics/particles_torch.py).

### 1.3 Sample-Level Mean-Field Context

All perturbations within the same sample \(s\) are coupled through a shared context vector
\[
c_s(t)\in\mathbb{R}^m.
\]

The full-path context map is mass-weighted and permutation-invariant:
\[
\bar{\psi}_s(t)
=
\frac{
\sum\limits_{k=(s,g)}\sum\limits_i w_{k,i}(t)\,\psi(z_{k,i}(t))
}{
\sum\limits_{k=(s,g)}\sum\limits_i w_{k,i}(t)
},
\]
\[
c_s(t)=\Phi(\bar{\psi}_s(t)).
\]

Here:

- \(\psi\) is the learned summary network
- \(\Phi\) is the learned context network

This is implemented in [`models/full_context_map.py`](models/full_context_map.py).

### 1.4 Control-Anchored Coefficient Fields

For perturbation \(g\), the model learns three coefficient fields:

- drift \(b_g(z,t,c)\)
- diffusion \(\sigma_g(z,t,c)\)
- growth \(r_g(z,t,c)\)

The full model uses the combined input
\[
u = [z,\phi(t),c],
\]
where \(\phi(t)\) is the time embedding.

Each field is control-anchored:
\[
f_g(u)=f_{\text{base}}(u)+M_f(u)a_g,
\]
where:

- \(a_g\in\mathbb{R}^p\) is the perturbation embedding
- \(a_{\text{ctrl}}=0\) exactly for controls

This gives:
\[
b_g(z,t,c)=f_g^{\text{drift}}(u),
\]
\[
\sigma_g(z,t,c)=\operatorname{softplus}\!\left(f_g^{\text{diff}}(u)\right)+\sigma_{\min},
\]
\[
r_g(z,t,c)=r_{\max}\tanh\!\left(f_g^{\text{growth}}(u)\right).
\]

This is implemented in [`models/full_coeff_nets.py`](models/full_coeff_nets.py).

### 1.5 Continuous-Time SDE

For a particle with key \(k=(s,g)\), the intended continuous-time dynamics are:

\[
dZ_t = b_g(Z_t,t,c_s(t))\,dt + \operatorname{diag}(\sigma_g(Z_t,t,c_s(t)))\,dW_t,
\]
\[
d\ell_t = r_g(Z_t,t,c_s(t))\,dt.
\]

Interpretation:

- drift moves particles through latent space
- diffusion spreads them stochastically
- growth changes their mass through log-weight dynamics
- context couples perturbations within the same sample

### 1.6 Euler-Maruyama Discretization

Let
\[
\Delta t = \frac{t_1-t_0}{N_{\text{steps}}}.
\]

Then the simulator applies:

\[
Z_{k,i}^{n+1}
=
Z_{k,i}^n
+
b_g(Z_{k,i}^n,t_n,c_s^n)\Delta t
+
\sigma_g(Z_{k,i}^n,t_n,c_s^n)\odot \sqrt{\Delta t}\,\xi_{k,i}^n,
\]

\[
\ell_{k,i}^{n+1}
=
\ell_{k,i}^n + r_g(Z_{k,i}^n,t_n,c_s^n)\Delta t,
\]

with
\[
\xi_{k,i}^n \sim \mathcal{N}(0,I_d).
\]

The discrete context is
\[
c_s^n = C\!\left(\{(Z_{k,i}^n,w_{k,i}^n)\}_{k=(s,g),i}\right).
\]

### 1.7 Terminal Statistics

At the end of the simulation, the code computes:

Total mass:
\[
\hat{M}_k = \sum_i w_{k,i}^T
\]

Weighted mean:
\[
\hat{m}_k = \sum_i \bar{w}_{k,i}^T z_{k,i}^T,
\qquad
\bar{w}_{k,i}^T = \frac{w_{k,i}^T}{\sum_j w_{k,j}^T}
\]

Trace of covariance:
\[
\operatorname{tr}(\hat{\Sigma}_k)
=
\sum_i \bar{w}_{k,i}^T \|z_{k,i}^T-\hat{m}_k\|_2^2
\]

These terminal statistics populate the `summary` table returned by the simulator.

## 2. Code Correspondence

### 2.1 `simulation/full_joint_sim.py` Line Map

The table below maps the main blocks in [`simulation/full_joint_sim.py`](simulation/full_joint_sim.py) to the mathematical objects above.

| Lines | Code block | Mathematical role |
| --- | --- | --- |
| `36-48` | constructor | Defines the endpoint family and latent dimension \(d\). |
| `50-54` | `dt` property | Computes \(\Delta t\). |
| `56-65` | `initialize_particles()` | Builds the particle approximation of \(\mu_k^0\). |
| `67-81` | `_noise_bank()` | Pre-samples Gaussian innovations \(\xi_{k,i}^n\). |
| `83-99` | `run()` preamble | Clones the initial particle system before simulation. |
| `101-102` | history/context allocation | Creates storage for trajectories and context diagnostics. |
| `104-105` | time loop | Iterates over \(t_n = n\Delta t\). |
| `106` | `model.context_values(particles)` | Computes \(c_s^n\) from the current joint particle population. |
| `107-121` | context summary rows | Stores the realized sample-level context and sample total mass. |
| `122-124` | `model.coefficients(...)` | Evaluates \(b_g,\sigma_g,r_g\). |
| `125` | `z_next = ...` | Euler-Maruyama update for \(Z_{k,i}^{n+1}\). |
| `126` | `logw_next = ...` | Forward-Euler update for \(\ell_{k,i}^{n+1}\). |
| `127-136` | particle state replacement | Promotes the simulation to the next step. |
| `138-153` | final context evaluation | Stores \(c_s(T)\) after the last update. |
| `155-158` | terminal measure construction | Converts particles into \(\hat{\mu}_k^T\). |
| `159-181` | terminal summary table | Computes \(\hat{M}_k\), \(\hat{m}_k\), and \(\operatorname{tr}(\hat{\Sigma}_k)\). |
| `185-193` | stability checks | Enforces finite particles, positive mass, and valid measures. |
| `196-205` | result packaging | Returns the complete simulation object. |

### 2.2 Supporting Modules

The simulator itself is intentionally thin. Most of the modeling assumptions live in the supporting modules.

#### A. Particle Mass Representation

- File: [`numerics/particles_torch.py`](numerics/particles_torch.py)
- Core idea:
  - particles store latent coordinates `z`
  - particles store log-weights `logw`
  - actual masses are reconstructed by exponentiation and rescaling

This file determines how means, variances, and terminal finite measures are computed.

#### B. Full Coefficient Model

- File: [`models/full_coeff_nets.py`](models/full_coeff_nets.py)
- Core idea:
  - build a control-anchored embedding for perturbations
  - combine state, time, and context into one feature vector
  - produce drift, diffusion, and growth through learned field heads

The exact control anchor comes from `ControlAnchoredEmbeddingStore`, where control perturbations are hard-fixed to zero.

#### C. Context Map

- File: [`models/full_context_map.py`](models/full_context_map.py)
- Core idea:
  - pool all particles in the same sample using mass-weighted bounded features
  - produce a sample-level context vector shared by all perturbations in that sample

This is the mechanism that makes the simulation genuinely mean-field coupled.

## 3. Training Objective For The Full Path

The new direct training harness is implemented in [`training/full_model.py`](training/full_model.py).

### 3.1 Endpoint Loss

For each key \(k\), the predicted terminal measure \(\hat{\mu}_k^T\) is compared to the observed terminal measure \(\mu_k^T\) with unbalanced Sinkhorn divergence:

\[
\mathcal{L}_{\text{endpoint},k}

=
\operatorname{UOT}_\varepsilon^\tau(\hat{\mu}_k^T,\mu_k^T).
\]

The batch endpoint loss is the mean over keys:

\[
\mathcal{L}_{\text{endpoint}}
=
\frac{1}{|\mathcal{K}|}\sum_{k\in\mathcal{K}}\mathcal{L}_{\text{endpoint},k}.
\]

### 3.2 Auxiliary Moment Losses

For each key \(k\), the training loop also penalizes errors in:

- total mass
- terminal mean vector
- terminal variance trace

\[
\mathcal{L}_{\text{aux},k}
=
\lambda_M(\hat{M}_k-M_k)^2
+
\lambda_m\|\hat{m}_k-m_k\|_2^2
+
\lambda_v(\operatorname{tr}(\hat{\Sigma}_k)-\operatorname{tr}(\Sigma_k))^2.
\]

Again this is averaged across keys.

### 3.3 Cross-Screen Delta Loss

For the two-screen Stage-II benchmark, the full-path trainer also penalizes errors in the cross-screen mean shift:

\[
\Delta \hat{m}_g = \hat{m}_{(s_2,g)} - \hat{m}_{(s_1,g)},
\qquad
\Delta m_g = m_{(s_2,g)} - m_{(s_1,g)}.
\]

\[
\mathcal{L}_{\text{screen-delta}}
=
\frac{1}{|\mathcal{G}|}\sum_{g\in\mathcal{G}}
\|\Delta \hat{m}_g - \Delta m_g\|_2^2.
\]

### 3.4 Regularization

The full trainer also includes regularization on:

- perturbation embeddings
- modulation heads
- diffusion head parameters
- overall network parameters
- context-map parameters

So the total loss is:

\[
\mathcal{L}_{\text{total}}
=
\mathcal{L}_{\text{endpoint}}
+
\mathcal{L}_{\text{aux}}
+
\lambda_\Delta \mathcal{L}_{\text{screen-delta}}
+
\mathcal{L}_{\text{reg}}.
\]

## 4. Performance Test

### 4.1 What Is Being Tested

The direct performance test for the full path is implemented in [`evaluation/full_model.py`](evaluation/full_model.py).

It trains:

- a full context-aware model
- a no-context ablation

on the synthetic Stage-II benchmark, then checks:

- simulation stability
- exact control anchoring
- terminal mass accuracy
- terminal mean accuracy
- terminal variance accuracy
- recovery of the cross-screen shift
- degradation when context is removed

### 4.2 Benchmark Setting

The default benchmark dataset inside [`evaluate_full_model()`](evaluation/full_model.py) is:

- seed: `29`
- screens: `screen1`, `screen2`
- perturbations: `ctrl`, `drift`, `diff`, `react`, `driver`
- observed cells:
  - `n_obs_p4 = 32`
  - `n_obs_p60 = 32`
- truth simulator:
  - `n_truth_particles = 1024`
  - `n_steps = 48`
- latent dimension: `1`

The default direct full-model training setting is:

- device: `cpu`
- seed: `29`
- epochs: `60`
- optimizer: Adam
- learning rate: `0.03`
- simulation steps during training: `12`
- `aux_screen_delta_mean_weight = 10.0`

The evaluator defaults to CPU on purpose because threshold-based acceptance tests are less brittle there.

### 4.3 Acceptance Criteria

The evaluator currently uses these thresholds:

- `control_shift_min = 0.02`
- `control_shift_error_max = 0.03`
- `screen_delta_error_max = 0.03`
- `full_mean_error_max = 0.05`
- `full_mass_error_max = 0.05`
- `full_variance_error_max = 0.03`
- `no_context_screen_delta_error_min = 0.05`
- `no_context_control_delta_error_min = 0.045`

### 4.4 Reproducible Command

Run from the repository root:

```bash
conda run --no-capture-output -n ml1 python3 - <<'PY'
from camfnd.evaluation.full_model import evaluate_full_model

ev = evaluate_full_model()
print("FULL_MODEL_OK", ev.ok)
print(ev.summary_table.to_string(index=False))
print(ev.thresholds)
PY
```

### 4.5 Observed Result

Observed in the local `ml1` environment on **March 24, 2026** with the default CPU configuration above:

```text
FULL_MODEL_OK True
model_name  stable  control_anchor_exact  mean_abs_mass_error  mean_abs_mean_error  mean_abs_variance_error  mean_abs_screen_delta_error  control_screen2_minus_screen1  control_screen2_minus_screen1_error  drift_screen2_minus_screen1  diff_screen2_minus_screen1  react_screen2_minus_screen1  driver_screen2_minus_screen1  endpoint_loss_mean  best_total_loss  mean_abs_context_value
      full    True                  True             0.010588             0.017649                 0.009732                     0.006041                       0.047888                             0.009043                     0.052356                    0.067871                     0.050504                      0.049828            0.001805         0.004396                0.266765
no_context    True                  True             0.008167             0.029429                 0.009426                     0.061250                       0.006626                             0.050305                     0.006569                   -0.018103                    -0.008543                     -0.002686            0.002057         0.004340                0.000000
```

### 4.6 Interpretation

These results show:

- the full-path simulator remains numerically stable
- the exact control anchor is preserved
- endpoint reconstruction is accurate
- the context-aware full model recovers the cross-screen shift well
- the no-context ablation fails to recover the screen effect

Most importantly, the direct full-path test now verifies the actual `FullJointSimulator` path instead of only the older Stage-II reference path.

## 5. Practical Notes

### 5.1 What The Simulator Supports vs What The Benchmark Uses

The simulator itself supports:

- general latent dimension \(d \ge 1\)
- vector-valued context
- state/time/context-dependent coefficient fields

The default benchmark used in the direct evaluator is still the synthetic Stage-II setting, which is currently:

- two-screen
- five-perturbation
- one-dimensional latent benchmark

So:

- the simulator is more general than the benchmark
- the benchmark is a controlled acceptance test, not the full space of supported models

### 5.2 What This Performance Test Does Not Measure

This document reports **predictive / acceptance performance**, not wall-clock profiling.

It does not currently benchmark:

- training speed across devices
- memory scaling with latent dimension
- throughput as `particles_per_atom` increases
- runtime sensitivity to context dimension

If runtime profiling is needed, that should be added as a separate benchmark note.

## 6. Short Summary

[`simulation/full_joint_sim.py`](simulation/full_joint_sim.py) simulates a coupled measure-valued SDE by:

1. representing each perturbation/sample endpoint as weighted particles
2. computing a sample-level mean-field context from all particles in a sample
3. applying control-anchored neural drift, diffusion, and growth fields
4. updating positions with Euler-Maruyama and masses through log-weight growth
5. reconstructing terminal finite measures and summary moments

The new direct full-path harness in [`evaluation/full_model.py`](evaluation/full_model.py) now validates this path end-to-end and confirms that the context-aware full model passes while the no-context ablation fails as expected.
