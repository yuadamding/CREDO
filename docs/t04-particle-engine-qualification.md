# T04 streaming particle-engine qualification

Last verified: 2026-08-15. Status: authoritative dev20 fixed-truth numerical
qualification. This is software evidence, not a cohort or biological result.

## Decision

T04 passes. The production streaming Euler–Maruyama kernel reproduced the
frozen deterministic-drift, Ornstein–Uhlenbeck, reaction-mass, absolute-weight
mean-field, and lifecycle truths. This unlocks the synthetic T05S drift and
T07S reaction tests. It does not qualify any trainable drift, diffusion,
reaction, pooled Renz model, decoder, context mechanism, or biological claim.

The exact test identity is `T04_PARTICLE_ENGINE`. Its generic trainable-channel
contract is fixed drift, fixed diffusion, fixed reaction, trainable ecology
off, and decoder off. A separate fixed-truth test qualifies absolute-mass pool
context aggregation; it does not fit or qualify an ecological mechanism.
There is no optimizer, checkpoint selection, post-selection refit, cohort,
representation, or GPU.

## Why T04 is independent

T01 asks whether pooled source counts support a promotable state
representation. T04 asks whether the numerical particle engine integrates
known dynamics correctly. The second question can and should be answered even
when a cohort-specific representation fails. Consequently:

```text
T01-v1/v2 failed ─────────────────────────┐
                                          │ no dependency
T04 fixed-truth engine → T05S drift / T07S reaction ─┘
```

T04 uses the same streaming integration functions as inference. Analytic
truth models are non-trainable adapters that supply fixed state steps,
diffusion, and reaction values to that kernel; they do not introduce a second
particle implementation.

## Frozen protocol

| Item | Contract |
| --- | --- |
| Device and dtype | CPU float64 |
| Particle grid | 64, 256, 1,024, 4,096 |
| Step grid | 8, 16, 32, 64 |
| OU repetitions | 50 independent seeds per grid point |
| Randomness | one persisted `torch.Generator` state per rollout |
| Publication | no-clobber, manifest-last committed directory |
| Primary metric | largest-grid OU variance relative error |
| OU variance limit | less than 10% |
| Reaction limit | relative error at most `1e-4` |
| Fixed pool-aggregation limit | absolute-weight reference error at most `1e-12` |
| Protected behavior | normalized weights, declared mass, ordering, replay, resume |

### Deterministic drift

For constant drift,

\[
dZ_t=v\,dt,\qquad v=0.75,
\]

the exact terminal state is `Z0 + vT`. A separate linear-drift refinement
case verifies that Euler truncation error decreases across the step grid; the
constant-drift case itself is algebraically exact under Euler integration.

### OU drift and diffusion

The one-dimensional truth is

\[
dZ_t=-0.2 Z_t\,dt+0.5\,dW_t,\qquad Z_0=0.25,\quad T=1.
\]

The continuous terminal moments are

\[
E[Z_T]=Z_0e^{-0.2T},\qquad
\operatorname{Var}(Z_T)=\frac{0.5^2}{2(0.2)}(1-e^{-2(0.2)T}).
\]

Each particle/step pair is aggregated over 50 independent seeds. The finest
grid mean must be within two aggregate Monte Carlo standard errors, its
variance relative error must be below 10%, the finest combined moment error
must improve on the coarsest, and the analytic Euler step error must decrease
strictly across the grid.

### Reaction mass

The engine evolves log mass. Two exact fixed rates are tested:

\[
r=-0.7\Rightarrow M_1=e^{-0.7}=0.496585\ldots,
\qquad
r=\log 2\Rightarrow M_1=2.
\]

Particle weights remain normalized separately; reconstructed absolute
weights `relative_mass × normalized_particle_weight` must sum to each
declared mass.

### Fixed absolute-weight pool aggregation

Two fixed states `[-1, +1]` begin with exposures `[9, 1]`. The physical-pool
mean is recomputed from absolute guide mass before each reaction update. A
manual fixed-step reference uses the same declared update order. The required
negative control incorrectly gives both guides equal within-guide weight; it
must diverge materially from the absolute-weight trajectory.

Pool aggregation is performed from log absolute masses using a pool-local
log-sum-exp. Direct masses and log masses must agree in the safe range, and a
`[1000, 998]` log-mass contrast must remain finite and reproduce the same
normalized state mean as `[0, -2]`.

### Lifecycle

The restart object stores only current particles, normalized weights,
relative log masses, generator state, and absolute grid step. T04 requires:

- bitwise deterministic CPU replay;
- bitwise equality between one-shot 32-step integration and a 13+19-step
  interrupted/resumed integration;
- no series/guide switching;
- finite normalized particle weights;
- reconstructed particle weights summing to declared masses; and
- no trainable T04 capacity checkpoint or scientific selection candidate.

## Results

Historical external evidence was recorded at workspace-relative path
`credo_v4_t04_particle_engine_20260815_r2/`; it is not shipped in this repository.
The qualification ID is
`ffa5d3968cde803675846d9f3ee54e29aa88b7a043bfe6f87aa3fd675f248569`,
the detailed receipt ID is
`f5dd7e7dbff1e916106a9f079ef441c365a285d277da885197ac060cc4312a8e`,
and the bound numerical implementation hash is
`61d22b5ab931594066a3653de9ee497a4156ca74f3758ad94d30e615e3d80cd9`.
The numerical environment hash is
`f6b82650180ceea1375bf1c9768ba4a90ee3ca6a4e17ac6da0e3966dcf1c5a55`.
An earlier passing bundle is retained at
`credo_v4_t04_particle_engine_20260815_pre_environment_binding`; it is
superseded because it did not bind Python/numerical-library versions. Its
scientific metrics agree, but it is not the promotion authority.

| Gate | Result | Limit | Status |
| --- | ---: | ---: | --- |
| Constant-drift maximum absolute error | 0 | `1e-12` | pass |
| Linear-drift refinement | strictly decreasing | required | pass |
| Finest OU mean | within two MC SE | required | pass |
| Finest OU variance relative error | 0.00445855 | 0.10 | pass |
| OU coarsest combined error | 0.0493619 | diagnostic | — |
| OU finest combined error | 0.00496401 | less than coarsest | pass |
| Depletion/doubling maximum relative error | `5.59e-16` | `1e-4` | pass |
| Fixed pool-aggregation maximum error | 0 | `1e-12` | pass |
| Normalized-context negative-control gap | greater than `1e-2` | at least `1e-2` | pass |
| Stabilized absolute series log mass | finite and reference-equal | required | pass |
| Deterministic replay | bitwise equal | required | pass |
| Interrupted/resumed replay | bitwise equal | required | pass |

The exact OU mean and variance were 0.2046826883 and 0.2060499712. The
per-seed 95% interval of largest-grid variance relative error was
`[0.00394026, 0.0475568]`; this interval characterizes Monte Carlo seeds, not
biological uncertainty.

## Package and repository validation

The final dev20 source passed 117 tests with one CUDA-only skip and 86.7746%
line/branch coverage under the repository coverage calculation. Ruff, mypy,
generated-schema parity, documentation links, and diff hygiene passed. Two
independent fixed-epoch builds produced byte-identical wheels and normalized
source distributions. `receipts/local-validation.json` binds the exact final
distribution hashes without introducing a self-reference into packaged
documentation.

A clean no-dependency wheel install against the preserved compatibility
environment passed T04 again. Because T04 now binds Python and numerical
library versions, that replay correctly received a distinct environment and
qualification ID; equality of scientific metrics rather than an invalid
cross-environment byte identity was required.

## Public surfaces

Python:

```python
from credo_count_sde_v4.numerics import qualify_particle_engine

qualify_particle_engine(output_directory)
```

Supported lifecycle API and CLI calls also verify the frozen CREDO dependency:

```python
from credo_count_sde_v4 import api

api.qualify_particle_engine(output_directory)
```

```bash
credo-v4 qualify-particle-engine --output T04_PARTICLE_ENGINE
```

The reusable implementation is
[`qualification.py`](../src/credo_count_sde_v4/numerics/qualification.py).
Restartable integration is in
[`particles.py`](../src/credo_count_sde_v4/numerics/particles.py), and stable
absolute log-mass aggregation is in
[`pool_bank.py`](../src/credo_count_sde_v4/model/pool_bank.py).

The committed result directory contains:

| Artifact | Meaning |
| --- | --- |
| `TEST_CONTRACT.json` | Fixed channel matrix and analytic baseline |
| `CONFIG.json` | dtype, grids, seeds, and truth parameters |
| `IMPLEMENTATION.sha256` | hashes of contracts, kernel, pool bank, and qualifier |
| `DETERMINISTIC_DRIFT.parquet` | constant and refinement cases |
| `OU_GRID.parquet` | all 16 particle/step aggregate moment rows |
| `REACTION_MASS.parquet` | depletion and doubling truth |
| `ECOLOGY_RESULTS.parquet` | absolute and incorrect normalized results |
| `LIFECYCLE_RESULTS.json` | replay, resume, weights, mass, and ordering gates |
| `TEST_RECEIPT.json` | detailed typed T04 decision |
| `COMPONENT_RECEIPT.json` | common component decision surface |
| `particle-engine.json` | content-addressed typed bundle |
| `artifacts.json`, `COMMITTED` | atomic publication evidence |
| `SHA256SUMS` | complete pre-publication file checksums |

## Interpretation and next action

T04 establishes that the engine can propagate known fixed drift, diffusion,
reaction, and absolute-weight mean-field dynamics with auditable restart
semantics. It does not establish that these channels are identifiable or
learnable from a biological cohort. The next eligible actions are T02A
raw-count/mass noise qualification and isolated synthetic T05S/T07S channel
recovery. Pooled Renz dynamics remain blocked by T01.

## Dev21 interpretation amendment

The immutable dev20 wire fields `ecology_absolute_weight_max_error` and
`stabilized_log_weight_pass` are legacy names. The first is the fixed
pool-aggregation reference error, not qualification of trainable ecology. The
second tested stabilized **pool-level absolute series log masses**, including
the `[1000, 998]` contrast; it did not test particle-level log weights. Dev21 keeps
the field readable for exact receipt compatibility and exposes the correctly
named `stabilized_absolute_log_mass_pass` interpretation in Python. Future
receipt schemas should use only the corrected name.

T04 qualified one homogeneous shared diagonal diffusion vector. It did not
evaluate state-, perturbation-, or time-dependent diffusion. It also tested
constant scalar reaction and a fixed state-mean pool summary, not joint
state-dependent reaction/selection or nonlinear particle-level ecology. CPU
float64 replay is qualified; CUDA/FP32 parity requires a separate
`T04G_DEVICE_PARITY` receipt before CUDA scientific use.
