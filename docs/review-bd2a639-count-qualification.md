# Count-representation qualification follow-up to bd2a639

Status: implementation and CPU validation; **no CUDA qualification or real-cohort
calibration has completed for this revision**. `bd2a639` remains the accepted
implementation foundation. The R48 forecasting baseline definitions, predictions,
scoring-v2 results and their scientific interpretation remain unchanged.

## Changes before expensive calibration

### Deterministic cross-source schedule

Training uses `source_interleaved_rotating_tail_v1`. Within each source, authorized
shards are shuffled for each epoch. Shuffled source rounds consume whole shards,
with one shard per source reserved for a final round. The final-round source order
rotates across epochs, so even unequal shard counts cannot always put the largest
source last. Every authorized shard has one contiguous visit, retaining the
existing single-shard count cache and within-shard row batching.

The sampler binds its version, seed, epoch and sorted authorized fitting
source/shard/payload/row-subset identities. Query outcomes, query payload hashes
and unrelated package-completion hashes never enter its numerical seed. Metadata
and count payload identities for the authorized fitting rows do enter it.
Each training epoch publishes the ordered shard addresses, their digest, its last
source and the consumed row-address digest. Calibration and refitting both use it;
audit and frozen encoding use their stable non-training traversal.

Tests cover exact replay, epoch changes, balanced final-source rotation with
unequal shard counts, unrelated-outcome invariance, and complete unique row
coverage. A separate fixture has genuinely different source distributions and
checks per-source audit errors. This eliminates fixed-source ordering by
construction; it does not prove better real-cohort optimization or globally
uniform cell shuffling. Large shard/source imbalance still merits profiling.

### Objective-matched condition comparator

The original `condition_composition` remains UMI-pooled with a 0.5 gene
pseudocount. It is not removed or relabeled as an equal-cell estimate.

`condition_cell_direction` averages `Y_i / L_i` over positive-depth training
halves within each condition, with each scored direction receiving equal weight.
Both directions of one fixed identity-keyed binomial split contribute. Audit
cells and query counts do not enter this estimate. The fixed split uses the
declared seed; optimization can still resplit its training cells by epoch.
This matches the primary score's weighting, without claiming the finite training
sample is the optimal predictor for an independent population.

Only `float64.tiny` is used as a numerical probability floor before normalization;
it is not a tuned biological pseudocount. Unseen genes can incur large audit error.
The comparator receipt records consumed cells, scored training-half denominators,
split seed and numerical floor. The depth-imbalanced regression independently
reconstructs the mean half composition and shows lower equal-cell training CE than
the UMI-pooled constant. It is a comparator test, not a biological improvement.

The primary count gate now requires improvement over the objective-matched
constant and separately selected factor, retaining the zero-latent comparison as
an additional guard. Zero is explicitly **not** an empirical fitting-latent mean
during calibration; this auxiliary check cannot qualify biological geometry.
Audit output now also separates each donor/condition source and reports both
equal-cell and UMI-weighted scores and denominators.

### Failed gate stops the default refit

`refit_representation()` rejects a failed count gate before initialization or
count reads. A deliberately requested engineering refit can use:

```bash
python -m credo_count_sde_v4.count_representation refit \
  --input-root /approved/prepared-package \
  --calibration /approved/run/calibration \
  --output /approved/run/new-diagnostic-refit \
  --diagnostic-reason 'Explicit reason for this non-promoting diagnostic'
```

The reason and diagnostic status are persisted; the failed gate remains false.
Neither a diagnostic refit nor a passing count gate sets representation or
scientific qualification true. Blank reasons are rejected. The ordinary path does
not supply this override. Small synthetic CUDA canaries deliberately do, because
they test execution with one epoch rather than optimize the component count gate.

## CUDA test coverage and numerical contract

The new `tests/gpu/test_count_representation_cuda.py` is selected by the existing
GPU test directory/marker workflow. It exercises the actual component on CUDA:
sparse projection, both molecule directions, finite nonzero gradients, clipping,
optimizer changes, calibration/audit, safetensors reload, repeated fresh refits,
frozen query export, and retrieval of individual-cell empirical laws.

Two fixed synthetic widths are used: six and **18,129 RNA features**, with default
512/128/48 architecture, rank-8 factor, batch size 256 and one calibration epoch.
The wide fixture uses 30% synthetic density beyond its first six genes. Previously
bound fitting-shard nnz/row metadata gives source-mean RNA density bounds of
19.35%–25.14%, allowing for at most one non-RNA feature per row. No raw count
payload was opened for this estimate. The 30% case is a conservative density
stress test above those means, not a measured cell-level tail distribution or
real-shard throughput sample. An 8-GiB
allocated/reserved ceiling is a bounded canary safeguard, not a 30-GB training
target or proof that a full cohort fits under this ceiling.

The tests freeze deterministic algorithms in error mode, highest float32 matmul
precision, disabled TF32, deterministic/non-benchmark cuDNN, and
`CUBLAS_WORKSPACE_CONFIG=:4096:8` **before constructing the specification**.
V2 binds these allowlisted controls alongside the existing environment identity.
A changed numerical setting fails the runtime guard. Unsupported deterministic
CUDA operations fail the test; there is no silent fallback or CPU substitution.

Repeated same-device refits compare tensor/state values at prespecified
`rtol=atol=1e-6`; they do not require CPU/CUDA byte equality. Qualification reports
include actual GPU model, library versions, bound specification/source, maximum
differences, peak CUDA allocated/reserved bytes and **process-lifetime** host peak
RSS. RSS includes setup and prior activity in that process, not an isolated
continuously sampled training-only peak. End-to-end canary time is not a cohort ETA.

The GPU workflow now fails early if CUDA is unavailable, runs with visible report
output, and uploads only the JUnit/qualification JSON artifacts. Adding these tests
or collecting them on CPU is not execution evidence. No remote CI was triggered
or code pushed for this local revision.

## Versioning and preservation

Representation specification and bundle advance to V2. Their V1 JSON schemas are
retained unchanged and are replayed using the archived bd2a639 source/wheel.
This does not modify any historical checkpoint or forecasting schema.
Existing baseline files, predictions, corrected metrics and input matrices were
not rewritten or rescored. Mean RNA retains target-transfer as its reference;
abundance retains persistence, without combining them into a new aggregate score.

## Current validation and execution boundary

The final fresh complete suite passed **591 tests, six expected CUDA skips,
one existing near-constant-input warning, and 85.6063% combined coverage** with
the unchanged 85% gate. Pytest took 944.40 seconds; command wall time was
973.48 seconds, with peak process RSS 1,406,908 KiB. Focused tests passed
28 cases with two CUDA skips and 92.2701% component coverage, separately from
the full suite. The installed wheel passed the same 28 cases and all 14 external
native adapter/workflow tests, including valid changed-protected-outcome invariance.
Formatting, Ruff, strict typing, schema, link and wheel checks passed.
The [validation receipt](../receipts/review-bd2a639-count-qualification.json)
binds these results and their exact source/test/log hashes. Historical 585-test
evidence remains specific to bd2a639.

The six-gene signal fixture passes the count gate (AE CE 0.554976 versus
matched constant 1.412178 and rank-1 factor 1.026651 at epoch 8). The different-
source fixture **fails** at its prespecified two-epoch budget (AE 1.412943 versus
matched constant 1.164508), despite beating zero-latent ablation. That negative
result is retained, not tuned into a pass. The null fixture also fails. These
small synthetic outcomes test the new comparison and stop policy; none is a
real-cohort representation or a donor-generalization result.

On 2026-09-13, the approved chain verified successfully to `ldragon3`. An exact
read-only path check found the scratch input-publication directory present, but
`/rsrch8/home/bcb/yding4/perturbseq`, its former input path and CREDO environment
absent. Scratch presence is not worker-visible input qualification. Under the
user's current Kubernetes visibility policy, an approved image/template/mount and
worker-visible project setup are required before CUDA execution or calibration.
No mounts, copies, new environments or cluster Jobs were created. The Seadragon
skill requires stopping at this missing execution authority, not guessing mounts.
The accessible parent was verified as yding4-owned, mode 0700, and direct `stat`
returned `No such file or directory` for the project, not a permission denial.

## Remaining scientific sequence

1. Qualify the exact intended CUDA worker with the component canary and a bounded
   representative real fitting-shard profile after approved access is established.
2. Calibrate only authorized D3/D4 Rest and Stim48hr rows at the unchanged epoch
   candidates; inspect complete exposure and per-source audit records. Stop on a
   failed gate; do not spend the default refit budget automatically.
3. Refit from scratch after that decision and assess fitting-only full-versus-half
   encoding stability by RNA-depth stratum, neighborhood preservation, supported
   guide/target contrasts within donor/condition, and donor/condition structure.
   Prespecify sampling and decision criteria before observing those diagnostics.
4. Freeze fitting geometry and encode D2 Rest without updates. Protected endpoint
   latent evaluation and state-dynamics fitting remain later separate work.

Half-count prediction alone does not qualify full-count transport geometry.
Batch correction remains `identity_unqualified`; omitting donor embeddings does
not establish removal of technical effects. D1 remains protected; no D2 Stim48hr
or Stim8hr fitting, reaction coupling or ecological context is introduced.
