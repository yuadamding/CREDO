# Review of 789537f: corrections and promotion boundary

Date: 2026-09-12. Base commit: `789537fef0ec26e2af814867f78642fa9f02cfcd`.
Review attachment SHA-256:
`f708e7cf10afe9979511a23260bffdfa5ab27cc98fa2eefb4428535583061d0a`.

The review's central findings are accepted. This correction repairs the access
identity and component metrics; it does not establish a full-cohort training
result, a biological discovery, or passing remote CI for an unpublished tree.
The original CREDO checkout, old run results and historical validation receipt
are preserved. Package version remains `4.0.0.dev40`; source hashes identify
these corrections. The frozen core recipe identifier is not a package-version
or validation-receipt identifier.

## A. Release integrity

The original [GitHub Actions run](https://github.com/yuadamding/CREDO/actions/runs/34674172587)
was checked directly: Python 3.11, 3.12 and 3.13 stopped at repository integrity;
the wheel job passed. Later Python gates were skipped, not failed tests.

Local investigation additionally found stale/missing generated schemas,
formatting drift, four nonportable historical artifact links, and strict mypy
errors. Corrections retain the existing checks, Python matrix and 85% total
coverage requirement with branches enabled. Most type edits are explicit NumPy array
annotations; learned-state module/buffer annotations and redundant casts were
also corrected. These are not changes to historical fitted weights or results.
Pytest now explicitly uses importlib mode, matching the validated local command
and avoiding collisions with unrelated installed top-level `tests` packages.
Generated schemas now reflect the already-existing Dev39/40 contracts plus the
new prepared-access V2 contract. Frozen legacy schema definitions are retained.

The repository manifest must be generated **after staging the intended file
inventory**, then staged and checked together with `sha256sum --check`. Passing
that check is not equivalent to passing the entire workflow. The old
`receipts/local-validation.json` remains historical dev37 evidence, SHA-256
`f2d494c5d7087a90707c1177099ade917cb4ac408b0122b720bb65a7f5492c0f`.
Current results must identify the base commit, corrected implementation hash,
environment and exact checks. A commit-bound green remote matrix still requires
publication and execution for the correction commit.

### Current local validation

The [new validation record](../receipts/review-789537f-local-validation.json)
binds corrected source hash
`5a93dccefc4c27d53c259e9e71213a4f50b9bc5a5be69d7eacecee6d80fc570a`,
the 161-file source/test/script/config inventory and the measured environment.
It is working-tree evidence, not a fabricated correction-commit receipt.

| Check | Observed result |
| --- | --- |
| Full CPU suite, first attempt | 509 passed, 4 CUDA skips, one pre-existing near-constant-correlation warning; 1008.28 seconds |
| First attempt coverage gate | Failed at 84.5077%; no test failures |
| Expanded evidence-boundary module | 34 passed, including 17 added cases; package source unchanged |
| Combined coverage | 85.1430% statement/branch total, passing the unchanged 85% threshold |
| Final test inventory | 530 collected; 526 distinct CPU cases exercised across full and targeted runs, 4 CUDA skips |
| Focused access/metric and external adapter tests | 51 and 10 passed, respectively; focused tests overlap the full suite |
| Formatting / lint / strict typing / schemas / documentation | Passed locally; 158 Python files formatted, 99 source files type checked |
| Package / wheel | Build and namespace/license audit passed; installed-wheel synthetic lifecycle and full reload verification passed |

Coverage is combined from the complete first suite and the expanded module,
not a second complete suite or the GitHub matrix. Branch-only coverage is
69.8215%; the configured 85% gate applies to the combined statement/branch
total. The additional tests cover calibrated abundance boundaries, process
evidence, lineage limits, checkpoint chronology and cross-wired dossier results.
No threshold, coverage omission or failure suppression was introduced.

The wheel uses a fresh package installation with existing `ml1` dependency
site-packages; it does not reproduce fresh GitHub dependency resolution.
Its four-update, 192-cell synthetic lifecycle is not R48 training. The final
wheel SHA-256 is
`f87fbab8816a2e06a439e6a9e674207f4cf5869a88488e9f07df5425cd3b19b4`.

## B. Biological identity and access

`PreparedAccess` V2 requires a byte-bound guide artifact, explicit column
projection and semantic SHA-256 of ordered
`(guide_index, guide_id, target_id, is_control)` records. Explicit numeric indices
permit physical catalog row reordering. When a package instead defines index
by row position, that choice must be declared; biological labels are not parsed
from names or inferred from equal support counts.

The population compiler checks supplied guide IDs against this authority
**before cell metadata access** and carries target/control identities forward.
Optional cell-level biological labels are checked against that same catalog.
Changed targets or control status invalidate the semantic binding even if a
caller supplies a fresh file-byte digest. Legacy V1 access objects cannot be
silently upgraded without this missing authority.

Each authorized shard can now carry a sorted, unique, strict-integer row
allowlist. Unauthorized requests fail before any count load. Public metadata,
stream traversal, population supports and denominators respect selected rows;
original physical row coordinates remain intact. Source/task/role identity and
the selection participate in access hashing.

This is an application guard, **not physical isolation**. Loading a restricted
row from a CSR shard still decompresses its physical shard. Protected rows must
not share predictor-readable files in a production isolation claim. A future
worker needs restricted mounts or an independently enforced reader service.

Parquet reads now preflight projected uncompressed metadata sizes and estimated
allocation, project columns and decode in batches. The limits constrain payloads
and conservative estimates, not peak process-tree RSS. CSR decompression,
temporary sparse copies, Python objects, libraries and worker multiplication
still need measured qualification. The row cursor is a single-reader traversal
cursor, **not optimizer-step resume**; it does not bind batch boundaries,
worker partitioning, optimizer state or particle-noise replay.

## C. Corrected component metrics

| Finding | Current correction | Boundary |
| --- | --- | --- |
| Pooled sign ignores perturbation identity | Donor/condition/guide/gene contrasts; macro aggregation only afterward | Equal-cell panel composition, not absolute expression |
| Training controls used as endpoint reference | Matched observed controls define truth; independently predicted control means define predictions | Static conditional diagnostic, not a sealed forecasting evaluator |
| Missing or tiny effects can look favorable | Fixed 0.05 absolute log-effect threshold, `1e-8` log pseudocount, explicit coverage; undefined stays undefined | Full unit support is required for the split gate |
| Sister-guide vectors misaligned by checkpoint | Join donor/condition/gene intersections and report every pair's support | Incomplete support fails; no inferred between-target variance |
| Learned dispersion absent from reported NB likelihood | Separate fitted `predictive_nb_log_likelihood` and `common_dispersion_mean_prediction_score` | Shared train-MOM score compares means; fitted-NB validation NLL selects checkpoints |
| Free guide multiplier interpreted as efficacy | Expose `latent_guide_scale`; mark biological efficiency unidentified | Legacy checkpoint key/artifact name retained; no centering or biological anchor added |
| Coefficient null statistic treated as discoveries | Explicit legacy parameter-exceedance diagnostic; null-calibration gate fails closed | Effect-scale selection calibration is still unimplemented |
| Entire dense datasets uploaded to device | Host-only input batches, training minibatch transfer and chunked validation; panel/payload caps | Not a full-cohort CSR trainer |

Adversarial tests establish that swapping two opposite perturbation predictions
changes sign accuracy from 1 to 0 despite identical pooled means. Sister guides
with checkpoint sets `{0,1}` and `{1,2}` compare only shared checkpoint 1, not
the two equal-length concatenations. Other tests cover missing controls,
undefined effects, malformed catalogs, nonzero control-target indices, fitted
dispersion changes, latent-scale nonidentifiability and minibatch transfers.
The exact NTC masking and physical-time ordering checks remain intact.

New detailed metric artifacts declare revision 2 and persist undefined values,
coverage, pair support, execution limits and interpretation restrictions. The
frozen V1 split summary cannot represent partial metric support: such a split
is marked ineligible there, while its available numerical scores remain in the
revision-2 artifact. Legacy variance/correlation sentinel slots are explicitly
labeled and must not be read as estimated biology.

The old universal four-split-type qualification remains separate. Its random
cell inner validation and maximum-index donor are not the prepared nested donor
design. Newly computed bundles cannot promote while null calibration is absent.
This correction does not retrospectively rewrite historical bundles or license
their biological interpretation.

## Real package verification

The external `credo_full_cohort_20260912/access_qualification_a2/` receipt records
a new access check at **2026-09-12 14:06:01 UTC**, using the existing local package
without rebuilding it. Ordered biological identity was verified for all 26,504
guides, including 992 controls; semantic digest:
`61f58f2c8472a048676b04f2890a13722d04e39c4f35c049b84bd17ea9810f66`.

The first R48 roles still resolve 7,416,636 fitting cells in 607 shards and
1,983,457 D2-Rest query cells in 152 shards. Five real-shard read canaries passed
exact RNA/technical/total-depth and reordered/duplicate-row checks; all 11
unauthorized sources were rejected by the query reader. This was a bounded
canary, not a scan of all expression or model training. Elapsed 44.17 seconds,
peak controller RSS 1,220,944 KiB (about 1.16 GiB). No D1 expression, D2 endpoint
expression or Stim8hr expression was opened. No GPU or learned model was used.
The historical a1 receipt remains unchanged and describes its old source.

## D. Remaining end-to-end R48 work — not completed by this correction

First experiment stays **fit D3+D4, inner validation D2, protect D1, seed 0,
Rest-to-Stim48hr, context disabled**. The adapter resolves these prepared views;
the full trainer/evaluator is not connected yet. Required next sequence:

1. Implement a content-bound baseline workflow over authorized fitting/query
   readers; freeze feature/guide orders, normalization and support/abstention.
2. Publish immutable D2 predictions before a separately scoped evaluator opens
   D2 Stim48hr truth. Prove that protected-truth changes cannot affect predictions.
   Do not open D1 or train transformations on D2 endpoints.
3. Report coverage, keyed effect accuracy and information-matched baseline
   errors. Valid execution does not require neural superiority.
4. Add fold-fitted count representation and empirical population-law training;
   preserve full-denominator reaction semantics and source-only inference.
5. Qualify worker memory/throughput and optimizer-level restart with fixed batch,
   worker, optimizer and stochastic-noise identities. Reader replay is insufficient.

Separate, versioned claim-specific profiles remain to be implemented: static
program interpretation; known-target nested-donor prediction; unseen-guide;
descriptor-supported unseen-target; genuine temporal extrapolation. Success in
the universal static-program diagnostic must not become an unrelated gate for
a valid known-target forecasting profile. GSE314342 condition/branch identities
must remain distinct from elapsed collection time: `[8,8,48]` is not a single
ordered physical-time trajectory, and Rest must not become fabricated time zero.

No promotion, complete inner-donor training run, GPU benchmark, calibrated
discovery set, or remote publication is established by this correction alone.
