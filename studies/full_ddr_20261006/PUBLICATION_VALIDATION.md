# Publication validation — October 10, 2026 (America/Chicago)

This record concerns source publication, not current remote job status or
scientific acceptance. No remote training, prediction, assessment, or launch
was performed for this publication.

## Identity and safety

- All 258 source/support files in `SOURCE_MANIFEST.json` were read back and
  matched by byte size and SHA-256 after publication.
- Both frozen phase Python inventories were checked against their retained
  BUILD records before copying. Shared package Python files also matched the
  live study worktree exactly.
- Staged text was scanned for private-key headers, common GitHub/AWS token
  patterns, and literal credential assignments; no matches were found. This
  pattern check is not a claim of exhaustive security certification.
- No new binary files, files over 10 MB, caches, biological input/output
  arrays, model weights, or live operational manifests were staged.
- The existing v4 source and frozen `vendor/credo-6f4f57c.tar` are unchanged.
  The root changes document the authorized exception, link this archive,
  exclude its immutable bytes from v4 lint, and update repository checksums.
- `git diff --cached --check` identifies only the retained extra EOF blank
  line in each phase's `neutral_admission.py`. These were not stripped because
  their original hashes identify the executed code. No conflict markers were
  introduced.

## Local checks

| Check | Result |
| --- | --- |
| Existing v4 full `python -m pytest -q` suite | 591 passed, 6 CUDA tests skipped, 97.65 s |
| Published run-source synthetic tests listed in README | 163 passed, 18.33 s |
| Four published package regression modules | 35 passed, 0.86 s |
| New v4 publication-integrity tests | 2 passed, 0.30 s |
| `python -m ruff check .` | Passed; exact archived snapshot explicitly excluded |
| `ruff format --check src tests scripts` | Passed, 178 files |
| `scripts/check_docs.py` | Passed |
| `verify_source.py` | 258 files verified without importing run code |
| Repository index manifest and SHA-256 readback | Passed after staging |

The full v4 suite was launched before the two new publication-integrity tests
were added; those two tests were then run separately. The total is 791 passed
tests across these invocations, not one monolithic test run. Warnings concerned
unavailable NVML, sparse-invariant checking, and a near-constant correlation
input. CUDA availability was not asserted or substituted with CPU results.

v4 checks used the existing `credo-count-sde-v4/.venv` interpreter with
`PYTHONPATH=src` pointing at the publication worktree. Run-source tests used
the existing `ml1` interpreter with the published snapshot on `PYTHONPATH`.
Thread counts were bounded to one for the numerical synthetic test runs.
No dependencies or live environments were changed.

Historical operational tests requiring excluded private receipts were not
replayed. The public snapshot is deliberately not a full execution bundle.
