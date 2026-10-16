# Publication validation — October 16, 2026 (America/Chicago)

This record concerns source publication only. No scientific fitting, checkpoint
loading, biological payload access, forecast regeneration, remote status query
or cluster mutation was performed for this publication.

## Identity and scope

- 235 archived source, test, protocol and configuration files, totaling
  2,970,747 bytes, match the current study workspace byte for byte. The largest
  archived file is 51,343 bytes. `SOURCE_MANIFEST.json` records each identity.
- All seven scientific observation-worker modules match the exact source
  manifests of the four v4 launch bundles. Their source-manifest SHA-256 is
  `d15446a6cf23283883547deb05ff6fe9ab6ebac009a8d761cef73afbc60b2c67`.
  The continuation, recovery and deployment-common sources were also checked
  directly against those immutable bundles. No live bundle was changed.
- The working-source inventory additionally contains unexecuted main-search
  components and the later progress helper. Inclusion is not qualification.
- Source-only text was checked for private-key headers, common GitHub/AWS token
  patterns, literal credential assignments and conflict markers; no matches
  were found. This is a scoped pattern check, not an exhaustive security audit.
- No arrays, native sample inventories, weights, forecast banks, input leases,
  live manifests, journals, generated launch bundles or caches were included.
- The existing v4 source, prior full-DDR publication and frozen vendor archive
  were not modified. Root edits only add the archive link, ownership boundary,
  immutable-source lint exclusion, publication tests and repository checksums.
- `git diff --cached --check` reports one pre-existing extra EOF blank line in
  `tests/test_observation_operations.py`. It is retained to preserve the source
  hash, not silently reformatted. No conflict marker was found.

## Checks performed

| Check | Result |
| --- | --- |
| Eight exported metadata-only regression modules | 218 passed |
| New and previous archive-integrity tests | 4 passed |
| Root `ruff check .` | Passed; byte-preserved study archives excluded |
| Root documentation links | Passed |
| Source verifier | 235 files verified; Python syntax and JSON parsed |
| Exact original/export byte comparison | Passed |
| Four v4 worker-source comparisons | Passed |

The first isolated test collection exposed a missing sibling-test import path.
Only the publication-specific pytest configuration was corrected to add `tests`
to its import path. The archived source and test bytes were not changed. The
subsequent complete selected run passed all 218 tests.

The metadata suite used the existing `ml1` Python environment. Root integrity
and lint checks used the existing count-SDE v4 virtual environment, with
`PYTHONPATH=src` bound to this publication worktree. No dependencies were
installed or changed. These checks made zero optimizer calls.

The full v4 numerical suite and the entire archived frontier suite were not
rerun for this publication. Historical test records remain historical; their
counts are not added to the 222 newly passing checks above. The archive README
lists the safe isolated command and explains the omitted private dependencies.
No CPU source check is presented as H100 qualification or a predictive result.
