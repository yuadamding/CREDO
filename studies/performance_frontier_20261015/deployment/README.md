# Observation-phase deployment adapter

This package prepares one scalar observation qualification Job, then separately
bound scalar observation fits. It is not a new scheduler, a launch receipt, or
evidence that the new CUDA worker has passed. No new runtime deadline or retry
is configured. Up to four independent H100 reservations are counted by the
local modern-Python admission journal, including reserved, Pending and uncertain
submissions. The host adapter does not allocate a fifth worker independently.

## Build

From the frontier study root, use an actual journal-reservation export:

```text
python3 -m deployment.build \
  --study-root /home/yding1995/opscc_sc/credo_performance_frontier_20261015_v1 \
  --retained-bundle <exact retained comparator assessment bundle> \
  --phase observation_qualification_v1 \
  --run-id <new exact scalar Kubernetes name> \
  --admission-json <absolute real reservation export> \
  --output <absolute new local package directory>
```

The required reservation schema is
`credo.frontier.observation-admission-export.v1`. It contains `run_id`,
`status="reserved"`, `storage_reserved_bytes`,
`scientific_update_reservation`, `aggregate_reserved_output_bytes`,
`live_h100_reservations`, and `journal_event_sha256`. Additional exact journal,
source, authority and capacity identities should be retained. The owner must
check the live journal again before scheduler mutation; a dated export is not
proof that four slots remain available.

The builder copies only the seven `SOURCE_MEMBERS` from the observation worker,
the listed adapter files, phase/protocol/parent authorities, the reservation and
exact pinned lifecycle/static-proof files. It imports no torch module, opens no
RNA arrays and calls no scheduler. Gzip/tar metadata and member order are fixed.
Existing package directories are refused.

Fit packages also include the four exact hash-pinned prepared-reader source
files under `bundle/retained_reader/`. Both `metadata_root` and `dataset_root`
reuse `/rsrch8/home/bcb/yding4/perturbseq/inputs/GSE314342_count_native_v1`;
`reader_source_root` points to that fit bundle's `retained_reader/`. No raw
counts, cell tables, historical weights or duplicate metadata trees are staged.
The seven-file numerical source manifest remains identical to the qualification
source when those worker files have not changed.

The archive has the retained publisher's exact `bundle/<member>` layout. The
source is inside `bundle/source/frontier/`. `INVENTORY.json` contains every
member except itself; `BUILD.json` binds its hash, archive hash, lifecycle and
worker-contract hashes, and builder source. The existing byte-only
`final_refit_v1/bundle/deployment.py::archive_payload` and
`publish_reserved_bundle` accept this format. The old publisher's `validated_roots`
and `publish_archive` hard-code historical paths and must **not** be called.
The new operations owner must perform exact new-root, owner, capacity and
canary checks around the reusable primitives.

## Remote layout and interface

Worker-visible root:
`/rsrch8/home/bcb/yding4/perturbseq/credo_performance_frontier_20261015_v1/<phase>`.
Controller evidence uses the matching phase beneath the scratch study root.

The immutable `bundle/` contains the lifecycle `PLAN.json`, numerical
`WORKER_CONTRACT.json`, source/protocol authorities and scalar manifest.
The wrapper creates `handshake/`; it does not create `output/`. The numerical
worker creates `output/` itself under its exclusive trajectory lease.

The wrapper runs the actual interface:

```text
/opt/credo/bin/python -m frontier.observation_worker qualify \
  --contract <bundle>/WORKER_CONTRACT.json --contract-sha256 <exact SHA256>
```

Production uses `fit`. A fit bundle additionally requires `--mode fit` and
`--fit-bindings <JSON>` with the exact ten `FIT_FIELDS` in `build.py`: the
subset-native metadata/reader/dataset identities and the verified same-source
qualification receipt. This adapter deliberately has no automatic resume or
retry path. Healthy checkpoint recovery requires a separately reviewed owner.

The actual controller-side interface is `operations/observation.py`, which
requires the current installed session-preflight receipt and reconciles the
live journal. `operations/status_observation.py` obtains one read-only exact-UID
snapshot; it does not poll, restart or resubmit. Imported successful numerical
receipts alone cannot release a slot: terminal lifecycle, exact Job/Pod absence
and retained output readback must also agree.

## Gates and closeout

The deployment sequence remains receipt-gated:

1. Reconcile the modern-Python journal and exact job identity; obtain fresh
   project/fileset capacity including all reserved outputs and 16 GiB reserve.
2. Stage the immutable allowlisted archive through the approved chain. Use
   exclusive writes, hash/readback and publication/removability canaries; never
   extract broadly, overwrite a root, or substitute a login visibility check
   for worker access.
3. Verify final complete inventory and deployment receipts. Then remove only
   the exact hash/owner/size-bound Mac relay archive and its empty relay
   directory. Retain an immutable closeout receipt; ambiguous publication or
   cleanup is a blocker, not permission to stage again. The separate operations
   adapter owns this transport step.
4. Run `control.py dryrun --plan <bundle>/PLAN.json`; the retained neutral proof
   is verified, not rerun. Run `create` once, capture the exact UID and stable
   suspended zero-Pod binding, then `launch`. The pinned core handles attach-only
   supervision and guarded activation. Pending/uncertain submissions retain
   their reservations.
5. The wrapper checks source hashes, owner/mount write-readback-removability and
   one full H100 before publishing `STARTUP.json`. The host audits the actual
   Pod/image, obtains fresh capacity, and publishes `STARTUP_RELEASE.json`.
   Waiting for this release has no runtime deadline. The numerical worker then
   configures CUDA in its own process and executes its bound mode.
6. Success requires terminal Kubernetes success, worker completion, same-source
   identities, exact output byte readback and bound release. Qualification also
   requires the paired FP32 result and exactly 13 disposable optimizer calls;
   its cap is 16, and it makes zero scientific updates or real-count reads.
   A fitting completion requires 20,000 accepted updates and input lease/exit
   records. The retained core captures terminal state and exact-UID cleanup.
   There is no automatic fitting successor.

Qualification reserves 16 GiB. Each of at most ten observation fits reserves
4 GiB. These and source/controller files share the 384-GiB study cap; the owner
must account for metadata publication as well as numerical output. A quota
snapshot is not a capacity guarantee for future workers. This package does not
qualify real input throughput, scientific correctness or independent prediction.

## Completed-fit intake

`operations/accept_observation.py` is a separate local operator; it does not
modify a submitted bundle or supervise a running fit. Use it after a requested
status check or bound notification establishes terminal completion. Do not run
it repeatedly as a polling substitute. Verify the approved access chain once
before collection. From the study root:

```text
python3 operations/accept_observation.py collect \
  --input <exact observation-fit launch package> \
  --output receipts/<new immutable intake name>.json
python3 operations/accept_observation.py reconcile \
  --input receipts/<that verified intake name>.json \
  --output receipts/<new immutable reconciliation name>.json
```

Collection takes one exact-UID snapshot and refuses running, Pending, failed or
uncleaned jobs before detailed intake. It then reads only five declared output
JSON files and four lifecycle JSON files, retaining their original bytes and
identities. No checkpoint tensors, RNA/count payloads, logs, unrelated paths or
directory scans are used. It reuses the bound lifecycle's terminal byte inventory
and reconciles its selected export, audit selection, input exit, fold, protocol,
qualification and completion identities. Native-input semantic hashes do not
include the receipt file's newline and use the producer's Unicode encoding rule.

The accepted artifact references feed the existing `ObservationArtifactAdmission`
contract; the worker's `load_observation` still verifies actual tensor bytes and
contents before use. Intake does not select a new checkpoint or constitute
main-model fitting/qualification permission.

Reconciliation is local and separately explicit. It revalidates intake against
the immutable launch package, then changes only the exact submitted task to
complete under the existing journal lock. It releases an H100 slot but retains
the full four-GiB storage reservation and scientific-update accounting. It does
not delete files, reduce storage reservations or start a successor. If a local
receipt write fails after the terminal event commits, a new explicit
reconciliation can reuse only the identical intake/event; it never applies a
second event or retries a model. Existing receipt names are never overwritten.

## Local verification

`python3 -m unittest tests.test_observation_deployment -v` performs only
fabricated metadata, source-byte, manifest, archive, and local temporary-file
checks. It has no optimizer calls, GPUs, scheduler/network requests or real-data
access. It does not substitute for the H100 qualification.
