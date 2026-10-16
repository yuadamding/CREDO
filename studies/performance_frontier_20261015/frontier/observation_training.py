"""Resumable fold-native observation fitting; no scheduler or remote operations.

Production requires a real, admitted representation_fit NativeHandle and CUDA.
The separate fabricated CPU entry point is dimension/update capped and cannot
accidentally execute the 20,000-update scientific defaults. Resume is explicit;
any failed attempt is terminal for this owner. Checkpoint deletion is absent.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import pandas as pd
from scipy import sparse
import torch

from .checkpoints import file_sha256, load_checkpoint, retention_plan, save_checkpoint
from .native_inputs import NativeBatch, NativeHandle
from .observation import ObservationFold, SparseR48, symmetric_umi_halves, technical_holdout


AUDIT_UPDATES = (256, 512, 1024, 2048, 4096, 8192, 12288, 16384, 20000)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class ObservationTrainingSpec:
    logical_batch: int = 2048
    maximum_updates: int = 20000
    audit_updates: tuple[int, ...] = AUDIT_UPDATES
    checkpoint_interval: int = 256
    features: int = 18129
    hidden_dims: tuple[int, int] = (512, 128)
    latent_dim: int = 48
    maximum_dense_bytes: int = 4 * 1024 ** 3
    synthetic_fixture: bool = False

    def __post_init__(self):
        if (any(type(n) is not int or n < 1 for n in (self.logical_batch, self.maximum_updates,
                self.checkpoint_interval, self.features, self.latent_dim, self.maximum_dense_bytes, *self.hidden_dims))
                or len(self.hidden_dims) != 2 or type(self.audit_updates) is not tuple
                or tuple(sorted(set(self.audit_updates))) != self.audit_updates
                or not self.audit_updates or self.audit_updates[-1] != self.maximum_updates
                or any(type(n) is not int or not 1 <= n <= self.maximum_updates for n in self.audit_updates)):
            raise ValueError("Positive fixed workload and sorted terminal-inclusive audit milestones required")
        if self.synthetic_fixture:
            if (self.maximum_updates > 8 or self.logical_batch > 32 or self.features > 64
                    or max(self.hidden_dims) > 64 or self.latent_dim > 48):
                raise ValueError("Fabricated CPU dimensions/update budget exceeded")
        elif (self.logical_batch, self.maximum_updates, self.audit_updates, self.checkpoint_interval,
              self.features, self.hidden_dims, self.latent_dim, self.maximum_dense_bytes) != (
                2048, 20000, AUDIT_UPDATES, 256, 18129, (512, 128), 48, 4 * 1024 ** 3):
            raise ValueError("Scientific observation protocol settings changed")

    @classmethod
    def fixture(cls, *, logical_batch=8, maximum_updates=4, audit_updates=(2, 4), checkpoint_interval=2):
        return cls(logical_batch=logical_batch, maximum_updates=maximum_updates, audit_updates=audit_updates,
                   checkpoint_interval=checkpoint_interval, features=7, hidden_dims=(9, 5), latent_dim=3,
                   maximum_dense_bytes=1024 ** 2, synthetic_fixture=True)

    @property
    def identity(self):
        return _hash({"specification": asdict(self), "optimizer": "AdamW", "lr": 1e-3,
                      "weight_decay": 1e-4, "betas": [0.9, 0.999], "epsilon": 1e-8,
                      "gradient_clip": 1.0, "initialization_seed": 0,
                      "technical_holdout": ["row_hash", 0, 0.05], "audit_thinning_seed": 1,
                      "training_thinning_seed": "accepted_update_index", "scale_excludes_holdout": True})


def _slice(batch, rows):
    rows = np.asarray(rows)
    positions = np.flatnonzero(rows) if rows.dtype == bool else rows
    return NativeBatch(batch.rna_counts[positions].tocsr(), batch.cells.iloc[positions].reset_index(drop=True),
                       tuple(batch.row_ids[int(row)] for row in positions), batch.rna_depth[positions], batch.positive_rna[positions])


class ObservationRows:
    """One fold's admitted rows; audits get an independent deterministic iterator."""

    def __init__(self, handle, fold):
        if not isinstance(handle, NativeHandle) or not isinstance(fold, ObservationFold):
            raise TypeError("Native representation-fit handle and exact observation fold required")
        if handle.access.role != "representation_fit":
            raise ValueError("Observation fitting cannot consume query, baseline or endpoint-truth handles")
        donors = tuple(sorted({handle.index.source_roles[source][0] for source in handle.access.source_ids}))
        if (donors != fold.training_donors or handle.index.rna_axis_sha256 != fold.feature_axis_sha256
                or handle.access.amendment_sha256 != _hash({"fold_id": fold.fold_id,
                    "training_donors": fold.training_donors, "task": "Stim8hr_to_Stim48hr", "Rest": "excluded"})):
            raise ValueError("Observation source differs from fold/donor/RNA bindings")
        self.handle, self.fold = handle, fold
        self.identity = handle.access.identity()
        self.features = len(handle.index.rna_positions)
        self.synthetic_fixture = False

    @classmethod
    def fabricated(cls, batches, fold):
        batches = tuple(batches)
        if not isinstance(fold, ObservationFold) or not batches or sum(len(b.row_ids) for b in batches) > 64:
            raise ValueError("Tiny explicit fabricated CPU batches required")
        if any(not isinstance(batch, NativeBatch) for batch in batches):
            raise TypeError("Fabricated fixtures still require NativeBatch identity")
        ids = tuple(item for batch in batches for item in batch.row_ids)
        if len(set(ids)) != len(ids):
            raise ValueError("Fabricated source row IDs must be globally unique")
        value = object.__new__(cls)
        value.handle, value.fold, value._batches = None, fold, batches
        value.features = batches[0].rna_counts.shape[1]
        value.synthetic_fixture = True
        records = []
        for batch in batches:
            if (not sparse.isspmatrix_csr(batch.rna_counts) or not batch.rna_counts.has_canonical_format
                    or batch.rna_counts.shape != (len(batch.row_ids), value.features)):
                raise ValueError("Canonical bounded synthetic CSR required")
            fold.admit_training(batch.cells.donor_id.tolist(), batch.cells.condition.tolist(), len(batch.row_ids))
            depth = np.asarray(batch.rna_counts.sum(1)).ravel()
            if not np.array_equal(depth, batch.rna_depth) or not np.array_equal(depth > 0, batch.positive_rna):
                raise ValueError("Fabricated RNA support metadata mismatch")
            records.append({"ids": batch.row_ids, "data": batch.rna_counts.data.tolist(),
                            "indices": batch.rna_counts.indices.tolist(), "indptr": batch.rna_counts.indptr.tolist(),
                            "donors": batch.cells.donor_id.tolist(), "conditions": batch.cells.condition.tolist()})
        value.identity = _hash({"fabricated_CPU_only": True, "fold": asdict(fold), "rows": records})
        return value

    def iterate(self, *, batch_size, epoch=0, cursor=None):
        if self.synthetic_fixture:
            start = 0 if cursor is None else cursor["chunk"]
            if cursor is not None and cursor != {"chunk": start, "epoch": epoch, "identity": self.identity}:
                raise ValueError("Synthetic cursor binding mismatch")
            # Native streams already hash-permute rows. Fixture order is fixed deliberately.
            for index in range(start, len(self._batches)):
                batch = self._batches[index]
                if len(batch.row_ids) > batch_size:
                    raise ValueError("Fixture chunks exceed declared streaming batch size")
                yield {"chunk": index + 1, "epoch": epoch, "identity": self.identity}, batch
        else:
            decoded = None if cursor is None else self.handle.index.legacy.StreamCursor.model_validate(cursor)
            for next_cursor, batch in self.handle.iterate_rows(batch_size=batch_size, seed=0, epoch=epoch, cursor=decoded):
                yield next_cursor.model_dump(mode="json"), batch

    def training_mask(self, batch):
        self.fold.admit_training(batch.cells.donor_id.tolist(), batch.cells.condition.tolist(), len(batch.row_ids))
        return batch.positive_rna & ~technical_holdout(batch.row_ids, fold=self.fold, seed=0, fraction=0.05)

    def partition(self, *, holdout, batch_size):
        for _, batch in self.iterate(batch_size=batch_size, epoch=0):
            selected = technical_holdout(batch.row_ids, fold=self.fold, seed=0, fraction=0.05)
            selected = batch.positive_rna & (selected if holdout else ~selected)
            if selected.any():
                yield _slice(batch, selected)


class LogicalRowSampler:
    """No raw cells in snapshots: exact native cursor plus intra-chunk remainder."""

    def __init__(self, source, batch_size, state=None):
        self.source, self.batch_size = source, batch_size
        self.state = {"position": 0, "epoch": 0, "cursor": None, "offset": 0,
                      "rows_examined": 0, "epoch_training_rows": 0, "source_identity": source.identity,
                      "batch_size": batch_size} if state is None else dict(state)
        if (set(self.state) != {"position", "epoch", "cursor", "offset", "rows_examined", "epoch_training_rows",
                               "source_identity", "batch_size"}
                or self.state["source_identity"] != source.identity or self.state["batch_size"] != batch_size
                or any(type(self.state[key]) is not int or self.state[key] < 0 for key in
                       ("position", "epoch", "offset", "rows_examined", "epoch_training_rows"))):
            raise ValueError("Sampler state/input bindings changed")
        self._block = None
        self._following = None

    def _load(self):
        while self._block is None:
            iterator = self.source.iterate(batch_size=self.batch_size, epoch=self.state["epoch"], cursor=self.state["cursor"])
            try:
                following, block = next(iterator)
            except StopIteration:
                if self.state["offset"] or not self.state["epoch_training_rows"]:
                    raise ValueError("No positive-RNA training rows or invalid terminal offset")
                self.state.update(epoch=self.state["epoch"] + 1, cursor=None, offset=0, epoch_training_rows=0)
                continue
            finally:
                iterator.close()
            if self.state["offset"] > len(block.row_ids):
                raise ValueError("Sampler offset exceeds bound source chunk")
            self._block, self._following = block, following

    def _advance(self):
        self.state.update(cursor=self._following, offset=0)
        self._block, self._following = None, None

    def next(self):
        pieces, filled = [], 0
        while filled < self.batch_size:
            self._load()
            block, start = self._block, self.state["offset"]
            choices = np.flatnonzero(self.source.training_mask(block) & (np.arange(len(block.row_ids)) >= start))
            selected = choices[:self.batch_size - filled]
            stop = int(selected[-1]) + 1 if len(selected) else len(block.row_ids)
            self.state["rows_examined"] += stop - start
            self.state["offset"] = stop
            if len(selected):
                pieces.append(_slice(block, selected))
                filled += len(selected)
                self.state["position"] += len(selected)
                self.state["epoch_training_rows"] += len(selected)
            if stop == len(block.row_ids):
                self._advance()
        ids = tuple(item for piece in pieces for item in piece.row_ids)
        if len(set(ids)) != len(ids):
            raise ValueError("Insufficient distinct fitting rows for one logical batch")
        return NativeBatch(sparse.vstack([piece.rna_counts for piece in pieces], format="csr"),
                           pd.concat([piece.cells for piece in pieces], ignore_index=True), ids,
                           np.concatenate([piece.rna_depth for piece in pieces]),
                           np.concatenate([piece.positive_rna for piece in pieces]))


def _safe_file(root, name):
    path = Path(root) / name
    if path.is_absolute() and any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Symlink publication forbidden")
    if Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("Publication must be a bound relative member")
    return path


def _publish_tensor(root, name, payload):
    path = _safe_file(root, name)
    with path.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    result = {"relative_uri": name, "sha256": file_sha256(path), "size_bytes": path.stat().st_size}
    with _safe_file(root, name + ".json").open("x") as stream:
        json.dump(result, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(Path(root), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return result


class ObservationFitter:
    def __init__(self, lease, source, spec, *, device, _resume=False):
        lease.assert_active()
        if not isinstance(source, ObservationRows) or not isinstance(spec, ObservationTrainingSpec):
            raise TypeError("Exact source and observation training specification required")
        if (source.synthetic_fixture != spec.synthetic_fixture or source.features != spec.features
                or lease.identity["input_sha256"] != source.identity
                or lease.identity["configuration_sha256"] != spec.identity
                or lease.identity["fold_id"] != source.fold.fold_id or lease.identity["seed"] != 0):
            raise ValueError("Trajectory/source/fold/configuration bindings disagree")
        device = torch.device(device)
        if (spec.synthetic_fixture and device.type != "cpu") or (not spec.synthetic_fixture and device.type != "cuda"):
            raise ValueError("Scientific fitting requires admitted CUDA; fixtures require CPU")
        if _safe_file(lease.root, "OBSERVATION_FAILED.json").exists():
            raise RuntimeError("This trajectory has a retained failed owner; recovery needs a separately bound decision")
        if not _resume and retention_plan(lease)["latest"] is not None:
            raise ValueError("Existing trajectory requires explicit resume, not a fresh model restart")
        self.lease, self.source, self.spec = lease, source, spec
        self.model = self._fresh_model(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4,
                                           betas=(0.9, 0.999), eps=1e-8)
        self.sampler = LogicalRowSampler(source, spec.logical_batch)
        self.accepted_updates = self.attempted_updates = 0
        self.optimizer_calls = 0
        self.audit_history, self.training_history, self.exposure_by_source = [], [], {}
        self.selected_candidate, self.last_checkpoint = None, None
        self.failed, self.finalized = False, False

    def _fresh_model(self, device):
        return SparseR48(self.source.fold, features=self.spec.features, hidden_dims=self.spec.hidden_dims,
                         latent_dim=self.spec.latent_dim, maximum_rows=self.spec.logical_batch,
                         maximum_dense_bytes=self.spec.maximum_dense_bytes, seed=0).to(device)

    @property
    def scope(self):
        return "synthetic_cpu_only" if self.spec.synthetic_fixture else "scientific"

    @classmethod
    def resume(cls, lease, source, spec, *, device, directory, expected_sha256):
        value = cls(lease, source, spec, device=device, _resume=True)
        restored = load_checkpoint(lease, directory, expected_sha256=expected_sha256, model=value.model,
                                   optimizer=value.optimizer, scheduler=None, execution_scope=value.scope)
        if restored["phase"] != "observation_fit":
            raise ValueError("Checkpoint is not an observation-fitting trajectory")
        value.accepted_updates, value.attempted_updates = restored["accepted_updates"], restored["attempted_updates"]
        state = restored["count_cache"]
        if (state.get("schema") != "credo.frontier.observation-engine.v1" or state.get("count_cache_active") is not False
                or state.get("specification_sha256") != spec.identity or value.accepted_updates > spec.maximum_updates
                or restored["sampler_state"]["position"] != value.accepted_updates * spec.logical_batch):
            raise ValueError("Observation checkpoint state or exposure accounting mismatch")
        value.sampler = LogicalRowSampler(source, spec.logical_batch, restored["sampler_state"])
        value.audit_history, value.exposure_by_source = state["audit_history"], state["exposure_by_source"]
        value.training_history, value.optimizer_calls = state["training_history"], state["optimizer_calls"]
        value.selected_candidate = state["selected_candidate"]
        if ([row["accepted_updates"] for row in value.audit_history] != [n for n in spec.audit_updates if n <= value.accepted_updates]
                or sum(value.exposure_by_source.values()) != value.accepted_updates * spec.logical_batch
                or value.optimizer_calls != value.accepted_updates
                or [row["accepted_updates"] for row in value.training_history] != list(range(1, value.accepted_updates + 1))):
            raise ValueError("Missing audit or fitting-row exposure history on resume")
        value.last_checkpoint = {"directory": str(directory), "checkpoint_sha256": expected_sha256}
        return value

    def _audit(self):
        """Full fitting technical holdout; FP64 row/direction reduction, no RNG drift."""
        was_training = self.model.training
        python_rng, numpy_rng, cpu_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        numerator, directions, cells = 0.0, 0, 0
        self.model.eval()
        try:
            with torch.no_grad():
                for batch in self.source.partition(holdout=True, batch_size=self.spec.logical_batch):
                    self.source.fold.admit_training(batch.cells.donor_id.tolist(), batch.cells.condition.tolist(), len(batch.row_ids))
                    halves = symmetric_umi_halves(batch.rna_counts, batch.row_ids, seed=1,
                        namespace=f"frontier-umi-v1:{self.source.fold.fold_id}", maximum_rows=self.spec.logical_batch)
                    cells += len(batch.row_ids)
                    for inputs, targets in (halves, halves[::-1]):
                        log_probability = torch.log_softmax(self.model.decoder(self.model.raw_encode(inputs)), -1)
                        rows = np.repeat(np.arange(targets.shape[0]), np.diff(targets.indptr))
                        row = torch.as_tensor(rows, dtype=torch.int64, device=self.model.device)
                        columns = torch.as_tensor(targets.indices, dtype=torch.int64, device=self.model.device)
                        counts = torch.as_tensor(targets.data.astype(np.float64), device=self.model.device)
                        depth = torch.zeros(len(batch.row_ids), dtype=torch.float64, device=self.model.device).index_add(0, row, counts)
                        nll = torch.zeros_like(depth).index_add(0, row, -counts * log_probability[row, columns].double())
                        valid = depth > 0
                        numerator += float((nll[valid] / depth[valid]).sum())
                        directions += int(valid.sum())
            if directions == 0 or not np.isfinite(numerator):
                raise ValueError("Technical holdout has no finite scored cell directions")
            return {"accepted_updates": self.accepted_updates, "risk": numerator / directions,
                    "sum_normalized_cross_entropy": numerator, "positive_cell_directions": directions,
                    "positive_RNA_cells": cells, "thinning_seed": 1, "independent_donors": False}
        finally:
            self.model.train(was_training)
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

    def _audit_and_publish(self):
        report = self._audit()
        payload = {"schema": "credo.frontier.observation-candidate.v1", "inference_only": True,
                   "identity": self.lease.identity, "audit": report,
                   "model_state": {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}}
        name = f"observation-candidate-{self.accepted_updates:09d}.pt"
        reference = _publish_tensor(self.lease.root, name, payload)
        report = {**report, "artifact": reference}
        self.audit_history.append(report)
        if self.selected_candidate is None or (report["risk"], report["accepted_updates"]) < (
                self.selected_candidate["risk"], self.selected_candidate["accepted_updates"]):
            self.selected_candidate = report

    def _checkpoint(self):
        self.last_checkpoint = save_checkpoint(self.lease, model=self.model, optimizer=self.optimizer,
            scheduler=None, accepted_updates=self.accepted_updates, attempted_updates=self.attempted_updates,
            phase="observation_fit", sampler_state=self.sampler.state,
            count_cache={"schema": "credo.frontier.observation-engine.v1", "count_cache_active": False,
                         "specification_sha256": self.spec.identity, "audit_history": self.audit_history,
                         "selected_candidate": self.selected_candidate, "exposure_by_source": self.exposure_by_source,
                         "training_history": self.training_history, "optimizer_calls": self.optimizer_calls},
            execution_scope=self.scope, milestone=self.accepted_updates in self.spec.audit_updates)

    def run_until(self, accepted_updates):
        """Bounded continuation; an explicit checkpoint boundary, never a deadline."""
        self.lease.assert_active()
        if self.failed or self.finalized:
            raise RuntimeError("Failed/finalized owner cannot continue or automatically retry")
        if (type(accepted_updates) is not int or not self.accepted_updates < accepted_updates <= self.spec.maximum_updates
                or (accepted_updates % self.spec.checkpoint_interval != 0 and accepted_updates not in self.spec.audit_updates)):
            raise ValueError("Continuation must end at a declared checkpoint/audit boundary within the workload")
        try:
            while self.accepted_updates < accepted_updates:
                batch = self.sampler.next()
                self.attempted_updates += 1
                self.optimizer.zero_grad(set_to_none=True)
                result = self.model.symmetric_count_loss(batch.rna_counts, row_ids=batch.row_ids,
                    donors=batch.cells.donor_id.tolist(), conditions=batch.cells.condition.tolist(),
                    thinning_seed=self.accepted_updates)
                loss = result["loss"]
                if loss is None or not bool(torch.isfinite(loss)):
                    raise ValueError("Observation loss lacks finite positive cell directions")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
                self.optimizer_calls += 1
                self.optimizer.step()
                if not all(bool(torch.isfinite(parameter).all()) for parameter in self.model.parameters()):
                    raise ValueError("Nonfinite parameter after optimizer attempt")
                self.accepted_updates += 1
                self.training_history.append({"accepted_updates": self.accepted_updates, "loss": float(loss.detach()),
                    "gradient_norm_before_clip": float(norm), "positive_cell_directions": result["scored_cell_directions"],
                    "fitting_row_exposures": len(batch.row_ids), "thinning_seed": self.accepted_updates - 1})
                for source, count in batch.cells.source_id.value_counts().items():
                    self.exposure_by_source[str(source)] = self.exposure_by_source.get(str(source), 0) + int(count)
                if self.accepted_updates in self.spec.audit_updates:
                    self._audit_and_publish()
                if self.accepted_updates % self.spec.checkpoint_interval == 0 or self.accepted_updates in self.spec.audit_updates:
                    self._checkpoint()
            return self.progress()
        except BaseException as error:
            self.failed = True
            self._record_failure(error, "fitting")
            raise

    def _record_failure(self, error, phase):
        """Retain failure counters; never disguise a failed owner as resumable work."""
        record = {"schema": "credo.frontier.observation-failure.v1", "identity": self.lease.identity,
                  "phase": phase, "exception_type": type(error).__name__, "accepted_updates": self.accepted_updates,
                  "attempted_updates": self.attempted_updates, "optimizer_calls": self.optimizer_calls,
                  "last_complete_checkpoint": self.last_checkpoint, "automatic_retry": False}
        try:
            with _safe_file(self.lease.root, "OBSERVATION_FAILED.json").open("x") as stream:
                json.dump(record, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException as publication_error:
            error.add_note(f"Failure receipt could not be published: {type(publication_error).__name__}")

    def progress(self):
        return {"accepted_updates": self.accepted_updates, "attempted_updates": self.attempted_updates,
                "fitting_row_exposures": sum(self.exposure_by_source.values()),
                "exposure_by_source": dict(self.exposure_by_source), "audit_history": list(self.audit_history),
                "training_history": list(self.training_history), "optimizer_calls": self.optimizer_calls,
                "selected_candidate": self.selected_candidate, "last_checkpoint": self.last_checkpoint,
                "failed": self.failed, "scientific_updates": 0 if self.spec.synthetic_fixture else self.accepted_updates,
                "disposable_optimizer_calls": self.optimizer_calls if self.spec.synthetic_fixture else 0}

    def finalize_selected(self):
        """Separate inference export, not optimizer resume from an older candidate."""
        self.lease.assert_active()
        if self.failed or self.finalized or self.accepted_updates != self.spec.maximum_updates or self.selected_candidate is None:
            raise RuntimeError("Final selection requires healthy completed fixed workload")
        try:
            return self._finalize_selected()
        except BaseException as error:
            self.failed = True
            self._record_failure(error, "selected_export")
            raise

    def _finalize_selected(self):
        record = self.selected_candidate
        reference = record["artifact"]
        path = _safe_file(self.lease.root, reference["relative_uri"])
        # Hash and restricted deserialization consume the same O_NOFOLLOW handle.
        before = path.stat()
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
            opened = os.fstat(stream.fileno())
            if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino) or opened.st_size != reference["size_bytes"]:
                raise ValueError("Selected inference candidate identity/size mismatch")
            digest = hashlib.sha256()
            while block := stream.read(1024 * 1024):
                digest.update(block)
            if digest.hexdigest() != reference["sha256"]:
                raise ValueError("Selected inference candidate hash mismatch")
            stream.seek(0)
            payload = torch.load(stream, map_location="cpu", weights_only=True)
            after = path.lstat()
            if any(getattr(before, key) != getattr(after, key) for key in
                   ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")):
                raise ValueError("Selected inference candidate changed during loading")
        if (payload.get("schema") != "credo.frontier.observation-candidate.v1" or payload.get("inference_only") is not True
                or payload.get("identity") != self.lease.identity
                or payload.get("audit") != {key: value for key, value in record.items() if key != "artifact"}):
            raise ValueError("Selected export does not match this trajectory and audit")
        selected = self._fresh_model(self.model.device)
        selected.load_state_dict(payload["model_state"], strict=True)
        selected.eval()

        def scale_batches():
            for batch in self.source.partition(holdout=False, batch_size=self.spec.logical_batch):
                yield batch.rna_counts, batch.cells.donor_id.tolist(), batch.cells.condition.tolist()

        scaling = selected.fit_scaling(scale_batches())
        selected.freeze()
        final = {"schema": "credo.frontier.selected-observation.v1", "inference_only": True,
                 "identity": self.lease.identity, "selected_candidate": record, "scaling": scaling,
                 "completed_updates": self.accepted_updates,
                 "model_state": {key: value.detach().cpu().clone() for key, value in selected.state_dict().items()}}
        publication = _publish_tensor(self.lease.root, "SELECTED_OBSERVATION.pt", final)
        self.finalized = True
        return selected, {"publication": publication, "scaling": scaling, "selected_updates": record["accepted_updates"],
                          "completed_updates": self.accepted_updates, "optimizer_resume": False,
                          "technical_holdout_excluded_from_scaling": True}
